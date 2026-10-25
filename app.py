"""
GA3 invoice action agent: A2A 1.0 HTTP+JSON.
Layers: protocol (routes) / storage (SQLite) / reasoning (one batched LLM call + cache).

Env:
  PUBLIC_BASE_URL   exact submitted base URL, e.g. https://agent.example.com/a2a/
  LLM_BASE_URL      OpenAI-compatible endpoint, e.g. https://api.groq.com/openai/v1
  LLM_API_KEY       key for that endpoint (never sent anywhere else)
  LLM_MODEL         model name
  ALLOWED_TOKENS    optional comma list; empty = any non-empty bearer is its own user
  REF_FORMAT        "inner" (default, ABC-1) or "bracketed" ([ABC-1])
  DB_PATH           default ./agent.db
"""
import asyncio, datetime, hashlib, json, os, re, sqlite3, uuid
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

BASE_URL = os.environ["PUBLIC_BASE_URL"]
PREFIX = urlparse(BASE_URL).path.rstrip("/")
LLM_BASE = os.environ.get("LLM_BASE_URL", "").rstrip("/")
LLM_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "")
ALLOWED = {t.strip() for t in os.environ.get("ALLOWED_TOKENS", "").split(",") if t.strip()}
REF_FORMAT = os.environ.get("REF_FORMAT", "inner")

A2A_MT = "application/a2a+json"
MT_BATCH = "application/vnd.ga5.invoice-claim-batch+json"
MT_PROP = "application/vnd.ga5.invoice-action-proposals+json"
MT_RES = "application/vnd.ga5.invoice-action-results+json"
MT_RCPT = "application/vnd.ga5.invoice-action-receipts+json"
ACTIONS = ["settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"]
TERMINAL = {"TASK_STATE_COMPLETED", "TASK_STATE_CANCELED", "TASK_STATE_FAILED", "TASK_STATE_REJECTED"}
MAX_BYTES = 500 * 1024

# ---------------------------------------------------------------- storage
db = sqlite3.connect(os.environ.get("DB_PATH", "agent.db"), check_same_thread=False, isolation_level=None)
db.execute("PRAGMA journal_mode=WAL")
db.executescript("""
CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, principal TEXT, batch_id TEXT, created REAL, json TEXT);
CREATE INDEX IF NOT EXISTS tasks_p ON tasks(principal, created);
CREATE TABLE IF NOT EXISTS idem(principal TEXT, mid TEXT, hash TEXT, task_id TEXT, PRIMARY KEY(principal, mid));
CREATE TABLE IF NOT EXISTS cache(k TEXT PRIMARY KEY, json TEXT);
""")
LOCK = asyncio.Lock()
INFLIGHT: dict = {}  # (principal, mid) -> asyncio.Future


def canon(o) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def get_task(principal, tid):
    r = db.execute("SELECT json FROM tasks WHERE id=? AND principal=?", (tid, principal)).fetchone()
    return json.loads(r[0]) if r else None


def put_task(principal, task, batch_id=None):
    db.execute(
        "INSERT INTO tasks(id,principal,batch_id,created,json) VALUES(?,?,?,strftime('%s','now'),?) "
        "ON CONFLICT(id) DO UPDATE SET json=excluded.json",
        (task["id"], principal, batch_id, canon(task)))


def batch_of(tid):
    return db.execute("SELECT batch_id FROM tasks WHERE id=?", (tid,)).fetchone()[0]


# ---------------------------------------------------------------- http helpers
def ok(body, status=200):
    return JSONResponse(body, status_code=status, media_type=A2A_MT)


def err(status, code, msg):
    return JSONResponse({"error": {"code": status, "status": code, "message": msg}},
                        status_code=status, media_type="application/json")


def auth(req: Request, needs_body=False):
    h = req.headers.get("authorization", "")
    if not h.startswith("Bearer ") or not h[7:].strip():
        return None, err(401, "UNAUTHENTICATED", "missing or invalid bearer token")
    tok = h[7:]
    if ALLOWED and tok not in ALLOWED:
        return None, err(403, "PERMISSION_DENIED", "forbidden")
    if req.headers.get("a2a-version") != "1.0":
        return None, err(400, "VERSION_NOT_SUPPORTED", "A2A-Version must be 1.0")
    if needs_body:
        ct = req.headers.get("content-type", "").split(";")[0].strip().lower()
        if ct != A2A_MT:
            return None, err(415, "UNSUPPORTED_MEDIA_TYPE", "use application/a2a+json")
    return sha("principal:" + tok), None


def view(task, history_length=None, include_artifacts=True):
    t = dict(task)
    if history_length is not None and history_length >= 0:
        t["history"] = t.get("history", [])[-history_length:] if history_length else []
    if not include_artifacts:
        t.pop("artifacts", None)
    return t


# ---------------------------------------------------------------- reasoning layer
REF_RE = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9_.:/#\-]{1,80})\]")
DECOY = re.compile(r"archiv|example|training|sample|template|illustrat|cover[ -]?sheet|historical|for reference only|do not use", re.I)


def strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from strings(v)


def paragraphs(pkg):
    out = []
    for s in strings(pkg):
        for p in re.split(r"\n\s*\n", s):
            if REF_RE.search(p):
                out.append(p)
    return out


def fmt_ref(r):
    return f"[{r}]" if REF_FORMAT == "bracketed" else r


def find_key(o, names):
    if isinstance(o, dict):
        for k, v in o.items():
            if k.lower() in names and not isinstance(v, (dict, list)):
                return v
        for v in o.values():
            r = find_key(v, names)
            if r is not None:
                return r
    elif isinstance(o, list):
        for v in o:
            r = find_key(v, names)
            if r is not None:
                return r
    return None


def fallback_facts(pkg):
    amt = find_key(pkg, {"amountminor", "amount_minor"})
    return {
        "vendorName": str(find_key(pkg, {"vendorname", "vendor", "supplier", "suppliername"}) or ""),
        "invoiceNumber": str(find_key(pkg, {"invoicenumber", "invoice_number", "invoiceno"}) or ""),
        "amountMinor": int(amt) if isinstance(amt, (int, float, str)) and str(amt).lstrip("-").isdigit() else 0,
        "currency": str(find_key(pkg, {"currency", "currencycode"}) or "INR"),
    }


KEYWORDS = {
    "reject_duplicate": r"already (been )?paid|duplicate of|previously settled|same invoice .*paid",
    "hold_invoice": r"pending verification|hold until|until .*verif|callback|bank.?detail change|awaiting confirmation",
    "open_exception": r"mismatch|conflict|discrepanc|does not reconcile|inconsistent",
    "request_approval": r"exceeds?|above .*authority|outside .*authority|approval required|beyond .*limit",
    "settle_invoice": r"within .*authority|fully reconciled|three-way match|matched",
}


def heuristic(pkg):
    best = None
    for p in paragraphs(pkg):
        refs = REF_RE.findall(p)
        if len(refs) != 3 or DECOY.search(p):
            continue
        for action, pat in KEYWORDS.items():
            if re.search(pat, p, re.I):
                best = (action, refs)
                break
        if best:
            break
    if not best:
        cand = [REF_RE.findall(p) for p in paragraphs(pkg) if not DECOY.search(p)]
        cand = [c for c in cand if len(c) == 3] or [[]]
        best = ("open_exception", cand[-1])
    return {"action": best[0], "facts": fallback_facts(pkg), "refs": best[1],
            "rationale": "Rule-based fallback decision from the current controlling paragraph."}


SYSTEM = """You are a senior accounts-payable control analyst. Decide exactly one action for EACH invoice package.

ACTIONS
- settle_invoice: invoice is valid, fully reconciled (PO, receipt, invoice agree) AND within autonomous/delegated authority, with no open verification.
- request_approval: commercially valid and reconciled, but amount or category exceeds delegated authority, so a named approver must sign off.
- hold_invoice: payment must pause until a specific stated verification completes (e.g. bank-detail change callback, missing certificate, pending goods receipt).
- reject_duplicate: the SAME commercial invoice (same vendor + same invoice number + same amount) was already paid. A payment for a DIFFERENT invoice number or amount is NOT a duplicate.
- open_exception: material records genuinely conflict (quantities, prices, vendor identity, amounts) and need an exception workflow.

HOW TO FIND THE CONTROLLING EVIDENCE
1. Ignore cover sheets, archived/historical cases, worked examples, training material, templates, and anything describing another invoice.
2. Read negation carefully: "not a duplicate", "no hold required", "does not exceed" cancel that action.
3. Ignore action words that are only mentioned, quoted, hypothetical or past ("would require approval if...").
4. The controlling paragraph is the current, authoritative finding about THIS invoice under the stated policy revision. It contains exactly three bracketed references. Return those three, exactly as written, without brackets, in the order they appear.

FACTS: take them for THIS invoice from the controlling case (not from archives, examples, or a prior paid invoice): vendorName exactly as written, invoiceNumber exactly as written, amountMinor = integer minor units (INR 1,234.50 -> 123450), currency as ISO code.

RATIONALE (2-3 sentences, under 450 characters): start with the action name; then explain how EACH of the three references supports it, writing each as [REF] followed by what it shows (e.g. "[PAY-22] shows the same invoice INV-9 was paid on 2 May"); end with why the other plausible actions do not apply.

Respond with JSON only:
{"decisions":[{"packageId":"...","action":"...","facts":{"vendorName":"...","invoiceNumber":"...","amountMinor":0,"currency":"INR"},"evidenceRefs":["A","B","C"],"rationale":"..."}]}"""


async def llm_batch(policy, pkgs):
    if not (LLM_BASE and LLM_MODEL) or not pkgs:
        return {}
    user = json.dumps({"policyRevision": policy, "packages": pkgs}, ensure_ascii=False)
    body = {"model": LLM_MODEL, "temperature": 0, "messages": [
        {"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        "response_format": {"type": "json_object"},
            "max_tokens": int(os.environ.get("MAX_TOKENS", "8000"))}
    if os.environ.get("LLM_REASONING"):
        body["reasoning_effort"] = os.environ["LLM_REASONING"]
    hdr = {"Authorization": f"Bearer {LLM_KEY}"} if LLM_KEY else {}
    async with httpx.AsyncClient(timeout=40) as c:
        for attempt in range(2):
            r = await c.post(f"{LLM_BASE}/chat/completions", json=body, headers=hdr)
            if r.status_code == 400 and "response_format" in body:
                body.pop("response_format")
                continue
            if r.status_code in (500, 502, 503) and attempt == 0:
                await asyncio.sleep(2)
                continue
            break
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"]
    txt = txt[txt.find("{"): txt.rfind("}") + 1]
    return {str(d.get("packageId")): d for d in json.loads(txt).get("decisions", [])}


def validate(pkg, d):
    """Return a clean decision or None."""
    if not d or d.get("action") not in ACTIONS:
        return None
    all_refs = set(REF_RE.findall(canon(pkg))) | set(REF_RE.findall("\n".join(strings(pkg))))
    refs = [str(r).strip().strip("[]") for r in d.get("evidenceRefs", [])]
    refs = [r for r in refs if r in all_refs]
    if len(refs) != 3 and refs:  # repair: take the paragraph that holds the first valid ref
        for p in paragraphs(pkg):
            pr = REF_RE.findall(p)
            if refs[0] in pr and len(pr) == 3:
                refs = pr
                break
    if len(refs) != 3 or len(set(refs)) != 3:
        return None
    fb = fallback_facts(pkg)
    f = d.get("facts") or {}
    try:
        amt = int(f.get("amountMinor"))
    except Exception:
        amt = fb["amountMinor"]
    facts = {"vendorName": str(f.get("vendorName") or fb["vendorName"]),
             "invoiceNumber": str(f.get("invoiceNumber") or fb["invoiceNumber"]),
             "amountMinor": amt, "currency": str(f.get("currency") or fb["currency"]).upper()}
    return {"action": d["action"], "facts": facts, "refs": refs, "rationale": str(d.get("rationale") or "")}


def pkg_key(pkg, policy):
    p = {k: v for k, v in pkg.items() if k != "packageId"}
    return sha(canon({"p": p, "policy": policy}))


def rationale(dec):
    r = dec["rationale"].strip()
    cited = sum(1 for x in dec["refs"] if x in r)
    if cited < 2 or dec["action"] not in r:
        refs = ", ".join(f"[{x}]" for x in dec["refs"])
        r = f"Action {dec['action']} is supported by {refs}. {r}"
    if len(r) < 60:
        r += " Decision based only on the current, non-archival evidence for this invoice."
    return r[:1500]


async def decide(policy, pkgs):
    out, todo = {}, []
    for p in pkgs:
        row = db.execute("SELECT json FROM cache WHERE k=?", (pkg_key(p, policy),)).fetchone()
        if row:
            out[p["packageId"]] = json.loads(row[0])
        else:
            todo.append(p)
    if todo:
        size = int(os.environ.get("CHUNK", "4"))

        async def one(ch):
            try:
                return await llm_batch(policy, ch)
            except Exception as ex:
                print("LLM ERROR:", repr(ex), getattr(getattr(ex, "response", None), "text", "")[:300], flush=True)
                return {}
        parts = await asyncio.gather(*(one(todo[i:i + size]) for i in range(0, len(todo), size)))
        got = {k: v for p in parts for k, v in p.items()}
        for p in todo:
            d = validate(p, got.get(str(p["packageId"])))
            if d:
                db.execute("INSERT OR REPLACE INTO cache(k,json) VALUES(?,?)", (pkg_key(p, policy), canon(d)))
            else:
                d = heuristic(p)
            out[p["packageId"]] = d
    return out


# ---------------------------------------------------------------- protocol layer
app = FastAPI()


@app.get("/.well-known/agent-card.json")
async def card():
    return JSONResponse({
        "name": "GA3 Invoice Action Agent",
        "description": "Reads invoice packages, proposes one evidence-backed action per invoice, and executes only grader-accepted actions with idempotent, durable task state.",
        "version": "1.0.0",
        "supportedInterfaces": [{"url": BASE_URL, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
        "capabilities": {"streaming": False, "pushNotifications": False},
        "securitySchemes": {"bearer": {"httpAuthSecurityScheme": {"scheme": "Bearer"}}},
        "securityRequirements": [{"schemes": {"bearer": {"list": []}}}],
        "defaultInputModes": [MT_BATCH, MT_RES],
        "defaultOutputModes": [MT_PROP, MT_RCPT],
        "skills": [{"id": "invoice_action_agent", "name": "Invoice action agent",
                    "description": "Choose settle, approval, hold, duplicate rejection or exception for each invoice package with cited evidence; execute accepted actions once.",
                    "tags": ["invoice", "accounts-payable", "a2a"],
                    "inputModes": [MT_BATCH, MT_RES], "outputModes": [MT_PROP, MT_RCPT]}],
    }, media_type="application/json")


def part_data(msg, mt):
    for p in msg.get("parts") or []:
        if isinstance(p, dict) and p.get("mediaType") == mt and isinstance(p.get("data"), dict):
            return p["data"]
    return None


@app.post(PREFIX + "/message:send")
async def send(req: Request):
    principal, e = auth(req, needs_body=True)
    if e:
        return e
    try:
        body = await req.json()
        msg = body["message"]
        mid = msg["messageId"]
        assert isinstance(mid, str) and mid and msg.get("role") == "ROLE_USER"
    except Exception:
        return err(400, "INVALID_ARGUMENT", "invalid message")
    h = sha(canon(msg))
    key = (principal, mid)

    async with LOCK:
        row = db.execute("SELECT hash, task_id FROM idem WHERE principal=? AND mid=?", key).fetchone()
        if row:
            if row[0] != h:
                return err(409, "IDEMPOTENCY_CONFLICT", "messageId reused with different content")
            fut = INFLIGHT.get(key)
        else:
            fut = None
        if row and not fut:
            return ok({"task": get_task(principal, row[1])})

        if not row:
            if msg.get("taskId"):  # ---------- continuation: fully handled under the lock
                return continuation(principal, key, h, msg)
            data = part_data(msg, MT_BATCH)
            pk = (data or {}).get("packages")
            if not data or not isinstance(pk, list) or not pk or not data.get("batchId"):
                return err(400, "INVALID_ARGUMENT", "expected invoice claim batch")
            ids = [p.get("packageId") for p in pk if isinstance(p, dict)]
            if len(ids) != len(pk) or None in ids or len(set(ids)) != len(ids):
                return err(400, "INVALID_ARGUMENT", "package ids must be unique")
            tid, cid = str(uuid.uuid4()), msg.get("contextId") or str(uuid.uuid4())
            hist_msg = dict(msg, taskId=tid, contextId=cid)
            task = {"id": tid, "contextId": cid, "status": {"state": "TASK_STATE_SUBMITTED", "timestamp": now()},
                    "artifacts": [], "history": [hist_msg]}
            put_task(principal, task, data["batchId"])
            db.execute("INSERT INTO idem VALUES(?,?,?,?)", (*key, h, tid))
            fut = asyncio.get_event_loop().create_future()
            INFLIGHT[key] = fut
            owner = True
        else:
            owner = False

    if not owner:  # concurrent duplicate: wait for the owner, no model call
        await asyncio.shield(fut)
        return ok({"task": get_task(principal, row[1])})

    try:
        decisions = await decide(data.get("policyRevision"), pk)
        proposals = []
        for p in pk:
            d = decisions[p["packageId"]]
            proposals.append({
                "packageId": p["packageId"],
                "actionId": "act_" + sha(f"{principal}|{tid}|{p['packageId']}")[:28],
                "action": d["action"], "facts": d["facts"],
                "evidenceRefs": [fmt_ref(r) for r in d["refs"]],
                "rationale": rationale(d)})
        async with LOCK:
            task = get_task(principal, tid)
            if task["status"]["state"] not in TERMINAL:  # canceled meanwhile -> leave alone
                task["artifacts"] = [{"artifactId": "proposals-" + tid, "name": "invoice-action-proposals",
                                      "parts": [{"mediaType": MT_PROP,
                                                 "data": {"batchId": data["batchId"], "proposals": proposals}}]}]
                task["status"] = {"state": "TASK_STATE_INPUT_REQUIRED", "timestamp": now()}
                put_task(principal, task)
    except Exception:
        async with LOCK:
            task = get_task(principal, tid)
            if task["status"]["state"] not in TERMINAL:
                task["status"] = {"state": "TASK_STATE_FAILED", "timestamp": now()}
                put_task(principal, task)
    finally:
        INFLIGHT.pop(key, None)
        fut.set_result(True)
    return ok({"task": get_task(principal, tid)})


def continuation(principal, key, h, msg):
    task = get_task(principal, msg["taskId"])
    if not task:
        return err(404, "TASK_NOT_FOUND", "task not found")
    if msg.get("contextId") != task["contextId"]:
        return err(400, "INVALID_ARGUMENT", "context mismatch")
    if task["status"]["state"] != "TASK_STATE_INPUT_REQUIRED":
        return err(409, "TASK_NOT_CONTINUABLE", "task is not awaiting input")
    data = part_data(msg, MT_RES)
    prop = task["artifacts"][0]["parts"][0]["data"]
    if not data or data.get("batchId") != prop["batchId"] or batch_of(task["id"]) != prop["batchId"]:
        return err(400, "INVALID_ARGUMENT", "batch mismatch")
    res = data.get("results")
    if not isinstance(res, list):
        return err(400, "INVALID_ARGUMENT", "results required")
    by_pkg = {p["packageId"]: p for p in prop["proposals"]}
    seen, accepted = set(), {}
    for r in res:
        if not isinstance(r, dict):
            return err(400, "INVALID_ARGUMENT", "bad result")
        p = by_pkg.get(r.get("packageId"))
        if (not p or r["packageId"] in seen or r.get("actionId") != p["actionId"]
                or r.get("action") != p["action"] or r.get("outcome") not in ("ACCEPTED", "REJECTED")):
            return err(400, "INVALID_ARGUMENT", "result does not match proposal")
        if r["outcome"] == "ACCEPTED":
            if not isinstance(r.get("receiptNonce"), str) or not r["receiptNonce"]:
                return err(400, "INVALID_ARGUMENT", "missing receipt")
            accepted[r["packageId"]] = r["receiptNonce"]
        seen.add(r["packageId"])
    if seen != set(by_pkg):
        return err(400, "INVALID_ARGUMENT", "results must cover every proposal")
    executions = [{"packageId": p["packageId"], "actionId": p["actionId"], "action": p["action"],
                   "receiptNonce": accepted[p["packageId"]], "facts": p["facts"],
                   "evidenceRefs": p["evidenceRefs"]}
                  for p in prop["proposals"] if p["packageId"] in accepted]
    task["history"].append(msg)
    task["artifacts"].append({"artifactId": "receipts-" + task["id"], "name": "invoice-action-receipts",
                              "parts": [{"mediaType": MT_RCPT,
                                         "data": {"batchId": prop["batchId"], "executions": executions}}]})
    task["status"] = {"state": "TASK_STATE_COMPLETED", "timestamp": now()}
    put_task(principal, task)
    db.execute("INSERT INTO idem VALUES(?,?,?,?)", (*key, h, task["id"]))
    return ok({"task": task})


def qint(req, name):
    v = req.query_params.get(name)
    return int(v) if v and v.lstrip("-").isdigit() else None


@app.get(PREFIX + "/tasks")
async def list_tasks(req: Request):
    principal, e = auth(req)
    if e:
        return e
    hl = qint(req, "historyLength")
    inc = req.query_params.get("includeArtifacts", "false").lower() == "true"
    size = min(qint(req, "pageSize") or 50, 100)
    rows = db.execute("SELECT json FROM tasks WHERE principal=? ORDER BY created DESC, rowid DESC LIMIT ?",
                      (principal, size)).fetchall()
    out, used = [], 64
    for (j,) in rows:
        t = view(json.loads(j), hl, inc)
        n = len(canon(t).encode()) + 2
        if used + n > MAX_BYTES:
            break
        out.append(t)
        used += n
    return ok({"tasks": out})


@app.api_route(PREFIX + "/tasks/{rest:path}", methods=["GET", "POST"])
async def task_route(rest: str, req: Request):
    if req.method == "POST" and rest.endswith(":cancel"):
        principal, e = auth(req)
        if e:
            return e
        async with LOCK:
            task = get_task(principal, rest[:-7])
            if not task:
                return err(404, "TASK_NOT_FOUND", "task not found")
            if task["status"]["state"] in TERMINAL:
                return err(409, "TASK_NOT_CANCELABLE", "task is terminal")
            task["status"] = {"state": "TASK_STATE_CANCELED", "timestamp": now()}
            put_task(principal, task)
            return ok(task)
    if req.method == "GET" and "/" not in rest and ":" not in rest:
        principal, e = auth(req)
        if e:
            return e
        task = get_task(principal, rest)
        if not task:
            return err(404, "TASK_NOT_FOUND", "task not found")
        return ok(view(task, qint(req, "historyLength")))
    return err(404, "NOT_FOUND", "not found")
