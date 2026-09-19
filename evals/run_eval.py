"""
Evaluation harness for the agentic orchestrator.

Run from the repo root (paths such as math_server.py are resolved from the cwd):

    python -m evals.run_eval --mode route --per-category 3     # cheap smoke test
    python -m evals.run_eval --mode all                        # full run

Modes:
  route      orchestrator only: routing accuracy (no DB/MCP needed)
  full       whole graph: routing, latency, retries, escalations, answer quality
  retrieval  pgvector hit@k / MRR against evals/retrieval_cases.jsonl
  all        full + retrieval
"""
import argparse
import asyncio
import contextlib
import io
import json
import math
import selectors
import sys
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from ma_chat import config, database, nodes
from ma_chat.graph import workflow

EVAL_DIR = Path(__file__).parent
ROUTES = ["MATH", "RAG", "GITHUB", "TAVILY", "SUMMARY_AGENT"]
AGENT_NODES = set(ROUTES)


class GradeSchema(TypedDict):
    correct: bool
    reasoning: str


# =====================================================
# HELPERS
# =====================================================
def log(msg):
    print(msg, file=sys.stderr, flush=True)


class Progress:
    def __init__(self, total):
        self.total, self.done = total, 0

    def tick(self, rec):
        self.done += 1
        status = "ERROR" if rec["error"] else f"{rec['expected']}->{rec['predicted']}"
        log(f"  [{self.done}/{self.total}] {rec['id']} {status} ({rec['latency_s']:.0f}s)")


def load_jsonl(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(json.loads(line))
    return rows


def build_messages(case):
    msgs = []
    for role, text in case.get("history", []):
        msgs.append(HumanMessage(content=text) if role == "user" else AIMessage(content=text))
    msgs.append(HumanMessage(content=case["question"]))
    return msgs


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / denom, (centre + margin) / denom


def rate(k, n):
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * k / n:.1f}%  (95% CI {100 * lo:.0f}-{100 * hi:.0f}%)"


def percentile(values, q):
    if not values:
        return None
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def answer_text(answer):
    """Agent answers are sometimes a list of content blocks ([{'type': 'text', 'text': ...}]); flatten to a string."""
    if isinstance(answer, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in answer)
    return answer


def norm(text):
    return "".join(str(text).lower().split())


# =====================================================
# GRADING
# =====================================================
async def llm_grade(model, case, answer, sem):
    prompt = (
        "You are grading an AI assistant's answer against a reference.\n"
        "Mark correct=true if the answer is consistent with the reference and does not contradict it; "
        "extra detail is fine. Mark false if it is wrong, missing the key facts, or is an error message.\n\n"
        f"QUESTION: {case['question']}\nREFERENCE: {case['reference']}\nANSWER: {answer}"
    )

    def _call():
        return model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json", "response_schema": GradeSchema},
        )

    for attempt in range(2):
        try:
            async with sem:
                resp = await asyncio.to_thread(_call)
            return bool(json.loads(resp.text)["correct"])
        except Exception as e:
            if attempt:
                log(f"[grader] failed for {case['id']}: {e}")
    return None


async def grade(model, case, answer, sem):
    """True/False, or None when the case has no ground truth (or the grader failed)."""
    if answer is None:
        return False if ("expect_all" in case or "reference" in case) else None
    if "expect_all" in case:
        return all(norm(e) in norm(answer) for e in case["expect_all"])
    if "reference" in case:
        return await llm_grade(model, case, answer, sem)
    return None


# =====================================================
# JOBS
# =====================================================
async def with_backoff(once, attempts, progress):
    """Re-run a case whose only problem was an API rate limit (429); each retry starts from a clean record."""
    for i in range(attempts):
        rec = await once()
        rec["attempts"] = i + 1
        if i == attempts - 1 or not (rec["error"] and "RateLimitError" in rec["error"]):
            break
        await asyncio.sleep(15 * (i + 1))
    progress.tick(rec)
    return rec


async def route_job(case, sem, timeout, progress, attempts):
    async def once():
        rec = {"id": case["id"], "expected": case["category"], "predicted": None, "error": None}
        t0 = time.perf_counter()
        try:
            state = {"question": case["question"], "messages": build_messages(case)}
            out = await asyncio.wait_for(nodes.node_orchestrator(state), timeout)
            rec["predicted"] = out["next_agent"]
            rec["rewritten"] = out["question"]
        except Exception as e:
            rec["error"] = repr(e)
        rec["latency_s"] = time.perf_counter() - t0
        return rec

    async with sem:
        return await with_backoff(once, attempts, progress)


async def full_job(app, case, sem, timeout, progress, attempts):
    async def once():
        rec = {"id": case["id"], "expected": case["category"], "predicted": None, "error": None,
               "first_answer": None, "final_answer": None, "retries": 0, "escalated": False}
        cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
        payload = {"question": case["question"], "messages": build_messages(case), "retry_count": 0}

        async def drive():
            async for chunk in app.astream(payload, config=cfg, stream_mode="updates"):
                for node, upd in chunk.items():
                    if node == "orchestrator":
                        if rec["predicted"] is None:
                            rec["predicted"] = upd["next_agent"]
                            rec["rewritten"] = upd["question"]
                    elif node in AGENT_NODES and rec["first_answer"] is None:
                        rec["first_answer"] = answer_text(upd.get("answer"))

        t0 = time.perf_counter()
        try:
            await asyncio.wait_for(drive(), timeout)
            snap = await app.aget_state(cfg)
            rec["escalated"] = bool(snap.next)
            rec["retries"] = snap.values.get("retry_count", 0)
            rec["final_answer"] = answer_text(snap.values.get("answer"))
            rec["critique"] = snap.values.get("system_critique")
            # The evaluator only escalates without a critique when Gemini itself errored (quota, outage, ...).
            rec["evaluator_failed"] = rec["escalated"] and not rec["critique"]
        except Exception as e:
            rec["error"] = repr(e)
        rec["latency_s"] = time.perf_counter() - t0
        return rec

    async with sem:
        return await with_backoff(once, attempts, progress)


async def retrieval_job(case, sem, top_k):
    rec = {"question": case["question"], "expected_source": case["expected_source"], "rank": None, "error": None}
    async with sem:
        try:
            rows = await database.retrieve(case["question"])
            needle = case.get("expected_text")
            for i, (source, _chunk_no, content) in enumerate(rows[:top_k], start=1):
                if source == case["expected_source"] and (not needle or norm(needle) in norm(content)):
                    rec["rank"] = i
                    break
        except Exception as e:
            rec["error"] = repr(e)
    return rec


# =====================================================
# SUMMARIES
# =====================================================
def summarize_routing(records):
    ok = [r for r in records if not r["error"]]
    hits = sum(r["predicted"] == r["expected"] for r in ok)
    per_cat = {}
    for cat in ROUTES:
        rs = [r for r in ok if r["expected"] == cat]
        if rs:
            per_cat[cat] = {"correct": sum(r["predicted"] == cat for r in rs), "n": len(rs)}
    confusion = defaultdict(Counter)
    for r in ok:
        confusion[r["expected"]][r["predicted"]] += 1
    return {"correct": hits, "n": len(ok), "errors": len(records) - len(ok),
            "per_category": per_cat, "confusion": {k: dict(v) for k, v in confusion.items()}}


def summarize_full(records):
    evaluator_failed = sum(bool(r.get("evaluator_failed")) for r in records)
    records = [r for r in records if not r.get("evaluator_failed")]  # not a quality signal; reported separately
    ok = [r for r in records if not r["error"]]
    accepted = [r for r in ok if not r["escalated"]]
    graded = [r for r in ok if r.get("first_correct") is not None]

    first_ok = [r for r in graded if r["first_correct"]]
    first_bad = [r for r in graded if not r["first_correct"]]
    first_accepted = lambda r: r["retries"] == 0 and not r["escalated"]  # noqa: E731

    final_graded = [r for r in graded if not r["escalated"]]
    recovered = [r for r in first_bad if not r["escalated"] and r["final_correct"]]

    lat = [r["latency_s"] for r in ok]
    lat_by_route = {}
    for route in ROUTES:
        vals = [r["latency_s"] for r in ok if r["predicted"] == route]
        if vals:
            lat_by_route[route] = {"n": len(vals), "p50": percentile(vals, .5), "p95": percentile(vals, .95)}

    return {
        "n": len(records), "errors": len(records) - len(ok), "evaluator_failed": evaluator_failed,
        "routing": summarize_routing(records),
        "auto_resolved": len(accepted), "escalated": len(ok) - len(accepted),
        "retry_distribution": dict(Counter(r["retries"] for r in ok)),
        "graded": len(graded),
        "first_pass_correct": len(first_ok),
        "final_correct_among_auto_resolved": [sum(r["final_correct"] for r in final_graded), len(final_graded)],
        "evaluator_accepted_correct_first_pass": [sum(first_accepted(r) for r in first_ok), len(first_ok)],
        "evaluator_rejected_incorrect_first_pass": [sum(not first_accepted(r) for r in first_bad), len(first_bad)],
        "incorrect_first_pass_recovered_by_retry": [len(recovered), len(first_bad)],
        "latency_p50": percentile(lat, .5), "latency_p95": percentile(lat, .95),
        "latency_by_route": lat_by_route,
    }


def summarize_retrieval(records, top_k):
    ok = [r for r in records if not r["error"]]
    out = {"n": len(ok), "errors": len(records) - len(ok), "hit_at": {}}
    for k in sorted({1, 3, top_k}):
        if k <= top_k:
            out["hit_at"][k] = sum(r["rank"] is not None and r["rank"] <= k for r in ok)
    out["mrr"] = sum(1 / r["rank"] for r in ok if r["rank"]) / len(ok) if ok else None
    return out


def print_routing(s, title="ROUTING ACCURACY"):
    print(f"\n== {title} ==")
    print(f"overall: {rate(s['correct'], s['n'])}" + (f"   [{s['errors']} errored, excluded]" if s["errors"] else ""))
    for cat, v in s["per_category"].items():
        print(f"  {cat:<14} {rate(v['correct'], v['n'])}")
    misses = [(e, p, c) for e, row in s["confusion"].items() for p, c in row.items() if p != e]
    if misses:
        print("  misroutes (expected -> predicted): " + ", ".join(f"{e}->{p} x{c}" for e, p, c in misses))


def print_full(s):
    print_routing(s["routing"])
    ok = s["n"] - s["errors"]
    print("\n== PIPELINE OUTCOMES ==")
    print(f"cases: {s['n']}   errored: {s['errors']}")
    if s["evaluator_failed"]:
        print(f"!! {s['evaluator_failed']} case(s) excluded: the Gemini evaluator itself failed (likely quota), "
              "which the graph treats as an escalation. Rerun those when quota resets.")
    print(f"auto-resolved: {rate(s['auto_resolved'], ok)}")
    print(f"escalated to human: {rate(s['escalated'], ok)}")
    print(f"retries per case: {dict(sorted(s['retry_distribution'].items()))}")

    print(f"\n== ANSWER QUALITY (graded cases: {s['graded']}) ==")
    print(f"first-pass correct (evaluator off equivalent): {rate(s['first_pass_correct'], s['graded'])}")
    print(f"final correct, auto-resolved only:             {rate(*s['final_correct_among_auto_resolved'])}")
    print(f"evaluator accepted first-pass answers that were correct:   {rate(*s['evaluator_accepted_correct_first_pass'])}")
    print(f"evaluator rejected first-pass answers that were incorrect: {rate(*s['evaluator_rejected_incorrect_first_pass'])}")
    print(f"incorrect first-pass answers fixed by retry:  {rate(*s['incorrect_first_pass_recovered_by_retry'])}")

    print("\n== LATENCY (end-to-end seconds) ==")
    if s["latency_p50"] is not None:
        print(f"overall: p50 {s['latency_p50']:.1f}s   p95 {s['latency_p95']:.1f}s")
    for route, v in s["latency_by_route"].items():
        print(f"  {route:<14} n={v['n']:<3} p50 {v['p50']:.1f}s   p95 {v['p95']:.1f}s")


def print_retrieval(s):
    print("\n== RETRIEVAL (pgvector) ==")
    for k, hits in s["hit_at"].items():
        print(f"hit@{k}: {rate(hits, s['n'])}")
    if s["mrr"] is not None:
        print(f"MRR: {s['mrr']:.3f}")


# =====================================================
# MAIN
# =====================================================
def select_cases(args):
    cases = load_jsonl(EVAL_DIR / args.cases)
    if args.category:
        cases = [c for c in cases if c["category"] in args.category]
    if args.per_category:
        seen = Counter()
        kept = []
        for c in cases:
            if seen[c["category"]] < args.per_category:
                kept.append(c)
                seen[c["category"]] += 1
        cases = kept
    return cases * args.repeats


async def run_route(args, cases):
    sem = asyncio.Semaphore(args.concurrency)
    progress = Progress(len(cases))
    with contextlib.redirect_stdout(io.StringIO()):
        recs = await asyncio.gather(*(route_job(c, sem, args.timeout, progress, args.attempts) for c in cases))
    summary = summarize_routing(recs)
    print_routing(summary)
    return {"summary": summary, "records": recs}


async def run_full(args, cases):
    await database.init_db_and_mcp()
    app = workflow.compile(checkpointer=MemorySaver())  # in-memory: leaves checkpoints.db untouched
    sem = asyncio.Semaphore(args.concurrency)
    progress = Progress(len(cases))
    log(f"running {len(cases)} cases through the full graph...")
    with contextlib.redirect_stdout(io.StringIO()):
        recs = await asyncio.gather(*(full_job(app, c, sem, args.timeout, progress, args.attempts) for c in cases))

    log("grading answers...")
    model = config.genai.GenerativeModel(args.grader_model)
    by_id = {c["id"]: c for c in cases}
    sem_g = asyncio.Semaphore(args.concurrency)

    async def grade_rec(r):
        if r["error"]:
            return
        case = by_id[r["id"]]
        r["first_correct"] = await grade(model, case, r["first_answer"], sem_g)
        r["final_correct"] = await grade(model, case, r["final_answer"], sem_g)

    await asyncio.gather(*(grade_rec(r) for r in recs))
    summary = summarize_full(recs)
    print_full(summary)
    return {"summary": summary, "records": recs}


async def run_retrieval(args):
    path = EVAL_DIR / "retrieval_cases.jsonl"
    cases = load_jsonl(path)
    if not cases:
        print(f"\n== RETRIEVAL == skipped: no cases in {path} (see the comments in that file)")
        return None
    if config.conn is None:
        await database.init_db()
    sem = asyncio.Semaphore(args.concurrency)
    recs = await asyncio.gather(*(retrieval_job(c, sem, config.TOP_K) for c in cases))
    summary = summarize_retrieval(recs, config.TOP_K)
    print_retrieval(summary)
    return {"summary": summary, "records": recs}


async def main(args):
    if args.generator_model:
        config.LLM_MODEL = args.generator_model  # nodes read config.LLM_MODEL at call time
    result = {"timestamp": datetime.now().isoformat(timespec="seconds"), "args": vars(args),
              "generator_model": config.LLM_MODEL, "in_loop_evaluator_model": config.GEMINI_MODEL,
              "grader_model": args.grader_model}
    try:
        if args.mode in ("route", "full", "all"):
            cases = select_cases(args)
            log(f"{len(cases)} cases selected")
            if args.mode == "route":
                result["route"] = await run_route(args, cases)
            else:
                result["full"] = await run_full(args, cases)
        if args.mode in ("retrieval", "all"):
            result["retrieval"] = await run_retrieval(args)
    finally:
        if config.conn:
            await config.conn.close()

    out_dir = EVAL_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{args.mode}_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nper-case results saved to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["route", "full", "retrieval", "all"], default="all")
    p.add_argument("--cases", default="cases.jsonl", help="case file inside evals/")
    p.add_argument("--category", nargs="+", choices=ROUTES, help="only these expected routes")
    p.add_argument("--per-category", type=int, help="take only the first N cases of each category")
    p.add_argument("--repeats", type=int, default=1, help="run each case N times (LLM output is nondeterministic)")
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--timeout", type=float, default=180, help="per-case timeout in seconds")
    p.add_argument("--attempts", type=int, default=3, help="max tries per case when the API rate-limits (429)")
    p.add_argument("--generator-model", help="override config.LLM_MODEL for this run (does not edit config.py)")
    p.add_argument("--grader-model", default=config.GEMINI_MODEL,
                   help="Gemini model used to grade reference-based cases")
    args = p.parse_args()
    asyncio.run(main(args), loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector()))
