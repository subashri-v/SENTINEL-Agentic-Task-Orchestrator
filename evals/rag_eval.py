"""
RAG generation evaluation (the retrieval half lives in run_eval.py --mode retrieval).

For each labeled question the real RAG agent retrieves + answers, then a judge model (a different model family
from the generator by default) scores:
  faithfulness   every claim in the answer is supported by the retrieved chunks
  correctness    the answer matches the reference answer
  citations      the sources cited are from the right document
and for questions the corpus cannot answer:
  abstention     the agent declines instead of inventing an answer

    python -m evals.rag_eval                 # 24 answerable + 8 unanswerable questions
    python -m evals.rag_eval --limit 4       # quick smoke test
"""
import argparse
import asyncio
import contextlib
import io
import json
import re
import selectors
import time
from datetime import datetime

from evals.run_eval import (EVAL_DIR, USAGE, config, database, install_usage_counter, load_jsonl, log, nodes,
                            norm, percentile, rate)

FAITHFUL_PROMPT = """You are checking an AI answer for faithfulness to source text.
CONTEXT is the only allowed source. Decide whether EVERY factual claim in the ANSWER is supported by the CONTEXT.
An answer that says the information is not available in the documents is faithful.
Respond with JSON only: {{"faithful": true or false, "unsupported_claims": ["..."]}}

CONTEXT:
{context}

ANSWER:
{answer}"""

CORRECT_PROMPT = """You are grading an AI answer against a reference answer.
correct=true if the answer contains the key facts of the reference and does not contradict it (extra detail is fine).
correct=false if it is wrong, misses the key facts, or declines to answer.
Respond with JSON only: {{"correct": true or false, "reason": "..."}}

QUESTION: {question}
REFERENCE: {reference}
ANSWER: {answer}"""

ABSTAIN_PROMPT = """The QUESTION cannot be answered from the user's documents. Did the ANSWER decline, i.e. say the
information is not in the documents, without asserting an answer to the question? abstained=false if the answer gives
a substantive answer to the question (even from general knowledge).
Respond with JSON only: {{"abstained": true or false}}

QUESTION: {question}
ANSWER: {answer}"""


def parse_json(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    text = text[text.find("{"): text.rfind("}") + 1]
    # judges often quote LaTeX (\frac, \sqrt): double any backslash that is not a valid JSON escape
    return json.loads(re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text))


async def judge(model, prompt, sem):
    for attempt in range(3):
        try:
            async with sem:
                resp = await config.llm_client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": prompt}], temperature=0)
            return parse_json(resp.choices[0].message.content)
        except Exception as e:
            if attempt == 2:
                log(f"[judge] failed: {e!r}"[:200])
            else:
                await asyncio.sleep(10 * (attempt + 1))
    return None


async def answer_job(case, kind, sem, timeout, progress):
    rec = {"kind": kind, "question": case["question"], "error": None, "expected_source": case.get("expected_source"),
           "reference": case.get("reference")}
    async with sem:
        for attempt in range(3):
            usage = {"calls": 0, "tokens": 0}
            USAGE.set(usage)
            rec["error"] = None
            t0 = time.perf_counter()
            try:
                rows = await database.retrieve(case["question"])
                rec["context"] = "\n\n".join(f"[{src} chunk {n}]\n{text}" for src, n, text in rows)
                out = await asyncio.wait_for(nodes.node_rag_agent({"question": case["question"]}), timeout)
                rec["answer"], rec["citations"] = out["answer"], out["citations"]
                if kind == "answerable":
                    needle = norm(case["expected_text"])
                    rec["retrieved_gold"] = any(s == case["expected_source"] and needle in norm(t) for s, _, t in rows)
            except Exception as e:
                rec["error"] = repr(e)
            rec["latency_s"] = time.perf_counter() - t0
            rec["llm_tokens"] = usage["tokens"]
            if not (rec["error"] and "RateLimitError" in rec["error"]):
                break
            await asyncio.sleep(15 * (attempt + 1))
    progress["done"] += 1
    log(f"  [{progress['done']}/{progress['total']}] {kind} {'ERROR' if rec['error'] else 'ok'} ({rec['latency_s']:.0f}s)")
    return rec


async def judge_record(rec, model, sem):
    if rec["error"] or rec.get("answer") is None:
        return
    if rec["kind"] == "answerable":
        f, c = await asyncio.gather(
            judge(model, FAITHFUL_PROMPT.format(context=rec["context"], answer=rec["answer"]), sem),
            judge(model, CORRECT_PROMPT.format(question=rec["question"], reference=rec["reference"], answer=rec["answer"]), sem))
        rec["faithful"] = f.get("faithful") if f else None
        rec["unsupported_claims"] = f.get("unsupported_claims") if f else None
        rec["correct"] = c.get("correct") if c else None
        cites = rec.get("citations") or []
        rec["cited_gold_source"] = any(x.startswith(rec["expected_source"]) for x in cites)
        rec["cited_only_gold_source"] = bool(cites) and all(x.startswith(rec["expected_source"]) for x in cites)
    else:
        a = await judge(model, ABSTAIN_PROMPT.format(question=rec["question"], answer=rec["answer"]), sem)
        rec["abstained"] = a.get("abstained") if a else None


def count(recs, key, cond=lambda r: True):
    pool = [r for r in recs if not r["error"] and r.get(key) is not None and cond(r)]
    return [sum(bool(r[key]) for r in pool), len(pool)]


def summarize(records):
    ans = [r for r in records if r["kind"] == "answerable"]
    una = [r for r in records if r["kind"] == "unanswerable"]
    ok = [r for r in records if not r["error"]]
    return {
        "answerable_n": len(ans), "unanswerable_n": len(una), "errors": len(records) - len(ok),
        "faithful": count(ans, "faithful"),
        "correct": count(ans, "correct"),
        "correct_when_gold_retrieved": count(ans, "correct", lambda r: r["retrieved_gold"]),
        "correct_when_gold_missed": count(ans, "correct", lambda r: not r["retrieved_gold"]),
        "cited_gold_source": count(ans, "cited_gold_source"),
        "cited_only_gold_source": count(ans, "cited_only_gold_source"),
        "abstained": count(una, "abstained"),
        "latency_p50": percentile([r["latency_s"] for r in ok], .5),
        "latency_p95": percentile([r["latency_s"] for r in ok], .95),
        "tokens_mean": sum(r["llm_tokens"] for r in ok) / len(ok) if ok else None,
    }


def report(s):
    print("\n== RAG GENERATION QUALITY ==")
    print(f"answerable questions: {s['answerable_n']}   unanswerable: {s['unanswerable_n']}   errored: {s['errors']}")
    print(f"faithfulness (all claims supported by retrieved text): {rate(*s['faithful'])}")
    print(f"answer correctness vs reference:                       {rate(*s['correct'])}")
    print(f"  when the gold chunk was retrieved:                   {rate(*s['correct_when_gold_retrieved'])}")
    print(f"  when it was not retrieved:                           {rate(*s['correct_when_gold_missed'])}")
    print(f"cited the correct source document:                     {rate(*s['cited_gold_source'])}")
    print(f"cited ONLY the correct source document:                {rate(*s['cited_only_gold_source'])}")
    print(f"abstained on unanswerable questions:                   {rate(*s['abstained'])}")
    k, n = s["abstained"]
    if n:
        print(f"  => answered questions it should have declined:       {rate(n - k, n)}")
    if s["tokens_mean"] is not None:
        print(f"\ncost/latency: {s['tokens_mean']:.0f} generator tokens per query, "
              f"p50 {s['latency_p50']:.1f}s, p95 {s['latency_p95']:.1f}s")


async def main(args):
    install_usage_counter()
    if args.generator_model:
        config.LLM_MODEL = args.generator_model
    if args.judge_model == config.LLM_MODEL:
        log("warning: the judge is the same model as the generator; prefer a different model family")

    answerable = load_jsonl(EVAL_DIR / "retrieval_cases.jsonl")
    unanswerable = load_jsonl(EVAL_DIR / "unanswerable_cases.jsonl")
    if args.limit:
        answerable, unanswerable = answerable[:args.limit], unanswerable[:max(1, args.limit // 2)]

    await database.init_db()
    sem = asyncio.Semaphore(args.concurrency)
    progress = {"done": 0, "total": len(answerable) + len(unanswerable)}
    jobs = ([answer_job(c, "answerable", sem, args.timeout, progress) for c in answerable] +
            [answer_job(c, "unanswerable", sem, args.timeout, progress) for c in unanswerable])
    log(f"answering {len(jobs)} questions with {config.LLM_MODEL}...")
    with contextlib.redirect_stdout(io.StringIO()):
        records = await asyncio.gather(*jobs)

    log(f"judging with {args.judge_model}...")
    await asyncio.gather(*(judge_record(r, args.judge_model, sem) for r in records))
    summary = summarize(records)
    report(summary)

    out_dir = EVAL_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"rag_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({"timestamp": datetime.now().isoformat(timespec="seconds"), "args": vars(args),
                               "generator_model": config.LLM_MODEL, "judge_model": args.judge_model,
                               "embedding_model": config.EMBEDDING_MODEL, "summary": summary, "records": records},
                              indent=2, default=str), encoding="utf-8")
    print(f"\nper-question results saved to {out}")
    await config.conn.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--judge-model", default="qwen/qwen3.8-27b", help="Groq model used as the judge")
    p.add_argument("--generator-model", help="override config.LLM_MODEL for this run")
    p.add_argument("--limit", type=int, help="only the first N answerable questions (and N/2 unanswerable)")
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--timeout", type=float, default=120)
    asyncio.run(main(p.parse_args()), loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector()))
