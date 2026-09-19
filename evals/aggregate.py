"""
Merge several run_eval result files into one report. For each case id the latest error-free record wins,
so a big run can be split across days/quotas and re-aggregated. Refuses to mix generator models.

    python -m evals.aggregate evals/results/route_A.json evals/results/route_B.json
"""
import argparse
import json
from pathlib import Path

from evals.run_eval import EVAL_DIR, load_jsonl, matches_expected, print_full, print_routing, summarize_full, summarize_routing


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+")
    p.add_argument("--exclude", nargs="+", default=[], metavar="CATEGORY",
                   help="drop cases whose expected route is one of these, e.g. SUMMARY_AGENT")
    p.add_argument("--regrade", action="store_true",
                   help="recompute programmatic (expect_all) grades from the recorded answers using the current cases.jsonl")
    p.add_argument("--allow-mixed-models", action="store_true")
    args = p.parse_args()

    runs = sorted((json.loads(Path(f).read_text(encoding="utf-8")) for f in args.files), key=lambda d: d["timestamp"])
    def check_models(section):
        # routing never calls the Gemini evaluator, so only the generator has to match there
        keys = ("generator_model",) if section == "route" else ("generator_model", "in_loop_evaluator_model")
        used = {tuple(d.get(k) for k in keys) for d in runs if section in d}
        if len(used) > 1 and not args.allow_mixed_models:
            raise SystemExit(f"[{section}] result files come from different models {used}. "
                             "Pass --allow-mixed-models to override.")
        return used.pop() if used else None

    cases = {c["id"]: c for c in load_jsonl(EVAL_DIR / "cases.jsonl")}

    def regrade(r):
        case = cases.get(r["id"], {})
        if args.regrade and "expect_all" in case and not r["error"]:
            for key, ans in (("first_correct", "first_answer"), ("final_correct", "final_answer")):
                r[key] = matches_expected(case, r[ans]) if r.get(ans) is not None else False
        return r

    for section, summarize, show in (("route", summarize_routing, print_routing), ("full", summarize_full, print_full)):
        used = check_models(section)
        merged = {}
        for d in runs:
            for r in d.get(section, {}).get("records", []):
                if r["expected"] in args.exclude:
                    continue
                if not r["error"] or r["id"] not in merged:
                    merged[r["id"]] = regrade(r)
        if merged:
            print(f"\n######## {section.upper()} (merged from {len(merged)} unique cases) ########")
            show(summarize(list(merged.values())))


if __name__ == "__main__":
    main()
