"""Summarize completed selection results; this is strictly post-evaluation analysis."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.selection_experiment import read_json, checkpoint_read, split_prediction, ARMS
from analysis.null_duplicate_ablation import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="output/selection_experiment/v2")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    evaluation = out / "full_results/evaluation"
    summary = read_json(evaluation / "summary.json")  # Refuse incomplete evaluation.
    details = read_json(evaluation / "per_question.json")
    manifest = read_json(out / "manifest.json")
    inputs = {r["question_id"]: r for r in read_json(out / "inputs.json")}
    assert len(details) == len(inputs) == summary["n"]
    gold_path = Path(manifest["question_source"]["path"])
    assert sha256(gold_path) == manifest["question_source"]["sha256"]
    gold = {r["question_id"]: r for r in read_json(gold_path)}
    outputs = {arm: read_json(out / "full_results" / f"selected_{arm}.json") for arm in ARMS}
    recoverable = {r["question_id"] for r in details if not r["scores"]["original"] and r["passing_candidate_count"]}
    report = {"n": summary["n"], "recoverable_failures": len(recoverable), "arms": {}}
    changes = []
    for arm in ARMS:
        recovered = summary["scores"][arm]["recovered_ids"]
        regressed = summary["scores"][arm]["regressed_ids"]
        report["arms"][arm] = {"recoveries": len(recovered), "regressions": len(regressed),
            "net_correct_change": len(recovered) - len(regressed),
            "recoveries_with_passing_saved_candidate": len(set(recovered) & recoverable),
            "remaining_recoverable_failures": len(recoverable - set(recovered))}
        for result in details:
            qid, db = result["question_id"], result["db_id"]
            selected_sql = split_prediction(outputs[arm][str(qid)], db)
            if selected_sql == inputs[qid]["selected"]:
                continue
            outcome = checkpoint_read(out / "pairs" / f"{qid}.json")["arms"][arm]
            before, after = result["scores"]["original"], result["scores"][arm]
            changes.append({"arm": arm, "question_id": qid, "db_id": db,
                "impact": "recovered" if after > before else "regressed" if after < before else "same_score",
                "original_correct": before, "new_correct": after,
                "question": inputs[qid]["question"], "evidence": inputs[qid]["evidence"],
                "original_sql": inputs[qid]["selected"], "selected_sql": selected_sql,
                "gold_sql": gold[qid]["SQL"], "reasoning": outcome.get("reasoning", "")})
    report["disagreement_vs_control"] = {
        "better_ids": [r["question_id"] for r in details if r["scores"]["disagreement"] > r["scores"]["control"]],
        "worse_ids": [r["question_id"] for r in details if r["scores"]["disagreement"] < r["scores"]["control"]]}
    responses = [read_json(p) for p in (out / "calls").glob("*/*/*.response.json")]
    report["models_in_cli_usage"] = dict(Counter(model for r in responses for model in r.get("modelUsage", {})))
    report["original_prediction_sources_unchanged"] = all(sha256(p) == expected for p, expected in manifest["prediction_sources"].items())
    assert report["original_prediction_sources_unchanged"]
    for arm in ARMS:
        assert summary["scores"][arm]["correct"] - summary["scores"]["original"]["correct"] == report["arms"][arm]["net_correct_change"]
    write_json(evaluation / "selection_analysis.json", report)
    if changes:
        with (evaluation / "changed_queries.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(changes[0]))
            writer.writeheader()
            writer.writerows(changes)
    lines = ["# Completed selection experiment", "",
        "All questions were processed before gold scoring. Both judges selected only frozen candidate SQL; no fixer ran. Original prediction hashes are unchanged.", "",
        "| Version | Correct | EX | Recoveries | Regressions | Net correct change |",
        "|---|---:|---:|---:|---:|---:|"]
    for name in ("original", *ARMS):
        value = summary["scores"][name]
        counts = report["arms"].get(name, {"recoveries": 0, "regressions": 0, "net_correct_change": 0})
        lines.append(f"| {name} | {value['correct']}/{summary['n']} | {value['accuracy_percent']:.2f}% | {counts['recoveries']} | {counts['regressions']} | {counts['net_correct_change']:+d} |")
    lines += ["", f"The execution-based gate reviewed {summary['triggered']} questions, including {summary['original_correct_in_trigger']} originally correct answers.",
        f"The fresh evaluation found {len(recoverable)} original failures with a passing saved candidate.", ""]
    for arm, value in report["arms"].items():
        lines.append(f"- {arm}: recovered {value['recoveries_with_passing_saved_candidate']} of these opportunities; {value['remaining_recoverable_failures']} remain.")
    comparison = report["disagreement_vs_control"]
    lines += ["", f"The new criteria beat the control on {len(comparison['better_ids'])} questions and lose to it on {len(comparison['worse_ids'])}.",
        "", "## Per-database correct answers", "",
        "| Database | Questions | Original | Control | Disagreement | Disagreement net |",
        "|---|---:|---:|---:|---:|---:|"]
    for db, base in summary["scores"]["original"]["by_database"].items():
        control = summary["scores"]["control"]["by_database"][db]["correct"]
        treatment = summary["scores"]["disagreement"]["by_database"][db]["correct"]
        lines.append(f"| {db} | {base['n']} | {base['correct']} | {control} | {treatment} | {treatment-base['correct']:+d} |")
    lines += ["", "## Interpretation and provenance", "",
        "- The control uses current production criteria with the new shared interface; it is not a replay of the historical judge.",
        "- The CLI model alias was sonnet. Models recorded in response usage: " + ", ".join(report["models_in_cli_usage"]) + ". Both arms used the same setup.",
        "- Dev was inspected during prompt development. These results are development evidence, not an independent private-test generalization estimate.",
        "- The CLI may use multiple internal model turns for one structured request. Its reported cost is not necessarily subscription billing.",
        f"- CLI usage summary: {json.dumps(summary['model_usage'])}",
        "- Execution matches do not prove semantic equivalence. The unmodified evaluator scores errors/timeouts as failures.",
        "", "The offline changed_queries.csv includes gold SQL and evaluation labels. Never supply it to an inference model.",
        "See summary.json for full scores/statuses, selection_analysis.json for comparisons, and per_question.json for all questions.", ""]
    (evaluation / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
