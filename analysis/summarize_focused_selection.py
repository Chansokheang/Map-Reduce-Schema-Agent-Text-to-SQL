"""Summarize a completed focused experiment, never used during inference."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.selection_experiment import read_json, checkpoint_read, write_json

# Text after this marker is hand-written analysis; regeneration preserves it.
MANUAL_MARKER = "\n<!-- manual-analysis -->\n"


def main():
    out = ROOT / "output/focused_selection/v1"
    summary = read_json(out / "full_results/evaluation/summary.json")
    details = read_json(out / "full_results/evaluation/per_question.json")
    stats = summary["scores"]["focused"]
    calls = list((out / "calls").glob("*/*/response.json"))
    failures, retention = Counter(), Counter()
    probes, probe_errors, ambiguous = 0, 0, 0
    for p in (out / "pairs").glob("*.json"):
        outcome = checkpoint_read(p)["arms"]["focused"]
        if outcome["status"] == "failed":
            failures[outcome.get("error", "unknown").split(":", 1)[0]] += 1
        if outcome["status"] == "abstained":
            retention[outcome["reasoning"].split(":", 1)[0]] += 1
    for p in (out / "calls").glob("*/alignment/validated.json"):
        ambiguous += checkpoint_read(p)["ambiguous"]
    for p in (out / "calls").glob("*/probes.json"):
        for entry in checkpoint_read(p):
            probes += 1
            probe_errors += bool(entry["observation"]["error"])
    diag = {"requests_with_responses": len(calls), "failed_reviews_by_type": dict(failures),
        "retention_reasons": dict(retention), "ambiguous_alignments": ambiguous,
        "probes": probes, "probe_errors": probe_errors,
        "recoveries": len(stats["recovered_ids"]), "regressions": len(stats["regressed_ids"]),
        "original_failures_with_passing_candidate": summary["original_failures_with_passing_candidate"]}
    opportunities = [r for r in details if not r["scores"]["original"] and r["passing_candidate_count"]]
    missed = Counter()
    for row in opportunities:
        if row["scores"]["focused"]:
            continue
        outcome = checkpoint_read(out / "pairs" / f"{row['question_id']}.json")["arms"]["focused"]
        missed[outcome.get("reasoning", outcome["status"]).split(":", 1)[0]] += 1
    diag["missed_recoverable_failure_reasons"] = dict(missed)
    write_json(out / "full_results/evaluation/diagnostics.json", diag)
    manifest = read_json(out / "manifest.json")
    questions = {q["question_id"]: q for q in read_json(manifest["question_source"]["path"])}
    inputs = {q["question_id"]: q for q in read_json(out / "inputs.json")}
    changes = []
    for row in details:
        qid = row["question_id"]
        outcome = checkpoint_read(out / "pairs" / f"{qid}.json")["arms"]["focused"]
        if outcome["sql"] != inputs[qid]["selected"]:
            changes.append({"question_id": qid, "db_id": row["db_id"],
                "question": questions[qid]["question"], "evidence": questions[qid].get("evidence", ""),
                "original_sql": inputs[qid]["selected"], "focused_sql": outcome["sql"],
                "gold_sql": questions[qid]["SQL"], "scores": row["scores"],
                "reasoning": outcome["reasoning"]})
    write_json(out / "full_results/evaluation/changed_queries_offline_only.json", changes)
    original = summary["scores"]["original"]
    net = stats["correct"] - original["correct"]
    lines = ["# Completed focused-selection experiment", "",
        f"Original: {original['correct']}/{summary['n']} ({original['accuracy_percent']:.2f}%).",
        f"Focused verification: {stats['correct']}/{summary['n']} ({stats['accuracy_percent']:.2f}%).",
        f"Recovered {diag['recoveries']} original failures and regressed {diag['regressions']} successes; net {net:+d}.", "",
        "## Actual execution", "",
        f"- {summary['triggered']} questions triggered review; all {summary['n']} were exported and scored.",
        f"- {len(calls)} logged model responses; {sum(failures.values())} failed question reviews retained originals.",
        f"- {ambiguous} alignments marked ambiguity; {probes} database probes, {probe_errors} probe errors.",
        f"- Statuses: {json.dumps(summary['statuses']['focused'])}.",
        f"- Model usage: {json.dumps(summary['model_usage'])}.", "",
        "## Interpretation", "",
        "This experiment checks independent answer requirements against frozen candidate SQL. It does not",
        "regenerate queries, train a selector or provide gold at inference. Model judgments remain fallible.",
        "Recovery/regression counts refer to strict local execution matching, not semantic proof.",
        "The original-input gold was used; revised dev_20251106 inputs/gold were not substituted.",
        "Original sources were checked at export. Production behavior was not changed.", "",
        "## Why recoverable failures remained", "", "```json", json.dumps(dict(missed), indent=2), "```", "",
        "## Changed execution outcomes", "",
        f"Recovered IDs: {stats['recovered_ids']}", "", f"Regressed IDs: {stats['regressed_ids']}", "",
        "## By database", "", "| Database | Original correct | Focused correct | N |", "|---|---:|---:|---:|"]
    for db, v in stats["by_database"].items():
        lines.append(f"| {db} | {original['by_database'][db]['correct']} | {v['correct']} | {v['n']} |")
    findings = ROOT / "analysis/focused_selection_findings.md"
    manual = ""
    if findings.exists() and MANUAL_MARKER in findings.read_text(encoding="utf-8"):
        manual = findings.read_text(encoding="utf-8").split(MANUAL_MARKER, 1)[1]
    findings.write_text("\n".join(lines) + "\n" + MANUAL_MARKER + manual, encoding="utf-8")
    print(json.dumps(diag, indent=2))


if __name__ == "__main__":
    main()
