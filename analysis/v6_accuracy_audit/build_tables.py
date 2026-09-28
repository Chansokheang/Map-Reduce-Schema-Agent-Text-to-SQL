"""Build review tables from a completed system-runtime audit."""
from collections import Counter
import csv
import json
from pathlib import Path
import re


def main():
    root = Path(__file__).resolve().parent
    source = root / "system_sqlite"
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    rows = sorted(
        map(json.loads, (source / "details.jsonl").read_text(encoding="utf-8").splitlines()),
        key=lambda r: r["question_id"],
    )
    fields = ["question_id", "database", "difficulty", "category", "question", "evidence",
              "gold_sql", "selected_sql", "gold_rows", "selected_rows", "gold_columns",
              "selected_columns", "correct_candidates", "candidate_answer_groups",
              "projection_mapping", "pred_only_sample", "gold_only_sample", "gold_error", "selected_error"]
    with (root / "cases.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            selected = r["results"]["selected"]
            writer.writerow(dict(
                question_id=r["question_id"], database=r["db_id"], difficulty=r["difficulty"],
                category=selected["category"], question=r["question"], evidence=r["evidence"],
                gold_sql=r["SQL"], selected_sql=selected["sql"], gold_rows=r["gold_row_count"],
                selected_rows=selected["row_count"], gold_columns=json.dumps(r["gold_columns"]),
                selected_columns=json.dumps(selected["columns"]),
                correct_candidates="; ".join(r["correct_candidates"]),
                candidate_answer_groups=json.dumps(r["candidate_result_groups"]),
                projection_mapping=json.dumps(selected["mapping"]),
                pred_only_sample=json.dumps(selected["pred_only"], ensure_ascii=False),
                gold_only_sample=json.dumps(selected["gold_only"], ensure_ascii=False),
                gold_error=r["gold_error"], selected_error=selected["error"],
            ))
    duplicate_ids = set(map(int, re.findall(
        r'"question_id"\s*:\s*(\d+)',
        (root / "Ground truth contains duplicates.txt").read_text(encoding="utf-8"))))
    supplement = {
        "all_five_agree": sum(len(r["candidate_result_groups"]) == 1 and
            len(r["candidate_result_groups"][0]) == 5 for r in rows),
        "all_five_agree_but_wrong_excluding_gold_errors": sum(
            len(r["candidate_result_groups"]) == 1 and len(r["candidate_result_groups"][0]) == 5
            and not r["correct_candidates"] and not r["gold_error"] for r in rows),
        "selected_wrong_no_passing_candidate": sum(not r["correct_candidates"] and
            r["results"]["selected"]["category"] != "strict" for r in rows),
        "selected_sql_absent_from_saved_candidates": sum(not any(
            r["results"]["selected"]["sql"].strip() == v["sql"].strip()
            for n, v in r["results"].items() if n.startswith("candidate_")) for r in rows),
        "duplicate_document_categories": dict(Counter(
            r["results"]["selected"]["category"] for r in rows if r["question_id"] in duplicate_ids)),
        "strict_pass_despite_different_row_counts": sum(
            r["results"]["selected"]["category"] == "strict" and
            r["results"]["selected"]["row_count"] != r["gold_row_count"] for r in rows),
        "selected_category_ids": {c: [r["question_id"] for r in rows
            if r["results"]["selected"]["category"] == c]
            for c in summary["categories"]["selected"] if c != "strict"},
    }
    (root / "supplement.json").write_text(json.dumps(supplement, indent=2), encoding="utf-8")
    compact = dict(summary)
    compact.pop("input_sha256")
    for key in ["selected_wrong_candidate_correct", "selected_correct_no_candidate_correct", "all_candidates_same_wrong"]:
        compact[key] = len(compact[key])
    compact["refined_comparison"] = {k:len(v) if isinstance(v,list) else v
                                     for k,v in compact["refined_comparison"].items()}
    print(json.dumps(compact, indent=2))
    print(json.dumps({k:v for k,v in supplement.items() if k != "selected_category_ids"}, indent=2))


if __name__ == "__main__":
    main()
