"""Offline, labelled review of the 88 saved-candidate selection opportunities."""
from collections import Counter
import csv
import difflib
import hashlib
import json
from pathlib import Path
import sys
import sqlglot

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.selection.judge import SQLJudge

GROUPS = {
    "output_attributes_and_representation":[58,165,248,280,288,323,436,469,630,866,873,898,907,936,967,978,1149,1177,1212,1406],
    "field_or_relationship_meaning":[56,77,412,449,453,667,851,887,891,952,1037,1175,1195,1238,1418,1419,1422,1510,1511,1524,1527],
    "population_and_join_membership":[48,54,83,107,214,239,326,335,337,384,557,689,1450],
    "aggregation_grain":[15,72,84,131,998,1218,1282,1458,1525],
    "predicates_arithmetic_and_types":[108,376,486,879,943,987,1093,1249,1260,1261,852,1108],
    "top_n_ties_and_row_selection":[101,476,580,590,794,1032],
    "null_and_zero_policy":[618,637,648,839,842,856,1178],
}


def normalize(sql):
    try:
        return sqlglot.parse_one(sql, read="sqlite").sql(dialect="sqlite", normalize=True)
    except Exception:
        return sql.strip()


def main():
    audit = ROOT / "analysis/v6_accuracy_audit/system_sqlite"
    metadata = json.loads((audit / "summary.json").read_text(encoding="utf-8"))
    source_hashes = {}
    for filename, expected in metadata["input_sha256"].items():
        if filename.startswith("output"):
            actual = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
            assert actual == expected, f"Saved predictions changed since audit: {filename}"
            source_hashes[filename] = actual
    rows = [json.loads(line) for line in (audit / "details.jsonl").read_text(encoding="utf-8").splitlines()]
    cases = [r for r in rows if r["results"]["selected"]["category"] != "strict" and r["correct_candidates"]]
    assignments = {qid:group for group,ids in GROUPS.items() for qid in ids}
    assert len(cases) == len(assignments) == sum(map(len,GROUPS.values())) == 88
    assert set(assignments) == {r["question_id"] for r in cases}
    records = []
    for r in sorted(cases, key=lambda r:r["question_id"]):
        selected = r["results"]["selected"]
        pool = {k:v for k,v in r["results"].items() if k.startswith("candidate_")}
        nearest = max(r["correct_candidates"], key=lambda k:difflib.SequenceMatcher(
            None, selected["sql"].lower(), pool[k]["sql"].lower()).ratio())
        records.append({"question_id":r["question_id"], "db_id":r["db_id"], "question":r["question"],
            "evidence":r["evidence"], "manual_primary_difference":assignments[r["question_id"]],
            "execution_diagnostic":selected["category"], "selected_sql":selected["sql"],
            "passing_saved_candidates":r["correct_candidates"], "nearest_passing_candidate":nearest,
            "nearest_passing_sql":pool[nearest]["sql"],
            "selected_exact_matches":[k for k,v in pool.items() if selected["sql"].strip()==v["sql"].strip()],
            "selected_parser_normalized_matches":[k for k,v in pool.items() if normalize(selected["sql"])==normalize(v["sql"])],
            "current_schema_gate_would_omit":not SQLJudge._candidates_differ_on_columns([v for v in pool.values() if not v["error"]]),
            "plurality_would_pass":r["result_plurality_correct"],
            "result_groups":r["candidate_result_groups"], "selected_execution":selected,
            "all_saved_candidates":pool,
            "offline_only":"Passing labels and gold-relative result diagnostics must never be supplied to inference."})
    disagreement = [r for r in rows if len(r["candidate_result_groups"])>1]
    summary = {"n":88, "source_prediction_hashes":source_hashes,
        "manual_primary_difference_counts":{k:len(v) for k,v in GROUPS.items()},
        "diagnostic_categories":dict(Counter(r["execution_diagnostic"] for r in records)),
        "number_of_passing_candidates":dict(Counter(len(r["passing_saved_candidates"]) for r in records)),
        "by_database":dict(Counter(r["db_id"] for r in records)),
        "plurality_recoveries_within_88":sum(r["plurality_would_pass"] for r in records),
        "plurality_full_dev_correct":sum(r["result_plurality_correct"] for r in rows),
        "exact_selected_membership":sum(bool(r["selected_exact_matches"]) for r in records),
        "parser_normalized_selected_membership":sum(bool(r["selected_parser_normalized_matches"]) for r in records),
        "schema_gate_omission_ids":[r["question_id"] for r in records if r["current_schema_gate_would_omit"]],
        "same_width_as_a_passing_candidate":sum(any(len(r["results"][k]["columns"])==len(r["results"]["selected"]["columns"]) for k in r["correct_candidates"]) for r in cases),
        "all_passing_candidates_same_physical_row_count":sum(all(r["results"][k]["row_count"]==r["results"]["selected"]["row_count"] for k in r["correct_candidates"]) for r in cases),
        "gold_empty_cases":sum(r["gold_row_count"]==0 for r in cases),
        "selected_empty_cases":sum(r["results"]["selected"]["row_count"]==0 for r in cases),
        "disagreement_trigger":{"n":len(disagreement),
            "currently_correct":sum(r["results"]["selected"]["category"]=="strict" for r in disagreement),
            "recoverable_failures":sum(r["results"]["selected"]["category"]!="strict" and bool(r["correct_candidates"]) for r in disagreement)},
        "scope":"Retrospective dev diagnosis. Manual categories are primary observed differences, may overlap conceptually, and do not prove the historical judge's causal reasoning."}
    out = ROOT / "analysis/selection_failure_review"
    out.mkdir(exist_ok=True)
    for filename,data in [("summary.json",summary),("offline_labelled_cases.json",records)]:
        (out / filename).write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding="utf-8")
    fields=["question_id","db_id","manual_primary_difference","question","evidence","execution_diagnostic",
            "selected_sql","nearest_passing_candidate","nearest_passing_sql","passing_saved_candidates"]
    with (out / "cases.csv").open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({k:json.dumps(r[k]) if isinstance(r[k],list) else r[k] for k in fields})
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
