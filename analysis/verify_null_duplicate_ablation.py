"""Verify completed pilot scores with the repository's unmodified evaluator worker."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.null_duplicate_ablation import sha256, write_json
from evaluation.evaluation import execute_model


def check(job):
    return {"question_id":job["question_id"], "arms":job["arms"], **execute_model(
        job["sql"], job["gold"], job["db_path"], job["question_id"], 30)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="output/null_duplicate_ablation/pilot")
    parser.add_argument("--questions", default="data/bird_data/dev.json")
    args = parser.parse_args()
    out = Path(args.out)
    # Require completed local evaluation before accessing labels.
    expected = {r["question_id"]:r for r in json.loads((out / "evaluation.json").read_text(encoding="utf-8"))}
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    gold_path = str(Path(args.questions).resolve())
    assert sha256(gold_path) == manifest["source_sha256"][gold_path]
    gold = {r["question_id"]:r for r in json.loads(Path(gold_path).read_text(encoding="utf-8"))}
    inputs = json.loads((out / "inputs.json").read_text(encoding="utf-8"))
    jobs = []
    for record in inputs:
        qid, db = record["question_id"], record["db_id"]
        pair = json.loads((out / "pairs" / f"{qid}.json").read_text(encoding="utf-8"))
        sqls = {"input":record["sql"], **{arm:pair.get("arms",{}).get(arm,{}).get("final_sql",record["sql"])
                                          for arm in manifest["arms"]}}
        grouped = {}
        for arm, sql in sqls.items():
            grouped.setdefault(sql, []).append(arm)
        for sql, arms in grouped.items():
            jobs.append({"question_id":qid, "sql":sql, "arms":arms, "gold":gold[qid]["SQL"],
                         "db_path":str(Path(manifest["db_root"]) / db / (db+".sqlite"))})
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(check, job) for job in jobs]
        results = []
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 20 == 0 or len(results) == len(jobs):
                print(f"Verified {len(results)}/{len(jobs)} unique query pairs", flush=True)
    results.sort(key=lambda r: (r["question_id"], r["arms"]))
    totals = {arm:0 for arm in ["input", *manifest["arms"]]}
    mismatches = []
    for result in results:
        for arm in result["arms"]:
            totals[arm] += result["res"]
            local = expected[result["question_id"]]["scores"][arm]
            if local != result["res"]:
                mismatches.append({"question_id":result["question_id"], "arm":arm,
                                   "local":local, "repository":result["res"]})
    report = {"n":len(inputs), "unique_query_pairs":len(jobs), "correct":totals,
              "mismatches":mismatches, "results":results,
              "evaluator_sha256":sha256(ROOT / "evaluation/evaluation.py")}
    write_json(out / "repository_check.json", report)
    print(json.dumps({k:v for k,v in report.items() if k!="results"}, indent=2))


if __name__ == "__main__":
    main()
