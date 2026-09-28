"""Recheck diagnostic passes using the repository's unmodified worker."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluation.evaluation import execute_model


def check(row):
    db = row["db_id"]
    return execute_model(row["results"]["selected"]["sql"], row["SQL"],
        str(ROOT / "data/bird_data/dev_databases" / db / (db+".sqlite")),
        row["question_id"], 30)


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent
    rows = list(map(json.loads, (folder / "system_sqlite/details.jsonl").read_text(encoding="utf-8").splitlines()))
    passes = [r for r in rows if r["results"]["selected"]["category"] == "strict"]
    with ProcessPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(check, passes))
    failures = [r for r in results if r["res"] != 1]
    result = {"rechecked_diagnostic_passes":len(passes), "repository_worker_failures":failures}
    (folder / "repository_pass_recheck.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
