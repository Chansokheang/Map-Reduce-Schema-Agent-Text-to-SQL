"""Evaluate a full-pipeline run on dev: selected.json vs selected_postprocessed.json.

Predictions are matched to gold by question key, so partial runs (e.g. -b 14 28) are scored
against the right gold SQL. Scoring uses the repository's unmodified
evaluation.evaluation.execute_model (set-of-rows execution match, same as run_evaluation.sh).
Reads gold from dev.json; for local evaluation only.

Usage (from the repository root):
  python -m experiments.full_pipeline.evaluate_outputs --output-dir output/full_v2
  python -m experiments.full_pipeline.evaluate_outputs --output-dir output/full_v2 --start 0 --end 10
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DELIMITER = "\t----- bird -----\t"
LEVELS = ("simple", "moderate", "challenging")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def score(job):
    from evaluation.evaluation import execute_model
    result = execute_model(job["sql"], job["gold"], job["db_path"], job["key"], job["timeout"])
    return job["key"], job["sql"], result["res"]


def split(value):
    sql, db_id = value.rsplit(DELIMITER, 1)
    return sql, db_id


def accuracy_table(keys, correct, questions):
    rows = {}
    for level in LEVELS + ("total",):
        chosen = [k for k in keys if level == "total" or questions[int(k)].get("difficulty") == level]
        n = len(chosen)
        rows[level] = {"count": n, "correct": sum(correct[k] for k in chosen),
                       "accuracy": round(100 * sum(correct[k] for k in chosen) / n, 2) if n else None}
    return rows


def print_table(title, table):
    print(f"\n{title}")
    print("{:12} {:>12} {:>12} {:>12} {:>12}".format("", *LEVELS, "total"))
    print("{:12} {:>12} {:>12} {:>12} {:>12}".format("count", *[table[l]["count"] for l in LEVELS + ("total",)]))
    print("{:12} {:>12} {:>12} {:>12} {:>12}".format("correct", *[table[l]["correct"] for l in LEVELS + ("total",)]))
    print("{:12} {:>12} {:>12} {:>12} {:>12}".format(
        "accuracy", *["-" if table[l]["accuracy"] is None else f"{table[l]['accuracy']:.2f}" for l in LEVELS + ("total",)]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True, help="Pipeline output folder containing selected.json")
    parser.add_argument("--pp-out", default=None, help="Default: <output-dir>/question_form_postprocess")
    parser.add_argument("--dev-json", default="data/bird_data/dev.json", help="Questions with gold SQL and difficulty")
    parser.add_argument("--databases-dir", default="data/bird_data/dev_databases")
    parser.add_argument("--start", type=int, default=None, help="First question index (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="Last question index (exclusive)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--report-dir", default=None, help="Default: <output-dir>/evaluation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    pp_out = Path(args.pp_out) if args.pp_out else output_dir / "question_form_postprocess"
    questions = read_json(args.dev_json)
    if any("SQL" not in q for q in questions[:1]):
        sys.exit("The questions file has no gold SQL; evaluation needs dev.json (or another file with SQL).")
    files = {"selected.json": output_dir / "selected.json",
             "selected_postprocessed.json": pp_out / "selected_postprocessed.json"}
    predictions = {name: read_json(path) for name, path in files.items() if path.exists()}
    if "selected.json" not in predictions:
        sys.exit(f"Not found: {files['selected.json']}")
    if "selected_postprocessed.json" not in predictions:
        print(f"Note: {files['selected_postprocessed.json']} not found; evaluating selected.json only.")

    start = args.start if args.start is not None else 0
    end = args.end if args.end is not None else len(questions)
    keys = sorted((k for k in predictions["selected.json"] if start <= int(k) < end), key=int)
    if not keys:
        sys.exit(f"selected.json has no questions in [{start}, {end}).")
    if args.start is None:
        start = int(keys[0])
    if args.end is None:
        end = int(keys[-1]) + 1
    missing = [str(i) for i in range(start, min(end, len(questions))) if str(i) not in predictions["selected.json"]]
    if missing:
        print(f"Warning: {len(missing)} questions in [{start}, {end}) are not in selected.json and are not scored "
              f"(first: {missing[:5]}).")
    for name, preds in predictions.items():
        absent = [k for k in keys if k not in preds]
        if absent:
            sys.exit(f"{name} is missing {len(absent)} of the evaluated keys (first: {absent[:5]}).")

    jobs, seen = [], set()
    for key in keys:
        q = questions[int(key)]
        for name, preds in predictions.items():
            sql, db_id = split(preds[key])
            if db_id != q["db_id"]:
                sys.exit(f"{name} key {key}: database {db_id} does not match dev.json ({q['db_id']}).")
            if (key, sql) not in seen:
                seen.add((key, sql))
                jobs.append({"key": key, "sql": sql, "gold": q["SQL"], "timeout": args.timeout,
                             "db_path": str(Path(args.databases_dir) / db_id / f"{db_id}.sqlite")})

    print(f"Scoring {len(keys)} questions in [{start}, {end}) ({len(jobs)} unique SQL executions)...", flush=True)
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(score, job) for job in jobs]):
            key, sql, res = future.result()
            results[(key, sql)] = res

    correct = {name: {k: results[(k, split(preds[k])[0])] for k in keys} for name, preds in predictions.items()}
    tables = {name: accuracy_table(keys, correct[name], questions) for name in predictions}
    for name, table in tables.items():
        print_table(f"=== {name} ===", table)

    report = {"output_dir": str(output_dir), "range": [start, end], "evaluated_keys": keys,
              "accuracy": tables, "per_question": []}
    pp_results = pp_out / "results"
    if "selected_postprocessed.json" in predictions:
        orig, post = correct["selected.json"], correct["selected_postprocessed.json"]
        recovered = [k for k in keys if post[k] > orig[k]]
        regressed = [k for k in keys if post[k] < orig[k]]
        print(f"\nPost-process effect: recovered {len(recovered)} {recovered}, regressed {len(regressed)} {regressed}, "
              f"net {len(recovered) - len(regressed):+d}")
        report.update(recovered=recovered, regressed=regressed)
        print("\n{:>6} {:<24} {:<24} {:>9} {:>14}".format("key", "form", "status", "original", "postprocessed"))
        for k in keys:
            info = read_json(pp_results / f"{k}.json") if (pp_results / f"{k}.json").exists() else {}
            row = {"key": k, "difficulty": questions[int(k)].get("difficulty"), "form": info.get("form"),
                   "status": info.get("status"), "original_correct": orig[k], "postprocessed_correct": post[k]}
            report["per_question"].append(row)
            if info.get("form") is not None:
                print("{:>6} {:<24} {:<24} {:>9} {:>14}".format(k, str(row["form"]), str(row["status"]),
                                                                orig[k], post[k]))
        if not any(r["form"] for r in report["per_question"]):
            print("  (no question in this range matched a post-process form)")
    else:
        report["per_question"] = [{"key": k, "difficulty": questions[int(k)].get("difficulty"),
                                   "original_correct": correct["selected.json"][k]} for k in keys]

    report_dir = Path(args.report_dir) if args.report_dir else output_dir / "evaluation"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"evaluation_{start}_{end}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
