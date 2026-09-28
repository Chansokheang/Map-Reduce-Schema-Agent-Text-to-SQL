"""
Permissive Evaluator — diagnostic companion to bird_evaluation.py

Categorizes each (predicted_sql, gold_sql) pair into four buckets:

  strict          set(pred) == set(gold)   (BIRD-pass)
  column_order    same values per row, different SELECT order
                  e.g. gold (City, LowGrade, School) vs pred (City, School, LowGrade)
  subset          gold's row values are a subset of pred's row values
                  e.g. gold ("San Diego",) vs pred ("San Diego", 5333)
                  (model knew the answer but over-projected)
  wrong           none of the above — real logic error / wrong filtering / etc.
  error           SQL failed to execute or timed out

Mirrors bird_evaluation.py CLI args so it integrates with the existing
workflow. Strict accuracy column will match BIRD's number; the additional
columns tell you how many "failures" are actually near-misses.

Usage:
  python -m evaluation.permissive_eval \
    --predicted_sql_path /path/to/preds/ \
    --ground_truth_path  /path/to/gold/ \
    --data_mode dev \
    --db_root_path /path/to/dev_databases/ \
    --diff_json_path /path/to/dev.json

This script does NOT change the pipeline or pred/gold files. Read-only.
"""

import argparse
import json
import multiprocessing as mp
import sqlite3
import sys

from func_timeout import FunctionTimedOut, func_timeout


def load_json(path):
    with open(path, "r") as f:
        return json.loads(f.read())


def _row_value_multiset(row):
    """Return a sorted tuple of stringified values, ignoring column position."""
    return tuple(sorted("" if v is None else str(v) for v in row))


def categorize(pred_rows, gold_rows):
    """
    Decide which bucket the (pred, gold) pair falls into.
    Both inputs are lists of tuples returned by cursor.fetchall().
    """
    p_set = set(pred_rows)
    g_set = set(gold_rows)

    if p_set == g_set:
        return "strict"

    # Column-order: same values per row regardless of SELECT order.
    p_sorted_set = {_row_value_multiset(r) for r in pred_rows}
    g_sorted_set = {_row_value_multiset(r) for r in gold_rows}
    if p_sorted_set == g_sorted_set:
        return "column_order"

    # Subset: each gold row's value-set fits inside some pred row's value-set,
    # AND the row counts match (so we're not silently allowing pred to add
    # rows that don't exist in gold).
    if pred_rows and gold_rows and len(pred_rows) == len(gold_rows):
        p_value_sets = [
            {("" if v is None else str(v)) for v in r} for r in pred_rows
        ]
        g_value_sets = [
            {("" if v is None else str(v)) for v in r} for r in gold_rows
        ]
        # For every gold value-set, there must be a pred value-set that
        # contains it. (Bag-style match, allowing reuse — fine for diagnostic.)
        if all(any(gv.issubset(pv) for pv in p_value_sets) for gv in g_value_sets):
            return "subset"

    return "wrong"


def execute_pair(predicted_sql, gold_sql, db_path):
    """Execute both SQLs, return (bucket, error_message_or_None)."""
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(predicted_sql)
        pred = cur.fetchall()
        cur.execute(gold_sql)
        gold = cur.fetchall()
        conn.close()
        return categorize(pred, gold), None
    except sqlite3.Error as e:
        return "error", f"sqlite: {e}"
    except Exception as e:
        return "error", f"other: {e}"


def execute_model(predicted_sql, gold_sql, db_path, idx, meta_time_out):
    try:
        bucket, err = func_timeout(
            meta_time_out, execute_pair, args=(predicted_sql, gold_sql, db_path)
        )
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        bucket, err = "error", "timeout"
    except Exception as e:
        bucket, err = "error", f"runner: {e}"
    return {"sql_idx": idx, "bucket": bucket, "error": err}


def package_sqls(sql_path, db_root_path, mode="gpt", data_mode="dev",
                 file_name="predict_dev.json"):
    sqls = []
    db_paths = []
    if mode == "gpt":
        # `file_name` lets you point at e.g. selected.json or
        # candidate_full_profile.json instead of the BIRD default
        # predict_dev.json — matches how evaluation/evaluation.py works.
        target = sql_path + file_name
        data = json.load(open(target, "r"))
        # Iterate by sorted integer key so the order matches dev.json.
        for k in sorted(data.keys(), key=lambda x: int(x) if x.isdigit() else x):
            sql_str = data[k]
            if isinstance(sql_str, str):
                if "\t----- bird -----\t" in sql_str:
                    sql, db_name = sql_str.split("\t----- bird -----\t")
                else:
                    sql, db_name = sql_str, "financial"
            else:
                sql, db_name = " ", "financial"
            sqls.append(sql)
            db_paths.append(db_root_path + db_name + "/" + db_name + ".sqlite")
    elif mode == "gt":
        with open(sql_path + data_mode + ".sql") as f:
            for line in f:
                sql, db_name = line.strip().split("\t")
                sqls.append(sql)
                db_paths.append(db_root_path + db_name + "/" + db_name + ".sqlite")
    return sqls, db_paths


# Kept module-global so multiprocessing callbacks can append into it.
exec_result: list = []


def _collect(result):
    exec_result.append(result)


def run_parallel(pairs, db_paths, num_cpus, meta_time_out):
    pool = mp.Pool(processes=num_cpus)
    for i, (pred, gold) in enumerate(pairs):
        pool.apply_async(
            execute_model,
            args=(pred, gold, db_paths[i], i, meta_time_out),
            callback=_collect,
        )
    pool.close()
    pool.join()


def sort_results(items):
    return sorted(items, key=lambda x: x["sql_idx"])


def report(results, diff_json_path):
    contents = load_json(diff_json_path)
    by_diff = {"simple": [], "moderate": [], "challenging": []}
    # Predictions may cover only a prefix of dev.json (partial runs).
    # Iterate over the result count, not the dev.json count.
    for r in results:
        idx = r["sql_idx"]
        if idx >= len(contents):
            continue
        d = contents[idx].get("difficulty", "")
        if d in by_diff:
            by_diff[d].append(r)

    levels = list(by_diff.keys()) + ["total"]
    buckets = ["strict", "column_order", "subset", "wrong", "error"]
    counts_per_level = {lv: {b: 0 for b in buckets} for lv in levels}

    for lv in by_diff:
        for r in by_diff[lv]:
            counts_per_level[lv][r["bucket"]] += 1
            counts_per_level["total"][r["bucket"]] += 1

    total_counts = {lv: sum(counts_per_level[lv].values()) for lv in levels}

    print()
    header = ["", *levels]
    fmt = "{:20} " + " ".join(["{:>14}"] * len(levels))
    print(fmt.format(*header))
    print(fmt.format("count", *[total_counts[lv] for lv in levels]))
    print("-" * (20 + 15 * len(levels)))

    for b in buckets:
        row = [counts_per_level[lv][b] for lv in levels]
        print(fmt.format(b, *row))

    print("-" * (20 + 15 * len(levels)))

    def acc(lv, *include):
        n_total = total_counts[lv] or 1
        return 100.0 * sum(counts_per_level[lv][b] for b in include) / n_total

    strict_row = [acc(lv, "strict") for lv in levels]
    perm_row = [acc(lv, "strict", "column_order", "subset") for lv in levels]

    print(fmt.format("strict %", *[f"{v:.2f}" for v in strict_row]))
    print(fmt.format("permissive %", *[f"{v:.2f}" for v in perm_row]))
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predicted_sql_path", type=str, required=True)
    parser.add_argument("--ground_truth_path", type=str, required=True)
    parser.add_argument("--data_mode", type=str, required=True, default="dev")
    parser.add_argument("--db_root_path", type=str, required=True)
    parser.add_argument("--num_cpus", type=int, default=1)
    parser.add_argument("--meta_time_out", type=float, default=30.0)
    parser.add_argument("--mode_gt", type=str, default="gt")
    parser.add_argument("--mode_predict", type=str, default="gpt")
    parser.add_argument("--diff_json_path", type=str, required=True)
    parser.add_argument(
        "--file_name", type=str, default="predict_dev.json",
        help="Prediction file name inside --predicted_sql_path "
             "(e.g. selected.json, candidate_full_profile.json)",
    )
    args = parser.parse_args()

    pred_queries, db_paths = package_sqls(
        args.predicted_sql_path, args.db_root_path,
        mode=args.mode_predict, data_mode=args.data_mode,
        file_name=args.file_name,
    )
    gold_queries, _ = package_sqls(
        args.ground_truth_path, args.db_root_path,
        mode="gt", data_mode=args.data_mode,
    )

    pairs = list(zip(pred_queries, gold_queries))
    run_parallel(pairs, db_paths, args.num_cpus, args.meta_time_out)
    results = sort_results(exec_result)

    print(f"Evaluated {len(results)} pairs")
    report(results, args.diff_json_path)


if __name__ == "__main__":
    main()
