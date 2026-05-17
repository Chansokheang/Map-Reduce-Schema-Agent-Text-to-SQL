"""
Unordered-comparison evaluator for BIRD predictions (key-based matching).

Differences vs. evaluation.py:
- Row comparison is order-independent at BOTH the row and column level.
  Each row is canonicalized by sorting its values with a (type, repr) key,
  so column-projection order within a row does not affect the verdict.
- Predictions are matched to ground truth BY KEY, not by position. Each key
  `k` in the prediction JSON is looked up as dev.json[int(k)] — supports
  sparse prediction files (e.g. only questions 0-9, 12, 23, 36).
- Ground truth SQL, db_id, and difficulty come directly from dev.json.
- `--predicted_sql_path` can be repeated (or given space-separated). When more
  than one is supplied, each directory is evaluated independently and the
  results are printed side-by-side with a summary table.

Usage (single directory):
    python -u evaluation/evaluation_unordered.py \
        --db_root_path data/bird_data/dev_databases/ \
        --predicted_sql_path output/final/ \
        --diff_json_path data/bird_data/dev.json \
        --file_name selected.json \
        --num_cpus 4 --meta_time_out 30

Usage (compare 3 directories):
    python -u evaluation/evaluation_unordered.py \
        --db_root_path data/bird_data/dev_databases/ \
        --predicted_sql_path output/final/ output/claude_headless_v2/ output/ver1/ \
        --diff_json_path data/bird_data/dev.json \
        --file_name selected.json \
        --num_cpus 4 --meta_time_out 30
"""

import argparse
import json
import multiprocessing as mp
import os
import sqlite3
import sys
from collections import Counter

from func_timeout import FunctionTimedOut, func_timeout


def load_json(path):
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def _canonical_row(row):
    return tuple(sorted((type(v).__name__, repr(v)) for v in row))


def _as_multiset(rows):
    return Counter(_canonical_row(r) for r in rows)


def execute_sql(predicted_sql, ground_truth, db_path):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found at {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(predicted_sql)
        predicted_res = cursor.fetchall()
        cursor.execute(ground_truth)
        ground_truth_res = cursor.fetchall()
    finally:
        conn.close()

    return 1 if _as_multiset(predicted_res) == _as_multiset(ground_truth_res) else 0


def execute_model(predicted_sql, ground_truth, db_place, qid, meta_time_out):
    try:
        res = func_timeout(
            meta_time_out,
            execute_sql,
            args=(predicted_sql, ground_truth, db_place),
        )
        err = ""
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        res, err = 0, "timeout"
    except Exception as e:
        res, err = 0, f"{type(e).__name__}: {e}"
    return {"question_id": qid, "res": res, "err": err}


def load_predictions(pred_path):
    raw = load_json(pred_path)
    delimiter = "\t----- bird -----\t"
    out = {}
    for key, entry in raw.items():
        sql, db_name = entry.split(delimiter)
        out[int(key)] = {"sql": sql.strip(), "db_id": db_name.strip()}
    return out


def slice_predictions(predictions, start, end):
    if start is None and end is None:
        return predictions, list(predictions.keys()), len(predictions)
    items = list(predictions.items())
    total = len(items)
    lo = start if start is not None else 0
    hi = end if end is not None else total
    sliced = items[lo:hi]
    return dict(sliced), [k for k, _ in sliced], total


def build_pairs(predictions, dev_data, db_root_path):
    """Align each predicted key to its dev.json entry by integer index."""
    pairs = []
    for qid in sorted(predictions.keys()):
        if qid >= len(dev_data):
            print(f"  [WARN] key {qid} exceeds dev.json size ({len(dev_data)}); skipping")
            continue
        pred = predictions[qid]
        gt_entry = dev_data[qid]
        db_id = pred["db_id"] or gt_entry["db_id"]
        if pred["db_id"] and pred["db_id"] != gt_entry["db_id"]:
            print(f"  [WARN] qid {qid} db_id mismatch: pred={pred['db_id']} gt={gt_entry['db_id']}")
        db_path = os.path.join(db_root_path, db_id, f"{db_id}.sqlite")
        pairs.append(
            {
                "qid": qid,
                "pred_sql": pred["sql"],
                "gt_sql": gt_entry["SQL"],
                "db_path": db_path,
                "difficulty": gt_entry.get("difficulty", "unknown"),
            }
        )
    return pairs


def run_parallel(pairs, num_cpus, meta_time_out):
    results = []
    pool = mp.Pool(processes=num_cpus)
    for p in pairs:
        pool.apply_async(
            execute_model,
            args=(p["pred_sql"], p["gt_sql"], p["db_path"], p["qid"], meta_time_out),
            callback=results.append,
        )
    pool.close()
    pool.join()
    return results


def evaluate_single(label, pred_dir, file_name, dev_data, db_root,
                    num_cpus, timeout, start, end, anchor_qids=None):
    """Run the full evaluation for one prediction directory and return a dict.

    If anchor_qids is provided, ignore start/end and keep only the predictions
    whose key is in anchor_qids (preserving anchor order). This lets the caller
    evaluate every directory on the exact same question set — useful when the
    first dir defines the questions and subsequent dirs must match qid-for-qid.
    """
    pred_path = os.path.join(pred_dir, file_name)
    if not os.path.exists(pred_path):
        print(f"[{label}] MISSING: {pred_path}")
        return {"label": label, "path": pred_path, "missing": True, "rows": [], "qids": []}

    print(f"\n[{label}] Loading predictions: {pred_path}")
    predictions = load_predictions(pred_path)

    if anchor_qids is not None:
        kept = {qid: predictions[qid] for qid in anchor_qids if qid in predictions}
        missing = [qid for qid in anchor_qids if qid not in predictions]
        predictions = kept
        picked = list(kept.keys())
        print(f"[{label}] Anchored to {len(anchor_qids)} qids from first dir: "
              f"{len(predictions)} kept → {picked}")
        if missing:
            print(f"[{label}] Missing qids in this file: {missing}")
    else:
        predictions, picked, total = slice_predictions(predictions, start, end)
        if start is not None or end is not None:
            print(f"[{label}] Positional slice [{start}, {end}) over {total} entries: "
                  f"{len(predictions)} keys → {picked}")
        else:
            print(f"[{label}] Evaluating all {len(predictions)} keys: {picked}")

    pairs = build_pairs(predictions, dev_data, db_root)
    print(f"[{label}] {len(pairs)} aligned pairs | workers={num_cpus} timeout={timeout}s")
    results = run_parallel(pairs, num_cpus, timeout)
    by_qid = {r["question_id"]: r for r in results}

    rows = []
    for p in pairs:
        r = by_qid.get(p["qid"], {"res": 0, "err": "missing"})
        rows.append(
            {
                "qid": p["qid"],
                "difficulty": p["difficulty"],
                "res": r["res"],
                "err": r.get("err", ""),
            }
        )
    return {
        "label": label,
        "path": pred_path,
        "missing": False,
        "rows": rows,
        "qids": [p["qid"] for p in pairs],
    }


def _acc(vals):
    return (sum(vals) / len(vals) * 100) if vals else 0.0


def report_single(run):
    """Single-directory report (backward-compatible with earlier output)."""
    rows = run["rows"]
    buckets = {"simple": [], "moderate": [], "challenging": []}
    for row in rows:
        if row["difficulty"] in buckets:
            buckets[row["difficulty"]].append(row["res"])

    print("\nPer-question results:")
    print("  qid  | difficulty    | match | error")
    print("  -----+---------------+-------+---------------------")
    for row in rows:
        mark = "OK" if row["res"] else "X "
        print(f"  {row['qid']:>4} | {row['difficulty']:<13} |  {mark}   | {row['err']}")

    levels = ["simple", "moderate", "challenging", "total"]
    counts = [len(buckets["simple"]), len(buckets["moderate"]),
              len(buckets["challenging"]), len(rows)]
    total_res = [r["res"] for r in rows]
    accs = [_acc(buckets["simple"]), _acc(buckets["moderate"]),
            _acc(buckets["challenging"]), _acc(total_res)]
    print("\n==================    ACCURACY (unordered rows & columns, key-matched)    ==================")
    print("{:15} {:15} {:15} {:15} {:15}".format("", *levels))
    print("{:15} {:<15} {:<15} {:<15} {:<15}".format("count", *counts))
    print("{:15} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f}".format("accuracy", *accs))


def report_compare(runs):
    """Side-by-side comparison across multiple prediction directories."""
    active = [r for r in runs if not r["missing"]]
    if not active:
        print("\nNo prediction files could be loaded.")
        return

    qid_to_diff = {}
    all_qids = set()
    for d in active:
        for row in d["rows"]:
            all_qids.add(row["qid"])
            qid_to_diff.setdefault(row["qid"], row["difficulty"])
    sorted_qids = sorted(all_qids)

    labels = [d["label"] for d in active]
    col_w = max(14, max(len(lbl) for lbl in labels) + 2)

    print("\n" + "=" * (25 + (col_w + 3) * len(labels)))
    print("Per-question match (OK = match, X = mismatch, - = not in file)")
    header = f"{'qid':>5} | {'difficulty':<12} | " + " | ".join(f"{lbl:^{col_w}}" for lbl in labels)
    print(header)
    print("-" * len(header))

    per_dir_rowmap = {d["label"]: {row["qid"]: row for row in d["rows"]} for d in active}

    for qid in sorted_qids:
        diff = qid_to_diff.get(qid, "unknown")
        cells = []
        for lbl in labels:
            row = per_dir_rowmap[lbl].get(qid)
            if row is None:
                cells.append(f"{'-':^{col_w}}")
            else:
                mark = "OK" if row["res"] else "X"
                cells.append(f"{mark:^{col_w}}")
        print(f"{qid:>5} | {diff:<12} | " + " | ".join(cells))

    print("\n" + "=" * (25 + (col_w + 3) * len(labels)))
    print("Summary accuracy (unordered rows & columns, key-matched):")
    print(f"{'metric':<22} | " + " | ".join(f"{lbl:^{col_w}}" for lbl in labels))
    print("-" * (24 + (col_w + 3) * len(labels)))

    for diff in ["simple", "moderate", "challenging", "total"]:
        cells = []
        for d in active:
            if diff == "total":
                vals = [r["res"] for r in d["rows"]]
            else:
                vals = [r["res"] for r in d["rows"] if r["difficulty"] == diff]
            cells.append(f"{len(vals)}x {_acc(vals):6.2f}%")
        print(f"{diff:<22} | " + " | ".join(f"{c:^{col_w}}" for c in cells))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unordered SQL evaluator with key-based matching to dev.json"
    )
    parser.add_argument("--predicted_sql_path", nargs="+", required=True,
                        help="One or more directories containing the prediction JSON file. "
                             "When multiple are given, results are shown side-by-side.")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Optional labels (one per --predicted_sql_path). "
                             "Default: directory basename.")
    parser.add_argument("--diff_json_path", type=str, required=True,
                        help="Path to BIRD dev.json (ground-truth source)")
    parser.add_argument("--db_root_path", type=str, required=True,
                        help="Directory containing <db_id>/<db_id>.sqlite folders")
    parser.add_argument("--file_name", type=str, default="selected.json")
    parser.add_argument("--num_cpus", type=int, default=1)
    parser.add_argument("--meta_time_out", type=float, default=30.0)
    # Accepted for compatibility with the shared runner but ignored here:
    parser.add_argument("--ground_truth_path", type=str, default=None)
    parser.add_argument("--data_mode", type=str, default="dev")
    parser.add_argument("--mode_gt", type=str, default="gt")
    parser.add_argument("--mode_predict", type=str, default="gpt")
    parser.add_argument("--start", type=int, default=None,
                        help="Positional slice start (0-indexed, inclusive) over the "
                             "prediction file's keys in file order")
    parser.add_argument("--end", type=int, default=None,
                        help="Positional slice end (exclusive) over the prediction "
                             "file's keys in file order")
    args = parser.parse_args()

    pred_dirs = args.predicted_sql_path
    labels = args.labels or [os.path.basename(os.path.normpath(p)) for p in pred_dirs]
    if len(labels) != len(pred_dirs):
        print("--labels count must match --predicted_sql_path count")
        sys.exit(1)

    print(f"Loading ground truth: {args.diff_json_path}")
    dev_data = load_json(args.diff_json_path)
    db_root = args.db_root_path.rstrip("/") + "/"

    runs = []
    anchor_qids = None
    for i, (lbl, path) in enumerate(zip(labels, pred_dirs)):
        if i == 0:
            run = evaluate_single(
                lbl, path, args.file_name, dev_data, db_root,
                args.num_cpus, args.meta_time_out, args.start, args.end,
            )
            if not run["missing"]:
                anchor_qids = run["qids"]
                print(f"\nAnchor qids (from {lbl}): {anchor_qids}")
        else:
            run = evaluate_single(
                lbl, path, args.file_name, dev_data, db_root,
                args.num_cpus, args.meta_time_out, args.start, args.end,
                anchor_qids=anchor_qids,
            )
        runs.append(run)

    if len(runs) == 1:
        report_single(runs[0])
    else:
        report_compare(runs)

    print(f"\nFinished unordered evaluation for {args.file_name}"
          f" across {len(runs)} director{'y' if len(runs) == 1 else 'ies'}")
