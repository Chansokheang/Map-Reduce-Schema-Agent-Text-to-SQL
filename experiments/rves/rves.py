"""R-VES (Reward-based Valid Efficiency Score), ported from BIRD's own evaluator.

Source: bird-bench/mini_dev, evaluation/evaluation_ves.py, downloaded and read directly rather
than paraphrased. The reward buckets and the aggregation are reproduced exactly:

    time_ratio = mean over `iterate_num` runs of  gold_time / predicted_time,
                 after dropping points beyond 3 standard deviations,
                 and 0 unless set(predicted_rows) == set(gold_rows)

    reward = 0     if time_ratio == 0        (wrong answer, timeout or error)
             1.25  if time_ratio >= 2        (at least twice as fast as gold)
             1.0   if 1    <= ratio < 2
             0.75  if 0.5  <= ratio < 1
             0.5   if 0.25 <= ratio < 0.5
             0.25  otherwise

    R-VES  = sum(sqrt(reward) * 100) / number_of_queries

The denominator is EVERY query, not only the correct ones, so a wrong query contributes 0 and
correctness gates efficiency. The ceiling is sqrt(1.25) * 100 = 111.80.

This is NOT the metric in evaluation/evaluation_ves.py in this repo, which is the older VES:
that one takes sqrt of the raw time ratio with no bucketing. Do not report one as the other.

Timing is wall-clock and therefore machine-dependent. BIRD's defaults are num_cpus=1 and
iterate_num=100 so the measurement does not compete with itself; this port keeps those defaults
and warns if iterate_num is lowered. Absolute R-VES is not comparable across machines.
"""
import argparse
import json
import math
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from func_timeout import FunctionTimedOut, func_timeout

ROOT = Path(__file__).resolve().parents[2]
DELIM = "\t----- bird -----\t"
BUCKETS = ((2.0, 1.25), (1.0, 1.0), (0.5, 0.75), (0.25, 0.5))
CEILING = math.sqrt(1.25) * 100


def clean_abnormal(values):
    """BIRD's outlier filter: keep points within 3 standard deviations."""
    arr = np.asarray(values)
    mean, std = np.mean(arr, axis=0), np.std(arr, axis=0)
    return [x for x in arr if mean - 3 * std < x < mean + 3 * std]


def reward_for(time_ratio):
    """BIRD's reward buckets, exactly."""
    if time_ratio == 0:
        return 0.0
    for lo, reward in BUCKETS:
        if time_ratio >= lo:
            return reward
    return 0.25


def _run(sql, db_path, return_time=False):
    conn = sqlite3.connect(db_path)
    try:
        start = time.time()
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        elapsed = time.time() - start
    finally:
        conn.close()
    return elapsed if return_time else rows


def iterated_time_ratio(predicted_sql, gold_sql, db_path, iterate_num):
    if set(_run(predicted_sql, db_path)) != set(_run(gold_sql, db_path)):
        return 0.0
    diffs = [_run(gold_sql, db_path, True) / _run(predicted_sql, db_path, True)
             for _ in range(iterate_num)]
    kept = clean_abnormal(diffs)
    return sum(kept) / len(kept) if kept else 0.0


def score_one(predicted_sql, gold_sql, db_path, iterate_num=100, meta_time_out=30.0):
    """(reward, time_ratio, status). A timeout or any error scores 0, as in BIRD."""
    try:
        ratio = func_timeout(meta_time_out * iterate_num, iterated_time_ratio,
                             args=(predicted_sql, gold_sql, db_path, iterate_num))
        return reward_for(ratio), ratio, "ok"
    except KeyboardInterrupt:
        raise
    except FunctionTimedOut:
        return 0.0, 0.0, "timeout"
    except Exception as exc:
        return 0.0, 0.0, "error: " + type(exc).__name__


def compute_rves(rewards):
    """sum(sqrt(reward) * 100) / n, over EVERY query."""
    if not rewards:
        return 0.0
    return sum(math.sqrt(r) * 100 for r in rewards) / len(rewards)


def sql_of(entry):
    """Strip this project's '<sql>\\t----- bird -----\\t<db_id>' wrapper."""
    return str(entry).split(DELIM)[0].strip()


LABELS = {1.25: ">=2x faster than gold", 1.0: "1-2x faster", 0.75: "0.5-1x (slower)",
          0.5: "0.25-0.5x", 0.25: "<0.25x", 0.0: "wrong / timeout / error"}


def report(rows):
    rewards = [r["reward"] for r in rows]
    print("\nR-VES over %d questions: %.2f   (ceiling %.2f)"
          % (len(rows), compute_rves(rewards), CEILING))
    correct = sum(1 for r in rewards if r > 0)
    print("  reward > 0: %d/%d = %.2f%%  - identical to execution accuracy"
          % (correct, len(rows), 100 * correct / len(rows)))
    print("\n%8s  %-32s%6s" % ("reward", "meaning", "n"))
    for val, cnt in sorted(Counter(rewards).items(), reverse=True):
        print("%8s  %-32s%6d" % (val, LABELS.get(val, ""), cnt))
    for field in ("difficulty", "db_id"):
        groups = {}
        for r in rows:
            groups.setdefault(r.get(field), []).append(r["reward"])
        print("\n%-24s%6s%9s" % (field, "n", "R-VES"))
        for key, vals in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            print("%-24s%6d%9.2f" % (key, len(vals), compute_rves(vals)))


def main():
    p = argparse.ArgumentParser(description="R-VES over a BIRD-format prediction file.")
    p.add_argument("--predictions", required=True)
    p.add_argument("--questions", default=str(ROOT / "data/bird_data/dev.json"))
    p.add_argument("--databases-dir", default=str(ROOT / "data/bird_data/dev_databases"))
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None, help="exclusive; default = all predictions")
    p.add_argument("--iterate-num", type=int, default=100, help="BIRD default 100")
    p.add_argument("--meta-time-out", type=float, default=30.0)
    p.add_argument("--out", default=None, help="per-question rewards; resumable if it exists")
    p.add_argument("--strict", action="store_true",
                   help="refuse to run below BIRD's default iterate-num instead of warning")
    args = p.parse_args()

    if args.iterate_num < 100:
        msg = ("iterate_num=%d is below BIRD's default of 100; the time ratio will be noisier "
               "and the result is not comparable to a published figure." % args.iterate_num)
        if args.strict:
            sys.exit("Error: " + msg)
        print("WARNING: " + msg, file=sys.stderr)

    qs = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    preds = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    end = args.end if args.end is not None else max(map(int, preds)) + 1
    keys = [str(i) for i in range(args.start, end) if str(i) in preds]
    print("scoring %d questions from %s" % (len(keys), args.predictions))

    cache = Path(args.out) if args.out else None
    done = json.loads(cache.read_text(encoding="utf-8")) if cache and cache.exists() else {}
    if done:
        print("resuming: %d already scored" % len(done))

    t0 = time.time()
    todo = [k for k in keys if k not in done]
    scored = 0
    for k in todo:
        q = qs[int(k)]
        db = Path(args.databases_dir) / q["db_id"] / (q["db_id"] + ".sqlite")
        reward, ratio, status = score_one(sql_of(preds[k]), q["SQL"], str(db),
                                          args.iterate_num, args.meta_time_out)
        done[k] = {"reward": reward, "time_ratio": ratio, "status": status,
                   "db_id": q["db_id"], "difficulty": q.get("difficulty")}
        scored += 1
        if cache:
            cache.write_text(json.dumps(done, indent=1), encoding="utf-8")
        if scored % 10 == 0 or scored == len(todo):
            # rate over work actually done in THIS run; cached skips would dilute it
            rate = (time.time() - t0) / scored
            print("  %d/%d of this run (%d/%d overall)  %.1fs/question  eta %.0f min"
                  % (scored, len(todo), len(done), len(keys), rate,
                     (len(todo) - scored) * rate / 60), flush=True)

    rows = [done[k] for k in keys if k in done]
    report(rows)
    bad = [k for k in keys if k in done and done[k]["status"] != "ok"]
    if bad:
        print("\n%d question(s) timed out or errored: %s"
              % (len(bad), " ".join("Q" + b for b in bad[:20])))


if __name__ == "__main__":
    main()
