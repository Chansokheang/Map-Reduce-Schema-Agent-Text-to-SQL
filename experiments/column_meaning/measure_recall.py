"""Offline recall: how often the column block contains the columns the gold query uses.

Gold SQL is read here for SCORING ONLY. Nothing in corpus.py, retriever.py or pipeline_patch.py
reads gold, and no rule in them was derived from a per-question result.

  python -m experiments.column_meaning.measure_recall --end 700
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import sqlglot
from sqlglot import exp

from experiments.column_meaning.retriever import retrieve, format_block

DEV = ROOT / "data/bird_data/dev.json"


def gold_columns(sql):
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return set()
    return {c.name.casefold() for c in tree.find_all(exp.Column) if c.name}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--questions", default=str(DEV))
    args = parser.parse_args()

    questions = json.load(open(args.questions, encoding="utf-8"))[args.start:args.end]
    per = defaultdict(Counter)
    lines = 0
    started = time.time()
    for q in questions:
        hits = retrieve(q["db_id"], q["question"], q.get("evidence", ""), limit=args.limit)
        shown = {h["column"].casefold() for h in hits}
        wanted = gold_columns(q["SQL"])
        d = q.get("difficulty", "?")
        lines += len(hits)
        per[d]["questions"] += 1
        per[d]["columns"] += len(wanted)
        per[d]["found"] += len(wanted & shown)
        per[d]["complete"] += bool(wanted) and wanted <= shown
        per["all"]["questions"] += 1
        per["all"]["columns"] += len(wanted)
        per["all"]["found"] += len(wanted & shown)
        per["all"]["complete"] += bool(wanted) and wanted <= shown
    elapsed = time.time() - started

    print(f"{len(questions)} questions, limit={args.limit}, "
          f"{lines/max(len(questions),1):.1f} lines per block, "
          f"{1000*elapsed/max(len(questions),1):.0f} ms per question\n")
    print(f"{'':<14}{'questions':>10}{'gold columns shown':>21}{'all shown':>12}")
    for d in ("simple", "moderate", "challenging", "all"):
        c = per.get(d)
        if not c:
            continue
        print(f"{d:<14}{c['questions']:>10}"
              f"{c['found']}/{c['columns']} = {100*c['found']/max(c['columns'],1):>5.1f}%".rjust(21)
              + f"{100*c['complete']/max(c['questions'],1):>11.1f}%")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
