"""Offline recall: does the retriever surface the string values gold SQL actually uses?

Gold is read here only to score the retriever; the retriever itself sees question and evidence
only. Writes output/matched_contents/recall.json and prints a summary.

  python -m experiments.matched_contents.measure_recall --limit 200
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import sqlglot
from sqlglot import exp
from experiments.matched_contents.indexer import DEFAULT_OUT, normalize
from experiments.matched_contents.retriever import retrieve, index_path

DELIMITER = "\t----- bird -----\t"


def gold_literals(sql):
    """String literals a prediction would have to reproduce, ignoring pure wildcards."""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return []
    out = []
    for literal in tree.find_all(exp.Literal):
        if not literal.is_string:
            continue
        raw = literal.this
        core = raw.strip("%_ ")
        if len(core) < 2 or re.fullmatch(r"[\d\W_]+", core):
            continue            # numbers, dates, punctuation and format strings are not lookups
        out.append({"literal": raw, "core": core})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dev-json", default="data/bird_data/dev.json")
    parser.add_argument("--index-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--limit", type=int, default=None, help="First N questions only")
    parser.add_argument("--hits", type=int, default=12, help="Matched contents shown per question")
    parser.add_argument("--out", default="output/matched_contents/recall.json")
    args = parser.parse_args()

    questions = json.loads(Path(args.dev_json).read_text(encoding="utf-8"))[:args.limit]
    stats = collections.Counter()
    per_question, misses = [], []
    for q in questions:
        if not index_path(q["db_id"], args.index_dir).exists():
            stats["no index"] += 1
            continue
        wanted = gold_literals(q["SQL"])
        hits = retrieve(q["db_id"], q["question"], q.get("evidence") or "", limit=args.hits,
                        index_dir=args.index_dir)
        found_norm = {normalize(h["value"]) for h in hits}
        found_pairs = {(normalize(h["value"]), h["table"].casefold(), h["column"].casefold()) for h in hits}
        stats["questions"] += 1
        stats["questions with hits"] += bool(hits)
        stats["hits"] += len(hits)
        if not wanted:
            stats["questions without a gold string literal"] += 1
            continue
        stats["questions with a gold string literal"] += 1
        covered = [w for w in wanted if normalize(w["core"]) in found_norm]
        stats["gold literals"] += len(wanted)
        stats["gold literals retrieved"] += len(covered)
        if len(covered) == len(wanted):
            stats["questions fully covered"] += 1
        elif covered:
            stats["questions partly covered"] += 1
        else:
            stats["questions not covered"] += 1
            if len(misses) < 30:
                misses.append({"question_id": q["question_id"], "db": q["db_id"], "question": q["question"][:110],
                               "wanted": [w["core"] for w in wanted][:4], "retrieved": sorted(found_norm)[:4]})
        per_question.append({"question_id": q["question_id"], "db": q["db_id"], "wanted": [w["core"] for w in wanted],
                             "retrieved": [h["value"] for h in hits],
                             "covered": [w["core"] for w in covered],
                             "with_column": [w["core"] for w in wanted
                                             if any(normalize(w["core"]) == v for v, _, _ in found_pairs)]})

    total_q = stats["questions with a gold string literal"]
    print(json.dumps({k: v for k, v in stats.items()}, indent=1))
    if total_q:
        print(f"\nliteral recall  {stats['gold literals retrieved']}/{stats['gold literals']} = "
              f"{100 * stats['gold literals retrieved'] / stats['gold literals']:.1f}%")
        print(f"question recall {stats['questions fully covered']}/{total_q} fully, "
              f"{stats['questions partly covered']} partly, {stats['questions not covered']} not covered")
        print(f"average hits shown per question: {stats['hits'] / max(stats['questions'], 1):.1f}")
    print("\nexamples where nothing gold needed was retrieved:")
    for m in misses[:12]:
        print(f"  Q{m['question_id']} [{m['db']}] wanted {m['wanted']}")
        print(f"      {m['question']}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"stats": dict(stats), "misses": misses, "per_question": per_question},
                              indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
