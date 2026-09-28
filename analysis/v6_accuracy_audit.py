"""Offline BIRD diagnostic audit; never supplies gold SQL to generation.

Uses the repository's set-of-tuples EX comparison. Additional categories are
diagnostics, not alternative accuracy scores or automatic SQL repairs.
SQLite connections are read-only and each execution has a progress timeout.
Run from the repository root: python analysis/v6_accuracy_audit.py
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
from pathlib import Path
import sqlite3
import sys
import time

DELIMITER = "\t----- bird -----\t"
STRATEGIES = ["full_schema", "sme_metadata", "minimal_profile", "focused_schema", "full_profile"]


def execute(conn, sql, timeout):
    start = time.monotonic()
    conn.set_progress_handler(lambda: int(time.monotonic() - start > timeout), 10000)
    try:
        cursor = conn.execute(sql)
        names = [c[0] for c in cursor.description or []]
        rows = cursor.fetchall()
        return rows, names, None, round(time.monotonic() - start, 4)
    except sqlite3.Error as exc:
        return [], [], str(exc), round(time.monotonic() - start, 4)


def projection_map(wide, narrow):
    """One fixed injective column mapping, verified over entire tuple sets.

    Unlike per-row value sorting, this cannot swap columns differently per row.
    Marginal values prune the search; bounded search is diagnostic only.
    """
    if not wide or not narrow:
        return None
    a, b = len(next(iter(wide))), len(next(iter(narrow)))
    if a < b:
        return None
    wa = [{r[i] for r in wide} for i in range(a)]
    nb = [{r[i] for r in narrow} for i in range(b)]
    options = [[i for i in range(a) if wa[i] == values] for values in nb]
    if any(not x for x in options):
        return None
    for count, mapping in enumerate(itertools.product(*options)):
        if count >= 10000:
            break
        if len(set(mapping)) == b and {tuple(r[i] for i in mapping) for r in wide} == narrow:
            return list(mapping)
    return None


def classify(pred, gold):
    p, g = set(pred), set(gold)
    if p == g:
        return "strict", None
    if p and g:
        pw, gw = len(pred[0]), len(gold[0])
        if pw >= gw:
            mapping = projection_map(p, g)
            if mapping is not None:
                return ("column_order" if pw == gw else "extra_columns"), mapping
        else:
            mapping = projection_map(g, p)
            if mapping is not None:
                return "missing_columns", mapping
        if pw == gw:
            # Requires identical non-null tuples and exclusively null-bearing
            # differences; not proof that removing NULL filters fixes the SQL.
            p_nonnull = {r for r in p if None not in r}
            g_nonnull = {r for r in g if None not in r}
            if p_nonnull == g_nonnull and p_nonnull:
                return "null_rows_only", None
            def rounded(rows):
                return {tuple(round(v, 9) if isinstance(v, float) else v for v in r) for r in rows}
            if rounded(p) == rounded(g):
                return "numeric_round9_match", None
    return "other_mismatch", None


def preview(rows):
    return [repr(row)[:700] for row in sorted(rows, key=repr)[:5]]


def audit_question(job):
    question, predictions, dbroot, timeout = job
    qid, db = question["question_id"], question["db_id"]
    dbpath = Path(dbroot) / db / (db + ".sqlite")
    conn = sqlite3.connect(dbpath.resolve().as_uri() + "?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    cache = {}
    def cached(sql):
        if sql not in cache:
            cache[sql] = execute(conn, sql, timeout)
        return cache[sql]
    gold, names, gold_error, gold_time = cached(question["SQL"])
    record = dict(question)
    record.update(gold_error=gold_error, gold_row_count=len(gold),
                  gold_unique_count=len(set(gold)), gold_columns=names,
                  gold_null_rows=sum(None in r for r in gold), gold_seconds=gold_time)
    results, groups, group_sets = {}, [], []
    for name, packed in predictions.items():
        sql, pred_db = packed.rsplit(DELIMITER, 1)
        if pred_db != db:
            raise ValueError(f"Database mismatch at {qid}: {pred_db} != {db}")
        rows, columns, error, elapsed = cached(sql)
        category, mapping = ("gold_error", None) if gold_error else (
            ("execution_error", None) if error else classify(rows, gold))
        pset, gset = set(rows), set(gold)
        results[name] = dict(sql=sql, category=category, mapping=mapping,
            error=error, row_count=len(rows), unique_count=len(pset), columns=columns,
            null_rows=sum(None in r for r in rows), seconds=elapsed,
            pred_only=preview(pset-gset), gold_only=preview(gset-pset))
        if name.startswith("candidate_") and not error:
            group_index = next((i for i, s in enumerate(group_sets) if s == pset), None)
            if group_index is None:
                groups.append([name]); group_sets.append(pset)
            else:
                groups[group_index].append(name)
    conn.close()
    record["results"] = results
    record["candidate_result_groups"] = groups
    correct = [name for name, r in results.items()
               if name.startswith("candidate_") and r["category"] == "strict"]
    record["correct_candidates"] = correct
    # Tie-break by strategy order, never by the gold label.
    winner = max(groups, key=len)[0] if groups else None
    record["result_plurality_candidate"] = winner
    record["result_plurality_correct"] = bool(winner and results[winner]["category"] == "strict")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", default="output/claude_headless_v6")
    parser.add_argument("--gold", default="data/bird_data/dev.json")
    parser.add_argument("--db-root", default="data/bird_data/dev_databases")
    parser.add_argument("--out", default="analysis/v6_accuracy_audit")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    filenames = ["selected"] + ["candidate_"+s for s in STRATEGIES]
    if (Path(args.pred_dir)/"refined_selected.json").exists():
        filenames.append("refined_selected")
    files = {name: Path(args.pred_dir)/(name+".json") for name in filenames}
    predictions = {name: json.loads(path.read_text(encoding="utf-8")) for name,path in files.items()}
    qids = {str(q["question_id"]) for q in gold}
    assert len(qids) == len(gold), "Duplicate ground-truth question IDs"
    for name,pred in predictions.items():
        assert set(pred) == qids, f"Question ID coverage mismatch in {name}"
    jobs = [(q, {name:p[str(q['question_id'])] for name,p in predictions.items()},
             args.db_root, args.timeout) for q in gold]
    records = []
    started = time.monotonic()
    with (out/"details.jsonl").open("w", encoding="utf-8") as stream:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(audit_question, job) for job in jobs]
            for future in as_completed(futures):
                row = future.result()
                records.append(row)
                stream.write(json.dumps(row, ensure_ascii=False)+"\n")
                stream.flush()
                if len(records) % 100 == 0:
                    print(f"Audited {len(records)}/{len(gold)} in {time.monotonic()-started:.0f}s", flush=True)
    records.sort(key=lambda r:r["question_id"])
    summary = {
        "count":len(records), "sqlite_version":sqlite3.sqlite_version,
        "timeout_per_query_seconds":args.timeout,
        "elapsed_seconds":round(time.monotonic()-started, 2),
        "input_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [Path(args.gold), *files.values()]},
        "categories":{n:dict(Counter(r['results'][n]['category'] for r in records)) for n in filenames},
        "candidate_oracle_count":sum(bool(r['correct_candidates']) for r in records),
        "selected_wrong_candidate_correct":[r['question_id'] for r in records
            if r['results']['selected']['category']!='strict' and r['correct_candidates']],
        "selected_correct_no_candidate_correct":[r['question_id'] for r in records
            if r['results']['selected']['category']=='strict' and not r['correct_candidates']],
        "plurality_correct_count":sum(r['result_plurality_correct'] for r in records),
        "candidate_result_group_counts":dict(Counter(len(r['candidate_result_groups']) for r in records)),
        "all_candidates_same_wrong":[r['question_id'] for r in records
            if len(r['candidate_result_groups'])==1 and not r['correct_candidates']],
        "by_database":{}, "by_difficulty":{},
    }
    for field, target in [('db_id','by_database'),('difficulty','by_difficulty')]:
        for value in sorted({r[field] for r in records}):
            subset=[r for r in records if r[field]==value]
            summary[target][value]={'n':len(subset), 'selected_correct':sum(r['results']['selected']['category']=='strict' for r in subset),
                'oracle_correct':sum(bool(r['correct_candidates']) for r in subset)}
    if 'refined_selected' in filenames:
        summary['refined_comparison']={
            'wrong_to_right':[r['question_id'] for r in records if r['results']['selected']['category']!='strict' and r['results']['refined_selected']['category']=='strict'],
            'right_to_wrong':[r['question_id'] for r in records if r['results']['selected']['category']=='strict' and r['results']['refined_selected']['category']!='strict'],
            'sql_changed':sum(r['results']['selected']['sql']!=r['results']['refined_selected']['sql'] for r in records)}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['input_sha256','refined_comparison']}, indent=2),flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
