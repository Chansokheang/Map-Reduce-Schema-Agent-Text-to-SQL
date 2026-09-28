"""Merge pilot repairs into separate full-dev files and evaluate all 1,534 questions."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.evaluate_matched_ablation import read_json, write_json, digest
from analysis.verify_null_duplicate_ablation import check
from evaluation.evaluation import package_sqls, compute_acc_by_diff, print_data

PILOT = ROOT / "output/null_duplicate_ablation/pilot"
DEST = PILOT / "full_dev"


def main():
    if DEST.exists():
        raise FileExistsError(f"Preserving existing full-dev export: {DEST}")
    source = ROOT / "output/claude_headless_v6/selected.json"
    gold_json = ROOT / "data/bird_data/dev.json"
    gold_sql = ROOT / "data/bird_data/dev.sql"
    hashes = {str(p):digest(p) for p in [source, gold_json, gold_sql]}
    manifest = read_json(PILOT / "manifest.json")
    assert hashes[str(source)] == manifest["source_sha256"][str(source)]
    original = read_json(source)
    assert list(original) == [str(i) for i in range(1534)]
    ids = set(manifest["question_ids"])
    assert len(ids) == 176
    predictions = {"original":original, "control":dict(original), "conditional":dict(original)}
    changes = {"control":[], "conditional":[]}
    delimiter = "\t----- bird -----\t"
    for qid in sorted(ids):
        pair = read_json(PILOT / "pairs" / f"{qid}.json")
        sql, db = original[str(qid)].rsplit(delimiter, 1)
        assert pair["question_id"] == qid and pair["db_id"] == db and pair["status"] == "complete"
        assert pair["input_sql"] == sql.strip()
        for arm in changes:
            repaired = pair["arms"][arm]["final_sql"]
            if repaired.strip() != sql.strip():
                predictions[arm][str(qid)] = repaired + delimiter + db
                changes[arm].append(qid)
    for arm in changes:
        assert len(predictions[arm]) == 1534
        actual = {int(k) for k,v in predictions[arm].items() if v != original[k]}
        assert actual == set(changes[arm]) and actual <= ids
    DEST.mkdir()
    (DEST / "original.json").write_bytes(source.read_bytes())
    for arm in changes:
        write_json(DEST / f"selected_{arm}.json", predictions[arm])
    filenames = {"original":"original.json", "control":"selected_control.json",
                 "conditional":"selected_conditional.json"}
    write_json(DEST / "merge_manifest.json", {"source":str(source), "source_sha256":hashes,
        "total_questions":1534, "pilot_question_ids":sorted(ids), "changed_question_ids":changes,
        "untouched_outside_pilot":1358, "files":filenames,
        "scoring":"Unmodified repository worker, 30 seconds per predicted/gold pair, two processes. Identical SQL for a question shares one execution result across files."})
    print(f"Created separate 1534-query files. Changed SQL: { {k:len(v) for k,v in changes.items()} }. Original preserved.", flush=True)
    db_root = str(ROOT / "data/bird_data/dev_databases") + "/"
    gt, gt_paths = package_sqls(str(gold_sql.parent)+"/", db_root, mode="gt", data_mode="dev")
    packaged = {}
    for arm, filename in filenames.items():
        sqls, db_paths = package_sqls(str(DEST)+"/", db_root, mode="gpt", data_mode="dev", file_name=filename)
        assert len(sqls) == len(gt) == 1534 and db_paths == gt_paths
        packaged[arm] = sqls
    jobs = []
    for qid in range(1534):
        grouped = {}
        for arm, sqls in packaged.items():
            grouped.setdefault(sqls[qid], []).append(arm)
        for sql, arms in grouped.items():
            jobs.append({"question_id":qid, "sql":sql, "gold":gt[qid], "db_path":gt_paths[qid], "arms":arms})
    print(f"Evaluating all three full files: {len(jobs)} unique query pairs, sharing identical work.", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(check, job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 100 == 0 or len(results) == len(jobs):
                print(f"Evaluated {len(results)}/{len(jobs)} unique query pairs", flush=True)
    scores = {arm:{} for arm in predictions}
    for result in results:
        for arm in result["arms"]:
            assert result["question_id"] not in scores[arm]
            scores[arm][result["question_id"]] = result["res"]
    summary = {"n":1534, "unique_query_pairs":len(jobs), "scores":{}, "changed_question_ids":changes}
    for arm in scores:
        assert len(scores[arm]) == 1534
        rows = [{"sql_idx":i, "res":scores[arm][i]} for i in range(1534)]
        write_json(DEST / f"{arm}.per_question.json", rows)
        simple, moderate, challenging, total, counts = compute_acc_by_diff(rows, str(gold_json), 0, 1534)
        text = io.StringIO()
        with redirect_stdout(text):
            print_data([simple, moderate, challenging, total], counts)
        (DEST / f"{arm}.evaluation.txt").write_text(text.getvalue(), encoding="utf-8")
        summary["scores"][arm] = {"correct":sum(scores[arm].values()), "accuracy_percent":total,
            "difficulty_accuracy":{"simple":simple, "moderate":moderate, "challenging":challenging},
            "recovered_vs_original":[i for i in range(1534) if scores[arm][i]>scores["original"][i]],
            "regressed_vs_original":[i for i in range(1534) if scores[arm][i]<scores["original"][i]]}
    assert all(digest(Path(p)) == h for p,h in hashes.items())
    write_json(DEST / "summary.json", summary)
    write_json(DEST / "integrity.json", {"original_files_unchanged":True, "source_sha256":hashes,
               "output_sha256":{f:digest(DEST/f) for f in filenames.values()},
               "outside_pilot_unchanged_in_each_file":1358})
    lines = ["# Full-dev evaluation with pilot repairs", "",
        "Each file contains all 1,534 original question IDs. Repairs were merged only for the 176 experiment questions; the other 1,358 predictions remain exactly unchanged. No new model calls were made.", "",
        "| Version | File | SQL changed | Correct | EX |", "|---|---|---:|---:|---:|"]
    for arm, filename in filenames.items():
        value = summary["scores"][arm]
        lines.append(f"| {arm} | [{filename}]({filename}) | {len(changes.get(arm, []))} | {value['correct']}/1534 | {value['accuracy_percent']:.2f}% |")
    lines += ["", "All three complete files were loaded and scored using the repository's unmodified evaluator worker, with two processes and a 30-second timeout per predicted/gold query pair. Identical SQL for the same question shares one execution result across versions, avoiding timing noise on unchanged queries.",
        "", "The original source file and ground truth are unchanged; see integrity.json. The copied original.json is byte-identical to the source. See merge_manifest.json for exact changed IDs, summary.json for recoveries/regressions, and each version's per_question.json and evaluation.txt for scores.",
        "", "These are full-dev scores for partial replacement by the existing pilot outcomes. The fixers were not run on the other 1,358 questions. The previously documented Q1295 control review fallback is retained. Timeouts are scored as failures by the repository evaluator."]
    (DEST / "README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
