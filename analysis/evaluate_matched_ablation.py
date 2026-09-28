"""Export the same pilot questions for all arms and run the repository evaluator CLI."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "output/null_duplicate_ablation/pilot"
DEST = PILOT / "matched_subset"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if DEST.exists():
        raise FileExistsError(f"Preserving existing subset evaluation: {DEST}")
    manifest = read_json(PILOT / "manifest.json")
    raw_path = ROOT / "output/claude_headless_v6/selected.json"
    gold_path = ROOT / "data/bird_data/dev.json"
    gold_sql_path = ROOT / "data/bird_data/dev.sql"
    protected = {str(p):digest(p) for p in [raw_path, gold_path, gold_sql_path]}
    assert protected[str(raw_path)] == manifest["source_sha256"][str(raw_path)]
    assert protected[str(gold_path)] == manifest["source_sha256"][str(gold_path)]
    raw = read_json(raw_path)
    gold = read_json(gold_path)
    gold_lines = gold_sql_path.read_text(encoding="utf-8").splitlines()
    ids = manifest["question_ids"]
    assert len(ids) == len(set(ids)) == 176
    frozen = {r["question_id"]:r for r in read_json(PILOT / "inputs.json")}
    predictions = {name:{} for name in ["original", "control", "conditional"]}
    mapping, questions, sql_lines = [], [], []
    delimiter = "\t----- bird -----\t"
    for index, qid in enumerate(ids):
        assert gold[qid]["question_id"] == qid
        db = gold[qid]["db_id"]
        sql, original_db = raw[str(qid)].rsplit(delimiter, 1)
        assert original_db == db and sql.strip() == frozen[qid]["sql"]
        assert gold_lines[qid].rsplit("\t",1) == [gold[qid]["SQL"], db]
        pair = read_json(PILOT / "pairs" / f"{qid}.json")
        assert pair["question_id"] == qid and pair["status"] == "complete"
        # Preserve each original prediction string verbatim.
        predictions["original"][str(index)] = raw[str(qid)]
        for arm in ["control", "conditional"]:
            predictions[arm][str(index)] = pair["arms"][arm]["final_sql"] + delimiter + db
        mapping.append({"subset_index":index, "original_question_id":qid, "db_id":db})
        questions.append({**gold[qid], "question_id":index, "original_question_id":qid})
        sql_lines.append(gold_lines[qid])
    DEST.mkdir()
    for name, data in predictions.items():
        write_json(DEST / f"{name}.json", data)
    write_json(DEST / "dev.json", questions)
    write_json(DEST / "index_mapping.json", mapping)
    (DEST / "dev.sql").write_text("\n".join(sql_lines)+"\n", encoding="utf-8")
    commands = []
    for name in predictions:
        command = [sys.executable, "-X", "utf8", "-u", str(ROOT / "evaluation/evaluation.py"),
                   "--db_root_path", str(ROOT / "data/bird_data/dev_databases") + "/",
                   "--predicted_sql_path", str(DEST) + "/", "--ground_truth_path", str(DEST) + "/",
                   "--data_mode", "dev", "--num_cpus", "2", "--mode_gt", "gt",
                   "--mode_predict", "gpt", "--diff_json_path", str(DEST / "dev.json"),
                   "--meta_time_out", "30", "--file_name", f"{name}.json", "--start", "0", "--end", "176"]
        commands.append({"arm":name, "argv":command})
    write_json(DEST / "commands.json", commands)
    print("Exported identical 176-question subsets, reindexed consistently from 0 to 175.", flush=True)
    for item in commands:
        name = item["arm"]
        print(f"Evaluating {name} with the unmodified repository CLI...", flush=True)
        with (DEST / f"{name}.evaluation.txt").open("w", encoding="utf-8") as log:
            subprocess.run(item["argv"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print((DEST / f"{name}.evaluation.txt").read_text(encoding="utf-8"), flush=True)
    assert all(digest(Path(p)) == expected for p,expected in protected.items())
    write_json(DEST / "integrity.json", {"original_files_unchanged":True, "source_sha256":protected,
               "prediction_counts":{name:len(value) for name,value in predictions.items()},
               "question_ids":ids, "evaluator_sha256":digest(ROOT / "evaluation/evaluation.py")})
    print("All three evaluations complete; original prediction and ground-truth files unchanged.", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
