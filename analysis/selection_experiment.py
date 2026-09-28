"""Separate, gold-blind reranking of frozen saved SQL candidates.

prepare -> inspect (optional, no model calls) -> run -> export -> evaluate.
Only evaluate reads gold SQL. Preparation allowlists fields from BIRD questions.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from contextlib import closing
import hashlib
import itertools
import json
from pathlib import Path
import random
import sqlite3
import sys
import time

import sqlglot
from sqlglot import exp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.null_duplicate_ablation import sha256, write_json
from analysis.selection_experiment_client import StructuredClaudeClient
from analysis.selection_experiment_prompt import prompt_arms

DELIMITER = "\t----- bird -----\t"
ARMS = ("control", "disagreement")
CANDIDATE_FILES = [f"candidate_{name}.json" for name in (
    "full_schema", "sme_metadata", "minimal_profile", "focused_schema", "full_profile")]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def split_prediction(value, expected_db):
    sql, db = value.rsplit(DELIMITER, 1)
    if db != expected_db or not sql.strip():
        raise ValueError("Prediction database mismatch or empty SQL")
    return sql


def execute(sql, db_path, timeout=30, max_rows=1000000):
    """Read-only, bounded execution. Truncated results cannot enter the gate."""
    started = time.monotonic()
    try:
        with closing(sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            conn.execute("PRAGMA query_only=ON")
            conn.set_progress_handler(lambda: int(time.monotonic() - started > timeout), 10000)
            cursor = conn.execute(sql)
            columns = [item[0] for item in cursor.description or []]
            rows = cursor.fetchmany(max_rows + 1)
            if len(rows) > max_rows:
                return {"error": "row_limit_exceeded", "columns": columns, "rows": []}
            return {"error": None, "columns": columns, "rows": rows}
    except sqlite3.Error as exc:
        return {"error": str(exc), "columns": [], "rows": []}


def display_value(value):
    if isinstance(value, bytes):
        return {"blob_hex": value.hex()[:256], "bytes": len(value)}
    if isinstance(value, str) and len(value) > 300:
        return {"text_prefix": value[:300], "characters": len(value)}
    return value


def display_rows(rows):
    return [[display_value(v) for v in row] for row in rows]


def examples(rows, limit=3):
    # Stable representative differences without sorting/materializing a huge set.
    import heapq
    return display_rows(heapq.nsmallest(limit, rows, key=repr))


def relevant_schema(schema, sqls):
    try:
        names = {table.name.casefold() for sql in sqls
                 for table in sqlglot.parse_one(sql, read="sqlite").find_all(exp.Table)}
        chosen = {name: value for name, value in schema.items() if name.casefold() in names}
        return chosen or schema
    except Exception:
        return schema  # Parse uncertainty must not silently discard metadata.


def build_packet(record, schema, db_path, seed, timeout=30, max_rows=1000000):
    sqls = list(dict.fromkeys([*record["candidates"], record["selected"]]))
    results = {sql: execute(sql, db_path, timeout, max_rows) for sql in sqls}
    result_sets = {sql: set(result["rows"]) for sql, result in results.items() if not result["error"]}
    classes = []
    class_ids = {}
    for sql, rows in result_sets.items():
        group = next((i for i, previous in enumerate(classes) if rows == previous), None)
        if group is None:
            group = len(classes)
            classes.append(rows)
        class_ids[sql] = group
    gate_groups = {class_ids[sql] for sql in record["candidates"] if sql in class_ids}
    random.Random(f"{seed}:{record['question_id']}").shuffle(sqls)
    options, sql_by_id = [], {}
    for index, sql in enumerate(sqls, 1):
        cid = f"C{index}"
        sql_by_id[cid] = sql
        result = results[sql]
        try:
            tree = sqlglot.parse_one(sql, read="sqlite")
            projection = [node.sql(dialect="sqlite") for node in tree.selects]
        except Exception:
            projection = None
        options.append({"id": cid, "sql": sql, "output_expressions": projection,
            "columns": result["columns"], "error": result["error"],
            "row_count": len(result["rows"]) if not result["error"] else None,
            "distinct_row_count": len(result_sets[sql]) if sql in result_sets else None,
            "null_counts": [sum(row[i] is None for row in result["rows"])
                            for i in range(len(result["columns"]))] if not result["error"] else None,
            "sample_rows": display_rows(result["rows"][:3]),
            "result_group": class_ids.get(sql)})
    differences = []
    for a, b in itertools.combinations(options, 2):
        if a["error"] or b["error"] or a["result_group"] == b["result_group"]:
            continue
        left, right = result_sets[a["sql"]], result_sets[b["sql"]]
        differences.append({"left_id": a["id"], "right_id": b["id"],
            "only_left_count": len(left - right), "only_right_count": len(right - left),
            "only_left_examples": examples(left - right), "only_right_examples": examples(right - left)})
    payload = {"question": record["question"], "evidence": record["evidence"],
        "schema": relevant_schema(schema, sqls), "candidates": options,
        "current_id": next(cid for cid, sql in sql_by_id.items() if sql == record["selected"]),
        "differences": differences,
        "comparison_note": "Unordered distinct row sets; column positions matter. Examples may be truncated. No reference answers supplied."}
    return {"question_id": record["question_id"], "triggered": len(gate_groups) > 1,
            "sql_by_id": sql_by_id, "payload": payload}


def parse_selection(raw, packet):
    data = json.loads(raw)
    if not isinstance(data, dict) or set(data) != {"selected_id", "reasoning"}:
        raise ValueError("Expected selected_id and reasoning only; SQL rewriting is forbidden")
    if not isinstance(data["reasoning"], str) or not data["reasoning"].strip():
        raise ValueError("Missing reasoning")
    cid = data["selected_id"]
    if cid is None:
        return {"status": "abstained", "selected_id": packet["payload"]["current_id"], "reasoning": data["reasoning"]}
    allowed = {c["id"] for c in packet["payload"]["candidates"] if c["error"] is None}
    if not isinstance(cid, str) or cid not in allowed:
        raise ValueError("Unknown or unsuccessful candidate ID")
    return {"status": "selected", **data}


def database_path(root, name):
    base = Path(root).resolve()
    path = (base / name / f"{name}.sqlite").resolve()
    if not path.is_relative_to(base) or Path(name).name != name:
        raise ValueError("Invalid database name")
    return path


def read_schema(db_path):
    with closing(sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)) as conn:
        ddl = conn.execute("SELECT name,sql FROM sqlite_schema WHERE type IN ('table','view') "
                           "AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
    schema = {name: {"ddl": sql} for name, sql in ddl}
    sources = {}
    names = {name.casefold(): name for name in schema}
    for path in sorted((db_path.parent / "database_description").glob("*.csv")):
        raw = path.read_bytes()
        try:
            description = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            description = raw.decode("cp1252")
        name = names.get(path.stem.casefold(), path.stem)
        schema.setdefault(name, {})["supplied_description_csv"] = description
        sources[str(path)] = sha256(path)
    if not sources:
        raise ValueError(f"No supplied descriptions found for {db_path.stem}")
    return schema, sources


def code_paths():
    return [Path(__file__), ROOT / "analysis/selection_experiment_prompt.py",
            ROOT / "analysis/selection_experiment_client.py",
            ROOT / "analysis/null_duplicate_ablation.py"]


def prepare(args):
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError("Use a new output directory; existing experiments are preserved")
    selected_path = Path(args.selected).resolve()
    questions_path = Path(args.questions).resolve()
    candidate_paths = [Path(args.candidates_dir).resolve() / name for name in CANDIDATE_FILES]
    predictions = read_json(selected_path)
    pools = [read_json(path) for path in candidate_paths]
    if any(set(pool) != set(predictions) for pool in pools):
        raise ValueError("Candidate and original question ID sets differ")
    # Discard gold, difficulty, and all unrecognized fields immediately.
    entries = [{"question_id": e["question_id"], "db_id": e["db_id"],
                "question": e["question"], "evidence": e.get("evidence", "")}
               for e in read_json(questions_path)]
    if not entries or any(type(e["question_id"]) is not int or e["question_id"] < 0 for e in entries):
        raise ValueError("Expected nonempty questions with nonnegative integer IDs")
    if len({str(e["question_id"]) for e in entries}) != len(entries):
        raise ValueError("Duplicate question IDs")
    by_id = {str(e["question_id"]): e for e in entries}
    if set(by_id) != set(predictions):
        raise ValueError("Questions and predictions must have the same ID set")
    records = []
    for qid in predictions:  # Preserve the original submission order.
        entry = by_id[qid]
        records.append({**entry, "selected": split_prediction(predictions[qid], entry["db_id"]),
            "candidates": [split_prediction(pool[qid], entry["db_id"]) for pool in pools]})
    schemas, metadata, databases = {}, {}, {}
    for db in sorted({r["db_id"] for r in records}):
        path = database_path(args.db_root, db)
        schemas[db], sources = read_schema(path)
        metadata.update(sources)
        stat = path.stat()
        databases[str(path)] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    out.mkdir(parents=True)
    write_json(out / "inputs.json", records)
    write_json(out / "schemas.json", schemas)
    write_json(out / "prompts.json", prompt_arms())
    (out / "original.json").write_bytes(selected_path.read_bytes())
    manifest = {"version": 1, "n": len(records), "seed": args.seed, "model": args.model,
        "db_root": str(Path(args.db_root).resolve()), "databases": databases,
        "sqlite_version": sqlite3.sqlite_version, "sqlglot_version": sqlglot.__version__,
        "sql_timeout": args.sql_timeout, "max_rows": args.max_rows,
        "cli_timeout": 180, "max_prompt_chars": 180000,
        "question_source": {"path": str(questions_path), "sha256": sha256(questions_path)},
        "prediction_sources": {str(p): sha256(p) for p in [selected_path, *candidate_paths]},
        "metadata_sources": metadata,
        "control_prompt_source_sha256": sha256(ROOT / "src/prompt/judge.py"),
        "code_sha256": {str(p.resolve()): sha256(p) for p in code_paths()},
        "frozen_sha256": {name: sha256(out / name) for name in (
            "inputs.json", "schemas.json", "prompts.json", "original.json")},
        "protocol": "Five-candidate successful result disagreement gate; current SQL is an additional option; paired ID-only judges; no fixer",
        "gold_access": "prepare allowlists question fields; inspect/run/export never load gold; evaluate is separate",
        "limitations": ["Database availability and SQLite execution required",
            "CLI does not expose a fixed sampling seed or temperature",
            "One seed gives a reproducible shuffled display order, not a balanced repeated-order study",
            "Database fingerprints use size and mtime, not content hashes"]}
    write_json(out / "manifest.json", manifest)
    print(f"Frozen {len(records)} questions in {out}. No model calls made.", flush=True)


def load_experiment(out):
    manifest = read_json(out / "manifest.json")
    for name, expected in manifest["frozen_sha256"].items():
        if sha256(out / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    for path, expected in manifest["code_sha256"].items():
        if sha256(path) != expected:
            raise ValueError(f"Experiment code changed; prepare a new run: {path}")
    if manifest["sqlite_version"] != sqlite3.sqlite_version or manifest["sqlglot_version"] != sqlglot.__version__:
        raise ValueError("Execution/parser runtime changed; prepare a new run")
    for path, expected in manifest["databases"].items():
        stat = Path(path).stat()
        if {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns} != expected:
            raise ValueError(f"Database changed: {path}")
    return manifest, read_json(out / "inputs.json"), read_json(out / "schemas.json")


def digest_object(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def checkpoint_write(path, value):
    write_json(path, {"sha256": digest_object(value), "data": value})


def checkpoint_read(path):
    wrapper = read_json(path)
    if digest_object(wrapper["data"]) != wrapper["sha256"]:
        raise ValueError(f"Checkpoint modified: {path}")
    return wrapper["data"]


def packet_for(record, schema, manifest, out):
    path = out / "packets" / f"{record['question_id']}.json"
    if path.exists():
        packet = checkpoint_read(path)
        if packet["question_id"] != record["question_id"] or set(packet["sql_by_id"].values()) != set(
                [record["selected"], *record["candidates"]]):
            raise ValueError("Packet does not match frozen question/candidates")
        return packet
    packet = build_packet(record, schema, database_path(manifest["db_root"], record["db_id"]),
        manifest["seed"], manifest["sql_timeout"], manifest["max_rows"])
    path.parent.mkdir(exist_ok=True)
    checkpoint_write(path, packet)
    return packet


def judge_packet(packet, system, client):
    try:
        raw = client.complete(json.dumps(packet["payload"], ensure_ascii=False),
                              system_prompt=system, max_tokens=2048, temperature=0)
        outcome = parse_selection(raw, packet)
    except Exception as exc:
        outcome = {"status": "failed", "selected_id": packet["payload"]["current_id"],
                   "reasoning": "Retained original after failed review", "error": f"{type(exc).__name__}: {exc}"}
    outcome["sql"] = packet["sql_by_id"][outcome["selected_id"]]
    return outcome


def process_record(record, schema, manifest, prompts, out, inspect_only=False, client_factory=StructuredClaudeClient):
    qid = record["question_id"]
    pair_path = out / "pairs" / f"{qid}.json"
    if pair_path.exists() and not inspect_only:
        return checkpoint_read(pair_path)
    packet = packet_for(record, schema, manifest, out)
    if inspect_only:
        return {"question_id": qid, "triggered": packet["triggered"]}
    pair = {"question_id": qid, "triggered": packet["triggered"], "arms": {}}
    order = list(ARMS)
    random.Random(f"arms:{manifest['seed']}:{qid}").shuffle(order)
    pair["arm_order"] = order
    for arm in order:
        destination = out / "calls" / str(qid) / arm
        destination.mkdir(parents=True, exist_ok=True)
        checkpoint = destination / "outcome.json"
        if checkpoint.exists():
            outcome = checkpoint_read(checkpoint)
        elif not packet["triggered"]:
            outcome = {"status": "not_triggered", "sql": record["selected"]}
        elif max(map(len, prompts.values())) + len(json.dumps(packet["payload"], ensure_ascii=False)) > manifest["max_prompt_chars"]:
            outcome = {"status": "failed", "sql": record["selected"], "error": "prompt_budget_exceeded"}
        else:
            client = client_factory(manifest["model"], destination, manifest["cli_timeout"])
            outcome = judge_packet(packet, prompts[arm], client)
        checkpoint_write(checkpoint, outcome)
        pair["arms"][arm] = outcome
    pair_path.parent.mkdir(exist_ok=True)
    checkpoint_write(pair_path, pair)
    return pair


def run(args):
    out = Path(args.out).resolve()
    manifest, records, schemas = load_experiment(out)
    prompts = read_json(out / "prompts.json")
    chosen = records[:args.limit] if args.limit else records
    # One coordinator per directory; checkpoints allow a later restart after interruption.
    lock = out / ".running"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write("A coordinator is using this directory. Remove only after it has stopped.\n")
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_record, r, schemas[r["db_id"]], manifest, prompts, out,
                                   args.command == "inspect") for r in chosen]
            for count, future in enumerate(as_completed(futures), 1):
                result = future.result()
                print(f"{count}/{len(chosen)} Q{result['question_id']}: review={result['triggered']}", flush=True)
    finally:
        lock.unlink()


def export(args):
    out = Path(args.out).resolve()
    manifest, records, _ = load_experiment(out)
    pairs = {}
    for record in records:
        path = out / "pairs" / f"{record['question_id']}.json"
        if not path.exists():
            raise ValueError("Finish both arms for every question before full export/evaluation")
        pairs[str(record["question_id"])] = checkpoint_read(path)
    original = read_json(out / "original.json")
    predictions = {arm: dict(original) for arm in ARMS}
    counts = {arm: {} for arm in ARMS}
    changed = {arm: [] for arm in ARMS}
    for record in records:
        qid = str(record["question_id"])
        pair = pairs[qid]
        if pair["question_id"] != record["question_id"] or set(pair["arms"]) != set(ARMS):
            raise ValueError("Incomplete or mismatched pair")
        for arm in ARMS:
            outcome = pair["arms"][arm]
            if outcome["sql"] not in [record["selected"], *record["candidates"]]:
                raise ValueError("Outcome SQL is not an exact frozen candidate")
            if (not pair["triggered"] or outcome["status"] != "selected") and outcome["sql"] != record["selected"]:
                raise ValueError("Fallback changed the original SQL")
            counts[arm][outcome["status"]] = counts[arm].get(outcome["status"], 0) + 1
            if outcome["sql"] != record["selected"]:
                predictions[arm][qid] = outcome["sql"] + DELIMITER + record["db_id"]
                changed[arm].append(record["question_id"])
    for path, expected in manifest["prediction_sources"].items():
        if sha256(path) != expected:
            raise ValueError(f"Original predictions changed since preparation: {path}")
    destination = out / "full_results"
    if destination.exists():
        raise FileExistsError("Full export already exists; it will not be overwritten")
    destination.mkdir()
    (destination / "original.json").write_bytes((out / "original.json").read_bytes())
    for arm in ARMS:
        write_json(destination / f"selected_{arm}.json", predictions[arm])
    write_json(destination / "manifest.json", {"n": manifest["n"], "statuses": counts,
        "triggered": sum(p["triggered"] for p in pairs.values()), "changed_ids": changed,
        "original_sources_unchanged": True,
        "sha256": {p.name: sha256(p) for p in destination.glob("*.json")}})
    print(json.dumps({"n": manifest["n"], "statuses": counts, "changed": {k: len(v) for k, v in changed.items()}}))


def score_job(job):
    # Imported only during the explicit, post-inference evaluate stage.
    from evaluation.evaluation import execute_model
    return {**execute_model(job["sql"], job["gold"], job["db_path"], job["question_id"], job["timeout"]),
            "question_id": job["question_id"], "names": job["names"]}


def evaluate(args):
    out = Path(args.out).resolve()
    manifest, records, _ = load_experiment(out)
    exported = out / "full_results"
    export_manifest = read_json(exported / "manifest.json")  # Must precede any gold access.
    for name, expected in export_manifest["sha256"].items():
        if sha256(exported / name) != expected:
            raise ValueError("Export changed after completion")
    destination = exported / "evaluation"
    if destination.exists():
        raise FileExistsError("Evaluation already exists; preserve it")
    questions = Path(args.questions or manifest["question_source"]["path"]).resolve()
    if sha256(questions) != manifest["question_source"]["sha256"]:
        raise ValueError("Gold question source differs from frozen preparation source")
    gold = {str(e["question_id"]): e for e in read_json(questions)}
    predictions = {"original": read_json(exported / "original.json"),
        **{arm: read_json(exported / f"selected_{arm}.json") for arm in ARMS}}
    jobs = []
    for record in records:
        qid, db = record["question_id"], record["db_id"]
        if gold[str(qid)]["db_id"] != db:
            raise ValueError("Gold database mismatch")
        sqls = {name: split_prediction(values[str(qid)], db) for name, values in predictions.items()}
        sqls.update({f"candidate_{i}": sql for i, sql in enumerate(record["candidates"])})
        grouped = {}
        for name, sql in sqls.items():
            grouped.setdefault(sql, []).append(name)
        jobs.extend({"question_id": qid, "sql": sql, "gold": gold[str(qid)]["SQL"],
            "db_path": str(database_path(manifest["db_root"], db)),
            "names": names, "timeout": manifest["sql_timeout"]} for sql, names in grouped.items())
    scores = {r["question_id"]: {} for r in records}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(score_job, job) for job in jobs]
        for count, future in enumerate(as_completed(futures), 1):
            result = future.result()
            for name in result["names"]:
                scores[result["question_id"]][name] = result["res"]
            if count % 100 == 0 or count == len(jobs):
                print(f"Evaluated {count}/{len(jobs)} unique SQL/reference pairs", flush=True)
    details = []
    for record in records:
        qid = record["question_id"]
        pair = checkpoint_read(out / "pairs" / f"{qid}.json")
        row = scores[qid]
        details.append({"question_id": qid, "db_id": record["db_id"], "triggered": pair["triggered"],
            "scores": {name: row[name] for name in predictions},
            "passing_candidate_count": sum(row[f"candidate_{i}"] for i in range(5)),
            "statuses": {arm: pair["arms"][arm]["status"] for arm in ARMS}})
    summary = {"n": len(records), "unique_sql_pairs": len(jobs), "scores": {},
        "triggered": sum(r["triggered"] for r in details),
        "original_correct_in_trigger": sum(r["triggered"] and r["scores"]["original"] for r in details),
        "original_failures_with_passing_candidate": sum(not r["scores"]["original"] and r["passing_candidate_count"] > 0 for r in details),
        "evaluator_sha256": sha256(ROOT / "evaluation/evaluation.py"),
        "scoring": "Unmodified repository execute_model; identical SQL shares one result. Errors/timeouts score zero. Passing means execution match, not semantic proof."}
    for name in predictions:
        correct = sum(r["scores"][name] for r in details)
        summary["scores"][name] = {"correct": correct, "accuracy_percent": 100 * correct / len(details),
            "recovered_ids": [r["question_id"] for r in details if r["scores"][name] > r["scores"]["original"]],
            "regressed_ids": [r["question_id"] for r in details if r["scores"][name] < r["scores"]["original"]],
            "by_database": {db: {"n": sum(r["db_id"] == db for r in details),
                "correct": sum(r["scores"][name] for r in details if r["db_id"] == db)}
                for db in sorted({r["db_id"] for r in details})}}
    summary["statuses"] = export_manifest["statuses"]
    # Provider usage and cost are reported when the CLI supplies them; never fabricated.
    responses = [read_json(p) for p in (out / "calls").glob("*/*/*.response.json")]
    costs = [r["total_cost_usd"] for r in responses if isinstance(r.get("total_cost_usd"), (int, float))]
    summary["model_usage"] = {"logged_responses": len(responses), "responses_with_cost": len(costs),
        "reported_total_cost_usd": sum(costs) if costs else None,
        "note": "CLI-reported cost may not represent subscription billing; full raw usage is in call logs."}
    destination.mkdir()
    write_json(destination / "per_question.json", details)
    write_json(destination / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "scores"}, indent=2))
    for name, value in summary["scores"].items():
        print(f"{name}: {value['correct']}/{len(details)} = {value['accuracy_percent']:.2f}%")


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "inspect", "run", "export", "evaluate"):
        sub = commands.add_parser(name)
        sub.add_argument("--out", default="output/selection_experiment/v2")
        if name == "prepare":
            sub.add_argument("--questions", default="data/bird_data/dev.json")
            sub.add_argument("--selected", default="output/claude_headless_v6/selected.json")
            sub.add_argument("--candidates-dir", default="output/claude_headless_v6")
            sub.add_argument("--db-root", default="data/bird_data/dev_databases")
            sub.add_argument("--seed", default="selection-v1-order-1")
            sub.add_argument("--model", default="sonnet")
            sub.add_argument("--sql-timeout", type=positive, default=30)
            sub.add_argument("--max-rows", type=positive, default=1000000)
        if name in ("inspect", "run", "evaluate"):
            sub.add_argument("--workers", type=positive, default=2)
        if name in ("inspect", "run"):
            sub.add_argument("--limit", type=positive, help="Process the first N original IDs; no gold-based subset")
        if name == "evaluate":
            sub.add_argument("--questions", help="Optional matching original question/gold JSON")
    args = parser.parse_args()
    {"prepare": prepare, "inspect": run, "run": run, "export": export, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
