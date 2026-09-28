"""Projection-alignment experiment: rewrite only the SELECT list of the selected SQL.

prepare -> run -> export -> evaluate. Only evaluate reads gold. Reuses the frozen,
gold-free inputs of output/selection_experiment/v2 and the repository evaluator.
Production code under src/ is never imported for mutation and never modified.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from contextlib import closing
import json
from pathlib import Path
import shutil
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import jsonschema
import sqlglot
from sqlglot import exp
import analysis.selection_experiment as base
from analysis.focused_selection_client import complete
from experiments.projection_alignment.prompt import PROMPT, SCHEMA

AGGREGATES = (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max, exp.AggFunc)


def code_paths():
    return [Path(__file__), Path(__file__).with_name("prompt.py"),
            ROOT / "analysis/focused_selection_client.py", ROOT / "analysis/selection_experiment.py"]


def table_columns(db_path):
    with closing(sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        names = [r[0] for r in conn.execute("SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        return {name.casefold(): [r[1] for r in conn.execute(f'PRAGMA table_info("{name}")')] for name in names}


def prepare(args):
    out, source = Path(args.out).resolve(), Path(args.source).resolve()
    if out.exists():
        raise FileExistsError("Choose an unused experiment directory")
    old = base.read_json(source / "manifest.json")
    for path, expected in {**old["prediction_sources"], **old["metadata_sources"]}.items():
        if base.sha256(path) != expected:
            raise ValueError(f"Original source changed: {path}")
    records = base.read_json(source / "inputs.json")
    out.mkdir(parents=True)
    for name in ("inputs.json", "schemas.json", "original.json"):
        shutil.copyfile(source / name, out / name)
    columns = {db: table_columns(base.database_path(old["db_root"], db)) for db in sorted({r["db_id"] for r in records})}
    base.write_json(out / "columns.json", columns)
    base.write_json(out / "prompts.json", {"align": PROMPT, "schema": SCHEMA})
    manifest = {"version": "projection-v1", "n": len(records), "source_experiment": str(source),
        "model": args.model or old["model"], "db_root": old["db_root"], "databases": old["databases"],
        "sql_timeout": old["sql_timeout"], "max_rows": old["max_rows"],
        "question_source": old["question_source"], "prediction_sources": old["prediction_sources"],
        "sqlite_version": sqlite3.sqlite_version, "sqlglot_version": sqlglot.__version__,
        "protocol": "One request per question; SELECT-list-only rewrite validated by AST rules; adopt only if it executes and is not newly empty",
        "gold_access": "Frozen gold-free inputs; evaluate separately reads matching original gold",
        "frozen_sha256": {name: base.sha256(out / name) for name in ("inputs.json", "schemas.json", "original.json", "columns.json", "prompts.json")},
        "code_sha256": {str(p.resolve()): base.sha256(p) for p in code_paths()}}
    base.write_json(out / "manifest.json", manifest)
    print(f"Prepared {len(records)} frozen questions; no model calls", flush=True)


def load(out):
    manifest = base.read_json(out / "manifest.json")
    for name, expected in manifest["frozen_sha256"].items():
        if base.sha256(out / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    for path, expected in manifest["code_sha256"].items():
        if base.sha256(path) != expected:
            raise ValueError(f"Experiment code changed; prepare a new run: {path}")
    return manifest, base.read_json(out / "inputs.json"), base.read_json(out / "schemas.json"), base.read_json(out / "columns.json")


def footprint(sql, db_path, timeout, max_rows):
    result = base.execute(sql, db_path, timeout, max_rows)
    if result["error"]:
        return {"error": result["error"], "columns": [], "row_count": None, "null_counts": None, "sample_rows": []}
    rows = result["rows"]
    return {"error": None, "columns": result["columns"], "row_count": len(rows),
            "null_counts": [sum(r[i] is None for r in rows) for i in range(len(result["columns"]))],
            "sample_rows": base.display_rows(rows[:3])}


def payload_for(record, schema, fp):
    # Question, evidence, schema of the tables the query reads, the SQL and its own execution shape.
    return {"question": record["question"], "evidence": record["evidence"],
            "schema": base.relevant_schema(schema, [record["selected"]]), "sql": record["selected"],
            "execution": {k: fp[k] for k in ("error", "columns", "row_count", "null_counts", "sample_rows")}}


def validate_answer(answer, payload):
    jsonschema.validate(answer, SCHEMA)
    positions = [s["position"] for s in answer["slots"]]
    if positions != list(range(1, len(positions) + 1)):
        raise ValueError("Slot positions must be 1..n in order")
    text = payload["question"] + "\n" + payload["evidence"]
    for slot in answer["slots"]:
        if slot["source_quote"] not in text:
            raise ValueError("Slot quote does not occur in question or evidence")
    if answer["change_needed"] and len(answer["select_list"]) != len(answer["slots"]):
        raise ValueError("select_list must have one expression per slot")


def _query_tables(select):
    """Map alias/table name (casefolded) -> table name (casefolded) for the outer FROM/JOINs."""
    tables = {}
    for node in select.find_all(exp.Table):
        if node.find_ancestor(exp.Select) is not select:
            continue
        name = node.name.casefold()
        tables[name] = name
        if node.alias:
            tables[node.alias.casefold()] = name
    return tables


def apply_rewrite(sql, select_list, columns):
    """Return the rewritten SQL or raise ValueError. Only the outer SELECT list may change."""
    tree = sqlglot.parse_one(sql, read="sqlite")
    if not isinstance(tree, exp.Select):
        raise ValueError("Only a single outer SELECT can be rewritten")
    if not select_list or len(select_list) > 6:
        raise ValueError("New select list must have 1-6 expressions")
    original = [e.sql(dialect="sqlite") for e in tree.expressions]
    original_norm = {sqlglot.parse_one(e, read="sqlite").sql(dialect="sqlite").casefold() for e in original}
    tables = _query_tables(tree)
    restricted = bool(tree.args.get("group")) or any(e.find(*AGGREGATES) for e in tree.expressions)
    new_exprs = []
    for text in select_list:
        try:
            node = sqlglot.parse_one(text, read="sqlite")
        except Exception as exc:
            raise ValueError(f"Unparseable expression: {exc}")
        inner = node.this if isinstance(node, exp.Alias) else node
        if isinstance(node, exp.Select) or node.find(exp.Select) or isinstance(inner, exp.Star):
            raise ValueError("Subqueries and * are not permitted in the select list")
        if inner.sql(dialect="sqlite").casefold() in original_norm:
            new_exprs.append(node)
            continue
        if not isinstance(inner, exp.Column):
            raise ValueError("New expressions must be plain columns of tables already read")
        if restricted:
            raise ValueError("Cannot add bare columns to an aggregated or grouped query")
        qualifier = inner.table.casefold() if inner.table else None
        if qualifier and qualifier not in tables:
            raise ValueError(f"Unknown table alias {inner.table}")
        candidates = [tables[qualifier]] if qualifier else sorted(set(tables.values()))
        matches = [t for t in candidates if inner.name.casefold() in {c.casefold() for c in columns.get(t, [])}]
        if len(matches) != 1:
            raise ValueError(f"Column {inner.name} is not an unambiguous column of the query's tables")
        new_exprs.append(node)
    if [e.sql(dialect="sqlite").casefold() for e in new_exprs] == [e.casefold() for e in original]:
        raise ValueError("Rewrite is identical to the original select list")
    rewritten = tree.copy()
    rewritten.set("expressions", new_exprs)
    # Everything except the select list must be untouched.
    check = rewritten.copy(); check.set("expressions", tree.expressions)
    if check.sql(dialect="sqlite") != tree.sql(dialect="sqlite"):
        raise ValueError("Rewrite changed something outside the select list")
    return rewritten.sql(dialect="sqlite")


def decide(record, answer, columns, db_path, timeout, max_rows, original_fp):
    if not answer["change_needed"]:
        return {"status": "retained", "sql": record["selected"], "reasoning": "Model found no projection violation: " + answer["reason"]}
    try:
        new_sql = apply_rewrite(record["selected"], answer["select_list"], columns)
    except ValueError as exc:
        return {"status": "rejected", "sql": record["selected"], "reasoning": f"Rewrite rejected by validator: {exc}"}
    fp = footprint(new_sql, db_path, timeout, max_rows)
    if fp["error"]:
        return {"status": "rejected", "sql": record["selected"], "reasoning": f"Rewrite failed to execute: {fp['error']}"}
    if fp["row_count"] == 0 and (original_fp["row_count"] or 0) > 0:
        return {"status": "rejected", "sql": record["selected"], "reasoning": "Rewrite returned no rows while the original did"}
    return {"status": "rewritten", "sql": new_sql, "reasoning": answer["reason"], "new_execution": fp}


def process(record, schema, columns, manifest, prompts, out, client=complete):
    qid = record["question_id"]
    destination = out / "results" / f"{qid}.json"
    if destination.exists():
        return base.checkpoint_read(destination)
    db_path = base.database_path(manifest["db_root"], record["db_id"])
    try:
        fp = footprint(record["selected"], db_path, manifest["sql_timeout"], manifest["max_rows"])
        payload = payload_for(record, schema, fp)
        folder = out / "calls" / str(qid)
        folder.mkdir(parents=True, exist_ok=True)
        answer_path = folder / "validated.json"
        if answer_path.exists():
            answer = base.checkpoint_read(answer_path)
        else:
            answer = client(payload, prompts["align"], SCHEMA, folder, manifest["model"])
            validate_answer(answer, payload)
            base.checkpoint_write(answer_path, answer)
        outcome = decide(record, answer, columns[record["db_id"]], db_path, manifest["sql_timeout"], manifest["max_rows"], fp)
    except Exception as exc:
        outcome = {"status": "failed", "sql": record["selected"], "reasoning": "Retained original after failure",
                   "error": f"{type(exc).__name__}: {exc}"}
    result = {"question_id": qid, "outcome": outcome}
    destination.parent.mkdir(exist_ok=True)
    base.checkpoint_write(destination, result)
    return result


def run(args):
    out = Path(args.out).resolve()
    manifest, records, schemas, columns = load(out)
    prompts = base.read_json(out / "prompts.json")
    records = records[:args.limit] if args.limit else records
    lock = out / ".running"
    with lock.open("x", encoding="utf-8") as f:
        f.write("Coordinator running; remove only after it stops")
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process, r, schemas[r["db_id"]], columns, manifest, prompts, out) for r in records]
            for count, future in enumerate(as_completed(futures), 1):
                result = future.result()
                o = result["outcome"]
                print(f"{count}/{len(records)} Q{result['question_id']}: {o['status']} {o.get('error', '')}", flush=True)
    finally:
        lock.unlink()


def export(args):
    out = Path(args.out).resolve()
    manifest, records, _, _ = load(out)
    for path, expected in manifest["prediction_sources"].items():
        if base.sha256(path) != expected:
            raise ValueError(f"Original predictions changed since preparation: {path}")
    predictions, counts, changed = {}, {}, []
    for record in records:
        qid = record["question_id"]
        result = base.checkpoint_read(out / "results" / f"{qid}.json")
        o = result["outcome"]
        if o["status"] != "rewritten" and o["sql"] != record["selected"]:
            raise ValueError("Non-rewritten outcome changed the original SQL")
        counts[o["status"]] = counts.get(o["status"], 0) + 1
        if o["sql"] != record["selected"]:
            changed.append(qid)
        predictions[str(qid)] = o["sql"] + base.DELIMITER + record["db_id"]
    if len(predictions) != manifest["n"]:
        raise ValueError("Export must contain every question")
    destination = out / "full_results"
    if destination.exists():
        raise FileExistsError("Full export already exists; it will not be overwritten")
    destination.mkdir()
    (destination / "original.json").write_bytes((out / "original.json").read_bytes())
    base.write_json(destination / "selected_projection.json", predictions)
    base.write_json(destination / "manifest.json", {"n": manifest["n"], "statuses": counts, "changed_ids": changed,
        "original_sources_unchanged": True, "sha256": {p.name: base.sha256(p) for p in destination.glob("*.json")}})
    print(json.dumps({"n": manifest["n"], "statuses": counts, "changed": len(changed)}))


def evaluate(args):
    out = Path(args.out).resolve()
    manifest, records, _, _ = load(out)
    exported = out / "full_results"
    export_manifest = base.read_json(exported / "manifest.json")
    for name, expected in export_manifest["sha256"].items():
        if base.sha256(exported / name) != expected:
            raise ValueError("Export changed after completion")
    destination = exported / "evaluation"
    if destination.exists():
        raise FileExistsError("Evaluation already exists; preserve it")
    questions = Path(manifest["question_source"]["path"])
    if base.sha256(questions) != manifest["question_source"]["sha256"]:
        raise ValueError("Gold question source differs from frozen preparation source")
    gold = {str(e["question_id"]): e for e in base.read_json(questions)}
    predictions = {"original": base.read_json(exported / "original.json"),
                   "projection": base.read_json(exported / "selected_projection.json")}
    jobs = []
    for record in records:
        qid, db = record["question_id"], record["db_id"]
        sqls = {name: base.split_prediction(values[str(qid)], db) for name, values in predictions.items()}
        grouped = {}
        for name, sql in sqls.items():
            grouped.setdefault(sql, []).append(name)
        jobs.extend({"question_id": qid, "sql": sql, "gold": gold[str(qid)]["SQL"],
                     "db_path": str(base.database_path(manifest["db_root"], db)), "names": names,
                     "timeout": manifest["sql_timeout"]} for sql, names in grouped.items())
    scores = {r["question_id"]: {} for r in records}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(base.score_job, job) for job in jobs]
        for count, future in enumerate(as_completed(futures), 1):
            result = future.result()
            for name in result["names"]:
                scores[result["question_id"]][name] = result["res"]
            if count % 100 == 0 or count == len(jobs):
                print(f"Evaluated {count}/{len(jobs)} unique SQL/reference pairs", flush=True)
    details = []
    for record in records:
        qid = record["question_id"]
        o = base.checkpoint_read(out / "results" / f"{qid}.json")["outcome"]
        details.append({"question_id": qid, "db_id": record["db_id"], "status": o["status"],
                        "scores": scores[qid], "changed": o["sql"] != record["selected"]})
    summary = {"n": len(records), "unique_sql_pairs": len(jobs), "statuses": export_manifest["statuses"],
               "evaluator_sha256": base.sha256(ROOT / "evaluation/evaluation.py"),
               "scoring": "Unmodified repository execute_model; passing means execution match, not semantic proof.", "scores": {}}
    for name in predictions:
        correct = sum(r["scores"][name] for r in details)
        summary["scores"][name] = {"correct": correct, "accuracy_percent": 100 * correct / len(details),
            "recovered_ids": [r["question_id"] for r in details if r["scores"][name] > r["scores"]["original"]],
            "regressed_ids": [r["question_id"] for r in details if r["scores"][name] < r["scores"]["original"]],
            "by_database": {db: {"n": sum(r["db_id"] == db for r in details), "correct": sum(r["scores"][name] for r in details if r["db_id"] == db)}
                            for db in sorted({r["db_id"] for r in details})}}
    responses = [base.read_json(p) for p in (out / "calls").glob("*/response.json")]
    summary["model_usage"] = {"logged_responses": len(responses),
        "reported_total_cost_usd": sum(r.get("total_cost_usd", 0) for r in responses if isinstance(r.get("total_cost_usd"), (int, float))),
        "models": sorted({m for r in responses for m in r.get("modelUsage", {})}),
        "note": "CLI reported list cost, not necessarily subscription billing"}
    destination.mkdir()
    base.write_json(destination / "summary.json", summary)
    base.write_json(destination / "per_question.json", details)
    for name, s in summary["scores"].items():
        print(f"{name}: {s['correct']}/{len(details)} = {s['accuracy_percent']:.2f}%  recovered {len(s['recovered_ids'])}  regressed {len(s['regressed_ids'])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "export", "evaluate"):
        p = subs.add_parser(name)
        p.add_argument("--out", default="output/projection_alignment/v1")
        if name == "prepare":
            p.add_argument("--source", default="output/selection_experiment/v2")
            p.add_argument("--model", default=None)
        if name in ("run", "evaluate"):
            p.add_argument("--workers", type=base.positive, default=4 if name == "run" else 2)
        if name == "run":
            p.add_argument("--limit", type=base.positive)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "export": export, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
