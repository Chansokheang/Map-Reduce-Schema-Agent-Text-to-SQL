"""Question-form output experiment: apply wording-based output conventions once.

prepare -> run -> export -> evaluate. Only evaluate reads gold. Questions are classified by
question text alone (patterns.py); only matched questions get one model call. The base is
the original selected SQL from the frozen, gold-free inputs of output/selection_experiment/v2.
src/ is never modified.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import collections
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
from experiments.projection_alignment.runner import footprint
from experiments.question_form_output.patterns import classify, FORMS
from experiments.question_form_output.prompt import prompt_for, SCHEMA, CONVENTIONS, BASE

FIXED_CLAUSES = ("from_", "joins", "where", "having", "order", "limit", "with_", "with")


def code_paths():
    here = Path(__file__).parent
    return [Path(__file__), here / "patterns.py", here / "prompt.py", ROOT / "experiments/projection_alignment/runner.py",
            ROOT / "analysis/focused_selection_client.py", ROOT / "analysis/selection_experiment.py"]


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
    forms = {str(r["question_id"]): classify(r["question"]) for r in records}
    base.write_json(out / "forms.json", forms)
    base.write_json(out / "prompts.json", {"base": BASE, "conventions": CONVENTIONS, "schema": SCHEMA})
    manifest = {"version": "question-form-v1", "n": len(records), "source_experiment": str(source),
        "model": args.model or old["model"], "db_root": old["db_root"], "databases": old["databases"],
        "sql_timeout": old["sql_timeout"], "max_rows": old["max_rows"],
        "question_source": old["question_source"], "prediction_sources": old["prediction_sources"],
        "sqlite_version": sqlite3.sqlite_version, "sqlglot_version": sqlglot.__version__,
        "form_counts": dict(collections.Counter(forms.values())),
        "protocol": "Question text classified into five forms; one request per matched question applying that form's "
                    "output convention; only SELECT list, DISTINCT and (count_then_list) GROUP BY may change; adopt only "
                    "if it executes and is not newly empty. Unmatched questions keep original SQL without a call.",
        "preregistered_reporting": "Full set with all five forms, plus per-form recovered/regressed. list_entity flagged "
                                   "in advance as risky (id-over-name did not hold on train gold).",
        "gold_access": "Frozen gold-free inputs; evaluate separately reads matching original gold",
        "frozen_sha256": {name: base.sha256(out / name) for name in ("inputs.json", "schemas.json", "original.json", "forms.json", "prompts.json")},
        "code_sha256": {str(p.resolve()): base.sha256(p) for p in code_paths()}}
    base.write_json(out / "manifest.json", manifest)
    print(json.dumps(manifest["form_counts"]), flush=True)


def load(out):
    manifest = base.read_json(out / "manifest.json")
    for name, expected in manifest["frozen_sha256"].items():
        if base.sha256(out / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    for path, expected in manifest["code_sha256"].items():
        if base.sha256(path) != expected:
            raise ValueError(f"Experiment code changed; prepare a new run: {path}")
    return manifest, base.read_json(out / "inputs.json"), base.read_json(out / "schemas.json"), base.read_json(out / "forms.json")


def _clause(tree, key):
    value = tree.args.get(key)
    if value is None:
        return None
    if isinstance(value, list):
        return [v.sql(dialect="sqlite") for v in value]
    return value.sql(dialect="sqlite")


def validate_rewrite(original, rewritten, form):
    """Raise ValueError unless only the permitted parts of the outer SELECT changed."""
    try:
        old = sqlglot.parse_one(original, read="sqlite")
        new = sqlglot.parse_one(rewritten, read="sqlite")
    except Exception as exc:
        raise ValueError(f"Unparseable SQL: {exc}")
    if not isinstance(old, exp.Select) or not isinstance(new, exp.Select):
        raise ValueError("Only a single outer SELECT can be rewritten")
    if new.sql(dialect="sqlite") == old.sql(dialect="sqlite"):
        raise ValueError("Rewrite is identical to the original")
    for key in FIXED_CLAUSES:
        if _clause(old, key) != _clause(new, key):
            raise ValueError(f"Rewrite changed a fixed clause: {key}")
    if form != "count_then_list" and _clause(old, "group") != _clause(new, "group"):
        raise ValueError("Rewrite changed GROUP BY")
    old_subqueries = {s.sql(dialect="sqlite") for e in old.expressions for s in e.find_all(exp.Select)}
    for e in new.expressions:
        if isinstance(e, exp.Star) or (isinstance(e, exp.Alias) and isinstance(e.this, exp.Star)):
            raise ValueError("* is not permitted in the output")
        if any(s.sql(dialect="sqlite") not in old_subqueries for s in e.find_all(exp.Select)):
            raise ValueError("New subqueries are not permitted in the output")
        if e.find(exp.Window) and form != "rank":
            raise ValueError("Window functions are only permitted for rank questions")
    return new.sql(dialect="sqlite")


def payload_for(record, schema, fp):
    return {"question": record["question"], "evidence": record["evidence"],
            "schema": base.relevant_schema(schema, [record["selected"]]), "sql": record["selected"],
            "execution": {k: fp[k] for k in ("error", "columns", "row_count", "null_counts", "sample_rows")}}


def decide(record, form, answer, db_path, manifest, original_fp):
    if not answer["change_needed"]:
        return {"status": "retained", "sql": record["selected"], "reasoning": "Model: " + answer["reason"]}
    try:
        new_sql = validate_rewrite(record["selected"], answer["sql"], form)
    except ValueError as exc:
        return {"status": "rejected", "sql": record["selected"], "reasoning": f"Validator: {exc}"}
    fp = footprint(new_sql, db_path, manifest["sql_timeout"], manifest["max_rows"])
    if fp["error"]:
        return {"status": "rejected", "sql": record["selected"], "reasoning": f"Rewrite failed to execute: {fp['error']}"}
    if fp["row_count"] == 0 and (original_fp["row_count"] or 0) > 0:
        return {"status": "rejected", "sql": record["selected"], "reasoning": "Rewrite returned no rows while the original did"}
    return {"status": "rewritten", "sql": new_sql, "reasoning": answer["reason"], "new_execution": fp}


def process(record, schema, form, manifest, out, client=complete):
    qid = record["question_id"]
    destination = out / "results" / f"{qid}.json"
    if destination.exists():
        return base.checkpoint_read(destination)
    if form is None:
        outcome = {"status": "not_matched", "sql": record["selected"]}
    else:
        db_path = base.database_path(manifest["db_root"], record["db_id"])
        try:
            fp = footprint(record["selected"], db_path, manifest["sql_timeout"], manifest["max_rows"])
            folder = out / "calls" / str(qid)
            folder.mkdir(parents=True, exist_ok=True)
            answer_path = folder / "validated.json"
            if answer_path.exists():
                answer = base.checkpoint_read(answer_path)
            else:
                answer = client(payload_for(record, schema, fp), prompt_for(form), SCHEMA, folder, manifest["model"])
                jsonschema.validate(answer, SCHEMA)
                base.checkpoint_write(answer_path, answer)
            outcome = decide(record, form, answer, db_path, manifest, fp)
        except Exception as exc:
            outcome = {"status": "failed", "sql": record["selected"], "reasoning": "Retained original after failure",
                       "error": f"{type(exc).__name__}: {exc}"}
    result = {"question_id": qid, "form": form, "outcome": outcome}
    destination.parent.mkdir(parents=True, exist_ok=True)
    base.checkpoint_write(destination, result)
    return result


def run(args):
    out = Path(args.out).resolve()
    manifest, records, schemas, forms = load(out)
    lock = out / ".running"
    with lock.open("x", encoding="utf-8") as f:
        f.write("Coordinator running; remove only after it stops")
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process, r, schemas[r["db_id"]], forms[str(r["question_id"])], manifest, out) for r in records]
            for count, future in enumerate(as_completed(futures), 1):
                result = future.result()
                o = result["outcome"]
                if o["status"] != "not_matched":
                    print(f"{count}/{len(records)} Q{result['question_id']} [{result['form']}]: {o['status']} {o.get('error', '')}", flush=True)
        print(f"done {len(records)}", flush=True)
    finally:
        lock.unlink()


def export(args):
    out = Path(args.out).resolve()
    manifest, records, _, _ = load(out)
    for path, expected in manifest["prediction_sources"].items():
        if base.sha256(path) != expected:
            raise ValueError(f"Original predictions changed since preparation: {path}")
    predictions, counts, changed = {}, collections.Counter(), []
    for record in records:
        qid = record["question_id"]
        result = base.checkpoint_read(out / "results" / f"{qid}.json")
        o = result["outcome"]
        if o["status"] != "rewritten" and o["sql"] != record["selected"]:
            raise ValueError("Non-rewritten outcome changed the original SQL")
        counts[f"{result['form']}:{o['status']}"] += 1
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
    base.write_json(destination / "selected_question_form.json", predictions)
    base.write_json(destination / "manifest.json", {"n": manifest["n"], "statuses": dict(counts), "changed_ids": changed,
        "original_sources_unchanged": True, "sha256": {p.name: base.sha256(p) for p in destination.glob("*.json")}})
    print(json.dumps({"statuses": dict(counts), "changed": len(changed)}))


def evaluate(args):
    out = Path(args.out).resolve()
    manifest, records, _, forms = load(out)
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
                   "question_form": base.read_json(exported / "selected_question_form.json")}
    jobs = []
    for record in records:
        qid, db = record["question_id"], record["db_id"]
        grouped = {}
        for name, values in predictions.items():
            grouped.setdefault(base.split_prediction(values[str(qid)], db), []).append(name)
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
            if count % 200 == 0 or count == len(jobs):
                print(f"Evaluated {count}/{len(jobs)} unique SQL/reference pairs", flush=True)
    details = []
    for record in records:
        qid = record["question_id"]
        result = base.checkpoint_read(out / "results" / f"{qid}.json")
        details.append({"question_id": qid, "db_id": record["db_id"], "form": result["form"],
                        "status": result["outcome"]["status"], "scores": scores[qid],
                        "changed": result["outcome"]["sql"] != record["selected"]})
    summary = {"n": len(records), "unique_sql_pairs": len(jobs), "statuses": export_manifest["statuses"],
               "evaluator_sha256": base.sha256(ROOT / "evaluation/evaluation.py"),
               "scoring": "Unmodified repository execute_model; passing means execution match, not semantic proof.",
               "scores": {}, "by_form": {}}
    for name in predictions:
        correct = sum(r["scores"][name] for r in details)
        summary["scores"][name] = {"correct": correct, "accuracy_percent": 100 * correct / len(details),
            "recovered_ids": [r["question_id"] for r in details if r["scores"][name] > r["scores"]["original"]],
            "regressed_ids": [r["question_id"] for r in details if r["scores"][name] < r["scores"]["original"]],
            "by_database": {db: {"n": sum(r["db_id"] == db for r in details), "correct": sum(r["scores"][name] for r in details if r["db_id"] == db)}
                            for db in sorted({r["db_id"] for r in details})}}
    for form in FORMS:
        rows = [r for r in details if r["form"] == form]
        summary["by_form"][form] = {"matched": len(rows), "rewritten": sum(r["changed"] for r in rows),
            "original_correct": sum(r["scores"]["original"] for r in rows),
            "question_form_correct": sum(r["scores"]["question_form"] for r in rows),
            "recovered_ids": [r["question_id"] for r in rows if r["scores"]["question_form"] > r["scores"]["original"]],
            "regressed_ids": [r["question_id"] for r in rows if r["scores"]["question_form"] < r["scores"]["original"]]}
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
    for form, v in summary["by_form"].items():
        print(f"  {form}: matched {v['matched']} rewritten {v['rewritten']} recovered {v['recovered_ids']} regressed {v['regressed_ids']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "export", "evaluate"):
        p = subs.add_parser(name)
        p.add_argument("--out", default="output/question_form_output/v1")
        if name == "prepare":
            p.add_argument("--source", default="output/selection_experiment/v2")
            p.add_argument("--model", default=None)
        if name in ("run", "evaluate"):
            p.add_argument("--workers", type=base.positive, default=4 if name == "run" else 2)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "export": export, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
