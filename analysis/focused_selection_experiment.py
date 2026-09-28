"""Independent requirements, per-requirement verification, conservative selection.

Reuses frozen gold-free v2 inputs and execution packets. Original artifacts and
production code are never edited. Explicit evaluate is the only gold-reading step.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sqlglot
from sqlglot import exp
import jsonschema
import analysis.selection_experiment as base
from analysis.focused_selection_client import complete
from analysis.focused_selection_prompt import ALIGNMENT, VERIFY, ALIGNMENT_SCHEMA, verification_schema


def prepare(args):
    out, source = Path(args.out).resolve(), Path(args.source).resolve()
    if out.exists():
        raise FileExistsError("Choose an unused experiment directory")
    old, records, schemas = base.load_experiment(source)
    for path, expected in {**old["prediction_sources"], **old["metadata_sources"]}.items():
        if base.sha256(path) != expected:
            raise ValueError(f"Original source changed: {path}")
    out.mkdir(parents=True)
    for name in ("inputs.json", "schemas.json", "original.json"):
        shutil.copyfile(source / name, out / name)
    # Packets contain no gold, correctness labels or historical judge decisions.
    (out / "packets").mkdir()
    for record in records:
        name = f"{record['question_id']}.json"
        packet = base.packet_for(record, schemas[record["db_id"]], old, source)
        base.checkpoint_write(out / "packets" / name, packet)
    base.write_json(out / "prompts.json", {"alignment": ALIGNMENT, "verify": VERIFY})
    manifest = {**old, "version": "focused-v1", "source_experiment": str(source),
        "model": args.model or old["model"], "jsonschema_version": __import__('importlib.metadata', fromlist=['version']).version('jsonschema'),
        "protocol": "Independent full-schema requirements, up to two bounded SQL probes, one check per requirement across all candidates; conservative deterministic switch; no rewrites",
        "gold_access": "Frozen allowlisted parent inputs only; evaluate separately reads matching original gold",
        "frozen_sha256": {name: base.sha256(out / name) for name in (
            "inputs.json", "schemas.json", "original.json", "prompts.json")},
        "code_sha256": {**old["code_sha256"], **{str(path.resolve()): base.sha256(path) for path in (
            Path(__file__), ROOT / "analysis/focused_selection_prompt.py", ROOT / "analysis/focused_selection_client.py")}}}
    base.write_json(out / "manifest.json", manifest)
    print(f"Prepared {len(records)} frozen questions; no model calls", flush=True)


def alignment_payload(record, schema):
    # Deliberately excludes candidates, question ID, labels and gold.
    return {"question": record["question"], "evidence": record["evidence"], "schema": schema}


def validate_alignment(answer, payload):
    jsonschema.validate(answer, ALIGNMENT_SCHEMA)
    sources = [payload["question"], payload["evidence"]]
    for info in payload["schema"].values():
        sources.extend(v for v in info.values() if isinstance(v, str))
    for index, req in enumerate(answer["requirements"], 1):
        if req["id"] != f"R{index}":
            raise ValueError("Requirements must have unique sequential IDs")
        if not any(req["source_quote"] in source for source in sources):
            raise ValueError("Requirement quote does not occur in permitted source")
    if answer["ambiguous"] and not answer["ambiguity_reason"].strip():
        raise ValueError("Missing ambiguity explanation")


def probe(sql, path, timeout=5, max_rows=20):
    """Reject nonqueries and database attachment even on a read-only connection."""
    try:
        trees = sqlglot.parse(sql, read="sqlite")
        if len(trees) != 1 or not isinstance(trees[0], (exp.Select, exp.SetOperation)):
            raise ValueError("Only one SELECT query is permitted")
        started = time.monotonic()
        denied = {sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH, sqlite3.SQLITE_PRAGMA}
        def authorize(action, a, b, db, trigger):
            if action in denied or (action == sqlite3.SQLITE_FUNCTION and
                    str(b).lower() in {"load_extension", "readfile", "writefile"}):
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            conn.execute("PRAGMA query_only=ON")
            conn.set_authorizer(authorize)
            conn.set_progress_handler(lambda: int(time.monotonic() - started > timeout), 1000)
            cursor = conn.execute(sql)
            rows = cursor.fetchmany(max_rows + 1)
            return {"error": None, "columns": [c[0] for c in cursor.description],
                "rows": base.display_rows(rows[:max_rows]), "truncated": len(rows) > max_rows}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "rows": [], "truncated": False}


def validate_check(check, ids):
    jsonschema.validate(check, verification_schema(ids))
    received = [item["candidate_id"] for item in check["assessments"]]
    if len(received) != len(set(received)) or set(received) != set(ids):
        raise ValueError("Check must cover every candidate exactly once")


def decide(packet, alignment, checks):
    current = packet["payload"]["current_id"]
    def retain(reason):
        return {"status": "abstained", "selected_id": current,
            "sql": packet["sql_by_id"][current], "reasoning": reason}
    if alignment["ambiguous"]:
        return retain("Unresolved input ambiguity: " + alignment["ambiguity_reason"])
    if len(checks) != len(alignment["requirements"]) or not all(c["supported"] for c in checks):
        return retain("A requirement is unsupported or a check is incomplete")
    ids = [c["id"] for c in packet["payload"]["candidates"]]
    for check in checks:
        validate_check(check, ids)
    verdicts = {cid: [next(a["verdict"] for a in c["assessments"] if a["candidate_id"] == cid)
                      for c in checks] for cid in ids}
    if "fail" not in verdicts[current]:
        return retain("No supported requirement clearly fails the original")
    qualified = [c for c in packet["payload"]["candidates"] if not c["error"]
                 and all(v == "pass" for v in verdicts[c["id"]])]
    if not qualified:
        return retain("No alternative passes every requirement")
    if len({c["result_group"] for c in qualified}) != 1:
        return retain("Multiple different answers pass; selection remains unresolved")
    winner = min(qualified, key=lambda c: c["id"])["id"]
    return {"status": "selected", "selected_id": winner, "sql": packet["sql_by_id"][winner],
        "reasoning": "Original fails a supported requirement; exactly one alternative result group passes every check"}


def stage(folder, payload, prompt, schema, model, client=complete):
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "validated.json"
    if result_path.exists():
        return base.checkpoint_read(result_path)
    answer = client(payload, prompt, schema, folder, model)
    jsonschema.validate(answer, schema)
    base.checkpoint_write(result_path, answer)
    return answer


def process(record, schema, manifest, prompts, out, client=complete):
    qid = record["question_id"]
    destination = out / "pairs" / f"{qid}.json"
    if destination.exists():
        return base.checkpoint_read(destination)
    packet = base.packet_for(record, schema, manifest, out)
    pair = {"question_id": qid, "triggered": packet["triggered"], "arms": {}}
    if not packet["triggered"]:
        outcome = {"status": "not_triggered", "sql": record["selected"]}
    else:
        try:
            folder = out / "calls" / str(qid)
            payload = alignment_payload(record, schema)
            alignment = stage(folder / "alignment", payload, prompts["alignment"], ALIGNMENT_SCHEMA, manifest["model"], client)
            validate_alignment(alignment, payload)
            if alignment["ambiguous"]:
                checks = []
            else:
                probe_path = folder / "probes.json"
                if probe_path.exists():
                    observations = base.checkpoint_read(probe_path)
                else:
                    observations = [{**p, "observation": probe(p["sql"], base.database_path(manifest["db_root"], record["db_id"]))}
                                    for p in alignment["probes"]]
                    base.checkpoint_write(probe_path, observations)
                ids = [c["id"] for c in packet["payload"]["candidates"]]
                checks = []
                for requirement in alignment["requirements"]:
                    # No current_id, original label, question ID or previous check verdicts.
                    check_payload = {**payload, "requirement": requirement,
                        "candidates": [{k: v for k, v in c.items() if k != "result_group"}
                                       for c in packet["payload"]["candidates"]],
                        "differences": packet["payload"]["differences"], "database_probes": observations}
                    check = stage(folder / requirement["id"], check_payload, prompts["verify"],
                                  verification_schema(ids), manifest["model"], client)
                    validate_check(check, ids)
                    checks.append(check)
            outcome = decide(packet, alignment, checks)
        except Exception as exc:
            outcome = {"status": "failed", "sql": record["selected"],
                "reasoning": "Retained original after failed verification", "error": f"{type(exc).__name__}: {exc}"}
    pair["arms"]["focused"] = outcome
    destination.parent.mkdir(exist_ok=True)
    base.checkpoint_write(destination, pair)
    return pair


def run(args):
    out = Path(args.out).resolve()
    manifest, records, schemas = base.load_experiment(out)
    prompts = base.read_json(out / "prompts.json")
    records = records[:args.limit] if args.limit else records
    lock = out / ".running"
    with lock.open("x", encoding="utf-8") as f:
        f.write("Coordinator running; remove only after it stops")
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process, r, schemas[r["db_id"]], manifest, prompts, out) for r in records]
            for count, future in enumerate(as_completed(futures), 1):
                pair = future.result()
                outcome = pair["arms"]["focused"]
                print(f"{count}/{len(records)} Q{pair['question_id']}: {outcome['status']} "
                      + outcome.get("error", ""), flush=True)
    finally:
        lock.unlink()


def export_or_evaluate(args):
    # Reuse tested full-set export/scoring, temporarily configuring its arm list
    # in this process only. No parent experiment or source file is modified.
    previous = base.ARMS
    try:
        base.ARMS = ("focused",)
        if args.command == "export":
            base.export(args)
        else:
            base.evaluate(args)
            summary_path = Path(args.out) / "full_results/evaluation/summary.json"
            summary = base.read_json(summary_path)
            responses = [base.read_json(p) for p in (Path(args.out) / "calls").glob("*/*/response.json")]
            summary["model_usage"] = {"logged_responses": len(responses),
                "reported_total_cost_usd": sum(r.get("total_cost_usd", 0) for r in responses),
                "models": sorted({m for r in responses for m in r.get("modelUsage", {})}),
                "note": "CLI reported list cost, not necessarily subscription billing"}
            base.write_json(summary_path, summary)
    finally:
        base.ARMS = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "export", "evaluate"):
        p = subs.add_parser(name)
        p.add_argument("--out", default="output/focused_selection/v1")
        if name == "prepare":
            p.add_argument("--source", default="output/selection_experiment/v2")
            p.add_argument("--model", default=None)
        if name in ("run", "evaluate"):
            p.add_argument("--workers", type=base.positive, default=4 if name == "run" else 2)
        if name == "run":
            p.add_argument("--limit", type=base.positive)
        if name == "evaluate":
            p.add_argument("--questions", default=None)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "export": export_or_evaluate, "evaluate": export_or_evaluate}[args.command](args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
