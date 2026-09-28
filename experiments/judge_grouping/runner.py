"""Judge-grouping A/B: does telling the judge which candidates agree change its accuracy?

Self-contained: the helpers it needs from earlier experiments are assembled here, so no
existing experiment file is imported or modified. Frozen, gold-free inputs and execution
packets are copied from output/selection_experiment/v2; gold is read only by `evaluate`.

  prepare -> run -> export -> evaluate

Arm A "no_grouping" mirrors the live pipeline judge (SQL, row count, up to three sample
rows). Arm B "grouping" adds the identical-result sets and a neutrality instruction.
Both arms see the same candidates in the same order and use the production judge prompt.
"""
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.judge_grouping.prompts import arms as prompt_arms

DELIMITER = "\t----- bird -----\t"
ARMS = ("no_grouping", "grouping")
SOURCE = "output/selection_experiment/v2"


# ----------------------------------------------------------------- assembled helpers
def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest_object(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def checkpoint_write(path, value):
    write_json(path, {"sha256": digest_object(value), "data": value})


def checkpoint_read(path):
    wrapper = read_json(path)
    if digest_object(wrapper["data"]) != wrapper["sha256"]:
        raise ValueError(f"Checkpoint modified: {path}")
    return wrapper["data"]


def split_prediction(value, expected_db):
    sql, db = value.rsplit(DELIMITER, 1)
    if db != expected_db or not sql.strip():
        raise ValueError("Prediction database mismatch or empty SQL")
    return sql


def database_path(root, name):
    base = Path(root).resolve()
    path = (base / name / f"{name}.sqlite").resolve()
    if not path.is_relative_to(base) or Path(name).name != name:
        raise ValueError("Invalid database name")
    return path


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


# ----------------------------------------------------------------- payloads
def payload_for(packet, arm):
    """Live-judge view of the candidates; arm B adds the identical-result sets."""
    source = packet["payload"]
    candidates = []
    for option in source["candidates"]:
        candidates.append({"id": option["id"], "sql": option["sql"], "error": option["error"],
                           "row_count": option["row_count"], "columns": option["columns"],
                           "sample_rows": option["sample_rows"][:3]})
    payload = {"question": source["question"], "evidence": source["evidence"], "schema": source["schema"],
               "candidates": candidates, "current_id": source["current_id"]}
    if arm == "grouping":
        groups = collections.defaultdict(list)
        for option in source["candidates"]:
            if option["error"] is None:
                groups[option["result_group"]].append(option["id"])
        payload["result_groups"] = [{"candidates": sorted(ids), "size": len(ids)}
                                    for ids in sorted(groups.values(), key=lambda g: sorted(g))]
        payload["result_groups_note"] = ("Candidates listed together returned identical result sets. "
                                         "Group size is not evidence of correctness.")
    return payload


def response_schema(payload):
    allowed = [c["id"] for c in payload["candidates"] if c["error"] is None]
    return {"type": "object", "properties": {
        "selected_id": {"type": ["string", "null"], "enum": [*allowed, None]},
        "reasoning": {"type": "string", "minLength": 1}},
        "required": ["selected_id", "reasoning"], "additionalProperties": False}


def complete(payload, system, folder, model, timeout=240):
    """Claude CLI with tools and MCP disabled, run outside the project; logs every call."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    request_path = folder / "request.json"
    if request_path.exists():
        raise FileExistsError("Request already attempted; do not silently repeat it")
    schema = response_schema(payload)
    user = json.dumps(payload, ensure_ascii=False)
    write_json(request_path, {"system": system, "user": payload, "json_schema": schema, "model": model})
    args = ["claude", "-p", "--model", model, "--safe-mode", "--tools", "",
            "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
            "--no-session-persistence", "--output-format", "json", "--json-schema",
            json.dumps(schema), "--system-prompt", system]
    result = subprocess.run(args, input=user, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", cwd=tempfile.gettempdir(), timeout=timeout)
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        write_json(folder / "response.json", {"returncode": result.returncode, "stdout": result.stdout,
                                              "stderr": result.stderr})
        raise ValueError("Non-JSON CLI response")
    write_json(folder / "response.json", data)
    if result.returncode or data.get("is_error"):
        raise ValueError("CLI request failed: " + str(data.get("result", data.get("subtype")))[:250])
    answer = data.get("structured_output")
    if not isinstance(answer, dict) or "selected_id" not in answer:
        raise ValueError("Missing structured output")
    if answer["selected_id"] is not None and answer["selected_id"] not in schema["properties"]["selected_id"]["enum"]:
        raise ValueError("Selected id outside the supplied candidates")
    return answer


def decide(packet, answer):
    current = packet["payload"]["current_id"]
    chosen = answer.get("selected_id") or current
    if chosen not in packet["sql_by_id"]:
        raise ValueError("Selected id is not a frozen candidate")
    status = "selected" if chosen != current else "retained"
    return {"status": status, "selected_id": chosen, "sql": packet["sql_by_id"][chosen],
            "reasoning": answer.get("reasoning", "")}


# ----------------------------------------------------------------- stages
def prepare(args):
    out, source = Path(args.out).resolve(), Path(args.source).resolve()
    if out.exists():
        raise FileExistsError("Choose an unused experiment directory")
    old = read_json(source / "manifest.json")
    for path, expected in {**old["prediction_sources"], **old["metadata_sources"]}.items():
        if sha256(path) != expected:
            raise ValueError(f"Original source changed: {path}")
    records = read_json(source / "inputs.json")
    out.mkdir(parents=True)
    for name in ("inputs.json", "schemas.json", "original.json"):
        shutil.copyfile(source / name, out / name)
    (out / "packets").mkdir()
    triggered = 0
    for record in records:
        name = f"{record['question_id']}.json"
        packet = checkpoint_read(source / "packets" / name)
        if packet["question_id"] != record["question_id"] or set(packet["sql_by_id"].values()) != set(
                [record["selected"], *record["candidates"]]):
            raise ValueError(f"Frozen packet does not match inputs for {name}")
        triggered += packet["triggered"]
        checkpoint_write(out / "packets" / name, packet)
    write_json(out / "prompts.json", prompt_arms())
    manifest = {"version": "judge-grouping-v1", "n": len(records), "triggered": triggered,
                "source_experiment": str(source), "model": args.model or old["model"],
                "db_root": old["db_root"], "sql_timeout": old["sql_timeout"],
                "question_source": old["question_source"], "prediction_sources": old["prediction_sources"],
                "cli_timeout": old["cli_timeout"], "max_prompt_chars": old["max_prompt_chars"],
                "arms": list(ARMS),
                "protocol": "Paired judges over frozen candidates. Arm no_grouping shows the live judge's view "
                            "(SQL, row count, <=3 sample rows). Arm grouping adds identical-result sets and a "
                            "neutrality instruction. Same production judge prompt, same candidates, ID-only answers.",
                "gold_access": "Frozen gold-free inputs and packets; evaluate separately reads matching original gold",
                "judge_prompt_source_sha256": sha256(ROOT / "src/prompt/judge.py"),
                "frozen_sha256": {name: sha256(out / name) for name in
                                  ("inputs.json", "schemas.json", "original.json", "prompts.json")},
                "code_sha256": {str(p.resolve()): sha256(p) for p in
                                (Path(__file__), Path(__file__).with_name("prompts.py"))}}
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"questions": len(records), "triggered": triggered, "arms": list(ARMS)}), flush=True)


def load(out):
    manifest = read_json(out / "manifest.json")
    for name, expected in manifest["frozen_sha256"].items():
        if sha256(out / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    for path, expected in manifest["code_sha256"].items():
        if sha256(path) != expected:
            raise ValueError(f"Experiment code changed; prepare a new run: {path}")
    if sha256(ROOT / "src/prompt/judge.py") != manifest["judge_prompt_source_sha256"]:
        raise ValueError("The production judge prompt changed since preparation")
    return manifest, read_json(out / "inputs.json"), read_json(out / "prompts.json")


def process(record, manifest, prompts, out, client=complete):
    qid = record["question_id"]
    destination = out / "pairs" / f"{qid}.json"
    if destination.exists():
        return checkpoint_read(destination)
    packet = checkpoint_read(out / "packets" / f"{qid}.json")
    pair = {"question_id": qid, "triggered": packet["triggered"], "arms": {}}
    for arm in ARMS:
        folder = out / "calls" / str(qid) / arm
        checkpoint = folder / "outcome.json"
        if checkpoint.exists():
            outcome = checkpoint_read(checkpoint)
        elif not packet["triggered"]:
            outcome = {"status": "not_triggered", "selected_id": packet["payload"]["current_id"],
                       "sql": record["selected"]}
        else:
            payload = payload_for(packet, arm)
            try:
                if len(json.dumps(payload, ensure_ascii=False)) + len(prompts[arm]) > manifest["max_prompt_chars"]:
                    raise ValueError("prompt_budget_exceeded")
                outcome = decide(packet, client(payload, prompts[arm], folder, manifest["model"],
                                                manifest["cli_timeout"]))
            except Exception as exc:
                outcome = {"status": "failed", "selected_id": packet["payload"]["current_id"],
                           "sql": record["selected"], "reasoning": "Retained original after failure",
                           "error": f"{type(exc).__name__}: {exc}"}
        folder.mkdir(parents=True, exist_ok=True)
        checkpoint_write(checkpoint, outcome)
        pair["arms"][arm] = outcome
    destination.parent.mkdir(exist_ok=True)
    checkpoint_write(destination, pair)
    return pair


def run(args):
    out = Path(args.out).resolve()
    manifest, records, prompts = load(out)
    records = records[:args.limit] if args.limit else records
    lock = out / ".running"
    with lock.open("x", encoding="utf-8") as f:
        f.write("Coordinator running; remove only after it stops")
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process, r, manifest, prompts, out) for r in records]
            for count, future in enumerate(as_completed(futures), 1):
                pair = future.result()
                if pair["triggered"]:
                    states = {arm: pair["arms"][arm]["status"] for arm in ARMS}
                    print(f"{count}/{len(records)} Q{pair['question_id']}: {states}", flush=True)
        print(f"done {len(records)}", flush=True)
    finally:
        lock.unlink()


def export(args):
    out = Path(args.out).resolve()
    manifest, records, _ = load(out)
    for path, expected in manifest["prediction_sources"].items():
        if sha256(path) != expected:
            raise ValueError(f"Original predictions changed since preparation: {path}")
    predictions = {arm: {} for arm in ARMS}
    counts = {arm: collections.Counter() for arm in ARMS}
    changed = {arm: [] for arm in ARMS}
    for record in records:
        qid = record["question_id"]
        pair = checkpoint_read(out / "pairs" / f"{qid}.json")
        for arm in ARMS:
            outcome = pair["arms"][arm]
            if outcome["sql"] not in [record["selected"], *record["candidates"]]:
                raise ValueError("Outcome SQL is not an exact frozen candidate")
            if outcome["status"] != "selected" and outcome["sql"] != record["selected"]:
                raise ValueError("Fallback changed the original SQL")
            counts[arm][outcome["status"]] += 1
            predictions[arm][str(qid)] = outcome["sql"] + DELIMITER + record["db_id"]
            if outcome["sql"] != record["selected"]:
                changed[arm].append(qid)
    destination = out / "full_results"
    if destination.exists():
        raise FileExistsError("Full export already exists; it will not be overwritten")
    destination.mkdir()
    (destination / "original.json").write_bytes((out / "original.json").read_bytes())
    for arm in ARMS:
        write_json(destination / f"selected_{arm}.json", predictions[arm])
    write_json(destination / "manifest.json", {"n": manifest["n"], "statuses": {a: dict(counts[a]) for a in ARMS},
        "changed_ids": changed, "original_sources_unchanged": True,
        "sha256": {p.name: sha256(p) for p in destination.glob("*.json")}})
    print(json.dumps({"statuses": {a: dict(counts[a]) for a in ARMS},
                      "changed": {a: len(changed[a]) for a in ARMS}}))


def score_job(job):
    from evaluation.evaluation import execute_model
    return job["names"], job["question_id"], execute_model(job["sql"], job["gold"], job["db_path"],
                                                           job["question_id"], job["timeout"])["res"]


def evaluate(args):
    out = Path(args.out).resolve()
    manifest, records, _ = load(out)
    exported = out / "full_results"
    export_manifest = read_json(exported / "manifest.json")
    for name, expected in export_manifest["sha256"].items():
        if sha256(exported / name) != expected:
            raise ValueError("Export changed after completion")
    destination = exported / "evaluation"
    if destination.exists():
        raise FileExistsError("Evaluation already exists; preserve it")
    questions = Path(manifest["question_source"]["path"])
    if sha256(questions) != manifest["question_source"]["sha256"]:
        raise ValueError("Gold question source differs from frozen preparation source")
    gold = {str(e["question_id"]): e for e in read_json(questions)}
    predictions = {"original": read_json(exported / "original.json"),
                   **{arm: read_json(exported / f"selected_{arm}.json") for arm in ARMS}}
    jobs = []
    for record in records:
        qid, db = record["question_id"], record["db_id"]
        grouped = {}
        for name, values in predictions.items():
            grouped.setdefault(split_prediction(values[str(qid)], db), []).append(name)
        jobs.extend({"question_id": qid, "sql": sql, "gold": gold[str(qid)]["SQL"], "names": names,
                     "db_path": str(database_path(manifest["db_root"], db)),
                     "timeout": manifest["sql_timeout"]} for sql, names in grouped.items())
    scores = {r["question_id"]: {} for r in records}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for count, future in enumerate(as_completed([pool.submit(score_job, j) for j in jobs]), 1):
            names, qid, res = future.result()
            for name in names:
                scores[qid][name] = res
            if count % 200 == 0 or count == len(jobs):
                print(f"Evaluated {count}/{len(jobs)} unique SQL/reference pairs", flush=True)
    details = []
    for record in records:
        qid = record["question_id"]
        pair = checkpoint_read(out / "pairs" / f"{qid}.json")
        details.append({"question_id": qid, "db_id": record["db_id"], "triggered": pair["triggered"],
                        "scores": scores[qid],
                        "statuses": {arm: pair["arms"][arm]["status"] for arm in ARMS},
                        "selected_ids": {arm: pair["arms"][arm].get("selected_id") for arm in ARMS}})
    summary = {"n": len(records), "unique_sql_pairs": len(jobs), "triggered": sum(d["triggered"] for d in details),
               "statuses": export_manifest["statuses"], "evaluator_sha256": sha256(ROOT / "evaluation/evaluation.py"),
               "scoring": "Unmodified repository execute_model; passing means execution match, not semantic proof.",
               "scores": {}}
    for name in predictions:
        correct = sum(d["scores"][name] for d in details)
        summary["scores"][name] = {"correct": correct, "accuracy_percent": 100 * correct / len(details),
            "recovered_ids": [d["question_id"] for d in details if d["scores"][name] > d["scores"]["original"]],
            "regressed_ids": [d["question_id"] for d in details if d["scores"][name] < d["scores"]["original"]]}
    a, b = ARMS
    summary["arm_comparison"] = {
        "same_choice": sum(d["selected_ids"][a] == d["selected_ids"][b] for d in details if d["triggered"]),
        "different_choice": sum(d["selected_ids"][a] != d["selected_ids"][b] for d in details if d["triggered"]),
        f"{b}_better": [d["question_id"] for d in details if d["scores"][b] > d["scores"][a]],
        f"{b}_worse": [d["question_id"] for d in details if d["scores"][b] < d["scores"][a]]}
    responses = [read_json(p) for p in (out / "calls").glob("*/*/response.json")]
    summary["model_usage"] = {"logged_responses": len(responses),
        "reported_total_cost_usd": sum(r.get("total_cost_usd", 0) for r in responses
                                       if isinstance(r.get("total_cost_usd"), (int, float))),
        "models": sorted({m for r in responses for m in r.get("modelUsage", {})}),
        "note": "CLI reported list cost, not necessarily subscription billing"}
    destination.mkdir()
    write_json(destination / "summary.json", summary)
    write_json(destination / "per_question.json", details)
    for name, s in summary["scores"].items():
        print(f"{name:<12} {s['correct']}/{len(details)} = {s['accuracy_percent']:.2f}%  "
              f"recovered {len(s['recovered_ids'])}  regressed {len(s['regressed_ids'])}")
    print(json.dumps(summary["arm_comparison"], indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "export", "evaluate"):
        p = subs.add_parser(name)
        p.add_argument("--out", default="output/judge_grouping/v1")
        if name == "prepare":
            p.add_argument("--source", default=SOURCE)
            p.add_argument("--model", default=None)
        if name in ("run", "evaluate"):
            p.add_argument("--workers", type=positive, default=4 if name == "run" else 2)
        if name == "run":
            p.add_argument("--limit", type=positive)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "export": export, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
