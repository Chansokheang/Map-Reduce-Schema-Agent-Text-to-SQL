"""Gold-blind, paired replay of the saved judge choice through two fixers.

Only two fixer instructions change. Source prompts, BIRD metadata, saved
candidates and production code are never modified. Gold SQL is loaded only
by the separate evaluate command after both arms finish.
"""
from copy import deepcopy
from contextlib import closing
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import asdict
import hashlib
import io
import math
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.prompt.fixer import FIXER_PROMPT
from src.selection.fixer import SQLFixer
from src.utils.llm_client import AnthropicExhaustedError

NULL_RULE = ("- **NULL handling**: NULL values alone are not a reason to refine. "
    "Add IS NOT NULL only when a condition or requirement in the question or "
    "evidence requires excluding those records; otherwise preserve them.")
DUPLICATE_RULE = ("- **Duplicates**: Duplicate output rows alone are not a reason "
    "to refine. Add DISTINCT only when the question or evidence requires unique "
    "results or when the query otherwise violates the requested entity or "
    "aggregation semantics; otherwise preserve the existing SQL.")


def prompt_arms():
    control = deepcopy(FIXER_PROMPT)
    treatment = deepcopy(FIXER_PROMPT)
    lines = treatment["system"].splitlines()
    replacements = 0
    for i, line in enumerate(lines):
        if line.startswith('- **Duplicates (MANDATORY,'):
            lines[i] = DUPLICATE_RULE
            replacements += 1
        elif line.startswith('- **NULL (MANDATORY when present)**:'):
            lines[i] = NULL_RULE
            replacements += 1
    if replacements != 2:
        raise ValueError("Source prompt changed: expected exactly two target instructions")
    treatment["system"] = "\n".join(lines)
    return {"control": control, "conditional": treatment}


def inference_question(entry):
    return {k: entry[k] for k in ("question_id", "db_id", "question", "evidence")}


def sample_questions(entries, per_database, seed):
    databases = sorted({e["db_id"] for e in entries})
    selected = []
    for database in databases:
        group = [inference_question(e) for e in entries if e["db_id"] == database]
        group.sort(key=lambda e: hashlib.sha256(
            f"{seed}:{database}:{e['question_id']}".encode()).hexdigest())
        selected.extend(group[:per_database])
    return sorted(selected, key=lambda e: e["question_id"])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def execute_readonly(sql, db_path, timeout=30):
    started = time.monotonic()
    with closing(sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        conn.execute("PRAGMA query_only=ON")
        conn.set_progress_handler(lambda: int(time.monotonic() - started > timeout), 10000)
        try:
            cursor = conn.execute(sql)
            return True, cursor.fetchall(), None
        except sqlite3.Error as exc:
            return False, [], str(exc)


def bird_schema(database, db_root):
    """Format existing DDL and supplied CSV descriptions without generating prose."""
    folder = Path(db_root) / database
    db_path = folder / (database + ".sqlite")
    with closing(sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        ddl = conn.execute("SELECT name, sql FROM sqlite_schema WHERE type='table' "
                           "AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
    pieces = [sql for _, sql in ddl]
    sources = {}
    for path in sorted((folder / "database_description").glob("*.csv")):
        raw = path.read_bytes()
        try:
            decoded = raw.decode("utf-8-sig")
            encoding = "utf-8-sig"
        except UnicodeDecodeError:
            decoded = raw.decode("cp1252")
            encoding = "cp1252"
        rows = list(csv.DictReader(io.StringIO(decoded)))
        pieces.append(f"BIRD supplied descriptions for {path.stem}:\n" +
                      json.dumps(rows, ensure_ascii=False))
        sources[str(path.resolve())] = {"sha256":sha256(path), "encoding":encoding}
    if not sources:
        raise ValueError(f"Missing supplied BIRD descriptions for {database}")
    return "\n\n".join(pieces), sources


def prepare(args):
    out = Path(args.out)
    if (out / "manifest.json").exists():
        raise ValueError("Manifest already exists; use run to resume the frozen experiment")
    out.mkdir(parents=True, exist_ok=True)
    entries = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    chosen = sample_questions(entries, args.per_database, args.seed)
    predictions = json.loads(Path(args.selected).read_text(encoding="utf-8"))
    records, metadata_sources = [], {}
    for db in sorted({e["db_id"] for e in chosen}):
        schema, sources = bird_schema(db, args.db_root)
        metadata_sources.update(sources)
        (out / f"{db}.schema.txt").write_text(schema, encoding="utf-8")
    for entry in chosen:
        sql, db = predictions[str(entry["question_id"])].rsplit("\t----- bird -----\t", 1)
        if db != entry["db_id"]:
            raise ValueError(f"Database mismatch for {entry['question_id']}")
        records.append({**entry, "sql":sql.strip()})
    write_json(out / "inputs.json", records)
    arms = prompt_arms()
    for name, prompt in arms.items():
        write_json(out / f"{name}.prompt.json", prompt)
    paths = [Path(args.questions), Path(args.selected), ROOT / "src/prompt/fixer.py",
             ROOT / "src/selection/fixer.py", Path(__file__)]
    generated_paths = [out / "inputs.json", *out.glob("*.schema.txt"), *out.glob("*.prompt.json")]
    manifest = {
        "design":"Paired fixer-only replay; saved judge choice frozen; exactly two prompt instructions differ",
        "model":args.model, "seed":args.seed, "per_database":args.per_database,
        "question_ids":[r["question_id"] for r in records], "n":len(records),
        "arms":list(arms), "max_iterations":3, "sql_timeout_seconds":30,
        "cli_timeout_seconds":180, "workers":args.workers,
        "db_root":str(Path(args.db_root).resolve()), "sqlite_version":sqlite3.sqlite_version,
        "source_sha256":{str(p.resolve()):sha256(p) for p in paths},
        "frozen_sha256":{str(p.resolve()):sha256(p) for p in generated_paths},
        "metadata_sources":metadata_sources,
        "gold_access":"Questions sanitized at preparation; run does not open gold; evaluate opens gold only after all pairs finish",
        "limitations":["Pilot stratified equally by database, not proportional to dev size",
                       "CLI does not expose a fixed random seed or temperature",
                       "Previously generated SQL and historical repairs held fixed in both arms"],
    }
    write_json(out / "manifest.json", manifest)
    print(f"Frozen {len(records)} questions across {len(set(r['db_id'] for r in records))} databases.")


class ReadOnlyFixer(SQLFixer):
    def _execute_sql(self, sql, db_path):
        return execute_readonly(sql, db_path, self.query_timeout)


class LoggedClaudeClient:
    def __init__(self, model, log_dir, timeout):
        self.model, self.log_dir, self.timeout = model, Path(log_dir), timeout
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.calls = 0

    def complete(self, prompt, system_prompt=None, max_tokens=1024, temperature=0):
        self.calls += 1
        prefix = self.log_dir / f"call_{self.calls}"
        write_json(prefix.with_suffix(".request.json"),
                   {"system":system_prompt, "user":prompt, "model":self.model})
        args = ["claude", "-p", "--model", self.model, "--safe-mode", "--tools", "",
                "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                "--no-session-persistence", "--output-format", "json",
                "--system-prompt", system_prompt or ""]
        try:
            process = subprocess.run(args, input=prompt, capture_output=True, text=True,
                encoding="utf-8", errors="replace", cwd=tempfile.gettempdir(), timeout=self.timeout)
            data = json.loads(process.stdout)
            write_json(prefix.with_suffix(".response.json"), data)
            if process.returncode or data.get("is_error"):
                raise ValueError(f"Claude error: {str(data.get('result', data.get('subtype')))[:300]}")
            return data["result"]
        except Exception as exc:
            # Fail the experimental arm instead of counting the fixer's silent
            # fallback after provider errors as a successful model decision.
            raise AnthropicExhaustedError(f"Inference failed: {type(exc).__name__}: {exc}") from exc


def run_pair(record, manifest, out):
    qid, db = record["question_id"], record["db_id"]
    destination = out / "pairs" / f"{qid}.json"
    destination.parent.mkdir(exist_ok=True)
    if destination.exists():
        return qid, "resumed"
    db_path = Path(manifest["db_root"]) / db / (db + ".sqlite")
    ok, rows, error = execute_readonly(record["sql"], db_path, manifest["sql_timeout_seconds"])
    if not ok:
        write_json(destination, {"question_id":qid, "status":"input_execution_error", "error":error})
        return qid, "input_execution_error"
    schema = (out / f"{db}.schema.txt").read_text(encoding="utf-8")
    pair = {"question_id":qid, "db_id":db, "status":"complete", "input_sql":record["sql"],
            "input_rows":len(rows), "input_nulls":SQLFixer._has_null_values(rows),
            "input_duplicates":SQLFixer._has_duplicate_rows(rows), "arms":{}}
    order = ["control", "conditional"] if qid % 2 == 0 else ["conditional", "control"]
    pair["arm_order"] = order
    for arm in order:
        checkpoint = out / "calls" / str(qid) / arm / "outcome.json"
        if checkpoint.exists():
            pair["arms"][arm] = json.loads(checkpoint.read_text(encoding="utf-8"))
            continue
        client = LoggedClaudeClient(manifest["model"], out / "calls" / str(qid) / arm,
                                    manifest["cli_timeout_seconds"])
        fixer = ReadOnlyFixer(client, manifest["max_iterations"], manifest["sql_timeout_seconds"])
        fixer.prompt_config = json.loads((out / f"{arm}.prompt.json").read_text(encoding="utf-8"))
        er = SimpleNamespace(sql=record["sql"], result=rows, row_count=len(rows))
        outcome = fixer.fix(SimpleNamespace(candidate_id=1), er, record["question"],
                            record["evidence"], db_path, schema)
        details = asdict(outcome)
        details.pop("final_rows")
        details["model_calls"] = client.calls
        if any(issue.startswith("fixer_error:") for issue in outcome.issues):
            raise ValueError(f"Fixer parse/review failure for Q{qid} / {arm}: {outcome.issues}")
        pair["arms"][arm] = details
        write_json(checkpoint, details)
    write_json(destination, pair)
    return qid, "complete"


def run(args):
    out = Path(args.out)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    for path, expected in manifest["frozen_sha256"].items():
        if sha256(path) != expected:
            raise ValueError(f"Frozen experiment input changed: {path}")
    for path, expected in manifest["source_sha256"].items():
        if Path(path).suffix == ".py" and sha256(path) != expected:
            raise ValueError(f"Frozen experiment code changed: {path}")
    records = json.loads((out / "inputs.json").read_text(encoding="utf-8"))
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=manifest["workers"]) as pool:
        futures = [pool.submit(run_pair, record, manifest, out) for record in records]
        for n, future in enumerate(as_completed(futures), 1):
            try:
                qid, status = future.result()
            except Exception:
                for pending in futures:
                    pending.cancel()
                raise
            print(f"{n}/{len(records)} Q{qid}: {status}; elapsed {time.monotonic()-started:.0f}s", flush=True)


def evaluate(args):
    out = Path(args.out)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    inputs = json.loads((out / "inputs.json").read_text(encoding="utf-8"))
    if any(not (out / "pairs" / f"{r['question_id']}.json").exists() for r in inputs):
        raise ValueError("Complete both inference arms before opening gold labels")
    gold_path = str(Path(args.questions).resolve())
    if sha256(gold_path) != manifest["source_sha256"].get(gold_path):
        raise ValueError("Gold file differs from the frozen experiment source")
    gold = {r["question_id"]:r for r in json.loads(Path(args.questions).read_text(encoding="utf-8"))}
    def score(record):
        qid, db = record["question_id"], record["db_id"]
        pair = json.loads((out / "pairs" / f"{qid}.json").read_text(encoding="utf-8"))
        db_path = Path(manifest["db_root"]) / db / (db + ".sqlite")
        gok, grows, gerror = execute_readonly(gold[qid]["SQL"], db_path, 30)
        result = {"question_id":qid, "db_id":db, "gold_error":gerror,
                  "pair_status":pair["status"], "input_nulls":pair.get("input_nulls"),
                  "input_duplicates":pair.get("input_duplicates"), "scores":{}, "errors":{}}
        sqls = {"input":record["sql"]}
        sqls.update({arm:pair.get("arms", {}).get(arm, {}).get("final_sql", record["sql"])
                     for arm in manifest["arms"]})
        cache = {}
        for name, sql in sqls.items():
            if sql not in cache:
                cache[sql] = execute_readonly(sql, db_path, 30)
            ok, rows, error = cache[sql]
            result["scores"][name] = int(gok and ok and set(rows) == set(grows))
            result["errors"][name] = error
        return result
    with ThreadPoolExecutor(max_workers=2) as pool:
        details = list(pool.map(score, inputs))
    write_json(out / "evaluation.json", details)
    recovered = [r["question_id"] for r in details if r["scores"]["conditional"] > r["scores"]["control"]]
    regressed = [r["question_id"] for r in details if r["scores"]["conditional"] < r["scores"]["control"]]
    n = len(details)
    discordant = len(recovered) + len(regressed)
    pvalue = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(len(recovered),len(regressed))+1))
                 / 2**discordant) if discordant else 1.
    summary = {"n":n, "model":manifest["model"], "correct":{
        name:sum(r["scores"][name] for r in details) for name in ["input", *manifest["arms"]]},
        "recovered_vs_control":recovered, "regressed_vs_control":regressed,
        "net_questions":len(recovered)-len(regressed), "exact_mcnemar_p":pvalue,
        "by_database":{}, "input_execution_errors":[r['question_id'] for r in details if r['pair_status']!='complete'],
        "gold_execution_errors":[r['question_id'] for r in details if r['gold_error']],
        "null_trigger_count":sum(bool(r["input_nulls"]) for r in details),
        "duplicate_trigger_count":sum(bool(r["input_duplicates"]) for r in details),
    }
    for db in sorted({r['db_id'] for r in details}):
        subset = [r for r in details if r['db_id']==db]
        summary['by_database'][db] = {'n':len(subset), **{
            arm:sum(r['scores'][arm] for r in subset) for arm in ['input', *manifest['arms']]}}
    for arm in manifest['arms']:
        summary[arm+'_vs_input'] = {
            'recovered':[r['question_id'] for r in details if r['scores'][arm]>r['scores']['input']],
            'regressed':[r['question_id'] for r in details if r['scores'][arm]<r['scores']['input']]}
    responses = [json.loads(path.read_text(encoding='utf-8')) for path in out.glob('calls/*/*/*.response.json')]
    summary['llm_calls'] = len(responses)
    summary['usage'] = {k:sum(r.get('usage',{}).get(k,0) for r in responses)
                        for k in ['input_tokens','output_tokens','cache_read_input_tokens','cache_creation_input_tokens']}
    write_json(out / "summary.json", summary)
    lines = ["**NULL/duplicate fixer ablation — paired pilot**", "",
        f"Model: {manifest['model']}. {n} questions, {manifest['per_database']} per database, selected by a frozen hash seed without gold labels.",
        "Only two fixer instructions changed. The saved judge choice, BIRD schema/descriptions, and all other rules were fixed.",
        "", "| Arm | Correct | EX |", "|---|---:|---:|"]
    lines += [f"| {arm} | {correct}/{n} | {correct/n*100:.2f}% |" for arm,correct in summary['correct'].items()]
    lines += ["", f"Conditional vs control: **{len(recovered)} recovered, {len(regressed)} regressed, net {len(recovered)-len(regressed):+d}** "
        f"({(len(recovered)-len(regressed))/n*100:+.2f} percentage points). Exact paired McNemar p={pvalue:.4f}.",
        f"Recovered IDs: {recovered}. Regressed IDs: {regressed}.", "",
        f"Input results triggered NULL presence in {summary['null_trigger_count']} cases and duplicates in {summary['duplicate_trigger_count']} cases.",
        f"Input execution errors: {summary['input_execution_errors']}; gold execution errors: {summary['gold_execution_errors']}.",
        "", "| Database | Input | Control | Conditional |", "|---|---:|---:|---:|"]
    lines += [f"| {db} | {v['input']}/{v['n']} | {v['control']}/{v['n']} | {v['conditional']}/{v['n']} |"
              for db,v in summary['by_database'].items()]
    lines += ["", "This is a database-balanced pilot, not full-dev EX or a private-test estimate. "
              "The two arms use independent model calls; the CLI does not offer a fixed sampling seed. "
              "Differences can include model variability and indirect effects of the remaining rules. "
              "Historical candidate-generation and execution-repair decisions were not rerun.",
              "", "Artifacts: manifest.json, frozen inputs and prompts, pairs/, complete model request/response logs under calls/, evaluation.json, summary.json."]
    (out / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "evaluate"])
    parser.add_argument("--out", default="output/null_duplicate_ablation/pilot")
    parser.add_argument("--questions", default="data/bird_data/dev.json")
    parser.add_argument("--selected", default="output/claude_headless_v6/selected.json")
    parser.add_argument("--db-root", default="data/bird_data/dev_databases")
    parser.add_argument("--per-database", type=int, default=16)
    parser.add_argument("--seed", default="null-duplicate-pilot-2026-09-08-v1")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    {"prepare":prepare, "run":run, "evaluate":evaluate}[args.command](args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
