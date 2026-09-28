"""Question-form output post-process for pipeline predictions.

Runs after src.pipeline has written selected.json. For each prediction whose question
matches an enabled wording form (experiments/question_form_output/patterns.py), one model
call applies that form's output convention (prompt.py). A rewrite is adopted only if it
changes nothing but the SELECT list / DISTINCT (and GROUP BY for count_then_list), executes,
and is not newly empty. Otherwise the pipeline's SQL is kept.

Inputs at inference: question, evidence, database files and supplied descriptions, the
pipeline's SQL and its own execution result. No gold SQL is read. The pipeline's
selected.json is never modified; results go to a separate folder.

Usage (from the repository root):
  python -m experiments.full_pipeline.postprocess --output-dir output/full_v1 \
      --questions data/bird_data/dev.json --databases-dir data/bird_data/dev_databases
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
import collections
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import jsonschema
import sqlglot
from sqlglot import exp
from experiments.question_form_output.patterns import classify, FORMS
from experiments.question_form_output.prompt import prompt_for, SCHEMA

DELIMITER = "\t----- bird -----\t"
DEFAULT_FORMS = ("rank", "count_then_list", "entity_then_attribute", "value_then_additive")
FIXED_CLAUSES = ("from_", "joins", "where", "having", "order", "limit", "with_", "with")
MAX_PROMPT_CHARS = 180000


# ---------------------------------------------------------------- small I/O helpers
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


def load_questions(path):
    """Keyed by position, matching the pipeline's selected.json keys. Gold fields are dropped."""
    return {str(i): {"question": e["question"], "evidence": e.get("evidence") or "", "db_id": e["db_id"]}
            for i, e in enumerate(read_json(path))}


# ---------------------------------------------------------------- schema and execution
def read_schema(db_path):
    """Table DDL plus the database's supplied description CSVs, as used in the tested layer."""
    db_path = Path(db_path)
    with closing(sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        ddl = conn.execute("SELECT name,sql FROM sqlite_schema WHERE type IN ('table','view') "
                           "AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
    schema = {name: {"ddl": sql} for name, sql in ddl}
    names = {name.casefold(): name for name in schema}
    for path in sorted((db_path.parent / "database_description").glob("*.csv")):
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("cp1252")
        schema.setdefault(names.get(path.stem.casefold(), path.stem), {})["supplied_description_csv"] = text
    return schema


def relevant_schema(schema, sql):
    try:
        wanted = {t.name.casefold() for t in sqlglot.parse_one(sql, read="sqlite").find_all(exp.Table)}
        chosen = {name: value for name, value in schema.items() if name.casefold() in wanted}
        return chosen or schema
    except Exception:
        return schema


def display_value(value):
    if isinstance(value, bytes):
        return {"blob_hex": value.hex()[:256], "bytes": len(value)}
    if isinstance(value, str) and len(value) > 300:
        return {"text_prefix": value[:300], "characters": len(value)}
    return value


def footprint(sql, db_path, timeout=30, max_rows=1000000):
    started = time.monotonic()
    try:
        with closing(sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            conn.execute("PRAGMA query_only=ON")
            conn.set_progress_handler(lambda: int(time.monotonic() - started > timeout), 10000)
            cursor = conn.execute(sql)
            columns = [c[0] for c in cursor.description or []]
            rows = cursor.fetchmany(max_rows + 1)
    except sqlite3.Error as exc:
        return {"error": str(exc), "columns": [], "row_count": None, "null_counts": None, "sample_rows": []}
    if len(rows) > max_rows:
        return {"error": "row_limit_exceeded", "columns": columns, "row_count": None, "null_counts": None, "sample_rows": []}
    return {"error": None, "columns": columns, "row_count": len(rows),
            "null_counts": [sum(r[i] is None for r in rows) for i in range(len(columns))],
            "sample_rows": [[display_value(v) for v in r] for r in rows[:3]]}


# ---------------------------------------------------------------- validation
def _clause(tree, key):
    value = tree.args.get(key)
    if value is None:
        return None
    if isinstance(value, list):
        return [v.sql(dialect="sqlite") for v in value]
    return value.sql(dialect="sqlite")


def validate_rewrite(original, rewritten, form):
    """Return the normalized rewrite, or raise ValueError if anything but the output changed."""
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


def _is_single_select(sql):
    try:
        return isinstance(sqlglot.parse_one(sql, read="sqlite"), exp.Select)
    except Exception:
        return False


def parse_answer(text):
    """Extract and validate the JSON answer from free text (API client)."""
    text = text.strip()
    decoder = json.JSONDecoder()
    for start in [i for i, ch in enumerate(text) if ch == "{"]:
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "change_needed" in value:
            jsonschema.validate(value, SCHEMA)
            return value
    raise ValueError("No valid JSON answer in model response")


# ---------------------------------------------------------------- model clients
class CliClient:
    """Claude Code CLI with tools and MCP disabled, run outside the project (as tested)."""

    def __init__(self, model="sonnet", timeout=240):
        self.model, self.timeout = model, timeout

    def args(self, system):
        return ["claude", "-p", "--model", self.model, "--safe-mode", "--tools", "",
                "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                "--no-session-persistence", "--output-format", "json", "--json-schema",
                json.dumps(SCHEMA), "--system-prompt", system]

    def complete(self, payload, system):
        result = subprocess.run(self.args(system), input=json.dumps(payload, ensure_ascii=False),
                                capture_output=True, text=True, encoding="utf-8", errors="replace",
                                cwd=tempfile.gettempdir(), timeout=self.timeout)
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            raise ValueError(f"Non-JSON CLI response (exit {result.returncode}): {result.stderr[:200]}")
        if result.returncode or data.get("is_error"):
            raise ValueError("CLI request failed: " + str(data.get("result", data.get("subtype")))[:250])
        answer = data.get("structured_output")
        jsonschema.validate(answer, SCHEMA)
        return answer, data


class ApiClient:
    """Any provider supported by src.utils.llm_client (text response, JSON parsed)."""

    def __init__(self, provider="anthropic", model="claude-sonnet-5"):
        from src.utils.llm_client import create_llm_client
        self.client = create_llm_client(provider=provider, model=model)
        self.model = model

    def complete(self, payload, system):
        text = self.client.complete(json.dumps(payload, ensure_ascii=False), system_prompt=system,
                                    max_tokens=2048, temperature=0)
        return parse_answer(text), {"text": text}


# ---------------------------------------------------------------- per-question processing
def process_one(key, sql, db_id, question, client, forms, databases_dir, work_dir, timeout=30):
    destination = Path(work_dir) / "results" / f"{key}.json"
    if destination.exists():
        cached = read_json(destination)
        if cached.get("original_sql") == sql and cached.get("db_id") == db_id:
            return cached  # Reuse only when the pipeline's SQL for this key is unchanged.
    form = classify(question["question"])
    result = {"key": key, "db_id": db_id, "form": form, "original_sql": sql}
    if form is None or form not in forms:
        result.update(status="not_matched" if form is None else "form_disabled", sql=sql)
    elif not _is_single_select(sql):
        # validate_rewrite would reject any rewrite of this SQL, so no model call is made.
        result.update(status="original_not_rewritable", sql=sql)
    else:
        db_path = Path(databases_dir) / db_id / f"{db_id}.sqlite"
        try:
            original_fp = footprint(sql, db_path, timeout)
            payload = {"question": question["question"], "evidence": question["evidence"],
                       "schema": relevant_schema(read_schema(db_path), sql), "sql": sql,
                       "execution": original_fp}
            system = prompt_for(form)
            if len(json.dumps(payload, ensure_ascii=False)) + len(system) > MAX_PROMPT_CHARS:
                raise ValueError("prompt_budget_exceeded")
            answer, raw = client.complete(payload, system)
            write_json(Path(work_dir) / "calls" / f"{key}.json",
                       {"form": form, "system": system, "payload": payload, "answer": answer, "raw": raw})
            result["reason"] = answer["reason"]
            if not answer["change_needed"]:
                result.update(status="retained", sql=sql)
            else:
                try:
                    new_sql = validate_rewrite(sql, answer["sql"], form)
                    new_fp = footprint(new_sql, db_path, timeout)
                    if new_fp["error"]:
                        raise ValueError(f"Rewrite failed to execute: {new_fp['error']}")
                    if new_fp["row_count"] == 0 and (original_fp["row_count"] or 0) > 0:
                        raise ValueError("Rewrite returned no rows while the original did")
                    result.update(status="rewritten", sql=new_sql)
                except ValueError as exc:
                    result.update(status="rejected", sql=sql, rejection=str(exc))
        except Exception as exc:
            result.update(status="failed", sql=sql, error=f"{type(exc).__name__}: {exc}")
    write_json(destination, result)
    return result


def postprocess(selected_path, questions_path, databases_dir, out_dir, client, forms=DEFAULT_FORMS,
                workers=4, timeout=30, log=print):
    selected_path, out_dir = Path(selected_path).resolve(), Path(out_dir).resolve()
    final_path = out_dir / "selected_postprocessed.json"
    if final_path == selected_path:
        raise ValueError("Refusing to overwrite the pipeline's selected.json")
    unknown = set(forms) - set(FORMS)
    if unknown:
        raise ValueError(f"Unknown forms: {sorted(unknown)}")
    before = sha256(selected_path)
    predictions = read_json(selected_path)
    questions = load_questions(questions_path)
    jobs = []
    for key, value in predictions.items():
        sql, db_id = value.rsplit(DELIMITER, 1)
        if key not in questions or questions[key]["db_id"] != db_id:
            raise ValueError(f"Prediction {key} does not match the questions file")
        jobs.append((key, sql, db_id))
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_one, key, sql, db_id, questions[key], client, forms,
                               databases_dir, out_dir, timeout): key for key, sql, db_id in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            r = future.result()
            results[r["key"]] = r
            if r["status"] not in ("not_matched", "form_disabled"):
                log(f"{done}/{len(jobs)} Q{r['key']} [{r['form']}]: {r['status']} {r.get('error', '')}")
    if sha256(selected_path) != before:
        raise RuntimeError("selected.json changed during post-processing")
    output = {key: results[key]["sql"] + DELIMITER + results[key]["db_id"] for key in predictions}
    write_json(final_path, output)
    counts = collections.Counter(f"{r['form']}:{r['status']}" for r in results.values())
    report = {"selected_source": str(selected_path), "selected_sha256": before,
              "questions_source": str(Path(questions_path).resolve()), "forms_enabled": list(forms),
              "client": type(client).__name__, "model": getattr(client, "model", None),
              "n": len(output), "statuses": dict(sorted(counts.items())),
              "changed_keys": [k for k in predictions if results[k]["status"] == "rewritten"],
              "output": str(final_path), "output_sha256": sha256(final_path)}
    write_json(out_dir / "report.json", report)
    log(f"Wrote {final_path} ({len(report['changed_keys'])} of {len(output)} changed)")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True, help="Pipeline output directory containing selected.json")
    parser.add_argument("--selected", default=None, help="Override path to the pipeline's selected.json")
    parser.add_argument("--questions", default="data/bird_data/dev.json", help="dev.json / test.json used by the pipeline")
    parser.add_argument("--databases-dir", default="data/bird_data/dev_databases")
    parser.add_argument("--out", default=None, help="Default: <output-dir>/question_form_postprocess")
    parser.add_argument("--forms", default=",".join(DEFAULT_FORMS),
                        help="Comma-separated forms to enable (all: " + ",".join(FORMS) + ")")
    parser.add_argument("--client", choices=("cli", "api"), default="cli")
    parser.add_argument("--provider", default="anthropic", help="Provider for --client api")
    parser.add_argument("--model", default=None, help="Default: sonnet (cli) / claude-sonnet-5 (api)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=30.0, help="SQL timeout in seconds")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    client = (CliClient(model=args.model or "sonnet") if args.client == "cli"
              else ApiClient(provider=args.provider, model=args.model or "claude-sonnet-5"))
    forms = tuple(f.strip() for f in args.forms.split(",") if f.strip())
    report = postprocess(args.selected or output_dir / "selected.json", args.questions, args.databases_dir,
                         args.out or output_dir / "question_form_postprocess", client, forms,
                         args.workers, args.timeout, log=lambda m: print(m, flush=True))
    print(json.dumps(report["statuses"], indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
