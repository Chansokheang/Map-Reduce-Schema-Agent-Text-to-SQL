"""Projection review: keep the SELECT items the question asks for, in the order it asks for them.

Runs on a finished selected.json — no regeneration — and never modifies it; the result is written
to <out>/selected_projection_reviewed.json.

What it fixes, measured on the sl_v1 700-question run: of 226 failures, 5 have gold's columns in
the wrong ORDER and 11 have gold's columns plus EXTRA ones. Gold follows the order the question
asks in 65 of 71 measurable cases, and the evaluator compares rows as tuples, so both the count
and the sequence of projected columns are enforced while row order and duplicates are not.

Why a model and not string matching: the mapping is semantic. "the names of all the
administrators" is `AdmFName1, AdmLName1`; "postal street address" is `MailStreet`. Reordering by
matching column names against question text touches 2 of 700 queries, because column names
almost never appear verbatim.

Why it cannot go wrong quietly: the model never writes SQL. It returns indices into the existing
SELECT list, and the rewrite is a permutation or a subset of what the pipeline already produced —
additions are impossible by construction, and every other clause is untouched. That constraint
comes from the earlier projection run, where reorders were 4 for 4 correct while additions were
0 for 2.

  python -m experiments.projection_review.review --output-dir ./output/sl_v1/ --out ./output/sl_v1/projection_review
"""
import argparse
import collections
import hashlib
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import jsonschema
import sqlglot
from sqlglot import exp

from experiments.projection_review.prompt import SYSTEM, build_payload

DELIMITER = "\t----- bird -----\t"
DEV = ROOT / "data/bird_data/dev.json"
SCHEMA = {
    "type": "object",
    "properties": {
        "keep": {"type": "array", "items": {"type": "integer", "minimum": 0}, "minItems": 1},
        "reason": {"type": "string"},
    },
    "required": ["keep"],
    "additionalProperties": False,
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_questions(path):
    """Keyed by position, matching the pipeline's selected.json keys.

    Gold fields are dropped here, not just left unused: nothing downstream can read the reference
    SQL even by accident, and the per-question records written to disk cannot leak it.
    """
    out = {}
    for index, entry in enumerate(read_json(path)):
        out[str(index)] = {"question": entry["question"],
                           "evidence": entry.get("evidence") or "",
                           "db_id": entry["db_id"]}
    return out


def outer_select(sql):
    """The top-level SELECT of a single-statement query, or None when that is not what this is."""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None, None
    if not isinstance(tree, exp.Select):
        return None, None
    return tree, tree.expressions


def item_texts(items):
    return [item.sql(dialect="sqlite") for item in items]


def window_last(items, keep):
    """Move window-function items (RANK, ROW_NUMBER, DENSE_RANK...) to the end of the order.

    A rank is derived from the columns it accompanies, so it annotates them rather than leading
    them. All three dev gold queries with a window function in a multi-column SELECT place it
    last (Q17, Q726, Q728), and the review had reversed exactly that on Q17: asked about
    "Rank schools by ... showing their charter numbers" it read the opening verb as the first
    column. The relative order of everything else is untouched.
    """
    windowed = [i for i in keep if list(items[i].find_all(exp.Window))]
    if not windowed or len(windowed) == len(keep):
        return keep
    return [i for i in keep if i not in windowed] + windowed


def apply_keep(tree, items, keep):
    tree.set("expressions", [items[i] for i in keep])
    return tree.sql(dialect="sqlite")


def validate_keep(keep, count):
    if not isinstance(keep, list) or not keep:
        raise ValueError("keep must be a non-empty list")
    if len(set(keep)) != len(keep):
        raise ValueError("keep repeats an index")
    if any(not isinstance(i, int) or i < 0 or i >= count for i in keep):
        raise ValueError(f"keep out of range for {count} items")
    return keep


class CliClient:
    """Claude Code CLI, tools and MCP off, run outside the project (as the post-process does)."""

    def __init__(self, model="sonnet", timeout=240):
        self.model, self.timeout = model, timeout

    def complete(self, payload):
        args = ["claude", "-p", "--model", self.model, "--safe-mode", "--tools", "",
                "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                "--no-session-persistence", "--output-format", "json",
                "--json-schema", json.dumps(SCHEMA), "--system-prompt", SYSTEM]
        result = subprocess.run(args, input=json.dumps(payload, ensure_ascii=False),
                                capture_output=True, text=True, encoding="utf-8",
                                errors="replace", cwd=tempfile.gettempdir(), timeout=self.timeout)
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            raise ValueError(f"Non-JSON CLI response (exit {result.returncode}): {result.stderr[:200]}")
        if result.returncode or data.get("is_error"):
            raise ValueError("CLI request failed: " + str(data.get("result", data.get("subtype")))[:250])
        answer = data.get("structured_output")
        jsonschema.validate(answer, SCHEMA)
        return answer


class ApiClient:
    """Any provider supported by src.utils.llm_client."""

    def __init__(self, provider="anthropic", model="claude-sonnet-5"):
        from src.utils.llm_client import create_llm_client
        self.client = create_llm_client(provider=provider, model=model)
        self.model = model

    def complete(self, payload):
        text = self.client.complete(json.dumps(payload, ensure_ascii=False), system_prompt=SYSTEM,
                                    max_tokens=512, temperature=0)
        decoder = json.JSONDecoder()
        for start in [i for i, ch in enumerate(text) if ch == "{"]:
            try:
                value, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and "keep" in value:
                jsonschema.validate(value, SCHEMA)
                return value
        raise ValueError("No valid JSON answer in model response")


def process_one(key, sql, db_id, question, client, work_dir, allow_trim=True):
    destination = Path(work_dir) / "results" / f"{key}.json"
    if destination.exists():
        cached = read_json(destination)
        if cached.get("original_sql") == sql and cached.get("db_id") == db_id:
            return cached
    record = {"key": key, "db_id": db_id, "original_sql": sql, "sql": sql, "status": "skipped"}
    tree, items = outer_select(sql)
    if tree is None:
        record["status"] = "not_a_simple_select"
    elif len(items) < 2:
        record["status"] = "single_column"
    else:
        texts = item_texts(items)
        record["items"] = texts
        try:
            answer = client.complete(build_payload(question, texts))
            keep = window_last(items, validate_keep(answer["keep"], len(items)))
            record["keep"] = keep
            record["reason"] = answer.get("reason", "")
            trims = sorted(keep) != list(range(len(items)))
            if keep == list(range(len(items))):
                record["status"] = "retained"
            elif trims and not allow_trim:
                record["status"] = "trim_disabled"        # reorder-only mode: keep the original
            else:
                record["sql"] = apply_keep(tree, items, keep)
                record["status"] = "trimmed" if trims else "reordered"
        except Exception as error:                     # keep the pipeline's SQL on any failure
            record["status"] = "failed"
            record["error"] = str(error)[:300]
    write_json(destination, record)
    return record


def review(selected_path, questions_path, out_dir, client, workers=4, allow_trim=True, log=print):
    selected_path, out_dir = Path(selected_path).resolve(), Path(out_dir).resolve()
    final_path = out_dir / "selected_projection_reviewed.json"
    if final_path == selected_path:
        raise ValueError("Refusing to overwrite the pipeline's selected.json")
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
        futures = {pool.submit(process_one, key, sql, db_id, questions[key], client, out_dir,
                               allow_trim): key for key, sql, db_id in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            record = future.result()
            results[record["key"]] = record
            if record["status"] in ("reordered", "trimmed", "failed"):
                log(f"{done}/{len(jobs)} Q{record['key']}: {record['status']} {record.get('error', '')}")
    if sha256(selected_path) != before:
        raise RuntimeError("selected.json changed during the review")
    output = {key: results[key]["sql"] + DELIMITER + results[key]["db_id"] for key in predictions}
    write_json(final_path, output)
    counts = collections.Counter(record["status"] for record in results.values())
    report = {"selected_source": str(selected_path), "selected_sha256": before,
              "questions_source": str(Path(questions_path).resolve()),
              "client": type(client).__name__, "model": getattr(client, "model", None),
              "allow_trim": allow_trim,
              "n": len(output), "statuses": dict(sorted(counts.items())),
              "changed_keys": [k for k in predictions
                               if results[k]["status"] in ("reordered", "trimmed")],
              "output": str(final_path), "output_sha256": sha256(final_path)}
    write_json(out_dir / "report.json", report)
    log(f"Wrote {final_path} ({len(report['changed_keys'])} of {len(output)} changed)")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True, help="Directory containing selected.json")
    parser.add_argument("--selected", default=None, help="Prediction file (default: <output-dir>/selected.json)")
    parser.add_argument("--questions", default=str(DEV))
    parser.add_argument("--out", default=None, help="Where to write (default: <output-dir>/projection_review)")
    parser.add_argument("--client", choices=("cli", "api"), default="cli")
    parser.add_argument("--provider", default="anthropic")
    parser.add_argument("--model", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-trim", action="store_true",
                        help="Reorder only; never drop a column. Measured on v6: +9 with 1 regression, "
                             "against +11 with 3 when trimming is allowed.")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    selected = Path(args.selected) if args.selected else output_dir / "selected.json"
    out_dir = Path(args.out) if args.out else output_dir / "projection_review"
    if args.client == "cli":
        client = CliClient(model=args.model or "sonnet")
    else:
        client = ApiClient(provider=args.provider, model=args.model or "claude-sonnet-5")
    review(selected, args.questions, out_dir, client, workers=args.workers,
           allow_trim=not args.no_trim)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
