"""Log what the schema agent actually saw and decided, including the tables it REJECTED.

schema_agent_output.jsonl only records tables scoring >= the threshold, so when a gold table is
dropped there is no record of its score or the reason. This captures both ends:

  * `SchemaWorker._llm_table_relevance` - the exact prompt the table was judged on (so we can see
    whether column descriptions were present) and the raw model response.
  * `SchemaManager.aggregate_results` - every table relevance before the threshold filter, with
    score, reason and columns, plus which ones were cut.

src/ is untouched; both are wrapped in memory. Enable with QASQL_SCHEMA_TRACE=1 and set
QASQL_SCHEMA_TRACE_OUT to the destination .jsonl (default experiments/schema_trace/trace.jsonl).
"""
import functools
import json
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
_LOCK = threading.Lock()


def _out():
    return Path(os.environ.get("QASQL_SCHEMA_TRACE_OUT",
                               ROOT / "experiments" / "schema_trace" / "trace.jsonl"))


def _write(record):
    p = _out()
    p.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:                                   # workers run in a thread pool
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _patch_worker():
    from src.agents.worker import SchemaWorker
    if getattr(SchemaWorker, "_qasql_trace", False):
        return True
    original = SchemaWorker._llm_table_relevance

    @functools.wraps(original)
    def _llm_table_relevance(self, table_name, table_readable_name, columns,
                             query_components, original_query=None, evidence=None):
        result = original(self, table_name, table_readable_name, columns, query_components,
                          original_query=original_query, evidence=evidence)
        described = sum(1 for c in columns if c.get("description"))
        _write({
            "kind": "table_call",
            "question": original_query,
            "evidence": evidence,
            "table": table_name,
            "readable_name": table_readable_name,
            "n_columns": len(columns),
            "columns_with_description": described,
            "column_names": [c.get("name") for c in columns],
            "components": query_components,
            "llm_result": result,
        })
        return result

    SchemaWorker._llm_table_relevance = _llm_table_relevance
    SchemaWorker._qasql_trace = True
    return True


def _patch_manager():
    from src.agents.manager import SchemaManager
    if getattr(SchemaManager, "_qasql_trace", False):
        return True
    original = SchemaManager.aggregate_results

    @functools.wraps(original)
    def aggregate_results(self, verification_results, schema, relevance_threshold=0.50):
        rows = []
        for r in verification_results:
            for t in getattr(r, "table_relevances", []) or []:
                rows.append({
                    "table": getattr(t, "table_name", None),
                    "score": getattr(t, "relevance_score", None),
                    "reason": getattr(t, "reason", None),
                    "columns": getattr(t, "relevant_columns", None),
                })
        out = original(self, verification_results, schema, relevance_threshold)
        kept = {x["table_name"] for x in (out or {}).get("table_relevances", [])}
        _write({
            "kind": "aggregate",
            "threshold": relevance_threshold,
            "evaluated": len(rows),
            "kept": sorted(kept),
            "dropped": sorted(r["table"] for r in rows if r["table"] not in kept),
            "all_relevances": sorted(rows, key=lambda r: -(r["score"] or 0)),
        })
        return out

    SchemaManager.aggregate_results = aggregate_results
    SchemaManager._qasql_trace = True
    return True


def install():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        return _patch_worker() and _patch_manager()
    except ImportError:
        return False


ENABLED = os.environ.get("QASQL_SCHEMA_TRACE") == "1"
