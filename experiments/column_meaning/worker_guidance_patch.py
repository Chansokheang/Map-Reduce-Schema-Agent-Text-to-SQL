"""Everything the column-meaning patch does, plus a column-selection instruction for the map agent.

Loaded by experiments/full_pipeline/patches/sitecustomize.py when QASQL_COLUMN_GUIDANCE=1.
A separate file from pipeline_patch.py on purpose: the cm_v1 configuration keeps reproducing
exactly, and this arm differs from it by one paragraph, appended to every schema worker prompt
after the retrieval blocks.

  QASQL_MATCHED_CONTENTS=1   values, at generation time only                    (mc_v1)
  QASQL_SCHEMA_LINKING=1     values + join columns, before the agent            (sl_v1)
  QASQL_COLUMN_MEANING=1     the above + the column-meaning block               (cm_v1)
  QASQL_COLUMN_GUIDANCE=1    the above + the column-selection instruction       (this file)

Why the map agent and not generation: the worker is what decides `relevant_columns` per table,
and it scores each table in isolation — it never sees that a similarly named column exists in
another table. The instruction tells it to use the descriptions and evidence it now has (the
`# Column meanings` block) when that happens. Generation prompts are left exactly as cm_v1 has
them, so the two arms differ in one place only.

Environment: as pipeline_patch.py, plus
  QASQL_COLUMN_GUIDANCE=1          enable
  QASQL_COLUMN_GUIDANCE_TEXT=...   override the instruction (default GUIDANCE below)
"""
import functools
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.matched_contents.pipeline_patch import resolve_database
from experiments.matched_contents.schema_agent_patch import _AppendingClient, retrieve_for_question
from experiments.column_meaning.pipeline_patch import full_block

GUIDANCE = (
    "**Column Selection:**\n"
    "- Carefully analyze column descriptions and evidences to choose the correct column when "
    "similar columns exist across tables."
)

_CURRENT = {"context": None}
_LOCK = threading.Lock()


def guidance():
    return os.environ.get("QASQL_COLUMN_GUIDANCE_TEXT") or GUIDANCE


def worker_suffix(db_id, question, evidence, hits, tables=None, focus_extra=()):
    """The retrieval blocks followed by the instruction. The instruction is always present.

    With no database resolved there is nothing to look up, so only the instruction is sent.
    """
    if not db_id:
        return guidance()
    block = full_block(db_id, question, evidence, hits, tables=tables, focus_extra=focus_extra)
    return f"{block}\n\n{guidance()}" if block else guidance()


def _patch_manager():
    from src.agents.manager import SchemaManager
    if getattr(SchemaManager, "_qasql_column_guidance", False):
        return True
    if getattr(SchemaManager, "_qasql_column_meaning", False) or \
            getattr(SchemaManager, "_qasql_schema_linking", False):
        return False                                   # another retrieval arm owns the agent
    original = SchemaManager.coordinate_workers

    @functools.wraps(original)
    def coordinate_workers(self, decomposed_query, schema, profile=None, evidence=None):
        db_id = resolve_database(schema)
        context = None
        if db_id:
            question = getattr(decomposed_query, "original_query", "") or ""
            context = {"db_id": db_id, "question": question, "evidence": evidence or "",
                       "hits": retrieve_for_question(db_id, question, evidence),
                       "tables": [str(t) for t in (schema or {})]}
        with _LOCK:
            previous = _CURRENT["context"]
            _CURRENT["context"] = context
        try:
            return original(self, decomposed_query, schema, profile=profile, evidence=evidence)
        finally:
            with _LOCK:
                _CURRENT["context"] = previous

    SchemaManager.coordinate_workers = coordinate_workers
    SchemaManager._qasql_column_guidance = True
    return True


def _patch_worker():
    from src.agents.worker import SchemaWorker
    if getattr(SchemaWorker, "_qasql_column_guidance", False):
        return True
    if getattr(SchemaWorker, "_qasql_column_meaning", False) or \
            getattr(SchemaWorker, "_qasql_schema_linking", False):
        return False
    original = SchemaWorker._llm_table_relevance

    @functools.wraps(original)
    def _llm_table_relevance(self, table_name, table_readable_name, columns, query_components,
                             original_query=None, evidence=None):
        def call():
            return original(self, table_name, table_readable_name, columns, query_components,
                            original_query=original_query, evidence=evidence)

        with _LOCK:
            context = _CURRENT["context"]
        if context:
            suffix = worker_suffix(context["db_id"], context["question"], context["evidence"],
                                   context["hits"], tables=context["tables"],
                                   focus_extra=(table_name,))
        else:
            suffix = guidance()                        # no database resolved: still instruct
        client = self.llm_client
        self.llm_client = _AppendingClient(client, suffix)
        try:
            return call()
        finally:
            self.llm_client = client

    SchemaWorker._llm_table_relevance = _llm_table_relevance
    SchemaWorker._qasql_column_guidance = True
    return True


def _patch_prompt_builder():
    """Generation prompts: identical to cm_v1 — blocks only, no instruction."""
    from experiments.column_meaning.pipeline_patch import _patch_prompt_builder as original
    return original()


def install():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        ok = _patch_manager() and _patch_worker()
    except ImportError:
        return False
    try:
        _patch_prompt_builder()
    except ImportError:
        pass
    return ok


ENABLED = os.environ.get("QASQL_COLUMN_GUIDANCE") == "1"
