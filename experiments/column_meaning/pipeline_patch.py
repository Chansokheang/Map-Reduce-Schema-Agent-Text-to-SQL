"""Matched values + join columns + column meanings, in front of the schema agent.

Loaded by experiments/full_pipeline/patches/sitecustomize.py when QASQL_COLUMN_MEANING=1.
src/ is never edited, and the earlier patches are left untouched so their runs still reproduce:

  QASQL_MATCHED_CONTENTS=1   values, at generation time only          (mc_v1)
  QASQL_SCHEMA_LINKING=1     values + join columns, before the agent  (sl_v1)
  QASQL_COLUMN_MEANING=1     the above, plus the column-meaning block (this file)

Why the extra block: on the sl_v1 700-question run the largest group of moderate failures was
"right tables, wrong columns" (24 of 86), and only 2 of those had the gold column anywhere in
the value block — the choice is between similarly named columns (`Free Meal Count` vs
`FRPM Count`), which values cannot settle but the shipped documentation describes. retriever.py
ranks the documented columns against the question with BM25 plus an exact-name lookup.

The block goes to the same two places as the others: every schema worker's prompt (so table and
column scoring can use it) and every generation prompt (restricted there to the visible tables).

Environment:
  QASQL_COLUMN_MEANING=1              enable
  QASQL_COLUMN_MEANING_LIMIT=10       ranked columns per prompt (named ones are extra)
  QASQL_COLUMN_MEANING_JSON=...       column_meaning.json (default data/column_meaning.json)
  QASQL_SCHEMA_LINKING_LIMIT=15       values shown to the schema agent
  QASQL_JOIN_LIMIT=12                 join edges per prompt
  QASQL_DATABASES_DIR=...             databases the foreign keys and descriptions come from
"""
import functools
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.matched_contents.pipeline_patch import resolve_database
from experiments.matched_contents.schema_agent_patch import (
    _AppendingClient, _limit, build_block, retrieve_for_question,
)

_CURRENT = {"context": None}
_LOCK = threading.Lock()


def column_block(db_id, question, evidence, tables=None):
    """The '# Column meanings' block for one prompt. '' when nothing matched or on failure."""
    from experiments.column_meaning.retriever import retrieve, format_block
    try:
        hits = retrieve(db_id, question or "", evidence or "",
                        tables=tuple(tables) if tables else None,
                        limit=_limit("QASQL_COLUMN_MEANING_LIMIT", 10))
    except Exception:
        return ""
    return format_block(hits)


@functools.lru_cache(maxsize=4096)
def _cached_column_block(db_id, question, evidence, tables):
    return column_block(db_id, question, evidence, tables or None)


def full_block(db_id, question, evidence, hits, tables=None, focus_extra=()):
    """Values + join columns + column meanings, in that order; '' when all three are empty."""
    parts = [build_block(db_id, hits, tables=tables, focus_extra=focus_extra),
             _cached_column_block(db_id, question or "", evidence or "",
                                  tuple(sorted(str(t) for t in tables)) if tables else None)]
    return "\n\n".join(p for p in parts if p)


def _patch_manager():
    from src.agents.manager import SchemaManager
    if getattr(SchemaManager, "_qasql_column_meaning", False):
        return True
    if getattr(SchemaManager, "_qasql_schema_linking", False):
        return False                                    # QASQL_SCHEMA_LINKING owns the agent
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
    SchemaManager._qasql_column_meaning = True
    return True


def _patch_worker():
    from src.agents.worker import SchemaWorker
    if getattr(SchemaWorker, "_qasql_column_meaning", False):
        return True
    if getattr(SchemaWorker, "_qasql_schema_linking", False):
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
        if not context:
            return call()
        block = full_block(context["db_id"], context["question"], context["evidence"],
                           context["hits"], tables=context["tables"], focus_extra=(table_name,))
        if not block:
            return call()
        client = self.llm_client
        self.llm_client = _AppendingClient(client, block)
        try:
            return call()
        finally:
            self.llm_client = client

    SchemaWorker._llm_table_relevance = _llm_table_relevance
    SchemaWorker._qasql_column_meaning = True
    return True


def _patch_prompt_builder():
    from src.generation.prompt_builder import PromptBuilder
    if getattr(PromptBuilder, "_qasql_column_meaning", False):
        return True
    if getattr(PromptBuilder, "_qasql_schema_linking", False) or \
            getattr(PromptBuilder, "_qasql_matched_contents", False):
        return False
    original = PromptBuilder.build

    @functools.lru_cache(maxsize=4096)
    def block_for(db_id, question, evidence, tables):
        hits = retrieve_for_question(db_id, question, evidence, tables=tables,
                                     limit=_limit("QASQL_MATCHED_CONTENTS_LIMIT", 12))
        return full_block(db_id, question, evidence, hits, tables=list(tables) or None)

    @functools.wraps(original)
    def build(self, nl_query, schema, strategy, focused_schema=None, profile=None, evidence=None):
        prompts = original(self, nl_query, schema, strategy, focused_schema=focused_schema,
                           profile=profile, evidence=evidence)
        visible = focused_schema if focused_schema else schema
        db_id = resolve_database(schema) or resolve_database(visible)
        if db_id:
            block = block_for(db_id, nl_query or "", evidence or "",
                              tuple(sorted(str(t) for t in (visible or {}))))
            if block:
                prompts["user"] = f"{prompts['user']}\n\n{block}"
                prompts["matched_contents"] = block
        return prompts

    PromptBuilder.build = build
    PromptBuilder._qasql_column_meaning = True
    return True


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


ENABLED = os.environ.get("QASQL_COLUMN_MEANING") == "1"
