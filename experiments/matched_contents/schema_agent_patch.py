"""Run the value retriever BEFORE the schema agent, and show join columns with the hits.

Loaded by experiments/full_pipeline/patches/sitecustomize.py when QASQL_SCHEMA_LINKING=1.
src/ is never edited; the two seams are wrapped at runtime.

What changes, compared with QASQL_MATCHED_CONTENTS=1 (pipeline_patch.py, left untouched so the
mc_v1 configuration still reproduces):

  1. Retrieval moves earlier. pipeline_patch retrieves at generation time, after the schema
     agent has already pruned tables, and restricts the lookup to the tables that survived —
     so it can never tell the agent that a question's literal lives in a table it dropped.
     Here `SchemaManager.coordinate_workers` retrieves once over the WHOLE schema before any
     table is scored, and every worker sees the same block: which tables and columns actually
     hold the question's values, including tables other than the one it is scoring.
  2. Join columns come with it. The database's declared foreign keys (join_paths.py) are shown
     for the tables the values point at, so a question that needs a join can be answered with
     the real join keys instead of guessed ones. Workers also see the edges touching their own
     table, which is what the existing "0.5 = needed for a JOIN" rule asks them to judge.
  3. Generation still gets a block, now matched contents + join columns for the visible tables.

Both blocks are database lookups: values from the value index, edges from the schema. No rules
are derived from questions, and nothing here reads gold SQL.

Environment:
  QASQL_SCHEMA_LINKING=1              enable
  QASQL_SCHEMA_LINKING_LIMIT=15       values shown to the schema agent (generation keeps
                                      QASQL_MATCHED_CONTENTS_LIMIT, default 12)
  QASQL_JOIN_LIMIT=12                 join edges shown per prompt
  QASQL_MATCHED_CONTENTS_INDEX=...    value index directory
  QASQL_DATABASES_DIR=...             databases the foreign keys are read from
"""
import functools
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.matched_contents.pipeline_patch import _index_dir, resolve_database

_CURRENT = {"context": None}          # workers run serially today, in threads if re-enabled
_LOCK = threading.Lock()


def _limit(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return int(default)


def retrieve_for_question(db_id, question, evidence, tables=None, limit=None):
    """Matched values over the given tables (all of them, before pruning). [] on any failure."""
    from experiments.matched_contents.retriever import retrieve
    try:
        return retrieve(db_id, question or "", evidence or "",
                        tables=list(tables) if tables else None,
                        limit=limit or _limit("QASQL_SCHEMA_LINKING_LIMIT", 15),
                        index_dir=_index_dir())
    except Exception:
        return []


def build_block(db_id, hits, tables=None, focus_extra=()):
    """'# Matched contents' + '# Join columns' for one prompt; '' when both are empty."""
    from experiments.matched_contents.retriever import format_block as format_values
    from experiments.matched_contents.join_paths import join_columns, format_block as format_joins

    focus = {hit["table"] for hit in hits} | {t for t in focus_extra if t}
    edges = join_columns(db_id, tables=tables, focus=focus or None,
                         limit=_limit("QASQL_JOIN_LIMIT", 12))
    return "\n\n".join(part for part in (format_values(hits), format_joins(edges)) if part)


class _AppendingClient:
    """The worker's LLM client with a block appended to every prompt it is given."""

    def __init__(self, inner, suffix):
        self._inner = inner
        self._suffix = suffix

    def complete(self, prompt=None, *args, **kwargs):
        if prompt is None:
            prompt = kwargs.pop("prompt", "")
        return self._inner.complete(f"{prompt}\n\n{self._suffix}", *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _patch_manager():
    from src.agents.manager import SchemaManager
    if getattr(SchemaManager, "_qasql_schema_linking", False):
        return True
    original = SchemaManager.coordinate_workers

    @functools.wraps(original)
    def coordinate_workers(self, decomposed_query, schema, profile=None, evidence=None):
        db_id = resolve_database(schema)
        context = None
        if db_id:
            question = getattr(decomposed_query, "original_query", "") or ""
            hits = retrieve_for_question(db_id, question, evidence)
            context = {"db_id": db_id, "hits": hits, "tables": [str(t) for t in (schema or {})]}
        with _LOCK:
            previous = _CURRENT["context"]
            _CURRENT["context"] = context
        try:
            return original(self, decomposed_query, schema, profile=profile, evidence=evidence)
        finally:
            with _LOCK:
                _CURRENT["context"] = previous

    SchemaManager.coordinate_workers = coordinate_workers
    SchemaManager._qasql_schema_linking = True
    return True


def _patch_worker():
    from src.agents.worker import SchemaWorker
    if getattr(SchemaWorker, "_qasql_schema_linking", False):
        return True
    original = SchemaWorker._llm_table_relevance

    @functools.wraps(original)
    def _llm_table_relevance(self, table_name, table_readable_name, columns, query_components,
                             original_query=None, evidence=None):
        with _LOCK:
            context = _CURRENT["context"]
        if not context:
            return original(self, table_name, table_readable_name, columns, query_components,
                            original_query=original_query, evidence=evidence)
        block = build_block(context["db_id"], context["hits"], tables=context["tables"],
                            focus_extra=(table_name,))
        if not block:
            return original(self, table_name, table_readable_name, columns, query_components,
                            original_query=original_query, evidence=evidence)
        client = self.llm_client
        self.llm_client = _AppendingClient(client, block)
        try:
            return original(self, table_name, table_readable_name, columns, query_components,
                            original_query=original_query, evidence=evidence)
        finally:
            self.llm_client = client

    SchemaWorker._llm_table_relevance = _llm_table_relevance
    SchemaWorker._qasql_schema_linking = True
    return True


def _patch_prompt_builder():
    """Generation-time block: same values as before, now followed by the join columns."""
    from src.generation.prompt_builder import PromptBuilder
    if getattr(PromptBuilder, "_qasql_schema_linking", False):
        return True
    if getattr(PromptBuilder, "_qasql_matched_contents", False):
        return False                                   # QASQL_MATCHED_CONTENTS owns the prompt
    original = PromptBuilder.build

    @functools.lru_cache(maxsize=4096)
    def block_for(db_id, question, evidence, tables):
        hits = retrieve_for_question(db_id, question, evidence, tables=tables,
                                     limit=_limit("QASQL_MATCHED_CONTENTS_LIMIT", 12))
        return build_block(db_id, hits, tables=list(tables) or None)

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
    PromptBuilder._qasql_schema_linking = True
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


ENABLED = os.environ.get("QASQL_SCHEMA_LINKING") == "1"
