"""Append a '# Matched contents' block to every generation prompt, without editing src/.

Loaded by experiments/full_pipeline/patches/sitecustomize.py when QASQL_MATCHED_CONTENTS=1.
It wraps src.generation.prompt_builder.PromptBuilder.build: the original prompt is
built first, then the values this question matches in the database are appended to the user
prompt. Nothing else changes, so a run with the flag differs from a run without it only by
that block.

The database is identified by matching the schema's table names against the value indexes in
output/matched_contents/index, because build does not receive a database id. One
lookup is cached per (database, question, tables), so all five strategies share it.

Environment:
  QASQL_MATCHED_CONTENTS=1          enable
  QASQL_MATCHED_CONTENTS_INDEX=...  index directory (default output/matched_contents/index)
  QASQL_MATCHED_CONTENTS_LIMIT=12   maximum values shown
"""
import functools
import os
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_TABLES = None


def _index_dir():
    return Path(os.environ.get("QASQL_MATCHED_CONTENTS_INDEX", ROOT / "output/matched_contents/index"))


def _database_tables():
    """database id -> set of indexed table names (casefolded), read once."""
    global _TABLES
    if _TABLES is None:
        _TABLES = {}
        for path in sorted(_index_dir().glob("*.sqlite")):
            try:
                with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
                    _TABLES[path.stem] = {t.casefold() for (t,) in
                                          conn.execute("SELECT DISTINCT table_name FROM value")}
            except sqlite3.Error:
                continue
    return _TABLES


def resolve_database(schema):
    """Best database for a schema dict, by table-name overlap. None when nothing matches."""
    tables = {str(t).casefold() for t in (schema or {})}
    if not tables:
        return None
    best, score = None, 0
    for db, indexed in _database_tables().items():
        overlap = len(tables & indexed)
        if overlap > score:
            best, score = db, overlap
    return best if score and score >= min(len(tables), 2) else None


@functools.lru_cache(maxsize=4096)
def _block(db_id, question, evidence, tables):
    from experiments.matched_contents.retriever import retrieve, format_block
    limit = int(os.environ.get("QASQL_MATCHED_CONTENTS_LIMIT", "12"))
    try:
        hits = retrieve(db_id, question, evidence or "", tables=list(tables) or None,
                        limit=limit, index_dir=_index_dir())
    except Exception:
        return ""
    return format_block(hits)


def install():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from src.generation.prompt_builder import PromptBuilder
    except ImportError:
        return False
    if getattr(PromptBuilder, "_qasql_matched_contents", False):
        return True
    original = PromptBuilder.build

    @functools.wraps(original)
    def build(self, nl_query, schema, strategy, focused_schema=None, profile=None, evidence=None):
        prompts = original(self, nl_query, schema, strategy, focused_schema=focused_schema,
                           profile=profile, evidence=evidence)
        visible = focused_schema if focused_schema else schema
        db_id = resolve_database(schema) or resolve_database(visible)
        if db_id:
            block = _block(db_id, nl_query or "", evidence or "",
                           tuple(sorted(str(t) for t in (visible or {}))))
            if block:
                prompts["user"] = f"{prompts['user']}\n\n{block}"
                prompts["matched_contents"] = block
        return prompts

    PromptBuilder.build = build
    PromptBuilder._qasql_matched_contents = True
    return True
