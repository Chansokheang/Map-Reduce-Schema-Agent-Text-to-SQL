"""Offline tests for the value index and retriever. No model calls, no gold."""
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.matched_contents import indexer, retriever


@pytest.fixture
def index(tmp_path):
    source = tmp_path / "demo.sqlite"
    with sqlite3.connect(source) as conn:
        conn.execute("CREATE TABLE schools (School TEXT, EdOpsName TEXT, County TEXT, Note TEXT, Rooms INTEGER)")
        conn.executemany("INSERT INTO schools VALUES (?,?,?,?,?)", [
            ("Alpha High", "Continuation School", "Los Angeles", "x" * 200, 3),
            ("Beta High", "State Special School", "Orange", "y" * 200, 4),
            ("Los Angeles County Online High", "Continuation School", "Los Angeles", "z" * 200, 5)])
    out = tmp_path / "index" / "demo.sqlite"
    indexer.build(source, out, log=lambda m: None)
    return tmp_path / "index"


def values(hits):
    return [h["value"] for h in hits]


def test_index_skips_free_text_and_non_text_columns(index):
    with sqlite3.connect(index / "demo.sqlite") as conn:
        columns = {c for (c,) in conn.execute("SELECT DISTINCT column_name FROM value")}
    assert columns == {"School", "EdOpsName", "County"}      # Note too long, Rooms not text


def test_exact_match_outranks_prefix_match(index):
    hits = retriever.retrieve("demo", "Which Los Angeles County school is listed?", index_dir=index)
    assert values(hits)[0] == "Los Angeles"


def test_plural_question_matches_singular_value(index):
    hits = retriever.retrieve("demo", "Please list the continuation schools.", index_dir=index)
    assert "Continuation School" in values(hits)


def test_trailing_punctuation_does_not_break_the_variant(index):
    assert "Continuation School" in values(retriever.retrieve("demo", "count continuation schools.", index_dir=index))


def test_quoted_phrase_is_found_and_ranked_first(index):
    hits = retriever.retrieve("demo", 'Schools of the "State Special School" kind in Orange', index_dir=index)
    assert values(hits)[0] == "State Special School"


def test_table_allow_list_and_limit(index):
    assert retriever.retrieve("demo", "Los Angeles", tables=["other"], index_dir=index) == []
    assert len(retriever.retrieve("demo", "Los Angeles Orange Continuation School", limit=2, index_dir=index)) == 2


def test_no_hits_gives_an_empty_block(index):
    assert retriever.format_block(retriever.retrieve("demo", "how many rooms are there", index_dir=index)) == ""


def test_block_lists_value_table_and_column(index):
    block = retriever.format_block(retriever.retrieve("demo", "continuation schools in Orange", index_dir=index))
    assert "# Matched contents" in block and "schools.EdOpsName" in block and "'Continuation School'" in block


def test_missing_index_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="No value index"):
        retriever.retrieve("absent", "question", index_dir=tmp_path)


@pytest.mark.parametrize("word,expected", [("schools", "school"), ("countries", "country"),
                                           ("school", "schools"), ("classes", "class")])
def test_number_variants(word, expected):
    assert expected in retriever.number_variants(word)
