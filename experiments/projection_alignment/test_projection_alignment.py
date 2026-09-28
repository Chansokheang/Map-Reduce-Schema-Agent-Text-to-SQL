"""Offline tests with a mock client. No model calls, no gold."""
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import analysis.selection_experiment as base
from experiments.projection_alignment import runner

COLUMNS = {"schools": ["CDSCode", "School", "Phone", "Ext", "Zip", "County"],
           "satscores": ["cds", "sname", "AvgScrRead"]}


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "demo" / "demo.sqlite"
    path.parent.mkdir()
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE schools (CDSCode TEXT PRIMARY KEY, School TEXT, Phone TEXT, Ext TEXT, Zip TEXT, County TEXT)")
        conn.execute("CREATE TABLE satscores (cds TEXT PRIMARY KEY, sname TEXT, AvgScrRead INTEGER)")
        conn.executemany("INSERT INTO schools VALUES (?,?,?,?,?,?)", [
            ("1", "Alpha High", "555-1", "11", "95203", "Alameda"), ("2", "Beta High", "555-2", None, "95203", "Orange")])
        conn.executemany("INSERT INTO satscores VALUES (?,?,?)", [("1", "Alpha High", 600), ("2", "Beta High", 500)])
    return path


def test_reorder_and_drop_extra_columns():
    sql = "SELECT School, Phone, Ext FROM schools WHERE Zip = '95203'"
    out = runner.apply_rewrite(sql, ["Phone", "Ext", "School"], COLUMNS)
    assert out.startswith("SELECT Phone, Ext, School FROM schools")
    out = runner.apply_rewrite("SELECT s.School, s.Phone FROM schools AS s", ["s.Phone"], COLUMNS)
    assert out == "SELECT s.Phone FROM schools AS s"


def test_add_plain_column_of_read_table_only():
    sql = "SELECT s.School FROM schools AS s INNER JOIN satscores AS t ON s.CDSCode = t.cds"
    assert "t.AvgScrRead" in runner.apply_rewrite(sql, ["s.School", "t.AvgScrRead"], COLUMNS)
    with pytest.raises(ValueError, match="alias"):
        runner.apply_rewrite(sql, ["s.School", "x.AvgScrRead"], COLUMNS)
    with pytest.raises(ValueError, match="unambiguous"):
        runner.apply_rewrite(sql, ["s.School", "s.nonexistent"], COLUMNS)


@pytest.mark.parametrize("select_list,match", [
    (["School", "(SELECT MAX(AvgScrRead) FROM satscores)"], "Subqueries"),
    (["*"], "Subqueries and \\*"),
    (["School", "COUNT(*)"], "plain columns"),
    (["School"], "identical"),
    ([], "1-6"),
])
def test_rejected_rewrites(select_list, match):
    with pytest.raises(ValueError, match=match):
        runner.apply_rewrite("SELECT School FROM schools", select_list, COLUMNS)


def test_grouped_query_allows_reorder_but_not_new_columns():
    sql = "SELECT County, COUNT(CDSCode) FROM schools GROUP BY County"
    assert runner.apply_rewrite(sql, ["COUNT(CDSCode)", "County"], COLUMNS).startswith("SELECT COUNT(CDSCode), County")
    with pytest.raises(ValueError, match="aggregated"):
        runner.apply_rewrite(sql, ["County", "COUNT(CDSCode)", "School"], COLUMNS)


def test_union_and_outside_changes_are_refused():
    with pytest.raises(ValueError, match="single outer SELECT"):
        runner.apply_rewrite("SELECT School FROM schools UNION SELECT sname FROM satscores", ["School"], COLUMNS)


def test_answer_validation_requires_real_quotes_and_slot_alignment():
    payload = {"question": "What is the phone number and extension for the school?", "evidence": ""}
    good = {"slots": [{"position": 1, "attribute": "phone", "source_quote": "phone number"},
                      {"position": 2, "attribute": "extension", "source_quote": "extension"}],
            "change_needed": True, "select_list": ["Phone", "Ext"], "reason": "ok"}
    runner.validate_answer(good, payload)
    bad = json.loads(json.dumps(good)); bad["slots"][0]["source_quote"] = "telephone"
    with pytest.raises(ValueError, match="quote"):
        runner.validate_answer(bad, payload)
    bad = json.loads(json.dumps(good)); bad["select_list"] = ["Phone"]
    with pytest.raises(ValueError, match="one expression per slot"):
        runner.validate_answer(bad, payload)


def test_decide_rejects_newly_empty_and_broken_rewrites(db):
    record = {"question_id": 1, "db_id": "demo", "question": "q", "evidence": "",
              "selected": "SELECT School, Phone FROM schools WHERE Zip = '95203'", "candidates": []}
    fp = runner.footprint(record["selected"], db, 30, 1000)
    ok = runner.decide(record, {"slots": [], "change_needed": True, "select_list": ["Phone", "School"], "reason": "r"}, COLUMNS, db, 30, 1000, fp)
    assert ok["status"] == "rewritten" and ok["sql"].startswith("SELECT Phone, School")
    kept = runner.decide(record, {"slots": [], "change_needed": False, "select_list": [], "reason": "fine"}, COLUMNS, db, 30, 1000, fp)
    assert kept["status"] == "retained" and kept["sql"] == record["selected"]
    rejected = runner.decide(record, {"slots": [], "change_needed": True, "select_list": ["COUNT(*)"], "reason": "r"}, COLUMNS, db, 30, 1000, fp)
    assert rejected["status"] == "rejected" and rejected["sql"] == record["selected"]


def test_process_with_mock_client_checkpoints_and_never_calls_twice(tmp_path, db):
    out = tmp_path / "exp"; (out / "results").mkdir(parents=True)
    record = {"question_id": 7, "db_id": "demo", "question": "List the phone number and school name for zip 95203.", "evidence": "",
              "selected": "SELECT School, Phone FROM schools WHERE Zip = '95203'", "candidates": []}
    manifest = {"db_root": str(db.parent.parent), "sql_timeout": 30, "max_rows": 1000, "model": "mock"}
    calls = []
    def client(payload, prompt, schema, folder, model):
        calls.append(payload)
        assert "SQL" not in payload and set(payload) == {"question", "evidence", "schema", "sql", "execution"}
        return {"slots": [{"position": 1, "attribute": "phone", "source_quote": "phone number"},
                          {"position": 2, "attribute": "school", "source_quote": "school name"}],
                "change_needed": True, "select_list": ["Phone", "School"], "reason": "order"}
    first = runner.process(record, {"schools": {"ddl": "CREATE TABLE schools (...)"}}, {"demo": COLUMNS}, manifest, {"align": "p"}, out, client)
    second = runner.process(record, {}, {"demo": COLUMNS}, manifest, {"align": "p"}, out, client)
    assert first == second and len(calls) == 1
    assert first["outcome"]["status"] == "rewritten" and first["outcome"]["sql"].startswith("SELECT Phone, School")
    assert base.checkpoint_read(out / "results" / "7.json") == first
