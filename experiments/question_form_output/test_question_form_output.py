"""Offline tests with a mock client. No model calls, no gold."""
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import analysis.selection_experiment as base
from experiments.question_form_output import runner
from experiments.question_form_output.patterns import classify
from experiments.question_form_output.prompt import prompt_for, CONVENTIONS


@pytest.mark.parametrize("question,form", [
    ("Rank heroes published by Marvel Comics by their height in descending order.", "rank"),
    ("How many cards have frame effect as extendedart? List out the id of those cards.", "count_then_list"),
    ("Who is the champion of the Canadian Grand Prix in 2008? Indicate his finish time.", "entity_then_attribute"),
    ("What is the postal street address for the school? Indicate the school's name.", "value_then_additive"),
    ("Which set is not available? Please include the set ID in your response.", "value_then_additive"),
    ("In which city is the school and what is its lowest grade? Indicate the school name.", "value_then_additive"),
    ("List all the expenses incurred by the vice president.", "list_entity"),
    ("List the full names of superheroes with missing weight.", None),
    ("Please list the countries of the gas stations with transactions in June, 2013.", None),
    ("List all races in 2017 and the hosting country order by date of the event.", None),
    ("What is the phone number of the school?", None),
])
def test_classify(question, form):
    assert classify(question) == form


def test_prompt_has_convention_per_form():
    for form in CONVENTIONS:
        assert CONVENTIONS[form] in prompt_for(form)


def test_validator_allows_output_changes_only():
    sql = "SELECT r.name, lt.milliseconds FROM lapTimes AS lt JOIN races AS r ON lt.raceId = r.raceId ORDER BY lt.milliseconds LIMIT 1"
    new = runner.validate_rewrite(sql, "SELECT lt.milliseconds FROM lapTimes AS lt JOIN races AS r ON lt.raceId = r.raceId ORDER BY lt.milliseconds LIMIT 1", "entity_then_attribute")
    assert new.startswith("SELECT lt.milliseconds FROM")
    for bad, msg in [
        ("SELECT lt.milliseconds FROM lapTimes AS lt JOIN races AS r ON lt.raceId = r.raceId ORDER BY lt.milliseconds LIMIT 2", "limit"),
        ("SELECT lt.milliseconds FROM lapTimes AS lt JOIN races AS r ON lt.raceId = r.raceId WHERE lt.lap = 1 ORDER BY lt.milliseconds LIMIT 1", "where"),
        ("SELECT lt.milliseconds FROM lapTimes AS lt ORDER BY lt.milliseconds LIMIT 1", "joins"),
        ("SELECT * FROM lapTimes AS lt JOIN races AS r ON lt.raceId = r.raceId ORDER BY lt.milliseconds LIMIT 1", r"\*"),
        (sql, "identical"),
    ]:
        with pytest.raises(ValueError, match=msg):
            runner.validate_rewrite(sql, bad, "entity_then_attribute")


def test_validator_group_by_and_window_rules():
    sql = "SELECT COUNT(id), GROUP_CONCAT(id) FROM cards WHERE frameEffects = 'extendedart'"
    assert runner.validate_rewrite(sql, "SELECT id FROM cards WHERE frameEffects = 'extendedart'", "count_then_list")
    grouped = "SELECT c.colour, COUNT(s.id) FROM superhero AS s JOIN colour AS c ON s.eye_colour_id = c.id GROUP BY c.colour"
    ranked = "SELECT c.colour, COUNT(s.id), RANK() OVER (ORDER BY COUNT(s.id) DESC) FROM superhero AS s JOIN colour AS c ON s.eye_colour_id = c.id GROUP BY c.colour"
    assert "RANK()" in runner.validate_rewrite(grouped, ranked, "rank")
    with pytest.raises(ValueError, match="Window"):
        runner.validate_rewrite(grouped, ranked, "value_then_additive")
    with pytest.raises(ValueError, match="GROUP BY"):
        runner.validate_rewrite(grouped, "SELECT c.colour FROM superhero AS s JOIN colour AS c ON s.eye_colour_id = c.id", "entity_then_attribute")


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "demo" / "demo.sqlite"
    path.parent.mkdir()
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE expense (expense_id TEXT PRIMARY KEY, expense_description TEXT, cost REAL)")
        conn.executemany("INSERT INTO expense VALUES (?,?,?)", [("e1", "Pizza", 10.0), ("e2", "Posters", 5.0)])
    return path


def test_process_unmatched_makes_no_call_and_matched_calls_once(tmp_path, db):
    out = tmp_path / "exp"
    manifest = {"db_root": str(db.parent.parent), "sql_timeout": 30, "max_rows": 1000, "model": "mock"}
    record = {"question_id": 3, "db_id": "demo", "question": "List all the expenses.", "evidence": "",
              "selected": "SELECT expense_description FROM expense", "candidates": []}
    calls = []
    def client(payload, prompt, schema, folder, model):
        calls.append(prompt)
        assert set(payload) == {"question", "evidence", "schema", "sql", "execution"}
        return {"change_needed": True, "sql": "SELECT expense_id FROM expense", "reason": "list entity"}
    skipped = runner.process({**record, "question_id": 4}, {}, None, manifest, out, client)
    assert skipped["outcome"]["status"] == "not_matched" and not calls
    first = runner.process(record, {"expense": {"ddl": "x"}}, "list_entity", manifest, out, client)
    second = runner.process(record, {}, "list_entity", manifest, out, client)
    assert first == second and len(calls) == 1
    assert first["outcome"]["status"] == "rewritten" and first["outcome"]["sql"] == "SELECT expense_id FROM expense"
    assert base.checkpoint_read(out / "results" / "3.json") == first
