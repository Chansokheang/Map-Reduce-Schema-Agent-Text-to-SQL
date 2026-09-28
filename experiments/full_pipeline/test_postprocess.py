"""Offline tests: no model calls, no gold."""
import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.full_pipeline import postprocess as pp
from experiments.question_form_output.prompt import prompt_for

VALIDATOR_CASES = [
    ("SELECT r.name, lt.milliseconds FROM lapTimes AS lt JOIN races AS r ON lt.raceId = r.raceId ORDER BY lt.milliseconds LIMIT 1",
     "SELECT lt.milliseconds FROM lapTimes AS lt JOIN races AS r ON lt.raceId = r.raceId ORDER BY lt.milliseconds LIMIT 1", "entity_then_attribute"),
    ("SELECT r.name, lt.milliseconds FROM lapTimes AS lt JOIN races AS r ON lt.raceId = r.raceId ORDER BY lt.milliseconds LIMIT 1",
     "SELECT lt.milliseconds FROM lapTimes AS lt ORDER BY lt.milliseconds LIMIT 1", "entity_then_attribute"),
    ("SELECT COUNT(id), GROUP_CONCAT(id) FROM cards WHERE frameEffects = 'extendedart'",
     "SELECT id FROM cards WHERE frameEffects = 'extendedart'", "count_then_list"),
    ("SELECT c.colour, COUNT(s.id) FROM superhero AS s JOIN colour AS c ON s.eye_colour_id = c.id GROUP BY c.colour",
     "SELECT c.colour, COUNT(s.id), RANK() OVER (ORDER BY COUNT(s.id) DESC) FROM superhero AS s JOIN colour AS c ON s.eye_colour_id = c.id GROUP BY c.colour", "rank"),
    ("SELECT c.colour, COUNT(s.id) FROM superhero AS s JOIN colour AS c ON s.eye_colour_id = c.id GROUP BY c.colour",
     "SELECT c.colour, COUNT(s.id), RANK() OVER (ORDER BY COUNT(s.id) DESC) FROM superhero AS s JOIN colour AS c ON s.eye_colour_id = c.id GROUP BY c.colour", "value_then_additive"),
    ("SELECT name FROM t WHERE a = 1", "SELECT * FROM t WHERE a = 1", "list_entity"),
    ("SELECT name FROM t WHERE a = 1", "SELECT name FROM t WHERE a = 1", "list_entity"),
]


def outcome(fn, *args):
    try:
        return ("ok", fn(*args))
    except ValueError as exc:
        return ("error", str(exc))


@pytest.mark.parametrize("original,rewritten,form", VALIDATOR_CASES)
def test_validator_matches_tested_experiment(original, rewritten, form):
    runner = pytest.importorskip("experiments.question_form_output.runner")
    assert outcome(pp.validate_rewrite, original, rewritten, form) == outcome(runner.validate_rewrite, original, rewritten, form)


def test_schema_payload_matches_tested_experiment():
    frozen = ROOT / "output/question_form_output/v1/schemas.json"
    db = ROOT / "data/bird_data/dev_databases/toxicology/toxicology.sqlite"
    if not frozen.exists() or not db.exists():
        pytest.skip("frozen experiment or database not available")
    assert pp.read_schema(db) == json.loads(frozen.read_text(encoding="utf-8"))["toxicology"]


def test_parse_answer_handles_fences_and_rejects_garbage():
    text = 'Here you go:\n```json\n{"change_needed": true, "sql": "SELECT 1", "reason": "r"}\n```'
    assert pp.parse_answer(text)["sql"] == "SELECT 1"
    with pytest.raises(ValueError):
        pp.parse_answer("no json here")
    with pytest.raises(Exception):
        pp.parse_answer('{"change_needed": "yes", "sql": "", "reason": "r"}')


def test_cli_client_disables_tools_and_mcp():
    args = pp.CliClient("sonnet").args("system")
    assert args[args.index("--tools") + 1] == ""
    assert "--strict-mcp-config" in args and args[args.index("--mcp-config") + 1] == '{"mcpServers":{}}'
    assert "--json-schema" in args and "--safe-mode" in args


def test_questions_file_gold_is_not_loaded(tmp_path):
    path = tmp_path / "dev.json"
    path.write_text(json.dumps([{"question_id": 0, "db_id": "d", "question": "q", "evidence": "e", "SQL": "GOLD"}]))
    assert pp.load_questions(path) == {"0": {"question": "q", "evidence": "e", "db_id": "d"}}


class MockClient:
    model = "mock"

    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def complete(self, payload, system):
        self.calls.append((payload, system))
        assert "SQL" not in payload and set(payload) == {"question", "evidence", "schema", "sql", "execution"}
        return self.answers[payload["question"]], {"mock": True}


@pytest.fixture
def setup(tmp_path):
    dbdir = tmp_path / "dbs" / "demo"
    dbdir.mkdir(parents=True)
    with sqlite3.connect(dbdir / "demo.sqlite") as conn:
        conn.execute("CREATE TABLE schools (CDSCode TEXT PRIMARY KEY, School TEXT, Phone TEXT, Ext TEXT)")
        conn.executemany("INSERT INTO schools VALUES (?,?,?,?)", [("1", "Alpha", "555", "1"), ("2", "Beta", "556", None)])
    questions = [
        {"question_id": 0, "db_id": "demo", "question": "What is the phone number and extension of Alpha? Indicate the school's name.", "evidence": ""},
        {"question_id": 1, "db_id": "demo", "question": "What is the phone number of Beta?", "evidence": ""},
        {"question_id": 2, "db_id": "demo", "question": "List all the schools.", "evidence": ""},
        {"question_id": 3, "db_id": "demo", "question": "Which school has code 2? Please give its phone.", "evidence": ""},
    ]
    qpath = tmp_path / "dev.json"
    qpath.write_text(json.dumps(questions))
    selected = {"0": "SELECT School, Phone, Ext FROM schools WHERE School = 'Alpha'",
                "1": "SELECT Phone FROM schools WHERE School = 'Beta'",
                "2": "SELECT School FROM schools",
                "3": "SELECT School, Phone FROM schools WHERE CDSCode = '2'"}
    out = tmp_path / "pipeline_out"
    out.mkdir()
    spath = out / "selected.json"
    spath.write_text(json.dumps({k: v + pp.DELIMITER + "demo" for k, v in selected.items()}, indent=4))
    return tmp_path, qpath, spath, out


def test_end_to_end_postprocess(setup):
    tmp_path, qpath, spath, out = setup
    before = spath.read_bytes()
    client = MockClient({
        "What is the phone number and extension of Alpha? Indicate the school's name.":
            {"change_needed": True, "sql": "SELECT Phone, Ext, School FROM schools WHERE School = 'Alpha'", "reason": "order"},
        "Which school has code 2? Please give its phone.":
            {"change_needed": True, "sql": "SELECT Phone FROM schools WHERE CDSCode = '9'", "reason": "changes filter"},
    })
    report = pp.postprocess(spath, qpath, tmp_path / "dbs", out / "pp", client, workers=2, log=lambda m: None)
    assert spath.read_bytes() == before
    final = json.loads((out / "pp" / "selected_postprocessed.json").read_text(encoding="utf-8"))
    assert list(final) == ["0", "1", "2", "3"]
    assert final["0"].startswith("SELECT Phone, Ext, School FROM schools")
    assert final["1"] == json.loads(before)["1"] and final["2"] == json.loads(before)["2"]
    assert final["3"] == json.loads(before)["3"]
    assert report["statuses"] == {"None:not_matched": 1, "entity_then_attribute:rejected": 1,
                                  "list_entity:form_disabled": 1, "value_then_additive:rewritten": 1}
    assert len(client.calls) == 2
    assert client.calls[0][1] in (prompt_for("value_then_additive"), prompt_for("entity_then_attribute"))
    # Resume: a second run reuses checkpoints and makes no new calls.
    pp.postprocess(spath, qpath, tmp_path / "dbs", out / "pp", client, workers=2, log=lambda m: None)
    assert len(client.calls) == 2


def test_rerun_with_changed_pipeline_sql_ignores_stale_checkpoint(setup):
    tmp_path, qpath, spath, out = setup
    answers = {"What is the phone number and extension of Alpha? Indicate the school's name.":
               {"change_needed": False, "sql": "SELECT 1", "reason": "fine"}}
    pp.postprocess(spath, qpath, tmp_path / "dbs", out / "pp4", MockClient(answers), log=lambda m: None)
    data = json.loads(spath.read_text(encoding="utf-8"))
    data["0"] = "SELECT Phone, Ext, School FROM schools WHERE School = 'Alpha'" + pp.DELIMITER + "demo"
    data["1"] = "SELECT Phone FROM schools" + pp.DELIMITER + "demo"
    spath.write_text(json.dumps(data), encoding="utf-8")
    client = MockClient(answers)
    pp.postprocess(spath, qpath, tmp_path / "dbs", out / "pp4", client, log=lambda m: None)
    final = json.loads((out / "pp4" / "selected_postprocessed.json").read_text(encoding="utf-8"))
    assert final["0"] == data["0"] and final["1"] == data["1"]
    assert len(client.calls) == 1


def test_refuses_to_overwrite_selected(setup):
    tmp_path, qpath, spath, out = setup
    (out / "selected_postprocessed.json").write_text("{}")
    with pytest.raises(ValueError, match="overwrite"):
        pp.postprocess(out / "selected_postprocessed.json", qpath, tmp_path / "dbs", out, MockClient({}), log=lambda m: None)


def test_unparseable_pipeline_sql_makes_no_call(setup):
    tmp_path, qpath, spath, out = setup
    data = json.loads(spath.read_text(encoding="utf-8"))
    data["0"] = "-- Error: Claude Code Headless error: [WinError 206] The filename or extension is too long" + pp.DELIMITER + "demo"
    spath.write_text(json.dumps(data), encoding="utf-8")
    client = MockClient({})
    report = pp.postprocess(spath, qpath, tmp_path / "dbs", out / "pp3", client, forms=("value_then_additive",), log=lambda m: None)
    assert report["statuses"]["value_then_additive:original_not_rewritable"] == 1
    assert not client.calls


def test_client_failure_keeps_original(setup):
    tmp_path, qpath, spath, out = setup

    class Broken:
        model = "broken"

        def complete(self, payload, system):
            raise RuntimeError("rate limited")

    report = pp.postprocess(spath, qpath, tmp_path / "dbs", out / "pp2", Broken(), log=lambda m: None)
    final = json.loads((out / "pp2" / "selected_postprocessed.json").read_text(encoding="utf-8"))
    assert final == json.loads(spath.read_text(encoding="utf-8"))
    assert report["statuses"]["value_then_additive:failed"] == 1
