import json
import sqlite3
import pytest
from analysis.selection_experiment import execute, build_packet, parse_selection
from analysis.selection_experiment import prepare, load_experiment, process_record, export, evaluate
from analysis.selection_experiment import read_json, write_json, sha256, ARMS, CANDIDATE_FILES, DELIMITER
from analysis.selection_experiment import judge_packet, checkpoint_read, relevant_schema
from types import SimpleNamespace
from analysis.selection_experiment_client import StructuredClaudeClient


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "demo.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE items(id INTEGER, value TEXT)")
        conn.executemany("INSERT INTO items VALUES (?, ?)", [(1, None), (2, "x"), (3, "x")])
    return path


def record():
    return dict(question_id=0, question="List stored values", evidence="",
        candidates=["SELECT value FROM items", "SELECT value FROM items WHERE value IS NOT NULL"],
        selected="SELECT value FROM items WHERE value IS NOT NULL", SQL="GOLD_SECRET", correct_candidates=["SECRET"])


def test_payload_is_allowlisted_and_retains_null_disagreement(database):
    packet = build_packet(record(), {"items": {"ddl": "CREATE TABLE items(id,value)",
        "description": "Supplied description"}}, database, "seed")
    assert packet["triggered"]
    assert "SECRET" not in json.dumps(packet)
    assert packet["payload"]["differences"]
    assert any(c["null_counts"] == [1] for c in packet["payload"]["candidates"])
    assert set(packet["sql_by_id"].values()) == set(record()["candidates"])


def test_duplicates_and_row_order_do_not_trigger_but_columns_do(database):
    rec = record()
    rec["candidates"] = ["SELECT value FROM items", "SELECT DISTINCT value FROM items ORDER BY value DESC"]
    rec["selected"] = "SELECT id FROM items"  # A sixth option cannot expand this gate.
    assert not build_packet(rec, {}, database, "seed")["triggered"]
    rec["candidates"] = ["SELECT id,value FROM items", "SELECT value,id FROM items"]
    assert build_packet(rec, {}, database, "seed")["triggered"]


def test_sql_readonly_deadline_and_incomplete_result(database):
    assert execute("DELETE FROM items", database)["error"]
    assert len(execute("SELECT * FROM items", database)["rows"]) == 3
    assert execute("SELECT * FROM items", database, max_rows=1)["error"] == "row_limit_exceeded"
    query = "WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n) SELECT SUM(x) FROM n"
    assert execute(query, database, timeout=.01)["error"] == "interrupted"


def test_selection_rejects_rewrites_invalid_ids_and_allows_abstention(database):
    packet = build_packet(record(), {}, database, "seed")
    assert parse_selection('{"selected_id":null,"reasoning":"ambiguous"}', packet)["status"] == "abstained"
    for data in [{"selected_id": "C99", "reasoning": "bad"},
                 {"selected_id": "C1", "reasoning": "bad", "selected_sql": "SELECT 1"},
                 {"selected_id": ["C1"], "reasoning": "bad"}]:
        with pytest.raises(ValueError):
            parse_selection(json.dumps(data), packet)


def test_failed_execution_cannot_be_selected_or_trigger_alone(database):
    rec = record()
    rec["candidates"] = ["SELECT id FROM items", "SELECT missing FROM items"]
    packet = build_packet(rec, {}, database, "seed")
    assert not packet["triggered"]
    bad = next(c["id"] for c in packet["payload"]["candidates"] if c["error"])
    with pytest.raises(ValueError):
        parse_selection(json.dumps({"selected_id": bad, "reasoning": "bad"}), packet)


def test_relevant_schema_includes_bare_column_alternatives():
    schema = {"cards": {"supplied_description_csv": "manaCost versus convertedManaCost"}, "unused": {}}
    assert relevant_schema(schema, ["SELECT manaCost FROM cards", "SELECT convertedManaCost FROM cards"]) == {"cards": schema["cards"]}


def test_provider_failure_is_logged_as_failure_and_keeps_original(database):
    class Broken:
        def complete(self, *args, **kwargs):
            raise RuntimeError("provider unavailable")
    packet = build_packet(record(), {}, database, "seed")
    outcome = judge_packet(packet, "criteria", Broken())
    assert outcome["status"] == "failed"
    assert outcome["sql"] == record()["selected"]
    assert "provider unavailable" in outcome["error"]


def test_cli_structured_output_keeps_tools_disabled_and_limits_ids(tmp_path, monkeypatch):
    import analysis.selection_experiment_client as adapter
    def fake_run(args, **kwargs):
        assert args[args.index("--tools") + 1] == ""
        assert "--safe-mode" in args and "--strict-mcp-config" in args
        schema = json.loads(args[args.index("--json-schema") + 1])
        assert schema["properties"]["selected_id"]["enum"] == ["C1", None]
        assert schema["additionalProperties"] is False
        return SimpleNamespace(returncode=0, stdout=json.dumps({
            "structured_output": {"selected_id": "C1", "reasoning": "Supported by evidence"},
            "total_cost_usd": .01}))
    monkeypatch.setattr(adapter.subprocess, "run", fake_run)
    client = StructuredClaudeClient("model", tmp_path, 30)
    response = client.complete(json.dumps({"candidates": [
        {"id": "C1", "error": None}, {"id": "C2", "error": "bad SQL"}]}))
    assert json.loads(response)["selected_id"] == "C1"
    assert read_json(tmp_path / "call_1.response.json")["total_cost_usd"] == .01


@pytest.fixture
def experiment(tmp_path):
    root = tmp_path / "databases"
    folder = root / "demo"
    descriptions = folder / "database_description"
    descriptions.mkdir(parents=True)
    with sqlite3.connect(folder / "demo.sqlite") as conn:
        conn.execute("CREATE TABLE items(id INTEGER, value TEXT)")
        conn.executemany("INSERT INTO items VALUES (?, ?)", [(1, None), (2, "x"), (3, "x")])
    (descriptions / "items.csv").write_text("original_column_name,column_description\nvalue,Keep this original description\n", encoding="utf-8")
    questions = tmp_path / "questions.json"
    write_json(questions, [dict(question_id=i, db_id="demo", question="List stored values", evidence="",
        SQL="SELECT value FROM items", difficulty="GOLD_SENTINEL") for i in range(3)])
    original = tmp_path / "selected.json"
    wrong = "SELECT value FROM items WHERE value IS NOT NULL"
    right = "SELECT value FROM items"
    # Preserve an original which is outside the five-candidate pool for the last question.
    write_json(original, {str(i): (right if i == 2 else wrong) + DELIMITER + "demo" for i in range(3)})
    for i, name in enumerate(CANDIDATE_FILES):
        write_json(tmp_path / name, {"0": (right if i == 0 else wrong) + DELIMITER + "demo",
            "1": wrong + DELIMITER + "demo", "2": wrong + DELIMITER + "demo"})
    args = SimpleNamespace(out=str(tmp_path / "experiment"), questions=str(questions), selected=str(original),
        candidates_dir=str(tmp_path), db_root=str(root), seed="test", model="fake",
        sql_timeout=30, max_rows=1000000, workers=1)
    prepare(args)
    return args


def test_preparation_is_gold_blind_and_refuses_overwrite(experiment):
    from pathlib import Path
    out = Path(experiment.out)
    assert "GOLD_SENTINEL" not in (out / "inputs.json").read_text()
    assert all(set(r) == {"question_id", "db_id", "question", "evidence", "selected", "candidates"}
               for r in read_json(out / "inputs.json"))
    with pytest.raises(FileExistsError):
        prepare(experiment)
    with pytest.raises(ValueError, match="Finish both arms"):
        export(experiment)
    # This must fail on the missing export before trying to read any gold.
    experiment.questions = "does-not-exist-gold.json"
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        evaluate(experiment)


def test_frozen_input_mutation_is_detected(experiment):
    from pathlib import Path
    out = Path(experiment.out)
    (out / "inputs.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen input changed"):
        load_experiment(out)


def test_inference_can_load_with_question_gold_file_unavailable(experiment):
    from pathlib import Path
    questions = Path(experiment.questions)
    questions.rename(questions.with_suffix(".unavailable"))
    manifest, records, schemas = load_experiment(Path(experiment.out))
    assert len(records) == 3
    assert "Keep this original description" in json.dumps(schemas)


def test_changing_gold_does_not_change_prepared_judge_inputs(experiment):
    from pathlib import Path
    old_out = Path(experiment.out)
    entries = read_json(experiment.questions)
    for entry in entries:
        entry["SQL"] = "SELECT 'DIFFERENT GOLD'"
        entry["difficulty"] = "different"
    write_json(experiment.questions, entries[::-1])
    experiment.out = str(old_out.with_name("different_gold"))
    prepare(experiment)
    assert read_json(old_out / "inputs.json") == read_json(Path(experiment.out) / "inputs.json")


def test_end_to_end_pair_resume_export_and_official_evaluation(experiment):
    from pathlib import Path
    out = Path(experiment.out)
    original_hash = sha256(experiment.selected)
    manifest, records, schemas = load_experiment(out)
    prompts = read_json(out / "prompts.json")
    class FakeClient:
        calls = []
        def __init__(self, *args): pass
        def complete(self, prompt, system_prompt, **kwargs):
            self.calls.append(prompt)
            packet = json.loads(prompt)
            chosen = next(c["id"] for c in packet["candidates"] if c["sql"] == "SELECT value FROM items")
            return json.dumps({"selected_id": chosen, "reasoning": "Question does not exclude missing values"})
    for rec in records:
        process_record(rec, schemas["demo"], manifest, prompts, out, client_factory=FakeClient)
    assert len(FakeClient.calls) == 2  # Only one disagreement, one call per arm.
    assert FakeClient.calls[0] == FakeClient.calls[1]
    assert "GOLD_SENTINEL" not in "".join(FakeClient.calls)
    for rec in records:
        process_record(rec, schemas["demo"], manifest, prompts, out, client_factory=FakeClient)
    assert len(FakeClient.calls) == 2  # Completed pairs resume without calls.
    export(experiment)
    for arm in ARMS:
        values = read_json(out / "full_results" / f"selected_{arm}.json")
        assert list(values) == ["0", "1", "2"]
        assert values["0"] == "SELECT value FROM items" + DELIMITER + "demo"
        assert values["1"] == read_json(experiment.selected)["1"]
        assert values["2"] == read_json(experiment.selected)["2"]  # Sixth option preserved.
    assert sha256(experiment.selected) == original_hash == sha256(out / "full_results/original.json")
    with pytest.raises(FileExistsError):
        export(experiment)
    evaluate(experiment)
    report = read_json(out / "full_results/evaluation/summary.json")
    assert report["scores"]["original"]["correct"] == 1
    assert report["scores"]["disagreement"]["correct"] == 2
    assert report["scores"]["disagreement"]["recovered_ids"] == [0]
    assert not report["scores"]["disagreement"]["regressed_ids"]
    assert sha256(experiment.selected) == original_hash
