import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from analysis.test_selection_experiment import database, experiment
from analysis import focused_selection_experiment as focused
from analysis import selection_experiment as base


def alignment():
    return {"requirements": [{"id": "R1", "requirement": "List stored values", "source_quote": "List stored values"}],
            "ambiguous": False, "ambiguity_reason": "", "probes": []}


def packet():
    return {"payload": {"current_id": "C1", "candidates": [
        {"id": "C1", "error": None, "result_group": 0},
        {"id": "C2", "error": None, "result_group": 1},
        {"id": "C3", "error": None, "result_group": 2}]},
        "sql_by_id": {"C1": "original", "C2": "alternative", "C3": "other"}}


def check(verdicts, supported=True):
    return {"supported": supported, "support_reason": "Explicit request", "assessments": [
        {"candidate_id": f"C{i}", "verdict": v, "reason": "Checked SQL"} for i, v in enumerate(verdicts, 1)]}


def test_requirements_cannot_see_candidates_or_gold_and_need_actual_quotes():
    source = {"question": "List stored values", "evidence": "", "selected": "SECRET", "SQL": "GOLD"}
    payload = focused.alignment_payload(source, {})
    assert set(payload) == {"question", "evidence", "schema"}
    focused.validate_alignment(alignment(), payload)
    bad = alignment()
    bad["requirements"][0]["source_quote"] = "Invented support"
    with pytest.raises(ValueError, match="quote"):
        focused.validate_alignment(bad, payload)


@pytest.mark.parametrize("verdicts,supported", [
    (["unknown", "pass", "fail"], True), (["pass", "pass", "fail"], True),
    (["fail", "pass", "pass"], True), (["fail", "unknown", "fail"], True),
    (["fail", "pass", "fail"], False)])
def test_uncertainty_or_multiple_answers_preserve_original(verdicts, supported):
    assert focused.decide(packet(), alignment(), [check(verdicts, supported)])["sql"] == "original"


def test_unique_fully_verified_result_group_can_replace_original():
    assert focused.decide(packet(), alignment(), [check(["fail", "pass", "fail"])])["sql"] == "alternative"
    p = packet()
    p["payload"]["candidates"][2]["result_group"] = 1
    assert focused.decide(p, alignment(), [check(["fail", "pass", "pass"])])["sql"] == "alternative"


def test_ambiguity_and_duplicate_candidate_assessments():
    a = alignment()
    a.update(ambiguous=True, ambiguity_reason="Conflicting sources")
    assert focused.decide(packet(), a, [])["sql"] == "original"
    c = check(["fail", "pass", "fail"])
    c["assessments"][2]["candidate_id"] = "C2"
    with pytest.raises(ValueError, match="exactly once"):
        focused.validate_check(c, ["C1", "C2", "C3"])


def test_probe_blocks_mutation_and_marks_truncation(database):
    for sql in ["DELETE FROM items", "ATTACH DATABASE ':memory:' AS other", "PRAGMA table_info(items)",
                "SELECT 1; DELETE FROM items", "SELECT load_extension('bad')"]:
        assert focused.probe(sql, database)["error"]
    observed = focused.probe("SELECT * FROM items", database, max_rows=1)
    assert observed["truncated"] and len(observed["rows"]) == 1
    assert len(focused.probe("SELECT * FROM items", database)["rows"]) == 3
    infinite = "WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n) SELECT SUM(x) FROM n"
    assert focused.probe(infinite, database, timeout=.01)["error"]


def test_full_export_is_separate_and_checks_are_independent(experiment):
    args = SimpleNamespace(source=experiment.out, out=str(Path(experiment.out).parent / "focused"), model="fake")
    focused.prepare(args)
    out = Path(args.out)
    manifest, records, schemas = base.load_experiment(out)
    prompts = base.read_json(out / "prompts.json")
    calls = []
    def fake(payload, prompt, schema, folder, model):
        calls.append(payload)
        assert "GOLD_SENTINEL" not in json.dumps(payload)
        assert "current_id" not in payload and "question_id" not in payload
        if "requirement" not in payload:
            assert "candidates" not in payload
            return alignment()
        assert "assessments" not in payload
        return {"supported": True, "support_reason": "Explicit request", "assessments": [
            {"candidate_id": c["id"], "verdict": "fail" if "IS NOT NULL" in c["sql"] else "pass",
             "reason": "Checks stored values"} for c in payload["candidates"]]}
    args.command = "export"
    with pytest.raises(ValueError, match="Finish"):
        focused.export_or_evaluate(args)
    for r in records:
        focused.process(r, schemas[r["db_id"]], manifest, prompts, out, fake)
    assert len(calls) == 2  # Only one triggered question, one requirement.
    focused.export_or_evaluate(args)
    assert base.ARMS == ("control", "disagreement")
    assert (out / "full_results/original.json").read_bytes() == Path(experiment.selected).read_bytes()
    selected = base.read_json(out / "full_results/selected_focused.json")
    assert selected["0"].startswith("SELECT value FROM items\t")
    assert len(selected) == 3
    with pytest.raises(FileExistsError):
        focused.export_or_evaluate(args)
    for r in records:
        focused.process(r, schemas[r["db_id"]], manifest, prompts, out, fake)
    assert len(calls) == 2  # Resume does not repeat model calls.
