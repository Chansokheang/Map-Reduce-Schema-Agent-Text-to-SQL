"""Offline tests with a mock client. No model calls, no gold."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.judge_grouping import runner
from experiments.judge_grouping.prompts import arms


def packet(triggered=True):
    return {"question_id": 7, "triggered": triggered,
            "sql_by_id": {"C1": "SELECT a FROM t", "C2": "SELECT b FROM t", "C3": "SELECT c FROM t"},
            "payload": {"question": "q", "evidence": "e", "schema": {"t": {"ddl": "CREATE TABLE t (a)"}},
                        "current_id": "C2", "comparison_note": "note",
                        "differences": [{"left_id": "C1", "right_id": "C2"}],
                        "candidates": [
                            {"id": "C1", "sql": "SELECT a FROM t", "error": None, "row_count": 3, "columns": ["a"],
                             "distinct_row_count": 3, "null_counts": [0], "sample_rows": [[1], [2], [3], [4]], "result_group": 0},
                            {"id": "C2", "sql": "SELECT b FROM t", "error": None, "row_count": 3, "columns": ["b"],
                             "distinct_row_count": 3, "null_counts": [0], "sample_rows": [[1], [2], [3]], "result_group": 0},
                            {"id": "C3", "sql": "SELECT c FROM t", "error": "boom", "row_count": None, "columns": [],
                             "distinct_row_count": None, "null_counts": None, "sample_rows": [], "result_group": None}]}}


def test_arm_a_hides_grouping_and_arm_b_states_it():
    a = runner.payload_for(packet(), "no_grouping")
    b = runner.payload_for(packet(), "grouping")
    assert "result_groups" not in a and "differences" not in a and "result_group" not in a["candidates"][0]
    assert "distinct_row_count" not in a["candidates"][0] and "null_counts" not in a["candidates"][0]
    assert len(a["candidates"][0]["sample_rows"]) == 3          # live judge shows at most three rows
    assert b["result_groups"] == [{"candidates": ["C1", "C2"], "size": 2}]
    assert "not evidence of correctness" in b["result_groups_note"]
    assert {k: v for k, v in b.items() if k not in ("result_groups", "result_groups_note")} == a


def test_prompts_differ_only_by_the_grouping_note():
    p = arms()
    assert p["grouping"].startswith(p["no_grouping"])
    extra = p["grouping"][len(p["no_grouping"]):]
    assert "never choose a candidate because more candidates" in extra.lower()
    assert "result_groups" not in p["no_grouping"]


def test_response_schema_excludes_failed_candidates():
    schema = runner.response_schema(runner.payload_for(packet(), "grouping"))
    assert schema["properties"]["selected_id"]["enum"] == ["C1", "C2", None]


@pytest.mark.parametrize("answer,status,sql", [
    ({"selected_id": "C1", "reasoning": "r"}, "selected", "SELECT a FROM t"),
    ({"selected_id": None, "reasoning": "r"}, "retained", "SELECT b FROM t"),
    ({"selected_id": "C2", "reasoning": "r"}, "retained", "SELECT b FROM t"),
])
def test_decide(answer, status, sql):
    outcome = runner.decide(packet(), answer)
    assert outcome["status"] == status and outcome["sql"] == sql


def test_decide_rejects_unknown_id():
    with pytest.raises(ValueError, match="frozen candidate"):
        runner.decide(packet(), {"selected_id": "C9", "reasoning": "r"})


def test_process_runs_both_arms_once_and_checkpoints(tmp_path):
    out = tmp_path / "exp"
    (out / "packets").mkdir(parents=True)
    runner.checkpoint_write(out / "packets" / "7.json", packet())
    record = {"question_id": 7, "db_id": "demo", "selected": "SELECT b FROM t",
              "candidates": ["SELECT a FROM t", "SELECT c FROM t"]}
    manifest = {"model": "mock", "cli_timeout": 10, "max_prompt_chars": 180000}
    prompts = arms()
    seen = []

    def client(payload, system, folder, model, timeout):
        seen.append((system, payload))
        return {"selected_id": "C1", "reasoning": "because"}

    first = runner.process(record, manifest, prompts, out, client)
    second = runner.process(record, manifest, prompts, out, client)
    assert first == second and len(seen) == 2
    assert {s for s, _ in seen} == set(prompts.values())
    assert [p.get("result_groups") is not None for _, p in seen].count(True) == 1
    for arm in runner.ARMS:
        assert first["arms"][arm]["sql"] == "SELECT a FROM t"


def test_untriggered_question_makes_no_call(tmp_path):
    out = tmp_path / "exp"
    (out / "packets").mkdir(parents=True)
    runner.checkpoint_write(out / "packets" / "7.json", packet(triggered=False))
    record = {"question_id": 7, "db_id": "demo", "selected": "SELECT b FROM t", "candidates": []}
    calls = []
    result = runner.process(record, {"model": "m", "cli_timeout": 10, "max_prompt_chars": 1000}, arms(), out,
                            lambda *a, **k: calls.append(1))
    assert not calls
    assert all(result["arms"][arm]["status"] == "not_triggered" for arm in runner.ARMS)


def test_client_failure_keeps_the_original(tmp_path):
    out = tmp_path / "exp"
    (out / "packets").mkdir(parents=True)
    runner.checkpoint_write(out / "packets" / "7.json", packet())
    record = {"question_id": 7, "db_id": "demo", "selected": "SELECT b FROM t", "candidates": ["SELECT a FROM t"]}

    def broken(*args, **kwargs):
        raise RuntimeError("rate limited")

    result = runner.process(record, {"model": "m", "cli_timeout": 10, "max_prompt_chars": 180000}, arms(), out, broken)
    for arm in runner.ARMS:
        assert result["arms"][arm]["status"] == "failed"
        assert result["arms"][arm]["sql"] == record["selected"]
