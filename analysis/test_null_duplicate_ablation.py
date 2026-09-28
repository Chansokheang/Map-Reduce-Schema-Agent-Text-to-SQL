import copy
import json
import sqlite3
import pytest
from types import SimpleNamespace
from analysis.null_duplicate_ablation import prompt_arms, inference_question, sample_questions
from analysis.null_duplicate_ablation import execute_readonly, ReadOnlyFixer, bird_schema
from analysis.null_duplicate_ablation import evaluate, write_json, sha256
from analysis.verify_null_duplicate_ablation import check
from src.prompt.fixer import FIXER_PROMPT


def test_only_two_instructions_change_and_source_is_untouched():
    original = copy.deepcopy(FIXER_PROMPT)
    arms = prompt_arms()
    assert arms["control"] == original
    assert FIXER_PROMPT == original
    before = arms["control"]["system"].splitlines()
    after = arms["conditional"]["system"].splitlines()
    assert len(before) == len(after)
    changed = [(a, b) for a, b in zip(before, after) if a != b]
    assert len(changed) == 2
    assert all("Duplicates" in a or "NULL (MANDATORY" in a for a, b in changed)
    assert arms["control"]["user_template"] == arms["conditional"]["user_template"]
    assert "CRITICAL — COUNT(DISTINCT id)" in arms["conditional"]["system"]


def test_question_sanitization_and_sampling_do_not_depend_on_gold():
    entries = [dict(question_id=i, db_id=str(i % 2), question=f"Question {i}",
                    evidence="provided hint", SQL=f"SECRET GOLD {i}", difficulty="simple")
               for i in range(20)]
    chosen = sample_questions(entries, 3, "fixed-before-evaluation")
    assert len(chosen) == 6
    assert "SECRET" not in str(chosen)
    assert all(set(e) == {"question_id", "db_id", "question", "evidence"} for e in chosen)
    altered = copy.deepcopy(entries)
    for entry in altered:
        entry["SQL"] = "different gold"
        entry["difficulty"] = "challenging"
    assert sample_questions(altered[::-1], 3, "fixed-before-evaluation") == chosen


def test_sql_is_readonly_and_query_execution_has_a_deadline(tmp_path):
    path = tmp_path / "db.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE items(value TEXT)")
        conn.execute("INSERT INTO items VALUES ('kept')")
    assert not execute_readonly("DELETE FROM items", path)[0]
    assert execute_readonly("SELECT value FROM items", path)[1] == [("kept",)]
    query = "WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n) SELECT SUM(x) FROM n"
    assert execute_readonly(query, path, timeout=0.01)[2] == "interrupted"


def test_paired_fixer_replay_preserves_input_and_exposes_null_regression(tmp_path):
    path = tmp_path / "db.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE items(value TEXT)")
        conn.executemany("INSERT INTO items VALUES (?)", [(None,), ("a",), ("a",)])
    original_rows = [(None,), ("a",), ("a",)]
    original = SimpleNamespace(sql="SELECT value FROM items", result=original_rows, row_count=3)
    class FakeClient:
        def __init__(self): self.calls = 0
        def complete(self, prompt, system_prompt, **kwargs):
            self.calls += 1
            if "NULL (MANDATORY when present)" in system_prompt and self.calls == 1:
                return json.dumps({"is_acceptable":False, "issues":["nulls and duplicates"],
                    "refined_sql":"SELECT DISTINCT value FROM items WHERE value IS NOT NULL"})
            return '{"is_acceptable":true,"issues":[],"refined_sql":""}'
    results = {}
    for name, prompt in prompt_arms().items():
        fixer = ReadOnlyFixer(FakeClient())
        fixer.prompt_config = prompt
        results[name] = fixer.fix(SimpleNamespace(candidate_id=1), original,
                                  "List stored values", "", path, "items(value TEXT)")
    assert results["control"].final_rows == [("a",)]
    assert results["conditional"].final_rows == original_rows
    assert original.result == original_rows
    assert original.sql == "SELECT value FROM items"


def test_schema_uses_supplied_description_verbatim(tmp_path):
    folder = tmp_path / "example"
    descriptions = folder / "database_description"
    descriptions.mkdir(parents=True)
    with sqlite3.connect(folder / "example.sqlite") as conn:
        conn.execute("CREATE TABLE items(value TEXT)")
    (descriptions / "items.csv").write_text(
        'original_column_name,column_description\nvalue,"Original line one\nOriginal line two"\n',
        encoding="utf-8", newline="")
    schema, sources = bird_schema("example", tmp_path)
    assert 'Original line one\\nOriginal line two' in schema
    assert "CREATE TABLE items(value TEXT)" in schema
    assert len(sources) == 1


def test_evaluation_requires_completed_pairs_before_reading_gold(tmp_path):
    write_json(tmp_path / "manifest.json", {})
    write_json(tmp_path / "inputs.json", [{"question_id":1}])
    with pytest.raises(ValueError, match="before opening gold"):
        evaluate(SimpleNamespace(out=tmp_path, questions=tmp_path / "missing_gold.json"))


def test_evaluation_ignores_duplicate_rows_but_preserves_column_order(tmp_path):
    db = tmp_path / "db" / "example"
    db.mkdir(parents=True)
    with sqlite3.connect(db / "example.sqlite") as conn:
        conn.execute("CREATE TABLE items(a INTEGER, b INTEGER)")
        conn.executemany("INSERT INTO items VALUES (?,?)", [(1,2), (1,2)])
    gold = tmp_path / "gold.json"
    write_json(gold, [{"question_id":1, "SQL":"SELECT DISTINCT a,b FROM items"}])
    write_json(tmp_path / "manifest.json", {
        "source_sha256":{str(gold.resolve()):sha256(gold)}, "db_root":str(tmp_path / "db"),
        "arms":["control", "conditional"], "model":"fake", "per_database":1})
    write_json(tmp_path / "inputs.json", [{"question_id":1, "db_id":"example",
                                         "sql":"SELECT a,b FROM items"}])
    (tmp_path / "pairs").mkdir()
    write_json(tmp_path / "pairs/1.json", {"status":"complete", "arms":{
        "control":{"final_sql":"SELECT b,a FROM items"},
        "conditional":{"final_sql":"SELECT a,b FROM items"}}})
    evaluate(SimpleNamespace(out=tmp_path, questions=gold))
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["correct"] == {"input":1, "control":0, "conditional":1}
    assert summary["recovered_vs_control"] == [1]
    assert summary["regressed_vs_control"] == []
    job = {"question_id":1, "db_path":str(db / "example.sqlite"),
           "gold":"SELECT DISTINCT a,b FROM items", "arms":["conditional"],
           "sql":"SELECT a,b FROM items"}
    assert check(job)["res"] == 1
    assert check({**job, "sql":"SELECT b,a FROM items"})["res"] == 0
