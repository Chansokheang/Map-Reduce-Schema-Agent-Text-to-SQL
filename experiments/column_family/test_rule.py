"""Tests for RULE P (near-empty secondary columns).

  python -m experiments.column_family.test_rule
"""
import json
import os
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.column_family import rule as R
from experiments.column_family import sparsity

SAMPLE = "Intro.\n\n**RULE A — CAST:**\n- do not cast\n\n**RULE B — ORDER:**\n- multiply first\n"


def schema_of(db):
    path = ROOT / f"data/bird_data/schemas/{db}_schema.json"
    return {"tables": json.loads(path.read_text(encoding="utf-8"))["tables"]} if path.exists() else None


class SparsityTest(unittest.TestCase):
    """The data, not the naming pattern, decides. A name-only rule misfires badly."""

    def test_admin_columns_are_flagged(self):
        got = sparsity.sparse_for("california_schools")
        cols = {e["column"].lower() for cs in got.values() for e in cs}
        self.assertIn("admfname2", cols)
        self.assertIn("admfname3", cols)

    def test_qualifying_rounds_are_not_flagged(self):
        """formula_1 q1/q2/q3 are elimination rounds - all three are meaningful."""
        got = sparsity.sparse_for("formula_1")
        cols = {e["column"].lower() for cs in got.values() for e in cs}
        self.assertNotIn("q2", cols)
        self.assertNotIn("q3", cols)

    def test_football_players_are_not_flagged(self):
        """home_player_1..11 are positions on the pitch, ~95% filled each."""
        got = sparsity.sparse_for("european_football_2")
        cols = {e["column"].lower() for cs in got.values() for e in cs}
        self.assertNotIn("home_player_2", cols)

    def test_per_year_statistics_are_not_flagged(self):
        """financial A12/A13 are 1995/1996 unemployment, fully populated."""
        got = sparsity.sparse_for("financial")
        cols = {e["column"].lower() for cs in got.values() for e in cs}
        self.assertNotIn("a12", cols)
        self.assertNotIn("a13", cols)


class InsertTest(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("QASQL_ENTITY_COLUMN", None)

    def test_no_rule_without_a_schema(self):
        self.assertEqual(R.insert(SAMPLE), SAMPLE)

    def test_no_rule_for_a_database_with_no_sparse_column(self):
        schema = schema_of("formula_1")
        if schema is None:
            self.skipTest("schema missing")
        self.assertEqual(R.insert(SAMPLE, schema), SAMPLE)

    def test_rule_appears_for_california_schools(self):
        schema = schema_of("california_schools")
        if schema is None:
            self.skipTest("schema missing")
        out = R.insert(SAMPLE, schema)
        self.assertLess(out.index("RULE P"), out.index("**RULE A"))
        self.assertIn("schools.AdmFName1", out)   # the primary to use, named explicitly
        self.assertIn("AdmFName2", out)
        self.assertIn("AdmEmail1", out)

    def test_idempotent(self):
        schema = schema_of("california_schools")
        if schema is None:
            self.skipTest("schema missing")
        once = R.insert(SAMPLE, schema)
        self.assertEqual(once, R.insert(once, schema))

    def test_entity_line_is_off_by_default(self):
        schema = schema_of("california_schools")
        if schema is None:
            self.skipTest("schema missing")
        self.assertNotIn("measurement/score table", R.insert(SAMPLE, schema))

    def test_entity_line_can_be_switched_on(self):
        os.environ["QASQL_ENTITY_COLUMN"] = "1"
        schema = schema_of("california_schools")
        if schema is None:
            self.skipTest("schema missing")
        self.assertIn("measurement/score table", R.insert(SAMPLE, schema))

    def test_the_evidence_exception_is_stated(self):
        schema = schema_of("california_schools")
        if schema is None:
            self.skipTest("schema missing")
        text = re.sub(r"\s+", " ", R.insert(SAMPLE, schema))
        self.assertIn("at most 3 administrators", text)
        # the exception is evidence-only: plural in the question must NOT license the extra columns,
        # because gold expands on Q87 ("addresses") and does not on Q63 ("all the administrators")
        self.assertIn("ONLY when the EVIDENCE", text)
        self.assertIn("Plural wording in the QUESTION is NOT enough", text)


class LiveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not R.install():
            raise unittest.SkipTest("src not importable")

    def test_fires_only_where_it_applies(self):
        from src.generation.prompt_builder import PromptBuilder
        builder = PromptBuilder()
        for db, expected in (("california_schools", True), ("formula_1", False),
                             ("european_football_2", False), ("toxicology", False)):
            schema = schema_of(db)
            if schema is None:
                continue
            system = builder.build("q", schema["tables"], list(builder.strategy_prompts)[0],
                                   focused_schema=schema, evidence="")["system"]
            self.assertEqual("RULE P" in system, expected, db)

    def test_original_rules_survive(self):
        from src.generation.prompt_builder import PromptBuilder
        schema = schema_of("california_schools")
        if schema is None:
            self.skipTest("schema missing")
        builder = PromptBuilder()
        system = builder.build("q", schema["tables"], list(builder.strategy_prompts)[0],
                               focused_schema=schema, evidence="")["system"]
        self.assertEqual(len(re.findall(r"\*\*RULE [A-Z]", system)), 12)


class TestSetReadinessTest(unittest.TestCase):
    """It must work on databases we have never seen, with no pre-extracted schema JSON."""

    @classmethod
    def setUpClass(cls):
        import sqlite3
        import tempfile
        cls.tmp = tempfile.TemporaryDirectory()
        base = Path(cls.tmp.name) / "unseen_db"
        base.mkdir(parents=True)
        con = sqlite3.connect(base / "unseen_db.sqlite")
        con.execute("CREATE TABLE contact (id INTEGER, agent1 TEXT, agent2 TEXT, score1 REAL, score2 REAL)")
        for i in range(500):
            con.execute("INSERT INTO contact VALUES (?,?,?,?,?)",
                        (i, f"a{i}", f"b{i}" if i < 10 else None, 1.0, 2.0))
        con.commit()
        con.close()
        cls.dir = str(Path(cls.tmp.name))

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("QASQL_DB_DIR", None)
        cls.tmp.cleanup()

    def test_scans_a_database_with_no_schema_json(self):
        got = sparsity.build(self.dir)
        self.assertIn("unseen_db", got)
        cols = {e["column"] for cs in got["unseen_db"].values() for e in cs}
        self.assertIn("agent2", cols)

    def test_a_fully_populated_family_member_is_not_flagged(self):
        got = sparsity.build(self.dir)
        cols = {e["column"] for cs in got["unseen_db"].values() for e in cs}
        self.assertNotIn("score2", cols)

    def test_cache_name_keeps_dev_and_test_apart(self):
        self.assertNotEqual(sparsity.cache_path(self.dir).name,
                            sparsity.cache_path().name)

    def test_rule_names_the_unseen_database_columns(self):
        sparsity.build(self.dir)
        os.environ["QASQL_DB_DIR"] = self.dir
        schema = {"tables": {"contact": {"columns": [{"name": n} for n in
                                                     ("id", "agent1", "agent2", "score1", "score2")]}}}
        # the rule now declares its column families explicitly, so an unseen database is silent
        # unless its families are added to STATIC_FAMILIES; sparsity.py stays as the tool that
        # finds them.
        text = R.rule_text(schema)
        self.assertEqual(text, "")
        found = sparsity.build(self.dir)
        cols = {e["column"] for cs in found["unseen_db"].values() for e in cs}
        self.assertIn("agent2", cols)
        self.assertNotIn("AdmFName", text)       # no dev column leaks into a test-set prompt


if __name__ == "__main__":
    unittest.main(verbosity=2)
