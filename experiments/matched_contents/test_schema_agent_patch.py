"""Tests for join_paths.py and schema_agent_patch.py.

  python -m experiments.matched_contents.test_schema_agent_patch
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.matched_contents import join_paths, schema_agent_patch
from experiments.matched_contents.retriever import index_path

DB = "california_schools"
HAS_INDEX = index_path(DB, schema_agent_patch._index_dir()).exists()
HAS_DB = join_paths.database_path(DB) is not None

SCHEMA = {
    "frpm": {"table_readable_name": "Free or reduced price meals",
             "columns": [{"name": "CDSCode"}, {"name": "School Name"}, {"name": "Charter School (Y/N)"}]},
    "satscores": {"table_readable_name": "SAT scores",
                  "columns": [{"name": "cds"}, {"name": "sname"}, {"name": "NumGE1500"}]},
    "schools": {"table_readable_name": "Schools",
                "columns": [{"name": "CDSCode"}, {"name": "School"}, {"name": "County"},
                            {"name": "EdOpsCode"}]},
}


class FakeClient:
    """Records every prompt and answers the worker's JSON contract."""

    def __init__(self):
        self.prompts = []

    def complete(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return '{"relevant": true, "score": 1.0, "reason": "test", "relevant_columns": ["CDSCode"]}'


@unittest.skipUnless(HAS_DB, "dev databases not available")
class JoinPathsTest(unittest.TestCase):
    def test_declared_foreign_keys_are_found(self):
        edges = join_paths.foreign_keys(DB)
        pairs = {(e[0], e[1], e[2], e[3]) for e in edges}
        self.assertIn(("frpm", "CDSCode", "schools", "CDSCode"), pairs)
        self.assertIn(("satscores", "cds", "schools", "CDSCode"), pairs)

    def test_focus_tables_come_first_and_others_are_dropped(self):
        edges = join_paths.join_columns(DB, tables=["frpm", "satscores", "schools"], focus=["frpm"])
        self.assertTrue(edges)
        self.assertEqual({"frpm"}, {e["table"] for e in edges})       # satscores-schools is not shown

    def test_unavailable_tables_are_excluded(self):
        self.assertEqual([], join_paths.join_columns(DB, tables=["satscores"], focus=["satscores"]))

    def test_block_formats_one_line_per_edge(self):
        block = join_paths.format_block(join_paths.join_columns(DB, focus=["frpm"]))
        self.assertIn("# Join columns", block)
        self.assertIn("- frpm.CDSCode = schools.CDSCode", block)

    def test_empty_edges_give_empty_block(self):
        self.assertEqual("", join_paths.format_block([]))


@unittest.skipUnless(HAS_INDEX and HAS_DB, "value index or dev databases not available")
class SchemaAgentPatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not schema_agent_patch.install():
            raise unittest.SkipTest("src.agents not importable")

    def _run_workers(self, question, evidence=""):
        from src.agents.manager import SchemaManager
        client = FakeClient()
        manager = SchemaManager(llm_client=client)
        decomposed = manager._heuristic_decompose(question)
        decomposed.original_query = question
        manager.coordinate_workers(decomposed_query=decomposed, schema=SCHEMA, evidence=evidence)
        return client.prompts

    def test_every_worker_sees_values_from_other_tables(self):
        prompts = self._run_workers("What is the phone number of the school 'Alameda High'?")
        self.assertTrue(prompts)
        with_values = [p for p in prompts if "# Matched contents" in p]
        self.assertEqual(len(prompts), len(with_values))
        # the worker scoring satscores is told the value lives in schools/frpm, not in its table
        satscores = next(p for p in prompts if "Table: satscores" in p)
        self.assertIn("Alameda High", satscores)

    def test_join_columns_are_shown_to_the_worker(self):
        prompts = self._run_workers("How many schools in Riverside are directly charter-funded?")
        joined = [p for p in prompts if "# Join columns" in p]
        self.assertTrue(joined)
        self.assertTrue(any("= schools.CDSCode" in p for p in joined))

    def test_context_is_cleared_after_the_call(self):
        self._run_workers("How many schools are there?")
        self.assertIsNone(schema_agent_patch._CURRENT["context"])

    def test_prompt_keeps_the_original_text(self):
        prompts = self._run_workers("How many schools in Riverside are directly charter-funded?")
        for prompt in prompts:
            self.assertIn("Return ONLY JSON:", prompt)
            self.assertLess(prompt.index("Return ONLY JSON:"), prompt.index("# "))

    def test_retrieval_covers_tables_the_agent_may_later_drop(self):
        hits = schema_agent_patch.retrieve_for_question(
            DB, "What is the phone number of the school 'Alameda High'?", "")
        self.assertTrue({h["table"] for h in hits} - {"frpm"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
