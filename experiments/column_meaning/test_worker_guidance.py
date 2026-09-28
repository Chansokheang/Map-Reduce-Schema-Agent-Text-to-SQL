"""Tests for the --column-guidance arm (cm_v1 + the map agent's column-selection instruction).

A separate module because the two arms patch the same methods: whichever installs first wins,
so each needs its own process.

  python -m experiments.column_meaning.test_worker_guidance
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.column_meaning.test_column_meaning import DB, HAS_DOCS, SCHEMA, FakeClient


@unittest.skipUnless(HAS_DOCS, "database descriptions not available")
class WorkerGuidanceTest(unittest.TestCase):
    """The --column-guidance arm: cm_v1 plus one instruction on the map agent's prompts."""

    @classmethod
    def setUpClass(cls):
        from experiments.column_meaning import worker_guidance_patch
        cls.patch = worker_guidance_patch
        from src.agents.worker import SchemaWorker
        if getattr(SchemaWorker, "_qasql_column_meaning", False):
            raise unittest.SkipTest("the column-meaning arm already patched the worker")
        if not worker_guidance_patch.install():
            raise unittest.SkipTest("src.agents not importable")

    def _prompts(self, question, evidence=""):
        from src.agents.manager import SchemaManager
        client = FakeClient()
        manager = SchemaManager(llm_client=client)
        decomposed = manager._heuristic_decompose(question)
        decomposed.original_query = question
        manager.coordinate_workers(decomposed_query=decomposed, schema=SCHEMA, evidence=evidence)
        return client.prompts

    def test_instruction_ends_every_worker_prompt(self):
        prompts = self._prompts("List the school names with a free meal count over 800.")
        self.assertTrue(prompts)
        for prompt in prompts:
            self.assertTrue(prompt.rstrip().endswith("similar columns exist across tables."))
            self.assertIn("**Column Selection:**", prompt)

    def test_blocks_still_precede_the_instruction(self):
        prompts = self._prompts("How many schools in Fresno are directly charter-funded?")
        for prompt in prompts:
            if "# Column meanings" in prompt:
                self.assertLess(prompt.index("# Column meanings"), prompt.index("**Column Selection:**"))

    def test_instruction_is_present_even_without_a_database(self):
        self.assertIn("**Column Selection:**", self.patch.worker_suffix(None, "q", "", []))

    def test_generation_prompts_do_not_get_the_instruction(self):
        from src.generation.prompt_builder import PromptBuilder
        builder = PromptBuilder()
        strategy = list(builder.strategy_prompts)[0]
        prompt = builder.build("List the school names.", SCHEMA, strategy, evidence="")["user"]
        self.assertNotIn("Column Selection", prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
