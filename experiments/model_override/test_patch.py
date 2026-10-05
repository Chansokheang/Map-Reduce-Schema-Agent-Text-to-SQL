"""Tests for per-component model selection.

  python -m experiments.model_override.test_patch
"""
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.model_override import patch as P

VARS = ("QASQL_MODEL_JUDGE", "QASQL_MODEL_FIXER", "QASQL_MODEL_GENERATION", "QASQL_MODEL_SCHEMA")


class ConfigTest(unittest.TestCase):
    def tearDown(self):
        for v in VARS:
            os.environ.pop(v, None)

    def test_nothing_applied_by_default(self):
        self.assertEqual(P.applied(), {})
        self.assertEqual(P.install(), {})

    def test_reads_each_component(self):
        os.environ["QASQL_MODEL_JUDGE"] = "m1"
        os.environ["QASQL_MODEL_FIXER"] = "m2"
        self.assertEqual(P.applied(), {"judge": "m1", "fixer": "m2"})


class PatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["QASQL_MODEL_JUDGE"] = "claude-opus-5-5"
        cls.done = P.install()

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("QASQL_MODEL_JUDGE", None)

    def test_judge_gets_the_named_model(self):
        from src.selection.judge import SQLJudge
        from src.utils.llm_client import create_llm_client
        judge = SQLJudge(llm_client=create_llm_client(provider="headless"))
        self.assertEqual(getattr(judge.llm_client, "model", None), "claude-opus-5-5")

    def test_other_components_are_untouched(self):
        """The whole point: -m switches everything, this must not."""
        from src.selection.fixer import SQLFixer
        from src.utils.llm_client import create_llm_client
        fixer = SQLFixer(llm_client=create_llm_client(provider="headless"))
        self.assertNotEqual(getattr(fixer.llm_client, "model", None), "claude-opus-5-5")

    def test_clients_are_reused_per_model(self):
        self.assertIs(P.client_for("claude-opus-5-5"), P.client_for("claude-opus-5-5"))

    def test_install_is_idempotent(self):
        before = P.install()
        self.assertEqual(before, P.install())

    def test_a_judge_without_a_client_is_left_alone(self):
        from src.selection.judge import SQLJudge
        self.assertIsNone(SQLJudge(llm_client=None).llm_client)


if __name__ == "__main__":
    unittest.main(verbosity=2)
