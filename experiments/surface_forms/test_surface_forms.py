"""Tests for the surface-form rule (RULE Q).

  python -m experiments.surface_forms.test_surface_forms
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.surface_forms import pipeline_patch, rule

SCHEMA = {"schools": {"table_readable_name": "schools",
                      "columns": [{"name": "CDSCode"}, {"name": "School"}, {"name": "Phone"}]}}


class RuleTextTest(unittest.TestCase):
    def test_rule_is_appended_once(self):
        first = rule.with_rule("BASE PROMPT")
        self.assertIn("RULE Q", first)
        self.assertEqual(first, rule.with_rule(first))

    def test_base_prompt_is_kept(self):
        self.assertTrue(rule.with_rule("BASE PROMPT").startswith("BASE PROMPT"))

    def test_empty_prompt_is_left_alone(self):
        self.assertEqual("", rule.with_rule(""))
        self.assertIsNone(rule.with_rule(None))

    def test_rule_names_all_three_constructs(self):
        text = rule.with_rule("x")
        for construct in ("WITH/CTE", "COALESCE", "IFNULL"):
            self.assertIn(construct, text, construct)

    def test_rule_states_the_evidence_exception(self):
        self.assertIn("unless the question or the evidence asks", rule.with_rule("x"))

    def test_rule_is_one_line_of_instruction(self):
        body = [ln for ln in rule.SURFACE_FORM_RULE.splitlines() if ln.startswith("- ")]
        self.assertEqual(1, len(body))


class PatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not pipeline_patch.install():
            raise unittest.SkipTest("src not importable")

    def test_every_strategy_system_prompt_carries_the_rule(self):
        from src.generation.prompt_builder import PromptBuilder
        builder = PromptBuilder()
        for strategy in builder.strategy_prompts:
            prompts = builder.build("List the phone numbers of chartered schools.", SCHEMA,
                                    strategy, evidence="")
            self.assertIn("RULE Q", prompts["system"], strategy)

    def test_existing_rules_are_kept_and_still_come_first(self):
        from src.generation.prompt_builder import PromptBuilder
        builder = PromptBuilder()
        strategy = list(builder.strategy_prompts)[0]
        system = builder.build("q", SCHEMA, strategy, evidence="")["system"]
        for kept in ("RULE A", "RULE J", "RULE K"):
            self.assertIn(kept, system)
        # appended, never promoted: RULE A must not be displaced (see rule.py)
        self.assertLess(system.index("RULE A"), system.index("RULE Q"))

    def test_user_prompt_is_not_touched(self):
        from src.generation.prompt_builder import PromptBuilder
        builder = PromptBuilder()
        strategy = list(builder.strategy_prompts)[0]
        self.assertNotIn("RULE Q", builder.build("q", SCHEMA, strategy, evidence="")["user"])

    def test_install_is_idempotent(self):
        from src.generation.prompt_builder import PromptBuilder
        builder = PromptBuilder()
        strategy = list(builder.strategy_prompts)[0]
        self.assertTrue(pipeline_patch.install())
        system = builder.build("q", SCHEMA, strategy, evidence="")["system"]
        self.assertEqual(1, system.count("RULE Q"))

    def test_fixer_prompt_is_left_alone(self):
        from src.prompt.fixer import FIXER_PROMPT
        self.assertNotIn("RULE Q", FIXER_PROMPT["system"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
