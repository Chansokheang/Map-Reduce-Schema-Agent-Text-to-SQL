"""Tests for the generation-rules patch.

  python -m experiments.generation_rules.test_generation_rules
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.generation_rules import pipeline_patch, rules

SCHEMA = {"schools": {"table_readable_name": "schools",
                      "columns": [{"name": "CDSCode"}, {"name": "School"}, {"name": "Phone"}]}}


class RuleTextTest(unittest.TestCase):
    def test_rules_are_appended_once(self):
        first = rules.with_rules("BASE PROMPT")
        self.assertIn("RULE L", first)
        self.assertIn("RULE M", first)
        self.assertIn("RULE N", first)
        self.assertEqual(first, rules.with_rules(first))

    def test_empty_prompt_is_left_alone(self):
        self.assertEqual("", rules.with_rules(""))

    def test_fixer_null_rule_is_replaced_once(self):
        before = "x\n" + rules.FIXER_NULL_RULE_OLD + "\ny"
        after = rules.corrected_fixer_prompt(before)
        self.assertNotIn(rules.FIXER_NULL_RULE_OLD, after)
        self.assertIn("only where it changes the answer", after)
        self.assertEqual(after, rules.corrected_fixer_prompt(after))

    def test_unknown_fixer_wording_is_not_mangled(self):
        self.assertEqual("nothing to replace", rules.corrected_fixer_prompt("nothing to replace"))


class PatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not pipeline_patch.install():
            raise unittest.SkipTest("src not importable")

    def test_every_strategy_system_prompt_carries_the_rules(self):
        from src.generation.prompt_builder import PromptBuilder
        builder = PromptBuilder()
        for strategy in builder.strategy_prompts:
            prompts = builder.build("List the phone numbers of chartered schools.", SCHEMA,
                                    strategy, evidence="")
            self.assertIn("RULE L", prompts["system"], strategy)
            self.assertIn("RULE M", prompts["system"], strategy)
            self.assertIn("RULE N", prompts["system"], strategy)

    def test_existing_rules_are_kept(self):
        from src.generation.prompt_builder import PromptBuilder
        builder = PromptBuilder()
        strategy = list(builder.strategy_prompts)[0]
        system = builder.build("q", SCHEMA, strategy, evidence="")["system"]
        for kept in ("RULE A", "RULE J", "RULE K"):
            self.assertIn(kept, system)

    def test_user_prompt_is_not_touched(self):
        from src.generation.prompt_builder import PromptBuilder
        builder = PromptBuilder()
        strategy = list(builder.strategy_prompts)[0]
        self.assertNotIn("RULE L", builder.build("q", SCHEMA, strategy, evidence="")["user"])

    def test_fixer_prompt_no_longer_mandates_null_filters(self):
        from src.prompt.fixer import FIXER_PROMPT
        self.assertNotIn("NULL (MANDATORY when present)", FIXER_PROMPT["system"])
        self.assertIn("only where it changes the answer", FIXER_PROMPT["system"])

    def test_fixer_duplicate_rule_is_left_as_it_was(self):
        from src.prompt.fixer import FIXER_PROMPT
        self.assertIn("Duplicates (MANDATORY", FIXER_PROMPT["system"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
