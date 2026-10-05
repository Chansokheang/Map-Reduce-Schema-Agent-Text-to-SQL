"""Tests for rule promotion.

  python -m experiments.rule_position.test_promote
"""
import json
import os
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.rule_position import promote as P

SAMPLE = """Intro text.

**RULE A — CAST:**
- do not cast

**RULE B — ORDER:**
- multiply first

**RULE K — SUPERLATIVE QUESTIONS:**
- use ORDER BY ... LIMIT 1

Rules:
1. Generate ONLY the SQL
"""


class BlockTest(unittest.TestCase):
    def test_finds_a_rule_with_its_bullets(self):
        block = P.rule_block(SAMPLE, "K")
        self.assertIn("RULE K", block)
        self.assertIn("ORDER BY ... LIMIT 1", block)
        self.assertNotIn("RULE A", block)

    def test_unknown_letter_is_none(self):
        self.assertIsNone(P.rule_block(SAMPLE, "Z"))


class PromoteTest(unittest.TestCase):
    def test_moves_the_rule_above_rule_a(self):
        out = P.promote(SAMPLE, ["K"])
        self.assertLess(out.index("RULE K"), out.index("**RULE A"))

    def test_appears_exactly_once(self):
        out = P.promote(SAMPLE, ["K"])
        self.assertEqual(out.count("**RULE K"), 1)

    def test_keeps_every_other_rule(self):
        out = P.promote(SAMPLE, ["K"])
        for kept in ("RULE A", "RULE B", "Generate ONLY the SQL"):
            self.assertIn(kept, out)

    def test_idempotent(self):
        once = P.promote(SAMPLE, ["K"])
        self.assertEqual(once, P.promote(once, ["K"]))

    def test_several_rules_keep_the_requested_order(self):
        out = P.promote(SAMPLE, ["K", "B"])
        self.assertLess(out.index("RULE K"), out.index("RULE B"))
        self.assertLess(out.index("RULE B"), out.index("**RULE A"))

    def test_unknown_letter_changes_nothing(self):
        self.assertEqual(P.promote(SAMPLE, ["Z"]), SAMPLE)

    def test_missing_anchor_changes_nothing(self):
        self.assertEqual(P.promote("no rules here", ["K"]), "no rules here")


class JudgeTest(unittest.TestCase):
    JSAMPLE = ("ROLE: reviewer.\n\nEVALUATION CRITERIA\n- RULE A - cast\n"
               "- RULE K - superlatives use ORDER BY LIMIT 1\n")

    def test_moves_the_bullet_above_the_criteria_heading(self):
        out = P.judge_promote(self.JSAMPLE, ["K"])
        self.assertLess(out.index("RULE K"), out.index("EVALUATION CRITERIA"))
        self.assertEqual(out.count("RULE K"), 1)
        self.assertIn("RULE A", out)

    def test_idempotent_and_safe_without_the_anchor(self):
        once = P.judge_promote(self.JSAMPLE, ["K"])
        self.assertEqual(once, P.judge_promote(once, ["K"]))
        self.assertEqual(P.judge_promote("no criteria", ["K"]), "no criteria")


class LiveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["QASQL_PROMOTE_RULES"] = "K"
        if not P.install():
            raise unittest.SkipTest("src not importable")

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("QASQL_PROMOTE_RULES", None)

    def test_every_strategy_puts_rule_k_first(self):
        from src.generation.prompt_builder import PromptBuilder
        path = ROOT / "data/bird_data/schemas/california_schools_schema.json"
        if not path.exists():
            self.skipTest("schema file missing")
        schema = json.loads(path.read_text(encoding="utf-8"))["tables"]
        builder = PromptBuilder()
        for strategy in builder.strategy_prompts:
            system = builder.build("q", schema, strategy, focused_schema=schema,
                                   evidence="")["system"]
            self.assertLess(system.index("RULE K"), system.index("**RULE A"), strategy.value)
            self.assertEqual(system.count("**RULE K"), 1, strategy.value)

    def test_all_eleven_rules_survive(self):
        from src.generation.prompt_builder import PromptBuilder
        path = ROOT / "data/bird_data/schemas/california_schools_schema.json"
        if not path.exists():
            self.skipTest("schema file missing")
        schema = json.loads(path.read_text(encoding="utf-8"))["tables"]
        system = PromptBuilder().build("q", schema, list(PromptBuilder().strategy_prompts)[0],
                                       evidence="")["system"]
        self.assertEqual(len(re.findall(r"\*\*RULE [A-Z]", system)), 11)

    def test_judge_prompt_is_promoted_too(self):
        from src.prompt import JUDGE_PROMPT
        system = JUDGE_PROMPT["system"]
        self.assertLess(system.index("RULE K"), system.index("EVALUATION CRITERIA"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
