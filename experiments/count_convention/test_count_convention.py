"""Tests for RULE O.

  python -m experiments.count_convention.test_count_convention
"""
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.count_convention import pipeline_patch, prompt

SCHEMA = {"Patient": {"table_readable_name": "patients",
                      "columns": [{"name": "ID"}, {"name": "SEX"}, {"name": "Admission"}]},
          "Laboratory": {"table_readable_name": "lab tests",
                         "columns": [{"name": "ID"}, {"name": "IGG"}, {"name": "Date"}]}}
# the same phrase families the rule quotes; note "should consider THE distinct atoms"
IMPERATIVE = re.compile(r"should consider (the )?distinct|number of (distinct|unique)|"
                        r"without repetitive|repetitive ones", re.I)


class TextTest(unittest.TestCase):
    def test_rule_added_once(self):
        first = prompt.with_count_rule("BASE RULES")
        self.assertIn(prompt.MARKER, first)
        self.assertEqual(first, prompt.with_count_rule(first))

    def test_empty_prompt_untouched(self):
        self.assertEqual("", prompt.with_count_rule(""))

    def test_rule_states_both_directions(self):
        text = re.sub(r"\s+", " ", prompt.COUNT_SECTION)   # the rule text is line-wrapped
        self.assertIn("MUST", text)                      # imperative present -> required
        self.assertIn("PREFER plain", text)              # absent -> preferred, not mandated
        self.assertIn("how many different", text)        # question-side override kept

    def test_rule_quotes_the_real_evidence_phrases(self):
        """The phrases come from dev evidence; all five families must be listed."""
        text = re.sub(r"\s+", " ", prompt.COUNT_SECTION).lower()
        for phrase in ("should consider distinct", "distinct/unique ones",
                       "number of distinct", "without repetitive", "repetitive ones"):
            self.assertIn(phrase, text)

    def test_phrases_match_what_dev_evidence_actually_says(self):
        dev = ROOT / "data/bird_data/dev.json"
        if not dev.exists():
            self.skipTest("dev.json not available")
            return
        qs = json.loads(dev.read_text(encoding="utf-8"))
        flagged = [q for q in qs if IMPERATIVE.search(q.get("evidence") or "")]
        self.assertEqual(18, len(flagged))                     # measured: 18 imperatives in dev
        deduped = sum(1 for q in flagged if re.search(r"DISTINCT", q["SQL"], re.I))
        self.assertGreaterEqual(deduped, 17)                   # the signal the rule relies on


class PatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not pipeline_patch.install():
            raise unittest.SkipTest("src not importable")

    def test_all_strategies_get_rule_o(self):
        from src.generation.prompt_builder import PromptBuilder
        builder = PromptBuilder()
        for strategy in builder.strategy_prompts:
            system = builder.build("how many patients?", SCHEMA, strategy, evidence="")["system"]
            self.assertIn(prompt.MARKER, system, strategy)

    def test_existing_rules_survive(self):
        from src.generation.prompt_builder import PromptBuilder
        builder = PromptBuilder()
        system = builder.build("q", SCHEMA, list(builder.strategy_prompts)[0], evidence="")["system"]
        for kept in ("RULE A", "RULE C", "RULE E", "RULE K"):
            self.assertIn(kept, system)

    def test_output_format_untouched(self):
        from src.generation.prompt_builder import PromptBuilder
        builder = PromptBuilder()
        system = builder.build("q", SCHEMA, list(builder.strategy_prompts)[0], evidence="")["system"]
        self.assertIn("Generate ONLY the SQL query", system)   # no extraction risk

    def test_user_prompt_not_touched(self):
        from src.generation.prompt_builder import PromptBuilder
        builder = PromptBuilder()
        prompts = builder.build("q", SCHEMA, list(builder.strategy_prompts)[0], evidence="ev")
        self.assertNotIn("RULE O", prompts["user"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
