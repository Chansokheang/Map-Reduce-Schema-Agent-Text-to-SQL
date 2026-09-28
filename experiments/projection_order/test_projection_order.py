"""Tests for the projection-order arm.

  python -m experiments.projection_order.test_projection_order
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.projection_order import pipeline_patch, prompt

SCHEMA = {"schools": {"table_readable_name": "schools",
                      "columns": [{"name": "CDSCode"}, {"name": "School"}, {"name": "Phone"}]}}
ANSWER = """outputs: phone number, extension, school name

```sql
SELECT Phone, Ext, School FROM schools WHERE Zip = '95203-3704'
```
"""


class TextTest(unittest.TestCase):
    def test_section_added_once(self):
        first = prompt.with_projection_order("BASE\n1. Generate ONLY the SQL query, no explanations")
        self.assertIn(prompt.MARKER, first)
        self.assertEqual(first, prompt.with_projection_order(first))

    def test_output_instructions_rewritten(self):
        base = ("x\n1. Generate ONLY the SQL query, no explanations\n"
                "9. Return only the SQL query, nothing else")
        out = prompt.with_projection_order(base)
        self.assertTrue(prompt.rewrote_output_instructions(out))

    def test_empty_prompt_untouched(self):
        self.assertEqual("", prompt.with_projection_order(""))

    def test_section_mentions_trailing_requests_and_conventional_groups(self):
        self.assertIn("goes LAST", prompt.PROJECTION_SECTION)
        self.assertIn("street, city, state, zip", prompt.PROJECTION_SECTION)


class PatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not pipeline_patch.install():
            raise unittest.SkipTest("src not importable, or the reasoning arm is active")

    def test_all_strategies_get_the_step(self):
        from src.generation.prompt_builder import PromptBuilder
        builder = PromptBuilder()
        for strategy in builder.strategy_prompts:
            system = builder.build("q", SCHEMA, strategy, evidence="")["system"]
            self.assertIn(prompt.MARKER, system, strategy)
            self.assertTrue(prompt.rewrote_output_instructions(system), strategy)

    def test_rules_a_to_k_survive(self):
        from src.generation.prompt_builder import PromptBuilder
        builder = PromptBuilder()
        system = builder.build("q", SCHEMA, list(builder.strategy_prompts)[0], evidence="")["system"]
        for kept in ("RULE A", "RULE E", "RULE K"):
            self.assertIn(kept, system)

    def test_rule_l_m_n_are_not_added(self):
        from src.generation.prompt_builder import PromptBuilder
        builder = PromptBuilder()
        system = builder.build("q", SCHEMA, list(builder.strategy_prompts)[0], evidence="")["system"]
        self.assertNotIn("RULE L", system)

    def test_extractor_takes_the_query_not_the_outputs_line(self):
        from src.generation.candidate_generator import CandidateGenerator
        generator = CandidateGenerator.__new__(CandidateGenerator)
        sql = generator._extract_sql(ANSWER)
        self.assertTrue(sql.upper().startswith("SELECT"))
        self.assertNotIn("outputs:", sql)


if __name__ == "__main__":
    unittest.main(verbosity=2)
