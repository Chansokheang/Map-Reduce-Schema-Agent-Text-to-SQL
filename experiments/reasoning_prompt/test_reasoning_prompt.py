"""Tests for the reasoning-prompt arm.

  python -m experiments.reasoning_prompt.test_reasoning_prompt
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.reasoning_prompt import pipeline_patch, prompt

SCHEMA = {"schools": {"table_readable_name": "schools",
                      "columns": [{"name": "CDSCode"}, {"name": "School"}, {"name": "FundingType"}]}}

# A realistic reasoning answer: prose, an intermediate query, then the final one.
ANSWER = """Step 1: Decompose. The question asks for the oldest driver's win total and full name.
Step 2: Map. wins lives in driverStandings, forename/surname in drivers.
Step 3: The oldest driver is a ranking, so ORDER BY dob ASC LIMIT 1.

```sql
SELECT driverId FROM drivers ORDER BY dob ASC LIMIT 1
```

That subquery returns the oldest driver. The final query:

```sql
SELECT SUM(ds.wins), d.forename, d.surname
FROM drivers d JOIN driverStandings ds ON d.driverId = ds.driverId
ORDER BY d.dob ASC LIMIT 1
```
"""


class TextTest(unittest.TestCase):
    def test_reasoning_section_added_once(self):
        first = prompt.with_reasoning("BASE\n1. Generate ONLY the SQL query, no explanations")
        self.assertIn(prompt.MARKER, first)
        self.assertEqual(first, prompt.with_reasoning(first))

    def test_contradicting_instructions_are_rewritten(self):
        base = ("rules\n1. Generate ONLY the SQL query, no explanations\n"
                "9. Return only the SQL query, nothing else")
        out = prompt.with_reasoning(base)
        self.assertNotIn("Generate ONLY the SQL query, no explanations", out)
        self.assertNotIn("Return only the SQL query, nothing else", out)
        self.assertTrue(prompt.rewrote_output_instructions(out))

    def test_empty_prompt_untouched(self):
        self.assertEqual("", prompt.with_reasoning(""))


class ExtractionTest(unittest.TestCase):
    def test_last_fenced_query_wins(self):
        sql = pipeline_patch.final_statement(ANSWER)
        self.assertIn("SUM(ds.wins)", sql)
        self.assertNotIn("SELECT driverId FROM drivers ORDER BY dob ASC LIMIT 1", sql)

    def test_shipped_extractor_would_have_taken_the_subquery(self):
        """The reason this patch exists."""
        import re
        first = re.search(r"```(?:sql)?\s*([\s\S]*?)\s*```", ANSWER, re.IGNORECASE).group(1).strip()
        self.assertEqual("SELECT driverId FROM drivers ORDER BY dob ASC LIMIT 1", first)

    def test_no_fence_returns_none(self):
        self.assertIsNone(pipeline_patch.final_statement("no code here"))

    def test_non_query_blocks_are_skipped(self):
        answer = "```sql\nSELECT 1\n```\ntext\n```\nnot a query\n```"
        self.assertEqual("SELECT 1", pipeline_patch.final_statement(answer))

    def test_with_statement_is_accepted(self):
        answer = "```sql\nWITH x AS (SELECT 1) SELECT * FROM x\n```"
        self.assertTrue(pipeline_patch.final_statement(answer).startswith("WITH"))


class PatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not pipeline_patch.install():
            raise unittest.SkipTest("src not importable")

    def test_every_strategy_gets_the_reasoning_section(self):
        from src.generation.prompt_builder import PromptBuilder
        builder = PromptBuilder()
        for strategy in builder.strategy_prompts:
            system = builder.build("q", SCHEMA, strategy, evidence="")["system"]
            self.assertIn(prompt.MARKER, system, strategy)
            self.assertTrue(prompt.rewrote_output_instructions(system), strategy)

    def test_existing_rules_survive(self):
        from src.generation.prompt_builder import PromptBuilder
        builder = PromptBuilder()
        system = builder.build("q", SCHEMA, list(builder.strategy_prompts)[0], evidence="")["system"]
        for kept in ("RULE A", "RULE E", "RULE J", "RULE K"):
            self.assertIn(kept, system)

    def test_generator_extracts_the_final_query(self):
        from src.generation.candidate_generator import CandidateGenerator
        generator = CandidateGenerator.__new__(CandidateGenerator)
        self.assertIn("SUM(ds.wins)", generator._extract_sql(ANSWER))

    def test_generator_still_handles_a_bare_query(self):
        from src.generation.candidate_generator import CandidateGenerator
        generator = CandidateGenerator.__new__(CandidateGenerator)
        self.assertIn("SELECT School", generator._extract_sql("SELECT School FROM schools"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
