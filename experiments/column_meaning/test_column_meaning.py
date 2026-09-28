"""Tests for the column-meaning corpus, retriever and patch.

  python -m experiments.column_meaning.test_column_meaning
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.column_meaning import corpus, pipeline_patch, retriever

DB = "california_schools"
HAS_DOCS = bool(corpus.columns(DB))

SCHEMA = {
    "frpm": {"table_readable_name": "Free or reduced price meals",
             "columns": [{"name": "CDSCode"}, {"name": "School Name"},
                         {"name": "Free Meal Count (Ages 5-17)"}, {"name": "FRPM Count (Ages 5-17)"}]},
    "satscores": {"table_readable_name": "SAT scores",
                  "columns": [{"name": "cds"}, {"name": "sname"}, {"name": "AvgScrRead"}]},
    "schools": {"table_readable_name": "Schools",
                "columns": [{"name": "CDSCode"}, {"name": "School"}, {"name": "City"},
                            {"name": "Website"}]},
}


class FakeClient:
    def __init__(self):
        self.prompts = []

    def complete(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return '{"relevant": true, "score": 1.0, "reason": "test", "relevant_columns": ["CDSCode"]}'


@unittest.skipUnless(HAS_DOCS, "database descriptions not available")
class CorpusTest(unittest.TestCase):
    def test_every_documented_column_has_searchable_text(self):
        rows = corpus.columns(DB)
        self.assertTrue(all(row["text"].strip() for row in rows))
        self.assertTrue(any(row["meaning"] for row in rows))      # column_meaning.json merged in

    def test_table_filter(self):
        rows = corpus.columns(DB, ("frpm",))
        self.assertEqual({"frpm"}, {row["table"] for row in rows})

    def test_unknown_database_is_empty_not_an_error(self):
        self.assertEqual([], corpus.columns("no_such_database"))


@unittest.skipUnless(HAS_DOCS, "database descriptions not available")
class RetrieverTest(unittest.TestCase):
    def test_wording_finds_the_described_column(self):
        hits = retriever.retrieve(DB, "How many test takers are there at the school in Fremont?")
        self.assertTrue(hits)
        self.assertIn("satscores.NumTstTakr", [f"{h['table']}.{h['column']}" for h in hits])

    def test_named_column_is_found_although_its_words_are_everywhere(self):
        # "School" shares its only word with every document, so BM25 alone never ranks it.
        hits = retriever.retrieve(DB, "List the names of schools in Alameda")
        named = [h for h in hits if h["match"] == "name"]
        self.assertIn("schools.School", [f"{h['table']}.{h['column']}" for h in named])

    def test_similar_columns_are_shown_with_their_descriptions(self):
        hits = retriever.retrieve(DB, "What is the free meal count for students aged 5 to 17?")
        names = [f"{h['table']}.{h['column']}" for h in hits]
        self.assertIn("frpm.Free Meal Count (Ages 5-17)", names)
        block = retriever.format_block(hits)
        self.assertIn("# Column meanings", block)
        self.assertTrue(any(line.count(":") >= 1 for line in block.splitlines()[2:]))

    def test_table_filter_limits_the_result(self):
        hits = retriever.retrieve(DB, "average reading score", tables=("satscores",))
        self.assertEqual({"satscores"}, {h["table"] for h in hits})

    def test_limit_applies_to_the_ranked_part(self):
        hits = retriever.retrieve(DB, "free meal count and enrollment", limit=3)
        self.assertLessEqual(sum(1 for h in hits if h["match"] == "meaning"), 3)

    def test_empty_question_gives_no_block(self):
        self.assertEqual("", retriever.format_block(retriever.retrieve(DB, "")))


@unittest.skipUnless(HAS_DOCS, "database descriptions not available")
class PatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not pipeline_patch.install():
            raise unittest.SkipTest("src.agents not importable, or another retrieval patch is active")

    def _run_workers(self, question, evidence=""):
        from src.agents.manager import SchemaManager
        client = FakeClient()
        manager = SchemaManager(llm_client=client)
        decomposed = manager._heuristic_decompose(question)
        decomposed.original_query = question
        manager.coordinate_workers(decomposed_query=decomposed, schema=SCHEMA, evidence=evidence)
        return client.prompts

    def test_workers_see_all_three_blocks(self):
        prompts = self._run_workers(
            "What is the free meal count for students aged 5-17 in the schools of Fresno?")
        self.assertTrue(prompts)
        for prompt in prompts:
            self.assertIn("# Column meanings", prompt)
        self.assertTrue(any("# Matched contents" in p for p in prompts))
        self.assertTrue(any("# Join columns" in p for p in prompts))

    def test_original_prompt_is_kept_and_blocks_are_appended(self):
        prompts = self._run_workers("How many schools are in Fresno?")
        for prompt in prompts:
            self.assertIn("Return ONLY JSON:", prompt)
            self.assertLess(prompt.index("Return ONLY JSON:"), prompt.index("# "))

    def test_context_is_cleared_after_the_call(self):
        self._run_workers("How many schools are in Fresno?")
        self.assertIsNone(pipeline_patch._CURRENT["context"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
