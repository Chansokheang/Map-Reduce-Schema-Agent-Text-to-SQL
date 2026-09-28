"""Tests for the projection review (no model calls: a stub client returns fixed indices).

  python -m experiments.projection_review.test_projection_review
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.projection_review import review as R
from experiments.projection_review.prompt import build_payload

DELIM = "\t----- bird -----\t"
SQL = "SELECT s.School, s.MailStreet FROM schools s WHERE s.Charter = 1"


class StubClient:
    """Returns a preset answer; records the payloads it was given."""

    model = "stub"

    def __init__(self, answer):
        self.answer, self.payloads = answer, []

    def complete(self, payload):
        self.payloads.append(payload)
        return dict(self.answer)


class ParsingTest(unittest.TestCase):
    def test_outer_select_items(self):
        tree, items = R.outer_select(SQL)
        self.assertEqual(["s.School", "s.MailStreet"], R.item_texts(items))

    def test_non_select_is_rejected(self):
        self.assertEqual((None, None), R.outer_select("UPDATE schools SET School = 'x'"))

    def test_unparseable_sql_is_rejected(self):
        self.assertEqual((None, None), R.outer_select("SELECT FROM WHERE ("))

    def test_reorder_keeps_every_other_clause(self):
        tree, items = R.outer_select(SQL)
        out = R.apply_keep(tree, items, [1, 0])
        self.assertIn("s.MailStreet, s.School", out)
        self.assertIn("WHERE s.Charter = 1", out)

    def test_subset_drops_an_item(self):
        tree, items = R.outer_select(SQL)
        self.assertNotIn("MailStreet", R.apply_keep(tree, items, [0]))


class ValidationTest(unittest.TestCase):
    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            R.validate_keep([], 2)

    def test_rejects_duplicates(self):
        with self.assertRaises(ValueError):
            R.validate_keep([0, 0], 2)

    def test_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            R.validate_keep([0, 5], 2)

    def test_accepts_permutation_and_subset(self):
        self.assertEqual([1, 0], R.validate_keep([1, 0], 2))
        self.assertEqual([1], R.validate_keep([1], 2))


class RunTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.selected = self.dir / "selected.json"
        self.questions = self.dir / "questions.json"
        self.selected.write_text(json.dumps({"0": SQL + DELIM + "california_schools"}), encoding="utf-8")
        self.questions.write_text(json.dumps(
            [{"question": "What is the postal street address of the chartered schools? "
                          "Indicate the school name.", "evidence": "", "db_id": "california_schools",
              "SQL": ""}]), encoding="utf-8")

    def _run(self, answer):
        client = StubClient(answer)
        report = R.review(self.selected, self.questions, self.dir / "out", client,
                          workers=1, log=lambda *a: None)
        result = json.loads((self.dir / "out" / "selected_projection_reviewed.json").read_text(encoding="utf-8"))
        return report, result["0"].rsplit(DELIM, 1)[0], client

    def test_reorder_is_applied_and_recorded(self):
        report, sql, client = self._run({"keep": [1, 0], "reason": "address first"})
        self.assertIn("s.MailStreet, s.School", sql)
        self.assertEqual(["0"], report["changed_keys"])
        self.assertEqual({"reordered": 1}, report["statuses"])

    def test_keeping_everything_leaves_the_sql_alone(self):
        report, sql, _ = self._run({"keep": [0, 1]})
        self.assertEqual(SQL, sql)
        self.assertEqual([], report["changed_keys"])

    def test_bad_answer_keeps_the_original(self):
        report, sql, _ = self._run({"keep": [0, 7]})
        self.assertEqual(SQL, sql)
        self.assertEqual({"failed": 1}, report["statuses"])

    def test_source_file_is_untouched(self):
        before = self.selected.read_bytes()
        self._run({"keep": [1, 0]})
        self.assertEqual(before, self.selected.read_bytes())

    def test_payload_carries_question_and_items(self):
        _, _, client = self._run({"keep": [0, 1]})
        payload = client.payloads[0]
        self.assertIn("postal street address", payload["question"])
        self.assertEqual(["s.School", "s.MailStreet"], [i["sql"] for i in payload["select_items"]])

    def test_second_run_reuses_the_checkpoint(self):
        self._run({"keep": [1, 0]})
        client = StubClient({"keep": [0, 1]})
        R.review(self.selected, self.questions, self.dir / "out", client, workers=1, log=lambda *a: None)
        self.assertEqual([], client.payloads)          # cached, no second call


class PayloadTest(unittest.TestCase):
    def test_missing_evidence_is_labelled(self):
        payload = build_payload({"question": "q"}, ["a"])
        self.assertEqual("(none)", payload["evidence"])




class NoGoldTest(unittest.TestCase):
    """The review must never see the reference SQL, even in memory or in its logs."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.questions = self.dir / "questions.json"
        self.questions.write_text(json.dumps([
            {"question_id": 0, "question": "List the school name and its phone.",
             "evidence": "", "db_id": "california_schools",
             "SQL": "SELECT SECRET_GOLD_MARKER FROM schools"}]), encoding="utf-8")

    def test_loaded_questions_carry_no_gold(self):
        loaded = R.load_questions(self.questions)
        self.assertEqual({"question", "evidence", "db_id"}, set(loaded["0"]))
        self.assertNotIn("SQL", loaded["0"])

    def test_payload_carries_no_gold(self):
        loaded = R.load_questions(self.questions)
        payload = build_payload(loaded["0"], ["a", "b"])
        self.assertNotIn("SECRET_GOLD_MARKER", json.dumps(payload))

    def test_records_written_to_disk_carry_no_gold(self):
        selected = self.dir / "selected.json"
        selected.write_text(json.dumps({"0": SQL + DELIM + "california_schools"}), encoding="utf-8")
        client = StubClient({"keep": [1, 0]})
        R.review(selected, self.questions, self.dir / "out", client, workers=1, log=lambda *a: None)
        for path in (self.dir / "out").rglob("*.json"):
            self.assertNotIn("SECRET_GOLD_MARKER", path.read_text(encoding="utf-8"), path.name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
