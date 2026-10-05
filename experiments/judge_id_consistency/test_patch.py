"""Tests for the judge id/sql reconciliation.

  python -m experiments.judge_id_consistency.test_patch
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.judge_id_consistency.patch import reconcile


class C:
    def __init__(self, cid, sql):
        self.candidate_id, self.sql = cid, sql


class R:
    def __init__(self, cid, sql, success=True):
        self.candidate_id, self.sql, self.success = cid, sql, success


CANDS = [C(i, f"SELECT col{i} FROM t") for i in (1, 2, 3, 4, 5)]


class ReconcileTest(unittest.TestCase):
    def test_consistent_pair_is_untouched(self):
        self.assertEqual((3, "SELECT col3 FROM t", "ok"),
                         reconcile(3, "SELECT col3 FROM t", CANDS, []))

    def test_mismatch_corrects_the_id_to_the_sql(self):
        """The Q101 case: judge returned candidate 5's SQL with selected_id=3."""
        cid, sql, action = reconcile(3, "SELECT col5 FROM t", CANDS, [])
        self.assertEqual(5, cid)
        self.assertEqual("SELECT col5 FROM t", sql)
        self.assertEqual("id_corrected", action)

    def test_whitespace_differences_do_not_count_as_mismatch(self):
        self.assertEqual("ok", reconcile(3, "SELECT   col3\nFROM t", CANDS, [])[2])

    def test_trailing_semicolon_does_not_count_as_mismatch(self):
        self.assertEqual("ok", reconcile(3, "SELECT col3 FROM t;", CANDS, [])[2])

    def test_unmatched_sql_falls_back_to_the_id(self):
        cid, sql, action = reconcile(3, "SELECT something_invented", CANDS, [])
        self.assertEqual(3, cid)
        self.assertEqual("SELECT col3 FROM t", sql)
        self.assertEqual("sql_replaced", action)

    def test_empty_sql_is_left_alone(self):
        self.assertEqual("ok", reconcile(3, "", CANDS, [])[2])

    def test_executed_sql_takes_precedence_over_candidate_text(self):
        """The executor's retry loop can rewrite a candidate; the judge sees the executed text."""
        ex = [R(4, "SELECT col4 FROM t WHERE x > 0")]
        cid, _, action = reconcile(1, "SELECT col4 FROM t WHERE x > 0", CANDS, ex)
        self.assertEqual(4, cid)
        self.assertEqual("id_corrected", action)

    def test_failed_execution_results_are_ignored(self):
        ex = [R(4, "SELECT col4 FROM t WHERE x > 0", success=False)]
        self.assertEqual("sql_replaced",
                         reconcile(1, "SELECT col4 FROM t WHERE x > 0", CANDS, ex)[2])

    def test_duplicate_sql_keeps_the_given_id_when_it_is_one_of_them(self):
        """Two candidates share the SQL and the given id is one of them - already consistent."""
        dup = [C(1, "SELECT a"), C(2, "SELECT a"), C(3, "SELECT b")]
        self.assertEqual((2, "SELECT a", "ok"), reconcile(2, "SELECT a", dup, []))

    def test_duplicate_sql_picks_a_match_when_the_id_is_not_one_of_them(self):
        dup = [C(1, "SELECT a"), C(2, "SELECT a"), C(3, "SELECT b")]
        cid, _, action = reconcile(3, "SELECT a", dup, [])
        self.assertIn(cid, (1, 2))
        self.assertEqual("ambiguous_ok", action)

    def test_unknown_id_with_unmatched_sql_is_reported_unresolved(self):
        self.assertEqual("unresolved", reconcile(99, "SELECT nothing", CANDS, [])[2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
