"""Tests for the R-VES port.

  python -m experiments.rves.test_rves

The bucket boundaries and the aggregation are checked against BIRD's source values, including
the awkward ones: the ceiling is sqrt(1.25)*100, not 125, and the denominator counts wrong
answers.
"""
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.rves import rves


class RewardBucketTest(unittest.TestCase):
    def test_zero_ratio_is_zero_reward(self):
        self.assertEqual(0.0, rves.reward_for(0))

    def test_boundaries_are_inclusive_below(self):
        # BIRD uses >=, so each boundary belongs to the FASTER bucket
        for ratio, expected in ((2.0, 1.25), (1.0, 1.0), (0.5, 0.75), (0.25, 0.5)):
            self.assertEqual(expected, rves.reward_for(ratio), ratio)

    def test_just_under_each_boundary_drops_a_bucket(self):
        for ratio, expected in ((1.999, 1.0), (0.999, 0.75), (0.499, 0.5), (0.249, 0.25)):
            self.assertEqual(expected, rves.reward_for(ratio), ratio)

    def test_very_fast_and_very_slow(self):
        self.assertEqual(1.25, rves.reward_for(50.0))
        self.assertEqual(0.25, rves.reward_for(0.001))


class AggregationTest(unittest.TestCase):
    def test_single_perfect_query(self):
        self.assertAlmostEqual(100.0, rves.compute_rves([1.0]))

    def test_ceiling_is_sqrt_of_125(self):
        self.assertAlmostEqual(math.sqrt(1.25) * 100, rves.compute_rves([1.25]))
        self.assertAlmostEqual(111.803, rves.compute_rves([1.25]), places=3)

    def test_wrong_answers_are_in_the_denominator(self):
        # one perfect, one wrong -> 50, not 100
        self.assertAlmostEqual(50.0, rves.compute_rves([1.0, 0.0]))

    def test_sqrt_is_applied_to_the_reward(self):
        self.assertAlmostEqual(math.sqrt(0.5) * 100, rves.compute_rves([0.5]))
        self.assertAlmostEqual(70.711, rves.compute_rves([0.5]), places=3)

    def test_empty(self):
        self.assertEqual(0.0, rves.compute_rves([]))

    def test_rves_equals_accuracy_when_every_correct_query_is_par(self):
        rewards = [1.0] * 70 + [0.0] * 30
        self.assertAlmostEqual(70.0, rves.compute_rves(rewards))


class OutlierTest(unittest.TestCase):
    def test_clean_abnormal_drops_a_wild_point(self):
        vals = [1.0] * 20 + [500.0]
        self.assertNotIn(500.0, rves.clean_abnormal(vals))

    def test_clean_abnormal_keeps_a_tight_cluster(self):
        vals = [1.0, 1.01, 0.99, 1.02]
        self.assertEqual(len(vals), len(rves.clean_abnormal(vals)))


class WrapperTest(unittest.TestCase):
    def test_strips_the_bird_delimiter(self):
        self.assertEqual("SELECT 1", rves.sql_of("SELECT 1" + rves.DELIM + "financial"))

    def test_leaves_a_bare_query_alone(self):
        self.assertEqual("SELECT 1", rves.sql_of("  SELECT 1  "))


class EndToEndTest(unittest.TestCase):
    """score_one against a real throwaway SQLite file."""

    @classmethod
    def setUpClass(cls):
        import sqlite3
        import tempfile
        cls.tmp = tempfile.mkdtemp()
        cls.db = str(Path(cls.tmp) / "t.sqlite")
        conn = sqlite3.connect(cls.db)
        conn.execute("CREATE TABLE t (a INTEGER, b TEXT)")
        conn.executemany("INSERT INTO t VALUES (?, ?)", [(i, "x%d" % i) for i in range(200)])
        conn.commit()
        conn.close()

    def test_matching_rows_score_above_zero(self):
        reward, ratio, status = rves.score_one(
            "SELECT a FROM t WHERE a < 10", "SELECT a FROM t WHERE a < 10", self.db, iterate_num=3)
        self.assertEqual("ok", status)
        self.assertGreater(reward, 0.0)
        self.assertGreater(ratio, 0.0)

    def test_row_order_does_not_matter(self):
        reward, _, status = rves.score_one(
            "SELECT a FROM t WHERE a < 10 ORDER BY a DESC",
            "SELECT a FROM t WHERE a < 10", self.db, iterate_num=3)
        self.assertEqual("ok", status)
        self.assertGreater(reward, 0.0)

    def test_wrong_rows_score_zero(self):
        reward, ratio, status = rves.score_one(
            "SELECT a FROM t WHERE a < 5", "SELECT a FROM t WHERE a < 10", self.db, iterate_num=3)
        self.assertEqual(0.0, reward)
        self.assertEqual(0.0, ratio)

    def test_broken_sql_scores_zero_without_raising(self):
        reward, _, status = rves.score_one(
            "SELECT nope FROM nowhere", "SELECT a FROM t", self.db, iterate_num=3)
        self.assertEqual(0.0, reward)
        self.assertTrue(status.startswith("error"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
