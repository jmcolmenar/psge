import math
import unittest

from examples.invalid_rule import (INVALID_THRESHOLD, agg_evo, clean, count_invalid,
                                   is_invalid, paired_diff, rel_diff_pct)

SENTINEL_NRMSE = 9999999   # calculate_dataset_nrmse(invalid_fitness)


class TestInvalidRule(unittest.TestCase):
    def test_r1_detects_sentinels_and_non_finite(self):
        for v in (SENTINEL_NRMSE, 1e6, INVALID_THRESHOLD, math.inf, math.nan, None, "x"):
            self.assertTrue(is_invalid(v), v)
        for v in (0.0, 23.84, 14.35, INVALID_THRESHOLD * 0.999):
            self.assertFalse(is_invalid(v), v)

    def test_r2_invalid_becomes_inf_not_dropped(self):
        self.assertEqual(clean(SENTINEL_NRMSE), math.inf)
        self.assertEqual(clean(0.5), 0.5)

    def test_r3_median_absorbs_one_failure_in_three(self):
        # Caso real: nguyen1, lam1_bet2, semilla de datos 402
        self.assertAlmostEqual(agg_evo([0.105, 6.13e-17, SENTINEL_NRMSE]), 0.105)
        self.assertEqual(agg_evo([0.1, SENTINEL_NRMSE, SENTINEL_NRMSE]), math.inf)

    def test_r4_paired_diff(self):
        self.assertEqual(paired_diff(math.inf, math.inf), 0.0)
        self.assertEqual(paired_diff(math.inf, 1.0), math.inf)
        self.assertEqual(paired_diff(1.0, math.inf), -math.inf)
        self.assertAlmostEqual(paired_diff(0.3, 0.5), -0.2)

    def test_r5_relative_diff(self):
        self.assertEqual(rel_diff_pct(math.inf, math.inf), 0.0)
        self.assertEqual(rel_diff_pct(math.inf, 0.5), math.inf)
        self.assertEqual(rel_diff_pct(0.5, math.inf), -100.0)
        self.assertAlmostEqual(rel_diff_pct(0.55, 0.5), 10.0)
        self.assertIsNone(rel_diff_pct(0.1, 0.0))

    def test_r6_count(self):
        self.assertEqual(count_invalid([0.1, SENTINEL_NRMSE, math.nan, 2.0]), 2)


if __name__ == "__main__":
    unittest.main()
