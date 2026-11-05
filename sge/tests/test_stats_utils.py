import itertools
import math
import unittest

import numpy as np

from examples.stats_utils import (bca_ci, f_sf, friedman_iman_davenport, holm, rank_biserial,
                                  rankdata, spearman_perm, wilcoxon_exact)


def brute_force_wilcoxon(d):
    d = np.asarray([v for v in d if v != 0], dtype=float)
    r = rankdata(np.abs(d))
    w = r[d > 0].sum()
    ws = [sum(r[i] for i in range(len(d)) if s[i]) for s in itertools.product([0, 1], repeat=len(d))]
    ws = np.array(ws)
    return min(1.0, 2 * min(np.mean(ws <= w + 1e-9), np.mean(ws >= w - 1e-9)))


class TestStats(unittest.TestCase):
    def test_rankdata_ties(self):
        np.testing.assert_allclose(rankdata([10, 20, 20, 30]), [1, 2.5, 2.5, 4])

    def test_wilcoxon_matches_brute_force(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            d = np.round(rng.normal(0.3, 1, size=rng.integers(4, 11)), 1)  # con empates
            if np.all(d == 0):
                continue
            self.assertAlmostEqual(wilcoxon_exact(d)[1], brute_force_wilcoxon(d), places=10)

    def test_wilcoxon_known_value(self):
        # n = 6, todas positivas: p exacto bilateral = 2 / 2^6
        self.assertAlmostEqual(wilcoxon_exact([1, 2, 3, 4, 5, 6])[1], 2 / 64)

    def test_wilcoxon_handles_inf(self):
        w, p, n = wilcoxon_exact([1, 2, math.inf, 4, 5, 6])
        self.assertEqual(n, 6)
        self.assertAlmostEqual(p, 2 / 64)

    def test_wilcoxon_one_sided(self):
        d = [-1, -2, -3, -4, -5, -6]
        self.assertAlmostEqual(wilcoxon_exact(d, "less")[1], 1 / 64)
        self.assertAlmostEqual(wilcoxon_exact(d, "greater")[1], 1.0)
        self.assertAlmostEqual(wilcoxon_exact(d)[1], 2 / 64)

    def test_holm(self):
        np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_f_distribution(self):
        # Valores críticos de tablas: F(2,10) al 5 % = 4,103; F(3,15) al 5 % = 3,287; F(1,1) en 1: p = 0,5
        self.assertAlmostEqual(f_sf(4.1028, 2, 10), 0.05, places=3)
        self.assertAlmostEqual(f_sf(3.2874, 3, 15), 0.05, places=3)
        self.assertAlmostEqual(f_sf(1.0, 1, 1), 0.5, places=6)

    def test_friedman(self):
        # Caso límite: orden idéntico en todos los bloques
        X = [[1, 2, 3, 4]] * 6
        r = friedman_iman_davenport(X)
        self.assertEqual(r["mean_ranks"], [1, 2, 3, 4])
        self.assertAlmostEqual(r["chi2_friedman"], 18.0)
        self.assertEqual(r["p"], 0.0)          # F infinita: separación perfecta
        r = friedman_iman_davenport([[1, 2], [2, 1], [1, 2], [2, 1]])
        self.assertAlmostEqual(r["chi2_friedman"], 0.0)
        self.assertAlmostEqual(r["p"], 1.0)

    def test_rank_biserial(self):
        self.assertEqual(rank_biserial([1, 2, 3]), 1.0)
        self.assertEqual(rank_biserial([-1, -2, -3]), -1.0)
        self.assertAlmostEqual(rank_biserial([1, -1]), 0.0)

    def test_bca_covers_and_reports_method(self):
        rng = np.random.default_rng(2)
        x = rng.normal(5, 1, 40)
        lo, hi, m = bca_ci(x, n_boot=4000)
        self.assertEqual(m, "bca")
        self.assertLess(lo, np.median(x))
        self.assertGreater(hi, np.median(x))
        self.assertEqual(bca_ci([1, 2, math.inf, 4], n_boot=500)[2], "percentile")

    def test_spearman(self):
        x = np.arange(20)
        rho, p, n = spearman_perm(x, x ** 2, n_perm=2000)
        self.assertAlmostEqual(rho, 1.0)
        self.assertLess(p, 0.01)
        rho, p, n = spearman_perm([1, 2, math.inf, 4], [1, 2, 3, 4], n_perm=200)
        self.assertEqual(n, 3)


if __name__ == "__main__":
    unittest.main()
