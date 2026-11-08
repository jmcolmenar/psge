"""El análisis prerregistrado del estudio de re-mapeo, probado con datos simulados de efecto conocido."""
import math
import unittest

import numpy as np

from examples.analyze_remap_study import analyze, report

LAMBDAS = [0.005, 0.01, 0.02, 0.05, 0.1]
REGIMES = [(20, 0.0), (20, 0.05), (100, 0.0), (100, 0.05)]
BENCH = [f"b{i}" for i in range(10)]


def simulate(lam_effect=0.0, pp_effect=0.0, nrmse_pp_factor=1.0, n_data=6, n_evo=2, seed=0,
             exact_frac=0.0, invalid_frac=0.0, slope_small_m=0.0):
    rng = np.random.default_rng(seed)
    conds = [("sgef", 0, 0.0, False)] + [(f"psge_lf{l:g}", i + 1, l, False) for i, l in enumerate(LAMBDAS)] \
        + [(f"psge_pp_lf{l:g}", i + 1, l, True) for i, l in enumerate(LAMBDAS)]
    rows = []
    for b_i, b in enumerate(BENCH):
        base = math.exp(rng.normal(-2, 1))
        for m, noise in REGIMES:
            for sd in range(n_data):
                data_eff = rng.normal(0, 0.5)
                for name, level, lam, is_pp in conds:
                    for se in range(n_evo):
                        slope = lam_effect + (slope_small_m if m == 20 else 0.0)
                        log_s = math.log(base) + data_eff + slope * level + (pp_effect if is_pp else 0.0) \
                            + rng.normal(0, 0.3)
                        s = math.exp(log_s)
                        if rng.uniform() < exact_frac:
                            s = 1e-30
                        nrmse = math.exp(rng.normal(-1, 0.2)) * (nrmse_pp_factor if is_pp else 1.0)
                        if rng.uniform() < invalid_frac:
                            s, nrmse = 9999999.0, 9999999.0
                        remap = 0.0 if (is_pp or level == 0) else 0.1 * level
                        rows.append({
                            "condition": name, "benchmark": b, "m": m, "noise": noise, "seed_data": 2000 + sd,
                            "seed_evo": 20000 + se, "apg_plus_atg": s, "apg_norm": s * 0.4, "atg_norm": s * 0.6,
                            "gen_gap_test_norm_fS": s * 2.0, "nrmse_test_fS": nrmse,
                            "entropy_S": 1.0 - 0.12 * level + rng.normal(0, 0.01),
                            "arm_S": {"trajectory": {"remap_rate": [remap] * 3}},
                            "arm_Sz": {"trajectory": {"remap_rate": [remap] * 3}},
                        })
    return rows


class TestAnalyzeRemap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.null, _ = analyze(simulate(seed=1), sensitivity=False)
        cls.effect, _ = analyze(simulate(lam_effect=0.3, pp_effect=-0.5, seed=2), sensitivity=True)

    def test_null_not_confirmed(self):
        P = self.null["primary"]
        self.assertFalse(P["H-A1"]["confirmed"])
        self.assertFalse(P["H-C1"]["confirmed"])

    def test_effects_confirmed_with_right_size(self):
        P = self.effect["primary"]
        self.assertTrue(P["H-A1"]["confirmed"])
        self.assertTrue(P["H-C1"]["confirmed"])
        self.assertAlmostEqual(P["H-A1"]["median_log_ratio_lmax_vs_sgef"], 1.5, delta=0.25)   # 0,3 x 5 niveles
        self.assertAlmostEqual(P["H-C1"]["median_log_ratio"], -0.5, delta=0.15)
        self.assertEqual(P["H-C1"]["n_benchmarks_pp_lower"], 10)
        self.assertLess(P["H-A1"]["p_holm"], 0.01)
        for v in self.effect["sensitivity_eps"].values():
            self.assertLess(v["H-A1_p_holm"], 0.05)
            self.assertLess(v["H-C1_p_holm"], 0.05)

    def test_holm_is_applied(self):
        P = self.null["primary"]
        ps = sorted([P["H-A1"]["p"], P["H-C1"]["p"]])
        adj = sorted([P["H-A1"]["p_holm"], P["H-C1"]["p_holm"]])
        self.assertAlmostEqual(adj[0], min(1.0, 2 * ps[0]))

    def test_manipulation_checks(self):
        M = self.effect["secondary"]["manipulation"]
        self.assertLess(M["entropy_page_p"], 0.01)
        self.assertLess(M["remap_psge_page_p"], 0.01)
        self.assertTrue(M["remap_pp_ok"])

    def test_non_inferiority(self):
        self.assertTrue(self.effect["secondary"]["NR"]["non_inferior"])        # NRMSE igual
        worse, _ = analyze(simulate(pp_effect=-0.5, nrmse_pp_factor=1.3, seed=3), sensitivity=False)
        self.assertFalse(worse["secondary"]["NR"]["non_inferior"])
        self.assertAlmostEqual(worse["secondary"]["NR"]["mean_rel_pct"], 30.0, delta=5.0)
        self.assertTrue(worse["primary"]["H-C1"]["confirmed"])               # opción A: no condiciona H-C1

    def test_generalization_tracking(self):
        E = self.effect["secondary"]["H-E"]
        self.assertGreater(E["mean_rho"]["apg_plus_atg"], 0.99)               # brecha = 2 x APG+ATG
        self.assertTrue(E["E1_confirmed"])

    def test_h_a3_interaction(self):
        res, _ = analyze(simulate(lam_effect=0.1, slope_small_m=0.3, seed=4), sensitivity=False)
        a3 = res["secondary"]["H-A3"]["small_m_minus_large_m"]
        self.assertLess(a3["p_greater"], 0.01)
        self.assertAlmostEqual(a3["median"], 1.5, delta=0.3)

    def test_exact_solutions_and_invalid_runs(self):
        res, _ = analyze(simulate(lam_effect=0.3, pp_effect=-0.5, exact_frac=0.1, invalid_frac=0.02, seed=5),
                         sensitivity=False)
        self.assertTrue(math.isfinite(res["primary"]["H-A1"]["p"]))
        self.assertTrue(math.isfinite(res["primary"]["H-C1"]["p"]))
        self.assertGreater(sum(v["n"] for v in res["invalid"].values()), 0)
        self.assertTrue(res["primary"]["H-A1"]["confirmed"])
        self.assertIn("H-A1", report(res))

    def test_incomplete_seeds_are_dropped_and_listed(self):
        rows = simulate(seed=6)
        rows = [r for r in rows if not (r["benchmark"] == "b0" and r["m"] == 20 and r["noise"] == 0.0
                                        and r["seed_data"] == 2000 and r["condition"] == "psge_pp_lf0.1"
                                        and r["seed_evo"] == 20001)]
        res, _ = analyze(rows, sensitivity=False)
        self.assertEqual(res["incomplete_data_seeds"], ["b0/m20_noise0/2000"])
        self.assertEqual(res["data_seeds_per_cell"]["b0/m20_noise0"], 5)


if __name__ == "__main__":
    unittest.main()
