import math
import re
import unittest

import numpy as np

from sge.stability import BENCHMARK_SPECS, generate_add_one_datasets


def grammar_vars(path):
    with open(path) as fh:
        return sorted(set(int(i) for i in re.findall(r"x\[(\d+)\]", fh.read())))


class TestBenchmarks(unittest.TestCase):
    def test_grammar_covers_all_variables(self):
        # Una gramática que no puede nombrar todas las variables hace el problema irresoluble.
        for name, spec in BENCHMARK_SPECS.items():
            self.assertEqual(grammar_vars(spec["grammar"]), list(range(spec["n_vars"])), name)

    def test_grammars_share_rules_except_variables(self):
        def rules(path):
            with open(path) as fh:
                return [l for l in fh.read().splitlines() if l and not l.startswith("<var>")]
        ref = rules("grammars/regression_1v.pybnf")
        for spec in BENCHMARK_SPECS.values():
            self.assertEqual(rules(spec["grammar"]), ref)

    def test_vladislavleva_domains(self):
        for name, d in (("vladislavleva4", 5), ("vladislavleva7", 2)):
            b = generate_add_one_datasets(name, m=100, seed_data=1, n_test=2000, n_val=0)
            xs = np.array([p[:-1] for p in b["S_plus_z"]])
            xt = np.array([p[:-1] for p in b["test_set"]])
            self.assertEqual(xs.shape[1], d)
            self.assertTrue(xs.min() >= 0.05 and xs.max() <= 6.05, name)
            self.assertTrue(xt.min() < 0.05 and xt.max() > 6.05, name)   # test incluye extrapolación
            self.assertTrue(xt.min() >= -0.25 and xt.max() <= 6.35, name)

    def test_vladislavleva4_value(self):
        f = BENCHMARK_SPECS["vladislavleva4"]["function"]
        self.assertAlmostEqual(f([3, 3, 3, 3, 3]), 2.0)          # 10 / 5
        self.assertAlmostEqual(f([4, 3, 3, 3, 3]), 10 / 6)

    def test_keijzer6_matches_harmonic_numbers(self):
        f = BENCHMARK_SPECS["keijzer6"]["function"]
        for n in range(1, 121):
            self.assertAlmostEqual(f([float(n)]), sum(1.0 / i for i in range(1, n + 1)), places=10)
        # suave y creciente entre enteros (la versión anterior era escalonada)
        xs = np.linspace(1, 120, 2000)
        ys = np.array([f([v]) for v in xs])
        self.assertTrue(np.all(np.diff(ys) > 0))
        self.assertTrue(f([1.0]) < f([1.5]) < f([2.0]))

    def test_keijzer6_domains(self):
        b = generate_add_one_datasets("keijzer6", m=100, seed_data=1, n_test=2000, n_val=0)
        xs = np.array([p[0] for p in b["S_plus_z"]])
        xt = np.array([p[0] for p in b["test_set"]])
        self.assertTrue(xs.min() >= 1 and xs.max() <= 50)
        self.assertTrue(xt.min() < 50 and xt.max() > 50 and xt.max() <= 120)
        self.assertEqual(len(set(xs)), 101)    # m = 100 puntos distintos ya es posible

    def test_pagie_finite_at_zero(self):
        f = BENCHMARK_SPECS["pagie1"]["function"]
        self.assertTrue(math.isfinite(f([0.0, 0.0])))
        self.assertAlmostEqual(f([1.0, 1.0]), 1.0)


if __name__ == "__main__":
    unittest.main()
