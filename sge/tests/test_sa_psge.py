"""
Sanity tests for the SA-PSGE fixes (session 2026-11-03):
  1. SA_ETA decoupled from LEARNING_FACTOR; grammar actually moves.
  2. Small uniform mixture (alpha) no longer anchors the PCFG at uniform.
  4. Validation set generation; test set optional and untouched in tuning.
  6. Dedicated RNG for SA-PSGE views (global np.random stream untouched).
"""
import copy
import os
import shutil
import tempfile
import unittest

import numpy as np

import sge.grammar as grammar
from sge.parameters import params
import importlib
sau = importlib.import_module("sge.operators.stability_aware_update")
from sge.stability import (
    generate_add_one_datasets,
    calculate_stability_metrics,
    calculate_pcfg_entropy,
    StabilityEvaluationFunction,
    TEST_SEED_OFFSET,
)

GRAMMAR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "grammars", "regression_1v.pybnf")


def _setup_grammar(seed):
    import sge.engine as engine
    tmp = tempfile.mkdtemp(prefix="sa_test_")
    opts = dict(params)
    opts.update({"GRAMMAR": GRAMMAR, "SEED": seed, "EXPERIMENT_NAME": tmp, "RUN": 1,
                 "MAX_TREE_DEPTH": 10, "MIN_TREE_DEPTH": 4, "POPSIZE": 60,
                 "VERBOSE": False, "PARAMETERS": None})
    grammar.reset()
    engine.setup(options_dict=opts)
    return engine, tmp


def _population(engine, eval_func, n):
    pop = []
    for _ in range(n):
        ind = engine.generate_random_individual()
        engine.evaluate(ind, eval_func)
        pop.append(ind)
    pop.sort(key=lambda i: i["fitness"])
    return pop


class TestDatasets(unittest.TestCase):

    def test_val_set_independent_and_test_unchanged(self):
        b = generate_add_one_datasets("nguyen5", m=20, seed_data=7, n_test=50, n_val=40)
        self.assertEqual(len(b["val_set"]), 40)
        self.assertEqual(len(b["test_set"]), 50)
        # val and test are different draws
        self.assertNotEqual(b["val_set"][0], b["test_set"][0])
        # test stream identical to the previous implementation
        rng = np.random.RandomState(7 + TEST_SEED_OFFSET)
        x = rng.uniform(-1.0, 1.0, size=50)
        self.assertAlmostEqual(b["test_set"][0][0], float(x[0]), places=12)

    def test_tuning_mode_without_test(self):
        b = generate_add_one_datasets("nguyen5", m=20, seed_data=7, n_test=0, n_val=40)
        self.assertEqual(b["test_set"], [])
        met = calculate_stability_metrics("x[0]", "x[0]*x[0]", b["S"], b["z"],
                                          test_set=None, var_y=b["var_y_S"], val_set=b["val_set"])
        self.assertIsNone(met["nrmse_test_fS"])
        self.assertIsNotNone(met["nrmse_val_fS"])
        self.assertGreater(met["nrmse_val_fS"], 0.0)

    def test_eval_function_does_not_touch_test_by_default(self):
        b = generate_add_one_datasets("nguyen5", m=20, seed_data=7, n_test=50)
        f = StabilityEvaluationFunction(dataset=b["S"], test_set=b["test_set"])
        _, info = f.evaluate("x[0]")
        self.assertTrue(np.isnan(info["test_error"]))
        f2 = StabilityEvaluationFunction(dataset=b["S"], test_set=b["test_set"], track_test_online=True)
        _, info2 = f2.evaluate("x[0]")
        self.assertFalse(np.isnan(info2["test_error"]))


class TestSAPSGE(unittest.TestCase):

    def setUp(self):
        self.engine, self.tmp = _setup_grammar(seed=123)
        b = generate_add_one_datasets("nguyen5", m=20, seed_data=3, n_test=0)
        self.eval_func = StabilityEvaluationFunction(dataset=b["S"])
        self.pop = _population(self.engine, self.eval_func, 60)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run_updates(self, eta, alpha, n=20):
        grammar.get_pcfg()[:] = self._uniform()
        for _ in range(n):
            sau.stability_aware_update(self.pop, self.eval_func, eta=eta, alpha=alpha)
        return calculate_pcfg_entropy(grammar.get_pcfg(), grammar.get_mask())

    def _uniform(self):
        g = copy.deepcopy(grammar.get_pcfg())
        mask = grammar.get_mask()
        for i in range(g.shape[0]):
            k = int(np.count_nonzero(mask[i]))
            g[i] = np.where(mask[i], 1.0 / k, 0.0)
        return g

    def test_old_defaults_anchor_at_uniform(self):
        """Regression: eta=0.01, alpha=0.05 keeps the PCFG ~uniform (SA-PSGE ~ SGE)."""
        h = self._run_updates(eta=0.01, alpha=0.05)
        self.assertGreater(h, 0.999)

    def test_new_setting_moves_grammar(self):
        h = self._run_updates(eta=1.0, alpha=0.005)
        self.assertLess(h, 0.99)

    def test_rows_remain_distributions_with_floor(self):
        self._run_updates(eta=5.0, alpha=0.0, n=50)
        g, mask = grammar.get_pcfg(), grammar.get_mask()
        for i in range(g.shape[0]):
            valid = mask[i].astype(bool)
            if valid.sum() <= 1:
                continue
            self.assertAlmostEqual(float(g[i, valid].sum()), 1.0, places=9)
            self.assertGreaterEqual(float(g[i, valid].min()), 1e-3 / (1 + 1e-3 * valid.sum()) - 1e-12)

    def test_views_do_not_consume_global_rng(self):
        np.random.seed(999)
        expected = np.random.uniform(size=5)
        np.random.seed(999)
        sau.stability_aware_update(self.pop, self.eval_func, eta=1.0, alpha=0.005)
        got = np.random.uniform(size=5)
        np.testing.assert_array_equal(expected, got)

    def test_sa_rng_reset_is_deterministic(self):
        sau.reset_sa_rng(42); a = sau.get_sa_rng().integers(0, 10**9, size=4)
        sau.reset_sa_rng(42); b = sau.get_sa_rng().integers(0, 10**9, size=4)
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
