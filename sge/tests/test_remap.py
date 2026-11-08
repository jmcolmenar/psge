"""Tests del estudio de adaptación y re-mapeo: PSGE-PP, tasa de re-mapeo, trayectoria, brecha y Page."""
import os
import tempfile
import unittest

import numpy as np

import sge
import sge.engine as eng
import sge.grammar as grammar
from sge.operators import phenotype_preserving as pp
from sge.parameters import params
from sge.stability import (BENCHMARK_SPECS, REMAP_CONFIRM_BENCHMARKS, StabilityEvaluationFunction,
                           calculate_stability_metrics, generate_add_one_datasets)
from examples.stats_utils import page_trend


def setup_grammar(path="grammars/regression.pybnf", max_depth=8, init_depth=5):
    grammar.reset()
    grammar.set_path(path)
    grammar.set_max_tree_depth(max_depth)
    grammar.set_min_init_tree_depth(init_depth)
    grammar.read_grammar()
    params['ADAPTIVE_MUTATION'] = False


def random_pcfg(rng, tiny=False):
    pcfg = np.zeros_like(grammar.get_pcfg())
    mask = grammar.get_mask()
    for i in range(pcfg.shape[0]):
        idx = np.where(mask[i])[0]
        p = rng.dirichlet(np.ones(len(idx)) * (0.3 if tiny else 2.0))
        pcfg[i, idx] = p / p.sum()
    return pcfg


def map_ind(ind, pcfg):
    mv = [0] * len(ind['genotype'])
    phen, depth, counter = grammar.mapping(pcfg, ind['genotype'], mv)
    ind.update(phenotype=phen, tree_depth=depth, grammar_counter=counter, mapping_values=mv)
    return ind


def population(n, pcfg, seed):
    np.random.seed(seed)
    pop = [eng.generate_random_individual() for _ in range(n)]
    return [map_ind(ind, pcfg) for ind in pop]


class TestRuleIntervals(unittest.TestCase):
    def test_select_rule_matches_mapping(self):
        # La réplica de la selección de reglas debe coincidir con la del mapeo en todos los codones usados,
        # en los dos regímenes (normal y límite de profundidad).
        setup_grammar()
        rng = np.random.default_rng(0)
        nts = list(grammar.get_non_terminals())
        limited_seen = 0
        for k in range(20):
            pcfg = random_pcfg(rng, tiny=(k % 2 == 1))
            for ind in population(30, pcfg, seed=k):
                for g, gene in enumerate(ind['genotype']):
                    for rule, codon, depth in gene[:ind['mapping_values'][g]]:
                        limited_seen += int(pp._limited(nts[g], depth))
                        self.assertEqual(pp.select_rule(pcfg, nts[g], codon, depth), rule)
        self.assertGreater(limited_seen, 0)


class TestPhenotypePreservation(unittest.TestCase):
    def test_reencoding_preserves_every_phenotype(self):
        setup_grammar()
        rng = np.random.default_rng(1)
        total_fail_inds = 0
        for k in range(30):
            old, new = random_pcfg(rng, tiny=(k % 3 == 0)), random_pcfg(rng, tiny=(k % 3 == 1))
            pop = population(40, old, seed=100 + k)
            stats = pp.reencode_population(pop, old, new)
            changed = sum(pp.map_copy(ind, new) != ind['phenotype'] for ind in pop)
            # Solo puede cambiar un individuo con algún codon imposible de conservar.
            self.assertLessEqual(changed, stats["individuals_with_failures"])
            if stats["failures"] == 0:
                self.assertEqual(pp.remap_rate(pop, new), 0.0)
            total_fail_inds += stats["individuals_with_failures"]
        self.assertLess(total_fail_inds, 30 * 40)   # los fallos son la excepción

    def test_reencoding_identity_when_grammar_unchanged(self):
        setup_grammar()
        pcfg = random_pcfg(np.random.default_rng(2))
        pop = population(30, pcfg, seed=7)
        before = [[[list(e) for e in g] for g in ind['genotype']] for ind in pop]
        stats = pp.reencode_population(pop, pcfg, pcfg.copy())
        self.assertEqual(stats["failures"], 0)
        for ind, b in zip(pop, before):
            for g, gb in zip(ind['genotype'], b):
                for e, eb in zip(g, gb):
                    self.assertAlmostEqual(e[1], eb[1], places=12)

    def test_without_reencoding_phenotypes_drift(self):
        setup_grammar()
        rng = np.random.default_rng(3)
        old, new = random_pcfg(rng), random_pcfg(rng, tiny=True)
        pop = population(50, old, seed=11)
        self.assertGreater(pp.remap_rate(pop, new), 0.0)

    def test_shared_entries_are_not_mutated(self):
        # Los descendientes pueden compartir entradas del genotipo con sus padres: re-codificar uno no
        # debe tocar al otro.
        setup_grammar()
        rng = np.random.default_rng(4)
        old, new = random_pcfg(rng), random_pcfg(rng)
        a = population(1, old, seed=5)[0]
        b = {k: v for k, v in a.items()}
        b['genotype'] = [g[:] for g in a['genotype']]       # copia superficial: entradas compartidas
        snapshot = [[list(e) for e in g] for g in a['genotype']]
        pp.reencode_individual(b, old, new)
        self.assertEqual([[list(e) for e in g] for g in a['genotype']], snapshot)

    def test_remap_rate_sample_is_deterministic_subset(self):
        setup_grammar()
        rng = np.random.default_rng(6)
        old, new = random_pcfg(rng), random_pcfg(rng, tiny=True)
        pop = population(60, old, seed=17)
        full = pp.remap_rate(pop, new)
        s1, s2 = pp.remap_rate(pop, new, sample=20), pp.remap_rate(pop, new, sample=20)
        self.assertEqual(s1, s2)
        self.assertEqual(pp.remap_rate(pop, new, sample=1000), full)
        self.assertTrue(0.0 <= s1 <= 1.0)

    def test_remap_rate_keeps_rng_state(self):
        setup_grammar()
        rng = np.random.default_rng(5)
        old, new = random_pcfg(rng), random_pcfg(rng, tiny=True)
        pop = population(30, old, seed=13)
        np.random.seed(99)
        state = np.random.get_state()
        pp.remap_rate(pop, new)
        after = np.random.get_state()
        self.assertTrue(all(np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y
                            for x, y in zip(state, after)))


def tiny_run(algorithm, lf, track, seed=123, gens=6):
    b = generate_add_one_datasets("nguyen1", m=20, seed_data=1, n_test=0, n_val=0)
    tmp = tempfile.mkdtemp()
    opts = {"POPSIZE": 40, "GENERATIONS": gens, "ELITISM": 2, "SEED": seed, "RUN": 1, "VERBOSE": False,
            "GRAMMAR": "grammars/regression_1v.pybnf", "EXPERIMENT_NAME": os.path.join(tmp, "x"),
            "ALGORITHM": algorithm, "LEARNING_FACTOR": lf, "TRACK_TRAJECTORY": track,
            "DUMP_GRAMMAR": not track, "DUMP_POPULATION": False, "DUMP_PROGRESS": not track,
            "MIN_TREE_DEPTH": 6, "MAX_TREE_DEPTH": 10, "SAVE_STEP": 1000}
    grammar.reset()
    best, pcfg = sge.evolutionary_algorithm(evaluation_function=StabilityEvaluationFunction(b["S"]),
                                            parameters_file="parameters/standard.yml", options_dict=opts)
    return best, pcfg, eng.get_trajectory()


class TestEngine(unittest.TestCase):
    def test_tracking_does_not_change_the_search(self):
        for alg, lf in (("psge", 0.1), ("psge_pp", 0.1), ("sge", 0.0)):
            b0, p0, _ = tiny_run(alg, lf, track=False)
            b1, p1, traj = tiny_run(alg, lf, track=True)
            self.assertEqual(b0['phenotype'], b1['phenotype'], alg)
            self.assertEqual(b0['fitness'], b1['fitness'], alg)
            self.assertTrue(np.array_equal(p0, p1), alg)
            self.assertEqual(len(traj["gen"]), 7)

    def test_psge_pp_has_no_remapping(self):
        _, _, traj = tiny_run("psge_pp", 0.1, track=True, gens=8)
        for rate, fails in zip(traj["remap_rate"], traj["pp_failures"]):
            if fails == 0:
                self.assertEqual(rate, 0.0)
        self.assertGreater(sum(traj["pp_moved"]), 0)
        _, _, traj_psge = tiny_run("psge", 0.1, track=True, gens=8)
        self.assertGreater(max(traj_psge["remap_rate"]), 0.0)
        self.assertEqual(sum(traj_psge["pp_codons"]), 0)

    def test_psge_pp_same_update_as_psge_in_first_generation(self):
        # La primera actualización parte de la misma población: misma KL en las dos variantes.
        _, _, a = tiny_run("psge", 0.05, track=True)
        _, _, b = tiny_run("psge_pp", 0.05, track=True)
        self.assertAlmostEqual(a["kl_update"][0], b["kl_update"][0], places=12)

    def test_sgef_has_flat_trajectory(self):
        _, _, traj = tiny_run("sge", 0.0, track=True)
        self.assertTrue(all(k == 0.0 for k in traj["kl_update"]))
        self.assertTrue(all(r == 0.0 for r in traj["remap_rate"]))
        self.assertTrue(all(abs(h - 1.0) < 1e-12 for h in traj["entropy"]))


class TestGeneralizationGap(unittest.TestCase):
    def test_gap_zero_for_true_model_without_noise(self):
        b = generate_add_one_datasets("nguyen1", m=20, seed_data=3, n_test=200, n_val=100)
        f = "x[0]*x[0]*x[0]+x[0]*x[0]+x[0]"
        mt = calculate_stability_metrics(f, f, b["S"], b["z"], test_set=b["test_set"], var_y=b["var_y_S"],
                                         val_set=b["val_set"])
        self.assertAlmostEqual(mt["gen_gap_test_norm_fS"], 0.0, places=12)
        self.assertAlmostEqual(mt["gen_gap_val_norm_fS"], 0.0, places=12)

    def test_gap_in_apg_atg_units(self):
        b = generate_add_one_datasets("nguyen1", m=20, seed_data=3, n_test=200, n_val=0)
        f = "x[0]"
        mt = calculate_stability_metrics(f, f, b["S"], b["z"], test_set=b["test_set"], var_y=b["var_y_S"])
        test_loss = np.mean([(p[0] - p[-1]) ** 2 for p in b["test_set"]]) / b["var_y_S"]
        train_loss = np.mean([(p[0] - p[-1]) ** 2 for p in b["S"]]) / b["var_y_S"]
        self.assertAlmostEqual(mt["gen_gap_test_norm_fS"], abs(test_loss - train_loss), places=10)
        self.assertIsNone(mt["gen_gap_val_norm_fS"])


class TestRemapBenchmarks(unittest.TestCase):
    def test_values(self):
        F = {k: BENCHMARK_SPECS[k]["function"] for k in REMAP_CONFIRM_BENCHMARKS}
        self.assertAlmostEqual(F["nguyen6"]([0.5]), np.sin(0.5) + np.sin(0.75))
        self.assertAlmostEqual(F["nguyen8"]([4.0]), 2.0)
        self.assertAlmostEqual(F["nguyen10"]([0.0, 0.3]), 0.0)
        self.assertAlmostEqual(F["nguyen11"]([0.25, 0.5]), 0.5)
        self.assertAlmostEqual(F["nguyen12"]([1.0, 1.0]), -0.5)
        self.assertAlmostEqual(F["keijzer4"]([0.0]), 0.0)
        self.assertAlmostEqual(F["keijzer9"]([2.0]), np.log(2.0 + np.sqrt(5.0)))
        self.assertAlmostEqual(F["keijzer11"]([1.0, 2.0]), 2.0)
        self.assertAlmostEqual(F["keijzer14"]([0.0, 0.0]), 4.0)
        self.assertAlmostEqual(F["keijzer15"]([1.0, 2.0]), 0.2 + 4.0 - 3.0)

    def test_domains_and_disjoint_from_previous_study(self):
        used = {"nguyen1", "nguyen5", "nguyen7", "nguyen9", "keijzer6", "pagie1", "vladislavleva4", "vladislavleva7"}
        self.assertEqual(len(REMAP_CONFIRM_BENCHMARKS), 10)
        self.assertFalse(used & set(REMAP_CONFIRM_BENCHMARKS))
        for name in REMAP_CONFIRM_BENCHMARKS:
            spec = BENCHMARK_SPECS[name]
            b = generate_add_one_datasets(name, m=100, noise_ratio=0.05, seed_data=2000, n_test=1000, n_val=0)
            xs = np.array([p[:-1] for p in b["S_plus_z"]])
            dom = np.array(spec.get("domain") or spec["train_domain"])
            self.assertEqual(xs.shape[1], spec["n_vars"], name)
            self.assertTrue(np.all(xs >= dom[:, 0]) and np.all(xs <= dom[:, 1]), name)
            self.assertTrue(np.all(np.isfinite([p[-1] for p in b["S_plus_z"] + b["test_set"]])), name)
        b = generate_add_one_datasets("keijzer4", m=20, seed_data=2000, n_test=2000, n_val=0)
        xt = np.array([p[0] for p in b["test_set"]])
        self.assertTrue(xt.min() >= 0.05 and xt.max() > 10.0 and xt.max() <= 10.05)


class TestPageTrend(unittest.TestCase):
    def test_matches_reference_values(self):
        # Referencia: scipy.stats.page_trend_test(X, method="exact") con los mismos datos.
        rng = np.random.default_rng(1)
        X = rng.normal(size=(8, 6))
        r = page_trend(X)
        self.assertEqual(r["L"], 634.0)
        self.assertAlmostEqual(r["p"], 0.019384780877464283, places=12)
        X = rng.normal(size=(8, 6)) + np.arange(6) * 0.3
        self.assertAlmostEqual(page_trend(X)["p"], 0.024268489091750514, places=12)

    def test_perfect_trend_minimum_p(self):
        X = np.tile(np.arange(6, dtype=float), (4, 1))
        r = page_trend(X)
        self.assertAlmostEqual(r["p"], (1 / 720) ** 4, places=20)
        self.assertGreater(page_trend(-X)["p"], 0.99)

    def test_ties_and_infinite(self):
        X = np.array([[1, 1, 2, 3], [0, 2, 2, np.inf], [1, 2, 3, 4]], dtype=float)
        r = page_trend(X)
        self.assertTrue(0 < r["p"] < 0.05)
        flat = page_trend(np.ones((5, 4)))
        self.assertAlmostEqual(flat["p"], 1.0)


if __name__ == "__main__":
    unittest.main()
