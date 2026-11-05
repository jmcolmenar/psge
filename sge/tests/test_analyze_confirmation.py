"""Prueba del análisis confirmatorio con datos SIMULADOS de efecto conocido (sin resultados reales)."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

# Efectos multiplicativos sobre APG+ATG y NRMSE-test por algoritmo (simulados)
EFFECTS = {"sge": (1.0, 1.0), "psge": (1.3, 1.0), "psge_kl": (1.3, 1.0),
           "sa_psge": (0.7, 1.0), "sa_psge_lam0": (1.3, 1.0)}


def simulate(root, n_data=12, n_evo=2):
    rng = np.random.default_rng(7)
    for bench in ("benchA", "benchB", "benchC"):
        for m, noise in ((20, 0.0), (100, 0.05)):
            for sd in range(n_data):
                base_s, base_n = rng.lognormal(0, 0.5), rng.lognormal(-1, 0.3)
                for alg, (fs, fn) in EFFECTS.items():
                    for se in range(n_evo):
                        s = base_s * fs * rng.lognormal(0, 0.05)
                        arm = {"n_case_evaluations": 1000 * (1.2 if alg.startswith("sa") else 1.0), "wall_sec": 1.0,
                               "cpu_sec": 1.0, "peak_rss_kb_process": 1000}
                        r = {"benchmark": bench, "m": m, "noise": noise, "algorithm": alg, "seed_data": sd,
                             "seed_evo": se, "apg_plus_atg": s, "apg_norm": s / 2, "atg_norm": s / 2,
                             "nrmse_test_fS": base_n * fn * rng.lognormal(0, 0.01), "nrmse_test_fSz": base_n,
                             "js_divergence": 0.01 * s, "arm_S": arm, "arm_Sz": arm}
                        d = os.path.join(root, bench, f"m{m}_noise{noise:g}", alg)
                        os.makedirs(d, exist_ok=True)
                        with open(os.path.join(d, f"{alg}_d{sd}_e{se}.json"), "w") as fh:
                            json.dump(r, fh)
    # una semilla incompleta y una ejecución inválida
    os.remove(os.path.join(root, "benchA", "m20_noise0", "psge_kl", "psge_kl_d0_e1.json"))
    f = os.path.join(root, "benchB", "m20_noise0", "sa_psge", "sa_psge_d1_e0.json")
    r = json.load(open(f)); r["nrmse_test_fS"] = 9999999; json.dump(r, open(f, "w"))


class TestAnalyzeConfirmation(unittest.TestCase):
    def test_recovers_simulated_effects(self):
        with tempfile.TemporaryDirectory() as root:
            simulate(root)
            out = os.path.join(root, "rep")
            cp = subprocess.run([sys.executable, "-m", "examples.analyze_confirmation", "--root", root,
                                 "--global_unit", "cell", "--alternative", "less", "--n_boot", "1000",
                                 "--output", out], capture_output=True, text=True)
            self.assertEqual(cp.returncode, 0, cp.stderr)
            rep = json.load(open(out + ".json"))
            integ = rep["integrity"]["benchA/m20_noise0"]
            self.assertEqual(integ["complete_data_seeds"], 11)          # la semilla 0 queda fuera
            self.assertEqual(rep["integrity"]["benchB/m20_noise0"]["invalid_runs"]["sa_psge"], 1)
            for k, cell in rep["cells"].items():
                c = cell["comparisons"]
                self.assertEqual(c["sa_psge_vs_psge"]["category"], "éxito", k)
                self.assertEqual(c["psge_kl_vs_psge"]["category"] if "psge_kl_vs_psge" in c else "nulo", "nulo")
                self.assertEqual(c["sa_psge_vs_sa_psge_lam0"]["category"], "éxito", k)
                self.assertGreater(c["psge_vs_sge"]["stab"]["rel"], 0.2, k)          # RQ1: +30 % simulado
                self.assertEqual(c["psge_vs_sge"]["category"], "PSGE más inestable", k)
                self.assertAlmostEqual(c["sa_psge_vs_psge"]["stab"]["rel"], 0.7 / 1.3 - 1, delta=0.08)   # -46 %
            g = {f["comparison"]: f for f in rep["global"]["stab"]["comparisons"]}
            self.assertEqual(g["sa_psge_vs_psge"]["treat_better_blocks"], 6)
            self.assertLess(g["sa_psge_vs_psge"]["p_holm"], 0.05)
            self.assertAlmostEqual(g["sa_psge_vs_psge"]["geo_mean_ratio"], 0.7 / 1.3, delta=0.05)
            self.assertLess(rep["global"]["stab"]["friedman"]["p"], 0.05)
            # Configuración por defecto (opción C): principal = benchmark unilateral, robustez = celdas bilateral.
            cp2 = subprocess.run([sys.executable, "-m", "examples.analyze_confirmation", "--root", root,
                                  "--n_boot", "200", "--output", out + "_c"], capture_output=True, text=True)
            self.assertEqual(cp2.returncode, 0, cp2.stderr)
            rep2 = json.load(open(out + "_c.json"))
            self.assertEqual(rep2["global"]["unit"], "benchmark")
            self.assertEqual(rep2["global"]["alternative_stability"], "less")
            self.assertEqual(rep2["global_robustness"]["unit"], "cell")
            prim = {f["comparison"]: f for f in rep2["global"]["stab"]["comparisons"]}
            rob = {f["comparison"]: f for f in rep2["global_robustness"]["stab"]["comparisons"]}
            # 3 benchmarks simulados, todos a favor: p unilateral = 1/8; 6 celdas a favor: p bilateral = 2/64
            self.assertAlmostEqual(prim["sa_psge_vs_psge"]["p"], 1 / 8)
            self.assertEqual(prim["sa_psge_vs_psge"]["alternative"], "less")
            self.assertEqual(prim["psge_vs_sge"]["alternative"], "greater")
            self.assertAlmostEqual(prim["psge_vs_sge"]["p"], 1 / 8)
            self.assertAlmostEqual(rob["sa_psge_vs_psge"]["p"], 2 / 64)
            self.assertEqual(rob["sa_psge_vs_psge"]["alternative"], "two-sided")
            nr = {f["comparison"]: f for f in rep2["global"]["nrmse"]["comparisons"]}
            self.assertEqual(nr["sa_psge_vs_psge"]["alternative"], "two-sided")   # NRMSE siempre bilateral
            self.assertIn("ROBUSTEZ", cp2.stdout)
            self.assertIn("NOTA", cp2.stdout)

if __name__ == "__main__":
    unittest.main()
