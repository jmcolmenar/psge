"""
Unit tests for stability metrics and dataset generation.
Verifies exact numerical formulas from Fong & Motani and project documents.
"""

import unittest
import numpy as np

from sge.stability import (
    calculate_point_squared_loss,
    calculate_stability_metrics,
    calculate_jensen_shannon,
    calculate_pcfg_entropy,
    generate_add_one_datasets,
)


class TestStabilityMetrics(unittest.TestCase):

    def test_numerical_example_from_document(self):
        """
        Validates the exact numerical example from
        'Explicacion_metricas_estabilidad_APG_ATG_SGE_PSGE_diseno_experimental.docx' (Section 3.2):
          S = {(1, 1), (2, 4), (3, 9), (4, 16)}
          z = (5, 25)
          f_S(x) = 0.95*x^2 + 0.20
          f_{S+z}(x) = 0.97*x^2 + 0.10
        """
        S = [
            [1.0, 1.0],
            [2.0, 4.0],
            [3.0, 9.0],
            [4.0, 16.0],
        ]
        z = [5.0, 25.0]

        # String representations of the models
        model_S = "0.95 * (x[0]**2) + 0.20"
        model_Sz = "0.97 * (x[0]**2) + 0.10"

        test_set = [[6.0, 36.0]]

        metrics = calculate_stability_metrics(
            model_S_phenotype=model_S,
            model_S_plus_z_phenotype=model_Sz,
            S=S,
            z=z,
            test_set=test_set,
            var_y=1.0  # raw loss unit
        )

        # Expected:
        # V(f_S, z) = (0.95*25 + 0.20 - 25)^2 = (23.95 - 25)^2 = 1.1025
        self.assertAlmostEqual(metrics["loss_fS_z"], 1.1025, places=5)

        # V(f_{S+z}, z) = (0.97*25 + 0.10 - 25)^2 = (24.35 - 25)^2 = 0.4225
        self.assertAlmostEqual(metrics["loss_fSz_z"], 0.4225, places=5)

        # APG = |1.1025 - 0.4225| = 0.68
        self.assertAlmostEqual(metrics["apg_raw"], 0.68, places=5)

        # Mean in-sample loss on S = 0.11125
        self.assertAlmostEqual(metrics["mean_in_sample_loss_fS"], 0.11125, places=5)

        # ATG = |0.4225 - 0.11125| = 0.31125
        self.assertAlmostEqual(metrics["atg_raw"], 0.31125, places=5)

        # APG + ATG = 0.68 + 0.31125 = 0.99125
        self.assertAlmostEqual(metrics["data_stability_raw"], 0.99125, places=5)

    def test_jensen_shannon_identical(self):
        """Identical distributions should have zero JS divergence."""
        pcfg = np.array([
            [0.5, 0.5, 0.0],
            [0.25, 0.25, 0.5],
        ])
        mask = np.array([
            [True, True, False],
            [True, True, True],
        ])

        js, _ = calculate_jensen_shannon(pcfg, pcfg, mask)
        self.assertAlmostEqual(js, 0.0, places=6)

    def test_jensen_shannon_divergent(self):
        """Different distributions should yield positive JS divergence in [0, 1]."""
        p1 = np.array([[1.0, 0.0], [0.5, 0.5]])
        p2 = np.array([[0.0, 1.0], [0.5, 0.5]])
        mask = np.array([[True, True], [True, True]])

        js, _ = calculate_jensen_shannon(p1, p2, mask)
        self.assertGreater(js, 0.0)
        self.assertLessEqual(js, 1.0)

    def test_dataset_generation_sizes(self):
        """Verify Add-One dataset generation properties."""
        bundle = generate_add_one_datasets("nguyen1", m=20, noise_ratio=0.0, seed_data=42, n_test=100)
        S = bundle["S"]
        z = bundle["z"]
        S_plus_z = bundle["S_plus_z"]
        test_set = bundle["test_set"]

        self.assertEqual(len(S), 20)
        self.assertEqual(len(S_plus_z), 21)
        self.assertEqual(len(test_set), 100)
        self.assertEqual(S_plus_z[-1], z)

        # Ensure z is not an exact duplicate of any point in S
        s_coords = [tuple(pt[:-1]) for pt in S]
        self.assertNotIn(tuple(z[:-1]), s_coords)


if __name__ == "__main__":
    unittest.main()
