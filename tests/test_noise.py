"""Checks for the Aria-1 noise emulation; no model behavior changes."""
import unittest
import numpy as np
from qiskit.quantum_info import Operator
from qrc.noise.compute import probabilities, features, reservoir_circuit
from qrc.noise.noise_model import aria1_rates, error_parameters, build_noise_model, apply_readout, z_expectations
from qrc.reservoir import generate_coupling_matrix
from qrc.trotter import build_trotter_evolution


class NoiseEmulationTests(unittest.TestCase):
    def test_scale_zero_equals_ideal_trotter_features(self):
        # Diagnostics `instrument` mirrors production's partial-trace protocol.
        from qrc.diagnostics import compute as diag
        x = np.random.default_rng(7).uniform(-.8, .8, (6, 7))
        j = generate_coupling_matrix(10, seed=0)
        for model, v in [('QR1', 1), ('QR2', 2)]:
            u, du = build_trotter_evolution(10, j, 1., 1, v)
            expected = np.array([diag.instrument(x[t - 3:t], Operator(u), Operator(du), v)[0] for t in [3, 5]])
            actual = features(probabilities(x, [3, 5], model, 1, 0), 0)
            np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0)

    def test_noise_shrinks_features_monotonically(self):
        x = np.random.default_rng(3).uniform(-.8, .8, (4, 7))
        amp = [np.abs(features(probabilities(x, [3], 'QR1', 1, s), error_parameters(s)['readout_flip'])).mean() for s in [0, 1, 4]]
        self.assertGreater(amp[0], amp[1]); self.assertGreater(amp[1], amp[2])

    def test_readout_map_scales_z_by_one_minus_two_p(self):
        p = np.random.default_rng(1).dirichlet(np.ones(1024))
        np.testing.assert_allclose(z_expectations(apply_readout(p, .03)), (1 - .06) * z_expectations(p), atol=1e-12)
        self.assertAlmostEqual(apply_readout(p, .03).sum(), 1, places=12)

    def test_noise_model_targets_ry_and_rxx_only(self):
        model = build_noise_model(1)
        self.assertEqual(set(model.noise_instructions), {'ry', 'rxx'})
        self.assertIsNone(build_noise_model(0))
        rates = aria1_rates()
        self.assertGreater(rates['records_used'], 20)
        self.assertTrue(0.98 < rates['fidelity_2q'] < 0.995 and rates['fidelity_1q'] > 0.999)

    def test_circuit_uses_resets_and_saves_each_endpoint(self):
        u, du = build_trotter_evolution(10, generate_coupling_matrix(10, seed=0), 1., 1, 2)
        ops = reservoir_circuit(np.zeros((3, 7)), u, du, 2).count_ops()
        self.assertEqual(ops['reset'], 14); self.assertEqual(ops['save_probabilities'], 2); self.assertEqual(ops['rxx'], 4 * 45)


if __name__ == '__main__':
    unittest.main()
