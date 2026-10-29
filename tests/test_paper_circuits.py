"""Gate-level checks for the paper-faithful hardware circuits; no backend is contacted."""
import unittest
import numpy as np
from qiskit.quantum_info import DensityMatrix, Operator, SparsePauliOp
from scipy.linalg import expm
from qrc.hardware.paper_circuits import (ROOT, PAPER_MODELS, IONQ_QIS_BASIS, paper_inputs, trotter_block,
                                     history_circuit, month_values, ideal_features)
from qrc.reservoir import build_ising_hamiltonian, build_z_observables, generate_coupling_matrix
from qrc.trotter import build_trotter_evolution


def exact(J, t):
    return expm(-1j * t * build_ising_hamiltonian(J.shape[0], J))


def gap(J, t, steps, order):
    return np.linalg.norm(Operator(trotter_block(J, t, steps, order)).data - exact(J, t), 2)


class PaperCircuitTests(unittest.TestCase):
    def test_paper_configuration_matches_notebook(self):
        data, target, couplings = paper_inputs()
        self.assertEqual(data.shape, (816, 15))
        self.assertEqual(PAPER_MODELS['QR1']['features'], ['RV', 'MKT', 'DP', 'IP', 'RV_q', 'STR', 'DEF'])
        self.assertEqual(PAPER_MODELS['QR2']['features'], ['RV', 'MKT', 'STR', 'RV_q', 'EP', 'INF', 'DEF'])
        from qrc.reservoir import load_coupling_matrices
        ms = load_coupling_matrices(str(ROOT / 'data/coeff_10.jld2'))
        np.testing.assert_array_equal(couplings['QR1'], ms[0])
        np.testing.assert_array_equal(couplings['QR2'], ms[1])
        np.testing.assert_array_equal(month_values(data, ['RV'], 3)[:, 0], target[:3])

    def test_first_order_matches_existing_builder(self):
        J = generate_coupling_matrix(4, seed=1)
        for steps in [1, 3]:
            u, du = build_trotter_evolution(4, J, 1., steps, 2)
            np.testing.assert_allclose(Operator(trotter_block(J, 1., steps, 1)).data, Operator(u).data, atol=1e-12)
            np.testing.assert_allclose(Operator(trotter_block(J, .5, steps, 1)).data, Operator(du).data, atol=1e-12)

    def test_second_order_is_strang_splitting_with_same_two_qubit_gates(self):
        J = generate_coupling_matrix(3, seed=2)
        dt = .4
        hxx = build_ising_hamiltonian(3, J) - SparsePauliOp(['IIZ', 'IZI', 'ZII']).to_matrix()
        hz = SparsePauliOp(['IIZ', 'IZI', 'ZII']).to_matrix()
        half = expm(-1j * dt / 2 * hz)
        step = half @ expm(-1j * dt * hxx) @ half
        np.testing.assert_allclose(Operator(trotter_block(J, 2 * dt, 2, 2)).data, step @ step, atol=1e-12)
        J10 = generate_coupling_matrix(10, seed=0)
        for steps in [1, 4]:
            one, two = trotter_block(J10, 1., steps, 1).count_ops(), trotter_block(J10, 1., steps, 2).count_ops()
            self.assertEqual(one['rxx'], two['rxx'])
            self.assertEqual(two['rxx'], 45 * steps)
            self.assertEqual(two['rz'], 10 * (steps + 1))

    def test_error_orders(self):
        J = generate_coupling_matrix(4, seed=3)
        # Global error falls ~2x per doubling for order 1 and ~4x for order 2.
        self.assertTrue(1.6 < gap(J, 1., 4, 1) / gap(J, 1., 8, 1) < 2.4)
        self.assertTrue(3.4 < gap(J, 1., 4, 2) / gap(J, 1., 8, 2) < 4.6)
        self.assertLess(gap(J, 1., 2, 2), gap(J, 1., 2, 1))

    def test_reset_circuits_reproduce_partial_trace_features(self):
        data, _, couplings = paper_inputs()
        for model in ['QR1', 'QR2']:
            spec = PAPER_MODELS[model]
            rows = data.iloc[600:604]
            features = ideal_features(rows, model, couplings[model], steps=2, order=2)[:, 3]
            values = month_values(rows, spec['features'], 3)
            for endpoint in range(1, spec['virtual_nodes'] + 1):
                qc = history_circuit(values, couplings[model], 2, 2, spec['virtual_nodes'], endpoint)
                self.assertTrue(set(qc.count_ops()) <= set(IONQ_QIS_BASIS))
                state = DensityMatrix.from_label('0' * 10).evolve(qc.remove_final_measurements(inplace=False))
                z = [state.expectation_value(o).real for o in build_z_observables(10)]
                np.testing.assert_allclose(z, features[(endpoint - 1) * 10:endpoint * 10], atol=1e-10)


if __name__ == '__main__':
    unittest.main()
