"""Qubit-order and Trotter gate-order conventions stated in the quantum module docstrings."""
import unittest
import numpy as np
from qiskit.quantum_info import DensityMatrix, Operator, SparsePauliOp, partial_trace
from scipy.linalg import expm
from qrc.reservoir import build_z_observables, encode_input, generate_coupling_matrix
from qrc.trotter import build_trotter_step


class QuantumConventionTests(unittest.TestCase):
    def test_known_state_qubit_order(self):
        # RY(pi) flips input qubit 0 only; hidden qubits 7-9 stay |0>.
        state = DensityMatrix.from_label('000').tensor(encode_input(np.array([1., 0, 0, 0, 0, 0, 0]), 7))
        z = [state.expectation_value(o).real for o in build_z_observables(10)]
        np.testing.assert_allclose(z, [-1] + [1] * 9, atol=1e-12)
        np.testing.assert_allclose(partial_trace(state, list(range(7))).data, DensityMatrix.from_label('000').data, atol=1e-12)

    def test_trotter_step_applies_xx_then_z(self):
        j = generate_coupling_matrix(2, seed=0)
        dt = 0.7
        hxx = SparsePauliOp('XX', j[0, 1]).to_matrix()
        hz = SparsePauliOp(['IZ', 'ZI'], [1., 1.]).to_matrix()
        step = Operator(build_trotter_step(2, j, dt)).data
        documented = expm(-1j * dt * hz) @ expm(-1j * dt * hxx)
        reversed_order = expm(-1j * dt * hxx) @ expm(-1j * dt * hz)
        self.assertLess(np.max(abs(step - documented)), 1e-12)
        self.assertGreater(np.max(abs(step - reversed_order)), 1e-2)


if __name__ == '__main__':
    unittest.main()
