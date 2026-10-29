"""Offline checks for the IonQ noise-study circuits; no backend is contacted."""
import unittest
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from qrc.hardware.paper_circuits import PAPER_MODELS, IONQ_QIS_BASIS, paper_inputs, ideal_features
from qrc.noise.noise_model import z_expectations
from qrc.ionq.circuits import (block_qubits, cloud_circuits, history_values,
                                          reset_history_circuit, z_from_counts)
from qrc.ionq.local import probabilities, features


def unrolled_z(qc):
    """Noiseless <Z> of the measured 10 qubits of an unrolled circuit (Aer statevector)."""
    sim = AerSimulator(method='statevector')
    bare = qc.remove_final_measurements(inplace=False)
    n = (qc.num_qubits - 3) // 7
    bare.save_probabilities(block_qubits(n - 1, n))
    return z_expectations(sim.run(transpile(bare, sim, optimization_level=0)).result().data()['probabilities'])


def reset_z(model, steps, values, J):
    """Noiseless <Z> per virtual node of the reset circuit (Aer density matrix)."""
    sim = AerSimulator(method='density_matrix')
    data = sim.run(reset_history_circuit(values, J, steps, PAPER_MODELS[model]['virtual_nodes'])).result().data()
    return np.concatenate([z_expectations(data[f'node{e}']) for e in range(PAPER_MODELS[model]['virtual_nodes'])])


class IonQNoiseCircuitTests(unittest.TestCase):
    def test_both_forms_reproduce_partial_trace_features(self):
        data, _, couplings = paper_inputs()
        rows = data.iloc[600:604]
        for model, steps in [('QR1', 2), ('QR2', 3), ('QR2', 1)]:
            features_ = ideal_features(rows, model, couplings[model], steps=steps, order=2)[:, 3]
            values = history_values(rows, PAPER_MODELS[model]['features'], 3)
            np.testing.assert_allclose(reset_z(model, steps, values, couplings[model]), features_, atol=1e-10)
            for e, qc in enumerate(cloud_circuits(model, 3, steps, 3, rows, couplings)):
                self.assertEqual(qc.num_qubits, 24)
                self.assertTrue(set(qc.count_ops()) <= set(IONQ_QIS_BASIS) - {'reset'})
                np.testing.assert_allclose(unrolled_z(qc), features_[e * 10:(e + 1) * 10], atol=1e-10)

    def test_short_histories_match_between_forms(self):
        # The cloud check uses 1- and 2-month histories (10 and 17 qubits).
        data, _, couplings = paper_inputs()
        for months, width in [(1, 10), (2, 17)]:
            qcs = cloud_circuits('QR2', 600, 3, months, data, couplings)
            ref = reset_z('QR2', 3, history_values(data, PAPER_MODELS['QR2']['features'], 600, months), couplings['QR2'])
            for e, qc in enumerate(qcs):
                self.assertEqual(qc.num_qubits, width)
                np.testing.assert_allclose(unrolled_z(qc), ref[e * 10:(e + 1) * 10], atol=1e-10)

    def test_gate_counts_match_hardware_selection(self):
        data, _, couplings = paper_inputs()
        qc1, = cloud_circuits('QR1', 600, 2, 3, data, couplings)
        qc2 = cloud_circuits('QR2', 600, 3, 3, data, couplings)
        self.assertEqual(qc1.count_ops()['rxx'], 270)
        self.assertEqual([q.count_ops()['rxx'] for q in qc2], [405, 540])

    def test_local_noise_shrinks_features(self):
        data, _, couplings = paper_inputs()
        rows = data.iloc[600:604]
        clean = ideal_features(rows, 'QR1', couplings['QR1'], steps=2, order=2)[:, 3]
        for noise in ['aria-1', 'forte-1']:
            noisy = features(probabilities((noise, 'QR1', 2, [603], 3, 1.0)), 0.0)[0]
            self.assertLess(np.abs(noisy).mean(), np.abs(clean).mean())
            np.testing.assert_array_less(np.abs(noisy), np.abs(clean) + 1e-12)

    def test_counts_bit_order(self):
        # clbit 0 is the rightmost character, as in Qiskit counts.
        np.testing.assert_allclose(z_from_counts({'0000000001': 3, '0000000000': 1}), [-0.5] + [1] * 9)
        np.testing.assert_allclose(z_from_counts({'1000000000': 1.0}), [1] * 9 + [-1])


if __name__ == '__main__':
    unittest.main()
