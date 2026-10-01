"""Checks for diagnostic protocol equivalence; no model behavior changes."""
import unittest
import numpy as np
from qiskit.quantum_info import DensityMatrix, Operator
from threadpoolctl import threadpool_limits
from compute import history_circuit, instrument
from quantum_reservoir_qiskit import generate_coupling_matrix, build_z_observables
from quantum_reservoir_trotter import build_trotter_evolution


class DiagnosticProtocolTests(unittest.TestCase):
    def test_reset_circuit_matches_partial_trace_protocol_at_both_endpoints(self):
        values=np.random.default_rng(31).uniform(-.7,.7,(3,7))
        j=generate_coupling_matrix(10,seed=0)
        with threadpool_limits(limits=1):
            u,du=build_trotter_evolution(10,j,1.,1,2)
            expected,probabilities,_=instrument(values,Operator(u),Operator(du),2)
            for endpoint in [1,2]:
                qc=history_circuit(values,j,1,2,endpoint).remove_final_measurements(inplace=False)
                # Deterministic density-matrix reset channel, not one random pure-state trajectory.
                state=DensityMatrix.from_label('0'*10).evolve(qc)
                actual=np.array([state.expectation_value(z).real for z in build_z_observables(10)])
                np.testing.assert_allclose(actual,expected[(endpoint-1)*10:endpoint*10],atol=2e-12,rtol=0)
                np.testing.assert_allclose(state.probabilities(),probabilities[endpoint-1],atol=2e-12,rtol=0)

    def test_full_history_resource_counts_include_repeated_endpoints(self):
        values=np.zeros((3,7));j=generate_coupling_matrix(10,seed=0)
        for steps in [1,2,4,8]:
            for v in [1,2]:
                total=0
                for endpoint in range(1,v+1):
                    qc=history_circuit(values,j,steps,v,endpoint);ops=qc.count_ops()
                    self.assertEqual(ops['rxx'],45*steps*(2+endpoint))
                    self.assertEqual(ops['ry'],21);self.assertEqual(ops['reset'],14);self.assertEqual(ops['measure'],10)
                    self.assertEqual(qc.num_qubits,10);total+=ops['rxx']
                self.assertEqual(total,(135 if v==1 else 315)*steps)

    def test_known_shot_variance_and_joint_correlation(self):
        signs=1-2*((np.arange(4)[:,None]>>np.arange(2))&1)
        prob=np.array([.5,0,0,.5]);means=prob@signs
        covariance=signs.T@(prob[:,None]*signs)-np.outer(means,means)
        np.testing.assert_allclose(covariance,np.ones((2,2)))
        # Correlated qubits cannot be treated as independent forecast noise.
        self.assertAlmostEqual(float(np.ones(2)@covariance@np.ones(2)),4)
        for shots in [100,1000,10000]:self.assertAlmostEqual(np.sqrt(covariance[0,0]/shots),1/np.sqrt(shots))


if __name__=='__main__':unittest.main()
