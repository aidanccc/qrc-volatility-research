"""Small real simulator integration, separate from the full historical verification."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from qrcstudy.colin_models import features,inputs
from qrcstudy.models import sequences
from quantum_reservoir_qiskit import build_ising_hamiltonian,compute_unitaries,generate_coupling_matrix,quantum_reservoir

class QuantumAdapterTests(unittest.TestCase):
    def test_adapter_and_cache_match_direct_simulation_with_missing_suffix(self):
        rng=np.random.default_rng(0);x=rng.uniform(-1,0,(5,7));x[-1,1]=np.nan
        with tempfile.TemporaryDirectory() as tmp:
            a=features(x,'QR1',0,tmp)
            u,du=compute_unitaries(build_ising_hamiltonian(10,generate_coupling_matrix(10,seed=0)),1.,1)
            direct=quantum_reservoir(pd.DataFrame(np.vstack([x[:4],np.zeros((1,7))])),list(range(7)),u,du,3,1,10).T
            np.testing.assert_allclose(a[:5],direct,atol=1e-12)
            self.assertTrue(np.isnan(a[5]).all())
            np.testing.assert_allclose(features(x,'QR1',0,tmp),a,atol=0,equal_nan=True)
