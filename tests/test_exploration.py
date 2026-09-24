import unittest
import numpy as np
from qrcstudy.exploration import choose_alpha,ridge_grid

class ExplorationTests(unittest.TestCase):
    def test_no_current_or_future_selection(self):
        p=np.zeros((30,6));y=np.arange(30.)
        a=choose_alpha(p,y,24);p[24:]=100;y[24:]=-100
        self.assertEqual(a,choose_alpha(p,y,24))
        with self.assertRaisesRegex(ValueError,'Insufficient'):choose_alpha(p,y,23)

    def test_affine_feature_invariance(self):
        rng=np.random.default_rng(0);x=rng.normal(size=(40,4));y=x[:,0]*2+3;z=rng.normal(size=4)
        scale=np.array([2.,.2,10.,3.]);offset=np.array([5.,-4.,9.,2.])
        np.testing.assert_allclose(ridge_grid(x,y,z),ridge_grid(x*scale+offset,y,z*scale+offset),atol=1e-10)
