import unittest
from abc import ABC

import numpy as np

from pygibbs.gibbs import hlm


class ConsistencyTest(ABC):

    def setUp(self, mod, nres=int(1e3), nobs=int(1e3), nvar=2, tol=(1e-1, 1e-1)):

        self.mod = mod
        self.tol = tol
        self.data, self.gt, self.hyper = self.mod._generate_fixture(nres, nobs, nvar)

    def test_map(self, niter=int(1e2)):

        map = self.mod.estimate(niter, *self.data, *self.hyper)
        for est, true in zip(map[1], self.gt[1]):
            np.testing.assert_allclose(est, true, *self.tol)

    def test_pev(self, niter=int(1e2)):

        samples = self.mod.sample(niter, *self.data, *self.hyper)
        for est, true in zip([np.mean(x, 0) for x in samples[1]], self.gt[1]):
            np.testing.assert_allclose(est, true, *self.tol)

    def test_map_eta(self):

        map = self.mod.estimate_eta(self.data, self.gt[1])
        for est, true in zip(map, self.gt[0]):
            np.testing.assert_allclose(est, true, *self.tol)

    def test_map_theta(self):

        map = self.mod.estimate_theta(self.data, self.gt[0], self.hyper)
        for est, true in zip(map, self.gt[1]):
            np.testing.assert_allclose(est, true, *self.tol)

    def test_logmargin(self):

        map = self.mod.estimate_theta(self.data, self.gt[0], self.hyper)
        np.testing.assert_allclose(self.mod.eval_logobserved(self.data, map),
                                   self.mod.eval_loglik(self.data, self.gt[0], map).sum(),
                                   *self.tol)


class ConsistencyTest_hlm(ConsistencyTest, unittest.TestCase):
    def setUp(self):
        super(ConsistencyTest_hlm, self).setUp(hlm)


if __name__ == '__main__':
    unittest.main()
