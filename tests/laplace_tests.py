import unittest
from abc import ABC

import numpy as np

from pygibbs.laplace import logit_mnorm


class ConsistencyTest(ABC):

    def setUp(self, mod, nobs=int(1e4), tol=(1e-2, 1e-2)):

        self.mod = mod
        self.tol = tol
        self.data, self.gt, self.hyper = self.mod._generate_fixture(nobs=nobs)
        self.posterior = self.mod.update(*self.data, *self.hyper)

    def test_map(self):

        for est, true in zip(self.mod.get_mode(*self.posterior), self.gt):
            np.testing.assert_allclose(est, true, *self.tol)

    def test_pev(self):

        for est, true in zip(self.mod.get_ev(*self.posterior), self.gt):
            np.testing.assert_allclose(est, true, *self.tol)

    def test_logmargin(self, nsamples=int(1e2)):

        minifix = self.mod._generate_fixture(nobs=nsamples)[0]
        np.testing.assert_allclose(self.mod.eval_logmargin(*minifix, *self.posterior),
                                   self.mod.eval_loglik(*minifix, *self.mod.get_ev(*self.posterior)).sum(),
                                   *self.tol)

    def test_param_sampling(self, nsamples=int(1e2)):

        for sample, true in zip(self.mod.sample_param(nsamples, *self.posterior), self.gt):
            np.testing.assert_allclose(np.mean(sample, 0), true, *self.tol)


class ConsistencyTest_logit_mnorm(ConsistencyTest, unittest.TestCase):
    def setUp(self):
        super(self.__class__, self).setUp(logit_mnorm)


if __name__ == '__main__':
    unittest.main()
