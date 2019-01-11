import unittest
from abc import ABC

import numpy as np

from pygibbs.conjugate import norm_chisq, mnorm_wishart, mnorm_mnorm_wishart, lm_mnorm_chisq, poisson_gamma, multinomial_dirichlet


class ConsistencyTest(ABC):

    def setUp(self, mod, nobs=int(1e5)):

        self.mod = mod
        self.nobs = nobs
        self.fixture = self.mod._generate_fixture(nobs=nobs)
        self.posterior = self.mod.update(*self.fixture[0], *self.fixture[2])

    def test_map(self, rtol=1e-2, atol=1e-2):

        for est, true in zip(self.mod.get_mode(*self.posterior), self.fixture[1]):
            np.testing.assert_allclose(est, true, rtol, atol)

    def test_pev(self, rtol=1e-2, atol=1e-2):

        for est, true in zip(self.mod.get_mode(*self.posterior), self.fixture[1]):
            np.testing.assert_allclose(est, true, rtol, atol)

    def test_logmargin(self, rtol=1e-2, atol=1e-2, nsamples=int(1e2)):

        minifix = self.mod._generate_fixture(nobs=nsamples)[0]
        np.testing.assert_allclose(self.mod.eval_logmargin(*minifix, *self.posterior),
                                   self.mod.eval_loglik(*minifix, *self.mod.get_ev(*self.posterior)).sum(),
                                   rtol, atol)

    def test_param_sampling(self, rtol=1e-2, atol=1e-2, nsamples=int(1e2)):

        for sample, true in zip(self.mod.sample_param(nsamples, *self.posterior), self.fixture[1]):
            np.testing.assert_allclose(np.mean(sample, 0), true, rtol, atol)


class ConsistencyTest_lm_mnorm_chisq(ConsistencyTest, unittest.TestCase):
    def setUp(self):
        super(self.__class__, self).setUp(lm_mnorm_chisq)


class ConsistencyTest_mnorm_mnorm_wishart(ConsistencyTest, unittest.TestCase):
    def setUp(self):
        super(self.__class__, self).setUp(mnorm_mnorm_wishart)


class ConsistencyTest_mnorm_wishart(ConsistencyTest, unittest.TestCase):
    def setUp(self):
        super(self.__class__, self).setUp(mnorm_wishart)


class ConsistencyTest_multinomial_dirichlet(ConsistencyTest, unittest.TestCase):
    def setUp(self):
        super(self.__class__, self).setUp(multinomial_dirichlet)


class ConsistencyTest_norm_chisq(ConsistencyTest, unittest.TestCase):
    def setUp(self):
        super(self.__class__, self).setUp(norm_chisq)


class ConsistencyTest_poisson_gamma(ConsistencyTest, unittest.TestCase):
    def setUp(self):
        super(self.__class__, self).setUp(poisson_gamma)


if __name__ == '__main__':
    unittest.main()
