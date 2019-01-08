import unittest
import doctest

import numpy as np

from pygibbs.conjugate import norm_chisq, mnorm_wishart, mnorm_mnorm_wishart, lm_mnorm_chisq, mlm_mnorm


def load_tests(loader, tests, ignore):

    modules = [norm_chisq, mnorm_wishart, mnorm_mnorm_wishart, lm_mnorm_chisq, mlm_mnorm]

    for mod in modules:
        tests.addTests(doctest.DocTestSuite(mod))
    return tests


class ConsistencyTest(unittest.TestCase):

    def setUp(self, nobs=int(1e5)):

        self.modules = [norm_chisq, mnorm_wishart, mnorm_mnorm_wishart, lm_mnorm_chisq]
        self.fixtures = [mod._generate_fixture(nobs=nobs) for mod in self.modules]
        self.posterior = [mod.update(*fixture[0], *fixture[2]) for mod, fixture in zip(self.modules, self.fixtures)]

    def test_consistency(self, rtol=1e-2, atol=1e-2):

        for mod, fixture, posterior in zip(self.modules, self.fixtures, self.posterior):
            np.testing.assert_allclose(mod.eval_logmargin(*fixture[0], *posterior),
                                       mod.eval_loglik(*fixture[0], *mod.get_ev(*posterior)).sum(),
                                       rtol, atol)
            for est, true in zip(mod.get_ev(*posterior), fixture[1]):
                np.testing.assert_allclose(est, true, rtol, atol)
            for est, true in zip(mod.get_mode(*posterior), fixture[1]):
                np.testing.assert_allclose(est, true, rtol, atol)
            for sample, true in zip(mod.sample_param(int(1e3), *posterior), fixture[1]):
                np.testing.assert_allclose(np.mean(sample, 0), true, rtol, atol)
