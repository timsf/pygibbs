import unittest

import numpy as np
from scipy.stats import norm, invwishart

from pygibbs.tools import linalg, densities


class LinalgTests(unittest.TestCase):

    def setUp(self):

        pass

    def test_quad(self):

        s = invwishart(2, np.diag(np.ones(2))).rvs(1)
        x = np.random.standard_normal((3, 2))

        np.testing.assert_allclose(linalg.eval_quad(x, s), [x_i @ np.linalg.inv(s) @ x_i for x_i in x])

    def test_mquad(self):

        s = invwishart(2, np.diag(np.ones(2))).rvs(1)
        x = np.random.standard_normal((3, 2))

        np.testing.assert_allclose(linalg.eval_matquad(x, s), x @ np.linalg.inv(s) @ x.T)

    def test_logdet(self):

        s = invwishart(2, np.diag(np.ones(2))).rvs(1)

        np.testing.assert_allclose(linalg.logdet_pd(s), np.prod(np.linalg.slogdet(s)))

    def test_precond_solve(self):

        s = invwishart(2, np.diag(np.ones(2))).rvs(1)
        x = np.random.standard_normal(2)

        np.testing.assert_allclose(linalg.precond_solve_pd(s, x), np.linalg.solve(s, x))

    def test_detquad(self):

        s = invwishart(2, np.diag(np.ones(2))).rvs(1)
        x = np.random.standard_normal((3, 2))
        cf, logdet = linalg.eval_detquad(x, s)

        np.testing.assert_allclose(np.sum(cf ** 2, 1), [x_i @ np.linalg.inv(s) @ x_i for x_i in x])
        np.testing.assert_allclose(logdet, np.prod(np.linalg.slogdet(s)))

    def test_double_detquad(self):

        s = invwishart(3, np.diag(np.ones(3))).rvs(1)
        t = invwishart(2, np.diag(np.ones(2))).rvs(1)
        x = np.random.standard_normal((3, 2))
        cf, logdet_s, logdet_t = linalg.eval_double_detquad(x, s, t)

        # np.testing.assert_allclose(cf.T @ cf, np.linalg.cholesky(np.linalg.inv(t)) @ x.T @ np.linalg.inv(s) @ x @ np.linalg.cholesky(np.linalg.inv(t)).T)
        np.testing.assert_allclose(logdet_s, np.prod(np.linalg.slogdet(s)))
        np.testing.assert_allclose(logdet_t, np.prod(np.linalg.slogdet(t)))


class DensityTests(unittest.TestCase):

    def setUp(self, nobs=3, nvar=2):

        self.x = np.random.standard_normal((nobs, nvar))
        self.mu = np.random.standard_normal((nobs, nvar))
        self.nu = np.random.standard_normal(1) ** 2
        self.sig = invwishart(nvar, np.identity(nvar)).rvs(1)
        self.tau = invwishart(nobs, np.identity(nobs)).rvs(1)

    def test_norm_root(self):

        np.testing.assert_allclose(densities.eval_norm(self.x, self.mu, np.diag(self.sig)),
                                   norm(self.mu, np.sqrt(np.diag(self.sig))).logpdf(self.x))

    def test_mvnorm_decomp(self):

        np.testing.assert_allclose(densities.eval_mvnorm(self.x, self.mu[0], np.diag(np.diag(self.sig))),
                                   densities.eval_norm(self.x, self.mu[0], np.diag(self.sig)).sum(1))

    def test_matnorm_decomp(self):

        np.testing.assert_allclose(densities.eval_matnorm(self.x, self.mu, self.tau, np.identity(self.sig.shape[0])),
                                   densities.eval_mvnorm(self.x.T, self.mu.T, self.tau).sum(0))

    def test_t_lim(self, lim=1e10, rtol=1e-4):

        np.testing.assert_allclose(densities.eval_norm(self.x, self.mu, np.diag(self.sig)),
                                   densities.eval_t(self.x, self.mu, np.diag(self.sig), np.repeat(lim, len(self.nu))),
                                   rtol)

    def test_mvt_lim(self, lim=1e10, rtol=1e-4):

        np.testing.assert_allclose(densities.eval_mvnorm(self.x, self.mu, self.sig),
                                   densities.eval_mvt(self.x, self.mu, self.sig, lim),
                                   rtol)

    def test_matt_lim(self, lim=1e10, rtol=1e-4):

        np.testing.assert_allclose(densities.eval_matnorm(self.x, self.mu, self.tau, self.sig),
                                   densities.eval_matt(self.x, self.mu, self.tau, self.sig, lim),
                                   rtol)


if __name__ == '__main__':
    unittest.main()
