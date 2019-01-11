import unittest
from abc import ABC

import numpy as np

from pygibbs.gibbs import hlm


class ConsistencyTest(ABC):

    def setUp(self, mod, nobs=int(1e3)):

        self.mod = mod
        self.fixture = self.mod._generate_fixture(int(1e3), nobs, 2)

    def test_map(self, niter=int(1e2), rtol=1e-1, atol=1e-1):

        map = self.mod.estimate(niter, *self.fixture[0], *self.fixture[2])
        for est, true in zip(map[1], self.fixture[1][1]):
            np.testing.assert_allclose(est, true, rtol, atol)

    def test_pev(self, niter=int(1e2), rtol=1e-1, atol=1e-1):

        samples = self.mod.sample(niter, *self.fixture[0], *self.fixture[2])
        for est, true in zip([np.mean(x, 0) for x in samples[1]], self.fixture[1][1]):
            np.testing.assert_allclose(est, true, rtol, atol)

    def test_map_eta(self, rtol=1e-1, atol=1e-1):

        map = self.mod.estimate_eta(self.fixture[0], self.fixture[1][1])
        for est, true in zip(map, self.fixture[1][0]):
            np.testing.assert_allclose(est, true, rtol, atol)

    def test_map_theta(self, rtol=1e-1, atol=1e-1):

        map = self.mod.estimate_theta(self.fixture[0], self.fixture[1][0], self.fixture[2])
        for est, true in zip(map, self.fixture[1][1]):
            np.testing.assert_allclose(est, true, rtol, atol)


class ConsistencyTest_hlm(ConsistencyTest, unittest.TestCase):

    def setUp(self):

        super(ConsistencyTest_hlm, self).setUp(hlm)


if __name__ == '__main__':
    unittest.main()
