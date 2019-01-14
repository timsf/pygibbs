"""
Provides routines for estimating poisson distributions, ie:
    x[·j]|l ~ Poisson(l[j])
    where l is the rate.

The prior distribution over (l,) is gamma:
    l[j]|(a, b) ~ Gamma(a[j], b[j])

The posterior distribution over (l,) is gamma:
    l[j]|(x, a, b) ~ Gamma(aN[j], bN[j])
"""

from typing import Tuple

import numpy as np
from scipy.special import gammaln

from pygibbs.tools.densities import eval_poisson as eval_loglik


Data = Tuple[np.ndarray]
Param = Tuple[np.ndarray]
Hyper = Tuple[np.ndarray, np.ndarray]


def update(x: np.ndarray, a: np.ndarray, b: np.ndarray, w: np.ndarray = None) -> Hyper:
    """Compute the hyperparameters of the posterior parameter distribution.

    :param x: (>0)[nobs, nvar]
    :param a: (>0)[nvar]
    :param b: (>0)[nvar]
    :param w: observation weights
    :returns: posterior hyperparameters
    """

    if w is not None:
        x = np.diag(w) @ x
    nobs, nvar = np.sum(np.isfinite(x), 0), x.shape[1]
    counts = np.where(nobs != 0, np.nansum(x, 0), np.zeros(nvar))

    aN = a + counts
    bN = b + nobs

    return aN, bN


def sample_param(ndraws: int, a: np.ndarray, b: np.ndarray) -> Param:
    """Draw samples from the parameter distribution given hyperparameters.

    :param ndraws: (>0) number of samples to be drawn
    :param a: (>0)[nvar]
    :param b: (>0)[nvar]
    :returns: samples from the parameter distribution
    """

    lam = np.random.gamma(a, 1 / b, (ndraws, len(a)))

    return lam,


def sample_data(ndraws: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Draw samples from the marginal data distribution given hyperparameters.

    :param ndraws: number of samples to be drawn
    :param a: (>0)[nvar]
    :param b: (>0)[nvar]
    :returns: samples from the data distribution
    """

    lam = sample_param(ndraws, a, b)[0]
    x = np.array([np.random.poisson(lam_i, len(a)).flatten() for lam_i in lam])

    return x


def eval_logmargin(x: np.ndarray, a: np.ndarray, b: np.ndarray, w: np.ndarray = None) -> float:
    """Evaluate the log marginal likelihood or evidence given data and hyperparameters.
    You can evaluate the predictive density by passing posterior instead of prior hyperparameters.

    :param x: (>0)[nobs, nvar]
    :param a: (>0)[nvar]
    :param b: (>0)[nvar]
    :param w: observation weights
    :returns: log marginal likelihood
    """

    def nc(a0, b0):
        return a0 * np.log(b0) - gammaln(a0)

    return float(np.sum(nc(a, b) - nc(*update(x, a, b, w)) - np.nansum(gammaln(x + 1), 0)))


def get_ev(a: np.ndarray, b: np.ndarray) -> Param:
    """Evaluate the expectation of parameters given hyperparameters.

    :param a: (>0)[nvar]
    :param b: (>0)[nvar]
    :returns: parameter expectations
    """

    return a / b,


def get_mode(a: np.ndarray, b: np.ndarray) -> Param:
    """Evaluate the mode of parameters given hyperparameters.

    :param a: (>0)[nvar]
    :param b: (>0)[nvar]
    :returns: parameter modes
    """

    return (a - 1) / b,


def _generate_fixture(nobs: int = 3, nvar: int = 2, seed: int = 666) -> (Data, Param, Hyper):
    """Generate a set of input data.

    :param nobs: (>0)
    :param nvar: (>0)
    :param seed: (>0) random number generator seed
    :returns: generated data, ground truth, hyperparameters
    """

    # ensure deterministic output
    np.random.seed(seed)

    # set ground truth
    lam = np.e

    # set input
    x = np.random.poisson(lam, (nobs, nvar))

    # set hyperparameters
    a = np.ones(nvar)
    b = np.ones(nvar)

    return (x,), (lam,), (a, b)
