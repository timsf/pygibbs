"""
Provides routines for estimating multivariate normals, ie:
    x[i·]|(mu, sig) ~ Mv-Normal(mu, sig)
    where mu is the mean vector and sig is the covariance matrix.

The prior distribution over (sig,) is inv-wishart:
    sig|(v, s) ~ Inv-Wishart(v, s)

The posterior distribution over (sig,) is inv-wishart:
    sig|(x, vN, sN) ~ Inv-Wishart(vN, sN)
"""

from typing import Tuple

import numpy as np
from scipy.special import multigammaln
from scipy.stats import invwishart

from pygibbs.tools.densities import eval_mvnorm as eval_loglik, eval_matt
from pygibbs.tools.linalg import logdet_pd


Data = Tuple[np.ndarray, np.ndarray]
Param = Tuple[np.ndarray]
Hyper = Tuple[float, np.ndarray]


def update(x: np.ndarray, mu: np.ndarray, v: float, s: np.ndarray, w: np.ndarray = None) -> Hyper:
    """Compute the hyperparameters of the posterior parameter distribution.

    :param x: [nvar, nobs]
    :param mu: [nvar]
    :param v: (>0)
    :param s: (PD)[nvar, nvar]
    :param w: (>0)[nobs]
    :returns: posterior hyperparameters
    """

    nobs, nvar = x.shape
    if nobs != 0:
        if w is not None:
            x = np.diag(w) @ x
        x_cov = np.mean([np.outer(x_i, x_i) for x_i in x - mu], 0)
    else:
        x_cov = np.zeros((nvar, nvar))

    vN = v + nobs
    sN = s + nobs * x_cov

    return vN, sN


def sample_param(ndraws: int, v: float, s: np.ndarray) -> Param:
    """Draw samples from the parameter distribution given hyperparameters.

    :param ndraws: (>0) number of samples to be drawn
    :param v: (>0)
    :param s: (PD)[nvar, nvar]
    :returns: samples from the parameter distribution
    """

    sig = invwishart.rvs(v, s, ndraws).reshape((ndraws, *s.shape))

    return sig,


def sample_data(ndraws: int, mu: np.ndarray, v: float, s: np.ndarray) -> np.ndarray:
    """Draw samples from the marginal data distribution given hyperparameters.

    :param ndraws: (>0) number of samples to be drawn
    :param mu: [nvar]
    :param v: (>0)
    :param s: (PD)[nvar, nvar]
    :returns: samples from the data distribution
    """

    z = np.random.standard_normal((ndraws, mu.shape[0]))
    sig, = sample_param(ndraws, v, s)
    x = mu + np.array([np.linalg.cholesky(sig_i) @ z_i for z_i, sig_i in zip(z, sig)])

    return x


def eval_logmargin(x: np.ndarray, mu: np.ndarray, v: float, s: np.ndarray, w: np.ndarray = None) -> float:
    """Evaluate the log marginal likelihood or evidence given data and hyperparameters.
    You can evaluate the predictive density by passing posterior instead of prior hyperparameters.

    :param x: [nvar, nobs]
    :param mu: [nvar]
    :param v: (>0)
    :param s: (PD)[nvar, nvar]
    :param w: (>0)[nobs]
    :returns: log marginal likelihood
    """

    nobs, nvar = x.shape
    def nc(v0, s0):
        return v0 / 2 * logdet_pd(s0) - multigammaln(v0 / 2, nvar)

    return nc(v, s) - nc(*update(x, mu, v, s, w)) - nobs * nvar / 2 * np.log(np.pi)


def get_ev(v: float, s: np.ndarray) -> Param:
    """Evaluate the expectation of parameters given hyperparameters.

    :param v: (>0)
    :param s: (PD)[nvar, nvar]
    :returns: parameter expectations
    """

    return s / (v - s.shape[0] - 1),


def get_mode(v: float, s: np.ndarray) -> Param:
    """Evaluate the mode of parameters given hyperparameters.

    :param v: (>0)
    :param s: (PD)[nvar, nvar]
    :returns: parameter modes
    """

    return s / (v + s.shape[0] + 1),


def _generate_fixture(nobs: int = 3, nvar: int = 2, seed: int = 666) -> (Data, Param, Hyper):
    """Generate a set of input data.

    :param nobs: (>0)
    :param nvar: (>0)
    :param seed: random number generator seed
    :returns: generated data, ground truth, hyperparameters
    """

    # ensure deterministic output
    np.random.seed(seed)

    # set input
    x = np.random.standard_normal((nobs, nvar))
    mu = np.zeros(nvar)

    # set ground truth
    sig = np.identity(nvar)

    # set hyperparameters
    v = nvar + 2
    s = np.diag(np.ones(nvar)) * v

    return (x, mu), (sig,), (v, s)

