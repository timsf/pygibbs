"""
Provides routines for estimating multivariate normals, ie:
    x[i·]|(mu, sig) ~ Mv-Normal(mu, sig)
    where mu is the mean vector and sig is the covariance matrix.

The prior distribution over (mu, sig) is normal-inv-wishart:
    mu|(sig, m, l) ~ Mv-Normal(m, sig / l)
    sig|(v, s) ~ Inv-Wishart(v, s)

The posterior distribution over (mu, sig) is normal-inv-wishart:
    mu|(x, sig, m, l) ~ Mv-Normal(mN, sig / lN)
    sig|(x, v, s) ~ Inv-Wishart(vN, sN)
"""

from typing import Tuple

import numpy as np
from scipy.special import multigammaln
from scipy.stats import invwishart

from pygibbs.tools.densities import eval_mvnorm as eval_loglik
from pygibbs.tools.linalg import logdet_pd


Data = Tuple[np.ndarray]
Param = Tuple[np.ndarray, np.ndarray]
Hyper = Tuple[np.ndarray, float, float, np.ndarray]


def update(x: np.ndarray, m: np.ndarray, l: float, v: float, s: np.ndarray, w: np.ndarray = None) -> Hyper:
    """Compute the hyperparameters of the posterior parameter distribution.

    :param x: [nobs, nvar]
    :param m: [nvar]
    :param l: (>0)
    :param v: (>0)
    :param s: (PD)[nvar, nvar]
    :param w: (>0)[nobs]
    :returns: posterior hyperparameters
    """

    nobs, nvar = x.shape
    if nobs != 0:
        if w is not None:
            x = np.diag(w) @ x
        x_mean = np.mean(x, 0)
        x_cov = np.cov(x.T, ddof=0)
    else:
        x_mean = np.zeros(nvar)
        x_cov = np.zeros((nvar, nvar))
    
    lN = l + nobs
    mN = (l * m + nobs * x_mean) / lN
    vN = v + nobs
    sN = s + nobs * (x_cov + l / lN * np.dot((x_mean - m), (x_mean - m).T))

    return mN, lN, vN, sN


def sample_param(ndraws: int, m: np.ndarray, l: float, v: float, s: np.ndarray) -> Param:
    """Draw samples from the parameter distribution given hyperparameters.

    :param ndraws: (>0) number of samples to be drawn
    :param m: [nvar]
    :param l: (>0)
    :param v: (>0)
    :param s: (PD)[nvar, nvar]
    :returns: samples from the parameter distribution
    """

    z = np.random.standard_normal((ndraws, m.shape[0]))
    sig = invwishart.rvs(v, s, ndraws).reshape((ndraws, *s.shape))
    mu = m + np.array([np.linalg.cholesky(sig_i / l) @ z_i for z_i, sig_i in zip(z, sig)])

    return mu, sig


def sample_data(ndraws: int, m: np.ndarray, l: float, v: float, s: np.ndarray) -> np.ndarray:
    """Draw samples from the marginal data distribution given hyperparameters.

    :param ndraws: (>0) number of samples to be drawn
    :param m: [nvar]
    :param l: (>0)
    :param v: (>0)
    :param s: (PD)[nvar, nvar]
    :returns: samples from the data distribution
    """

    z = np.random.standard_normal((ndraws, m.shape[0]))
    mu, sig = sample_param(ndraws, m, l, v, s)
    x = mu + np.array([np.linalg.cholesky(sig_i) @ z_i for z_i, sig_i in zip(z, sig)])

    return x


def eval_logmargin(x: np.ndarray, m: np.ndarray, l: float, v: float, s: np.ndarray, w: np.ndarray = None) -> float:
    """Evaluate the log marginal likelihood or evidence given data and hyperparameters.
    You can evaluate the predictive density by passing posterior instead of prior hyperparameters.

    :param x: [nobs, nvar]
    :param m: [nvar]
    :param l: (>0)
    :param v: (>0)
    :param s: (PD)[nvar, nvar]
    :param w: (>0)[nobs]
    :returns: log marginal likelihood
    """

    def nc(_, l0, v0, s0):
        return (v0 * logdet_pd(s0) + nvar * np.log(l0)) / 2 - multigammaln(v0 / 2, nvar)
    nobs, nvar = x.shape

    return nc(m, l, v, s) - nc(*update(x, m, l, v, s, w)) - nobs * nvar / 2 * np.log(np.pi)


def get_ev(m: np.ndarray, l: float, v: float, s: np.ndarray) -> Param:
    """Evaluate the expectation of parameters given hyperparameters.

    :param m: [nvar]
    :param l: (>0)
    :param v: (>0)
    :param s: (PD)[nvar, nvar]
    :returns: parameter expectations
    """

    return m, s / (v - len(m) - 1)


def get_mode(m: np.ndarray, l: float, v: float, s: np.ndarray) -> Param:
    """Evaluate the mode of parameters given hyperparameters.

    :param m: [nvar]
    :param l: (>0)
    :param v: (>0)
    :param s: (PD)[nvar, nvar]
    :returns: parameter modes
    """

    return m, s / (v + len(m) + 1)


def _generate_fixture(nobs: int = 3, nvar: int = 2, seed: int = 666) -> (Data, Param, Hyper):
    """Generate a set of input data.

    :param nobs:
    :param nvar:
    :param seed: random number generator seed
    :returns: generated data, ground truth, hyperparameters
    """

    # ensure deterministic output
    np.random.seed(seed)

    # set input
    x = np.random.standard_normal((nobs, nvar))

    # set ground truth
    mu = np.zeros(nvar)
    sig = np.identity(nvar)

    # set hyperparameters
    m = np.zeros(nvar)
    l = 1
    v = nvar + 2
    s = np.diag(np.ones(nvar)) * v

    return (x,), (mu, sig), (m, l, v, s)
