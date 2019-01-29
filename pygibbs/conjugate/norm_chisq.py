"""
Provides routines for estimating normals, ie:
    x[·j]|(mu, sig) ~ Normal(mu[j], sig[j])
    where mu is the mean and sig is the standard deviation.

The prior distribution over (sig,) is scaled-inv-chi^2:
    sig[j]|(v, s) ~ Scaled-Inv-Chi^2(v[j], s[j])

The posterior distribution over (sig,) is scaled-inv-chi^2:
    sig[j]|(x[·j], mu, vN, sN) ~ Scaled-Inv-Chi^2(vN[j], sN[j])
"""

from typing import Tuple

import numpy as np
from scipy.special import gammaln

from pygibbs.tools.densities import eval_norm as eval_loglik


Data = Tuple[np.ndarray, np.ndarray]
Param = Tuple[np.ndarray]
Hyper = Tuple[np.ndarray, np.ndarray]


def update(x: np.ndarray, mu: np.ndarray, v: np.ndarray, s: np.ndarray, w: np.ndarray = None) -> Hyper:
    """Compute the hyperparameters of the posterior parameter distribution.

    :param x: [nvar, nobs]
    :param mu: [nvar]
    :param v: (>0)[nvar]
    :param s: (>0)[nvar]
    :param w: (>0)[nobs]
    :returns: posterior hyperparameters
    """

    nobs, nvar = np.sum(np.isfinite(x), 0), x.shape[1]
    if w is not None:
        x = np.diag(w) @ x
    x_var = np.where(nobs != 0, np.nanmean(np.square(x - mu), 0), np.zeros(nvar))

    vN = v + nobs
    sN = s + nobs * x_var

    return vN, sN


def sample_param(ndraws: int, v: np.ndarray, s: np.ndarray) -> Param:
    """Draw samples from the parameter distribution given hyperparameters.

    :param ndraws: (>0) number of samples to be drawn
    :param v: (>0)[nvar]
    :param s: (>0)[nvar]
    :returns: samples from the parameter distribution
    """

    sig = s / np.random.chisquare(v, (ndraws, s.shape[0]))

    return sig,


def sample_data(ndraws: int, mu: np.ndarray, v: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Draw samples from the marginal data distribution given hyperparameters.

    :param ndraws: (>0) number of samples to be drawn
    :param mu: [nvar]
    :param v: (>0)[nvar]
    :param s: (>0)[nvar]
    :returns: samples from the data distribution
    """

    z = np.random.standard_normal((ndraws, mu.shape[0]))
    sig, = sample_param(ndraws, v, s)
    sqrt_sig = np.sqrt(sig)
    x = mu + np.array([sqrt_sig_i * z_i for z_i, sqrt_sig_i in zip(z, sqrt_sig)])

    return x


def eval_logmargin(x: np.ndarray, mu: np.ndarray, v: np.ndarray, s: np.ndarray, w: np.ndarray = None) -> float:
    """Evaluate the log marginal likelihood or evidence given data and hyperparameters.
    You can evaluate the predictive density by passing posterior instead of prior hyperparameters.

    :param x: [nvar, nobs]
    :param mu: [nvar]
    :param v: (>0)[nvar]
    :param s: (>0)[nvar]
    :param w: (>0)[nobs]
    :returns: log marginal likelihood
    """

    def nc(v0, s0):
        return v0 / 2 * np.log(s0) - gammaln(v0 / 2)
    nobs = np.sum(np.isfinite(x), 0)

    return float(np.sum(nc(v, s) - nc(*update(x, mu, v, s, w)) - nobs / 2 * np.log(np.pi)))


def get_ev(v: np.ndarray, s: np.ndarray) -> Param:
    """Evaluate the expectation of parameters given hyperparameters.

    :param v: (>0)[nvar]
    :param s: (>0)[nvar]
    :returns: parameter expectations
    """

    return s / (v - 2),


def get_mode(v: np.ndarray, s: np.ndarray) -> Param:
    """Evaluate the mode of parameters given hyperparameters.

    :param v: (>0)[nvar]
    :param s: (>0)[nvar]
    :returns: parameter modes
    """

    return s / (v + 2),


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
    sig = np.ones(nvar)

    # set hyperparameters
    v = np.ones(nvar)
    s = np.ones(nvar) * v

    return (x, mu), (sig,), (v, s)
