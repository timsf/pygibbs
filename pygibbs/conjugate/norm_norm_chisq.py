"""
Provides routines for estimating normals, ie:
    x[·j]|(mu, sig) ~ Normal(mu[j], sig[j])
    where mu is the mean and sig is the standard deviation.

The prior distribution over (mu, sig) is normal-scaled-inv-chi^2:
    mu[j]|(sig, m, l) ~ Normal(m[j], sig[j] / l[j])
    sig[j]|(v, s) ~ Scaled-Inv-Chi^2(v[j], s[j])

The posterior distribution over (mu, sig) is normal-scaled-inv-chi^2:
    mu[j]|(sig, m, l) ~ Normal(mN[j], sig[j] / lN[j])
    sig[j]|(x, v, s) ~ Scaled-Inv-Chi^2(vN[j], sN[j])
"""

from typing import Tuple

import numpy as np
from scipy.special import gammaln

from pygibbs.tools.densities import eval_norm as eval_loglik


Data = Tuple[np.ndarray]
Param = Tuple[np.ndarray, np.ndarray]
Hyper = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def update(x: np.ndarray, m: np.ndarray, l: np.ndarray, v: np.ndarray, s: np.ndarray, w: np.ndarray = None) -> Hyper:
    """Compute the hyperparameters of the posterior parameter distribution.

    :param x: [nobs, nvar]
    :param m: [nvar]
    :param l: (>0)[nvar]
    :param v: (>0)[nvar]
    :param s: (>0)[nvar]
    :param w: (>0)[nobs]
    :returns: posterior hyperparameters
    """

    nobs, nvar = np.sum(np.isfinite(x), 0), x.shape[1]
    if w is not None:
        x = np.diag(w) @ x
    x_mean = np.where(nobs != 0, np.nanmean(x, 0), np.zeros(nvar))
    x_var = np.where(nobs != 0, np.nanvar(x, 0), np.zeros(nvar))

    lN = l + nobs
    mN = (l * m + nobs * x_mean) / lN
    vN = v + nobs
    sN = s + nobs * (x_var + l / lN * np.square(x_mean - m))

    return mN, lN, vN, sN


def sample_param(ndraws: int, m: np.ndarray, l: np.ndarray, v: np.ndarray, s: np.ndarray) -> Param:
    """Draw samples from the parameter distribution given hyperparameters.

    :param ndraws: (>0) number of samples to be drawn
    :param m: [nvar]
    :param l: (>0)[nvar]
    :param v: (>0)[nvar]
    :param s: (>0)[nvar]
    :returns: samples from the parameter distribution
    """

    sig = s / np.random.chisquare(v, (ndraws, s.shape[0]))
    mu = np.random.normal(m, np.sqrt(sig / l), (ndraws, m.shape[0]))

    return mu, sig


def sample_data(ndraws: int, m: np.ndarray, l: np.ndarray, v: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Draw samples from the marginal data distribution given hyperparameters.

    :param ndraws: (>0) number of samples to be drawn
    :param m: [nvar]
    :param l: (>0)[nvar]
    :param v: (>0)[nvar]
    :param s: (>0)[nvar]
    :returns: samples from the data distribution
    """

    z = np.random.standard_normal((ndraws, m.shape[0]))
    mu, sig = sample_param(ndraws, m, l, v, s)
    sqrt_sig = np.sqrt(sig)
    x = mu + np.array([sqrt_sig_i * z_i for z_i, sqrt_sig_i in zip(z, sqrt_sig)])

    return x


def eval_logmargin(x: np.ndarray, m: np.ndarray, l: np.ndarray, v: np.ndarray, s: np.ndarray, w: np.ndarray = None) -> float:
    """Evaluate the log marginal likelihood or evidence given data and hyperparameters.
    You can evaluate the predictive density by passing posterior instead of prior hyperparameters.

    :param x: [nobs, nvar]
    :param m: [nvar]
    :param l: (>0)[nvar]
    :param v: (>0)[nvar]
    :param s: (>0)[nvar]
    :param w: (>0)[nobs]
    :returns: log marginal likelihood
    """

    def nc(_, l0, v0, s0):
        return (v0 * np.log(s0) + np.log(l0)) / 2 - gammaln(v0 / 2)
    nobs = np.sum(np.isfinite(x), 0)

    return float(np.sum(nc(m, l, v, s) - nc(*update(x, m, l, v, s, w)) - nobs / 2 * np.log(np.pi)))


def get_ev(m: np.ndarray, l: np.ndarray, v: np.ndarray, s: np.ndarray) -> Param:
    """Evaluate the expectation of parameters given hyperparameters.

    :param m: [nvar]
    :param l: (>0)[nvar]
    :param v: (>0)[nvar]
    :param s: (>0)[nvar]
    :returns: parameter expectations
    """

    return m, s / (v - 2)


def get_mode(m: np.ndarray, l: np.ndarray, v: np.ndarray, s: np.ndarray) -> Param:
    """Evaluate the mode of parameters given hyperparameters.

    :param m: [nvar]
    :param l: (>0)[nvar]
    :param v: (>0)[nvar]
    :param s: (>0)[nvar]
    :returns: parameter modes
    """

    return m, s / (v + 2)


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
    sig = np.ones(nvar)

    # set hyperparameters
    m = np.zeros(nvar)
    l = np.ones(nvar)
    v = np.ones(nvar)
    s = np.ones(nvar)

    return (x,), (mu, sig), (m, l, v, s)
