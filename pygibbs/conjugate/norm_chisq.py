"""
Provides routines for estimating normal conjugate, such that:
    X[·j]|(mu, sig2) ~ Normal(mu[j], sig2[j])
    where mu is the mean and sig2 is the standard deviation.

The prior distribution over (sig2,) is scaled-inv-chi^2:
    sig2[j]|(v0, s0) ~ Scaled-Inv-Chi^2(v0[j], s0[j])

The posterior distribution over (sig2,) is scaled-inv-chi^2:
    sig2[j]|(X[·j], mu, vN, sN) ~ Scaled-Inv-Chi^2(vN[j], sN[j])

Data
----
X : Matrix[nobs, nvar]

Parameters
----------
mu (known) : Vector[nvar]
sig2 >= 0 : Vector[nvar]

Hyperparameters
---------------
v >= 0 : Vector[nvar]
s >= 0 : Vector[nvar]
"""

import numpy as np
from scipy.special import gammaln
from scipy.stats import norm

from pygibbs.tools.densities import eval_norm as eval_loglik


def update(X, mu, v0, s0):
    """Compute the hyperparameters of the posterior parameter distribution.

    Parameters
    ----------
    X, mu, v0, s0 : see module docstring

    Returns
    -------
    tuple (2 * np.ndarray)
        posterior hyperparameters (vN, sN)
    """

    nobs, nvar = np.sum(np.isfinite(X), 0), X.shape[1]
    X_var = np.where(nobs != 0, np.nanmean(np.square(X - mu), 0), np.zeros(nvar))

    vN = v0 + nobs
    sN = s0 + nobs * X_var

    return (vN, sN)


def weighted_update(X, W, mu, v0, s0):
    """Compute weighted hyperparameters of the posterior parameter distribution.

    Parameters
    ----------
    X : see module docstring
    W : np.ndarray in R+^(nobs, nvar)
        weight array
    mu, v0, s0 : see module docstring
        notice that X may not contain np.nan here

    Returns
    -------
    tuple (2 * np.ndarray)
        posterior hyperparameters (vN, sN)
    """

    nobs, nvar = np.sum(np.isfinite(X), 0), X.shape[1]
    X_mean = np.where(nobs != 0, np.nanmean(W * X, 0), np.zeros(nvar))
    X_var = np.where(nobs != 0, np.nanmean(W * np.square(X - mu), 0), np.zeros(nvar))

    vN = v0 + nobs
    sN = s0 + nobs * X_var

    return (vN, sN)


def sample_param(ndraws, v, s):
    """Draw samples from the parameter distribution given hyperparameters.

    Parameters
    ----------
    ndraws : int in N+
        number of draws to be sampled
    X, v, s : see module docstring

    Returns
    -------
    tuple (np.ndarray in R+^(ndraws, nvar))
        parameter draws (sig2,)
    """

    sig2 = s / np.random.chisquare(v, (ndraws, s.shape[0]))

    return (sig2,)


def sample_data(ndraws, mu, v, s):
    """Draw samples from the marginal data distribution given hyperparameters.

    Parameters
    ----------
    ndraws : int in N+
        number of draws to be sampled
    mu, v, s : see module docstring

    Returns
    -------
    np.ndarray in R^(ndraws, nvar)
        data draws
    """

    sig2 = sample_param(ndraws, v, s)

    X = np.array([
        np.random.normal(mu, np.sqrt(sig2_i), mu.shape[0]).flatten()
        for sig2_i in sig2])

    return X


def eval_logmargin(X, mu, v0, s0):
    """Evaluate the log marginal likelihood or evidence given data and hyperparameters. You can evaluate the predictive density by passing posterior instead of prior hyperparameters.

    Parameters
    ----------
    X, mu, v0, s0 : see module docstring

    Returns
    -------
    float
        log marginal likelihood
    """

    nobs = np.sum(np.isfinite(X), 0)
    vN, sN = update(X, mu, v0, s0)

    nc_lik = -nobs / 2 * np.log(2 * np.pi)
    nc_prior = -gammaln(v0 / 2) + v0 / 2 * np.log(s0)
    nc_post = -gammaln(vN / 2) + vN / 2 * np.log(sN)

    return np.sum(nc_lik + nc_prior - nc_post)


def get_ev(v, s):
    """Evaluate the expectation of parameters given hyperparameters.

    Parameters
    ----------
    v, s : see module docstring

    Returns
    -------
    tuple (np.ndarray,)
        parameter expectations (sig2,)
    """

    return (s / (v - 2),)


def get_mode(v, s):
    """Evaluate the mode of parameters given hyperparameters.

    Parameters
    ----------
    v, s : see module docstring

    Returns
    -------
    tuple (np.ndarray,)
        parameter modes (sig2,)
    """

    return (s / (v + 2),)
