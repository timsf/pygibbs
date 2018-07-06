"""
Provides routines for estimating multivariate normal conjugate, such that:
    X[i·]|(mu, Sig) ~ Mv-Normal(mu, Sig)
    where mu is the mean vector and Sig is the covariance matrix.

The prior distribution over (mu, Sig) is inv-wishart:
    Sig|(v0, S0) ~ Inv-Wishart(v0, S0)

The posterior distribution over (mu, Sig) is inv-wishart:
    Sig|(X, mu, vN, SN) ~ Inv-Wishart(vN, SN)

Data
----
X : Matrix[nobs, nvar]

Parameters
----------
mu (known) : Vector[nvar]
Sig : Matrix[nvar, nvar]

Hyperparameters
---------------
v >= 0 : Real
S >= 0 (positive semi-definite): Matrix[nvar, nvar]
"""

import numpy as np
from scipy.special import multigammaln
from scipy.stats import multivariate_normal, invwishart

from pygibbs.tools.densities import eval_mvnorm as eval_loglik


def update(X, mu, v0, S0):
    """Compute the hyperparameters of the posterior parameter distribution.

    Parameters
    ----------
    X, mu, v0, S0 : see module docstring

    Returns
    -------
    tuple (float, np.ndarray)
        posterior hyperparameters (vN, SN)
    """

    nobs, nvar = X.shape
    if nobs != 0:
        X_demean = X - mu
        X_cov = np.mean([np.outer(X_demean[i], X_demean[i]) for i in range(nobs)], 0)
    else:
        X_mean = np.zeros(nvar)
        X_cov = np.zeros((nvar, nvar))

    vN = v0 + nobs
    SN = S0 + nobs * X_cov

    return (vN, SN)


def weighted_update(X, w, mu, v0, S0):
    """Compute weighted hyperparameters of the posterior parameter distribution.

    Parameters
    ----------
    X : see module docstring
    w : np.ndarray in R+^nobs
        weight vector
    mu, v0, S0 : see module docstring

    Returns
    -------
    tuple (float, np.ndarray)
        posterior hyperparameters (vN, SN)
    """

    nobs, nvar = X.shape
    if nobs != 0:
        X_mean = np.mean(w[:, np.newaxis] * X, 0)
        X_demean = X - mu
        X_cov = np.mean([w[i] * np.outer(X_demean[i], X_demean[i]) for i in range(nobs)], 0)
    else:
        X_mean = np.zeros(nvar)
        X_cov = np.zeros((nvar, nvar))

    vN = v0 + nobs
    SN = S0 + nobs * X_cov

    return (vN, SN)


def sample_param(ndraws, v, S):
    """Draw samples from the parameter distribution given hyperparameters.

    Parameters
    ----------
    ndraws : int in N+
        number of draws to be sampled
    v, S : see module docstring

    Returns
    -------
    tuple (np.ndarray in PSD_nvar^ndraws,)
        parameter draws (Sig,)
    """

    Sig = invwishart.rvs(v, S, ndraws)
    if ndraws == 1:
        Sig = Sig[np.newaxis]

    return (Sig,)


def sample_data(ndraws, mu, v, S):
    """Draw samples from the marginal data distribution given hyperparameters.

    Parameters
    ----------
    ndraws : int in N+
        number of draws to be sampled
    mu, v, S : see module docstring

    Returns
    -------
    np.ndarray in R^(ndraws, nvar)
        data draws
    """

    Sig = sample_param(ndraws, mu, v, S)

    X = np.array([
        np.random.multivariate_normal(mu, Sig_i, 1).flatten()
        for Sig_i in Sig])

    return X


def eval_logmargin(X, mu, v0, S0):
    """Evaluate the log marginal likelihood or evidence given data and hyperparameters. You can evaluate the predictive density by passing posterior instead of prior hyperparameters.

    Parameters
    ----------
    X, mu, v0, S0 : see module docstring

    Returns
    -------
    float
        log marginal likelihood
    """

    nobs, nvar = X.shape
    vN, SN = update(X, mu, v0, S0)

    logdet = lambda A: np.prod(np.linalg.slogdet(A))
    nc_lik = -np.prod(X.shape) / 2 * np.log(np.pi)
    nc_prior = -multigammaln(v0 / 2, nvar) + v0 / 2 * logdet(S0)
    nc_post = -multigammaln(vN / 2, nvar) + vN / 2 * logdet(SN)

    return nc_lik + nc_prior - nc_post


def get_ev(v, S):
    """Evaluate the expectation of parameters given hyperparameters.

    Parameters
    ----------
    v, S : see module docstring

    Returns
    -------
    tuple (np.ndarray,)
        parameter expectations (Sig,)
    """

    return (S / (v - S.shape[0] - 1),)


def get_mode(v, S):
    """Evaluate the mode of parameters given hyperparameters.

    Parameters
    ----------
    v, S : see module docstring

    Returns
    -------
    tuple (np.ndarray,)
        parameter modes (Sig,)
    """

    return (S / (v + S.shape[0] + 1),)
