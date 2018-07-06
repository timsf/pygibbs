"""
Provides routines for estimating multivariate normal conjugate, such that:
    X[i·]|(mu, Sig) ~ Mv-Normal(mu, Sig)
    where mu is the mean vector and Sig is the covariance matrix.

The prior distribution over (mu, Sig) is normal-inv-wishart:
    mu|(Sig, m0, l0) ~ Mv-Normal(m0, Sig / l0)
    Sig|(v0, S0) ~ Inv-Wishart(v0, S0)

The posterior distribution over (mu, Sig) is normal-inv-wishart:
    mu|(X, Sig, mN, lN) ~ Mv-Normal(mN, Sig / lN)
    Sig|(X, vN, SN) ~ Inv-Wishart(vN, SN)

Data
----
X : Matrix[nobs, nvar]

Parameters
----------
mu : Vector[nvar]
Sig : Matrix[nvar, nvar]

Hyperparameters
---------------
m : Vector[nvar]
l >= 0 : Real
v >= 0 : Real
S >= 0 (positive semi-definite): Matrix[nvar, nvar]
"""

import numpy as np
from scipy.special import multigammaln
from scipy.stats import multivariate_normal, invwishart

from pygibbs.tools.densities import eval_mvnorm as eval_loglik


def update(X, m0, l0, v0, S0):
    """Compute the hyperparameters of the posterior parameter distribution.

    Parameters
    ----------
    X, m0, l0, v0, S0 : see module docstring

    Returns
    -------
    tuple (np.ndarray, float, float, np.ndarray)
        posterior hyperparameters (mN, lN, vN, SN)
    """

    nobs, nvar = X.shape
    if nobs != 0:
        X_mean = np.mean(X, 0)
        X_cov = np.cov(X.T)
    else:
        X_mean = np.zeros(nvar)
        X_cov = np.zeros((nvar, nvar))
    
    lN = l0 + nobs
    mN = (l0 * m0 + nobs * X_mean) / lN
    vN = v0 + nobs
    SN = S0 + nobs * (X_cov + l0 / lN * np.dot((X_mean - m0), (X_mean - m0).T))

    return (mN, lN, vN, SN)


def weighted_update(X, w, m0, l0, v0, S0):
    """Compute weighted hyperparameters of the posterior parameter distribution.

    Parameters
    ----------
    X : see module docstring
    w : np.ndarray in R+^nobs
        weight vector
    m0, l0, v0, S0 : see module docstring

    Returns
    -------
    tuple (np.ndarray, float, float, np.ndarray)
        posterior hyperparameters (mN, lN, vN, SN)
    """

    nobs, nvar = X.shape
    if nobs != 0:
        X_mean = np.mean(w[:,np.newaxis] * X, 0)
        X_demean = X - X_mean
        X_cov = np.mean([
            w[i] * np.outer(X_demean[i], X_demean[i]) for i in range(nobs)], 0)
    else:
        X_mean = np.zeros(nvar)
        X_cov = np.zeros((nvar, nvar))
    
    lN = l0 + nobs
    mN = (l0 * m0 + nobs * X_mean) / lN
    vN = v0 + nobs
    SN = S0 + nobs * (X_cov + l0 / lN * np.dot((X_mean - m0), (X_mean - m0).T))

    return (mN, lN, vN, SN)


def sample_param(ndraws, m, l, v, S):
    """Draw samples from the parameter distribution given hyperparameters.

    Parameters
    ----------
    ndraws : int in N+
        number of draws to be sampled
    m, l, v, S : see module docstring

    Returns
    -------
    tuple (np.ndarray in R^(ndraws, nvar), np.ndarray in PSD_nvar^ndraws)
        parameter draws (mu, Sig)
    """

    Sig = invwishart.rvs(v, S, ndraws)
    if ndraws == 1:
        Sig = Sig[np.newaxis]
    mu = np.array([
        np.random.multivariate_normal(m, Sig_i / l, 1).flatten()
        for Sig_i in Sig])

    return (mu, Sig)


def sample_data(ndraws, m, l, v, S):
    """Draw samples from the marginal data distribution given hyperparameters.

    Parameters
    ----------
    ndraws : int in N+
        number of draws to be sampled
    m, l, v, S : see module docstring

    Returns
    -------
    np.ndarray in R^(ndraws, nvar)
        data draws
    """

    mu, Sig = sample_param(ndraws, m, l, v, S)

    X = np.array([
        np.random.multivariate_normal(mu_i, Sig_i, 1).flatten()
        for mu_i, Sig_i in zip(mu, Sig)])

    return X


def eval_logmargin(X, m0, l0, v0, S0):
    """Evaluate the log marginal likelihood or evidence given data and hyperparameters. You can evaluate the predictive density by passing posterior instead of prior hyperparameters.

    Parameters
    ----------
    X, m0, l0, v0, S0 : see module docstring

    Returns
    -------
    float
        log marginal likelihood
    """

    nobs, nvar = X.shape
    mN, lN, vN, BN = update(X, m0, l0, v0, S0)

    logdet = lambda A: np.prod(np.linalg.slogdet(A))
    nc_lik = -np.prod(X.shape) / 2 * np.log(np.pi)
    nc_prior = -multigammaln(v0 / 2, nvar) + (v0 * logdet(S0) + nvar * logdet(L0)) / 2
    nc_post = -multigammaln(vN / 2, nvar) + (vN * logdet(SN) + nvar * logdet(LN)) / 2

    return nc_lik + nc_prior - nc_post


def get_ev(m, l, v, S):
    """Evaluate the expectation of parameters given hyperparameters.

    Parameters
    ----------
    m, l, v, S : see module docstring

    Returns
    -------
    tuple (np.ndarray, np.ndarray)
        parameter expectations (mu, Sig)
    """

    return (m, S / (v - len(m) - 1))


def get_mode(m, l, v, S):
    """Evaluate the mode of parameters given hyperparameters.

    Parameters
    ----------
    m, l, v, S : see module docstring

    Returns
    -------
    tuple (np.ndarray, np.ndarray)
        parameter modes (mu, Sig)
    """

    return (m, S / (v + len(m) + 1))
