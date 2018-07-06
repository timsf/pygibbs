"""
Provides routines for Bayesian linear regression, where
   Y[i·]|(X, Bet, sig2) ~ Mv-Normal(Bet[i·] * X, sig2[i] * I)
   where Y are responses, X are covariates, Bet is the coefficient matrix, sig is the standard deviation of residuals and I is the identity.

The prior distribution over (Bet, sig2) is normal-scaled-inv-chi^2:
   Bet[i·]|(sig2, M0, L0) ~ Mv-Normal(M0[i·], sig2[i] * L0^-1)
   sig2[i]|(v0, s0) ~ Scaled-Inv-Chi^2(v0[i], s0[i])

The posterior distribution over (Bet, sig2) is normal-scaled-inv-chi^2:
   Bet[i·]|(Y[i·], sig2, M0, l0) ~ Mv-Normal(MN[i·], sig2[i] * LN^-1)
   sig2[i]|(Y[i·], v0, s0) ~ Scaled-Inv-Chi^2(vN[i], sN[i])

Data
----
Y : Matrix[nres, nobs]
X : Matrix[nvar, nobs]

Parameters
----------
Bet : Matrix[nres, nvar]
sig2 >= 0 : Vector[nres]

Hyperparameters
---------------
M : Matrix[nres, nvar]
L >= 0 (positive semi-definite): Matrix[nvar, nvar]
v >= 0: Vector[nres]
s >= 0: Vector[nres]
"""

import numpy as np
from scipy.special import gammaln
from scipy.stats import norm, matrix_normal

from pygibbs.tools.densities import eval_lm as eval_loglik


def update(Y, X, M0, L0, v0, s0):
    """Compute the hyperparameters of the posterior parameter distribution.

    Parameters
    ----------
    Y, X, M0, L0, v0, s0 : see module docstring

    Returns
    -------
    tuple (4 * np.ndarray)
        posterior hyperparameters (MN, LN, vN, sN)
    """

    S0 = M0 @ L0
    Sxx = X @ X.T
    Sxy = Y @ X.T
    syy = np.sum(np.square(Y), 1)
    
    LN = L0 + Sxx
    MN = np.linalg.solve(LN.T, (S0 + Sxy).T).T
    vN = v0 + Y.shape[1]
    sN = s0 + syy - np.sum(MN * (S0 + Sxy), 1) + np.sum(M0 * S0, 1)

    return (MN, LN, vN, sN)


def sample_param(ndraws, M, L, v, s):
    """Draw samples from the parameter distribution given hyperparameters.

    Parameters
    ----------
    ndraws : int in N+
        number of draws to be sampled
    M, L, v, s : see module docstring

    Returns
    -------
    tuple (np.ndarray in R^(ndraws, nvar+1, ntar), np.ndarray in R+^(ndraws, ntar))
        parameter draws (Bet, sig2)
    """

    T = np.linalg.inv(L)
    sig2 = s / np.random.chisquare(v, (ndraws, len(s)))
    Bet = np.array([
        matrix_normal.rvs(M, np.diag(sig2_i), T, 1) for sig2_i in sig2])

    return (Bet, sig2)


def sample_data(ndraws, X, M, L, v, s):
    """Draw samples from the marginal data distribution given hyperparameters.

    Parameters
    ----------
    ndraws : int in N+
        number of draws to be sampled
    X, M, L, v, s : see module docstring

    Returns
    -------
    np.ndarray in R^(ndraws, ntar, nobs)
        data draws
    """

    Bet, sig2 = sample_param(ndraws, M, L, v, s)
    Mu = np.tensordot(Bet, X, 1)

    Y = np.random.normal(Mu, np.sqrt(sig2)[:,:,np.newaxis])

    return Y


def eval_logmargin(Y, X, M0, L0, v0, s0):
    """Evaluate the log marginal likelihood or evidence given data and hyperparameters. You can evaluate the predictive density by passing posterior instead of prior hyperparameters.

    Parameters
    ----------
    Y, X, M0, L0, v0, s0 : see module docstring

    Returns
    -------
    float
        log marginal likelihood
    """
    MN, LN, vN, sN = update(Y, X, M0, L0, v0, s0)

    logdet = lambda A: np.prod(np.linalg.slogdet(A))
    nc_lik = -Y.shape[1] / 2 * np.log(2 * np.pi)
    nc_prior = -gammaln(v0 / 2) + v0 / 2 * np.log(s0) + logdet(L0) / 2
    nc_post = -gammaln(vN / 2) + vN / 2 * np.log(sN) + logdet(LN) / 2

    return np.sum(nc_lik + nc_prior - nc_post)


def get_ev(M, L, v, s):
    """Evaluate the expectation of parameters given hyperparameters.

    Parameters
    ----------
    M, L, v, s : see module docstring

    Returns
    -------
    tuple (np.ndarray, np.ndarray)
        parameter expectations (Bet, sig2)
    """

    return (M, s / (v - 2))


def get_mode(M, L, v, s):
    """Evaluate the mode of parameters given hyperparameters.

    Parameters
    ----------
    M, L, v, s : see module docstring

    Returns
    -------
    tuple (np.ndarray, np.ndarray)
        parameter modes (Bet, sig2)
    """

    return (M, s / (v + 2))
