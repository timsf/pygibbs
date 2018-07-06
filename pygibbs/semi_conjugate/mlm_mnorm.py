"""
Factor modelling library for models of the form:
   Y[i·]|(X, Bet, Sig) ~ Mv-Normal(Bet[i·] * X, Sig)
   Bet[i·]|(m0, S0) ~ Mv-Normal(m0, S0)
   where Y are responses, X are covariates, Bet is the coefficient matrix and Sig a covariance matrix.

The prior distribution over Bet is normal:
   Bet[i·]|(m0, S0) ~ Mv-Normal(m0, S0)

The posterior distribution over Bet is normal:
   Bet[i·]|(Y[i·], m0, S0) ~ Mv-Normal(MN[i], SN)

Data
----
Y : Matrix[nres, nobs]
X : Matrix[nvar, nobs]

Parameters
----------
Bet : Matrix[nres, nvar]
Sig >= 0 (positive semi-definite, known) : Matrix[nres, nres]

Hyperparameters
---------------
m : Vector[nvar]
S : Matrix[nvar, nvar]
"""

import numpy as np

from pygibbs.tools.densities import eval_mvnorm as eval_loglik


def marginalize(X, Sig, m0, S0):
    """Marginalize y ~ N(bX, Sig) over b ~ N(m0, S0).

    Parameters
    ----------
    X : np.ndarray
        linear transformation of integration variable
    Sig : np.ndarray
        covariance of marginal variable
    m0 : np.ndarray
        mean of integration variable
    S0 : np.ndarray
        covariance of integration variable

    Returns
    -------
    tuple (np.ndarray in R, np.ndarray in PSD)
        marginal mean, marginal covariance
    """

    return (m0 @ X, X.T @ S0 @ X + Sig)


def update(Y, X, Sig, m0, S0):
    """Update B ~ N(m0, S0) after observing Y ~ N(BX, Sig).

    Parameters
    ----------
    Y : np.ndarray
        observations
    X : np.ndarray
        linear transformation of random variable
    Sig : np.ndarray
        covariance of observation
    m0 : np.ndarray
        prior mean
    S0 : np.ndarray
        prior covariance

    Returns
    -------
    tuple (np.ndarray in R, np.ndarray in PSD)
        posterior mean, posterior covariance
    """

    mX, SX = marginalize(X, Sig, m0, S0)

    S0X = S0 @ X
    SXi = np.linalg.inv(SX)

    MN = m0 + (S0X @ SXi @ (Y - mX).T).T
    SN = S0 - (S0X @ SXi @ S0X.T)

    return (np.where(np.isnan(MN), m0, MN), SN)


def sample_param(ndraws, M, S):
    """Draw samples from the parameter distribution given hyperparameters.

    Parameters
    ----------
    ndraws : int in N+
        number of draws to be sampled
    M, S : see module docstring

    Returns
    -------
    tuple (np.ndarray in R^(ndraws, nvar+1, ntar))
        parameter draws
    """

    return (M + np.random.multivariate_normal(np.zeros(M.shape[1]), S, (ndraws, M.shape[0])),)


def sample_data(ndraws, X, Sig, M, S):
    """Draw samples from the marginal data distribution given hyperparameters.

    Parameters
    ----------
    ndraws : int in N+
        number of draws to be sampled
    X, Sig, M, S : see module docstring

    Returns
    -------
    np.ndarray in R^(ndraws, ntar, nobs)
        data draws
    """

    Bet = sample_param(ndraws, M, S)[0]

    Y = Bet @ X + np.random.multivariate_normal(np.zeros(X.shape[1]), Sig, (ndraws, M.shape[0]))

    return Y


def eval_logmargin(Y, X, Sig, m0, S0):
    """Evaluate the observed likelihood.

    Parameters
    ----------
    Y : np.ndarray
        observations
    X : np.ndarray
        linear transformation of random variable
    Sig : np.ndarray
        covariance of observation
    m0 : np.ndarray
        prior mean
    S0 : np.ndarray
        prior covariance

    Returns
    -------
    float
        log observed likelihood
    """

    margin = marginalize(X, Sig, m0, S0)

    return np.nansum(eval_loglik(Y, *margin))


def get_ev(M, S):
    """Evaluate the expectation of parameters given hyperparameters.

    Parameters
    ----------
    M, S : see module docstring

    Returns
    -------
    tuple (np.ndarray,)
        parameter expectations (Bet,)
    """

    return (M,)


def get_mode(M, S):
    """Evaluate the mode of parameters given hyperparameters.

    Parameters
    ----------
    M, S : see module docstring

    Returns
    -------
    tuple (np.ndarray,)
        parameter modes (Bet,)
    """

    return (M,)
