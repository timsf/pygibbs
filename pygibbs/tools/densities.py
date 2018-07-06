import numpy as np
from scipy.stats import multivariate_normal, norm


def eval_mvnorm(X, mu, Sig):
    """Evaluate the log likelihood given data and parameters.

    Parameters
    ----------
    X, mu, Sig :

    Returns
    -------
    np.ndarray in R^(nobs)
        log density at each observation
    """

    return multivariate_normal.logpdf(X, mu, Sig)


def eval_norm(X, mu, sig2):
    """Evaluate the log likelihood given data and parameters.

    Parameters
    ----------
    X, mu, sig2 :

    Returns
    -------
    np.ndarray in R^(nobs)
        log density at each observation
    """

    return np.nansum(norm.logpdf(X, mu, np.sqrt(sig2)), 1)


def eval_lm(Y, X, Bet, sig2):
    """Evaluate the log likelihood given data and parameters.

    Parameters
    ----------
    Y, X, Bet, sig2 :

    Returns
    -------
    np.ndarray in R^nobs
        log marginal likelihood
    """

    Mu = np.dot(Bet, X)

    return np.nansum(norm.logpdf(Y.T, Mu.T, np.sqrt(sig2)), 1)


def eval_mvlm(Y, X, Bet, Sig):
    """Evaluate the log likelihood given data and parameters.

    Parameters
    ----------
    Y, X, Bet, Sig :

    Returns
    -------
    np.ndarray in R^nvar
        log marginal likelihood
    """

    Mu = np.dot(Bet, X)

    return multivariate_normal.logpdf((Y - Mu).T, cov=Sig)