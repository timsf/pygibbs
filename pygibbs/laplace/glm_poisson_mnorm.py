"""
Provides routines for Bayesian poisson regression, ie:
   y[i]|(x, bet) ~ Poisson(exp(x[i·] * bet))
   where y are responses, x are covariates and bet is the coefficient matrix

The prior distribution over (bet, sig) is normal:
   bet|(m, l) ~ Mv-Normal(m, l^-1)

The posterior distribution over (bet, sig) is approximately normal:
   bet|(y, m, l) ~ Mv-Normal(mN, lN^-1)
"""

from typing import Tuple

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import gammaln, logsumexp
from scipy.stats import poisson

from pygibbs.tools.laplace import fit_approx, est_integral
from pygibbs.tools.linalg import logdet_pd


Data = Tuple[np.ndarray, np.ndarray]
Param = Tuple[np.ndarray]
Hyper = Tuple[np.ndarray, np.ndarray]


def update(y: np.ndarray, x: np.ndarray, m: np.ndarray, l: np.ndarray, w: np.ndarray = None) -> Hyper:
    """Compute the hyperparameters of the posterior parameter distribution.

    :param y: [nobs]
    :param x: [nobs, nvar]
    :param m: [nres, nvar]
    :param l: (PD)[nvar, nvar]
    :param w: (>0)[nobs]
    :returns: posterior hyperparameters
    """

    if w is not None:
        x = x @ np.diag(w)
        y = y @ np.diag(w)

    def log_posterior(bet):
        lin_pred = x @ bet
        gam = bet - m
        return y @ lin_pred - np.exp(logsumexp(lin_pred)) - gam @ l @ gam / 2

    def grad_log_posterior(bet):
        fitted = np.exp(x @ bet)
        return (y - fitted) @ x - l @ (bet - m)

    def hess_log_posterior(bet):
        fitted = np.exp(x @ bet)
        return -x.T @ np.diag(fitted) @ x - l

    return fit_approx(np.zeros(m.shape), log_posterior, grad_log_posterior, hess_log_posterior)


def sample_param(ndraws: int, m: np.ndarray, l: np.ndarray) -> Param:
    """Draw samples from the parameter distribution given hyperparameters.

    :param ndraws: (>0) number of samples to be drawn
    :param m: [nres, nvar]
    :param l: (PD)[nvar, nvar]
    :returns: samples from the parameter distribution
    """

    z = np.random.standard_normal((ndraws, m.shape[0]))

    return m + solve_triangular(np.linalg.cholesky(l).T, z.T).T,


def sample_data(ndraws: int, x: np.ndarray, m: np.ndarray, l: np.ndarray) -> np.ndarray:
    """Draw samples from the marginal data distribution given hyperparameters.

    :param ndraws: (>0) number of samples to be drawn
    :param x: [nobs, nvar]
    :param m: [nres, nvar]
    :param l: (PD)[nvar, nvar]
    :returns: samples from the parameter distribution
    """

    bet, = sample_param(ndraws, m, l)

    return np.random.poisson(np.exp(bet @ x))


def eval_logmargin(y: np.ndarray, x: np.ndarray, m: np.ndarray, l: np.ndarray, w: np.ndarray = None) -> float:
    """Evaluate the log marginal likelihood or evidence given data and hyperparameters.
    You can evaluate the predictive density by passing posterior instead of prior hyperparameters.

    :param y: [nobs]
    :param x: [nobs, nvar]
    :param m: [nres, nvar]
    :param l: (PD)[nvar, nvar]
    :param w: (>0)[nobs]
    :returns: log marginal likelihood
    """

    def log_lik(bet):
        lin_pred = x @ bet
        return y @ lin_pred - np.exp(logsumexp(lin_pred)) - np.sum(gammaln(y + 1))

    def log_prior(bet):
        gam = bet - m
        return (logdet_pd(l) - m.shape[0] * np.log(2 * np.pi) - gam @ l @ gam) / 2

    return est_integral(1000, *update(y, x, m, l, w), log_lik, log_prior)[0]


def eval_loglik(y: np.ndarray, x: np.ndarray, bet: np.ndarray) -> np.ndarray:
    """Evaluate the log likelihood given data and parameters.

    :param y: [nobs]
    :param x: [nobs, nvar]
    :returns: log likelihood
    """

    return poisson.logpmf(y, np.exp(x @ bet))


def get_ev(m: np.ndarray, l: np.ndarray) -> Param:
    """Evaluate the expectation of parameters given hyperparameters.

    :param m: [nres, nvar]
    :param l: (PD)[nvar, nvar]
    :returns: parameter expectations
    """

    return m,


def get_mode(m: np.ndarray, l: np.ndarray) -> Param:
    """Evaluate the mode of parameters given hyperparameters.

    :param m: [nres, nvar]
    :param l: (PD)[nvar, nvar]
    :returns: parameter modes
    """

    return m,


def _generate_fixture(nobs: int = 3, nvar: int = 2, seed: int = 666) -> (Data, Param, Hyper):
    """Generate a set of input data.

    :param nobs: (>0)
    :param nvar: (>0)
    :param seed: random number generator seed
    :returns: generated data, ground truth, hyperparameters
    """

    # ensure deterministic output
    np.random.seed(seed)

    # set ground truth
    bet = np.ones(nvar)

    # set input
    x = np.random.standard_normal((nobs, nvar))
    y = np.random.poisson(np.exp(x @ bet))

    # set hyperparameters
    m = np.zeros(nvar)
    l = np.identity(nvar)

    return (y, x), (bet,), (m, l)
