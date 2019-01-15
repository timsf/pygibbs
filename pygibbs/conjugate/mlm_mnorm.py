"""location
Provides routines for Bayesian multivariate linear regression, ie:
   y[i·]|(x, bet, sig) ~ Mv-Normal(bet[i·] * x, sig)
   where y are responses, x are covariates, bet is the coefficient matrix and sig a covariance matrix.

The prior distribution over (bet,) is normal:
   bet[i·]|(m, s) ~ Mv-Normal(m, s)

The posterior distribution over (bet,) is normal:
   bet[i·]|(y[i·], m, s) ~ Mv-Normal(mN[i·], sN)
"""

from typing import Tuple

import numpy as np
from scipy.linalg import solve_triangular

from pygibbs.tools.densities import eval_mvnorm


Data = Tuple[np.ndarray, np.ndarray, np.ndarray]
Param = Tuple[np.ndarray]
Hyper = Tuple[np.ndarray, np.ndarray]


def update(y: np.ndarray, x: np.ndarray, sig: np.ndarray, m: np.ndarray, s: np.ndarray, w: np.ndarray = None) -> Hyper:
    """Compute the hyperparameters of the posterior parameter distribution.

    :param y: [nres, nobs]
    :param x: [nvar, nobs]
    :param sig: (PD)[nobs, nobs]
    :param m: [nres, nvar]
    :param s: (PD)[nvar, nvar]
    :param w: (>0)[nobs]
    :returns: posterior hyperparameters
    """

    if w is not None:
        x = x @ np.diag(w)
        y = y @ np.diag(w)

    xs = x.T @ s
    l = np.linalg.cholesky(xs @ x + sig)
    a = solve_triangular(l, xs, lower=True)
    b = solve_triangular(l, (y - m @ x).T, lower=True)

    mN = m + b.T @ a
    sN = s - a.T @ a

    return np.where(np.isnan(mN), m, mN), sN


def sample_param(ndraws: int, m: np.ndarray, s: np.ndarray) -> Param:
    """Draw samples from the parameter distribution given hyperparameters.

    :param ndraws: (>0) number of samples to be drawn
    :param m: [nres, nvar]
    :param s: (PD)[nvar, nvar]
    :returns: samples from the parameter distribution
    """

    nres, nvar = m.shape
    z = np.random.standard_normal((ndraws, *m.shape))

    bet = m + np.array([z_i @ np.linalg.cholesky(s).T for z_i in z])

    return bet,


def sample_data(ndraws: int, x: np.ndarray, sig: np.ndarray, m: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Draw samples from the marginal data distribution given hyperparameters.

    :param ndraws: (>0) number of samples to be drawn
    :param x: [nvar, nobs]
    :param sig: (PD)[nobs, nobs]
    :param m: [nres, nvar]
    :param s: (PD)[nvar, nvar]
    :returns: samples from the data distribution
    """

    nres, nobs = m.shape[0], x.shape[1]
    z = np.random.standard_normal((ndraws, nres, nobs))
    bet, = sample_param(ndraws, m, s)
    y = bet @ x + np.array([np.linalg.cholesky(sig) @ z_i for z_i in z])

    return y


def eval_logmargin(y: np.ndarray, x: np.ndarray, sig: np.ndarray, m: np.ndarray, s: np.ndarray, w: np.ndarray = None) -> float:
    """Evaluate the log marginal likelihood or evidence given data and hyperparameters.
    You can evaluate the predictive density by passing posterior instead of prior hyperparameters.

    :param y: [nres, nobs]
    :param x: [nvar, nobs]
    :param sig: (PD)[nobs, nobs]
    :param m: [nres, nvar]
    :param s: (PD)[nvar, nvar]
    :param w: (>0)[nobs]
    :returns: log marginal likelihood
    """

    if w is not None:
        x = x @ np.diag(w)
        y = y @ np.diag(w)
    mx, sx = m @ x, x.T @ s @ x + sig

    return float(np.nansum(eval_mvnorm(y - mx, np.zeros(y.shape[1]), sx)))


def eval_loglik(y: np.ndarray, x: np.ndarray, sig: np.ndarray, bet: np.ndarray) -> np.ndarray:
    """Evaluate the log density given parameters

    :param y: [nres, nobs]
    :param x: [nvar, nobs]
    :param sig: (PD)[nobs, nobs]
    :param bet: [nres, nvar]
    :returns: log likelihood
    """

    return eval_mvnorm(y, bet @ x, sig)


def get_ev(m: np.ndarray, s: np.ndarray) -> Param:
    """Evaluate the expectation of parameters given hyperparameters.

    :param m: [nres, nvar]
    :param s: (PD)[nvar, nvar]
    :returns: parameter expectations
    """

    return m,


def get_mode(m: np.ndarray, s: np.ndarray) -> Param:
    """Evaluate the mode of parameters given hyperparameters.

    :param m: [nres, nvar]
    :param s: (PD)[nvar, nvar]
    :returns: parameter modes
    """

    return m,


def _generate_fixture(nres: int = 4, nobs: int = 3, nvar: int = 2, seed: int = 666) -> (Data, Param, Hyper):
    """Generate a set of input data.

    :param nres: (>0)
    :param nobs: (>0)
    :param nvar: (>0)
    :param seed: random number generator seed
    :returns: generated data, ground truth, hyperparameters
    """

    # ensure deterministic output
    np.random.seed(seed)

    # set ground truth
    bet = np.ones((nres, nvar))

    # set input
    x = np.random.standard_normal((nvar, nobs))
    y = bet @ x + np.random.standard_normal((nres, nobs))
    sig = np.identity(nobs)

    # set hyperparameters
    m = np.zeros((nres, nvar))
    s = np.identity(nvar)

    return (y, x, sig), (bet,), (m, s)
