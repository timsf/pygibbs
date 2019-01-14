"""
Provides routines for Bayesian linear regression, ie:
   y|(x, bet, sig) ~ Mv-Normal(bet * x, sig)
   where y are responses, x are covariates, bet is the coefficient matrix, sig is the residual variance and I is the identity.

The prior distribution over (bet, sig) is normal-wishart:
   bet|(sig, m, l) ~ Mat-Normal(m, l^-1, sig)
   sig|(v, s) ~ Inv-Wishart(v, s)

The posterior distribution over (bet, sig) is normal-wishart:
   bet|(y, sig, m, l) ~ Mat-Normal(mN, lN^-1, sig)
   sig|(y, v, s) ~ Inv-Wishart(vN, sN)
"""

from typing import Tuple

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import multigammaln
from scipy.stats import invwishart

from pygibbs.tools.densities import eval_mvnorm
from pygibbs.tools.linalg import precond_solve_pd, logdet_pd


Data = Tuple[np.ndarray, np.ndarray]
Param = Tuple[np.ndarray, np.ndarray]
Hyper = Tuple[np.ndarray, np.ndarray, float, np.ndarray]


def update(y: np.ndarray, x: np.ndarray, m: np.ndarray, l: np.ndarray, v: float, s: np.ndarray, w: np.ndarray = None) -> Hyper:
    """Compute the hyperparameters of the posterior parameter distribution.

    :param y: [nres, nobs]
    :param x: [nvar, nobs]
    :param m: [nres, nvar]
    :param l: (PD)[nvar, nvar]
    :param v: (>0)
    :param s: (>0)[nvar]
    :param w: (>0)[nobs]
    :returns: posterior hyperparameters
    """

    if w is not None:
        x = x @ np.diag(w)
        y = y @ np.diag(w)

    ss0 = m @ l
    sxx = x @ x.T
    sxy = y @ x.T
    syy = y @ y.T

    lN = l + sxx
    mN = precond_solve_pd(lN.T, (ss0 + sxy).T).T
    vN = v + y.shape[1]
    sN = s + syy - mN @ (ss0 + sxy).T + m @ ss0.T

    return mN, lN, vN, sN


def sample_param(ndraws: int, m: np.ndarray, l: np.ndarray, v: float, s: np.ndarray) -> Param:
    """Draw samples from the parameter distribution given hyperparameters.

    :param ndraws: (>0) number of samples to be drawn
    :param m: [nres, nvar]
    :param l: (PD)[nvar, nvar]
    :param v: (>0)
    :param s: (>0)[nvar]
    :returns: samples from the parameter distribution
    """

    nres, nvar = m.shape
    z = np.random.standard_normal((ndraws, *m.shape))
    sig = invwishart.rvs(v, s, ndraws).reshape((ndraws, *s.shape))

    sqrt_li = solve_triangular(np.linalg.cholesky(l), np.identity(nvar), lower=True)
    bet = m + np.array([np.linalg.cholesky(sig_i) @ z_i @ sqrt_li for z_i, sig_i in zip(z, sig)])

    return bet, sig


def sample_data(ndraws: int, x: np.ndarray, m: np.ndarray, l: np.ndarray, v: float, s: np.ndarray) -> np.ndarray:
    """Draw samples from the marginal data distribution given hyperparameters.

    :param ndraws: (>0) number of samples to be drawn
    :param x: [nvar, nobs]
    :param m: [nres, nvar]
    :param l: (PD)[nvar, nvar]
    :param v: (>0)
    :param s: (>0)[nvar]
    :returns: samples from the data distribution
    """

    nres, nobs = m.shape[0], x.shape[1]
    z = np.random.standard_normal((ndraws, nres, nobs))
    bet, sig = sample_param(ndraws, m, l, v, s)

    y = np.array([bet_i @ x + np.linalg.cholesky(sig_i) @ z_i for z_i, bet_i, sig_i in zip(z, bet, sig)])

    return y


def eval_logmargin(y: np.ndarray, x: np.ndarray, m: np.ndarray, l: np.ndarray, v: float, s: np.ndarray, w: np.ndarray = None) -> float:
    """Evaluate the log marginal likelihood or evidence given data and hyperparameters.
    You can evaluate the predictive density by passing posterior instead of prior hyperparameters.

    :param y: [nres, nobs]
    :param x: [nvar, nobs]
    :param m: [nres, nvar]
    :param l: (PD)[nvar, nvar]
    :param v: (>0)
    :param s: (>0)[nvar]
    :param w: (>0)[nobs]
    :returns: log marginal likelihood
    """

    def nc(_, l0, v0, s0):
        return y.shape[0] * logdet_pd(l0) / 2 - multigammaln(v0 / 2, y.shape[0]) + v0 / 2 * logdet_pd(s0)

    return float(np.sum(nc(m, l, v, s) - nc(*update(y, x, m, l, v, s, w)) - np.prod(y.shape) / 2 * np.log(np.pi)))


def eval_loglik(y: np.ndarray, x: np.ndarray, bet: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """Evaluate the log density given parameters

    :param y: [nres, nobs]
    :param x: [nvar, nobs]
    :param bet: [nres, nvar]
    :param sig: [nres]
    :returns: log likelihood
    """

    return eval_mvnorm(y.T, (bet @ x).T, sig)


def get_ev(m: np.ndarray, l: np.ndarray, v: float, s: np.ndarray) -> Param:
    """Evaluate the expectation of parameters given hyperparameters.

    :param m: [nres, nvar]
    :param l: (PD)[nvar, nvar]
    :param v: (>0)
    :param s: (>0)[nvar]
    :returns: parameter expectations
    """

    return m, s / (v - s.shape[0] - 1)


def get_mode(m: np.ndarray, l: np.ndarray, v: float, s: np.ndarray) -> Param:
    """Evaluate the mode of parameters given hyperparameters.

    :param m: [nres, nvar]
    :param l: (PD)[nvar, nvar]
    :param v: (>0)
    :param s: (>0)[nvar]
    :returns: parameter modes
    """

    return m, s / (v + s.shape[0] + 1)


def _generate_fixture(nres: int = 3, nobs: int = 2, nvar: int = 1, seed: int = 666) -> (Data, Param, Hyper):
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
    sig = np.identity(nres)

    # set input
    x = np.random.standard_normal((nvar, nobs))
    y = bet @ x + np.random.standard_normal((nres, nobs))

    # set hyperparameters
    m = np.zeros((nres, nvar))
    l = np.identity(nvar)
    v = nres
    s = np.ones(nres) * v

    return (y, x), (bet, sig), (m, l, v, s)
