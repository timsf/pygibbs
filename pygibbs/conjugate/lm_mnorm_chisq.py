"""
Provides routines for Bayesian linear regression, ie:
   y[i·]|(x, bet, sig) ~ Mv-Normal(bet[i·] * x, sig[i] * I)
   where y are responses, x are covariates, bet is the coefficient matrix, sig is the standard deviation of residuals and I is the identity.

The prior distribution over (bet, sig) is normal-scaled-inv-chi^2:
   bet[i·]|(sig, m, l) ~ Mv-Normal(m[i·], sig[i] * l^-1)
   sig[i]|(v, s) ~ Scaled-Inv-Chi^2(v[i], s[i])

The posterior distribution over (bet, sig) is normal-scaled-inv-chi^2:
   bet[i·]|(y[i·], sig, m, l) ~ Mv-Normal(mN[i·], sig[i] * lN^-1)
   sig[i]|(y[i·], v, s) ~ Scaled-Inv-Chi^2(vN[i], sN[i])
"""

from typing import Tuple

import numpy as np
from scipy.special import gammaln
from scipy.linalg import solve, solve_triangular

from pygibbs.tools.densities import eval_lm as eval_loglik
from pygibbs.tools.linalg import eval_matquad, logdet_pd


Data = Tuple[np.ndarray, np.ndarray]
Param = Tuple[np.ndarray, np.ndarray]
Hyper = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def update(y: np.ndarray, x: np.ndarray, m: np.ndarray, l: np.ndarray, v: np.ndarray, s: np.ndarray, w: np.ndarray = None) -> Hyper:
    """Compute the hyperparameters of the posterior parameter distribution.

    :param y: [nres, nobs]
    :param x: [nvar, nobs]
    :param m: [nres, nvar]
    :param l: (PD)[nvar, nvar]
    :param v: (>0)[nvar]
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
    syy = np.sum(np.square(y), 1)

    lN = l + sxx
    mN = solve(lN.T, (ss0 + sxy).T, sym_pos=True).T
    vN = v + y.shape[1]
    sN = s + syy - np.sum(mN * (ss0 + sxy), 1) + np.sum(m * ss0, 1)

    return mN, lN, vN, sN


def sample_param(ndraws: int, m: np.ndarray, l: np.ndarray, v: np.ndarray, s: np.ndarray) -> Param:
    """Draw samples from the parameter distribution given hyperparameters.

    :param ndraws: (>0) number of samples to be drawn
    :param m: [nres, nvar]
    :param l: (PD)[nvar, nvar]
    :param v: (>0)[nvar]
    :param s: (>0)[nvar]
    :returns: samples from the parameter distribution
    """

    nres, nvar = m.shape
    z = np.random.standard_normal((ndraws, *m.shape))
    sig = s / np.random.chisquare(v, (ndraws, nres))

    sqrt_li = solve_triangular(np.linalg.cholesky(l), np.identity(nvar), lower=True)
    sqrt_sig2 = np.sqrt(sig)
    bet = m + np.array([sqrt_sig2_i[:, np.newaxis] * z_i @ sqrt_li for z_i, sqrt_sig2_i in zip(z, sqrt_sig2)])

    return bet, sig


def sample_data(ndraws: int, x: np.ndarray, m: np.ndarray, l: np.ndarray, v: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Draw samples from the marginal data distribution given hyperparameters.

    :param ndraws: (>0) number of samples to be drawn
    :param x: [nvar, nobs]
    :param m: [nres, nvar]
    :param l: (PD)[nvar, nvar]
    :param v: (>0)[nvar]
    :param s: (>0)[nvar]
    :returns: samples from the data distribution
    """

    nres, nobs = m.shape[0], x.shape[1]
    z = np.random.standard_normal((ndraws, nres, nobs))
    bet, sig = sample_param(ndraws, m, l, v, s)

    sqrt_sig2 = np.sqrt(sig)
    y = np.array([bet_i @ x + sqrt_sig2_i[:, np.newaxis] * z_i for z_i, bet_i, sqrt_sig2_i in zip(z, bet, sqrt_sig2)])

    return y


def eval_logmargin(y: np.ndarray, x: np.ndarray, m: np.ndarray, l: np.ndarray, v: np.ndarray, s: np.ndarray, w: np.ndarray = None) -> float:
    """Evaluate the log marginal likelihood or evidence given data and hyperparameters.
    You can evaluate the predictive density by passing posterior instead of prior hyperparameters.

    :param y: [nres, nobs]
    :param x: [nvar, nobs]
    :param m: [nres, nvar]
    :param l: (PD)[nvar, nvar]
    :param v: (>0)[nvar]
    :param s: (>0)[nvar]
    :param w: (>0)[nobs]
    :returns: log marginal likelihood
    """

    def nc(_, l0, v0, s0):
        return logdet_pd(l0) / 2 - gammaln(v0 / 2) + v0 / 2 * np.log(s0)

    return float(np.sum(nc(m, l, v, s) - nc(*update(y, x, m, l, v, s, w)) - y.shape[1] / 2 * np.log(np.pi)))


def get_ev(m: np.ndarray, l: np.ndarray, v: np.ndarray, s: np.ndarray) -> Param:
    """Evaluate the expectation of parameters given hyperparameters.

    :param m: [nres, nvar]
    :param l: (PD)[nvar, nvar]
    :param v: (>0)[nvar]
    :param s: (>0)[nvar]
    :returns: parameter expectations
    """

    return m, s / (v - 2)


def get_mode(m: np.ndarray, l: np.ndarray, v: np.ndarray, s: np.ndarray) -> Param:
    """Evaluate the mode of parameters given hyperparameters.

    :param m: [nres, nvar]
    :param l: (PD)[nvar, nvar]
    :param v: (>0)[nvar]
    :param s: (>0)[nvar]
    :returns: parameter modes
    """

    return m, s / (v + 2)


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
    sig = np.ones(nres)

    # set input
    x = np.random.standard_normal((nvar, nobs))
    y = bet @ x + np.random.standard_normal((nres, nobs))

    # set hyperparameters
    m = np.zeros((nres, nvar))
    l = np.diag(np.ones(nvar))
    v = np.ones(nres) + 2
    s = np.ones(nres) * v

    return (y, x), (bet, sig), (m, l, v, s)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
