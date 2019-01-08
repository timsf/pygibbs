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

Data
----
nres : int(>0) # number of responses
nobs : int(>0) # number of observations per response
nvar : int(>0) # number of covariates
y : np.ndarray[nres, nobs] # response matrix
x : np.ndarray[nvar, nobs] # covariate matrix
w : np.ndarray(>0)[nobs] # observation weights

Parameters
----------
bet : np.ndarray[nres, nvar] # coefficient matrix
sig : np.ndarray(>0)[nres] # residual variance vector

Hyperparameters
---------------
m : np.ndarray[nres, nvar] # coefficient location matrix
l : np.ndarray(PD)[nvar, nvar] # coefficient dof matrix
v : np.ndarray(>0)[nres] # residual variance dof vector
s : np.ndarray(>0)[nres] # residual variance location vector
"""

import numpy as np
from scipy.special import gammaln
from scipy.linalg import solve, solve_triangular

from pygibbs.tools.densities import eval_lm as eval_loglik, eval_mvt
from pygibbs.tools.linalg import eval_matquad, logdet_pd


def update(y: np.ndarray, x: np.ndarray, m: np.ndarray, l: np.ndarray, v: np.ndarray, s: np.ndarray, w: np.ndarray=None
           ) -> 4 * (np.ndarray,):
    """Compute the hyperparameters of the posterior parameter distribution.

    :param y:
    :param x:
    :param m:
    :param l:
    :param v:
    :param s:
    :param w:
    :returns: posterior hyperparameters

    >>> data, _, param = _generate_fixture(3, 2, 1, seed=666)
    >>> y, x = data
    >>> m, l, v, s = param
    >>> update(y, x, m, l, v, s)
    (array([[1.21127945],
           [0.20207457],
           [0.24735536]]), array([[1.90965336]]), array([5., 5., 5.]), array([6.11815054, 3.12300744, 3.80927815]))
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


def marginalize(x: np.ndarray, m: np.ndarray, l: np.ndarray, v: np.ndarray, s: np.ndarray) -> 4 * (np.ndarray,):
    """Compute the parameters of the marginal likelihood

    :param x:
    :param m:
    :param l:
    :param v:
    :param s:
    :returns: parameters of marginal likelihood

    >>> data, _, param = _generate_fixture(3, 2, 1, seed=666)
    >>> _, x = data
    >>> m, l, v, s = param
    >>> marginalize(x, m, l, v, s)
    (array([[0., 0.],
           [0., 0.],
           [0., 0.]]), array([1., 1., 1.]), array([[1.679286  , 0.39558226],
           [0.39558226, 1.23036736]]), array([3., 3., 3.]))
    """

    return m @ x, s / v, np.identity(x.shape[1]) + eval_matquad(x.T, l), v


def sample_param(ndraws: int, m: np.ndarray, l: np.ndarray, v: np.ndarray, s: np.ndarray) -> 2 * (np.ndarray,):
    """Draw samples from the parameter distribution given hyperparameters.

    :param ndraws: number of samples to be drawn
    :param m:
    :param l:
    :param v:
    :param s:
    :returns: samples from the parameter distribution

    >>> _, _, param = _generate_fixture(3, 2, 1, seed=666)
    >>> m, l, v, s = param
    >>> sample_param(1, m, l, v, s)
    (array([[[ 1.20724636],
            [-1.43612397],
            [ 0.42275744]]]), array([[3.55184779, 3.33464167, 0.48209583]]))
    """

    nres, nvar = m.shape
    z = np.random.standard_normal((ndraws, *m.shape))
    sig = s / np.random.chisquare(v, (ndraws, nres))

    sqrt_li = solve_triangular(np.linalg.cholesky(l), np.identity(nvar), lower=True)
    sqrt_sig2 = np.sqrt(sig)
    bet = m + np.array([sqrt_sig2_i[:,np.newaxis] * z_i @ sqrt_li for z_i, sqrt_sig2_i in zip(z, sqrt_sig2)])

    return bet, sig


def sample_data(ndraws: int, x: np.ndarray, m: np.ndarray, l: np.ndarray, v: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Draw samples from the marginal data distribution given hyperparameters.

    :param ndraws: number of samples to be drawn
    :param x:
    :param m:
    :param l:
    :param v:
    :param s:
    :returns: samples from the data distribution

    >>> data, _, param = _generate_fixture(3, 2, 1, seed=666)
    >>> _, x = data
    >>> m, l, v, s = param
    >>> sample_data(1, x, m, l, v, s)
    array([[[ 0.59966755, -1.41389558],
            [ 0.25662034, -1.28957295],
            [ 0.07671978, -1.19512987]]])
    """

    nres, nobs = m.shape[0], x.shape[1]
    z = np.random.standard_normal((ndraws, nres, nobs))
    bet, sig = sample_param(ndraws, m, l, v, s)

    sqrt_sig2 = np.sqrt(sig)
    y = np.array([bet_i @ x + sqrt_sig2_i[:,np.newaxis] * z_i for z_i, bet_i, sqrt_sig2_i in zip(z, bet, sqrt_sig2)])

    return y


def eval_logmargin(y: np.ndarray, x: np.ndarray, m: np.ndarray, l: np.ndarray, v: np.ndarray, s: np.ndarray, w: np.ndarray=None
                   ) -> float:
    """Evaluate the log marginal likelihood or evidence given data and hyperparameters. you can evaluate the predictive density by passing posterior instead of prior hyperparameters.

    :param y:
    :param x:
    :param m:
    :param l:
    :param v:
    :param s:
    :param w:
    :returns: log marginal likelihood

    >>> data, _, param = _generate_fixture(3, 2, 1, seed=666)
    >>> y, x = data
    >>> m, l, v, s = param
    >>> eval_logmargin(y, x, m, l, v, s)
    -8.963161695276913
    """

    def nc(_, l, v, s):
         return logdet_pd(l) / 2 - gammaln(v / 2) + v / 2 * np.log(s)

    return float(np.sum(nc(m, l, v, s) - nc(*update(y, x, m, l, v, s, w)) - y.shape[1] / 2 * np.log(np.pi)))


def get_ev(m: np.ndarray, l: np.ndarray, v: np.ndarray, s: np.ndarray) -> 2 * (np.ndarray,):
    """Evaluate the expectation of parameters given hyperparameters.

    :param m:
    :param l:
    :param v:
    :param s:
    :returns: parameter expectations

    >>> _, _, param = _generate_fixture(3, 2, 1, seed=666)
    >>> m, l, v, s = param
    >>> get_ev(m, l, v, s)
    (array([[0.],
           [0.],
           [0.]]), array([3., 3., 3.]))
    """

    return m, s / (v - 2)


def get_mode(m: np.ndarray, l: np.ndarray, v: np.ndarray, s: np.ndarray) -> 2 * (np.ndarray,):
    """Evaluate the mode of parameters given hyperparameters.

    :param m:
    :param l:
    :param v:
    :param s:
    :returns: parameter modes

    >>> _, _, param = _generate_fixture(3, 2, 1, seed=666)
    >>> m, l, v, s = param
    >>> get_mode(m, l, v, s)
    (array([[0.],
           [0.],
           [0.]]), array([0.6, 0.6, 0.6]))
    """

    return m, s / (v + 2)


def _generate_fixture(nres: int = 3, nobs: int = 2, nvar: int = 1, seed: int = 666) -> (2 * (np.ndarray,), 2 * (np.ndarray,), 4 * (np.ndarray,)):
    """Generate a set of input data.

    :param nres:
    :param nobs:
    :param nvar:
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
