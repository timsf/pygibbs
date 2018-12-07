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
l : np.ndarray(PSD)[nvar, nvar] # coefficient dof matrix
v : np.ndarray(>0)[nres] # residual variance dof vector
s : np.ndarray(>0)[nres] # residual variance location vector
"""

import numpy as np
from scipy.special import gammaln
from scipy.linalg import solve, solve_triangular

from pygibbs.tools.densities import eval_lm as eval_loglik, eval_matquad


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

    >>> data, param = _generate_fixture(3, 2, 1, seed=666)
    >>> y, x = data
    >>> m, l, v, s = param
    >>> update(y, x, m, l, v, s)
    (array([[ 0.7349347 ],
           [-0.27427018],
           [-0.22898939]]), array([[1.90965336]]), array([5., 5., 5.]), array([4.17193638, 3.19520305, 3.79091217]))
    >>> w = np.ones(2)
    >>> update(y, x, m, l, v, s, w=w)
    (array([[ 0.7349347 ],
           [-0.27427018],
           [-0.22898939]]), array([[1.90965336]]), array([5., 5., 5.]), array([4.17193638, 3.19520305, 3.79091217]))
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

    return (mN, lN, vN, sN)


def marginalize(x: np.ndarray, m: np.ndarray, l: np.ndarray, v: np.ndarray, s: np.ndarray) -> 4 * (np.ndarray,):
    """Compute the parameters of the marginal likelihood

    :param x:
    :param m:
    :param l:
    :param v:
    :param s:
    :returns: parameters of marginal likelihood

    >>> data, param = _generate_fixture(3, 2, 1, seed=666)
    >>> _, x = data
    >>> m, l, v, s = param
    >>> marginalize(x, m, l, v, s)
    (array([[0., 0.],
           [0., 0.],
           [0., 0.]]), array([1., 1., 1.]), array([[0.320714  , 0.60441774],
           [0.60441774, 0.76963264]]), array([3., 3., 3.]))
    """

    return (m @ x, s / v, np.identity(x.shape[0]) - eval_matquad(x.T, l), v)


def sample_param(ndraws: int, m: np.ndarray, l: np.ndarray, v: np.ndarray, s: np.ndarray) -> 2 * (np.ndarray,):
    """Draw samples from the parameter distribution given hyperparameters.

    :param ndraws: number of samples to be drawn
    :param m:
    :param l:
    :param v:
    :param s:
    :returns: samples from the parameter distribution

    >>> _, param = _generate_fixture(3, 2, 1, seed=666)
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

    return (bet, sig)


def sample_data(ndraws: int, x: np.ndarray, m: np.ndarray, l: np.ndarray, v: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Draw samples from the marginal data distribution given hyperparameters.

    :param ndraws: number of samples to be drawn
    :param x:
    :param m:
    :param l:
    :param v:
    :param s:
    :returns: samples from the data distribution

    >>> data, param = _generate_fixture(3, 2, 1, seed=666)
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

    >>> data, param = _generate_fixture(3, 2, 1, seed=666)
    >>> y, x = data
    >>> m, l, v, s = param
    >>> eval_logmargin(y, x, m, l, v, s)
    -8.051015631484027
    >>> w = np.ones(2)
    >>> eval_logmargin(y, x, m, l, v, s, w=w)
    -8.051015631484027
    """

    def nc(_, l, v, s):
         return np.prod(np.linalg.slogdet(l)) / 2 - gammaln(v / 2) + v / 2 * np.log(s)

    return float(np.sum(nc(m, l, v, s) - nc(*update(y, x, m, l, v, s, w)) - y.shape[1] / 2 * np.log(np.pi)))


def get_ev(m: np.ndarray, _, v: np.ndarray, s: np.ndarray) -> 2 * (np.ndarray,):
    """Evaluate the expectation of parameters given hyperparameters.

    :param m:
    :param l:
    :param v:
    :param s:
    :returns: parameter expectations

    >>> _, param = _generate_fixture(3, 2, 1, seed=666)
    >>> m, l, v, s = param
    >>> get_ev(m, l, v, s)
    (array([[0.],
           [0.],
           [0.]]), array([3., 3., 3.]))
    """

    return (m, s / (v - 2))


def get_mode(m: np.ndarray, _, v: np.ndarray, s: np.ndarray) -> 2 * (np.ndarray,):
    """Evaluate the mode of parameters given hyperparameters.

    :param m:
    :param l:
    :param v:
    :param s:
    :returns: parameter modes

    >>> _, param = _generate_fixture(3, 2, 1, seed=666)
    >>> m, l, v, s = param
    >>> get_mode(m, l, v, s)
    (array([[0.],
           [0.],
           [0.]]), array([0.6, 0.6, 0.6]))
    """

    return (m, s / (v + 2))


def _generate_fixture(nres: int, nobs: int, nvar: int, seed: int=666) -> (2 * (np.ndarray,), 4 * (np.ndarray,)):
    """Generate a set of input data.

    :param nres:
    :param nobs:
    :param nvar:
    :param seed: random number generator seed
    :returns: generated data, generated hyperparameters

    >>> data, param = _generate_fixture(3, 2, 1, seed=666)
    >>> data
    (array([[ 1.17346801,  0.90904807],
           [-0.57172145, -0.10949727],
           [ 0.01902826, -0.94376106]]), array([[0.82418808, 0.479966  ]]))
    >>> param
    (array([[0.],
           [0.],
           [0.]]), array([[1.]]), array([3., 3., 3.]), array([3., 3., 3.]))
    """

    # ensure deterministic output
    np.random.seed(seed)

    # set input
    x = np.random.standard_normal((nvar, nobs))
    y = np.random.standard_normal((nres, nobs))

    # set hyperparameters
    m = np.zeros((nres, nvar))
    l = np.diag(np.ones(nvar))
    v = np.ones(nres) + 2
    s = np.ones(nres) * v

    return ((y, x), (m, l, v, s))


if __name__ == '__main__':
    import doctest
    doctest.testmod()
