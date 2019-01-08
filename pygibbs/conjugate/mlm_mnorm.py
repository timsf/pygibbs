"""
Provides routines for Bayesian multivariate linear regression, ie:
   y[i·]|(x, bet, sig) ~ Mv-Normal(bet[i·] * x, sig)
   where y are responses, x are covariates, bet is the coefficient matrix and sig a covariance matrix.

The prior distribution over (bet,) is normal:
   bet[i·]|(m, s) ~ Mv-Normal(m, s)

The posterior distribution over (bet,) is normal:
   bet[i·]|(y[i·], m, s) ~ Mv-Normal(mN[i·], sN)

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
sig : np.ndarray(PD)[nobs, nobs] # residual variance matrix (known)

Hyperparameters
---------------
m : np.ndarray[nvar] # coefficient location vector
s : np.ndarray(PD)[nvar, nvar] # coefficient scale matrix
"""

import numpy as np

from pygibbs.tools.densities import eval_mvlm as eval_loglik, eval_mvnorm


def update(y: np.ndarray, x: np.ndarray, sig: np.ndarray, m: np.ndarray, s: np.ndarray, w: np.ndarray=None) -> 2 * (np.ndarray,):
    """Compute the hyperparameters of the posterior parameter distribution.

    :param y:
    :param x:
    :param sig:
    :param m:
    :param s:
    :param w:
    :returns: posterior hyperparameters

    >>> data, _, param = _generate_fixture(3, 2, 1, seed=666)
    >>> y, x, sig = data
    >>> m, s = param
    >>> update(y, x, sig, m, s)
    (array([[1.21127945],
           [0.20207457],
           [0.24735536]]), array([[0.52365525]]))
    """

    if w is not None:
        x = x @ np.diag(w)
    mx, sx = marginalize(x, sig, m, s)

    s0x = s @ x
    sxi = np.linalg.inv(sx)

    mN = m + (s0x @ sxi @ (y - mx).T).T
    sN = s - (s0x @ sxi @ s0x.T)

    return np.where(np.isnan(mN), m, mN), sN


def marginalize(x: np.ndarray, sig: np.ndarray, m: np.ndarray, s: np.ndarray) -> 2 * (np.ndarray,):
    """Compute the parameters of the marginal likelihood

    :param x:
    :param sig:
    :param m:
    :param s:
    :returns: parameters of marginal likelihood

    >>> data, _, param = _generate_fixture(3, 2, 1, seed=666)
    >>> _, x, sig = data
    >>> m, s = param
    >>> marginalize(x, sig, m, s)
    (array([[0., 0.],
           [0., 0.],
           [0., 0.]]), array([[1.679286  , 0.39558226],
           [0.39558226, 1.23036736]]))
    """

    return m @ x, x.T @ s @ x + sig


def sample_param(ndraws: int, m: np.ndarray, s: np.ndarray) -> (np.ndarray,):
    """Draw samples from the parameter distribution given hyperparameters.

    :param ndraws: number of samples to be drawn
    :param m:
    :param s:
    :returns: samples from the parameter distribution

    >>> _, _, param = _generate_fixture(3, 2, 1, seed=666)
    >>> m, s = param
    >>> sample_param(1, m, s)
    (array([[[ 0.64057315],
            [-0.78644317],
            [ 0.60886999]]]),)
    """

    nres, nvar = m.shape
    bet = m + np.random.multivariate_normal(np.zeros(nvar), s, (ndraws, nres))

    return bet,


def sample_data(ndraws: int, x: np.ndarray, sig: np.ndarray, m: np.ndarray, s: np.ndarray):
    """Draw samples from the marginal data distribution given hyperparameters.

    :param ndraws: number of samples to be drawn
    :param x:
    :param sig:
    :param m:
    :param s:
    :returns: samples from the data distribution

    >>> data, _, param = _generate_fixture(3, 2, 1, seed=666)
    >>> _, x, sig = data
    >>> m, s = param
    >>> sample_data(1, x, sig, m, s)
    array([[[-0.40305909,  1.28567558],
            [-1.38509515, -0.6761986 ],
            [ 0.04123602, -0.79655609]]])
    """

    nres, nobs = m.shape[0], x.shape[1]
    bet, = sample_param(ndraws, m, s)
    y = bet @ x + np.random.multivariate_normal(np.zeros(nobs), sig, (ndraws, nres))

    return y


def eval_logmargin(y: np.ndarray, x: np.ndarray, sig: np.ndarray, m: np.ndarray, s: np.ndarray, w: np.ndarray=None):
    """Evaluate the log marginal likelihood or evidence given data and hyperparameters. you can evaluate the predictive density by passing posterior instead of prior hyperparameters.

    :param y:
    :param x:
    :param sig:
    :param m:
    :param s:
    :param w:
    :returns: log marginal likelihood

    >>> data, _, param = _generate_fixture(3, 2, 1, seed=666)
    >>> y, x, sig = data
    >>> m, s = param
    >>> eval_logmargin(y, x, sig, m, s)
    -8.509231870747989
    """

    if w is not None:
        x = x @ np.diag(w)
        y = y @ np.diag(w)
    mx, sx = marginalize(x, sig, m, s)

    return np.nansum(eval_mvnorm(y - mx, np.zeros(y.shape[1]), sx))


def get_ev(m: np.ndarray, s: np.ndarray):
    """Evaluate the expectation of parameters given hyperparameters.

    :param m:
    :param s:
    :returns: parameter expectations

    >>> _, _, param = _generate_fixture(3, 2, 1, seed=666)
    >>> m, s = param
    >>> get_ev(m, s)
    (array([[0.],
           [0.],
           [0.]]),)
    """

    return m,


def get_mode(m: np.ndarray, s: np.ndarray):
    """Evaluate the mode of parameters given hyperparameters.

    :param m:
    :param s:
    :returns: parameter modes

    >>> _, _, param = _generate_fixture(3, 2, 1, seed=666)
    >>> m, s = param
    >>> get_mode(m, s)
    (array([[0.],
           [0.],
           [0.]]),)
    """

    return m,


def _generate_fixture(nres: int = 3, nobs: int = 2, nvar: int = 1, seed: int = 666) -> (3 * (np.ndarray,), (np.ndarray,), 2 * (np.ndarray,)):
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

    # set input
    x = np.random.standard_normal((nvar, nobs))
    y = bet @ x + np.random.standard_normal((nres, nobs))
    sig = np.identity(nobs)

    # set hyperparameters
    m = np.zeros((nres, nvar))
    s = np.identity(nvar)

    return (y, x, sig), (bet,), (m, s)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
