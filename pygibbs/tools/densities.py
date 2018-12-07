import numpy as np
from scipy.special import gammaln, multigammaln
from scipy.linalg import solve, solve_triangular


def eval_norm(x: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """Evaluate the log density given parameters

    :param x: data vector
    :param mu: mean vector
    :param sig: variance vector
    :returns: log density for each independent element of x
    """

    d = (x - mu) ** 2 / sig
    k = -d / 2
    nc = -(np.log(sig) + np.log(2 * np.pi) / 2)

    return nc + k


def eval_mvnorm(x: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """Evaluate the log density given parameters

    :param x: data matrix, rows corresponding to observations and columns to variables
    :param mu: mean vector
    :param sig: covariance matrix. if 1-dimensional, assume diagonal matrix
    :returns: log density for each row of x
    """

    nvar = x.shape[1]

    if len(sig.shape) == 2:
        det_sig = logdet(sig)
    else:
        det_sig = np.sum(np.log(sig))

    d = eval_quad(x - mu, sig)
    k = -d / 2
    nc = -(det_sig + np.log(2 * np.pi)) / 2

    return nc + k


def eval_lm(y: np.ndarray, x: np.ndarray, bet: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """Evaluate the log density given parameters

    :param y: response matrix
    :param x: covariate matrix
    :param bet: coefficient matrix
    :param sig: covariance matrix. if 1-dimensional, assume diagonal matrix
    :returns: log density
    """

    return np.nansum(eval_norm(y.T, (bet @ x).T, np.sqrt(sig)), 1)


def eval_mvlm(y: np.ndarray, x: np.ndarray, bet: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """Evaluate the log density given parameters

    :param y: response matrix
    :param x: covariate matrix
    :param bet: coefficient matrix
    :param sig: covariance matrix. if 1-dimensional, assume diagonal matrix
    :returns: log density
    """

    return eval_mvnorm(y.T, (bet @ x).T, sig)


def eval_t(x: np.ndarray, mu: np.ndarray, sig: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """Evaluate the log density given parameters

    :param x: data vector
    :param mu: mean vector
    :param sig: variance vector
    :param nu: degrees of freedom vector
    :returns: log density for each independent element of x
    """

    d = (x - mu) ** 2 / sig
    k = -(nu + 1) / 2 * np.log(d / nu + 1)
    nc = gammaln((nu + 1) / 2) - (gammaln(nu / 2) + (np.log(nu) + np.log(np.pi) + np.sqrt(sig)) / 2)

    return nc + k


def eval_mvt(x: np.ndarray, mu: np.ndarray, sig: np.ndarray, nu: float) -> np.ndarray:
    """Evaluate the log density given parameters

    :param x: data matrix, rows corresponding to observations and columns to variables
    :param mu: mean vector
    :param sig: covariance matrix. if 1-dimensional, assume diagonal matrix
    :param nu: degrees of freedom
    :returns: log density for each row of x
    """

    nvar = x.shape[1]

    if len(sig.shape) == 2:
        det_sig = logdet(sig)
    else:
        det_sig = np.sum(np.log(sig))

    d = eval_quad(x - mu, sig)
    k = -(nu + nvar) / 2 * np.log(d / nu + 1)
    nc = gammaln((nu + nvar) / 2) - (gammaln(nu / 2) + nvar * (np.log(nu) + np.log(np.pi)) / 2 + det_sig / 2)

    return nc + k


def eval_matt(x: np.ndarray, mu: np.ndarray, sig: np.ndarray, tau: np.ndarray, nu: float) -> float:
    """Evaluate the log density given parameters

    :param x: data matrix, rows corresponding to observations and columns to variables
    :param mu: mean matrix
    :param sig: row covariance matrix. if 1-dimensional, assume diagonal matrix
    :param tau: column covariance matrix. if 1-dimensional, assume diagonal matrix
    :param nu: degrees of freedom
    :returns: log density at x
    """

    nobs, nvar = x.shape

    if len(tau.shape) == 2:
        det_tau = logdet(tau)
    else:
        det_tau = np.sum(np.log(tau))

    if len(sig.shape) == 2:
        d = logdet(np.identity(nobs) + solve(sig, np.identity(nobs), sym_pos=True) @ eval_matquad(x - mu, tau))
        det_sig = logdet(sig)
    else:
        d = logdet(np.identity(nobs) + eval_matquad(x - mu, tau) / sig)
        det_sig = np.sum(np.log(sig))

    nc = multigammaln((nu + nobs + nvar - 1) / 2, nvar) \
         - multigammaln((nu + nvar - 1) / 2, nvar) \
         - (nobs * nvar * np.log(np.pi) + nvar * det_sig + nobs * det_tau) / 2
    k = -(nu + nobs + nvar - 1) / 2 * d

    return nc + k


def eval_quad(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Evaluate the quadratic form x[i].T @ inv(s) @ x[i] for each row i in x.

    :param x:
    :param s: inverse scaling matrix. if 1-dimensional, assume diagonal matrix
    :returns: evaluated quadratic forms

    >>> from scipy.stats import wishart
    >>> s = wishart(2, np.diag(np.ones(2))).rvs(1)
    >>> x = np.random.standard_normal((3, 2))
    >>> np.all(np.isclose(eval_quad(x, s), [x_i @ np.linalg.inv(s) @ x_i for x_i in x]))
    True
    """

    if len(s.shape) == 2:
        cf = x @ solve_triangular(np.linalg.cholesky(s).T, np.identity(x.shape[1]))
    else:
        cf = x / s

    return np.sum(cf ** 2, 1)


def eval_matquad(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Evaluate the quadratic form x @ inv(s) @ x.T.

    :param x:
    :param s: inverse scaling matrix. if 1-dimensional, assume diagonal matrix
    :returns: evaluated quadratic form

    >>> from scipy.stats import wishart
    >>> s = wishart(2, np.diag(np.ones(2))).rvs(1)
    >>> x = np.random.standard_normal((3, 2))
    >>> np.all(np.isclose(eval_matquad(x, s), x @ np.linalg.inv(s) @ x.T))
    True
    """

    if len(s.shape) == 2:
        cf = x @ solve_triangular(np.linalg.cholesky(s).T, np.identity(x.shape[1]))
    else:
        cf = x / s

    return cf @ cf.T


def logdet(a: np.ndarray) -> float:
    """Evaluate log determinant of the matrix a

    :param a: square matrix
    :returns: log determinant of a
    """

    return float(np.prod(np.linalg.slogdet(a)))