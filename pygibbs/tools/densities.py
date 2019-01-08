import numpy as np
from scipy.special import gammaln, multigammaln

from tools.linalg import eval_detquad, eval_double_detquad


def eval_norm(x: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """Evaluate the log density given parameters

    :param x: data vector
    :param mu: mean vector
    :param sig: variance vector
    :returns: log density for each independent element of x
    """

    d = (x - mu) ** 2 / sig
    kern = -d / 2
    cons = -(np.log(sig) + np.log(2 * np.pi)) / 2

    return cons + kern


def eval_mvnorm(x: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """Evaluate the log density given parameters

    :param x: data matrix, rows corresponding to observations and columns to variables
    :param mu: mean vector
    :param sig: covariance matrix. if 1-dimensional, assume diagonal matrix
    :returns: log density for each row of x
    """

    nvar = x.shape[1]
    d, logdet_sig = eval_detquad(x - mu, sig)
    kern = -np.sum(d ** 2, 1) / 2
    cons = -(logdet_sig + nvar * np.log(2 * np.pi)) / 2

    return cons + kern


def eval_matnorm(x: np.ndarray, mu: np.ndarray, sig: np.ndarray, tau: np.ndarray) -> float:
    """Evaluate the log density given parameters

    :param x: data matrix
    :param mu: mean matrix
    :param sig: row covariance matrix. if 1-dimensional, assume diagonal matrix
    :param tau: column covariance matrix. if 1-dimensional, assume diagonal matrix
    :returns: log density at x
    """

    rows, cols = x.shape
    d, logdet_sig, logdet_tau = eval_double_detquad(x - mu, sig, tau)
    kern = -float(np.sum(d ** 2) / 2)
    cons = -(rows * cols * np.log(2 * np.pi) + cols * logdet_sig + rows * logdet_tau) / 2

    return cons + kern


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
    :param mu: location vector
    :param sig: scale vector
    :param nu: degrees of freedom vector
    :returns: log density for each independent element of x
    """

    d = (x - mu) ** 2 / sig
    kern = -(nu + 1) / 2 * np.log(d / nu + 1)
    cons = gammaln((nu + 1) / 2) - (gammaln(nu / 2) + (np.log(nu) + np.log(np.pi) + np.log(sig)) / 2)

    return kern + cons


def eval_mvt(x: np.ndarray, mu: np.ndarray, sig: np.ndarray, nu: float) -> np.ndarray:
    """Evaluate the log density given parameters

    :param x: data matrix, rows corresponding to observations and columns to variables
    :param mu: mean vector
    :param sig: covariance matrix. if 1-dimensional, assume diagonal matrix
    :param nu: degrees of freedom
    :returns: log density for each row of x
    """

    nvar = x.shape[1]
    d, logdet_sig = eval_detquad(x - mu, sig)
    kern = -(nu + nvar) / 2 * np.log(np.sum(d ** 2, 1) / nu + 1)
    cons = gammaln((nu + nvar) / 2) - gammaln(nu / 2) - (nvar * (np.log(nu) + np.log(np.pi)) + logdet_sig) / 2

    return cons + kern


def eval_matt(x: np.ndarray, mu: np.ndarray, sig: np.ndarray, tau: np.ndarray, nu: float) -> float:
    """Evaluate the log density given parameters

    :param x: data matrix
    :param mu: mean matrix
    :param sig: row covariance matrix. if 1-dimensional, assume diagonal matrix
    :param tau: column covariance matrix. if 1-dimensional, assume diagonal matrix
    :param nu: degrees of freedom
    :returns: log density at x
    """

    rows, cols = x.shape
    d, logdet_sig, logdet_tau = eval_double_detquad(x - mu, sig, tau)
    if d.shape[0] > d.shape[1]:
        d = d.T @ d
    else:
        d = d @ d.T
    kern = -(nu + rows + cols - 1) / 2 * np.prod(np.linalg.slogdet(np.identity(d.shape[0]) + d / nu))
    cons = multigammaln((nu + rows + cols - 1) / 2, rows + cols) \
        - multigammaln((nu + cols - 1) / 2, cols) \
        - multigammaln((nu + rows - 1) / 2, rows) \
        - (rows * cols * (np.log(nu) + 2 * np.log(np.pi)) + cols * logdet_sig + rows * logdet_tau) / 2

    return cons + kern
