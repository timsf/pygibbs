import numpy as np
from scipy.linalg import cho_factor, solve_triangular


def eval_quad(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Evaluate the quadratic form x[i].T @ inv(s) @ x[i] for each row i in x.

    :param x:
    :param s: inverse scaling matrix. if 1-dimensional, assume diagonal matrix
    :return: evaluated quadratic forms
    """

    l, _ = cho_factor(s, lower=True)
    root = solve_triangular(l, x.T, lower=True).T

    return np.sum(root ** 2, 1)


def eval_matquad(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Evaluate the quadratic form x @ inv(s) @ x.T.

    :param x:
    :param s: inverse scaling matrix. if 1-dimensional, assume diagonal matrix
    :return: evaluated quadratic form
    """

    l, _ = cho_factor(s, lower=True)
    root = solve_triangular(l, x.T, lower=True).T

    return root @ root.T


def logdet_pd(s: np.ndarray) -> float:
    """Evaluate log determinant of the PD matrix s

    :param s: PD matrix
    :return: log determinant of s
    """

    r, _ = cho_factor(s)

    return float(np.sum(np.log(np.diag(r))) * 2)


def eval_detquad(x, s):
    """

    :param x:
    :param s:
    :return:
    """

    if len(s.shape) == 2:
        l, _ = cho_factor(s, lower=True)
        root = solve_triangular(l, x.T, lower=True).T
        logdet_s = np.sum(np.log(np.diag(l))) * 2
    else:
        root = x / np.sqrt(s)
        logdet_s = np.sum(np.log(s))

    return root, logdet_s


def eval_double_detquad(x, s, t):
    """

    :param x:
    :param s:
    :param t:
    :return:
    """

    root, logdet_s = eval_detquad(x.T, s)
    root, logdet_t = eval_detquad(root.T, t)
        
    return root, logdet_s, logdet_t

