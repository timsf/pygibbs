import numpy as np


def softplus(x: np.ndarray) -> np.ndarray:
    """Compute the softplus function, i.e. the anti-derivative of the expit function.

    :param x:
    :returns: softplus at x
    """

    return np.array([np.logaddexp(0, xi) for xi in x])