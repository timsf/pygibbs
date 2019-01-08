from typing import Callable

import numpy as np


def em(niter: int,
       data: tuple,
       priors: tuple,
       init: tuple,
       eval_obj: Callable,
       update_block1: Callable,
       update_block2: Callable,
       tol: float = 1e-4) -> (tuple, tuple):
    """Find the MAP estimator of the model with the EM algorithm.

    :param niter: maximum number of iterations
    :param data: observed quantities
    :param priors: prior hyperparameters for block2
    :param init: initial block1 variable values
    :param eval_obj: objective function
    :param update_block1: e-step function
    :param update_block2: m-step function
    :param tol: minimum objective increment
    :returns: block1 estimates, block 2 estimates
    """

    # initialize
    block1, block2 = init, None
    obj = [-np.inf]
    
    for t in range(niter):

        # M-step
        block2 = update_block2(*data, *block1, *priors)

        # E-step
        block1 = update_block1(*data, *block2)

        # stopping criterion
        obj.append(eval_obj(*data, *block2))
        if obj[t + 1] - obj[t] < tol:
            break

    return block1, block2


def gibbs(ndraws: int,
          data: tuple,
          priors: tuple,
          init: tuple,
          sample_block1: Callable,
          sample_block2: Callable) -> (tuple, tuple):
    """Sample from the posterior distribution given data and hyperparameters.

    :param ndraws: number of draws to be sampled
    :param data: observed quantities
    :param priors: prior hyperparameters for block2
    :param init: initial block1 and block2 values
    :param sample_block1: block1 given block2 sampling function
    :param sample_block2: block2 given block1 sampling function
    :returns: block1 draws, block2 draws
    """

    # initialize
    block1_init, block2_init = init
    block1 = [[x] for x in block1_init]
    block2 = [[x] for x in block2_init]

    for t in range(ndraws - 1):

        # sample block2 given glock1
        for x, y in zip(block2, sample_block2(*data, *[z[-1] for z in block1], *priors)):
            x.append(y)
        
        # sample block1 given block2
        for x, y in zip(block1, sample_block1(*data, *[z[-1] for z in block2])):
            x.append(y)
        
    return tuple([np.array(x) for x in block1]), tuple([np.array(x) for x in block2])
