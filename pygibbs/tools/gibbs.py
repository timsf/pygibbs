from typing import TypeVar, Tuple, Callable

import numpy as np


Data = TypeVar('Data')
Priors = TypeVar('Priors')
Block1 = TypeVar('Block1')
Block2 = TypeVar('Block2')


def em(niter: int,
       data: Data,
       priors: Priors,
       init_block1: Block1,
       eval_obj: Callable[[Data, Block2], float],
       update_block1: Callable[[Data, Block2], Block1],
       update_block2: Callable[[Data, Block1, Priors], Block2],
       tol: float = 1e-4) -> Tuple[Block1, Block2]:
    """Find the MAP estimator of the model with the EM algorithm.

    :param niter: maximum number of iterations
    :param data: observed quantities
    :param priors: prior hyperparameters for block2
    :param init_block1: initial block1 variable values
    :param eval_obj: objective function
    :param update_block1: e-step function
    :param update_block2: m-step function
    :param tol: minimum objective increment
    :returns: block1 estimates, block 2 estimates
    """

    # initialize
    block1, block2 = init_block1, None
    obj = [-np.inf]

    for t in range(niter):

        # M-step
        block2 = update_block2(data, block1, priors)

        # E-step
        block1 = update_block1(data, block2)

        # stopping criterion
        obj.append(eval_obj(data, block2))
        if obj[t + 1] - obj[t] < tol:
            break

    return block1, block2


def gibbs(ndraws: int,
          data: Data,
          priors: Priors,
          init_block1: Block1,
          init_block2: Block2,
          sample_block1: Callable[[Data, Block2], Block1],
          sample_block2: Callable[[Data, Block1, Priors], Block2]) -> Tuple[Block1, Block2]:
    """Sample from the posterior distribution given data and hyperparameters.

    :param ndraws: number of draws to be sampled
    :param data: observed quantities
    :param priors: prior hyperparameters for block2
    :param init_block1: initial block1 values
    :param init_block2: initial block2 values
    :param sample_block1: block1 given block2 sampling function
    :param sample_block2: block2 given block1 sampling function
    :returns: block1 draws, block2 draws
    """

    # initialize
    block1 = [init_block1]
    block2 = [init_block2]

    for t in range(ndraws - 1):

        # sample block2 given block1
        block2.append(sample_block2(data, block1[-1], priors))

        # sample block1 given block2
        block1.append(sample_block1(data, block2[-1]))

    return tuple([np.array(x) for x in zip(*block1)]), tuple([np.array(x) for x in zip(*block2)])

