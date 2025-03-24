import numpy as np
from typing import Tuple


def weights_inits(init_type: str, **kwargs):
    """
    Calls the correct weight init function depending on the name of the initialization

    :param init_type: name of the desired initialization (key of a dict{name: func_pointer})
    :param kwargs: contains other arguments for the different init functions
    :return: calls the correct init function and returns its return value
    """
    inits = {'fixed': fixed_init, 'uniform': rand_init}
    return inits[init_type](**kwargs)


def fixed_init(shape: Tuple[int, ...], weight_value: float, **kwargs):
    """
    Initialize all weights and biases to the same value (i.e. 'weight_value')

    :param shape: shape of the weight matrix
    :param weight_value: value of all the weights in the matrix
    :param kwargs: put to let the user pass other arguments for potentially other initializations without causing errors
    :return: a weight matrix of the specified shape where all the weights have the same value
    """
    return np.full(shape=shape, fill_value=weight_value)


def rand_init(shape: Tuple[int, ...], lower_lim: float = -0.05, upper_lim: float = 0.05, **kwargs):
    """
    Random uniform initialization of the weights.
    Note: if this function is used to initialize the bias, it will set it to 0.0

    :param shape: shape of the weight matrix
    :param lower_lim: smallest weight_value that the weights can assume
    :param upper_lim: largest weight_value that the weights can assume
    :param kwargs: put to let the user pass other arguments for potentially other initializations without causing errors
    :return: weight matrix of the specified shape with random uniform weights in the interval [lower_lim, upper_lim]
    """
    if shape[0] == 1:  # it means we're initializing the bias
        return fixed_init(shape=shape, weight_value=0.)  # bias initialized to 0
    return np.random.uniform(low=lower_lim, high=upper_lim, size=shape)  # else return a random uniform weight matrix
