import numpy as np
import warnings
from typing import Union, Tuple, List


class Function:
    """
    Class representing a generic function
    Attributes:
        func (function "pointer"): "points" to the Python function executing the function itself
        name (str): name of the function
    """
    def __init__(self, func, name):
        """
        Constructor
        :param func: (function "pointer") "points" to the Python function executing the function itself
        :param name: (str) name of the function
        """
        self.func = func
        self.name = name


class DerivableFunction(Function):
    """
    Class representing a derivable function which derivative is needed and immediate
    Attributes:
        func (function "pointer"): "points" to the Python function executing the function itself
        deriv (function "pointer"): "points" to the Python function executing the derivative of the function itself
    """
    def __init__(self, func, name, deriv):
        """
        Constructor
        :param func: (function "pointer") "points" to the Python function executing the function itself
        :param name: (str) name of the function
        :param deriv: (function "pointer") "points" to the Python function executing the function itself
        """
        super().__init__(func=func, name=name)
        self.deriv = deriv


# ACTIVATION FUNCTIONS & RESPECTIVE DERIVATIVES
def identity(x: np.ndarray) -> np.ndarray:
    return x


def identity_deriv(x: np.ndarray) -> np.ndarray:
    return np.ones(shape=x.shape)


def sigmoid(x: np.ndarray) -> np.ndarray:
    warnings.filterwarnings('error')    # to enter the "except" in case of RuntimeWarning
    try:
        ones = np.ones(shape=x.shape)
        exp = np.exp(-x)
        return np.divide(ones, np.add(ones, exp))
    except RuntimeWarning:
        warnings.filterwarnings('default')  # restore warnings' settings
        return np.full(shape=x.shape, fill_value=0.0)


def sigmoid_deriv(x: np.ndarray) -> np.ndarray:
    sig = sigmoid(x)
    ones = np.ones(shape=x.shape)
    return np.multiply(sig, np.subtract(ones, sig))    # elementwise multiplication


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_deriv(x: np.ndarray) -> np.ndarray:
    return np.subtract(np.ones(shape=x.shape), np.power(np.tanh(x), 2))


# sort of "static objects" representing the activation functions,
# they are the only instantiation of the respective function
Identity = DerivableFunction(identity, 'identity', identity_deriv)
Sigmoid = DerivableFunction(sigmoid, 'sigmoid', sigmoid_deriv)
Tanh = DerivableFunction(tanh, 'tanh', tanh_deriv)
act_funcs = {
    'identity': Identity,
    'linear': Identity,
    'sigmoid': Sigmoid,
    'tanh': Tanh,
}


# LOSS FUNCTIONS
def mse(predicted: np.ndarray, target: np.ndarray) -> float:
    """
    Compute the Mean Squared Error
    :param predicted: numpy ndarray (patterns, dimensions...) representing the outputs of the net
    :param target: numpy ndarray (patterns, dimensions...) representing the ground-truth outputs
    :return: a float representing the mean squared error for all the predicted patterns
    """
    squared_err = 0.5 * np.square(np.subtract(target, predicted))   # multiply by 0.5 to make derivative simpler
    mean_squared_err = np.mean(squared_err, axis=0)
    try:
        return float(mean_squared_err)  # if the output is a numpy array with 1 value
    except TypeError:
        # if the output is numpy array with more than 1 values
        try:
            return float(np.mean(mean_squared_err, axis=1))
        except np.AxisError:
            return float(np.mean(mean_squared_err))


def mse_deriv(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Computes the derivative of the squared loss for each pattern (w.r.t. the predictions of the net)
    :param predicted: numpy ndarray (patterns, dimensions...) representing the outputs of the net
    :param target: numpy ndarray (patterns, dimensions...) representing the ground-truth outputs
    :return: a numpy ndarray containing the derivative of the squared error for each predicted pattern
    """
    return np.subtract(predicted, target)


MSE = DerivableFunction(mse, 'MSE', mse_deriv)
loss_funcs = {'mse': MSE}


# REGULARIZATION FUNCTIONS
def l1reg(coeff: float, W: np.ndarray, B: np.ndarray):
    """
    Compute L1 (lasso) regularization
    :param coeff: regularization coefficient
    :param W: numpy ndarray containing each layer's weights, flattened and concatenated (single 1-dim vector of all weights)
    :param B: numpy ndarray containing each layer's biases, flattened and concatenated (single 1-dim vector of all biases)
    :return: L1 regularization function value
    """
    # W and B should already be flattened, but we do it here as well for "safety"
    # numpy's norm with ord=1 applied to vectors computes the L1 regularization, for matrices it does a different operation, but here we have a vector
    return coeff * np.linalg.norm(np.concatenate((W.flatten(), B.flatten())), ord=1)


def l1reg_deriv(coeff: float, matrix: np.ndarray):
    """
    Computes the derivative of the L1 (lasso) regularization
    :param coeff: regularization coefficient
    :param matrix: weight/biases matrix/vector
    :return: derivative of L1 reg w.r.t each weight/bias
    """
    return coeff * np.sign(matrix)
