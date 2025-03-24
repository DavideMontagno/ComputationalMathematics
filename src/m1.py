"""
This code implements the Unified Momentum approach presented in the paper "Unified convergence analysis of stochastic
momentum methods for convex and non-convex optimization", Tianbao Yang, Qihang Lin, Zhe Li, 2016
"""

import time
import pickle
import math
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from datetime import datetime

from functions import Function, act_funcs, loss_funcs, l1reg, l1reg_deriv
from weight_initializations import weights_inits
from m1_utilities import *


class Layer:
    """
    Implements a layer of a neural network.

    Attributes:
        weight: matrix of the layer's weights
        bias: vector of the layer's biases
        activation: activation function of the layer
        inputs: the layers' input
        weighted_sum: contains the multiplication of the layer's input by the layer's weights
        grad_weights: gradient of the cost function w.r.t. the layer's weights
        grad_bias: gradient of the cost function w.r.t. the layer's bias
    """

    def __init__(self, weight: np.ndarray, bias: np.ndarray, activation: str = 'linear'):
        """
        :param weight: matrix of the layer's weights
        :param bias: vector of the layer's biases
        :param activation: string indicating the activation function
        """
        self.weight = weight
        self.bias = bias
        self.activation = act_funcs[activation]
        self.inputs = None
        self.weighted_sum = None
        self.grad_weights = None
        self.grad_bias = None

    def forward_pass(self, x: np.ndarray):
        """
        Perform a forward pass through the layer
        :param x: input to the layer
        :return: output of the layer
        """
        self.inputs = x
        self.weighted_sum = np.matmul(x, self.weight)
        sum_plus_bias = np.add(self.weighted_sum, self.bias)
        out = self.activation.func(sum_plus_bias)
        return out

    def backward_pass(self, upstream_err, verbose=False):
        """
        Computes a backward pass through the layer to compute its weights' gradient
        :param upstream_err: derivative of the error w.r.t. the current layer's output
        :param verbose: if True, prints some debugging statements
        :return: upstream gradient for the previous layer, gradient of weights, gradient of biases
        """
        # derivative of the activation function w.r.t. the weighted sum
        dOut_dNet = self.activation.deriv(self.weighted_sum)
        # compute deltas
        delta = np.multiply(upstream_err, dOut_dNet)
        # compute the upstream error for the next layer in the backprop (i.e. the previous one)
        new_upstream_err = delta @ self.weight.T
        # compute the gradients for the current layer's weights and biases
        self.grad_bias = np.mean(delta, axis=0)
        self.grad_weights = np.mean(np.matmul(np.transpose(self.inputs, axes=(0, 2, 1)), delta), axis=0)
        # return flat vector of weights and biases to later compute gradient's norm
        flat_weights_biases = np.concatenate((self.grad_weights.flatten(), self.grad_bias.flatten()))
        # prints of the various dimensions of the variables involved, for debugging purposes
        if verbose:
            print("dOut_dNet: ", dOut_dNet.shape)
            print("delta: ", delta.shape)
            print("weight: ", self.weight.shape)
            print("weight transpose: ", self.weight.T.shape)
            print("new_upstream_err: ", new_upstream_err.shape)
            print("grad_bias: ", self.grad_bias.shape)
            print("bias: ", self.bias.shape)
            print("inputs: ", self.inputs.shape)
            print("inputs transposed: ", np.transpose(self.inputs, axes=(0, 2, 1)).shape)
            print("grad_weights: ", self.grad_weights.shape, '\n\n')
        return new_upstream_err, flat_weights_biases


class NeuralNetwork:
    """
    Implements a fully-connected feed-forward neural network.
    It corresponds to the model 'M1' in our report (report/report.pdf)

    Attributes:
        layers: list of Layer objects, i.e. the layers of te nn
    """

    def __init__(self,
                 dims: Tuple[int, ...],
                 activations: Union[Tuple[str, ...], str] = 'linear',
                 init_type: str = 'uniform',
                 **kwargs):
        """
        :param dims: tuple of the dimensions of the net's layers, the 1st is the dim of the input vector
        :param activations: activation functions for the layers:
            if string: same activation for every layer
            if tuple of 2 elements: 1st element as activation for all the layers except the last one, 2nd element as last activation
            if longer tuple: must have the same length as the number of layers, specify specific activation for each layer
        :param init_type: type of weight initialization
        """

        # checks on arguments
        if isinstance(activations, str):
            activations = (activations,)
        assert len(activations) <= len(dims)
        if len(activations) < len(dims):
            assert len(activations) <= 2
            if len(activations) == 1:
                activations = [activations[0]] * (len(dims) - 1)
            else:   # len(activations) == 2
                activations = [activations[0]] * (len(dims) - 2) + [activations[1]]

        # create and initialize the layers (i.e. the matrices of the net's weights)
        self.layers = []
        for i in range(len(dims) - 1):
            weight = weights_inits(init_type, shape=(dims[i], dims[i + 1]), **kwargs)
            bias = weights_inits(init_type, shape=(1, dims[i + 1]), **kwargs)
            self.layers.append(Layer(weight, bias, activations[i]))

    def forward(self, x: np.ndarray):
        """
        Performs a forward pass on the whole network
        :param x: input of the network
        :return: output of the network
        """
        for layer in self.layers:
            x = layer.forward_pass(x)
        return x

    def backward(self, dErr_dOut):
        """
        Performs a backward pass through the network to propagate the error and set the gradients
        :param dErr_dOut: derivative of the error w.r.t. the output of the network
        """
        net_flat_weights_biases = []
        upstream_err = dErr_dOut
        # backpropagation
        for layer in reversed(self.layers):
            upstream_err, layer_flat_weights_biases = layer.backward_pass(upstream_err=upstream_err)
            net_flat_weights_biases = np.concatenate((net_flat_weights_biases, layer_flat_weights_biases))
        # compute gradient's norm to check convergence
        grad_norm = np.linalg.norm(net_flat_weights_biases, ord=2)
        return grad_norm

    def _init_y(self):
        """
        Initialize the Y sequences for the first step of weight update in the unified momentum approach (citation on top)
        :return: the aforementioned Y sequences for weights and biases
        """
        y_weight_prec, y_bias_prec = [], []
        for layer in self.layers:
            y_weight_prec.append(layer.weight)
            y_bias_prec.append(layer.bias)
        return y_weight_prec, y_bias_prec

    def get_weights_biases(self):
        """
        :return: 2 lists:
            one containing each layer's weights, flattened and concatenated;
            one containing each layer's biases, flattened and concatenated
        """
        W = self.layers[0].weight.flatten()
        B = self.layers[0].bias.flatten()
        for layer in self.layers[1:]:
            W = np.concatenate((W, layer.weight.flatten()))
            B = np.concatenate((B, layer.bias.flatten()))
        return W, B

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int, batch_size: int, val_split: float = 0.,
            x_test: np.ndarray = None, y_test: np.ndarray = None, lr: float = 0.1, alpha: float = 0., s: float = 0,
            loss: str = 'mse', l1_coeff: float = 0., norm_threshold=0., save_model=False, model_path=None,
            save_plot=False, plot_path=None):
        """
        Performs the training of the network
        :param x: input data
        :param y: targets / labels
        :param epochs: number of epochs
        :param batch_size: batch size
        :param val_split: (float) percentage of the dataset to be used for testing
        :param x_test: data to be used for validation (passed explicitly)
        :param y_test: targets for the validation data (passed explicitly)
        :param lr: learning rate
        :param alpha: momentum coefficient
        :param s: term to choose the type of momentum:
            s = 0               --> classical momentum (heavy ball)
            s = 1               --> Nesterov's accelerated gradient
            s = 1 / (1 - alpha) --> GD/SGD with learning rate equal to lr / (1 - alpha)
                NOTE: it's also possible to set alpha=0 and have a GD/SGS with learning rate directly equal to lr
        :param loss: loss function
        :param save_model: if True, saves the model
        :param model_path: dirs where to save the model if save_model=True
        :param save_plot: if True, saves the plot of the training
        :param plot_path: dirs where to save the plots if save_plot=True
        :param l1_coeff: (L1) regularization coefficient
        :param norm_threshold: bound norm gradient for early stopping
        :return: history of training
        """

        # add "1" in dimensions in case of vectors -> e.g. (3,)->(1, 3)
        x, y = np.array(x), np.array(y)
        if len(x.shape) < 3:
            x = x[:, np.newaxis, :]
        if len(y.shape) < 2:
            y = y[:, np.newaxis]
        if len(y.shape) < 3:
            y = y[:, np.newaxis, :]

        # create validation set
        if x_test is None and val_split > 0:
            x_tr, x_test, y_tr, y_test = train_test_split(x, y, test_size=val_split, shuffle=True)
        else:
            x_tr, y_tr = x, y

        loss = loss_funcs[loss]
        # added norm history, time elapsed per epoch
        loss_hist = {'tr': [], 'val': [], 'norm': [], 'elapsed': []}

        first_update = True  # used for the weight update

        # cycle through epochs
        for ep in range(epochs):
            start_time = time.time()
            epoch_tr_loss = []
            grad_norms = []

            # shuffle
            indexes = list(range(len(x_tr)))
            np.random.shuffle(indexes)
            x_tr = x_tr[indexes]
            y_tr = y_tr[indexes]

            # create progress bar
            limit = math.ceil(len(x_tr) / batch_size)
            iterat = range(limit)
            progbar = tqdm(iterable=iterat, total=limit)
            progbar.set_description(f"Epoch [{ep + 1}/{epochs}]")

            # cycle through batches
            for batch_index in iterat:
                start = batch_index * batch_size
                end = start + batch_size
                x_tr_batch = x_tr[start: end]
                y_tr_batch = y_tr[start: end]

                # forward pass and compute error
                out = self.forward(x_tr_batch)
                curr_err = loss.func(predicted=out, target=y_tr_batch)
                W_vec, B_vec = self.get_weights_biases()
                curr_err += l1reg(l1_coeff, W_vec, B_vec)
                epoch_tr_loss.append(curr_err)

                # compute error derivative
                dErr_dOut = loss.deriv(predicted=out, target=y_tr_batch)
                # backward pass -> sets each layer's gradients
                curr_grad_norm = self.backward(dErr_dOut)
                grad_norms.append(curr_grad_norm)

                # # weight update SGD (with NO regularization) (for debug purposes)
                # for layer in self.layers:
                #     layer.weight += -lr * layer.grad_weights
                #     layer.bias += -lr * layer.grad_bias

                # weight update through unified momentum (citation on top of the script)
                for idx, layer in enumerate(self.layers):
                    if first_update:
                        first_update = False
                        ys_weight_prec, ys_bias_prec = self._init_y()

                    # add the regularization term to the weights/biases' gradients
                    layer.grad_weights += l1reg_deriv(l1_coeff, layer.weight)
                    layer.grad_bias += l1reg_deriv(l1_coeff, layer.bias)

                    y_weight = layer.weight - lr * layer.grad_weights
                    y_bias = layer.bias - lr * layer.grad_bias

                    ys_weight = layer.weight - s * lr * layer.grad_weights
                    ys_bias = layer.bias - s * lr * layer.grad_bias

                    layer.weight = y_weight + alpha * (ys_weight - ys_weight_prec[idx])
                    layer.bias = y_bias + alpha * (ys_bias - ys_bias_prec[idx])

                    ys_weight_prec[idx] = ys_weight
                    ys_bias_prec[idx] = ys_bias

                # update progress bar after each minibatch
                progbar.update()
                progbar.set_postfix(tr_loss=f"{curr_err:.8f}")

            # compute and save some data at the end of the epoch
            mean_tr_loss = np.mean(epoch_tr_loss)
            stop_time = (time.time() - start_time)
            avg_grad_norm = np.mean(grad_norms)
            loss_hist['tr'].append(mean_tr_loss)
            loss_hist['norm'].append(avg_grad_norm)
            loss_hist['elapsed'].append(stop_time)

            # validation
            if val_split > 0 or x_test is not None:
                val_loss = self.evaluate(x_test, y_test, loss)
                loss_hist['val'].append(val_loss)
                progbar.set_postfix(epoch_mean_tr_loss=f"{mean_tr_loss:.4f}",
                                    val_loss=f"{val_loss:.4f}",
                                    grad_norm=f"{avg_grad_norm:.4f}")
            else:
                progbar.set_postfix(epoch_mean_tr_loss=f"{0.5 * mean_tr_loss:.8f}",
                                    grad_norm=f"{avg_grad_norm:.4f}")

            # close progress bar
            progbar.close()

            # check for convergence, i.e. gradient norm <= threshold. In case exit the training
            if avg_grad_norm < norm_threshold:
                print(f"\nStopped before:\nEpoch: {ep}\nGradient norm: {avg_grad_norm} (threshold: {norm_threshold})")
                break  # exit training

        # save model if required
        if save_model:
            self.save(model_path)

        # save plot of learning curve
        if save_plot:
            now = datetime.now()
            filename = str(now.date()) + "_" + now.strftime("%H%M") + ".png"
            if plot_path is None:
                plot_path = filename
            elif os.path.isdir(plot_path):
                plot_path += '/' if plot_path[-1] != '/' else ''
                plot_path += filename
            plt.plot(loss_hist['tr'])
            if val_split > 0 or x_test is not None:
                plt.plot(loss_hist['val'])
                plt.legend(["Training loss", "Validation loss"])
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.ylim((-0.1, 2))
            plt.grid()
            plt.savefig(plot_path)
        return loss_hist

    def evaluate(self, x: np.ndarray, y: np.ndarray, loss: Union[str, Function]):
        """
        Perform validation/testing of the model
        :param x: input data
        :param y: target data
        :param loss: the loss object or the name of the loss as string
        :return: the error of the model on the data x
        """
        out = self.forward(x)
        loss = loss if isinstance(loss, Function) else loss_funcs[loss]
        return loss.func(predicted=out, target=y)

    def save(self, path=None):
        """
        Saves the model in json format
        :param path: dirs where to save the model
        """
        now = datetime.now()
        filename = str(now.date()) + "_" + now.strftime("%H%M") + ".json"
        if path is None:
            path = filename
        elif os.path.isdir(path):
            path += '/' if path[-1] != '/' else ''
            path += filename
        with open(path, 'wb') as f:
            pickle.dump(self, f)
