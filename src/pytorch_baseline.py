import json
import math
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.modules.loss as loss
from datetime import datetime
from torch import tensor
from torch import nn
from torch import Tensor
from tqdm import tqdm

from m1_utilities import read_dataset_CUP


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


class L1Loss(loss.MSELoss):
    """
    Custom PyTorch loss corresponding to a MSE with L1 regularization
    """
    def __init__(self, lambd: float, size_average=None, reduce=None) -> None:
        super(L1Loss, self).__init__(size_average, reduce)
        self.lambd = lambd

    def forward(self, predicted: Tensor, target: Tensor, model: nn.Module) -> Tensor:
        return l1_loss(predicted, target, self.lambd, model)


def l1_loss(predicted, target, lambd, model):
    """
    Function executing the computation of MSE + L1 reg
    :param predicted: predicted outputs of the network
    :param target: actual targets
    :param lambd: regularization coefficient
    :param model: PyTorch neural network
    :return: MSE + L1 regularization
    """
    mse = 0.5 * torch.mean((target - predicted) ** 2)
    weights_and_biases = np.concatenate((
        torch.detach(model[0].weight).cpu().numpy().flatten(),
        torch.detach(model[0].bias).cpu().numpy().flatten()))
    for i in range(1, len(model)):
        if isinstance(model[i], nn.Linear):
            weights_and_biases = np.concatenate((weights_and_biases,
                                                 torch.detach(model[i].weight).cpu().numpy().flatten(),
                                                 torch.detach(model[i].bias).cpu().numpy().flatten()))
    l1_reg = lambd * np.linalg.norm(weights_and_biases, ord=1)
    return mse + l1_reg


def train_model(model, x, y, epochs, lr, momentum, nesterov, l1_reg_coeff, bs=None):
    """
    Performs training of a PyTorch model
    :param model: PyTorch neural network
    :param x: input data
    :param y: data targets
    :param epochs: number of epochs of training
    :param lr: learning rate
    :param momentum: momentum coefficient
    :param nesterov: if momentum > 0 -> if True, perform NAG, otherwise HB
    :param l1_reg_coeff: L1 regularization coefficient
    :param bs: batch size
    :return: training loss history
    """
    bs = len(x) if bs is None else bs
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov)
    criterion = L1Loss(lambd=l1_reg_coeff)

    # progress bar
    progbar = tqdm(range(epochs), total=epochs)

    # training cycle
    loss = None  # just to avoid reference before assigment
    tr_loss = []
    for ep in range(epochs):
        model.train()

        # shuffle
        indexes = torch.randperm(x.shape[0])
        x = x[indexes]
        y = y[indexes]

        n_batches = math.ceil(len(x) / bs)
        ep_loss = 0

        optimizer.zero_grad()

        try:
            x = torch.from_numpy(x).float().to(device)
            y = torch.from_numpy(y).float().to(device)
        except TypeError:
            pass

        outputs = model(x)
        loss = criterion(model=model, predicted=outputs, target=y)
        ep_loss += loss.item()

        # backprop
        loss.backward()
        optimizer.step()

        # update progress bar
        progbar.update()
        progbar.set_postfix(train_loss=f"{loss.item():.8f}")

        # save history of the epoch
        ep_loss /= n_batches
        tr_loss.append(ep_loss)

    progbar.close()
    return tr_loss


def init_weights(layer):
    if isinstance(layer, nn.Linear):
        torch.nn.init.uniform_(tensor=layer.weight, a=-0.05, b=0.05)
        torch.nn.init.uniform_(tensor=layer.bias, a=-0.05, b=0.05)


if __name__ == '__main__':
    # define model
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.Sigmoid(),
        nn.Linear(5, 2),
    )
    model = model.float()

    # weights initialization
    model.apply(init_weights)

    # read data
    x, y = read_dataset_CUP()

    # training
    tr_loss = train_model(model, x, y, epochs=17500, lr=0.01, momentum=0.7, nesterov=False, l1_reg_coeff=0.00001)
    dir_path = ""
    file_name = ""
    os.makedirs(dir_path, exist_ok=True)
    data = {}
    if os.path.exists(dir_path + file_name):
        with open(dir_path + file_name, 'r') as f:
            data = json.load(f)
    date_time_of_run = datetime.now().strftime("%d/%m/%Y %H:%M")
    data[date_time_of_run] = {'tr_loss': tr_loss}
    with open(dir_path + file_name, 'w') as f:
        json.dump(data, f)
