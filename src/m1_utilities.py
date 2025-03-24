import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from m1 import *
import json
from typing import Union, List, Tuple
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler


def load_network(path: str):
    """
    Load a NeuralNetwork object from disk
    :param path: the dirs to the object to load's file
    :return: the loaded object
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def read_dataset_CUP(path="../datasets/ML_CUP/ML-CUP20-TR.csv"):
    """
    Reads the CUP dataset
    :param path: path of the file containing the data
    :return: inputs and targets (normalized)
    """
    col_names = ['id', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'target_x', 'target_y']
    x = pd.read_csv(path, sep=',', names=col_names, skiprows=range(7), usecols=range(1, 11))
    y = pd.read_csv(path, sep=',', names=col_names, skiprows=range(7), usecols=range(11, 13))
    # normalization
    x_normalized = x.copy()
    y_normalized = y.copy()
    for col in x.columns:
        max_val = x[col].max()
        min_val = x[col].min()
        x_normalized[col] = (x[col] - min_val) / (max_val - min_val)
    for col in y.columns:
        max_val = y[col].max()
        min_val = y[col].min()
        y_normalized[col] = (y[col] - min_val) / (max_val - min_val)
    return x_normalized.to_numpy(), y_normalized.to_numpy()


def pick_best_models(alg_type: str, dirs: Union[str, list] = "../training_histories", n_models: int = 1,
                     only_sigmoid: bool = False, select: str = 'last'):
    """
    Reads all the training histories in the directories in "dirs" and picks the best models for each training algorithm (SGD / NAG / CM)
    :param dirs: directory or list of directories containing the training histories
    :param alg_type: name of the algorithm {"sgd", "nesterov_momentum", "heavy_ball"}
    :param n_models: number of best models to consider
    :param only_sigmoid: if true, only consider model that have sigmoid activation function
    :param select: if 'best' select models depending on their best error; if 'last' select models depending on their last error
    :return: list of the n_models best models
    """
    if isinstance(dirs, str):
        dirs = [dirs]

    best_models = []
    for d in dirs:
        # iterate over files
        files = os.listdir(d)  # list of files in the directory
        for file in files:
            # if required, skip the models that don't use sigmoid
            if only_sigmoid and 'sigmoid' not in file:
                continue

            # read the data in the current file
            file_name = os.path.join(d, file)
            with open(file_name) as f:
                data = json.load(f)

            # if we're on the correct algorithm type (heavy ball / nesterov / no momentum)
            if alg_type in data["histories"]:
                # for every run of that algorithm on that model
                for date_time_of_run in data["histories"][alg_type].keys():
                    results = data["histories"][alg_type][date_time_of_run]
                    last_value_tr = results['tr'][-1]
                    best_value_tr, best_value_tr_epoch = np.min(results['tr']), np.argmin(results['tr'])
                    curr_model_entry = {
                        "file_name": file_name,
                        "architecture": data["architecture"],
                        "hyperparams": data["hyperparams"],
                        "alg_type": alg_type,
                        "date_time_of_run": date_time_of_run,
                        "results": results,
                        "last_value_tr": last_value_tr,
                        "best_value_tr": best_value_tr,
                        "best_value_tr_epoch": best_value_tr_epoch
                    }

                    # if we already have 'n_models' saved as current best models
                    if len(best_models) >= n_models:
                        if select == 'last':
                            key = 'last_value_tr'
                            curr_err = last_value_tr
                        else:
                            key = 'best_value_tr'
                            curr_err = best_value_tr
                        errors = [item[key] for item in best_models]
                        # if we find a model that is better than the worst in the top n_models, we switch it
                        if curr_err < max(errors):
                            best_models[np.argmax(errors)] = curr_model_entry
                    # if we haven't encountered n_models yet, save these first models in the list of the best ones
                    else:
                        best_models.append(curr_model_entry)
    return best_models


def grid_search(results_dir: str = "../grid_search_results/"):
    """
    Performs a grid search (hyper-parameters' values specified within this function). The results are saved.
    :param results_dir: path to the directory where to save the results
    """
    x, y = read_dataset_CUP()

    # parameters configurations for the various tests
    dims_list = ((len(x[0]), 5, 2),)
    # acts_list = (('sigmoid', 'linear'),)
    acts_list = ('sigmoid',)
    epochs_list = (17500,)
    batch_sizes = (len(x),)
    lrs = (0.2,)
    alphas = (0.7,)
    l1_coeffs = (0.00001,)
    norm_threshold = 0.

    # grid search cycle
    for dims in dims_list:
        for acts in acts_list:
            for epochs in epochs_list:
                for bs in batch_sizes:
                    for lr in lrs:
                        for alpha in alphas:
                            for l1_coeff in l1_coeffs:
                                # set the file name to save this run's training results
                                history_file_name = "dims"
                                for d in dims:
                                    history_file_name += str(d) + "-"
                                history_file_name = history_file_name[:-1]
                                history_file_name += "_" + acts[0] + "_"
                                history_file_name += "ep" + str(epochs) + "_"
                                if bs == len(x):
                                    history_file_name += "batch_"
                                else:
                                    history_file_name += "bs" + str(bs) + "_"
                                history_file_name += "lr" + str(lr) + "_"
                                history_file_name += "alpha" + str(alpha) + "_"
                                history_file_name += "l1-" + str(l1_coeff) + ".json"

                                # create directory to save the training results
                                Path(results_dir).mkdir(parents=True, exist_ok=True)

                                # train the same model with heavy ball and Nesterov's momentum;
                                # if alpha (momentum coeff) is zero, then train it without momentum.
                                # Save a json file for each model containing its architecture, hyperparams and
                                # training histories for each type of momentum (and without momentum too)
                                s = 0   # for classical momentum (heavy ball)
                                net = NeuralNetwork(dims, acts)
                                date_time_of_run = datetime.now().strftime("%d/%m/%Y %H:%M")

                                try:
                                    hist_hb = net.fit(x=x, y=y, epochs=epochs, batch_size=bs, lr=lr,
                                                      alpha=alpha, s=s, l1_coeff=l1_coeff, norm_threshold=norm_threshold)

                                    if alpha != 0:
                                        # training for Nesterov's momentum too
                                        net = NeuralNetwork(dims, acts)
                                        s = 1   # for Nesterov's momentum
                                        date_time_of_run2 = datetime.now().strftime("%d/%m/%Y %H:%M")
                                        hist_nm = net.fit(x=x, y=y, epochs=epochs, batch_size=bs, lr=lr, alpha=alpha, s=s,
                                                          l1_coeff=l1_coeff, norm_threshold=norm_threshold)
                                        # save the histories
                                        if os.path.isfile(results_dir + history_file_name):
                                            # if the file for this model already exists, add the new data to it
                                            with open(results_dir + history_file_name, 'r') as f:
                                                content = json.load(f)
                                                content['histories']['heavy_ball'][date_time_of_run] = hist_hb
                                                content['histories']['nesterov_momentum'][date_time_of_run2] = hist_nm
                                        else:
                                            # the file does not already exist
                                            content = {
                                                'architecture': {'dims': dims, 'activations': acts},
                                                'hyperparams': {
                                                    'epochs': epochs,
                                                    'batch_size': bs,
                                                    'learning_rate': lr,
                                                    'alpha': alpha,
                                                    'l1_coeff:': l1_coeff
                                                },
                                                'histories': {
                                                    'heavy_ball': {date_time_of_run: hist_hb},
                                                    'nesterov_momentum': {date_time_of_run2: hist_nm},
                                                }
                                            }

                                    else:
                                        # it means that hist_hb contains the history of a training without momentum,
                                        # because regardless of s, alpha was 0.
                                        hist_sgd = hist_hb
                                        if os.path.isfile(results_dir + history_file_name):
                                            # if the file for this model already exists, add the new data to it
                                            with open(results_dir + history_file_name, 'r') as f:
                                                content = json.load(f)
                                                content['histories']['sgd'][date_time_of_run] = hist_sgd
                                        else:
                                            # the file does not already exist
                                            content = {
                                                'architecture': {'dims': dims, 'activations': acts},
                                                'hyperparams': {
                                                    'epochs': epochs,
                                                    'batch_size': bs,
                                                    'learning_rate': lr,
                                                    'alpha': alpha,
                                                    'l1_coeff:': l1_coeff
                                                },
                                                'histories': {
                                                    'sgd': {date_time_of_run: hist_sgd}
                                                }
                                            }

                                    # actually write the training results on the file
                                    with open(results_dir + history_file_name, 'w') as f:
                                        json.dump(content, f, indent='\t')
                                except RuntimeWarning:
                                    pass


def add_plot(path: str, opt_err: float, label=None):
    """
    Add some training curves to a plot being created
    :param path: path of the file containing the training history
    :param opt_err: value of the optimal error J* (used to compute the error gap)
    :param label: label of the curve
    """
    with open(path, 'r') as f:
        hist = json.load(f)
    last_run = sorted(list(hist.keys()))[-1]
    loss = np.array(hist[last_run]['tr_loss'])
    gap = (loss - opt_err) / opt_err
    plt.plot(gap, label=label)


def show_err_and_plot(models: List, opt_err: float, labels=None):
    """
    Prints some error data about the models and plots them
    :param models: list of model entries, return value of pick_best_models
    :param opt_err: value of the optimal error J* (used to compute the error gap)
    :param labels: list of labels for each model
    """
    for i, model in enumerate(models):
        err_gap = (np.array(model['results']['tr']) - opt_err) / opt_err
        file_name = model['file_name'].split('/')[-1]
        print(f"File name: {file_name}\nArchitecture: {model['architecture']}\nHyperparams: {model['hyperparams']}")
        print(f"Last error: {model['last_value_tr']}\nLast error gap: {err_gap[-1]}")
        print(f"Best error {model['best_value_tr']} (at position {model['best_value_tr_epoch']})")
        print(f"Best error gap: {err_gap[model['best_value_tr_epoch']]}\n")
        label = file_name[:-5] + f"_{model['alg_type']}" if labels is None else labels[i]
        plt.plot(err_gap, label=label)
    plt.yscale('log')
    plt.xlabel("Epochs")
    plt.ylabel("Error gap (log scale)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == '__main__':
    # Executing GridSearch
    gird_search_results_dir = "../grid_search_results/"
    grid_search(gird_search_results_dir)

    # Pick and compare best models
    dirs = [gird_search_results_dir]
    select = 'last'
    best_sgd = pick_best_models(dirs=dirs, alg_type="sgd", n_models=1, only_sigmoid=True, select=select)
    best_hb = pick_best_models(dirs=dirs, alg_type="heavy_ball", n_models=1, only_sigmoid=True, select=select)
    best_nag = pick_best_models(dirs=dirs, alg_type="nesterov_momentum", n_models=1, only_sigmoid=True, select=select)
    
    # Show error gaps
    opt_err = 0.0029    # example optimal error
    all_picked_models = best_sgd + best_hb + best_nag
    show_err_and_plot(models=all_picked_models, opt_err=opt_err, labels=("Best SGD", "Best HB", "Best NAG"))
