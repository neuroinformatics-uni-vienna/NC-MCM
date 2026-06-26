"""
@authors:
Akshey Kumar
Vittorio Boarini
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from scipy import signal



def prep_data(x, b, win=15, n_steps=1):
    """
    Prepares the data for the BundleNet algorithm by formatting the input neuronal and behavioral traces.

    Parameters:
        x : np.ndarray
            Raw neuronal traces of shape (t, n), where n is the number of neurons and t is the number of time steps.
        b : np.ndarray
            Raw behavioral traces of shape (t,) for discrete or (t, num_beh) for continuous behaviour.
        win : int, optional
            Length of the window to feed as input to the algorithm. If win > 1, a slice of the time series is used
            as input.
        n_steps : int, optional
            Number of unrolling steps. Default is 1 (standard BunDLeNet behaviour).
            When n_steps > 1, returns sequences of n_steps+1 windows per sample for multi-step training.

    Returns:
        x_ : np.ndarray
            Shape (m, n_steps+1, win, n). Zero-copy strided view; x_[i, step, t, n] == x[i+step+t, n].
        b_ : np.ndarray
            Discrete:   shape (m,) for n_steps=1, (m, n_steps) for n_steps>1.
            Continuous: shape (m, num_beh) for n_steps=1, (m, n_steps, num_beh) for n_steps>1.

    """
    if x.shape[0] != b.shape[0]:
        raise ValueError("The number of time steps in x must match the length of b.")

    T = x.shape[0]
    m = T - win - n_steps + 1
    if m <= 0:
        raise ValueError("The window must be smaller than number of time steps.")

    swv = np.lib.stride_tricks.sliding_window_view

    # Slide a window of size `win` along time: (T-win+1, N, win) → (T-win+1, win, N)
    # Then slide a window of size n_steps+1 over those: (m, win, N, n_steps+1) → (m, n_steps+1, win, N)
    # Result: x_[i, step, t, n] == x[i + step + t, n]
    all_windows = swv(x, win, axis=0).transpose(0, 2, 1)
    x_ = swv(all_windows, n_steps + 1, axis=0).transpose(0, 3, 1, 2)

    if b.ndim == 1:
        # b_steps[i, step] == b[win + i + step]
        b_steps = swv(b[win:], n_steps, axis=0)            # (m, n_steps)
        return x_, b_steps[:, 0] if n_steps == 1 else b_steps
    else:
        # b_steps[i, step, k] == b[win + i + step, k]
        b_steps = swv(b[win:], n_steps, axis=0).transpose(0, 2, 1)  # (m, n_steps, num_beh)
        return x_, b_steps[:, 0] if n_steps == 1 else b_steps


def timeseries_train_test_split(x_paired, b_1):
    """
    Perform a train-test split for time series data without shuffling, based on a specific fold.

    Parameters:
        x_paired : np.ndarray
            Paired neuronal traces of shape (m, 2, win, n), where m is the number of paired windows,
            2 represents the current and next time steps, win-1 is the length of each window excluding the last time 
            step,and n is the number of neurons.
        b_1 : np.ndarray
            behavioral traces corresponding to the next time step, of shape (m,). Each value represents the behavioral
            data corresponding to the next time step in the paired neuronal traces.

    Returns:
        x_train : np.ndarray
            Training set of paired neuronal traces, of shape (m_train, 2, win, n), where m_train is the number of 
            paired windows in the training set.
        x_test : np.ndarray
            Test set of paired neuronal traces, of shape (m_test, 2, win, n), where m_test is the number of paired 
            windows in the test set.
        b_train_1 : np.ndarray
            behavioral traces corresponding to the next time step in the training set, of shape (m_train,).
        b_test_1 : np.ndarray
            behavioral traces corresponding to the next time step in the test set, of shape (m_test,).

    """
    # Train test split 
    kf = KFold(n_splits=7)
    for i, (train_index, test_index) in enumerate(kf.split(x_paired)):
        if i == 4:
            # Train test split based on a fold
            x_train, x_test = x_paired[train_index], x_paired[test_index]
            b_train_1, b_test_1 = b_1[train_index], b_1[test_index]

            return x_train, x_test, b_train_1, b_test_1


def torch_batch_prep(x_, b_, device, batch_size=100, shuffle=True):
    """
    Prepare datasets for PyTorch by creating batches.

    Parameters:
        x_ : np.ndarray
            Input data of shape (n_samples, ...).
        b_ : np.ndarray
            Target data of shape (n_samples, ...).
        device : torch.device
            Device where the tensors should be created.
        batch_size : int, optional
            Size of the batches to be created. Default is 100.
        shuffle : bool, optional
            Defines whether the data should be reshuffled at every epoch

    Returns:
        dataloader : torch.utils.data.DataLoader
            PyTorch DataLoader containing batches of input data and target data.

    This function prepares datasets for PyTorch by creating batches. It takes input data 'x_' and target data 'b_'
    and creates a PyTorch dataloader from them.

    The function returns the prepared batch dataloader, which will be used for training the PyTorch model.
    """
    tensor_x = torch.tensor(x_, dtype=torch.float, device=device)
    tensor_b = torch.tensor(b_, device=device)
    dataset = TensorDataset(tensor_x, tensor_b)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


class GaussianNoise(nn.Module):
    def __init__(self, mean=0, stddev=0.05):
        super(GaussianNoise, self).__init__()
        self.mean = mean
        self.stddev = stddev

    def forward(self, x):
        if self.training and self.stddev > 0:
            return x + torch.normal(self.mean, self.stddev, size=x.shape, device=x.device)
        return x

