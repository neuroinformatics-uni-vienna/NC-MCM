"""
@authors:
Akshey Kumar
Vittorio Boarini
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from scipy import signal



def prep_data(x, b, win=15):
    """
    Prepares the data for the BundleNet algorithm by formatting the input neuronal and behavioral traces.

    Parameters:
        x : np.ndarray
            Raw neuronal traces of shape (t, n), where n is the number of neurons and t is the number of time steps.
        b : np.ndarray
            Raw behavioral traces of shape (t,), representing the behavioral data corresponding to the neuronal
            traces.
        win : int, optional
            Length of the window to feed as input to the algorithm. If win > 1, a slice of the time series is used 
            as input.

    Returns:
        x_paired : np.ndarray
            Paired neuronal traces of shape (m, 2, win, n), where m is the number of paired windows,
            2 represents the current and next time steps, win is the length of each window,
            and n is the number of neurons.
        b_1 : np.ndarray
            Behavioral traces corresponding to the next time step, of shape (m,). Each value represents
            the behavioral data corresponding to the next time step in the paired neuronal traces.

    """
    if x.shape[0] != b.shape[0]:
        raise ValueError("The number of time steps in x must match the length of b.")

    if win + 1 > x.shape[0]:
        raise ValueError("The window must be smaller than number of time steps.")

    win += 1
    x_win = np.zeros((x.shape[0] - win + 1, win, x.shape[1]))
    for i, _ in enumerate(x_win):
        x_win[i] = x[i:i + win]

    xwin0, xwin1 = x_win[:, :-1, :], x_win[:, 1:, :]
    b_1 = b[win - 1:]
    x_paired = np.array([xwin0, xwin1])
    x_paired = np.transpose(x_paired, axes=(1, 0, 2, 3))

    return x_paired, b_1


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


class CustomDataset(Dataset):
    def __init__(self, x_, b_, device):
        """
        Initialize the dataset.

        Parameters:
        x_ :np.ndarray
            Input data of shape (n_samples, ...)
            or
            (np.ndarray, np.ndarray, np.ndarray)
            Tuple of np.ndarray of shape (n_samples, ...) for subssystem BunDLe-Net.
        b_ : np.ndarray
            Target data of shape (n_samples, ...).
        device : torch.device
            Device where the tensors should be created.
        """
        self.x_ = (x_,) if not isinstance(x_, tuple) else x_
        self.b_ = b_
        self.device = device

    def __len__(self):
        return len(self.b_)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.

        Parameters:
            idx : int
                Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing a tuple of input tensors and the corresponding target tensor.
        """
        x_tuple = tuple(torch.tensor(x[idx], dtype=torch.float, device=self.device) for x in self.x_)
        b = torch.tensor(self.b_[idx], device=self.device)

        return x_tuple, b


def torch_batch_prep(x_, b_, device, batch_size=100, shuffle=True):
    """
    Prepare datasets for PyTorch by creating batches.

    Parameters:
        x_ : list or tuple of lists
            Input data, where x_ can be a single list or a tuple of lists/arrays.
        b_ : list
            Target data.
        batch_size : int, optional
            Size of the batches to be created. Default is 100.
        device : torch.device
            Device where the tensors should be created.
        batch_size : int, optional
            Size of the batches to be created. Default is 100.
        shuffle : bool, optional
            Defines whether the data should be reshuffled at every epoch
    Returns:
        DataLoader : PyTorch DataLoader
            DataLoader containing batches of input data and target data.
    """
    dataset = CustomDataset(x_, b_, device=device)
    batch_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return batch_loader


class GaussianNoise(nn.Module):
    def __init__(self, mean=0, stddev=0.05):
        super(GaussianNoise, self).__init__()
        self.mean = mean
        self.stddev = stddev

    def forward(self, x):
        if self.training and self.stddev > 0:
            return x + torch.normal(self.mean, self.stddev, size=x.shape, device=x.device)
        return x

