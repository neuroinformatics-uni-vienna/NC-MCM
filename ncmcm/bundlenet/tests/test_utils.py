import torch
import numpy as np
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split, torch_batch_prep
import pytest

def test_prep_data_typical_case():
    X = np.random.rand(50, 10)
    B = np.random.rand(50)
    win = 5

    X_paired, B_1 = prep_data(X, B, win)

    assert X_paired.shape == (X.shape[0] - win, 2, win, X.shape[1])
    assert B_1.shape == (B.shape[0] - win,)
    assert np.array_equal(X_paired[1:, 0, :, :], X_paired[:-1, 1, :, :])


def test_prep_data_single_time_slice():
    X = np.random.rand(50, 10)
    B = np.random.rand(50)
    win = 1

    X_paired, B_1 = prep_data(X, B, win)

    assert X_paired.shape == (X.shape[0] - win, 2, win, X.shape[1])
    assert B_1.shape == (B.shape[0] - win,)
    assert np.array_equal(X_paired[1:, 0, :, :], X_paired[:-1, 1, :, :])


def test_prep_data_large_window():
    X = np.random.rand(50, 10)
    B = np.random.rand(50)
    win = 30

    X_paired, B_1 = prep_data(X, B, win)

    assert X_paired.shape == (X.shape[0] - win, 2, win, X.shape[1])
    assert B_1.shape == (B.shape[0] - win,)
    assert np.array_equal(X_paired[1:, 0, :, :], X_paired[:-1, 1, :, :])
    

def test_prep_data_raises_on_shape_mismatch():
    with pytest.raises(ValueError):
        prep_data(np.zeros((50, 10)), np.zeros(40), win=5)
        

def test_prep_data_raises_on_window_too_large():
    with pytest.raises(ValueError):
        prep_data(np.zeros((10, 5)), np.zeros(10), win=10)


def test_prep_data():
    X = np.random.rand(50, 10)
    B = np.random.rand(50)
    win = 5

    X_paired, B_1 = prep_data(X, B, win)

    np.testing.assert_array_equal(B_1, B[win:])
    np.testing.assert_array_equal(X_paired, [[X[i:i+win], X[i+1:i+1+win]] for i in range(len(X) - win)])


def test_timeseries_train_test_split_shapes():
    X = np.random.rand(50, 10)
    B = np.random.rand(50)
    win = 5

    X_paired, B_1 = prep_data(X, B, win)
    X_train, X_test, B_train_1, B_test_1 = timeseries_train_test_split(X_paired, B_1)

    assert X_train.shape[0] == B_train_1.shape[0]
    assert X_test.shape[0] == B_test_1.shape[0]


def test_timeseries_train_test_split_contents():
    X = np.random.rand(50, 10)
    B = np.random.rand(50)

    X_train, X_test, B_train, B_test = timeseries_train_test_split(X, B)

    i_split = 4
    n_splits = 7
    fold_sizes = np.full(n_splits, len(X) // n_splits, dtype=int)
    fold_sizes[: len(X) % n_splits] += 1

    split_start = np.sum(fold_sizes[:i_split])
    split_end = split_start + fold_sizes[i_split]

    expected_X_test = X[split_start:split_end]
    expected_X_train = np.concatenate((X[:split_start], X[split_end:]))
    expected_B_test = B[split_start:split_end]
    expected_B_train = np.concatenate((B[:split_start], B[split_end:]))
    
    np.testing.assert_array_equal(X_test, expected_X_test)
    np.testing.assert_array_equal(X_train, expected_X_train)
    np.testing.assert_array_equal(B_test, expected_B_test)
    np.testing.assert_array_equal(B_train, expected_B_train)


def test_torch_batch_prep():
    batch_size = 20
    X = np.random.rand(50, 10).astype(np.float32)
    B = np.random.rand(50)

    device = torch.device('cpu')
    dataloader = torch_batch_prep(X, B, device=device, batch_size=batch_size, shuffle=False)

    assert isinstance(dataloader, torch.utils.data.DataLoader)

    for i, (batch_x, batch_b) in enumerate(dataloader):
        assert isinstance(batch_x, torch.Tensor)
        assert isinstance(batch_b, torch.Tensor)

        np.testing.assert_array_equal(batch_x, X[i*batch_size: (i*batch_size) + batch_size])
