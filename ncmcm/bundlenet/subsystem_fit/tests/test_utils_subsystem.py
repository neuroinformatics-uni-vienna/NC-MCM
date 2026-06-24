import pytest
import torch
import numpy as np
from ncmcm.bundlenet.subsystem_fit.utils_subsystem import (
    prep_data, timeseries_train_test_split, torch_batch_prep, CustomDataset, GaussianNoise
)


# --- prep_data ---

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


def test_prep_data():
    X = np.random.rand(50, 10)
    B = np.random.rand(50)
    win = 5

    X_paired, B_1 = prep_data(X, B, win)

    np.testing.assert_array_equal(B_1, B[win:])
    np.testing.assert_array_equal(
        X_paired, [[X[i:i + win], X[i + 1:i + 1 + win]] for i in range(len(X) - win)]
    )


def test_prep_data_raises_on_shape_mismatch():
    with pytest.raises(ValueError):
        prep_data(np.zeros((50, 10)), np.zeros(40), win=5)


def test_prep_data_raises_on_window_too_large():
    with pytest.raises(ValueError):
        prep_data(np.zeros((10, 5)), np.zeros(10), win=10)


# --- timeseries_train_test_split ---

def test_timeseries_train_test_split_shapes():
    X = np.random.rand(50, 10)
    B = np.random.rand(50)
    win = 5

    X_paired, B_1 = prep_data(X, B, win)
    X_train, X_test, B_train, B_test = timeseries_train_test_split(X_paired, B_1)

    assert X_train.shape[0] == B_train.shape[0]
    assert X_test.shape[0] == B_test.shape[0]
    assert X_train.shape[0] + X_test.shape[0] == X_paired.shape[0]


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


# --- CustomDataset ---

def test_custom_dataset_single_array():
    X = np.random.rand(20, 5).astype(np.float32)
    B = np.random.randint(0, 3, size=(20,))
    device = torch.device('cpu')

    dataset = CustomDataset(X, B, device)

    assert len(dataset) == 20
    x_tuple, b = dataset[0]
    assert isinstance(x_tuple, tuple)
    assert len(x_tuple) == 1
    assert x_tuple[0].shape == (5,)
    assert isinstance(b, torch.Tensor)


def test_custom_dataset_tuple_input():
    X1 = np.random.rand(20, 3).astype(np.float32)
    X2 = np.random.rand(20, 4).astype(np.float32)
    X3 = np.random.rand(20, 5).astype(np.float32)
    B = np.random.randint(0, 3, size=(20,))
    device = torch.device('cpu')

    dataset = CustomDataset((X1, X2, X3), B, device)

    assert len(dataset) == 20
    x_tuple, b = dataset[0]
    assert isinstance(x_tuple, tuple)
    assert len(x_tuple) == 3
    assert x_tuple[0].shape == (3,)
    assert x_tuple[1].shape == (4,)
    assert x_tuple[2].shape == (5,)


# --- torch_batch_prep ---

def test_torch_batch_prep_single_input():
    batch_size = 10
    X = np.random.rand(50, 10).astype(np.float32)
    B = np.random.randint(0, 5, size=(50,))
    device = torch.device('cpu')

    dataloader = torch_batch_prep(X, B, device=device, batch_size=batch_size, shuffle=False)

    assert isinstance(dataloader, torch.utils.data.DataLoader)
    for i, (x_batch, b) in enumerate(dataloader):
        # DataLoader collates the tuple from __getitem__ into a list of tensors
        assert isinstance(x_batch, list)
        assert len(x_batch) == 1
        np.testing.assert_array_equal(x_batch[0], X[i*batch_size:(i+1)*batch_size])
        assert b.shape == (batch_size,)


def test_torch_batch_prep_tuple_input():
    X1 = np.random.rand(50, 3).astype(np.float32)
    X2 = np.random.rand(50, 4).astype(np.float32)
    X3 = np.random.rand(50, 5).astype(np.float32)
    B = np.random.randint(0, 5, size=(50,))
    device = torch.device('cpu')

    dataloader = torch_batch_prep((X1, X2, X3), B, device=device, batch_size=10, shuffle=False)

    x_batch, b = next(iter(dataloader))
    # DataLoader collates the tuple from __getitem__ into a list of tensors
    assert isinstance(x_batch, list)
    assert len(x_batch) == 3
    assert x_batch[0].shape == (10, 3)
    assert x_batch[1].shape == (10, 4)
    assert x_batch[2].shape == (10, 5)
    assert b.shape == (10,)


def test_torch_batch_prep_covers_all_samples():
    X = np.random.rand(50, 10).astype(np.float32)
    B = np.random.randint(0, 5, size=(50,))
    device = torch.device('cpu')

    dataloader = torch_batch_prep(X, B, device=device, batch_size=10, shuffle=False)

    total = sum(b.shape[0] for _, b in dataloader)
    assert total == 50


# --- GaussianNoise ---

def test_gaussian_noise_train():
    mean = 0.0
    stddev = 0.1
    X = torch.randn(50, 10)

    noise = GaussianNoise(mean=mean, stddev=stddev)
    noise.train()
    output = noise(X)

    torch.testing.assert_close((output - X).mean(), torch.tensor(mean), atol=0.05, rtol=0)
    torch.testing.assert_close((output - X).std(), torch.tensor(stddev), atol=0.05, rtol=0)


def test_gaussian_noise_eval():
    X = torch.randn(50, 10)

    noise = GaussianNoise(mean=0, stddev=0.1)
    noise.eval()
    output = noise(X)

    torch.testing.assert_close(output, X)
