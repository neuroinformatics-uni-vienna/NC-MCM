import numpy as np
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split


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


def test_timeseries_train_test_split():
    X = np.random.rand(50, 10)
    B = np.random.rand(50)
    win = 5

    X_paired, B_1 = prep_data(X, B, win)
    X_train, X_test, B_train_1, B_test_1 = timeseries_train_test_split(X_paired, B_1)

    assert X_train.shape[0] == B_train_1.shape[0]
    assert X_test.shape[0] == B_test_1.shape[0]