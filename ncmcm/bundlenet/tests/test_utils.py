import numpy as np
from ncmcm.bundlenet.utils import prep_data


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


