import torch
import numpy as np
from ncmcm.bundlenet.utils import (
    prep_data, timeseries_train_test_split, torch_batch_prep,
    segment_trial_boundaries, LazyTrialWindowDataset, prep_data_trials_lazy,
    trial_train_test_split_lazy, segment_trials, prep_data_trials,
    trial_train_test_split,
)


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


# ---------------------------------------------------------------------------
# Lazy trial-based utilities
# ---------------------------------------------------------------------------

def _make_toy_trial_data(n_neurons=4, rng_seed=0):
    """Build a small deterministic dataset with 3 clear trials.

    Trial boundaries are defined by the label 0 ('intertrial').
    Layout (50 timesteps):
      t=0..9   intertrial (10 steps)
      t=10..24 choosing    (15 steps)
      t=25..34 intertrial (10 steps)
      t=35..49 choosing    (15 steps)
    """
    rng = np.random.default_rng(rng_seed)
    T = 50
    X = rng.random((T, n_neurons)).astype(np.float32)
    B = np.zeros(T, dtype=np.int64)
    B[10:25] = 1   # choosing
    B[35:50] = 1   # choosing
    b_labels_dict = {0: 'intertrial', 1: 'choosing'}
    return X, B, b_labels_dict


def test_segment_trial_boundaries_count():
    X, B, b_labels_dict = _make_toy_trial_data()
    boundaries = segment_trial_boundaries(B, b_labels_dict, 'intertrial')
    # Two trials: [0,25) and [25,50)
    assert len(boundaries) == 2
    assert boundaries[0] == (0, 25)
    assert boundaries[1] == (25, 50)


def test_lazy_trial_dataset_shape():
    X, B, b_labels_dict = _make_toy_trial_data()
    win = 5
    boundaries = segment_trial_boundaries(B, b_labels_dict, 'intertrial')
    ds = LazyTrialWindowDataset(X, B, boundaries, win)
    M, two, w, N = ds.shape
    assert two == 2
    assert w == win
    assert N == X.shape[1]
    assert M == len(ds)


def test_lazy_trial_dataset_matches_eager():
    """Each window produced by LazyTrialWindowDataset must equal prep_data_trials."""
    X, B, b_labels_dict = _make_toy_trial_data()
    win = 5

    # --- Eager path ---
    segments = segment_trials(X, B, b_labels_dict, 'intertrial')
    X_paired_eager, B_1_eager, trial_ids_eager = prep_data_trials(segments, win=win)

    # --- Lazy path ---
    boundaries = segment_trial_boundaries(B, b_labels_dict, 'intertrial')
    ds = LazyTrialWindowDataset(X, B, boundaries, win)

    assert len(ds) == len(X_paired_eager), (
        f"Pair count mismatch: lazy={len(ds)}, eager={len(X_paired_eager)}"
    )
    np.testing.assert_array_equal(ds.B_1, B_1_eager)
    np.testing.assert_array_equal(ds.trial_ids, trial_ids_eager)

    for i in range(len(ds)):
        np.testing.assert_allclose(
            ds[i], X_paired_eager[i], rtol=1e-5, atol=1e-6,
            err_msg=f"Window mismatch at index {i}"
        )


def test_prep_data_trials_lazy_matches_eager():
    """prep_data_trials_lazy returns same B_1 and trial_ids as prep_data_trials."""
    X, B, b_labels_dict = _make_toy_trial_data()
    win = 5

    # Eager
    segments = segment_trials(X, B, b_labels_dict, 'intertrial')
    _, B_1_eager, trial_ids_eager = prep_data_trials(segments, win=win)

    # Lazy
    ds, B_1_lazy, trial_ids_lazy = prep_data_trials_lazy(
        X, B, b_labels_dict, trial_start_state='intertrial', win=win
    )
    np.testing.assert_array_equal(B_1_lazy, B_1_eager)
    np.testing.assert_array_equal(trial_ids_lazy, trial_ids_eager)


def test_trial_train_test_split_lazy_sizes():
    """Lazy split produces the same number of pairs as eager split."""
    X, B, b_labels_dict = _make_toy_trial_data()
    win = 5

    # Eager
    segments = segment_trials(X, B, b_labels_dict, 'intertrial')
    X_paired, B_1, trial_ids = prep_data_trials(segments, win=win)
    (X_tr, B_tr), (X_te, B_te) = trial_train_test_split(
        X_paired, B_1, trial_ids, test_ratio=0.5, random_state=0
    )

    # Lazy
    ds, B_1_l, trial_ids_l = prep_data_trials_lazy(
        X, B, b_labels_dict, trial_start_state='intertrial', win=win
    )
    (tr_sub, B_tr_l), (te_sub, B_te_l) = trial_train_test_split_lazy(
        ds, B_1_l, trial_ids_l, test_ratio=0.5, random_state=0
    )

    assert len(tr_sub) == len(X_tr)
    assert len(te_sub) == len(X_te)
    assert len(B_tr_l) == len(B_tr)
    assert len(B_te_l) == len(B_te)
