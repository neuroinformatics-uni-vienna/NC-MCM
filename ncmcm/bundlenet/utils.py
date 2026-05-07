"""
@authors:
Akshey Kumar
Vittorio Boarini
Kerim Atak
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import KFold, TimeSeriesSplit


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
        ValueError("The number of time steps in x must match the length of b.")

    if win > x.shape[0]:
        ValueError("The window must be smaller than number of time steps.")

    win += 1
    x_win = np.zeros((x.shape[0] - win + 1, win, x.shape[1]))
    for i, _ in enumerate(x_win):
        x_win[i] = x[i:i + win]

    xwin0, xwin1 = x_win[:, :-1, :], x_win[:, 1:, :]
    b_1 = b[win - 1:]
    x_paired = np.array([xwin0, xwin1])
    x_paired = np.transpose(x_paired, axes=(1, 0, 2, 3))

    return x_paired, b_1


def make_hybrid_b(b_encoded, *continuous_arrays):
    """Combine a discrete label array with one or more continuous behaviour arrays
    into a single float32 array suitable for ``b_type='hybrid'`` training.

    Parameters
    ----------
    b_encoded : array-like, shape (T,)
        Integer (or float) class indices, e.g. the output of a LabelEncoder.
    *continuous_arrays : array-like, each shape (T,) or (T, k)
        One or more continuous behaviour signals that have already been
        normalised to a fixed range (e.g. [-1, 1]).  Each 1-D array is
        reshaped to (T, 1) before concatenation.

    Returns
    -------
    b_hybrid : np.ndarray, shape (T, 1 + n_continuous), dtype float32
        Column 0 holds the class indices; subsequent columns hold the
        continuous values.
    """
    b_encoded = np.asarray(b_encoded, dtype=np.float32).reshape(-1, 1)
    parts = [b_encoded]
    for arr in continuous_arrays:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        parts.append(arr)
    return np.concatenate(parts, axis=1)


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


def timeseries_train_test_split_cv(x_paired, b_1, n_splits=5):
    """
    Perform train-test splits for time series data using TimeSeriesSplit.
    Returns all splits from the cross-validation.
    
    Parameters:
        x_paired : np.ndarray
            Paired neuronal traces of shape (m, 2, win, n), where m is the number of paired windows,
            2 represents the current and next time steps, win-1 is the length of each window excluding the last time 
            step, and n is the number of neurons.
        b_1 : np.ndarray
            Behavioral traces corresponding to the next time step, of shape (m,). Each value represents the behavioral
            data corresponding to the next time step in the paired neuronal traces.
        n_splits : int, optional
            Number of splits for TimeSeriesSplit. Default is 5.
    
    Returns:
        splits : list of tuples
            List of (x_train, x_test, b_train_1, b_test_1) tuples, one for each fold.
            Each tuple contains:
                x_train : np.ndarray - Training set of paired neuronal traces
                x_test : np.ndarray - Test set of paired neuronal traces
                b_train_1 : np.ndarray - Behavioral traces for training set
                b_test_1 : np.ndarray - Behavioral traces for test set
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    
    for train_index, test_index in tscv.split(x_paired):
        x_train, x_test = x_paired[train_index], x_paired[test_index]
        b_train_1, b_test_1 = b_1[train_index], b_1[test_index]
        splits.append((x_train, x_test, b_train_1, b_test_1))
    
    return splits


def torch_batch_prep(x_, b_, device, batch_size=100, shuffle=True):
    """
    Prepare datasets for PyTorch by creating batches.

    Parameters:
        x_ : np.ndarray or Dataset (e.g., LazyWindowedDataset, Subset)
            Input data of shape (n_samples, ...) or a PyTorch Dataset.
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
    # Check if x_ is a Dataset (lazy or Subset)
    if isinstance(x_, Dataset):
        # Use lazy batch dataset to avoid materializing all data at once
        dataset = LazyBatchDataset(x_, b_, device)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        # Original behavior for numpy arrays
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


# ---------------------------------------------------------------------------
# Lazy utilities
# ---------------------------------------------------------------------------


class LazyWindowedDataset(Dataset):
    """
    Memory-efficient Dataset that generates windowed samples on-the-fly.
    Instead of materializing all windows in memory,
    this creates windows only when requested during training.
    
    Behaves like a numpy array for compatibility with existing code.
    """
    def __init__(self, x, b, win=15):
        """
        Args:
            x: Raw data (T, N) where T=timesteps, N=neurons
            b: Behavior labels (T,)
            win: Window size (will be adjusted to win-1 in output, matching prep_data behavior)
        """
        self.x = x
        self.b = b
        self.win = win + 1  # Add 1 for the next timestep
        self.num_samples = len(x) - self.win + 1
        self._n_neurons = x.shape[1]
        
    def __len__(self):
        return self.num_samples
    
    @property
    def shape(self):
        """Return shape as if this were a materialized numpy array"""
        return (self.num_samples, 2, self.win - 1, self._n_neurons)
    
    def __getitem__(self, idx):
        """
        Returns a paired window matching prep_data output format.
        Supports integer indexing and array/list indexing.
        """
        if isinstance(idx, (int, np.integer)):
            # Single index - return one sample
            if idx < 0:
                idx = self.num_samples + idx
            if idx < 0 or idx >= self.num_samples:
                raise IndexError(f"Index {idx} out of bounds for dataset with {self.num_samples} samples")
            
            # Extract window
            window = self.x[idx:idx + self.win]  # (win, n_neurons)
            
            # Split into current and next
            x_current = window[:-1]  # (win-1, n_neurons)
            x_next = window[1:]      # (win-1, n_neurons)
            
            # Stack them
            x_paired = np.stack([x_current, x_next], axis=0)  # (2, win-1, n_neurons)
            
            return x_paired.astype(np.float32)
        
        elif isinstance(idx, (list, np.ndarray)):
            # Array indexing - return multiple samples
            return np.array([self[i] for i in idx])
        
        elif isinstance(idx, slice):
            # Slice indexing
            indices = range(*idx.indices(self.num_samples))
            return np.array([self[i] for i in indices])
        
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")
    
    def __array__(self):
        """Support numpy array conversion (materializes all data - use sparingly!)"""
        return np.array([self[i] for i in range(self.num_samples)])


def prep_data_lazy(x, b, win=15):
    """
    MEMORY-EFFICIENT version of prep_data that returns the same structure as prep_data,
    but uses a lazy Dataset for x_paired instead of materializing all windows in memory.
    
    Returns:
        x_paired : LazyWindowedDataset
            Lazy dataset of paired neuronal traces. Behaves like shape (m, 2, win, n) array
            but generates windows on-demand. Supports indexing like numpy arrays.
        b_1 : np.ndarray
            Behavioral traces corresponding to the next time step, of shape (m,).
    
    This is a drop-in replacement for prep_data() that reduces memory drastically.
    """
    # Create lazy dataset for x_paired
    x_paired = LazyWindowedDataset(x, b, win=win)
    
    # Extract behavior labels (lightweight, just a view/slice)
    win_adjusted = win + 1
    b_1 = b[win_adjusted - 1:]
    
    return x_paired, b_1


def timeseries_train_test_split_lazy(x_paired, b_1):
    """
    MEMORY-EFFICIENT version of timeseries_train_test_split that works with LazyWindowedDataset.
    
    This is a drop-in replacement for timeseries_train_test_split() with the same signature.
    
    Parameters:
        x_paired : LazyWindowedDataset
            Lazy dataset of paired neuronal traces from prep_data_lazy()
        b_1 : np.ndarray
            Behavioral traces corresponding to the next time step, of shape (m,)
    
    Returns:
        x_train : Subset or LazyWindowedDataset
            Training set of paired neuronal traces (lazy)
        x_test : Subset or LazyWindowedDataset
            Test set of paired neuronal traces (lazy)
        b_train_1 : np.ndarray
            Behavioral traces for training set
        b_test_1 : np.ndarray
            Behavioral traces for test set
    """
    from torch.utils.data import Subset
    
    total_samples = len(x_paired)
    indices = np.arange(total_samples)
    
    # Use KFold logic to match original behavior
    kf = KFold(n_splits=7, shuffle=False)
    for i, (train_index, test_index) in enumerate(kf.split(indices)):
        if i == 4:  # Use fold 4 like original implementation
            x_train = Subset(x_paired, train_index)
            x_test = Subset(x_paired, test_index)
            b_train_1 = b_1[train_index]
            b_test_1 = b_1[test_index]
            return x_train, x_test, b_train_1, b_test_1
    
    # Fallback (shouldn't reach here)
    split_idx = int(total_samples * 6/7)
    x_train = Subset(x_paired, indices[:split_idx])
    x_test = Subset(x_paired, indices[split_idx:])
    b_train_1 = b_1[:split_idx]
    b_test_1 = b_1[split_idx:]
    return x_train, x_test, b_train_1, b_test_1


class LazyBatchDataset(Dataset):
    """
    Wrapper dataset that combines a lazy windowed dataset with behavior labels
    and handles conversion to tensors on-the-fly.
    """
    def __init__(self, x_dataset, b_labels, device):
        """
        Args:
            x_dataset: LazyWindowedDataset, Subset, or similar Dataset
            b_labels: numpy array of behavior labels
            device: torch device to place tensors on
        """
        self.x_dataset = x_dataset
        self.b_labels = b_labels
        self.device = device
        
    def __len__(self):
        return len(self.x_dataset)
    
    def __getitem__(self, idx):
        """Return a single sample as tensors on the specified device"""
        x = self.x_dataset[idx]
        b = self.b_labels[idx]
        
        # Convert to tensors
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        b_tensor = torch.tensor(b, device=self.device)
        
        return x_tensor, b_tensor


def timeseries_train_test_split_cv_lazy(x_paired, b_1, n_splits=5):
    """
    MEMORY-EFFICIENT version of timeseries_train_test_split_cv that works with LazyWindowedDataset.
    Performs train-test splits for time series data using TimeSeriesSplit.
    Returns all splits from the cross-validation.
    
    Parameters:
        x_paired : LazyWindowedDataset
            Lazy dataset of paired neuronal traces from prep_data_lazy()
        b_1 : np.ndarray
            Behavioral traces corresponding to the next time step, of shape (m,)
        n_splits : int, optional
            Number of splits for TimeSeriesSplit. Default is 5.
    
    Returns:
        splits : list of tuples
            List of (x_train, x_test, b_train_1, b_test_1) tuples, one for each fold.
            Each tuple contains:
                x_train : Subset - Training set of paired neuronal traces (lazy)
                x_test : Subset - Test set of paired neuronal traces (lazy)
                b_train_1 : np.ndarray - Behavioral traces for training set
                b_test_1 : np.ndarray - Behavioral traces for test set
    """
    from torch.utils.data import Subset
    
    total_samples = len(x_paired)
    indices = np.arange(total_samples)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    
    for train_index, test_index in tscv.split(indices):
        x_train = Subset(x_paired, train_index)
        x_test = Subset(x_paired, test_index)
        b_train_1 = b_1[train_index]
        b_test_1 = b_1[test_index]
        splits.append((x_train, x_test, b_train_1, b_test_1))
    
    return splits


# ---------------------------------------------------------------------------
# Trial-based utilities
# ---------------------------------------------------------------------------


def segment_trials(X, B, trial_start_indices=None, b_detect=None):
    """
    Segment a full-session time series into individual trials using explicit
    trial start indices.

    Label-name based segmentation (e.g. ``trial_start_state`` / ``'intertrial'``)
    has been removed. Pass ``trial_start_indices`` (array-like of ints) as the
    canonical trial start positions (e.g. from
    ``BanditTaskNeuroPixelsDataset.trial_start_indices``).

    Parameters
    ----------
    X : np.ndarray, shape (T, N)
        Neuronal traces for the full session.
    B : np.ndarray, shape (T,) or (T, k)
        Behavioral array for the full session.
    trial_start_indices : array-like of int
        First timepoint index of each trial.
    b_detect : np.ndarray, shape (T,), optional
        Ignored (kept for API compatibility).

    Returns
    -------
    segments : list of (np.ndarray, np.ndarray)
        Each element is ``(X_trial, B_trial)`` with shapes ``(t_i, N)`` and
        ``(t_i, ...)`` respectively, where ``t_i`` is the length of trial *i*.
    """
    if trial_start_indices is None:
        raise ValueError(
            "segment_trials requires 'trial_start_indices' (BanditTaskNeuroPixelsDataset.trial_start_indices)."
            " Label-name based segmentation has been removed."
        )

    return segments_from_trial_starts(X, B, trial_start_indices)


def prep_data_trials(trial_segments, win=15):
    """
    Window each trial independently and concatenate the results.

    Each trial borrows the last ``win`` timesteps of the preceding trial as
    context for its earliest windows.  This means every trial (except the
    first) produces ``len(trial)`` pairs instead of ``len(trial) - win``, and
    short trials that would otherwise be skipped can still contribute data.
    The behavioural label predicted by every pair always belongs to the current
    trial (when a full ``win``-step context is available from the predecessor).

    Trial 0 has no predecessor and behaves as before: it produces
    ``len(trial_0) - win`` pairs.  A trial is only skipped if
    ``len(context) + len(trial) <= win`` (i.e. effectively empty).

    Parameters
    ----------
    trial_segments : list of (np.ndarray, np.ndarray)
        Output of :func:`segment_trials`: each element is ``(X_trial, B_trial)``.
    win : int, optional
        Window length passed to :func:`prep_data`. Default is 15.

    Returns
    -------
    X_paired : np.ndarray, shape (M, 2, win, N)
        All paired neuronal windows concatenated across trials.
    B_1 : np.ndarray, shape (M,) or (M, k)
        Behavioral labels for the next time step, concatenated across trials.
        Works with both 1-D discrete labels and 2-D hybrid labels.
    trial_ids : np.ndarray, shape (M,), dtype int64
        Integer trial index (0-based) for each pair in ``X_paired``.
    """
    all_x_paired = []
    all_b_1 = []
    all_trial_ids = []

    prev_X_tail = None  # last `win` timesteps of the previous trial
    prev_B_tail = None

    for trial_id, (X_t, B_t) in enumerate(trial_segments):
        if prev_X_tail is not None:
            # Prepend context from the previous trial.
            # With len(prev_X_tail) == win, b_1 from prep_data equals B_t[0:],
            # so all predicted labels belong to the current trial.
            X_input = np.concatenate([prev_X_tail, X_t], axis=0)
            B_input = np.concatenate([prev_B_tail, B_t], axis=0)
        else:
            X_input = X_t
            B_input = B_t

        # Stash tail of this trial as context for the next one
        ctx = min(win, len(X_t))
        prev_X_tail = X_t[-ctx:]
        prev_B_tail = B_t[-ctx:]

        if len(X_input) <= win:
            continue  # Too short to produce any pairs even with context

        x_p, b_1 = prep_data(X_input, B_input, win)
        all_x_paired.append(x_p)
        all_b_1.append(b_1)
        all_trial_ids.append(np.full(len(x_p), trial_id, dtype=np.int64))

    if not all_x_paired:
        raise ValueError(
            "No trials produced any pairs. "
            "Ensure that at least some trials are longer than `win` timesteps."
        )

    X_paired = np.concatenate(all_x_paired, axis=0)
    B_1 = np.concatenate(all_b_1, axis=0)
    trial_ids = np.concatenate(all_trial_ids, axis=0)

    return X_paired, B_1, trial_ids


def trial_train_test_split(X_paired, B_1, trial_ids, test_ratio=0.2, random_state=None):
    """
    Randomly split pairs into train/test sets at the *trial* level.

    All pairs belonging to a given trial are kept together — a trial is
    entirely in train or entirely in test, never split across both.  This
    prevents data leakage between train and test sets and allows free
    shuffling of pairs within the training set.

    Parameters
    ----------
    X_paired : np.ndarray, shape (M, 2, win, N)
        Paired neuronal traces, e.g. from :func:`prep_data_trials`.
    B_1 : np.ndarray, shape (M,) or (M, k)
        Behavioral labels for the next time step.
    trial_ids : np.ndarray, shape (M,)
        Trial index for each pair, e.g. from :func:`prep_data_trials`.
    test_ratio : float, optional
        Fraction of trials to allocate to the test set. Default is 0.2.
    random_state : int or None, optional
        Seed for the random number generator, for reproducibility.

    Returns
    -------
    (X_train, B_train) : tuple of np.ndarray
        Training pairs and corresponding behavioral labels.
    (X_test, B_test) : tuple of np.ndarray
        Test pairs and corresponding behavioral labels.
    """
    rng = np.random.default_rng(random_state)
    unique_trials = np.unique(trial_ids).copy()
    rng.shuffle(unique_trials)

    n_test = max(1, int(np.round(len(unique_trials) * test_ratio)))
    test_trial_set = set(unique_trials[-n_test:].tolist())

    train_mask = np.array([tid not in test_trial_set for tid in trial_ids])
    test_mask = ~train_mask

    return (X_paired[train_mask], B_1[train_mask]), (X_paired[test_mask], B_1[test_mask])


# ---------------------------------------------------------------------------
# Lazy trial-based utilities
# ---------------------------------------------------------------------------


def segment_trial_boundaries(B_1d, trial_start_indices=None):
    """Return raw ``(start, end)`` index pairs for each trial — no data copied.

    Label-name based segmentation has been removed. Supply explicit
    ``trial_start_indices`` (preferred) which will be converted to
    ``(start, end)`` pairs.

    Parameters
    ----------
    B_1d : np.ndarray, shape (T,)
        1-D integer behavioural label array for the full session. Only used to
        determine the total length when converting start indices to boundaries.
    trial_start_indices : array-like of int
        Explicit trial start indices.

    Returns
    -------
    boundaries : list of (int, int)
        Each element is ``(start, end)`` — exclusive end index — covering
        exactly one trial's timesteps in ``B_1d``.
    """
    if trial_start_indices is None:
        raise ValueError(
            "segment_trial_boundaries requires 'trial_start_indices'. "
            "Label-name based segmentation has been removed."
        )

    return boundaries_from_trial_starts(trial_start_indices, len(B_1d))


class LazyTrialWindowDataset(Dataset):
    """Memory-efficient Dataset for trial-based windowing.

    Identical window format to :class:`LazyWindowedDataset` — each item is
    ``(2, win, N)`` float32 — but pairs never cross trial boundaries and
    context borrowing (the same logic as :func:`prep_data_trials`) is
    applied on-the-fly.

    Precomputes only three small integer arrays (``pair_offsets``,
    ``B_1``, ``trial_ids``) and holds raw ``X`` and ``B`` by reference so
    no large copies are ever made.

    Parameters
    ----------
    X : np.ndarray, shape (T, N)
        Full-session neuronal traces.
    B : np.ndarray, shape (T,) or (T, k)
        Full-session behavioural array (discrete or hybrid).
    trial_boundaries : list of (int, int)
        Output of :func:`segment_trial_boundaries`.
    win : int
        Window length (same semantics as :func:`prep_data`).
    """

    def __init__(self, X, B, trial_boundaries, win):
        self.X = X
        self.B = B
        self.win = win
        self._N = X.shape[1]

        # Build flat index arrays by mimicking prep_data_trials logic
        pair_offsets = []   # (abs_context_start, step_offset) per pair
        b_1_rows = []
        trial_id_rows = []

        prev_t_len = None  # length of previous trial (determines context size)
        for trial_id, (t_start, t_end) in enumerate(trial_boundaries):
            t_len = t_end - t_start
            # Determine context window prepended from previous trial.
            # Matches prep_data_trials: ctx = min(win, len(prev_trial)).
            if prev_t_len is not None:
                ctx = min(win, prev_t_len)
                context_start = t_start - ctx
                total_len = ctx + t_len
            else:
                context_start = t_start
                total_len = t_len

            prev_t_len = t_len

            if total_len <= win:
                continue  # too short

            # Number of pairs this trial produces (matches prep_data_trials exactly)
            n_pairs = total_len - win  # prep_data uses win+1 internally → win steps
            abs_start = context_start

            for step in range(n_pairs):
                pair_offsets.append((abs_start, step))
                # b_1 index: abs_start + step + win  (the label at the next step)
                b_idx = abs_start + step + win
                b_1_rows.append(b_idx)
                trial_id_rows.append(trial_id)

        if not pair_offsets:
            raise ValueError(
                "No trials produced any pairs. "
                "Ensure that at least some trials are longer than `win` timesteps."
            )

        self._pair_offsets = np.array(pair_offsets, dtype=np.int64)  # (M, 2)
        self.trial_ids = np.array(trial_id_rows, dtype=np.int64)      # (M,)

        # B_1: precompute labels — tiny compared to X_paired
        b_idx_arr = np.array(b_1_rows, dtype=np.int64)
        self.B_1 = B[b_idx_arr]  # (M,) or (M, k)

    def __len__(self):
        return len(self._pair_offsets)

    @property
    def shape(self):
        """Emulate the shape of the materialised X_paired array."""
        return (len(self._pair_offsets), 2, self.win, self._N)

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            if idx < 0:
                idx = len(self) + idx
            abs_start, step = self._pair_offsets[idx]
            i = abs_start + step
            x0 = self.X[i:i + self.win]        # (win, N)
            x1 = self.X[i + 1:i + 1 + self.win]  # (win, N)
            return np.stack([x0, x1], axis=0).astype(np.float32)  # (2, win, N)
        elif isinstance(idx, (list, np.ndarray)):
            return np.array([self[i] for i in idx])
        elif isinstance(idx, slice):
            return np.array([self[i] for i in range(*idx.indices(len(self)))])
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

    def __array__(self):
        """Materialise all windows — use only for debugging."""
        return np.array([self[i] for i in range(len(self))])


def prep_data_trials_lazy(X, B, b_labels_dict=None, win=15, trial_start_indices=None):
    """Memory-efficient variant of :func:`prep_data_trials`.

    Returns a :class:`LazyTrialWindowDataset` instead of a materialised
    ``X_paired`` array. The dataset generates windows on-the-fly during
    training so peak RAM is dominated by raw ``X`` (``T × N × 4B``) rather
    than ``X_paired`` (``M × 2 × win × N × 4B``).

    NOTE: Label-name based segmentation (``trial_start_state`` / ``intertrial``)
    has been removed to avoid ambiguity between `b_mode='full'` and
    `b_mode='decision'`. When using trial-based workflows, pass
    ``trial_start_indices`` (from the dataloader) which is the canonical
    source of trial boundaries.

    Parameters
    ----------
    X : np.ndarray, shape (T, N)
        Full-session neuronal traces (float32 recommended).
    B : np.ndarray, shape (T,) or (T, k)
        Full-session behavioural array.
    b_labels_dict : dict {int: str} or None
        Optional label mapping. Kept for compatibility but ignored when
        ``trial_start_indices`` is supplied.
    win : int, optional
        Window length. Default 15.
    trial_start_indices : array-like of int or None, optional
        Explicit trial start indices from the dataloader
        (``BanditTaskNeuroPixelsDataset.trial_start_indices``). This is now
        required for trial-based windowing; a ValueError is raised when it is
        not provided.

    Returns
    -------
    dataset : LazyTrialWindowDataset
        Lazy dataset; ``dataset.B_1`` and ``dataset.trial_ids`` carry the
        precomputed label and trial-id arrays.
    B_1 : np.ndarray
        Alias for ``dataset.B_1`` — behavioural labels per pair.
    trial_ids : np.ndarray
        Alias for ``dataset.trial_ids`` — trial index per pair.
    """
    if trial_start_indices is None:
        raise ValueError(
            "prep_data_trials_lazy requires 'trial_start_indices' (BanditTaskNeuroPixelsDataset.trial_start_indices)."
            " Label-based segmentation has been removed to avoid ambiguity."
        )

    boundaries = boundaries_from_trial_starts(trial_start_indices, len(X))
    dataset = LazyTrialWindowDataset(X, B, boundaries, win)
    return dataset, dataset.B_1, dataset.trial_ids


def trial_train_test_split_lazy(dataset, B_1, trial_ids, test_ratio=0.2, random_state=None):
    """Trial-level train/test split for a :class:`LazyTrialWindowDataset`.

    Identical split logic to :func:`trial_train_test_split` but returns
    ``torch.utils.data.Subset`` objects instead of numpy arrays so that the
    split sets remain lazy.

    Parameters
    ----------
    dataset : LazyTrialWindowDataset
        The full lazy dataset returned by :func:`prep_data_trials_lazy`.
    B_1 : np.ndarray, shape (M,) or (M, k)
        Behavioural labels (``dataset.B_1``).
    trial_ids : np.ndarray, shape (M,)
        Trial indices (``dataset.trial_ids``).
    test_ratio : float, optional
        Fraction of trials held out as test set. Default 0.2.
    random_state : int or None, optional
        RNG seed for reproducibility.

    Returns
    -------
    (train_subset, B_train) : (Subset, np.ndarray)
    (test_subset, B_test) : (Subset, np.ndarray)
    """
    from torch.utils.data import Subset

    rng = np.random.default_rng(random_state)
    unique_trials = np.unique(trial_ids).copy()
    rng.shuffle(unique_trials)

    n_test = max(1, int(np.round(len(unique_trials) * test_ratio)))
    test_trial_set = set(unique_trials[-n_test:].tolist())

    all_indices = np.arange(len(dataset))
    train_mask = np.array([tid not in test_trial_set for tid in trial_ids])
    test_mask = ~train_mask

    train_subset = Subset(dataset, all_indices[train_mask].tolist())
    test_subset = Subset(dataset, all_indices[test_mask].tolist())

    return (train_subset, B_1[train_mask]), (test_subset, B_1[test_mask])


# ---------------------------------------------------------------------------
# Trial-start-index-based utilities (preferred over label-name segmentation)
# ---------------------------------------------------------------------------


def boundaries_from_trial_starts(trial_start_indices, total_length):
    """Convert an array of trial start indices to ``(start, end)`` boundary pairs.

    This is the preferred replacement for :func:`segment_trial_boundaries` when
    trial boundaries come from the dataloader
    (``BanditTaskNeuroPixelsDataset.trial_start_indices``) rather than from a
    behavioral label name.  It works for any ``b_mode`` including ``'decision'``
    where ``'intertrial'`` may not appear as a label.

    Parameters
    ----------
    trial_start_indices : array-like of int
        First timepoint index of each trial, e.g.
        ``BanditTaskNeuroPixelsDataset.trial_start_indices``.
    total_length : int
        Total number of timesteps in the session (``len(X)``).

    Returns
    -------
    list of (int, int)
        Each tuple is ``(start, end)`` with exclusive *end*, covering exactly
        one trial's timesteps.  Trials are returned in chronological order.
    """
    starts = np.asarray(trial_start_indices, dtype=np.int64)
    ends = np.concatenate([starts[1:], [total_length]])
    return list(zip(starts.tolist(), ends.tolist()))


def segments_from_trial_starts(X, B, trial_start_indices):
    """Segment a full-session time series into trials using explicit start indices.

    This is the preferred replacement for :func:`segment_trials` when trial
    boundaries come from the dataloader
    (``BanditTaskNeuroPixelsDataset.trial_start_indices``) rather than from a
    behavioral label name.  It works for any ``b_mode``.

    Parameters
    ----------
    X : np.ndarray, shape (T, N)
        Neuronal traces for the full session.
    B : np.ndarray, shape (T,) or (T, k)
        Behavioral array for the full session.
    trial_start_indices : array-like of int
        First timepoint index of each trial, as provided by the dataloader.

    Returns
    -------
    segments : list of (np.ndarray, np.ndarray)
        Each element is ``(X_trial, B_trial)`` — same contract as
        :func:`segment_trials`.
    """
    boundaries = boundaries_from_trial_starts(trial_start_indices, len(X))
    return [(X[s:e], B[s:e]) for s, e in boundaries]

