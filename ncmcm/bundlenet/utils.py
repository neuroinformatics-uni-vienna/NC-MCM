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


def segment_trials(X, B, b_labels_dict, trial_start_state='intertrial', b_detect=None):
    """
    Segment a full-session time series into individual trials.

    A new trial begins whenever the behavioral state transitions *into*
    ``trial_start_state``.  Any data before the first such transition (i.e.
    a partial trial at the start of the session) is discarded.

    Parameters
    ----------
    X : np.ndarray, shape (T, N)
        Neuronal traces for the full session.
    B : np.ndarray, shape (T,) or (T, k)
        Behavioral array for the full session.  Slices of this are returned
        in the output segments.  May be 2-D for hybrid mode.
    b_labels_dict : dict {int: str}
        Mapping from integer label to state name, as returned by
        ``BanditTaskNeuroPixelsDataset.b_labels_dict``.
    trial_start_state : str, optional
        Name of the behavioral state that marks the beginning of a new trial.
        Default is ``'intertrial'``.
    b_detect : np.ndarray, shape (T,), optional
        1-D integer array used *only* for boundary detection.  Pass this when
        ``B`` is 2-D (e.g. hybrid mode) so that boundaries can still be found
        from the discrete class column.  If ``None``, ``B`` itself is used
        (it must then be 1-D integer).

    Returns
    -------
    segments : list of (np.ndarray, np.ndarray)
        Each element is ``(X_trial, B_trial)`` with shapes ``(t_i, N)`` and
        ``(t_i, ...)`` respectively, where ``t_i`` is the length of trial *i*.
    """
    label_to_id = {v: k for k, v in b_labels_dict.items()}
    if trial_start_state not in label_to_id:
        raise ValueError(
            f"State '{trial_start_state}' not found in b_labels_dict. "
            f"Available states: {list(label_to_id.keys())}"
        )
    start_label = label_to_id[trial_start_state]

    b_1d = b_detect if b_detect is not None else B
    is_start = b_1d == start_label
    # Indices where the state transitions INTO start_label
    transition_points = np.where(is_start[1:] & ~is_start[:-1])[0] + 1

    if is_start[0]:
        # Session begins with the start state — include t=0
        boundaries = np.concatenate([[0], transition_points])
    else:
        # First partial trial (before the first transition) is discarded
        boundaries = transition_points

    segments = []
    for i, start in enumerate(boundaries):
        end = int(boundaries[i + 1]) if i + 1 < len(boundaries) else len(X)
        segments.append((X[start:end], B[start:end]))

    return segments


def prep_data_trials(trial_segments, win=15):
    """
    Window each trial independently and concatenate the results.

    Calls :func:`prep_data` on every trial in *trial_segments*, so windows
    never cross trial boundaries.  Trials shorter than ``win + 1`` timesteps
    are silently skipped (they would produce zero pairs).

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

    for trial_id, (X_t, B_t) in enumerate(trial_segments):
        if len(X_t) <= win:
            continue  # Too short to produce any pairs
        x_p, b_1 = prep_data(X_t, B_t, win)
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

