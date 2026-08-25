import numpy as np
from torch.utils.data import TensorDataset, Subset

from rich import print

def _find_minimal_window(idx_pool: np.ndarray, B_arr: np.ndarray, behavior, needed: int):
    """Finds the shortest contiguous index window (within idx_pool) that contains at least
    `needed` occurrences of `behavior`. Returns the array of indices (subset of idx_pool)
    contained in that window, or None if idx_pool doesn't contain enough occurrences.
    """
    idx_pool = np.sort(idx_pool)
    mask = B_arr[idx_pool] == behavior
    positions = idx_pool[mask]  # actual dataset indices where the behavior occurs, within idx_pool

    if len(positions) < needed:
        return None

    best_len = None
    best_start, best_end = None, None
    for i in range(len(positions) - needed + 1):
        start = positions[i]
        end = positions[i + needed - 1]
        length = end - start + 1
        if best_len is None or length < best_len:
            best_len = length
            best_start, best_end = start, end

    window_range = np.arange(best_start, best_end + 1)
    # Only keep indices that actually belong to idx_pool (idx_pool may have holes from
    # previous corrections, so the raw range can contain indices already on the other side).
    window = np.intersect1d(window_range, idx_pool)
    return window


def _no_leakage_split(
    dataset: TensorDataset,
    train_size: float = 0.8,
    random_state: int = 42,
    block_size: int = 2000,
    force_behavioral_presence: bool = False,
    balance_tolerance: float = 0.1,
    max_correction_passes: int = 10,
    window_size: int = None,
    strategy: str = 'lossy'
) -> tuple[Subset, Subset]:
    """Splits the dataset into training and test sets without leakage, preserving temporal order.

    Args:
        dataset (TensorDataset): The dataset to be split.
        train_size (float, optional): The fraction of the dataset to be used for training. Defaults to 0.8.
        random_state (int, optional): Kept for API compatibility; the split is deterministic
            (no shuffling is performed). Defaults to 42.
        block_size (int, optional): Size of the blocks used to determine the initial contiguous
            split point. Defaults to 2000.
        force_behavioral_presence (bool, optional): If True, imbalance corrections are applied
            automatically without asking for user confirmation. If False, the user is warned and
            asked whether to proceed with the correction. Defaults to False.
        balance_tolerance (float, optional): Maximum allowed relative deviation between a
            behavior's actual train fraction and the target train fraction before a correction is
            triggered. Defaults to 0.1 (10%).
        max_correction_passes (int, optional): Maximum number of correction passes to run, since
            fixing one behavior's balance can slightly perturb others. Defaults to 10.

    Returns:
        tuple[Subset, Subset]: The training and test subsets of the dataset.
    """

    if strategy == 'lossy':
        print("Using [bold]lossy[/bold] strategy\n -> Some samples may dropped to ensure no leakage and balanced behavior representation.")
        return _lossy_no_leakage_split(
            dataset,
            train_size,
            random_state,
            window_size
        )



def _lossy_no_leakage_split(dataset, train_size, random_state, window_size):
    dataset_length = len(dataset)
    train_length = int(dataset_length * train_size)
    test_length = dataset_length - train_length

    # We make a cut in the dataset at the train_length index.
    train_indices = list(range(train_length))
    test_indices = list(range(train_length, dataset_length))

    # Cut off the last 'window_size' samples from the training set to avoid leakage.
    if window_size is not None and window_size > 0:
        if window_size >= len(train_indices):
            raise ValueError("Window size is too large for the training set.")
        train_indices = train_indices[:-window_size]

    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    print(f"[bold cyan]============ Data Preparation Recap ============[/bold cyan]")
    print(f"Total samples    : {dataset_length:5}")
    print(f"Training samples : {len(train_indices):5}")
    print(f"Testing samples  : {len(test_indices):5}")
    print(f"[bold cyan]================================================[/bold cyan]")

    return train_subset, test_subset, train_indices, test_indices