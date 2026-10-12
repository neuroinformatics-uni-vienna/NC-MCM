import numpy as np
from torch.utils.data import TensorDataset, Subset

from .printer import print_status

def _no_leakage_split(
    dataset: TensorDataset,
    train_size: float = 0.8,
    random_state: int = 42,
    block_size: int = 2000,
    force_behavioral_presence: bool = False,
    balance_tolerance: float = 0.1,
    max_correction_passes: int = 10,
    window_size: int = None,
    strategy: str = 'lossy',
    strategy_kwargs: dict = None
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
        return _lossy_no_leakage_split(
            dataset,
            random_state,
            window_size,
            **strategy_kwargs
        )
    elif strategy == 'block':
        return _block_no_leakage_split(
            dataset,
            random_state,
            window_size,
            **strategy_kwargs
        )
    elif strategy == 'multiblock':
        return _multiblock_no_leakage_split(
            dataset,
            **strategy_kwargs
        )
    elif strategy == 'autocorr':
        return _autocorrelation_based_split(
            dataset,
            train_size,
            **strategy_kwargs
        )
    elif strategy == 'random':
        return _random_split(
            dataset,
            **strategy_kwargs
        )
    
    raise ValueError(f"Unknown no_leakage_split strategy: {strategy}. Must be one of 'lossy', 'block', or 'multiblock'.")


def _lossy_no_leakage_split(dataset, random_state, window_size, **kwargs):
    dataset_length = len(dataset)
    train_size = kwargs.get('train_size', 0.8)
    train_length = int(dataset_length * train_size)
    offset = kwargs.get('offset', 10)

    # We make a cut in the dataset at the train_length index.
    train_indices = list(range(train_length))
    test_indices = list(range(train_length, dataset_length))

    # Cut off the last 'window_size' samples from the training set to avoid leakage.
    if window_size is not None and window_size > 0:
        if window_size >= len(train_indices):
            raise ValueError("Window size is too large for the training set.")
        train_indices = train_indices[:-window_size-offset]

    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    print_status(message="Data preparation recap - Simple Split", details={
        "Total samples": dataset_length,
        "Training samples": len(train_indices),
        "Testing samples": len(test_indices)
    })
    return train_subset, test_subset, train_indices, test_indices



def _block_no_leakage_split(dataset, **kwargs):
    total_samples = len(dataset)
    train_size = kwargs.get('train_size', 0.8)
    
    # 1. Deduct all buffer gaps from the total length to find the usable sample count

    small_interval = kwargs.get('small_interval', 15)
    large_interval = kwargs.get('large_interval', 50)
    total_buffers = 2 * kwargs.get('small_interval', 15) + kwargs.get('large_interval', 50)
    usable_samples = total_samples - total_buffers
    
    assert usable_samples > 0, "Buffer intervals exceed total dataset length!"
    
    # 2. Divide usable space equally into two subparts (Part 1: A-B, Part 2: C-D)
    half_usable = usable_samples // 2
    
    train_per_half = int(half_usable * train_size)
    test_per_half = half_usable - train_per_half
    
    # 3. Calculate index boundaries for each block
    # Block A (Train)
    start_A = 0
    end_A = start_A + train_per_half
    
    # Block B (Test)
    start_B = end_A + small_interval
    end_B = start_B + test_per_half
    
    # Block C (Train)
    start_C = end_B + large_interval
    end_C = start_C + train_per_half
    
    # Block D (Test)
    start_D = end_C + small_interval
    end_D = start_D + test_per_half
    
    # 4. Construct index lists for train and test splits
    train_indices = list(range(start_A, end_A)) + list(range(start_C, end_C))
    test_indices = list(range(start_B, end_B)) + list(range(start_D, end_D))
    
    # Summary report
    print_status(message="Data preparation recap - Single Block", details={
        "Total dataset length": total_samples,
        "Purged samples (gap)": total_samples - len(train_indices) - len(test_indices),
        "Training samples": len(train_indices),
        "Testing samples": len(test_indices)
    })

    
    # Wrap indices into PyTorch Subsets
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)
    
    return train_subset, test_subset, train_indices, test_indices


def _multiblock_no_leakage_split(dataset, **kwargs):
    total_samples = len(dataset)


    # Iteratively split the dataset into multiple blocks
    # Train Block, Small Interval, Test Block, Large Interval, Train Block, Small Interval, Test Block, ...

    # 2. Divide usable space into multiple blocks
    small_interval = kwargs.get('small_interval', 15)
    large_interval = kwargs.get('large_interval', 30)
    train_size = kwargs.get('train_size', 0.8)
    B = kwargs.get('num_blocks', 3)  # Number of train-test block pairs
    workable_length = (total_samples // B - (small_interval + large_interval))
    train_length = int(workable_length * train_size)
    test_length = workable_length - train_length

    B = (total_samples + large_interval) // (train_length + small_interval + test_length + large_interval)

    train_indices = []
    test_indices = []
    
    for i in range(B):
        start_train_block = i * (train_length + small_interval + test_length + large_interval)
        end_train_block = start_train_block + train_length

        start_test_block = end_train_block + small_interval
        end_test_block = start_test_block + test_length

        train_indices.extend(range(start_train_block, end_train_block))
        test_indices.extend(range(start_test_block, end_test_block))

    # Then reach the end of the dataset with the test set
    if total_samples > sum(len(indices) for indices in [train_indices, test_indices]):
        remaining_start = sum(len(indices) for indices in [train_indices, test_indices]) + B * (small_interval + large_interval) - (large_interval) 
        remaining_end = total_samples
        test_indices.extend(range(remaining_start, remaining_end))

    print_status(message="Data preparation recap - Multiblock", details={
        "Total dataset length": total_samples,
        "Purged samples (gap)": total_samples - len(train_indices) - len(test_indices),
        "Training samples": len(train_indices),
        "Testing samples": len(test_indices)
    })
    
    # Wrap indices into PyTorch Subsets
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)
    
    return train_subset, test_subset, train_indices, test_indices

def _autocorrelation_based_split(dataset, train_size=0.7, **kwargs):
    length = len(dataset)

    train_length = kwargs.get('train_length', int(100 * train_size))
    test_length = kwargs.get('test_length', int(train_length * (1 - train_size) / train_size))
    buffer = kwargs.get('buffer', 15)  # Buffer to avoid leakage

    test_indices, train_indices = [], []
    last_train_index = 0
    last_test_index = 0
    autocorr_threshold = kwargs.get('autocorr_threshold', 0.1)

    neuronal_data = dataset.tensors[0].cpu().numpy()
    relevant_neuronal_data = neuronal_data[:, -1, :]
    print(relevant_neuronal_data.shape)

    available_data = True
    while available_data:
        # 1. add to the training set the first train_length samples
        if len(train_indices) + train_length <= length:
            final_current_train_index = min(last_train_index + train_length, length - 1)
            train_indices.extend(range(last_train_index, final_current_train_index))
            print_status(f"Added training samples: {last_train_index} to {final_current_train_index}")
        else:
            print("Not enough samples left for a full training block. Ending split.")
            break

        # 2. Compute the autocorrelation of the last sample in the training set w.r.t. the rest of the dataset
        last_train_sample = relevant_neuronal_data[final_current_train_index]


        remaining_data = relevant_neuronal_data[final_current_train_index + 1:]
        print_status(f"Remaining data length for autocorrelation check: {len(remaining_data)}")
        if len(remaining_data) == 0:
            break

        # Compute autocorrelation
        autocorrs = np.array([np.corrcoef(last_train_sample, remaining_data[i])[0, 1] for i in range(len(remaining_data))])

        # 3. Find the first index where the autocorrelation drops below the threshold
        below_threshold_indices = np.where(autocorrs < autocorr_threshold)[0]
        if len(below_threshold_indices) == 0:
            print_status("[bold yellow]Autocorrelation never drops below the threshold. Test set will be empty for this block.[/bold yellow]")
            last_train_index = final_current_train_index + 1
            continue

        first_idx = max(buffer, below_threshold_indices[0])
        first_below_threshold = min(final_current_train_index + first_idx, length - 1)
        
        print_status(f"First index below autocorr threshold: {first_below_threshold} (autocorr={autocorrs[first_idx]:.4f})")

        final_current_test_index = min(first_below_threshold + test_length, length - 1)
        test_indices.extend(range(first_below_threshold, final_current_test_index + 1))
        print(f"Added testing samples: {first_below_threshold} to {final_current_test_index}")


        # Update the last indices for the next iteration
        last_train_index = first_below_threshold + test_length + buffer

        if last_train_index >= length:
            available_data = False

    # Summary report
    print_status(message="Data preparation recap - Autocorrelation-based Split", details={
        "Total dataset length": length,
        "Purged samples (gap)": length - len(train_indices) - len(test_indices),
        "Training samples": len(train_indices),
        "Testing samples": len(test_indices)
    })

    # Wrap indices into PyTorch Subsets
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    return train_subset, test_subset, train_indices, test_indices

def _random_split(dataset, **kwargs):
    from sklearn.model_selection import train_test_split

    train_size = kwargs.get('train_size', 0.8)
    random_state = kwargs.get('random_state', 42)
    stratified = kwargs.get('stratified', True)

    total_samples = len(dataset)
    train_length = int(total_samples * train_size)


    train_indices, test_indices = train_test_split(
        np.arange(total_samples),
        train_size=train_length,
        random_state=random_state,
        stratify=dataset.tensors[2].cpu().numpy() if stratified else None
    )

    # Summary report
    print_status(message="Data preparation recap - Random Split", details={
        "Total dataset length": total_samples,
        "Purged samples (gap)": total_samples - len(train_indices) - len(test_indices),
        "Training samples": len(train_indices),
        "Testing samples": len(test_indices)
    })

    # Wrap indices into PyTorch Subsets
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    return train_subset, test_subset, train_indices, test_indices