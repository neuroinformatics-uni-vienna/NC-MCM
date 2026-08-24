import numpy as np
from torch.utils.data import TensorDataset, Subset


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
) -> tuple[Subset, Subset]:
    """Splits the dataset into training and test sets without leakage, preserving temporal order.

    Strategy:
        1. The dataset (a time series) is divided into contiguous blocks of `block_size` samples.
        2. An initial contiguous train/test split is made at the block boundary closest to
           `train_size`.
        3. Behavioral label counts are checked in both sets: every label should be present in
           both sets, and its proportion between train/test should roughly match `train_size` /
           (1 - train_size), within `balance_tolerance`.
        4. For any behavior that is missing or over/under-represented on one side, the shortest
           contiguous window on the donor side (the side with the excess) that contains enough
           occurrences of that behavior is relocated to the other side. This is a heuristic: the
           window may contain other behaviors' samples too, so it is kept as short as possible to
           avoid moving more data than necessary.

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
    total_len = len(dataset)
    train_frac_target = train_size
    test_frac_target = 1.0 - train_size

    B_arr = np.ravel(dataset.tensors[2].cpu().numpy())  # behavioral labels are in the third tensor
    all_behaviors = set(np.unique(B_arr).tolist())
    behavior_total_counts = {b: int(np.sum(B_arr == b)) for b in all_behaviors}

    print("Histogram of behavioral labels in the dataset:", np.unique(B_arr, return_counts=True))

    # --- 1) Build blocks of `block_size`, used only to pick the initial split point ---
    num_of_blocks = total_len // block_size
    block_sizes = [block_size] * num_of_blocks
    if total_len % block_size != 0:
        block_sizes.append(total_len % block_size)
        num_of_blocks += 1

    cum_sizes = np.cumsum(block_sizes)
    print(f"Number of blocks: {num_of_blocks}, Block sizes: {block_sizes}")

    # --- 2) Initial contiguous split at the block boundary closest to train_size ---
    target_split_block = int(np.argmin(np.abs(cum_sizes - int(total_len * train_frac_target))))
    target_split_block = max(1, min(num_of_blocks - 1, target_split_block))
    train_end = int(cum_sizes[target_split_block - 1])

    train_indices = np.arange(0, train_end)
    test_indices = np.arange(train_end, total_len)

    print(f"Initial split: {len(train_indices)} training samples, {len(test_indices)} test samples.")

    # --- 3) Check presence + balance of each behavior ---
    def compute_counts(idx_arr):
        return {b: int(np.sum(B_arr[idx_arr] == b)) for b in all_behaviors} if len(idx_arr) else {b: 0 for b in all_behaviors}

    def find_imbalanced(train_idx, test_idx):
        train_counts = compute_counts(train_idx)
        test_counts = compute_counts(test_idx)
        imbalanced = {}
        for b in all_behaviors:
            total = behavior_total_counts[b]
            if total == 0:
                continue
            train_frac_actual = train_counts[b] / total
            deviates = abs(train_frac_actual - train_frac_target) > balance_tolerance
            missing = train_counts[b] == 0 or test_counts[b] == 0
            if deviates or missing:
                imbalanced[b] = (train_counts[b], test_counts[b], train_frac_actual)
        return imbalanced, train_counts, test_counts

    imbalanced, train_counts, test_counts = find_imbalanced(train_indices, test_indices)

    if imbalanced:
        print("\033[93mWarning: some behaviors are missing or unbalanced across train/test:\033[0m")
        for b, (tr_c, te_c, tr_f) in imbalanced.items():
            print(f"  behavior={b}: train_count={tr_c}, test_count={te_c}, "
                  f"train_frac={tr_f:.3f} (target={train_frac_target:.3f})")

        if not force_behavioral_presence:
            print("\033[93mDo you want to proceed with automatic correction? (y/n)\033[0m")
            user_input = input().lower()
            if user_input == 'n':
                exit(1)

        # --- 4) Correct imbalances by relocating minimal windows between sides ---
        for _pass in range(max_correction_passes):
            imbalanced, train_counts, test_counts = find_imbalanced(train_indices, test_indices)
            if not imbalanced:
                break

            for b, (tr_c, te_c, tr_f) in imbalanced.items():
                total = behavior_total_counts[b]
                target_train_count = int(round(train_frac_target * total))
                deficit_train = target_train_count - tr_c  # >0 => train needs more, <0 => test needs more

                if deficit_train > 0:
                    donor_idx, recipient = test_indices, "train"
                    needed = deficit_train
                elif deficit_train < 0:
                    donor_idx, recipient = train_indices, "test"
                    needed = -deficit_train
                else:
                    continue

                window = _find_minimal_window(donor_idx, B_arr, b, needed)
                if window is None or len(window) == 0:
                    print(f"\033[93mWarning: not enough occurrences of behavior {b} on the donor "
                          f"side to correct the balance. Leaving as is.\033[0m")
                    continue

                donor_size_before = len(donor_idx)
                if len(window) > 0.5 * donor_size_before:
                    print(f"\033[93mWarning: correcting behavior {b} requires moving a window of "
                          f"{len(window)} samples out of {donor_size_before} on the donor side "
                          f"(>50%). Proceeding anyway, but this may cost significant information "
                          f"to that side.\033[0m")

                if recipient == "train":
                    train_indices = np.union1d(train_indices, window)
                    test_indices = np.setdiff1d(test_indices, window)
                else:
                    test_indices = np.union1d(test_indices, window)
                    train_indices = np.setdiff1d(train_indices, window)

                print(f"Moved {len(window)} samples (window containing behavior {b}) to {recipient}.")

        imbalanced, train_counts, test_counts = find_imbalanced(train_indices, test_indices)
        if imbalanced:
            print("\033[93mWarning: after correction, some behaviors are still imbalanced or "
                  "missing from one side:\033[0m")
            for b, (tr_c, te_c, tr_f) in imbalanced.items():
                print(f"  behavior={b}: train_count={tr_c}, test_count={te_c}, train_frac={tr_f:.3f}")

    train_indices = np.sort(train_indices)
    test_indices = np.sort(test_indices)

    print(f"Final split: {len(train_indices)} training samples, {len(test_indices)} test samples "
          f"({len(train_indices) / total_len:.3f} / {len(test_indices) / total_len:.3f}).")

    # Write a table of the final counts for each behavior in train/test
    print("\nFinal counts of behavioral labels in train/test:")
    print(f"{'Behavior':<15} {'Train Count':<12} {'Test Count':<12} {'Train Fraction':<15}")
    print("-" * 60)
    for b in sorted(all_behaviors):
        tr_c = train_counts[b]
        te_c = test_counts[b]
        tr_f = tr_c / behavior_total_counts[b] if behavior_total_counts[b] > 0 else 0.0
        print(f"{b:<15} {tr_c:<12} {te_c:<12} {tr_f:<15.3f}")
    # And add a final row for the total counts
    total_train = sum(train_counts.values())
    total_test = sum(test_counts.values())
    total_frac = total_train / (total_train + total_test) if (total_train + total_test) > 0 else 0.0
    print("-" * 60)
    print(f"{'Total':<15} {total_train:<12} {total_test:<12} {total_frac:<15.3f}")
    
    train_subset = Subset(dataset, train_indices.tolist())
    test_subset = Subset(dataset, test_indices.tolist())

    return train_subset, test_subset, train_indices, test_indices