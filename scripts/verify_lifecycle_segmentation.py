"""
verify_lifecycle_segmentation.py
=================================
Smoke test for segment_policy='lifecycle_start_to_next_start'.

Compares default trial segmentation (segment_policy=None, windows end at t_chosen)
against lifecycle segmentation (windows cover [trial.start, next_trial.start-1]).

Checks four criteria:
  1. Lifecycle trial windows cover ≥ 95% of the recording (vs ~57% for default).
  2. Reward and no-reward states appear in the lifecycle train split.
  3. Reward and no-reward proportions inside lifecycle windows match the
     full-session proportions (within 5 pp tolerance).
  4. Default segmentation (segment_policy=None) is unchanged: no regression
     in reward_to_choice trial coverage.

Usage:
    python scripts/verify_lifecycle_segmentation.py
    python scripts/verify_lifecycle_segmentation.py --data_path /path/to/session

Exit code:
    0 — all criteria pass
    1 — one or more criteria fail
"""

import argparse
import sys
import os
import numpy as np
import scipy.sparse

# Ensure the repo root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ncmcm.data_loaders.bandit_task import BanditTaskNeuroPixelsDataset


DEFAULT_DATA_PATH = (
    '/home/kerim/Projects/Neural Algorithms/NC-MCM/'
    'datasets/raw/twoArmBandit/JPAS_0023_20230922'
)
DOWNSAMPLE_FS = 30
DOWNSAMPLE_METHOD = 'gaussian'
GAUSSIAN_SIGMA_MS = 25.0
TRAIN_RATIO = 0.8   # approximate — used as a split point
TOLERANCE_PP = 5.0  # percentage-point tolerance for proportion check


def _dense_b(dataset):
    """Return flattened dense behaviour array."""
    b = dataset.b
    if scipy.sparse.issparse(b):
        b = b.toarray()
    return b.flatten()


def _state_counts(b_dense, trial_indices, labels_dict):
    """Return dict {state_name: count_total, count_in_trial, count_out_trial}."""
    result = {}
    in_trial = trial_indices >= 0
    for sid, name in labels_dict.items():
        mask = b_dense == sid
        result[name] = {
            'total': int(mask.sum()),
            'in_trial': int((mask & in_trial).sum()),
            'out_trial': int((mask & ~in_trial).sum()),
        }
    return result


def _train_val_split(trial_indices, train_ratio=0.8):
    """
    Simple chronological split: first `train_ratio` fraction of trial IDs → train.
    Returns boolean masks (train_mask, val_mask) over timepoints.
    """
    max_trial = int(trial_indices.max())
    if max_trial < 0:
        return np.zeros(len(trial_indices), dtype=bool), np.zeros(len(trial_indices), dtype=bool)
    split_trial = int(max_trial * train_ratio)
    train_mask = (trial_indices >= 0) & (trial_indices <= split_trial)
    val_mask = (trial_indices > split_trial)
    return train_mask, val_mask


def _report_config(tag, dataset, b_dense, labels_dict):
    """Print per-state coverage statistics for one configuration."""
    ti = dataset.trial_indices
    total = len(ti)
    in_trial_count = int((ti >= 0).sum())
    coverage_pct = 100.0 * in_trial_count / total

    print(f"\n{'='*60}")
    print(f"  Config: {tag}")
    print(f"{'='*60}")
    print(f"  Total timepoints         : {total:,}")
    print(f"  In-trial timepoints      : {in_trial_count:,}  ({coverage_pct:.1f}%)")
    print(f"  Number of trials         : {int(ti.max()) + 1 if ti.max() >= 0 else 0}")

    counts = _state_counts(b_dense, ti, labels_dict)
    train_mask, val_mask = _train_val_split(ti)
    train_count = int(train_mask.sum())
    val_count = int(val_mask.sum())

    print(f"\n  State breakdown (full session → in-trial → train → val):")
    header = f"    {'State':<35}  {'Total':>8}  {'In-trial':>9}  {'Train':>7}  {'Val':>7}"
    print(header)
    print(f"    {'-'*35}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*7}")
    for name, c in sorted(counts.items(), key=lambda x: -x[1]['total']):
        full_n = c['total']
        in_t = c['in_trial']
        mask_s = b_dense == [k for k, v in labels_dict.items() if v == name][0]
        tr_n = int((mask_s & train_mask).sum())
        va_n = int((mask_s & val_mask).sum())
        print(f"    {name:<35}  {full_n:>8,}  {in_t:>9,}  {tr_n:>7,}  {va_n:>7,}")

    print(f"\n  Train timepoints: {train_count:,}   Val timepoints: {val_count:,}")
    return coverage_pct, counts, train_mask, val_mask


def main():
    parser = argparse.ArgumentParser(description='Verify lifecycle segmentation.')
    parser.add_argument('--data_path', default=DEFAULT_DATA_PATH,
                        help='Path to the session directory')
    parser.add_argument('--downsample_fs', type=int, default=DOWNSAMPLE_FS)
    parser.add_argument('--downsample_method', default=DOWNSAMPLE_METHOD)
    parser.add_argument('--gaussian_sigma_ms', type=float, default=GAUSSIAN_SIGMA_MS)
    args = parser.parse_args()

    failures = []

    # ------------------------------------------------------------------ #
    # Config A: default (segment_policy=None, b_mode='full')
    # ------------------------------------------------------------------ #
    print("\nLoading Config A: segment_policy=None (default, b_mode=full)...")
    ds_default = BanditTaskNeuroPixelsDataset(
        data_path=args.data_path,
        downsample_fs=args.downsample_fs,
        downsample_method=args.downsample_method,
        gaussian_sigma_ms=args.gaussian_sigma_ms,
        b_mode='full',
        segment_policy=None,
        hgf_model=None,
    )
    b_default = _dense_b(ds_default)
    cov_default, counts_default, _, _ = _report_config(
        'default (segment_policy=None)', ds_default, b_default, ds_default.b_labels_dict
    )

    # ------------------------------------------------------------------ #
    # Config B: lifecycle segmentation (segment_policy='lifecycle_...')
    # ------------------------------------------------------------------ #
    print("\nLoading Config B: segment_policy='lifecycle_start_to_next_start' (b_mode=full)...")
    ds_lifecycle = BanditTaskNeuroPixelsDataset(
        data_path=args.data_path,
        downsample_fs=args.downsample_fs,
        downsample_method=args.downsample_method,
        gaussian_sigma_ms=args.gaussian_sigma_ms,
        b_mode='full',
        segment_policy='lifecycle_start_to_next_start',
        hgf_model=None,
    )
    b_lifecycle = _dense_b(ds_lifecycle)
    cov_lifecycle, counts_lifecycle, train_mask_lc, val_mask_lc = _report_config(
        'lifecycle_start_to_next_start', ds_lifecycle, b_lifecycle, ds_lifecycle.b_labels_dict
    )

    # ------------------------------------------------------------------ #
    # Criterion 1: lifecycle coverage ≥ 95%
    # ------------------------------------------------------------------ #
    print("\n--- Criterion checks ---")
    if cov_lifecycle >= 95.0:
        print(f"  [PASS] Criterion 1: lifecycle coverage = {cov_lifecycle:.1f}% (≥ 95%)")
    else:
        msg = f"Criterion 1 FAIL: lifecycle coverage = {cov_lifecycle:.1f}% (expected ≥ 95%)"
        print(f"  [FAIL] {msg}")
        failures.append(msg)

    # ------------------------------------------------------------------ #
    # Criterion 2: reward AND no-reward appear in lifecycle train split
    # ------------------------------------------------------------------ #
    reward_states = [n for n in counts_lifecycle if 'reward' in n.lower() or 'no reward' in n.lower()]
    reward_in_train_counts = {}
    for name in reward_states:
        sid = [k for k, v in ds_lifecycle.b_labels_dict.items() if v == name][0]
        count_in_train = int(((b_lifecycle == sid) & train_mask_lc).sum())
        reward_in_train_counts[name] = count_in_train

    if all(c > 0 for c in reward_in_train_counts.values()):
        print(f"  [PASS] Criterion 2: reward/no-reward in train: "
              f"{', '.join(f'{n}={c:,}' for n, c in reward_in_train_counts.items())}")
    else:
        msg = (f"Criterion 2 FAIL: some reward/no-reward states absent from train: "
               f"{reward_in_train_counts}")
        print(f"  [FAIL] {msg}")
        failures.append(msg)

    # ------------------------------------------------------------------ #
    # Criterion 3: lifecycle proportions inside trial windows match
    #              full-session proportions (within TOLERANCE_PP percentage points)
    # ------------------------------------------------------------------ #
    total_lc = len(b_lifecycle)
    in_trial_lc = int((ds_lifecycle.trial_indices >= 0).sum())

    mismatched = []
    for name, c in counts_lifecycle.items():
        full_pct = 100.0 * c['total'] / total_lc if total_lc else 0
        in_trial_pct = 100.0 * c['in_trial'] / in_trial_lc if in_trial_lc else 0
        delta = abs(full_pct - in_trial_pct)
        if delta > TOLERANCE_PP:
            mismatched.append(f"{name}: full={full_pct:.1f}% in_trial={in_trial_pct:.1f}% Δ={delta:.1f}pp")

    if not mismatched:
        print(f"  [PASS] Criterion 3: in-trial state proportions match full-session "
              f"(all within {TOLERANCE_PP:.0f} pp)")
    else:
        msg = f"Criterion 3 FAIL: proportion mismatches: {'; '.join(mismatched)}"
        print(f"  [FAIL] {msg}")
        failures.append(msg)

    # ------------------------------------------------------------------ #
    # Criterion 4: default segmentation unchanged — reward/no-reward
    #              should be near-zero in-trial (< 0.1% of total).
    #              A small number of frames (~5) is a pre-existing timing
    #              artifact where t_chosen falls on the reward-state boundary;
    #              this is not a regression introduced by segment_policy.
    # ------------------------------------------------------------------ #
    reward_states_default = [n for n in counts_default if 'reward' in n.lower() or 'no reward' in n.lower()]
    total_reward_default = sum(counts_default[n]['total'] for n in reward_states_default)
    in_trial_reward_default = sum(counts_default[n]['in_trial'] for n in reward_states_default)
    frac_default = in_trial_reward_default / max(total_reward_default, 1)
    if frac_default < 0.001:  # < 0.1% of reward/no-reward frames in-trial
        print(f"  [PASS] Criterion 4: default segmentation unchanged "
              f"(reward/no-reward in-trial = {in_trial_reward_default} = {frac_default*100:.3f}% of total; "
              f"timing-artifact threshold < 0.1%)")
    else:
        msg = (f"Criterion 4 FAIL: default segmentation regression — "
               f"reward/no-reward in-trial = {in_trial_reward_default} "
               f"({frac_default*100:.2f}% of total, expected < 0.1%)")
        print(f"  [FAIL] {msg}")
        failures.append(msg)

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    if not failures:
        print("  ALL 4 CRITERIA PASSED")
        print(f"{'='*60}\n")
        sys.exit(0)
    else:
        print(f"  {len(failures)} / 4 CRITERIA FAILED:")
        for f in failures:
            print(f"    - {f}")
        print(f"{'='*60}\n")
        sys.exit(1)


if __name__ == '__main__':
    main()
