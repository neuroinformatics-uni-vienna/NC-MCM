#!/usr/bin/env python3
"""
Target-mode diagnostics for T0 / T1 / T2 / T2b.

For each configuration, instantiates BanditTaskNeuroPixelsDataset, prints the
final state labels (after trimming and any relabelling), reports class counts,
runs check_state_transitions(), and saves plots + a diagnostics.json.

Output layout
-------------
results/target_mode_diagnostics/
    T0_full_none/
        diagnostics.json
        state_counts_full.png
        state_counts_train.png
        state_counts_val.png
    T1_full_hold_to_choosing/
        ...
    T2_full_choosing_to_outcome/
        ...
    T2b_full_choosing_to_correctness/
        ...

Usage
-----
    python scripts/target_mode_diagnostics.py
    python scripts/target_mode_diagnostics.py --session JPAS_0023_20230922
    python scripts/target_mode_diagnostics.py --tags T0_full_none T1_full_hold_to_choosing
    python scripts/target_mode_diagnostics.py --recompute_cache
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ncmcm.data_loaders.bandit_task import BanditTaskNeuroPixelsDataset


# ---------------------------------------------------------------------------
# Target configuration catalogue
# ---------------------------------------------------------------------------

TARGET_CONFIGS = {
    'T0_full_none': {
        'label': 'T0 — Old full multi-state baseline (no fusion)',
        'b_mode': 'full',
        'choosing_state_mode': 'side',
        'state_transitions': None,
        'apply_hold_transitions': 'none',
        'description': (
            'b_mode=full, choosing_state_mode=side, apply_hold_transitions=none. '
            'Historical / DAP-style. Expected states: intertrial, hold, '
            'choosing left, choosing right, reward, no reward.'
        ),
    },
    'T1_full_hold_to_choosing': {
        'label': 'T1 — Hold + choice-side fusion',
        'b_mode': 'full',
        'choosing_state_mode': 'side',
        'state_transitions': BanditTaskNeuroPixelsDataset.HOLD_TO_CHOOSING_TRANSITIONS,
        'apply_hold_transitions': 'HOLD_TO_CHOOSING_TRANSITIONS',
        'description': (
            'b_mode=full, choosing_state_mode=side, '
            'apply_hold_transitions=HOLD_TO_CHOOSING_TRANSITIONS. '
            'Each contiguous hold segment + its following contiguous '
            'choosing-left/right segment are merged into a single '
            '"hold --> choosing left/right" state. '
            'Expected added states: hold --> choosing left, hold --> choosing right.'
        ),
    },
    'T2_full_choosing_to_outcome': {
        'label': 'T2 — Choice-side + outcome fusion',
        'b_mode': 'full',
        'choosing_state_mode': 'side',
        'state_transitions': BanditTaskNeuroPixelsDataset.CHOOSING_TO_OUTCOME_TRANSITIONS,
        'apply_hold_transitions': 'CHOOSING_TO_OUTCOME_TRANSITIONS',
        'description': (
            'b_mode=full, choosing_state_mode=side, '
            'apply_hold_transitions=CHOOSING_TO_OUTCOME_TRANSITIONS. '
            'Each contiguous choosing-left/right segment + its following '
            'reward/no-reward segment are merged. '
            'Expected added states: choosing left --> reward, '
            'choosing left --> no reward, choosing right --> reward, '
            'choosing right --> no reward.'
        ),
    },
    'T2b_full_choosing_to_correctness': {
        'label': 'T2b — Correctness fusion (optional comparison)',
        'b_mode': 'full',
        'choosing_state_mode': 'side',
        'state_transitions': BanditTaskNeuroPixelsDataset.CHOOSING_TO_CORRECTNESS_TRANSITIONS,
        'apply_hold_transitions': 'CHOOSING_TO_CORRECTNESS_TRANSITIONS',
        'description': (
            'b_mode=full, choosing_state_mode=side, '
            'apply_hold_transitions=CHOOSING_TO_CORRECTNESS_TRANSITIONS. '
            'Both choosing sides collapse into choosing reward / choosing no reward. '
            'Expected added states: choosing reward, choosing no reward.'
        ),
    },
}


# ---------------------------------------------------------------------------
# Canonical dataset settings (matching Prompt 029 / 032 production runs)
# ---------------------------------------------------------------------------

CANONICAL = {
    'downsample_fs': 30,
    'downsample_method': 'gaussian',
    'good_neurons_only': False,
    'hgf_model': 'binary2',
    'hgf_column': 'x_1_expected_mean',
    'gaussian_sigma_ms': 25.0,
    'normalize_method': None,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trial_split(trial_indices, seed=42, test_ratio=0.2):
    """80/20 trial-level split matching production trial_random_state=42."""
    unique_trials = np.unique(trial_indices)
    unique_trials = unique_trials[unique_trials >= 0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(unique_trials))
    shuffled = unique_trials[perm]
    n_val = max(1, int(len(shuffled) * test_ratio))
    val_set = set(shuffled[-n_val:].tolist())
    train_mask = np.array([t >= 0 and t not in val_set for t in trial_indices])
    val_mask   = np.array([t >= 0 and t in val_set     for t in trial_indices])
    return train_mask, val_mask, len(shuffled) - n_val, n_val


def _state_counts(b_dense, b_labels_dict):
    """Return {state_name: count} ordered by state integer ID."""
    c = Counter(b_dense.tolist())
    return {b_labels_dict[int(k)]: int(v) for k, v in sorted(c.items())}


def _imbalance_ratio(counts):
    vals = list(counts.values())
    if not vals or min(vals) == 0:
        return float('inf')
    return max(vals) / min(vals)


def _save_bar_plot(state_counts, title, out_path, imbalance_threshold=3.0):
    labels = list(state_counts.keys())
    counts = list(state_counts.values())
    total  = sum(counts) or 1
    proportions = [c / total * 100 for c in counts]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4), 5))
    bars = ax.bar(labels, counts, color='steelblue', edgecolor='k', linewidth=0.5)
    for bar, prop in zip(bars, proportions):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            f'{prop:.1f}%',
            ha='center', va='bottom', fontsize=8,
        )
    ratio = _imbalance_ratio(state_counts)
    note = f'  [max/min ratio: {ratio:.1f}x]' if ratio > imbalance_threshold else ''
    ax.set_title(f'{title}{note}', fontsize=9)
    ax.set_ylabel('Timepoint count')
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Per-target diagnostic
# ---------------------------------------------------------------------------

def run_diagnostics(tag, cfg, data_path, out_base, recompute_cache):
    label = cfg['label']
    sep = '=' * 62
    print(f'\n{sep}')
    print(f'  {tag}')
    print(f'  {label}')
    print(sep)
    print(f'  {cfg["description"]}')

    out_dir = out_base / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = BanditTaskNeuroPixelsDataset(
        data_path=str(data_path),
        state_transitions=cfg['state_transitions'] or {},
        b_mode=cfg['b_mode'],
        choosing_state_mode=cfg['choosing_state_mode'],
        recompute_cache=recompute_cache,
        **CANONICAL,
    )

    b_dense   = ds.b.toarray().flatten()
    state_map = ds.b_labels_dict   # int -> name

    # --- full-session counts ---
    full_counts = _state_counts(b_dense, state_map)
    total = len(b_dense)

    print(f'\nState integer mapping (after trim + relabel):')
    for k, v in sorted(state_map.items()):
        print(f'  {k} -> "{v}"')

    print(f'\nFull-session state counts  (T = {total:,} timepoints):')
    for name, cnt in full_counts.items():
        print(f'  {name:45s}  {cnt:>9,}  ({cnt/total*100:.1f}%)')
    ir = _imbalance_ratio(full_counts)
    print(f'  → imbalance ratio (max/min): {ir:.1f}x', end='')
    if ir > 5:
        print('  [SEVERE]')
    elif ir > 2:
        print('  [moderate]')
    else:
        print('  [mild]')

    # --- train / val split ---
    train_mask, val_mask, n_train_trials, n_val_trials = _trial_split(ds.trial_indices)
    b_train = b_dense[train_mask]
    b_val   = b_dense[val_mask]
    train_counts = _state_counts(b_train, state_map)
    val_counts   = _state_counts(b_val,   state_map)

    n_train = int(train_mask.sum())
    n_val   = int(val_mask.sum())

    print(f'\nTrain set  ({n_train_trials} trials,  {n_train:,} timepoints):')
    for name, cnt in train_counts.items():
        print(f'  {name:45s}  {cnt:>9,}  ({cnt/n_train*100:.1f}%)')

    print(f'\nValidation set  ({n_val_trials} trials,  {n_val:,} timepoints):')
    for name, cnt in val_counts.items():
        print(f'  {name:45s}  {cnt:>9,}  ({cnt/n_val*100:.1f}%)')

    # --- transition check ---
    print(f'\ncheck_state_transitions():')
    result = ds.check_state_transitions()
    n_invalid = len(result['invalid_transitions'])
    print(f'  valid:                   {result["valid"]}')
    print(f'  observed transition types: {len(result["observed_transitions"])}')
    print(f'  invalid transitions:       {n_invalid}')
    if n_invalid:
        # group by reason
        for inv in result['invalid_transitions'][:15]:
            reason = inv.get('reason', '')
            print(f'    {inv["from"]:30s} --> {inv["to"]:30s}  '
                  f'({inv["count"]:,} times)  {reason}')
        if n_invalid > 15:
            print(f'    ... ({n_invalid - 15} more)')

    # --- save outputs ---
    _save_bar_plot(full_counts,  f'{tag}: full session',   out_dir / 'state_counts_full.png')
    _save_bar_plot(train_counts, f'{tag}: train set',      out_dir / 'state_counts_train.png')
    _save_bar_plot(val_counts,   f'{tag}: validation set', out_dir / 'state_counts_val.png')

    summary = {
        'tag': tag,
        'label': label,
        'description': cfg['description'],
        'configuration': {
            'b_mode':                 cfg['b_mode'],
            'choosing_state_mode':    cfg['choosing_state_mode'],
            'apply_hold_transitions': cfg['apply_hold_transitions'],
            **CANONICAL,
        },
        'state_integer_mapping': {str(k): v for k, v in sorted(state_map.items())},
        'state_names': [state_map[k] for k in sorted(state_map)],
        'n_states': len(state_map),
        'full_session': {
            'n_timepoints': int(total),
            'state_counts': full_counts,
            'state_proportions': {k: round(v / total, 6) for k, v in full_counts.items()},
            'imbalance_ratio': round(ir, 2),
        },
        'train_set': {
            'n_trials': n_train_trials,
            'n_timepoints': n_train,
            'state_counts': train_counts,
            'state_proportions': {k: round(v / n_train, 6) for k, v in train_counts.items()},
        },
        'validation_set': {
            'n_trials': n_val_trials,
            'n_timepoints': n_val,
            'state_counts': val_counts,
            'state_proportions': {k: round(v / n_val, 6) for k, v in val_counts.items()},
        },
        'transition_check': {
            'valid': result['valid'],
            'n_observed_transition_types': len(result['observed_transitions']),
            'n_invalid_transitions': n_invalid,
            'invalid_transitions': [
                {
                    'from':   inv['from'],
                    'to':     inv['to'],
                    'count':  inv['count'],
                    'reason': inv.get('reason', ''),
                }
                for inv in result['invalid_transitions']
            ],
        },
    }

    diag_path = out_dir / 'diagnostics.json'
    with open(diag_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved → {out_dir}/')

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Target mode diagnostics for T0/T1/T2/T2b')
    parser.add_argument('--session', default='JPAS_0023_20230922',
                        help='Session folder name under datasets/raw/twoArmBandit/')
    parser.add_argument('--out_base', default='results/target_mode_diagnostics',
                        help='Output base directory (relative to repo root)')
    parser.add_argument('--recompute_cache', action='store_true',
                        help='Force recompute dataset (ignore cached pkl)')
    parser.add_argument('--tags', nargs='+', default=list(TARGET_CONFIGS.keys()),
                        help='Which target tags to run (default: all)')
    args = parser.parse_args()

    data_path = REPO_ROOT / 'datasets' / 'raw' / 'twoArmBandit' / args.session
    out_base  = REPO_ROOT / args.out_base

    if not data_path.exists():
        sys.exit(f'ERROR: data_path not found: {data_path}')

    summaries = {}
    for tag in args.tags:
        if tag not in TARGET_CONFIGS:
            print(f'WARNING: unknown tag {tag!r}, skipping')
            continue
        summaries[tag] = run_diagnostics(
            tag, TARGET_CONFIGS[tag], data_path, out_base, args.recompute_cache
        )

    # -----------------------------------------------------------------------
    # Cross-target summary table
    # -----------------------------------------------------------------------
    print(f'\n\n{"=" * 62}')
    print('CROSS-TARGET SUMMARY')
    print(f'{"=" * 62}')
    header = f'{"Tag":<40}  {"States":>6}  {"Imbalance":>10}'
    print(header)
    print('-' * len(header))
    for tag, s in summaries.items():
        print(f'{tag:<40}  {s["n_states"]:>6}  {s["full_session"]["imbalance_ratio"]:>9.1f}x')

    print()
    for tag, s in summaries.items():
        print(f'\n{tag}  ({s["n_states"]} states):')
        for name, cnt in s['full_session']['state_counts'].items():
            pct = s['full_session']['state_proportions'][name] * 100
            print(f'  {name:45s}  {cnt:>9,}  ({pct:.1f}%)')

    print(f'\nAll outputs: {out_base}')


if __name__ == '__main__':
    main()
