#!/usr/bin/env python3
"""
diagnose_bandit_segment_target_specs.py — Per-spec diagnostics for all
registered BanditSpecs.

For each BanditSpec in bandit_specs.BANDIT_SPECS this script:

  1. Maps the spec to old constructor parameters via spec_to_old_params().
  2. Instantiates BanditTaskNeuroPixelsDataset (using cache if available).
  3. Reports:
       a. Segment coverage: fraction of timepoints with trial_index >= 0.
       b. Per-state counts: full session vs within trial windows.
       c. Reward/no-reward presence in trial windows (critical for T0/T1/T2/T2b).
       d. Label consistency within segments: for choice-label targets, are all
          timepoints in a segment labeled with the same choice?
       e. Segment statistics: count, mean/min/max frames per segment.
       f. check_state_transitions() result.
       g. is_consistent flag from the spec.
  4. Saves a JSON summary to results/segment_target_diagnostics/<spec_name>/.
  5. Prints a compact comparison table at the end.

Specs that require reloading (decision, decision_strict) are computed
fresh (no prior cache exists). Use --skip_uncached to skip them.

Usage
-----
    python scripts/diagnose_bandit_segment_target_specs.py
    python scripts/diagnose_bandit_segment_target_specs.py --session JPAS_0023_20230922
    python scripts/diagnose_bandit_segment_target_specs.py --specs T0_phase_full_side reward_to_choice
    python scripts/diagnose_bandit_segment_target_specs.py --skip_uncached
    python scripts/diagnose_bandit_segment_target_specs.py --recompute_cache
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ncmcm.data_loaders.bandit_task import BanditTaskNeuroPixelsDataset
from ncmcm.data_loaders.bandit_specs import (
    BANDIT_SPECS,
    BanditSpec,
    old_params_to_spec,
    spec_to_old_params,
)


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy types and tuple keys."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANONICAL = dict(
    downsample_fs=30,
    downsample_method='gaussian',
    good_neurons_only=False,
)

# These b_modes have no pre-existing cache (not run before this script).
# They require a full reload, which may take several minutes.
UNCACHED_B_MODES = {'decision', 'decision_strict'}

# States that carry behavioral outcome information — these should ideally
# appear only OUTSIDE trial windows for the 'full' target policies.
REWARD_STATES = {'reward', 'no reward', 'no_reward'}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(
    session_path: Path,
    old_params: dict,
    recompute_cache: bool,
) -> BanditTaskNeuroPixelsDataset:
    """Instantiate BanditTaskNeuroPixelsDataset for the given old params."""
    # state_transitions are stored as string keys in old_params;
    # look up the actual dict from BanditTaskNeuroPixelsDataset class constants.
    transitions_key = old_params.get('apply_hold_transitions', 'none').upper()
    transitions_dict = _resolve_state_transitions(transitions_key)

    return BanditTaskNeuroPixelsDataset(
        data_path=str(session_path),
        b_mode=old_params.get('b_mode', 'full'),
        choosing_state_mode=old_params.get('choosing_state_mode', 'side'),
        state_transitions=transitions_dict,
        recompute_cache=recompute_cache,
        **CANONICAL,
    )


def _resolve_state_transitions(key: str) -> dict:
    """Map apply_hold_transitions string to actual state_transitions dict."""
    key = key.upper().strip()
    if key in ('NONE', ''):
        return {}
    _class_attrs = {
        'HOLD_TO_CHOOSING_TRANSITIONS': BanditTaskNeuroPixelsDataset.HOLD_TO_CHOOSING_TRANSITIONS,
        'CHOOSING_TO_OUTCOME_TRANSITIONS': BanditTaskNeuroPixelsDataset.CHOOSING_TO_OUTCOME_TRANSITIONS,
        'CHOOSING_TO_CORRECTNESS_TRANSITIONS': BanditTaskNeuroPixelsDataset.CHOOSING_TO_CORRECTNESS_TRANSITIONS,
    }
    if key in _class_attrs:
        return _class_attrs[key]
    raise ValueError(f"Unknown apply_hold_transitions key: {key!r}")


# ---------------------------------------------------------------------------
# Per-spec analysis
# ---------------------------------------------------------------------------

def _serialize_transitions_result(tv: dict) -> dict:
    """Convert check_state_transitions() dict to JSON-serializable form.

    The method may return dicts with tuple keys like ('choosing left', 'reward').
    JSON requires string keys.
    """
    out = {}
    for k, v in tv.items():
        key = str(k) if isinstance(k, tuple) else k
        if isinstance(v, dict):
            out[key] = _serialize_transitions_result(v)
        elif isinstance(v, list):
            serialized_list = []
            for item in v:
                if isinstance(item, dict):
                    serialized_list.append({
                        str(ki) if isinstance(ki, tuple) else ki: _np_to_python(vi)
                        for ki, vi in item.items()
                    })
                else:
                    serialized_list.append(_np_to_python(item))
            out[key] = serialized_list
        else:
            out[key] = _np_to_python(v)
    return out


def _np_to_python(v):
    """Convert numpy scalar to Python native type."""
    if hasattr(v, 'item'):
        return v.item()
    return v

def analyse_spec(
    spec: BanditSpec,
    dataset: BanditTaskNeuroPixelsDataset,
) -> dict:
    """Compute diagnostics for one spec.

    Returns a dict suitable for JSON serialization.
    """
    b_dense = dataset.b.toarray().squeeze()   # (T,)
    trial_idx = dataset.trial_indices          # (T,) int32; -1 = outside trial
    b_labels = dataset.b_labels               # list of label names
    T = len(b_dense)

    # --- Segment coverage ---
    in_trial_mask = trial_idx >= 0
    seg_coverage = float(np.mean(in_trial_mask))

    # --- State distribution full session ---
    full_counts = Counter()
    for label_id, count in zip(*np.unique(b_dense, return_counts=True)):
        name = b_labels[label_id] if label_id < len(b_labels) else f'id={label_id}'
        full_counts[name] = int(count)

    # --- State distribution WITHIN trial windows ---
    in_trial_counts = Counter()
    for label_id, count in zip(*np.unique(b_dense[in_trial_mask], return_counts=True)):
        name = b_labels[label_id] if label_id < len(b_labels) else f'id={label_id}'
        in_trial_counts[name] = int(count)

    # --- State distribution OUTSIDE trial windows ---
    out_trial_counts = Counter()
    out_mask = ~in_trial_mask
    if out_mask.any():
        for label_id, count in zip(*np.unique(b_dense[out_mask], return_counts=True)):
            name = b_labels[label_id] if label_id < len(b_labels) else f'id={label_id}'
            out_trial_counts[name] = int(count)

    # --- Reward/no-reward in trial windows ---
    reward_in_trial: dict[str, int] = {}
    reward_in_trial_total = 0
    for state_name in REWARD_STATES:
        # match against partial label names
        matched = [n for n in full_counts if state_name in n.lower()]
        for m in matched:
            cnt = in_trial_counts.get(m, 0)
            reward_in_trial[m] = cnt
            reward_in_trial_total += cnt

    # --- Segment statistics ---
    unique_trial_ids = np.unique(trial_idx[in_trial_mask])
    n_segments = int(len(unique_trial_ids))
    frames_per_segment = np.array([
        int(np.sum(trial_idx == tid)) for tid in unique_trial_ids
    ])
    seg_stats = {}
    if n_segments > 0:
        seg_stats = {
            'count': n_segments,
            'mean_frames': float(np.mean(frames_per_segment)),
            'min_frames': int(np.min(frames_per_segment)),
            'max_frames': int(np.max(frames_per_segment)),
            'std_frames': float(np.std(frames_per_segment)),
        }
    else:
        seg_stats = {'count': 0}

    # --- Label homogeneity within segments (for choice-label modes) ---
    # A segment is "homogeneous" if all its timepoints carry the same label.
    homogeneity_result = _check_label_homogeneity(b_dense, trial_idx, unique_trial_ids)

    # --- check_state_transitions ---
    try:
        tv_raw = dataset.check_state_transitions()
        # check_state_transitions() may return a bool OR a dict with tuple keys.
        # Normalize to JSON-serializable form.
        if isinstance(tv_raw, dict):
            transitions_valid = _serialize_transitions_result(tv_raw)
        else:
            transitions_valid = bool(tv_raw)
    except Exception as exc:
        transitions_valid = f'error: {exc}'

    # --- Reward states EXPECTED to be absent? ---
    # For is_consistent=True specs, reward states should have ZERO in-trial count.
    reward_expectation = 'PASS' if reward_in_trial_total == 0 else 'FAIL'
    # But for some specs (T2/T2b fused), low counts are expected — flag separately.
    severe_reward_contamination = reward_in_trial_total > n_segments  # more than 1 per segment

    result = {
        'spec_name': spec.name,
        'is_consistent': spec.is_consistent,
        'known_issues': list(spec.known_issues),
        'old_params': spec_to_old_params(spec),
        'total_timepoints': T,
        'segment_coverage_fraction': round(seg_coverage, 4),
        'segment_coverage_pct': round(100 * seg_coverage, 2),
        'state_labels': b_labels,
        'full_session_counts': dict(full_counts),
        'in_trial_counts': dict(in_trial_counts),
        'out_of_trial_counts': dict(out_trial_counts),
        'reward_states_in_trial': reward_in_trial,
        'reward_states_in_trial_total': reward_in_trial_total,
        'reward_expectation': reward_expectation,
        'severe_reward_contamination': severe_reward_contamination,
        'segment_stats': seg_stats,
        'label_homogeneity': homogeneity_result,
        'transitions_valid': transitions_valid,
    }
    return result


def _check_label_homogeneity(
    b_dense: np.ndarray,
    trial_idx: np.ndarray,
    unique_trial_ids: np.ndarray,
) -> dict:
    """Check whether each segment is labeled with a single behavioral state.

    Relevant for choice-label modes (decision, decision_strict,
    reward_to_choice) where each segment should be one homogeneous
    choice label.  For full-phase modes, multi-label segments are expected.
    """
    if len(unique_trial_ids) == 0:
        return {'checked': False, 'reason': 'no segments'}

    n_homogeneous = 0
    n_mixed = 0
    mixed_examples: list[dict] = []

    for tid in unique_trial_ids:
        seg_labels = b_dense[trial_idx == tid]
        unique_in_seg = np.unique(seg_labels)
        if len(unique_in_seg) == 1:
            n_homogeneous += 1
        else:
            n_mixed += 1
            if len(mixed_examples) < 3:
                mixed_examples.append({
                    'trial_id': int(tid),
                    'unique_label_ids': unique_in_seg.tolist(),
                    'n_timepoints': len(seg_labels),
                })

    return {
        'checked': True,
        'n_homogeneous': n_homogeneous,
        'n_mixed': n_mixed,
        'pct_homogeneous': round(100 * n_homogeneous / max(1, n_homogeneous + n_mixed), 1),
        'mixed_examples': mixed_examples,
    }


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

_TABLE_COLS = [
    ('spec_name', 30),
    ('is_consistent', 13),
    ('seg_cov_%', 9),
    ('n_segs', 7),
    ('mean_fr', 8),
    ('reward_in_trial', 15),
    ('hom_%', 8),
    ('transitions', 11),
]


def _print_table(rows: list[dict]) -> None:
    header = '  '.join(col.ljust(w) for col, w in _TABLE_COLS)
    print('\n' + '=' * len(header))
    print('SEGMENT / TARGET SPEC DIAGNOSTICS SUMMARY')
    print('=' * len(header))
    print(header)
    print('-' * len(header))
    for r in rows:
        seg_cov = r.get('segment_coverage_pct', 'N/A')
        n_segs = r.get('segment_stats', {}).get('count', 'N/A')
        mean_fr = r.get('segment_stats', {}).get('mean_frames')
        mean_fr_s = f'{mean_fr:.1f}' if mean_fr is not None else 'N/A'
        rit = r.get('reward_states_in_trial_total', 'N/A')
        hom = r.get('label_homogeneity', {}).get('pct_homogeneous', 'N/A')
        hom_s = f'{hom:.1f}' if isinstance(hom, float) else str(hom)
        tv = r.get('transitions_valid', 'N/A')
        tv_s = 'True' if tv is True else ('False' if tv is False else str(tv)[:10])

        row_vals = [
            r.get('spec_name', '?'),
            str(r.get('is_consistent', '?')),
            str(seg_cov),
            str(n_segs),
            mean_fr_s,
            str(rit),
            hom_s,
            tv_s,
        ]
        print('  '.join(v.ljust(w) for v, (_, w) in zip(row_vals, _TABLE_COLS)))
    print('=' * len(header))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run per-spec diagnostics for all BanditSpecs.'
    )
    parser.add_argument(
        '--session',
        default='JPAS_0023_20230922',
        help='Session identifier. Default: JPAS_0023_20230922',
    )
    parser.add_argument(
        '--specs',
        nargs='+',
        default=None,
        help='Subset of spec names to run. Default: all registered specs.',
    )
    parser.add_argument(
        '--skip_uncached',
        action='store_true',
        help=(
            'Skip specs whose b_mode has no prior cache '
            '(decision, decision_strict). These require a full reload.'
        ),
    )
    parser.add_argument(
        '--recompute_cache',
        action='store_true',
        help='Force recompute even if a cache file exists.',
    )
    parser.add_argument(
        '--out_base',
        default=str(REPO_ROOT / 'results' / 'segment_target_diagnostics'),
        help='Base output directory. Default: results/segment_target_diagnostics/',
    )
    args = parser.parse_args()

    # Locate session data
    bandit_root = REPO_ROOT / 'datasets' / 'raw' / 'twoArmBandit'
    session_dirs = list(bandit_root.glob(f'{args.session}*'))
    if not session_dirs:
        print(f'[ERROR] No session directory found matching: {args.session}')
        sys.exit(1)
    session_path = session_dirs[0]
    print(f'Session: {session_path.name}')

    # Select specs
    all_specs = list(BANDIT_SPECS.values())
    if args.specs:
        unknown = [s for s in args.specs if s not in BANDIT_SPECS]
        if unknown:
            print(f'[ERROR] Unknown spec names: {unknown}')
            print(f'Available: {list(BANDIT_SPECS.keys())}')
            sys.exit(1)
        all_specs = [BANDIT_SPECS[s] for s in args.specs]

    # Skip proposed / not-implemented specs (no old_params)
    runnable = []
    for spec in all_specs:
        params = spec_to_old_params(spec)
        if not params or params.get('b_mode') is None:
            print(f'[SKIP] {spec.name} — no old_params (not yet implemented)')
            continue
        if args.skip_uncached and params.get('b_mode') in UNCACHED_B_MODES:
            print(f'[SKIP] {spec.name} — b_mode={params["b_mode"]} not cached; use --skip_uncached=False to include')
            continue
        runnable.append(spec)

    if not runnable:
        print('[WARNING] No runnable specs after filtering.')
        sys.exit(0)

    out_base = Path(args.out_base)
    out_base.mkdir(parents=True, exist_ok=True)
    table_rows: list[dict] = []
    all_results: dict[str, dict] = {}

    for spec in runnable:
        params = spec_to_old_params(spec)
        print(f'\n{"=" * 60}')
        print(f'Spec: {spec.name}')
        print(f'  b_mode          = {params.get("b_mode")}')
        print(f'  choosing_mode   = {params.get("choosing_state_mode")}')
        print(f'  transitions     = {params.get("apply_hold_transitions")}')
        print(f'  is_consistent   = {spec.is_consistent}')
        if spec.known_issues:
            for issue in spec.known_issues:
                print(f'  [!] {issue[:80]}')

        try:
            print('  Loading dataset...')
            ds = load_dataset(session_path, params, args.recompute_cache)
            print(f'  Loaded — T={ds.trial_indices.shape[0]}, labels={ds.b_labels}')

            result = analyse_spec(spec, ds)
            all_results[spec.name] = result

            # Print compact summary
            print(f'  Segment coverage  : {result["segment_coverage_pct"]:.1f}%')
            print(f'  N segments        : {result["segment_stats"].get("count", "?")}')
            mf = result["segment_stats"].get("mean_frames")
            if mf is not None:
                print(f'  Mean frames/seg   : {mf:.1f}')
            rit = result['reward_states_in_trial_total']
            print(f'  Reward in windows : {rit} ({result["reward_expectation"]})')
            hom = result['label_homogeneity']
            if hom.get('checked'):
                print(f'  Homogeneous segs  : {hom["pct_homogeneous"]:.1f}%  '
                      f'({hom["n_mixed"]} mixed)')
            print(f'  Transitions valid : {result["transitions_valid"]}')

            # Save JSON
            spec_out_dir = out_base / spec.name
            spec_out_dir.mkdir(parents=True, exist_ok=True)
            with open(spec_out_dir / 'diagnostics.json', 'w') as f:
                json.dump(result, f, indent=2, cls=_NumpyEncoder)
            print(f'  Saved to {spec_out_dir.relative_to(REPO_ROOT)}')

            table_rows.append(result)

        except Exception as exc:
            print(f'  [ERROR] {exc}')
            traceback.print_exc()
            table_rows.append({
                'spec_name': spec.name,
                'is_consistent': spec.is_consistent,
                'error': str(exc),
            })

    # Save combined summary
    combined_path = out_base / 'all_specs_summary.json'
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2, cls=_NumpyEncoder)
    print(f'\nCombined summary: {combined_path.relative_to(REPO_ROOT)}')

    _print_table(table_rows)

    # Print mismatch report
    mismatches = [r for r in table_rows if not r.get('is_consistent', True)
                  and 'error' not in r]
    if mismatches:
        print('LABEL–WINDOW MISMATCH DETECTED in:')
        for r in mismatches:
            rit = r.get('reward_states_in_trial_total', '?')
            hom = r.get('label_homogeneity', {}).get('pct_homogeneous', '?')
            print(f'  {r["spec_name"]}: reward_in_trial={rit}, '
                  f'hom={hom}%')
        print()

    consistent = [r for r in table_rows if r.get('is_consistent') and 'error' not in r]
    if consistent:
        print('CONSISTENT specs (label boundaries == trial_index boundaries):')
        for r in consistent:
            print(f'  {r["spec_name"]}')
        print()


if __name__ == '__main__':
    main()
