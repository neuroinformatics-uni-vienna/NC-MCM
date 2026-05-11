"""
Regenerate neural_behavioural_overview.html for existing grid-search run directories,
overlaying diagonal hatching on the behaviour subplot to mark test-trial windows,
upward triangle markers at each trial's choosing timestep, and dashed epoch lines
at block transitions labelled 'Better L' / 'Better R'.

Usage
-----
# Single run directory:
python scripts/regenerate_neural_behavioural_html.py \
    results/grid_search_20260510_222915/run_000_...

# All runs in a grid-search directory:
python scripts/regenerate_neural_behavioural_html.py \
    results/grid_search_20260510_222915/run_*

# Or pass the grid-search directory itself (processes all run_* subdirs):
python scripts/regenerate_neural_behavioural_html.py \
    results/grid_search_20260510_222915
"""

import argparse
import json
from pathlib import Path

import numpy as np

from ncmcm.data_loaders.bandit_task import BanditTaskNeuroPixelsDataset
from ncmcm.visualisers.neuronal_behavioural import plotting_neuronal_behavioural_plotly


def load_dataset_for_run(cfg: dict):
    """Instantiate BanditTaskNeuroPixelsDataset using the run's saved config."""
    b_type = cfg.get('b_type', 'discrete')
    hgf_model = cfg.get('hgf_model') if b_type == 'hybrid' else None
    hgf_column = cfg.get('hgf_column') if b_type == 'hybrid' else None

    kwargs = {}
    if hgf_model is not None:
        kwargs['hgf_model'] = hgf_model
    if hgf_column is not None:
        kwargs['hgf_column'] = hgf_column

    dataset = BanditTaskNeuroPixelsDataset(
        data_path=cfg['data_path'],
        downsample_fs=int(cfg['downsample_fs']),
        downsample_method=cfg['downsample_method'],
        good_neurons_only=bool(cfg['good_neurons_only']),
        state_transitions=cfg.get('apply_hold_transitions') if cfg.get('apply_hold_transitions') != 'none' else None,
        normalize_method=cfg.get('normalize_method') if cfg.get('normalize_method') != 'None' else None,
        choosing_state_mode=cfg.get('choosing_state_mode', 'side'),
        gaussian_sigma_ms=float(cfg.get('gaussian_sigma_ms', 25.0)),
        b_mode=cfg.get('b_mode', 'full'),
        **kwargs,
    )
    return dataset


def build_split_mask(trial_start_indices: np.ndarray, test_trial_ids: np.ndarray, T: int) -> np.ndarray:
    """Boolean mask of length T where True = timestep belongs to a test trial."""
    mask = np.zeros(T, dtype=bool)
    n_trials = len(trial_start_indices)
    for tid in np.unique(test_trial_ids):
        t0 = int(trial_start_indices[tid])
        t1 = int(trial_start_indices[tid + 1]) if tid + 1 < n_trials else T
        mask[t0:t1] = True
    return mask


def compute_vis_markers(dataset, data_path: Path):
    """Compute trial choice triangles, block epoch markers, and per-trial sample boundaries.

    Returns
    -------
    trial_markers : list of (sample_idx, choice_int)
    epoch_markers : list of (sample_idx, label_str)
    t_chosen_samples : np.ndarray shape (n_trials,) — sample index of t_chosen per trial
    t_choosing_samples : np.ndarray shape (n_trials,) — sample index of t_choosing per trial
    """
    trial_markers = []
    epoch_markers = []
    t_chosen_samples = None
    t_choosing_samples = None

    try:
        with open(data_path / 'metrics.json') as f:
            metrics_data = json.load(f)
        trials_raw = metrics_data['metrics']['trials']
        behavioral_time = dataset.behavioral_time  # ms, shape (T,)
        T = len(behavioral_time)

        # Sort by start — same ordering used by dataset._create_trial_indices
        usable_trials = sorted(
            [t for t in trials_raw if t.get('start') is not None],
            key=lambda t: t['start'],
        )
        n_trials = len(usable_trials)
        t_chosen_samples = np.full(n_trials, -1, dtype=np.int64)
        t_choosing_samples = np.full(n_trials, -1, dtype=np.int64)

        for i, trial in enumerate(usable_trials):
            t_ch = trial.get('t choosing')
            t_cho = trial.get('t chosen')
            choice = trial.get('choice', '').lower()

            if t_ch is not None:
                idx = int(np.clip(np.searchsorted(behavioral_time, t_ch), 0, T - 1))
                t_choosing_samples[i] = idx
                if choice in ('l', 'r'):
                    trial_markers.append((idx, 0 if choice == 'l' else 1))

            if t_cho is not None:
                t_chosen_samples[i] = int(np.clip(np.searchsorted(behavioral_time, t_cho), 0, T - 1))

    except Exception as exc:
        print(f"  Warning: could not compute trial markers: {exc}")

    try:
        bi = dataset.block_indices
        bl = dataset.block_labels
        block_change_samples = np.where(np.diff(bi) != 0)[0] + 1
        for s in block_change_samples:
            lbl = bl[int(s)]
            nice_lbl = 'Better L' if 'left' in str(lbl).lower() else 'Better R'
            epoch_markers.append((int(s), nice_lbl))
    except Exception as exc:
        print(f"  Warning: could not compute epoch markers: {exc}")

    return trial_markers, epoch_markers, t_chosen_samples, t_choosing_samples


def regenerate_run(run_dir: Path, force: bool = False):
    run_dir = Path(run_dir)
    config_path = run_dir / 'config.json'
    if not config_path.exists():
        print(f"[SKIP] {run_dir.name}: no config.json")
        return

    val_ids_path = run_dir / 'data' / 'trial_ids_validation.npy'
    if not val_ids_path.exists():
        print(f"[SKIP] {run_dir.name}: trial_ids_validation.npy not found "
              f"(run was not trial-based, no hatching to add)")
        return

    output_path = run_dir / 'figures' / 'neural_behavioural_overview.html'

    with open(config_path) as f:
        cfg = json.load(f)

    test_trial_ids = np.load(val_ids_path)

    # Try to load trial_start_indices from disk; fall back to reloading the dataset.
    tsi_path = run_dir / 'data' / 'trial_start_indices.npy'
    dataset = load_dataset_for_run(cfg)
    if tsi_path.exists():
        trial_start_indices = np.load(tsi_path)
        print(f"[INFO] {run_dir.name}: loaded trial_start_indices from disk")
    else:
        trial_start_indices = dataset.trial_start_indices
        np.save(tsi_path, trial_start_indices)
        print(f"[INFO] {run_dir.name}: saved trial_start_indices.npy ({len(trial_start_indices)} trials)")

    x = dataset.x.T.toarray().astype('float32')
    b = dataset.b.toarray().flatten()
    b_labels = dataset.b_labels
    b_colors = dataset.get_color_map_for_plotting()

    data_path = Path(cfg['data_path'])
    trial_markers, epoch_markers, t_chosen_samples, t_choosing_samples = compute_vis_markers(dataset, data_path)
    print(f"[INFO] {run_dir.name}: {len(trial_markers)} trial markers, {len(epoch_markers)} epoch markers")

    # Build split_mask: hatch from t_chosen[N-1] to t_choosing[N] so the hatch
    # aligns with the decision_strict color block boundary.
    T = x.shape[0]
    split_mask = None
    if len(test_trial_ids) > 0:
        split_mask = np.zeros(T, dtype=bool)
        if t_chosen_samples is not None and t_choosing_samples is not None:
            for tid in np.unique(test_trial_ids):
                tid = int(tid)
                t0 = int(t_chosen_samples[tid - 1]) if tid > 0 else 0
                t1 = int(t_choosing_samples[tid])
                if 0 <= t0 < t1 <= T:
                    split_mask[t0:t1] = True
        else:
            # Fallback to trial_start_indices boundaries
            n_trials = len(trial_start_indices)
            for tid in np.unique(test_trial_ids):
                t0 = int(trial_start_indices[tid])
                t1 = int(trial_start_indices[tid + 1]) if tid + 1 < n_trials else T
                split_mask[t0:t1] = True

    if split_mask is not None:
        n_test_trials = len(np.unique(test_trial_ids))
        n_test_ts = int(split_mask.sum())
        print(f"[INFO] {run_dir.name}: {n_test_trials} test trials → "
              f"{n_test_ts}/{T} timesteps hatched ({100*n_test_ts/T:.1f}%)")

    fig = plotting_neuronal_behavioural_plotly(
        x, b, b_names=b_labels, b_colors=b_colors, show_fig=False,
        split_mask=split_mask,
        trial_markers=trial_markers,
        epoch_markers=epoch_markers,
    )

    fig.write_html(str(output_path))
    print(f"[DONE] {run_dir.name}: HTML written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Regenerate neural_behavioural_overview.html with test-trial hatching')
    parser.add_argument('run_dirs', nargs='+', metavar='RUN_DIR',
                        help='Run directories to process. Can also pass a grid-search '
                             'directory to process all run_* subdirs.')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite even if the HTML already exists')
    args = parser.parse_args()

    dirs_to_process = []
    for d in args.run_dirs:
        p = Path(d)
        if not p.exists():
            print(f"[WARN] {d} does not exist, skipping")
            continue
        # If the path is a grid-search parent, expand to run_* children (dirs only)
        children = sorted(c for c in p.glob('run_*') if c.is_dir())
        if children:
            dirs_to_process.extend(children)
        else:
            dirs_to_process.append(p)

    print(f"Processing {len(dirs_to_process)} run director{'y' if len(dirs_to_process)==1 else 'ies'}")

    for run_dir in dirs_to_process:
        try:
            regenerate_run(run_dir, force=args.force)
        except Exception as exc:
            print(f"[ERROR] {run_dir.name}: {exc}")


if __name__ == '__main__':
    main()
