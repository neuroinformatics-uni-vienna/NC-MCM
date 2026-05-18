#!/usr/bin/env python3
"""
Regenerate common figures for all runs in a grid-search directory.

This script recreates:
 - latent_time_series_{train,validation}.html and .png
 - phase_space_dynamics_{split}_{view}.png (front/top/right)
 - phase_space_continuous_{split}_HGF_belief_{view}.png (if present)
 - training_loss.png
 - neural_behavioural_overview.html (by calling scripts/regenerate_neural_behavioural_html.py)

Usage:
    python scripts/regenerate_gridsearch_figures.py results/grid_search_20260510_222915

"""
import argparse
import json
import sys
import subprocess
from pathlib import Path
import numpy as np
import os

# Use non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ncmcm.visualisers.latent_space import LatentSpaceVisualiser
from ncmcm.data_loaders.bandit_task import BanditTaskNeuroPixelsDataset


def load_dataset_for_run(cfg: dict):
    kwargs = {}
    if cfg.get('b_type') == 'hybrid':
        kwargs['hgf_model'] = cfg.get('hgf_model')
        kwargs['hgf_column'] = cfg.get('hgf_column')

    dataset = BanditTaskNeuroPixelsDataset(
        data_path=cfg['data_path'],
        downsample_fs=int(cfg['downsample_fs']) if cfg.get('downsample_fs') is not None else None,
        downsample_method=cfg.get('downsample_method', 'count'),
        good_neurons_only=bool(cfg.get('good_neurons_only', True)),
        state_transitions=cfg.get('apply_hold_transitions') if cfg.get('apply_hold_transitions') != 'none' else None,
        normalize_method=cfg.get('normalize_method') if cfg.get('normalize_method') not in (None, 'None') else None,
        choosing_state_mode=cfg.get('choosing_state_mode', 'side'),
        gaussian_sigma_ms=float(cfg.get('gaussian_sigma_ms', 25.0)),
        b_mode=cfg.get('b_mode', 'full'),
        **kwargs,
    )
    return dataset


def plot_training_loss(loss_array, test_loss_array, output_dir: Path):
    is_hybrid = loss_array.shape[1] == 5
    n_plots = 5 if is_hybrid else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

    labels = [
        r"$\mathcal{L}_{\mathrm{Markov}}$",
        r"$\mathcal{L}_{\mathrm{Behaviour}}$",
        r"Total loss $\mathcal{L}$"
    ]
    if is_hybrid:
        labels += [
            r"$\mathcal{L}_{\mathrm{Discrete}}$ (CE component)",
            r"$\mathcal{L}_{\mathrm{Continuous}}$ (MSE component)"
        ]

    # Ensure axes iterable
    if n_plots == 1:
        axes = [axes]

    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.semilogy(loss_array[:, i], label='Train', linewidth=2)
        ax.semilogy(test_loss_array[:, i], label='Test', linewidth=2, linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / 'figures' / 'training_loss.png'
    os.makedirs(plot_path.parent, exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Training loss plot saved to {plot_path}")


def regenerate_run(run_dir: Path, force: bool = False, generate_gif: bool = False, generate_3d_html: bool = False):
    run_dir = Path(run_dir)
    config_path = run_dir / 'config.json'
    if not config_path.exists():
        print(f"[SKIP] {run_dir.name}: no config.json")
        return

    print(f"Processing {run_dir}")
    with open(config_path) as f:
        cfg = json.load(f)

    data_dir = run_dir / 'data'
    fig_dir = run_dir / 'figures'
    os.makedirs(fig_dir, exist_ok=True)

    # Load dataset (may use cache)
    try:
        dataset = load_dataset_for_run(cfg)
    except Exception as exc:
        print(f"  Warning: could not load dataset for {run_dir.name}: {exc}")
        dataset = None

    # For each split, regenerate latent + phase-space visuals
    for split in ('train', 'validation'):
        y_path = data_dir / f'latent_trajectories_{split}.npy'
        b_path = data_dir / f'behaviour_labels_{split}.npy'
        if not y_path.exists() or not b_path.exists():
            print(f"  Skipping {split}: latent/behaviour files missing")
            continue

        print(f"  Regenerating latent/phase visuals for {split}...")
        y = np.load(y_path)
        b = np.load(b_path)

        # Segment ids (trial ids) if present
        seg_path = data_dir / f'trial_ids_{split}.npy'
        segment_ids = np.load(seg_path) if seg_path.exists() else None

        colors = None
        b_labels = None
        try:
            if dataset is not None:
                colors = dataset.get_rgb_colors_for_visualizer()
                b_labels = dataset.b_labels
        except Exception:
            colors = None
            b_labels = None

        vis = LatentSpaceVisualiser(y, b, b_labels or {}, show_points=True, colors=colors, segment_ids=segment_ids)

        # Latent time series (HTML + PNG)
        html_path = fig_dir / f'latent_time_series_{split}.html'
        png_path = fig_dir / f'latent_time_series_{split}.png'
        try:
            vis.plot_latent_timeseries_plotly(show_fig=False, filename=str(html_path))
            print(f"    Wrote {html_path}")
        except Exception as exc:
            print(f"    Error writing HTML latent timeseries: {exc}")

        try:
            vis.plot_latent_timeseries(show_fig=False, filename=str(png_path))
            print(f"    Wrote {png_path}")
        except Exception as exc:
            print(f"    Error writing PNG latent timeseries: {exc}")

        # Phase space views
        phase_views = [((0, 0), 'front'), ((90, 0), 'top'), ((0, 90), 'right')]
        for (elev, azim), view_name in phase_views:
            outpath = fig_dir / f'phase_space_dynamics_{split}_{view_name}.png'
            try:
                fig, ax = vis.plot_phase_space(show_fig=False, axis_view=(elev, azim), filename=str(outpath))
                plt.close(fig)
                print(f"    Wrote {outpath}")
            except Exception as exc:
                print(f"    Error writing phase space ({view_name}): {exc}")

        # Continuous-variable phase space (HGF belief)
        # name patterns: hgf_belief_{split}.npy  or hgf_beliefs.npy
        hgf_candidates = [data_dir / f'hgf_belief_{split}.npy', data_dir / 'hgf_belief.npy', data_dir / 'hgf_beliefs.npy']
        hgf_path = next((p for p in hgf_candidates if p.exists()), None)
        if hgf_path is not None:
            try:
                c = np.load(hgf_path)
                safe_label = 'HGF_belief'
                for (elev, azim), view_name in phase_views:
                    outpath = fig_dir / f'phase_space_continuous_{split}_{safe_label}_{view_name}.png'
                    try:
                        fig, ax = vis.plot_phase_space_continuous(c=c, label='HGF belief', show_fig=False, axis_view=(elev, azim), filename=str(outpath))
                        plt.close(fig)
                        print(f"    Wrote {outpath}")
                    except Exception as exc:
                        print(f"    Error writing continuous phase space ({view_name}): {exc}")
            except Exception as exc:
                print(f"    Could not load HGF beliefs: {exc}")

    # Training loss
    loss_path = data_dir / 'loss_array.npy'
    test_loss_path = data_dir / 'test_loss_array.npy'
    if loss_path.exists() and test_loss_path.exists():
        try:
            loss_array = np.load(loss_path)
            test_loss_array = np.load(test_loss_path)
            plot_training_loss(loss_array, test_loss_array, run_dir)
        except Exception as exc:
            print(f"  Warning: could not plot training loss: {exc}")

    # Regenerate neural_behavioural_overview.html using existing helper script
    regen_script = Path(__file__).parent / 'regenerate_neural_behavioural_html.py'
    if regen_script.exists():
        try:
            cmd = [sys.executable, str(regen_script), str(run_dir)]
            if force:
                cmd += ['--force']
            print(f"  Calling: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"  Warning: regenerate_neural_behavioural_html failed for {run_dir}: {exc}")
    else:
        print(f"  regenerate_neural_behavioural_html.py not found; skipping neural_behavioural regeneration")


def main():
    parser = argparse.ArgumentParser(description='Regenerate figures for a grid-search directory')
    parser.add_argument('grid_dirs', nargs='+', help='Grid-search directory or individual run_* directories')
    parser.add_argument('--force', action='store_true', help='Overwrite existing files')
    args = parser.parse_args()

    dirs_to_process = []
    for d in args.grid_dirs:
        p = Path(d)
        if not p.exists():
            print(f"[WARN] {d} does not exist, skipping")
            continue
        children = sorted(c for c in p.glob('run_*') if c.is_dir())
        if children:
            dirs_to_process.extend(children)
        else:
            dirs_to_process.append(p)

    print(f"Processing {len(dirs_to_process)} run directories")
    for run_dir in dirs_to_process:
        try:
            regenerate_run(run_dir, force=args.force)
        except Exception as exc:
            print(f"[ERROR] {run_dir}: {exc}")


if __name__ == '__main__':
    main()
