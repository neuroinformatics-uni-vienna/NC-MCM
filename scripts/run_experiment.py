#!/usr/bin/env python3
"""
Experiment archive pipeline for BunDLe-Net bandit runs.

Creates a structured ``results/experiments/<id>/`` folder containing all
training artifacts, decoding outputs, time-resolved analysis, and geometry
plots for a given trained BunDLe-Net run.

Usage
-----
    # Full pipeline on an existing run:
    python scripts/run_experiment.py --run_dir <path_to_run_dir>

    # Dry-run — create folder + manifest only, no analysis:
    python scripts/run_experiment.py --run_dir <path> --dry_run

    # Skip behavior decoding (reuse existing results from run_dir):
    python scripts/run_experiment.py --run_dir <path> --skip_decoding

    # Copy existing decoding outputs instead of re-running:
    python scripts/run_experiment.py --run_dir <path> --reuse_decoding

Folder structure produced
------------------------
    results/experiments/<experiment_id>/
      manifest.json
      config.json
      run_summary.json
      status.json
      logs/
      model/           bundlenet_model.pt
      data/            *.npy (latent trajectories, labels, etc.)
      training/        training_curve.png, loss_array.npy, ...
      behavior_decoding/  decoding_summary.json, *.pdf, ...
      time_analysis/   accuracy_over_trial_time.png, ...
      geometry/        latent_space_projections.png, pca_summary.json, ...
      mve_trial_based/ status.json  (placeholder until MVE is integrated)
      reports/         experiment_report.md
"""

import argparse
import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Repo root and ncmcm import
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT  = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from ncmcm.experiment_archive import (
    generate_experiment_id,
    create_experiment_folder,
    build_manifest,
    save_manifest,
    generate_report,
)


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description='Create a structured experiment archive from a BunDLe-Net run.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        '--run_dir', required=True,
        help='Path to an existing BunDLe-Net run folder (must contain config.json).',
    )
    ap.add_argument(
        '--out_base', default=None,
        help='Base directory for experiments. Default: results/experiments/',
    )
    ap.add_argument(
        '--experiment_id', default=None,
        help='Override the auto-generated experiment ID.',
    )
    ap.add_argument(
        '--skip_decoding', action='store_true',
        help='Skip behavior decoding entirely.',
    )
    ap.add_argument(
        '--reuse_decoding', action='store_true',
        help='Copy existing decoding outputs from <run_dir>/data/decoding/ '
             'instead of re-running the decoder.',
    )
    ap.add_argument(
        '--skip_analysis', action='store_true',
        help='Skip time and geometry analysis.',
    )
    ap.add_argument(
        '--dry_run', action='store_true',
        help='Create folder structure and manifest only; do not run any analysis.',
    )
    return ap.parse_args()


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    args   = _parse_args()
    run_dir = Path(args.run_dir).resolve()

    if not run_dir.is_dir():
        print(f'ERROR: run_dir not found: {run_dir}', file=sys.stderr)
        sys.exit(1)

    config_path = run_dir / 'config.json'
    if not config_path.exists():
        print(f'ERROR: config.json not found in {run_dir}', file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    # ── Determine output base ──────────────────────────────────────────────
    out_base = Path(args.out_base) if args.out_base else REPO_ROOT / 'results' / 'experiments'

    # ── Generate experiment ID ─────────────────────────────────────────────
    exp_id = args.experiment_id or generate_experiment_id(config, run_dir.name)
    print(f'Experiment ID : {exp_id}')

    # ── Create folder structure ────────────────────────────────────────────
    paths   = create_experiment_folder(out_base, exp_id)
    exp_dir = paths['root']
    print(f'Experiment dir: {exp_dir}')

    # ── Write config + run_summary ─────────────────────────────────────────
    _copy_config(run_dir, exp_dir, config)

    # ── Write initial status ───────────────────────────────────────────────
    started_at = datetime.datetime.now().isoformat()
    _write_status(exp_dir, 'running' if not args.dry_run else 'dry_run',
                  started_at=started_at)

    if args.dry_run:
        _write_manifest(run_dir, config, exp_id, exp_dir, {}, args)
        print('[dry_run] Folder structure created. Manifest written. Exiting.')
        return

    artifact_paths: dict[str, str] = {
        'config':      str(exp_dir / 'config.json'),
        'run_summary': str(exp_dir / 'run_summary.json'),
    }

    # ── Model checkpoint ───────────────────────────────────────────────────
    model_src = run_dir / 'model' / 'bundlenet_model.pt'
    if model_src.exists():
        dst = paths['model'] / 'bundlenet_model.pt'
        shutil.copy2(model_src, dst)
        artifact_paths['model_checkpoint'] = str(dst)
        print('Copied model checkpoint.')
    else:
        print('WARNING: model checkpoint not found.')

    # ── Data arrays ────────────────────────────────────────────────────────
    _copy_glob(run_dir / 'data', paths['data'], patterns=('*.npy', '*.json'))
    print('Copied data arrays.')

    # ── Training artifacts ─────────────────────────────────────────────────
    _copy_glob(run_dir / 'figures', paths['training'], patterns=('*.png', '*.pdf'))
    for fname in ('loss_array.npy', 'test_loss_array.npy',
                  'disc_loss_array.npy', 'cont_loss_array.npy'):
        src = run_dir / 'data' / fname
        if src.exists():
            shutil.copy2(src, paths['training'] / fname)
    _generate_training_curve(paths['training'])
    print('Training artifacts copied/generated.')

    # ── Behavior decoding ──────────────────────────────────────────────────
    dec_dir = paths['behavior_decoding']
    if not args.skip_decoding:
        if args.reuse_decoding:
            src_dec = run_dir / 'data' / 'decoding'
            if src_dec.exists():
                _copy_glob(src_dec, dec_dir)
                print(f'Reused existing decoding outputs from {src_dec}')
            else:
                print('WARNING: --reuse_decoding set but no existing decoding dir found; '
                      'running decoder.')
                _run_decoding(run_dir, dec_dir)
        else:
            print('\n--- Running behavior decoding ---')
            ret = _run_decoding(run_dir, dec_dir)
            if ret != 0:
                print(f'WARNING: decoding exited with code {ret}')
    if (dec_dir / 'decoding_summary.json').exists():
        artifact_paths['decoding_summary'] = str(dec_dir / 'decoding_summary.json')

    # ── Geometry + time analysis ───────────────────────────────────────────
    if not args.skip_analysis:
        print('\n--- Running geometry analysis ---')
        _run_geometry_analysis(run_dir, paths['geometry'], config)

        print('\n--- Running time analysis ---')
        _run_time_analysis(run_dir, paths['time_analysis'], config)

    # ── MVE placeholder ────────────────────────────────────────────────────
    _write_json(paths['mve_trial_based'] / 'status.json', {
        'status': 'not_run',
        'note':   'Trial-based MVE not yet integrated into the experiment pipeline.',
        'created_at': datetime.datetime.now().isoformat(),
    })

    # ── Manifest ───────────────────────────────────────────────────────────
    command = f'python scripts/run_experiment.py --run_dir {run_dir}'
    manifest = _write_manifest(run_dir, config, exp_id, exp_dir, artifact_paths, args,
                               command=command)

    # ── Report ─────────────────────────────────────────────────────────────
    report_path = generate_report(manifest, exp_dir)
    print(f'Report : {report_path}')

    # ── Final status ───────────────────────────────────────────────────────
    completed_at = datetime.datetime.now().isoformat()
    _write_status(exp_dir, 'completed',
                  started_at=started_at, completed_at=completed_at)

    print(f'\n{"="*70}')
    print(f'Experiment archive complete!')
    print(f'  {exp_dir}')
    print(f'{"="*70}')


# ===========================================================================
# Helpers
# ===========================================================================

def _copy_config(run_dir: Path, exp_dir: Path, config: dict) -> None:
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    rs = run_dir / 'run_summary.json'
    if rs.exists():
        shutil.copy2(rs, exp_dir / 'run_summary.json')


def _write_status(
    exp_dir: Path,
    status: str,
    started_at: str = '',
    completed_at: str = '',
) -> None:
    _write_json(exp_dir / 'status.json', {
        'status':       status,
        'started_at':   started_at,
        'completed_at': completed_at,
        'updated_at':   datetime.datetime.now().isoformat(),
    })


def _write_manifest(
    run_dir: Path,
    config: dict,
    exp_id: str,
    exp_dir: Path,
    artifact_paths: dict,
    args: argparse.Namespace,
    command: str = '',
) -> dict:
    manifest = build_manifest(
        run_dir, config, exp_id, artifact_paths, command=command,
    )
    save_manifest(manifest, exp_dir)
    print('manifest.json written.')
    return manifest


def _copy_glob(src: Path, dst: Path, patterns: tuple[str, ...] = ('*',)) -> None:
    """Copy files matching *patterns* from src to dst."""
    dst.mkdir(parents=True, exist_ok=True)
    if not src.is_dir():
        return
    for pat in patterns:
        for f in src.glob(pat):
            if f.is_file():
                shutil.copy2(f, dst / f.name)


def _write_json(path: Path, data: dict) -> None:
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def _run_decoding(run_dir: Path, out_dir: Path) -> int:
    """Invoke bandit_behaviour_decoding.py as a subprocess."""
    script = SCRIPT_DIR / 'bandit_behaviour_decoding.py'
    result = subprocess.run(
        [sys.executable, str(script), str(run_dir), '--out', str(out_dir)],
        cwd=REPO_ROOT,
    )
    return result.returncode


def _generate_training_curve(training_dir: Path) -> None:
    """Generate training_curve.png from loss_array.npy if present."""
    loss_path = training_dir / 'loss_array.npy'
    if not loss_path.exists():
        return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        loss      = np.load(loss_path)   # (n_epochs, 5)
        test_path = training_dir / 'test_loss_array.npy'
        test_loss = np.load(test_path) if test_path.exists() else None

        col_names = ['total', 'markovian', 'behaviour', 'discrete', 'continuous']
        epochs    = np.arange(1, len(loss) + 1)

        fig, ax = plt.subplots(figsize=(10, 4))
        for i, name in enumerate(col_names[:loss.shape[1]]):
            col = loss[:, i]
            if np.any(col != 0):
                ax.plot(epochs, col, lw=1.5, label=f'train/{name}')
        if test_loss is not None:
            for i, name in enumerate(col_names[:test_loss.shape[1]]):
                col = test_loss[:, i]
                if np.any(col != 0):
                    ax.plot(epochs, col, lw=1.0, ls='--',
                            label=f'val/{name}', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (log scale)')
        ax.set_title('Training loss curves')
        ax.set_yscale('log')
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(training_dir / 'training_curve.png', dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f'WARNING: could not generate training curve: {e}')


def _run_geometry_analysis(run_dir: Path, out_dir: Path, config: dict) -> None:
    """Generate PCA variance and latent-space scatter plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    data_dir       = run_dir / 'data'
    lat_train_path = data_dir / 'latent_trajectories_train.npy'
    lat_val_path   = data_dir / 'latent_trajectories_validation.npy'
    lab_val_path   = data_dir / 'behaviour_labels_validation.npy'

    if not lat_train_path.exists():
        print('WARNING: latent_trajectories_train.npy not found; skipping geometry.')
        return

    lat_train  = np.load(lat_train_path)
    lat_val    = np.load(lat_val_path) if lat_val_path.exists() else lat_train
    lab_val    = np.load(lab_val_path).astype(int) if lab_val_path.exists() else None
    b_labels   = config.get('b_labels', [])
    latent_dim = int(config.get('latent_dim', lat_train.shape[1]))

    n_comps = min(3, latent_dim, lat_train.shape[1])
    pca = PCA(n_components=n_comps).fit(lat_train)
    ev  = pca.explained_variance_ratio_

    # PCA explained variance bar chart
    fig, ax = plt.subplots(figsize=(6, 3))
    bar_colors = ['#2196F3', '#FF5722', '#4CAF50'][:n_comps]
    ax.bar(range(n_comps), ev * 100, color=bar_colors)
    ax.set_xticks(range(n_comps))
    ax.set_xticklabels([f'PC{i+1}' for i in range(n_comps)])
    ax.set_ylabel('Explained variance (%)')
    ax.set_title(f'PCA explained variance  (latent_dim={latent_dim})')
    for i, e in enumerate(ev):
        ax.text(i, e * 100 + 0.3, f'{e*100:.1f}%', ha='center', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / 'pca_explained_variance.png', dpi=150)
    plt.close(fig)

    # 2D projection scatter plots
    if latent_dim >= 2 and lab_val is not None:
        colors    = plt.cm.tab10.colors
        dim_pairs = [(0, 1), (0, 2), (1, 2)] if latent_dim >= 3 else [(0, 1)]
        valid_pairs = [(xi, yi) for xi, yi in dim_pairs if xi < lat_val.shape[1] and yi < lat_val.shape[1]]

        fig, axes = plt.subplots(1, len(valid_pairs), figsize=(5 * len(valid_pairs), 4))
        if len(valid_pairs) == 1:
            axes = [axes]

        for ax, (xi, yi) in zip(axes, valid_pairs):
            for lab in sorted(np.unique(lab_val).tolist()):
                m    = lab_val == lab
                name = b_labels[lab] if b_labels and lab < len(b_labels) else str(lab)
                ax.scatter(
                    lat_val[m, xi], lat_val[m, yi],
                    c=[colors[lab % len(colors)]], alpha=0.10, s=2,
                    rasterized=True, label=name,
                )
            ax.set_xlabel(f'y{xi}')
            ax.set_ylabel(f'y{yi}')
            ax.legend(fontsize=7, markerscale=5)
        fig.suptitle('Latent space (validation set)', fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / 'latent_space_projections.png', dpi=150,
                    bbox_inches='tight')
        plt.close(fig)

    # Save PCA JSON summary
    _write_json(out_dir / 'pca_summary.json', {
        'explained_variance_ratio': ev.tolist(),
        'latent_dim':               latent_dim,
        'n_components_fit':         n_comps,
        'n_train_samples':          int(len(lat_train)),
        'n_val_samples':            int(len(lat_val)),
    })


def _run_time_analysis(run_dir: Path, out_dir: Path, config: dict) -> None:
    """Generate time-resolved accuracy plots over normalised within-trial time."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy import stats as scipy_stats
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    data_dir       = run_dir / 'data'
    lat_train_path = data_dir / 'latent_trajectories_train.npy'
    lat_val_path   = data_dir / 'latent_trajectories_validation.npy'
    lab_train_path = data_dir / 'behaviour_labels_train.npy'
    lab_val_path   = data_dir / 'behaviour_labels_validation.npy'
    tid_val_path   = data_dir / 'trial_ids_validation.npy'

    if not (lat_val_path.exists() and lab_val_path.exists()):
        print('WARNING: validation latent/label files not found; skipping time analysis.')
        return

    lat_train = np.load(lat_train_path)
    lab_train = np.load(lab_train_path).astype(int)
    lat_val   = np.load(lat_val_path)
    lab_val   = np.load(lab_val_path).astype(int)
    b_labels  = config.get('b_labels', [])

    # Train a linear classifier on train split
    clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
    clf.fit(lat_train, lab_train)
    preds   = clf.predict(lat_val)
    correct = (preds == lab_val).astype(int)

    val_acc  = float(accuracy_score(lab_val, preds))
    val_bacc = float(balanced_accuracy_score(lab_val, preds))
    majority = float(max(np.bincount(lab_val) / len(lab_val)))

    # Time-resolved curves (requires trial_ids)
    has_trials = tid_val_path.exists()
    if has_trials:
        import pandas as pd

        tid_val = np.load(tid_val_path).astype(int)
        df = pd.DataFrame({
            'trial_id': tid_val,
            'correct':  correct,
            'label':    lab_val,
        })
        df['within_trial_idx'] = df.groupby('trial_id').cumcount()
        nwm = df.groupby('trial_id')['within_trial_idx'].transform('max')
        df['normalized_time'] = df['within_trial_idx'] / nwm.clip(lower=1)

        N_BINS    = 20
        bin_edges = np.linspace(0, 1, N_BINS + 1)
        bin_ctrs  = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_idx   = pd.cut(df['normalized_time'], bins=bin_edges,
                           labels=False, include_lowest=True)

        def _bin_stat(series, idx):
            acc = np.array([
                series[idx == i].mean() if (idx == i).any() else np.nan
                for i in range(N_BINS)
            ])
            sem = np.array([
                scipy_stats.sem(series[idx == i]) if (idx == i).sum() > 1 else np.nan
                for i in range(N_BINS)
            ])
            return acc, sem

        bin_acc, bin_sem = _bin_stat(df['correct'], bin_idx)

        # ── Overall time-resolved figure ──────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.fill_between(bin_ctrs, bin_acc - bin_sem, bin_acc + bin_sem,
                        alpha=0.18, color='steelblue')
        ax.plot(bin_ctrs, bin_acc, '-o', color='steelblue', ms=4, lw=2,
                label='accuracy')
        ax.axhline(majority, color='grey', ls='--', lw=1.5,
                   label=f'Majority baseline ({majority:.3f})')
        ax.axhline(0.5, color='lightgray', ls=':', lw=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Normalised trial time (0=start, 1=choice)')
        ax.set_ylabel('Accuracy')
        ax.set_title(
            f'Time-resolved accuracy\n'
            f'acc={val_acc:.4f}  bacc={val_bacc:.4f}  '
            f'n_trials={df["trial_id"].nunique()}'
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / 'accuracy_over_trial_time.png', dpi=150)
        plt.close(fig)

        # ── Per-class time-resolved figure ────────────────────────────────
        colors = plt.cm.tab10.colors
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.axhline(majority, color='grey', ls='--', lw=1.5,
                   label=f'Majority ({majority:.3f})')
        for lab in sorted(np.unique(lab_val).tolist()):
            sub_mask = df['label'] == lab
            ba, _    = _bin_stat(df.loc[sub_mask, 'correct'], bin_idx[sub_mask])
            name = b_labels[lab] if b_labels and lab < len(b_labels) else str(lab)
            ax.plot(bin_ctrs, ba, '-o', color=colors[lab % len(colors)],
                    ms=3, lw=1.5, label=name)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Normalised trial time')
        ax.set_ylabel('Accuracy')
        ax.set_title('Time-resolved accuracy by class')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / 'accuracy_over_time_per_class.png', dpi=150)
        plt.close(fig)

        # ── Save CSV ──────────────────────────────────────────────────────
        df[['trial_id', 'within_trial_idx', 'normalized_time',
            'label', 'correct']].to_csv(
            out_dir / 'validation_time_resolved.csv', index=False
        )

    # Always save summary JSON
    _write_json(out_dir / 'time_analysis_summary.json', {
        'val_acc':          val_acc,
        'val_balanced_acc': val_bacc,
        'majority_baseline': majority,
        'has_trial_ids':    has_trials,
    })


# ===========================================================================

if __name__ == '__main__':
    main()
