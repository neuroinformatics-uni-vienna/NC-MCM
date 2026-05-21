"""
Manifest construction and serialisation.

The manifest captures all provenance information needed to reproduce or
audit an experiment: git state, hyperparameters, data splits, and paths
to every generated artifact.
"""

import datetime
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def get_git_info(repo_root: Path) -> dict:
    """Return git provenance information from *repo_root*."""
    def _run(*args):
        try:
            return subprocess.check_output(
                list(args), cwd=repo_root, stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            return 'unknown'

    commit_hash = _run('git', 'rev-parse', 'HEAD')
    short_hash  = _run('git', 'rev-parse', '--short', 'HEAD')
    branch      = _run('git', 'rev-parse', '--abbrev-ref', 'HEAD')
    status_raw  = _run('git', 'status', '--short')

    return {
        'commit_hash':    commit_hash,
        'short_hash':     short_hash,
        'branch':         branch,
        'status_summary': status_raw or '(clean)',
    }


# ---------------------------------------------------------------------------
# Data-split helpers
# ---------------------------------------------------------------------------

def _class_counts(data_dir: Path, split: str) -> dict:
    """Return {str(class_int): count} for one split, or {}."""
    lbl_path = data_dir / f'behaviour_labels_{split}.npy'
    if not lbl_path.exists():
        return {}
    try:
        labels = np.load(lbl_path).astype(int)
        return {str(k): int(v) for k, v in sorted(Counter(labels.tolist()).items())}
    except Exception:
        return {}


def _trial_info(data_dir: Path, split: str) -> dict:
    """Return trial-count summary for one split, or {}."""
    t_path = data_dir / f'trial_ids_{split}.npy'
    if not t_path.exists():
        return {}
    try:
        ids = np.load(t_path).astype(int)
        return {
            'n_windows': int(len(ids)),
            'n_trials':  int(len(set(ids.tolist()))),
            'trial_id_min': int(ids.min()),
            'trial_id_max': int(ids.max()),
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------

def build_manifest(
    run_dir: Path,
    config: dict,
    experiment_id: str,
    artifact_paths: dict,
    command: str = '',
    extra: dict | None = None,
) -> dict:
    """Build the complete manifest dictionary.

    Parameters
    ----------
    run_dir:
        The BunDLe-Net run directory (already trained).
    config:
        Contents of ``run_dir/config.json``.
    experiment_id:
        Human-readable ID string for this experiment archive.
    artifact_paths:
        Dict mapping artifact labels to absolute path strings.
    command:
        The command string used to invoke the pipeline.
    extra:
        Optional extra keys merged at the top level.
    """
    # Find repo root
    repo_root = Path(__file__).parents[2]

    git_info     = get_git_info(repo_root)
    run_summary  = _load_json_safe(run_dir / 'run_summary.json')

    data_path = config.get('data_path', '')
    session   = Path(data_path).name if data_path else 'unknown'
    data_dir  = run_dir / 'data'

    # Class counts and trial splits for each data split
    class_counts: dict = {}
    trial_splits: dict = {}
    for split in ('train', 'validation'):
        cc = _class_counts(data_dir, split)
        if cc:
            class_counts[split] = cc
        ti = _trial_info(data_dir, split)
        if ti:
            trial_splits[split] = ti

    manifest = {
        'experiment_id': experiment_id,
        'created_at':    datetime.datetime.now().isoformat(),
        'source_run_dir': str(run_dir),
        'command':       command,
        'git':           git_info,
        'session':       session,
        'data_path':     data_path,
        'training': {
            'start_timestamp':  run_summary.get('start_timestamp', ''),
            'completed_at':     run_summary.get('completed_at', ''),
            'execution_time':   run_summary.get('execution_time', ''),
            'status':           run_summary.get('status', 'unknown'),
            'final_losses':     run_summary.get('metrics', {}),
        },
        'model': {
            'b_mode':                  config.get('b_mode', ''),
            'choosing_state_mode':     config.get('choosing_state_mode', 'side'),
            'apply_hold_transitions':  config.get('apply_hold_transitions', 'none'),
            'b_type':                  config.get('b_type', ''),
            'alpha':                   config.get('alpha', None),
            'latent_dim':              config.get('latent_dim', ''),
            'context_policy':          config.get('context_policy', ''),
            'trial_random_state':      config.get('trial_random_state', None),
            'n_epochs':                config.get('n_epochs', ''),
            'window':                  config.get('window', ''),
            'normalize_method':        config.get('normalize_method', ''),
            'downsample_fs':           config.get('downsample_fs', ''),
            'b_labels':                config.get('b_labels', []),
            'state_mapping':           {str(i): name for i, name in
                                        enumerate(config.get('b_labels', []))},
            'n_classes':               config.get('n_classes', ''),
            'hgf_model':               config.get('hgf_model', None),
            'hgf_column':              config.get('hgf_column', None),
        },
        'data': {
            'class_counts': class_counts,
            'trial_splits': trial_splits,
        },
        'artifacts': artifact_paths,
    }

    if extra:
        manifest.update(extra)

    return manifest


def save_manifest(manifest: dict, experiment_dir: Path) -> Path:
    """Write manifest.json to *experiment_dir*."""
    out = Path(experiment_dir) / 'manifest.json'
    with open(out, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_json_safe(path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}
