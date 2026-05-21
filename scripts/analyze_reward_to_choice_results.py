#!/usr/bin/env python3
"""
Post-run analysis for reward_to_choice clean training runs.

Answers 6 key research questions:
  Q1. Does choice decodability remain high under corrected (reward→choice) interval?
  Q2. Is early decodability still mostly stay trials (easier to read)?
  Q3. Does switch-trial pre-choice decodability rise above baseline at early timesteps?
  Q4. Does hybrid HGF supervision leave binary choice accuracy nearly unchanged vs discrete?
  Q5. Does hybrid produce richer geometry than discrete (PCA variance)?
  Q6. Is the continuous HGF target decodable from the hybrid latent space?

Usage:
    python scripts/analyze_reward_to_choice_results.py \\
        --hybrid  results/grid_search_..._hybrid_alpha_050/run_...  \\
        --discrete results/grid_search_..._discrete_only/run_...   \\
        [--out results/analysis/reward_to_choice_analysis]

Requires:
  - Both run folders to contain data/latent_trajectories_{train,validation}.npy
  - Both run folders to contain data/behaviour_labels_{train,validation}.npy
  - Both run folders to contain data/trial_ids_{train,validation}.npy
  - Hybrid run to contain data/hgf_belief_train.npy  (and _validation.npy)
  - Both run folders to contain config.json
  - (Optional) data/decoding/decoding_summary.json from bandit_behaviour_decoding.py
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# ── Tiny linear probe ─────────────────────────────────────────────────────────

def fit_linear_probe(X_tr, y_tr, X_te, y_te, max_iter=500):
    """Fit a simple L2-regularised logistic regression probe."""
    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_tr)
    y_te_enc = le.transform(y_te)
    clf = LogisticRegression(max_iter=max_iter, C=1.0, solver='lbfgs',
                              multi_class='auto', n_jobs=1)
    clf.fit(X_tr, y_tr_enc)
    pred = clf.predict(X_te)
    acc  = accuracy_score(y_te_enc, pred)
    bacc = balanced_accuracy_score(y_te_enc, pred)
    return acc, bacc, pred, le


# ── Load a run folder ─────────────────────────────────────────────────────────

def load_run(run_dir: Path, split_train='train', split_val='validation'):
    data = data_dir = run_dir / 'data'
    def _npy(stem):
        p = data_dir / f'{stem}.npy'
        if not p.exists():
            return None
        return np.load(p)

    Y_tr = _npy(f'latent_trajectories_{split_train}')
    Y_va = _npy(f'latent_trajectories_{split_val}')
    b_tr = _npy(f'behaviour_labels_{split_train}')
    b_va = _npy(f'behaviour_labels_{split_val}')
    tid_tr = _npy(f'trial_ids_{split_train}')
    tid_va = _npy(f'trial_ids_{split_val}')
    hgf_tr = _npy('hgf_belief_train')
    hgf_va = _npy('hgf_belief_validation')
    trial_start = _npy('trial_start_indices')    # (n_trials,) timepoints in full recording

    with open(run_dir / 'config.json') as f:
        config = json.load(f)

    # Decoding summary (optional, written by bandit_behaviour_decoding.py)
    dec_path = data_dir / 'decoding' / 'decoding_summary.json'
    decoding = None
    if dec_path.exists():
        with open(dec_path) as f:
            decoding = json.load(f)

    return dict(
        Y_tr=Y_tr, Y_va=Y_va,
        b_tr=b_tr, b_va=b_va,
        tid_tr=tid_tr, tid_va=tid_va,
        hgf_tr=hgf_tr, hgf_va=hgf_va,
        trial_start=trial_start,
        config=config,
        decoding=decoding,
    )


# ── Stay/switch labels ─────────────────────────────────────────────────────────

def compute_stay_switch(b_tr, b_va, tid_tr, tid_va):
    """
    Compute stay/switch label for each window in the validation set.

    A trial is "stay" if its choice matches the previous trial's choice;
    "switch" otherwise.  The very first trial (sorted by id) has no
    predecessor and is excluded from the stay/switch map.

    Returns:
        stay_switch_val : np.ndarray of shape (n_val_windows,) with values
                          0 = stay, 1 = switch, -1 = first trial / unknown
    """
    # Per-trial choice: combine train+val and take the most common label per trial
    all_b   = np.concatenate([b_tr, b_va])
    all_tid = np.concatenate([tid_tr, tid_va])
    unique_trials = np.unique(all_tid)

    # All windows in a trial have the same label (reward_to_choice mode)
    trial_choice = {}
    for t in unique_trials:
        mask = all_tid == t
        choices, counts = np.unique(all_b[mask], return_counts=True)
        trial_choice[t] = int(choices[counts.argmax()])

    # Compute stay/switch for sorted trial ids
    sorted_trials = sorted(unique_trials)
    stay_switch = {}   # trial_id → 0 (stay) / 1 (switch)
    for i in range(1, len(sorted_trials)):
        curr = sorted_trials[i]
        prev = sorted_trials[i - 1]
        stay_switch[curr] = 0 if trial_choice[curr] == trial_choice[prev] else 1

    # Map to validation windows
    ss_val = np.full(len(b_va), -1, dtype=np.int8)
    for i, t in enumerate(tid_va):
        ss_val[i] = stay_switch.get(int(t), -1)

    return ss_val, trial_choice, stay_switch


# ── Temporal position within trial ───────────────────────────────────────────

def compute_trial_temporal_positions(trial_ids):
    """
    For each window, compute its relative position (0..1) within its trial.

    Returns:
        rel_pos : np.ndarray shape (N,) in [0, 1]
        abs_pos : np.ndarray shape (N,) absolute index within trial
    """
    rel_pos = np.zeros(len(trial_ids), dtype=np.float32)
    abs_pos = np.zeros(len(trial_ids), dtype=np.int32)

    unique_trials = np.unique(trial_ids)
    for t in unique_trials:
        mask = np.where(trial_ids == t)[0]
        n = len(mask)
        abs_pos[mask] = np.arange(n)
        if n > 1:
            rel_pos[mask] = np.arange(n) / (n - 1)
        else:
            rel_pos[mask] = 0.5

    return rel_pos, abs_pos


# ── Q5: PCA variance explained ────────────────────────────────────────────────

def pca_variance(Y_tr, Y_va, n_components=None):
    Y_all = np.concatenate([Y_tr, Y_va], axis=0)
    if n_components is None:
        n_components = Y_all.shape[1]
    pca = PCA(n_components=n_components)
    pca.fit(Y_all)
    return pca.explained_variance_ratio_


# ── Temporal early/mid/late analysis ─────────────────────────────────────────

def temporal_decodability(Y_tr, b_tr, Y_va, b_va, tid_va,
                           stay_switch_va, n_thirds=3):
    """
    Train a probe on all training windows.
    Evaluate on early / mid / late thirds of validation trials.
    Also split by stay vs switch.

    Returns dict with keys:
      'all': (acc, bacc)
      'early': (acc, bacc)
      'mid': (acc, bacc)    [only if n_thirds >= 3]
      'late': (acc, bacc)
      'stay_all': (acc, bacc)
      'switch_all': (acc, bacc)
      'stay_early': (acc, bacc)
      'switch_early': (acc, bacc)
      'stay_late': (acc, bacc)
      'switch_late': (acc, bacc)
    """
    # Fit probe on all training windows
    le = LabelEncoder()
    y_tr_enc = le.fit_transform(b_tr)
    clf = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs',
                              multi_class='auto', n_jobs=1)
    clf.fit(Y_tr, y_tr_enc)

    rel_pos, _ = compute_trial_temporal_positions(tid_va)

    results = {}

    # Threshold splits
    early_thr  = 1.0 / n_thirds
    late_thr   = (n_thirds - 1) / n_thirds

    all_idx   = np.arange(len(Y_va))
    early_idx = np.where(rel_pos <= early_thr)[0]
    mid_idx   = np.where((rel_pos > early_thr) & (rel_pos <= late_thr))[0]
    late_idx  = np.where(rel_pos > late_thr)[0]

    stay_idx   = np.where(stay_switch_va == 0)[0]
    switch_idx = np.where(stay_switch_va == 1)[0]

    def _eval(idx_set, label):
        if len(idx_set) < 5:
            results[label] = (float('nan'), float('nan'))
            return
        X   = Y_va[idx_set]
        y_t = le.transform(b_va[idx_set])
        pred = clf.predict(X)
        acc  = accuracy_score(y_t, pred)
        bacc = balanced_accuracy_score(y_t, pred)
        results[label] = (acc, bacc)

    _eval(all_idx,   'all')
    _eval(early_idx, 'early')
    _eval(mid_idx,   'mid')
    _eval(late_idx,  'late')
    _eval(stay_idx,          'stay_all')
    _eval(switch_idx,        'switch_all')
    _eval(np.intersect1d(stay_idx,   early_idx), 'stay_early')
    _eval(np.intersect1d(switch_idx, early_idx), 'switch_early')
    _eval(np.intersect1d(stay_idx,   late_idx),  'stay_late')
    _eval(np.intersect1d(switch_idx, late_idx),  'switch_late')

    counts = {
        'n_stay':   len(stay_idx),
        'n_switch': len(switch_idx),
        'n_early':  len(early_idx),
        'n_late':   len(late_idx),
        'n_stay_early':   len(np.intersect1d(stay_idx,   early_idx)),
        'n_switch_early': len(np.intersect1d(switch_idx, early_idx)),
        'n_stay_late':    len(np.intersect1d(stay_idx,   late_idx)),
        'n_switch_late':  len(np.intersect1d(switch_idx, late_idx)),
    }

    results['_counts'] = counts
    return results


# ── HGF R² probe (sklearn, lighter than torch) ───────────────────────────────

def hgf_linear_r2(Y_tr, hgf_tr, Y_va, hgf_va, n_repeats=5):
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    r2_vals = []
    for _ in range(n_repeats):
        ridge = Ridge(alpha=1.0, fit_intercept=True)
        ridge.fit(Y_tr, hgf_tr.ravel())
        pred = ridge.predict(Y_va)
        r2_vals.append(r2_score(hgf_va.ravel(), pred))
    return float(np.mean(r2_vals)), float(np.std(r2_vals))


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_temporal_decodability(results_hybrid, results_disc, out_path):
    """Figure comparing temporal decodability (all/early/late × stay/switch × hybrid/discrete)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle('Temporal Choice Decodability — reward_to_choice', fontsize=14, fontweight='bold')

    for ax, (res, title) in zip(axes, [(results_hybrid, 'Hybrid'), (results_disc, 'Discrete-only')]):
        keys    = ['early', 'mid', 'late']
        keys_ss = [('stay_early', 'switch_early'), ('stay_mid', 'switch_mid'), ('stay_late', 'switch_late')]
        positions = [0, 1, 2]
        width = 0.25

        acc_all  = [res.get(k, (float('nan'), float('nan')))[1] for k in keys]
        acc_stay = [res.get(f'stay_{k}', (float('nan'), float('nan')))[1] for k in keys]
        acc_sw   = [res.get(f'switch_{k}', (float('nan'), float('nan')))[1] for k in keys]

        x = np.array(positions)
        b1 = ax.bar(x - width, acc_all,  width, color='steelblue', alpha=0.8, label='All trials')
        b2 = ax.bar(x,         acc_stay, width, color='seagreen',  alpha=0.8, label='Stay')
        b3 = ax.bar(x + width, acc_sw,   width, color='tomato',    alpha=0.8, label='Switch')

        ax.axhline(0.5, color='k', linestyle='--', linewidth=0.8, alpha=0.5, label='Chance (balanced)')
        ax.set_xticks(x); ax.set_xticklabels(['Early\n(1st third)', 'Mid\n(middle)', 'Late\n(last third)'])
        ax.set_ylabel('Balanced Accuracy', fontsize=11)
        ax.set_ylim([0, 1.05])
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_stay_switch_all(results_hybrid, results_disc, out_path):
    """Bar plot: stay vs switch, all timesteps, hybrid vs discrete."""
    fig, ax = plt.subplots(figsize=(8, 5))
    cats   = ['All', 'Stay', 'Switch']
    keys_h = ['all', 'stay_all', 'switch_all']
    keys_d = ['all', 'stay_all', 'switch_all']

    x   = np.arange(len(cats))
    w   = 0.35
    bac_h = [results_hybrid.get(k, (float('nan'), float('nan')))[1] for k in keys_h]
    bac_d = [results_disc.get(k, (float('nan'), float('nan')))[1] for k in keys_d]

    ax.bar(x - w/2, bac_h, w, color='#4A90D9', alpha=0.85, label='Hybrid')
    ax.bar(x + w/2, bac_d, w, color='#E88040', alpha=0.85, label='Discrete-only')
    ax.axhline(0.5, color='k', linestyle='--', linewidth=0.8, alpha=0.5, label='Chance (balanced)')
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=12)
    ax.set_ylabel('Balanced Accuracy (validation)', fontsize=11)
    ax.set_title('Stay / Switch Choice Decodability — All timesteps', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.set_ylim([0, 1.05]); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_pca_variance(evr_hybrid, evr_disc, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    dims = np.arange(1, len(evr_hybrid) + 1)
    ax.bar(dims - 0.2, evr_hybrid * 100, 0.35, color='#4A90D9', alpha=0.85, label='Hybrid')
    ax.bar(dims + 0.2, evr_disc   * 100, 0.35, color='#E88040', alpha=0.85, label='Discrete-only')
    ax.set_xlabel('Latent dimension', fontsize=11)
    ax.set_ylabel('Explained variance (%)', fontsize=11)
    ax.set_title('PCA Explained Variance of Latent Space', fontsize=13, fontweight='bold')
    ax.set_xticks(dims); ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_hgf_scatter(Y_va, hgf_va, out_path):
    """Predicted vs true HGF belief (Ridge regression, single fit for scatter)."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    ridge = Ridge(alpha=1.0)
    Y_all = Y_va  # we only have validation for plotting; use all for quick scatter
    ridge.fit(Y_all, hgf_va.ravel())
    pred = ridge.predict(Y_all)
    r2 = r2_score(hgf_va.ravel(), pred)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(hgf_va, pred, alpha=0.2, s=8, color='steelblue')
    lo, hi = min(hgf_va.min(), pred.min()), max(hgf_va.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='Identity')
    ax.set_xlabel('True HGF belief (x₁)', fontsize=11)
    ax.set_ylabel('Predicted HGF belief', fontsize=11)
    ax.set_title('Ridge Probe: HGF Belief from Hybrid Latent', fontsize=12, fontweight='bold')
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, va='top', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_confusion(Y_tr, b_tr, Y_va, b_va, label_names, title, out_path):
    acc, bacc, pred, le = fit_linear_probe(Y_tr, b_tr, Y_va, b_va)
    labels = le.classes_
    names  = [label_names.get(int(l), str(l)) for l in labels]
    cm = confusion_matrix(le.transform(b_va), pred, labels=np.arange(len(labels)))
    cm_pct = cm / (cm.sum(axis=1, keepdims=True) + 1e-9) * 100

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax,
                xticklabels=names, yticklabels=names,
                cbar_kws={'label': '%'}, vmin=0, vmax=100)
    ax.set_title(f'{title}\nAcc={acc:.3f}  BalancedAcc={bacc:.3f}', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45); ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")
    return acc, bacc


# ── Markdown report ───────────────────────────────────────────────────────────

CHOICE_NAMES = {0: 'Left', 1: 'Right'}


def write_report(out_dir, hybrid_dir, disc_dir, run_hybrid, run_disc,
                 temp_hybrid, temp_disc, evr_hybrid, evr_disc,
                 acc_hybrid, bacc_hybrid, acc_disc, bacc_disc,
                 hgf_r2_hybrid, hgf_r2_hybrid_std,
                 ss_counts_hybrid, ss_counts_disc):
    from datetime import datetime
    ts = datetime.now().strftime('%Y-%m-%d %H:%M')

    def _acc(k, r):
        v = r.get(k)
        if v is None: return 'N/A'
        return f'{v[0]:.3f} / {v[1]:.3f} (acc/bacc)'

    report = f"""# Reward-to-Choice Clean Run Analysis
Generated: {ts}

## Run Folders
- **Hybrid** (α=0.5):  `{hybrid_dir}`
- **Discrete-only**:   `{disc_dir}`

## Commit
{_git_head()}

---

## Q1 — Choice Decodability (corrected interval)

Linear probe (LogisticRegression) trained on train windows, evaluated on validation windows.

| Model | Acc | Balanced Acc |
|-------|-----|-------------|
| Hybrid | {acc_hybrid:.3f} | {bacc_hybrid:.3f} |
| Discrete-only | {acc_disc:.3f} | {bacc_disc:.3f} |
| Chance (balanced) | 0.500 | 0.500 |

---

## Q2 — Stay / Switch Decodability (all timesteps)

| Model | All (bacc) | Stay (bacc) | Switch (bacc) |
|-------|------------|-------------|---------------|
| Hybrid | {temp_hybrid.get('all',(float('nan'), float('nan')))[1]:.3f} | {temp_hybrid.get('stay_all',(float('nan'), float('nan')))[1]:.3f} | {temp_hybrid.get('switch_all',(float('nan'), float('nan')))[1]:.3f} |
| Discrete | {temp_disc.get('all',(float('nan'), float('nan')))[1]:.3f} | {temp_disc.get('stay_all',(float('nan'), float('nan')))[1]:.3f} | {temp_disc.get('switch_all',(float('nan'), float('nan')))[1]:.3f} |

Window counts (hybrid validation):  
Stay={ss_counts_hybrid['n_stay']}, Switch={ss_counts_hybrid['n_switch']}

---

## Q3 — Pre-choice Decodability: Temporal Breakdown

Balanced accuracy at different stages of the trial (1/3 thirds).

### Hybrid
| Stage | All (bacc) | Stay (bacc) | Switch (bacc) |
|-------|------------|-------------|---------------|
| Early | {temp_hybrid.get('early',(float('nan'), float('nan')))[1]:.3f} | {temp_hybrid.get('stay_early',(float('nan'), float('nan')))[1]:.3f} | {temp_hybrid.get('switch_early',(float('nan'), float('nan')))[1]:.3f} |
| Mid   | {temp_hybrid.get('mid',  (float('nan'), float('nan')))[1]:.3f} | {temp_hybrid.get('stay_mid',  (float('nan'), float('nan')))[1]:.3f} | {temp_hybrid.get('switch_mid',  (float('nan'), float('nan')))[1]:.3f} |
| Late  | {temp_hybrid.get('late', (float('nan'), float('nan')))[1]:.3f} | {temp_hybrid.get('stay_late', (float('nan'), float('nan')))[1]:.3f} | {temp_hybrid.get('switch_late', (float('nan'), float('nan')))[1]:.3f} |

### Discrete-only
| Stage | All (bacc) | Stay (bacc) | Switch (bacc) |
|-------|------------|-------------|---------------|
| Early | {temp_disc.get('early',(float('nan'), float('nan')))[1]:.3f} | {temp_disc.get('stay_early',(float('nan'), float('nan')))[1]:.3f} | {temp_disc.get('switch_early',(float('nan'), float('nan')))[1]:.3f} |
| Mid   | {temp_disc.get('mid',  (float('nan'), float('nan')))[1]:.3f} | {temp_disc.get('stay_mid',  (float('nan'), float('nan')))[1]:.3f} | {temp_disc.get('switch_mid',  (float('nan'), float('nan')))[1]:.3f} |
| Late  | {temp_disc.get('late', (float('nan'), float('nan')))[1]:.3f} | {temp_disc.get('stay_late', (float('nan'), float('nan')))[1]:.3f} | {temp_disc.get('switch_late', (float('nan'), float('nan')))[1]:.3f} |

*If switch early < switch late, switch trials are harder to read before the choice is made — consistent with less stereotyped pre-choice dynamics.*

---

## Q4 — Hybrid vs Discrete Accuracy

| Metric | Hybrid | Discrete | Δ (hybrid − disc) |
|--------|--------|----------|--------------------|
| Acc    | {acc_hybrid:.3f} | {acc_disc:.3f} | {acc_hybrid - acc_disc:+.3f} |
| BalAcc | {bacc_hybrid:.3f} | {bacc_disc:.3f} | {bacc_hybrid - bacc_disc:+.3f} |

*If |Δ| < 0.03, HGF supervision does not appreciably hurt choice decodability.*

---

## Q5 — Latent Geometry (PCA variance explained)

| Dim | Hybrid | Discrete-only |
|-----|--------|---------------|
""" + "\n".join(
        f"| {i+1} | {evr_hybrid[i]*100:.1f}% | {evr_disc[i]*100:.1f}% |"
        for i in range(min(len(evr_hybrid), len(evr_disc)))
    ) + f"""
| Total | {evr_hybrid.sum()*100:.1f}% | {evr_disc.sum()*100:.1f}% |

*Higher total variance in hybrid suggests more distributed use of latent dimensions (richer geometry).*

---

## Q6 — HGF Belief Decodability (hybrid only)

Ridge regression probe (α=1.0), {5} repeats:
- R² = **{hgf_r2_hybrid:.3f}** ± {hgf_r2_hybrid_std:.3f}

*R² > 0.10 indicates the latent space encodes meaningful belief information beyond chance.*

---

## Files Generated
- `figures/temporal_decodability.pdf`
- `figures/stay_switch_all.pdf`
- `figures/pca_variance.pdf`
- `figures/confusion_hybrid.pdf`
- `figures/confusion_discrete.pdf`
- `figures/hgf_scatter.pdf`  (hybrid only)
- `analysis_summary.json`
- `analysis_report.md`  ← this file
"""
    return report


def _git_head():
    import subprocess
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                        stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--hybrid', required=True, metavar='DIR',
                   help='Hybrid (b_type=hybrid) run folder path')
    p.add_argument('--discrete', required=True, metavar='DIR',
                   help='Discrete-only run folder path')
    p.add_argument('--out', default=None, metavar='DIR',
                   help='Output directory (default: results/analysis/rtc_analysis_{timestamp})')
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    hybrid_dir = Path(args.hybrid)
    disc_dir   = Path(args.discrete)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(args.out) if args.out else Path('results/analysis') / f'rtc_analysis_{ts}'
    fig_dir = out_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Reward-to-Choice Analysis")
    print(f"  Hybrid:   {hybrid_dir}")
    print(f"  Discrete: {disc_dir}")
    print(f"  Output:   {out_dir}")
    print(f"{'='*60}\n")

    # ── Load runs ─────────────────────────────────────────────────────────────
    print("Loading hybrid run...")
    rh = load_run(hybrid_dir)
    print(f"  Y_train={rh['Y_tr'].shape}  Y_val={rh['Y_va'].shape}")
    print(f"  b_train labels: {np.unique(rh['b_tr'])}")
    print(f"  HGF available: {rh['hgf_tr'] is not None}")

    print("\nLoading discrete run...")
    rd = load_run(disc_dir)
    print(f"  Y_train={rd['Y_tr'].shape}  Y_val={rd['Y_va'].shape}")
    print(f"  b_train labels: {np.unique(rd['b_tr'])}")

    # ── Q1: Choice decodability ───────────────────────────────────────────────
    print("\n--- Q1: Choice decodability (all val windows) ---")
    acc_h, bacc_h, _, _ = fit_linear_probe(rh['Y_tr'], rh['b_tr'], rh['Y_va'], rh['b_va'])
    acc_d, bacc_d, _, _ = fit_linear_probe(rd['Y_tr'], rd['b_tr'], rd['Y_va'], rd['b_va'])
    print(f"  Hybrid:   acc={acc_h:.3f}  bacc={bacc_h:.3f}")
    print(f"  Discrete: acc={acc_d:.3f}  bacc={bacc_d:.3f}")

    # ── Confusion matrices ────────────────────────────────────────────────────
    print("\nPlotting confusion matrices...")
    acc_h_cm, bacc_h_cm = plot_confusion(
        rh['Y_tr'], rh['b_tr'], rh['Y_va'], rh['b_va'],
        CHOICE_NAMES, 'Hybrid — Choice Confusion',
        fig_dir / 'confusion_hybrid.pdf',
    )
    acc_d_cm, bacc_d_cm = plot_confusion(
        rd['Y_tr'], rd['b_tr'], rd['Y_va'], rd['b_va'],
        CHOICE_NAMES, 'Discrete-only — Choice Confusion',
        fig_dir / 'confusion_discrete.pdf',
    )

    # ── Stay/switch ───────────────────────────────────────────────────────────
    print("\n--- Q2: Stay/switch labels ---")
    ss_h, trial_choice_h, _ = compute_stay_switch(rh['b_tr'], rh['b_va'], rh['tid_tr'], rh['tid_va'])
    ss_d, trial_choice_d, _ = compute_stay_switch(rd['b_tr'], rd['b_va'], rd['tid_tr'], rd['tid_va'])

    n_stay_h   = int((ss_h == 0).sum())
    n_switch_h = int((ss_h == 1).sum())
    n_stay_d   = int((ss_d == 0).sum())
    n_switch_d = int((ss_d == 1).sum())
    print(f"  Hybrid val:   stay={n_stay_h}  switch={n_switch_h}  unknown={int((ss_h==-1).sum())}")
    print(f"  Discrete val: stay={n_stay_d}  switch={n_switch_d}  unknown={int((ss_d==-1).sum())}")

    # ── Q2+Q3: Temporal decodability ─────────────────────────────────────────
    print("\n--- Q2/Q3: Temporal decodability (early/mid/late × stay/switch) ---")
    temp_h = temporal_decodability(rh['Y_tr'], rh['b_tr'], rh['Y_va'], rh['b_va'],
                                    rh['tid_va'], ss_h)
    temp_d = temporal_decodability(rd['Y_tr'], rd['b_tr'], rd['Y_va'], rd['b_va'],
                                    rd['tid_va'], ss_d)

    for name, res in [('Hybrid', temp_h), ('Discrete', temp_d)]:
        print(f"\n  {name}:")
        for k, v in res.items():
            if k.startswith('_'): continue
            acc_, bacc_ = v
            print(f"    {k:20s}  acc={acc_:.3f}  bacc={bacc_:.3f}")

    plot_temporal_decodability(temp_h, temp_d, fig_dir / 'temporal_decodability.pdf')
    plot_stay_switch_all(temp_h, temp_d, fig_dir / 'stay_switch_all.pdf')

    # ── Q5: PCA variance ─────────────────────────────────────────────────────
    print("\n--- Q5: PCA variance explained ---")
    evr_h = pca_variance(rh['Y_tr'], rh['Y_va'])
    evr_d = pca_variance(rd['Y_tr'], rd['Y_va'])
    print(f"  Hybrid:   {[f'{v*100:.1f}%' for v in evr_h]}  total={evr_h.sum()*100:.1f}%")
    print(f"  Discrete: {[f'{v*100:.1f}%' for v in evr_d]}  total={evr_d.sum()*100:.1f}%")
    plot_pca_variance(evr_h, evr_d, fig_dir / 'pca_variance.pdf')

    # ── Q6: HGF decodability ─────────────────────────────────────────────────
    hgf_r2_mean = hgf_r2_std = float('nan')
    if rh['hgf_tr'] is not None and rh['hgf_va'] is not None:
        print("\n--- Q6: HGF decodability (hybrid latent) ---")
        # Use train+val together for the scatter; report on val only for R²
        hgf_r2_mean, hgf_r2_std = hgf_linear_r2(
            rh['Y_tr'], rh['hgf_tr'], rh['Y_va'], rh['hgf_va'], n_repeats=5)
        print(f"  R² = {hgf_r2_mean:.3f} ± {hgf_r2_std:.3f}")
        plot_hgf_scatter(rh['Y_va'], rh['hgf_va'], fig_dir / 'hgf_scatter.pdf')
    else:
        print("\n  No HGF arrays found — skipping Q6")

    # ── Counts for report ─────────────────────────────────────────────────────
    ss_counts_h = temp_h.get('_counts', {})
    ss_counts_d = temp_d.get('_counts', {})
    if not ss_counts_h:
        ss_counts_h = {'n_stay': n_stay_h, 'n_switch': n_switch_h}
    if not ss_counts_d:
        ss_counts_d = {'n_stay': n_stay_d, 'n_switch': n_switch_d}

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        'timestamp':       ts,
        'git_head':        _git_head(),
        'hybrid_run_dir':  str(hybrid_dir),
        'discrete_run_dir': str(disc_dir),
        'Q1': {
            'hybrid_acc':   acc_h, 'hybrid_bacc':   bacc_h,
            'discrete_acc': acc_d, 'discrete_bacc': bacc_d,
        },
        'Q2_Q3': {
            'hybrid':   {k: list(v) for k, v in temp_h.items() if not k.startswith('_')},
            'discrete': {k: list(v) for k, v in temp_d.items() if not k.startswith('_')},
            'hybrid_counts':   ss_counts_h,
            'discrete_counts': ss_counts_d,
        },
        'Q4': {
            'delta_acc':  acc_h - acc_d,
            'delta_bacc': bacc_h - bacc_d,
        },
        'Q5_pca': {
            'hybrid_explained_variance':   evr_h.tolist(),
            'discrete_explained_variance': evr_d.tolist(),
        },
        'Q6_hgf': {
            'r2_mean': hgf_r2_mean,
            'r2_std':  hgf_r2_std,
        },
    }
    with open(out_dir / 'analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary JSON: {out_dir / 'analysis_summary.json'}")

    # ── Markdown report ───────────────────────────────────────────────────────
    report = write_report(
        out_dir, str(hybrid_dir), str(disc_dir),
        rh, rd,
        temp_h, temp_d,
        evr_h, evr_d,
        acc_h, bacc_h, acc_d, bacc_d,
        hgf_r2_mean, hgf_r2_std,
        ss_counts_h, ss_counts_d,
    )
    with open(out_dir / 'analysis_report.md', 'w') as f:
        f.write(report)
    print(f"  Report:      {out_dir / 'analysis_report.md'}")

    print(f"\n{'='*60}")
    print("DONE.  Summary:")
    print(f"  Q1  Hybrid acc/bacc:     {acc_h:.3f} / {bacc_h:.3f}")
    print(f"  Q1  Discrete acc/bacc:   {acc_d:.3f} / {bacc_d:.3f}")
    print(f"  Q4  Δ(hybrid−disc):      {acc_h - acc_d:+.3f} acc  /  {bacc_h - bacc_d:+.3f} bacc")
    print(f"  Q5  PCA hybrid total:    {evr_h.sum()*100:.1f}%  (disc: {evr_d.sum()*100:.1f}%)")
    print(f"  Q6  HGF R²:              {hgf_r2_mean:.3f} ± {hgf_r2_std:.3f}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
