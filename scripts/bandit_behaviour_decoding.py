"""
Behaviour decoding evaluation from a BunDLeNet run folder.

Usage:
    python bandit_behaviour_decoding.py <path_to_run_folder>

Loads pre-saved latent trajectories and behaviour labels from the run folder,
trains linear decoders and saves results under <run_folder>/data/decoding/.

Produces the same figure format as bandit_microvariable_evaluation.py so that
direct (raw neural → behaviour) and latent-space (BunDLeNet → behaviour)
decoding can be placed side-by-side.

Evaluates:
  - Discrete state classification: unweighted + weighted CE, permutation chance,
    2×3 summary PDF matching microvariable layout
  - HGF belief regression (only if hgf_belief_{train,validation}.npy exist):
    R² distribution, decoder vs chance, predicted vs true scatter
"""

import sys
import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_recall_fscore_support, r2_score,
)
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ── Config ────────────────────────────────────────────────────────────────────
N_RUNS       = 5
N_EPOCHS     = 100
BATCH_SIZE   = 256
LR           = 0.01
N_PERMUTATIONS = 200
HYBRID_ALPHA = 0.1   # α·CE_norm + (1-α)·MSE  (matches BunDLeNet default)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# State names matching BanditTaskNeuroPixelsDataset defaults
_DEFAULT_STATE_NAMES = {
    0: 'waiting', 1: 'intertrial', 2: 'hold',
    3: 'choosing left', 4: 'choosing right', 5: 'reward', 6: 'no reward',
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_loader(X, y, dtype_y):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=dtype_y)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=True)


def predict_logits(model, X):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        return model(X_t).cpu().numpy()


def predict_regression(model, X):
    return predict_logits(model, X).squeeze()


# Auto-detect train/validation split files in a run data directory.
def _detect_train_validation_splits(data_dir):
    data_dir = Path(data_dir)
    latent_files = list(data_dir.glob('latent_trajectories*.npy'))
    label_files  = list(data_dir.glob('behaviour_labels*.npy'))

    def _extract_split(name, prefix):
        stem = name[:-4] if name.endswith('.npy') else name
        if stem == prefix:
            return ''
        if stem.startswith(prefix + '_'):
            return stem[len(prefix) + 1:]
        if stem.startswith(prefix):
            return stem[len(prefix):].lstrip('._-')
        return stem[len(prefix):] if stem.startswith(prefix) else stem

    latent_map = { _extract_split(p.name, 'latent_trajectories'): p for p in latent_files }
    label_map  = { _extract_split(p.name, 'behaviour_labels'): p for p in label_files }

    candidate_splits = sorted(set(latent_map.keys()) & set(label_map.keys()))
    # Fallback heuristics: try to match 'train'/'validation' tokens inside file names
    if not candidate_splits:
        tokens_train = ['train', 'training']
        tokens_val = ['validation', 'val', 'test']
        found_train = None
        found_val = None
        for p in latent_files:
            for t in tokens_train:
                if t in p.name:
                    found_train = t; break
            if found_train:
                break
        for p in label_files:
            for t in tokens_val:
                if t in p.name:
                    found_val = t; break
            if found_val:
                break
        if found_train and found_val:
            # map detected tokens to canonical names
            for p in latent_files:
                if found_train in p.name:
                    latent_map['train'] = p
                elif found_val in p.name:
                    latent_map['validation'] = p
            for p in label_files:
                if found_train in p.name:
                    label_map['train'] = p
                elif found_val in p.name:
                    label_map['validation'] = p
            candidate_splits = sorted(set(latent_map.keys()) & set(label_map.keys()))

    if not candidate_splits:
        raise RuntimeError(
            f"Could not find matching latent+label split files in {data_dir}. "
            f"Expected files like 'latent_trajectories_train.npy' and 'behaviour_labels_train.npy'. "
            f"Found latent files: {[p.name for p in latent_files]}; label files: {[p.name for p in label_files]}"
        )

    preferred_val = ['validation', 'val', 'test']
    if 'train' in candidate_splits:
        train_split = 'train'
        val_split = next((v for v in preferred_val if v in candidate_splits), None)
        if val_split is None:
            others = [s for s in candidate_splits if s != 'train']
            if others:
                val_split = others[0]
            else:
                raise RuntimeError("Found only 'train' split but no validation/test split.")
    else:
        # choose a validation-like if present, else pick two available splits
        val_split = next((v for v in preferred_val if v in candidate_splits), None)
        if val_split:
            train_split = next(s for s in candidate_splits if s != val_split)
        elif len(candidate_splits) >= 2:
            train_split, val_split = candidate_splits[0], candidate_splits[1]
        else:
            raise RuntimeError("Unable to identify train/validation splits from files in data dir.")

    X_train_path = latent_map.get(train_split) or latent_map.get('')
    y_train_path = label_map.get(train_split) or label_map.get('')
    X_val_path   = latent_map.get(val_split)   or latent_map.get('')
    y_val_path   = label_map.get(val_split)   or label_map.get('')
    return train_split, val_split, X_train_path, y_train_path, X_val_path, y_val_path


# ── Discrete decoder (one variant) ───────────────────────────────────────────

def _run_discrete_variant(X_train, y_train, X_val, y_val,
                           state_labels, class_weights_tensor,
                           use_weighted, label):
    n_classes  = len(state_labels)
    latent_dim = X_train.shape[1]
    crit_weight = class_weights_tensor if use_weighted else None

    loader_train = make_loader(X_train, y_train.astype(np.int64), torch.long)

    val_acc_list  = []
    val_f1_list   = []
    val_conf_sum  = np.zeros((n_classes, n_classes))
    val_preds_all = []
    val_true_all  = []

    print(f"\n  [{label}]")
    for run in range(N_RUNS):
        model = nn.Linear(latent_dim, n_classes).to(DEVICE)
        opt   = optim.Adam(model.parameters(), lr=LR)
        crit  = nn.CrossEntropyLoss(weight=crit_weight)

        model.train()
        for _ in range(N_EPOCHS):
            for xb, yb in loader_train:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                crit(model(xb), yb).backward()
                opt.step()

        logits = predict_logits(model, X_val)
        pred   = logits.argmax(axis=1)
        true   = y_val.astype(np.int64)

        acc = accuracy_score(true, pred)
        f1  = f1_score(true, pred, average=None, labels=state_labels, zero_division=0)

        val_acc_list.append(acc)
        val_f1_list.append(f1)
        val_conf_sum += confusion_matrix(true, pred, labels=state_labels)
        val_preds_all.append(pred)
        val_true_all.append(true)
        print(f"    Run {run+1:2d}/{N_RUNS}  acc={acc:.3f}")

    val_acc_list = np.array(val_acc_list)
    val_f1_list  = np.array(val_f1_list)
    avg_conf     = val_conf_sum / N_RUNS

    print(f"  → {label} val acc: {val_acc_list.mean():.3f} ± {val_acc_list.std():.3f}")
    return dict(
        val_acc_list=val_acc_list,
        val_f1_list=val_f1_list,
        avg_conf=avg_conf,
        val_preds_all=val_preds_all,
        val_true_all=val_true_all,
    )


# ── Discrete decoding (both variants + figures) ───────────────────────────────

def run_discrete_decoding(X_train, y_train, X_val, y_val,
                           out_dir, session_dir, state_names_dict):
    state_labels = np.array(sorted(np.unique(np.concatenate([y_train, y_val]))))
    n_classes    = len(state_labels)
    state_names  = [state_names_dict.get(int(s), f'S{s}') for s in state_labels]
    x_pos        = np.arange(n_classes)

    print(f"\n{'#'*60}")
    print(f"### DISCRETE DECODING — LATENT SPACE ({n_classes} classes) ###")
    print(f"{'#'*60}")
    print(f"Train={len(X_train)}  Val={len(X_val)}  N_runs={N_RUNS}  Epochs={N_EPOCHS}")

    # Class weights from training distribution
    total_train = len(y_train)
    counts = {lbl: int((y_train == lbl).sum()) for lbl in state_labels}
    class_weights = {
        lbl: total_train / (n_classes * counts[lbl]) if counts[lbl] > 0 else 1.0
        for lbl in state_labels
    }
    weights_list         = [class_weights[l] for l in state_labels]
    class_weights_tensor = torch.FloatTensor(weights_list).to(DEVICE)

    res_uw = _run_discrete_variant(X_train, y_train, X_val, y_val,
                                    state_labels, class_weights_tensor,
                                    use_weighted=False, label='UNWEIGHTED')
    res_w  = _run_discrete_variant(X_train, y_train, X_val, y_val,
                                    state_labels, class_weights_tensor,
                                    use_weighted=True,  label='WEIGHTED')

    # Permutation chance baseline
    print(f"\n  Estimating chance accuracy ({N_PERMUTATIONS} permutations)...")
    chance_acc = np.array([
        accuracy_score(y_val, np.random.choice(y_val, size=y_val.shape))
        for _ in range(N_PERMUTATIONS)
    ])
    print(f"  Chance: {chance_acc.mean():.3f} ± {chance_acc.std():.3f}")

    t_uw, p_uw = stats.ttest_ind(res_uw['val_acc_list'], chance_acc)
    t_w,  p_w  = stats.ttest_ind(res_w['val_acc_list'],  chance_acc)

    # ── Save data files ───────────────────────────────────────────────────────
    np.savetxt(out_dir / f'acc_list_val_{session_dir}_unweighted.txt', res_uw['val_acc_list'])
    np.savetxt(out_dir / f'acc_list_val_{session_dir}_weighted.txt',   res_w['val_acc_list'])
    np.savetxt(out_dir / f'acc_list_chance_{session_dir}.txt',         chance_acc)
    np.save(out_dir / f'all_f1_scores_val_{session_dir}_unweighted.npy', res_uw['val_f1_list'])
    np.save(out_dir / f'all_f1_scores_val_{session_dir}_weighted.npy',   res_w['val_f1_list'])

    # ── Derived quantities ────────────────────────────────────────────────────
    uw_f1_means = res_uw['val_f1_list'].mean(axis=0)
    uw_f1_std   = res_uw['val_f1_list'].std(axis=0)
    w_f1_means  = res_w['val_f1_list'].mean(axis=0)
    w_f1_std    = res_w['val_f1_list'].std(axis=0)

    norm_conf_uw = res_uw['avg_conf'] / (res_uw['avg_conf'].sum(axis=1, keepdims=True) + 1e-9) * 100
    norm_conf_w  = res_w['avg_conf']  / (res_w['avg_conf'].sum(axis=1, keepdims=True)  + 1e-9) * 100

    avg_prec, avg_rec = [], []
    for si in range(n_classes):
        precs, recs = [], []
        for pred, true in zip(res_uw['val_preds_all'], res_uw['val_true_all']):
            p, r, _, _ = precision_recall_fscore_support(
                true, pred, labels=state_labels, zero_division=0, average=None)
            precs.append(p[si]); recs.append(r[si])
        avg_prec.append(np.mean(precs)); avg_rec.append(np.mean(recs))

    # ── 2×3 Summary figure ───────────────────────────────────────────────────
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Latent Space Decoder — {session_dir}', fontsize=15, fontweight='bold')

    # (0,0) Accuracy boxplot
    ax = axs[0, 0]
    bp = ax.boxplot(
        [res_uw['val_acc_list'], res_w['val_acc_list'], chance_acc],
        positions=[1, 2, 3], widths=0.6, patch_artist=True, showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
    for patch, col in zip(bp['boxes'], ['#4C9BE8', '#6BBF6B', '#AAAAAA']):
        patch.set_facecolor(col); patch.set_alpha(0.8)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(['Unweighted', 'Weighted', 'Chance'])
    ax.set_ylabel('Validation Accuracy', fontsize=12)
    ax.set_title('Decoder Accuracy vs Chance', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y'); ax.set_ylim([0, 1.0])
    ax.text(0.5, 0.98,
            f'UW: t={t_uw:.1f}, p={p_uw:.1e}\nW:  t={t_w:.1f}, p={p_w:.1e}',
            transform=ax.transAxes, ha='center', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

    # (0,1) Confusion matrix — Unweighted
    ax = axs[0, 1]
    sns.heatmap(norm_conf_uw, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax,
                xticklabels=state_names, yticklabels=state_names,
                cbar_kws={'label': '%'}, vmin=0, vmax=100)
    ax.set_title('Confusion Matrix — Unweighted (%)', fontweight='bold')
    ax.tick_params(axis='x', rotation=45); ax.tick_params(axis='y', rotation=0)

    # (0,2) Confusion matrix — Weighted
    ax = axs[0, 2]
    sns.heatmap(norm_conf_w, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax,
                xticklabels=state_names, yticklabels=state_names,
                cbar_kws={'label': '%'}, vmin=0, vmax=100)
    ax.set_title('Confusion Matrix — Weighted (%)', fontweight='bold')
    ax.tick_params(axis='x', rotation=45); ax.tick_params(axis='y', rotation=0)

    # (1,0) Per-state F1 UW vs W
    ax = axs[1, 0]
    bw = 0.35
    ax.bar(x_pos - bw/2, uw_f1_means, bw, yerr=uw_f1_std, label='Unweighted',
           color='#4C9BE8', alpha=0.85, capsize=4, error_kw=dict(elinewidth=1.2))
    ax.bar(x_pos + bw/2, w_f1_means,  bw, yerr=w_f1_std,  label='Weighted',
           color='#6BBF6B', alpha=0.85, capsize=4, error_kw=dict(elinewidth=1.2))
    ax.set_xticks(x_pos); ax.set_xticklabels(state_names, rotation=45, ha='right')
    ax.set_ylabel('F1', fontsize=12)
    ax.set_title('Per-State F1 — UW vs W (mean ± std)', fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y'); ax.set_ylim([0, 1.05])

    # (1,1) Precision / Recall / F1 — Unweighted
    ax = axs[1, 1]
    pw = 0.25
    ax.bar(x_pos - pw,  avg_prec,    pw, label='Precision', color='#5B8DD9', alpha=0.85)
    ax.bar(x_pos,       avg_rec,     pw, label='Recall',    color='#E8864C', alpha=0.85)
    ax.bar(x_pos + pw,  uw_f1_means, pw, label='F1',        color='#6BBF6B', alpha=0.85)
    ax.set_xticks(x_pos); ax.set_xticklabels(state_names, rotation=45, ha='right')
    ax.set_title('Precision / Recall / F1 — Unweighted', fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y'); ax.set_ylim([0, 1.05])

    # (1,2) Summary text
    ax = axs[1, 2]; ax.axis('off')
    summary_t = (
        f"LATENT SPACE DECODER — {session_dir}\n{'─'*36}\n"
        f"(BunDLeNet latent space, fixed train/val split)\n\n"
        f"Val Accuracy\n"
        f"  Unweighted : {res_uw['val_acc_list'].mean():.3f} ± {res_uw['val_acc_list'].std():.3f}\n"
        f"  Weighted   : {res_w['val_acc_list'].mean():.3f} ± {res_w['val_acc_list'].std():.3f}\n"
        f"  Chance     : {chance_acc.mean():.3f} ± {chance_acc.std():.3f}\n"
        f"  W − UW     : {res_w['val_acc_list'].mean() - res_uw['val_acc_list'].mean():+.3f}\n\n"
        f"Macro F1 (val, mean)\n"
        f"  Unweighted : {uw_f1_means.mean():.3f}\n"
        f"  Weighted   : {w_f1_means.mean():.3f}\n"
        f"  W − UW     : {w_f1_means.mean() - uw_f1_means.mean():+.3f}\n\n"
        f"vs Chance (t-test)\n"
        f"  UW : t={t_uw:.2f}, p={p_uw:.2e}\n"
        f"  W  : t={t_w:.2f}, p={p_w:.2e}\n\n"
        f"N_runs={N_RUNS}  ·  Epochs={N_EPOCHS}\n"
        f"Permutations={N_PERMUTATIONS}"
    )
    ax.text(0.05, 0.97, summary_t, transform=ax.transAxes, fontsize=11,
            va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

    plt.tight_layout()
    fig_path = out_dir / f'summary_{session_dir}.pdf'
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved summary figure → {fig_path}")

    return {
        'unweighted_val_acc':     float(res_uw['val_acc_list'].mean()),
        'unweighted_val_acc_std': float(res_uw['val_acc_list'].std()),
        'weighted_val_acc':       float(res_w['val_acc_list'].mean()),
        'weighted_val_acc_std':   float(res_w['val_acc_list'].std()),
        'chance_acc':             float(chance_acc.mean()),
        't_weighted_vs_chance':   float(t_w),
        'p_weighted_vs_chance':   float(p_w),
    }


def _process_decoding_for_data_dir(data_dir, session_label, config, latent_dim):
    """Process decoding for a given data directory (run/data or fold_X/data).

    Raises RuntimeError if expected latent/label split files are not found.
    """
    data_dir = Path(data_dir)
    # detect splits (will raise RuntimeError if not found)
    train_split, val_split, X_train_path, y_train_path, X_val_path, y_val_path = _detect_train_validation_splits(data_dir)

    print(f"Detected splits: train='{train_split}'  val='{val_split}'")
    print(f"  X_train -> {X_train_path.name}")
    print(f"  y_train -> {y_train_path.name}")
    print(f"  X_val   -> {X_val_path.name}")

    X_train = np.load(X_train_path)
    X_val   = np.load(X_val_path)

    # Load discrete labels
    y_train = np.load(y_train_path).astype(int)
    y_val   = np.load(y_val_path).astype(int)

    print(f"Samples     : train={len(X_train)}, val={len(X_val)}")
    print(f"Labels      : {sorted(np.unique(y_train).tolist())}")

    # Detect trial IDs (trial-based runs) if present
    trial_ids_train = None
    trial_ids_val = None
    for p in data_dir.glob('trial_ids*.npy'):
        if train_split and train_split in p.name:
            try:
                trial_ids_train = np.load(p)
            except Exception:
                print(f"Warning: could not load {p}")
        if val_split and val_split in p.name:
            try:
                trial_ids_val = np.load(p)
            except Exception:
                print(f"Warning: could not load {p}")
    if trial_ids_train is not None or trial_ids_val is not None:
        print("Trial IDs detected (trial-based run). Decoding will run per-sample; trial-aware metrics can be added if desired.")

    # Check for HGF files
    hgf_train_path = data_dir / 'hgf_belief_train.npy'
    hgf_val_path   = data_dir / 'hgf_belief_validation.npy'
    has_hgf = hgf_train_path.exists() and hgf_val_path.exists()
    if has_hgf:
        hgf_train = np.load(hgf_train_path).astype(np.float32)
        hgf_val   = np.load(hgf_val_path).astype(np.float32)
        print(f"HGF belief  : found  range=[{hgf_train.min():.3f}, {hgf_train.max():.3f}]")
    else:
        hgf_train = hgf_val = None
        print("HGF belief  : not found — skipping HGF decoding")

    # Output directory for decoding results
    out_dir = data_dir / 'decoding'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build state name mapping from config (fallback to default)
    if 'b_labels' in config and isinstance(config['b_labels'], list) and len(config['b_labels']) > 0:
        state_names_dict = {i: config['b_labels'][i] for i in range(len(config['b_labels']))}
    else:
        state_names_dict = _DEFAULT_STATE_NAMES

    metrics = {}

    # Discrete decoding
    metrics['discrete'] = run_discrete_decoding(
        X_train, y_train, X_val, y_val,
        out_dir, session_label, state_names_dict,
    )

    # HGF decoding
    if has_hgf:
        metrics['hgf'] = run_hgf_decoding(
            X_train, hgf_train, X_val, hgf_val,
            out_dir, session_label,
        )

    # Hybrid decoding
    if has_hgf:
        metrics['hybrid'] = run_hybrid_decoding(
            X_train, y_train, hgf_train,
            X_val, y_val, hgf_val,
            out_dir, session_label, state_names_dict,
        )

    # Save summary JSON
    summary = {
        'session_dir': session_label,
        'n_runs':      N_RUNS,
        'n_epochs':    N_EPOCHS,
        'hybrid_alpha': HYBRID_ALPHA,
        'latent_dim':  latent_dim,
        'decoder':     'linear (fixed train/val split from BunDLeNet run)',
        'metrics':     metrics,
    }
    with open(out_dir / 'decoding_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Final printout (concise)
    d = metrics['discrete']
    print(f"\n{'='*60}")
    print(f"RESULTS saved to: {out_dir}")
    print(f"  Discrete acc (UW) : {d['unweighted_val_acc']:.3f} ± {d['unweighted_val_acc_std']:.3f}")
    print(f"  Discrete acc (W)  : {d['weighted_val_acc']:.3f} ± {d['weighted_val_acc_std']:.3f}")
    print(f"  Chance acc        : {d['chance_acc']:.3f}")
    if has_hgf:
        h = metrics['hgf']
        print(f"  HGF R²            : {h['val_r2_mean']:.3f} ± {h['val_r2_std']:.3f}")
        hy = metrics['hybrid']
        print(f"  Hybrid acc (UW)   : {hy['unweighted_val_acc']:.3f} ± {hy['unweighted_val_acc_std']:.3f}")
        print(f"  Hybrid acc (W)    : {hy['weighted_val_acc']:.3f} ± {hy['weighted_val_acc_std']:.3f}")
        print(f"  Hybrid R² (cont,UW): {hy['unweighted_cont_r2_mean']:.3f} ± {hy['unweighted_cont_r2_std']:.3f}")
    print(f"{'='*60}")



# ── HGF decoder ───────────────────────────────────────────────────────────────

def run_hgf_decoding(X_train, hgf_train, X_val, hgf_val, out_dir, session_dir):
    print(f"\n{'#'*60}")
    print(f"### HGF BELIEF DECODING — LATENT SPACE ###")
    print(f"{'#'*60}")
    latent_dim = X_train.shape[1]
    mse_fn     = nn.MSELoss()

    loader_train = make_loader(
        X_train, hgf_train.astype(np.float32).reshape(-1, 1), torch.float32
    )

    val_r2_list  = []
    last_pred    = last_true = None

    for run in range(N_RUNS):
        model = nn.Linear(latent_dim, 1).to(DEVICE)
        opt   = optim.Adam(model.parameters(), lr=LR)

        model.train()
        for _ in range(N_EPOCHS):
            for xb, yb in loader_train:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                mse_fn(model(xb), yb).backward()
                opt.step()

        pred = predict_regression(model, X_val)
        r2   = r2_score(hgf_val, pred)

        val_r2_list.append(r2)
        last_pred = pred
        last_true = hgf_val
        print(f"  Run {run+1:2d}/{N_RUNS}  R²={r2:.3f}")

    val_r2_list = np.array(val_r2_list)
    print(f"\n  Val R²: {val_r2_list.mean():.3f} ± {val_r2_list.std():.3f}")

    # Permutation baseline (val only — no train split available for R²)
    print(f"  Estimating chance R² ({N_PERMUTATIONS} permutations)...")
    chance_r2 = np.array([
        r2_score(hgf_val, np.random.permutation(hgf_val))
        for _ in range(N_PERMUTATIONS)
    ])
    print(f"  Chance R²: {chance_r2.mean():.3f} ± {chance_r2.std():.3f}")

    t_stat, p_val = stats.ttest_ind(val_r2_list, chance_r2)
    print(f"  t-test vs chance: t={t_stat:.2f}, p={p_val:.2e}")

    # ── Save data ─────────────────────────────────────────────────────────────
    np.savetxt(out_dir / f'r2_val_{session_dir}.txt',    val_r2_list)
    np.savetxt(out_dir / f'r2_chance_{session_dir}.txt', chance_r2)

    # ── Plot 1: R² decoder vs chance ─────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    bp = ax1.boxplot([val_r2_list, chance_r2],
                     positions=[1, 2], widths=0.6, patch_artist=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
    for patch, col in zip(bp['boxes'], ['skyblue', 'lightgray']):
        patch.set_facecolor(col)
    ax1.set_xticks([1, 2]); ax1.set_xticklabels(['Belief Decoder', 'Chance (permutation)'])
    ax1.set_ylabel('R²', fontsize=13)
    ax1.set_title(f'Belief Decoder vs Chance — {session_dir}', fontsize=15, fontweight='bold')
    ax1.text(0.5, 0.95,
             f'Val R² vs Chance: t={t_stat:.2f}, p={p_val:.2e}',
             transform=ax1.transAxes, ha='center', va='top', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax1.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out_dir / f'decoder_vs_chance_{session_dir}.pdf', bbox_inches='tight')
    plt.close()

    # ── Plot 2: predicted vs true ─────────────────────────────────────────────
    if last_pred is not None:
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        ax2.scatter(last_true, last_pred, alpha=0.3, s=10, color='steelblue', label='Predictions')
        lo = min(last_true.min(), last_pred.min())
        hi = max(last_true.max(), last_pred.max())
        ax2.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='Identity (perfect)')
        r2_last = r2_score(last_true, last_pred)
        ax2.set_xlabel('True HGF belief', fontsize=13)
        ax2.set_ylabel('Predicted HGF belief', fontsize=13)
        ax2.set_title(f'Predicted vs True Belief — {session_dir}', fontsize=13, fontweight='bold')
        ax2.text(0.05, 0.95, f'R² = {r2_last:.3f}',
                 transform=ax2.transAxes, va='top', fontsize=13,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax2.legend(); ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f'predicted_vs_true_{session_dir}.pdf', bbox_inches='tight')
        plt.close()

    print(f"  Saved HGF figures → {out_dir}")

    return {
        'val_r2_mean':    float(val_r2_list.mean()),
        'val_r2_std':     float(val_r2_list.std()),
        'chance_r2_mean': float(chance_r2.mean()),
        't_vs_chance':    float(t_stat),
        'p_vs_chance':    float(p_val),
    }


# ── Hybrid decoder ─────────────────────────────────────────────────────────────

def _run_hybrid_variant(X_train, y_disc_train, hgf_train,
                         X_val, y_disc_val, hgf_val,
                         state_labels, class_weights_tensor,
                         use_weighted, label):
    import math
    n_classes  = len(state_labels)
    latent_dim = X_train.shape[1]
    output_dim = n_classes + 1
    ce_norm    = math.log(n_classes)
    crit_weight = class_weights_tensor if use_weighted else None

    # Build train loader: X, (disc_label, hgf_belief)
    X_t       = torch.tensor(X_train, dtype=torch.float32)
    disc_t    = torch.tensor(y_disc_train.astype(np.int64), dtype=torch.long)
    hgf_t     = torch.tensor(hgf_train.astype(np.float32).reshape(-1, 1), dtype=torch.float32)
    loader_train = DataLoader(
        torch.utils.data.TensorDataset(X_t, disc_t, hgf_t),
        batch_size=BATCH_SIZE, shuffle=True,
    )

    ce_fn  = nn.CrossEntropyLoss(weight=crit_weight)
    mse_fn = nn.MSELoss()

    val_acc_list  = []
    val_f1_list   = []
    val_conf_sum  = np.zeros((n_classes, n_classes))
    val_preds_all = []
    val_true_all  = []
    val_r2_list   = []

    print(f"\n  [{label}]")
    for run in range(N_RUNS):
        model = nn.Linear(latent_dim, output_dim).to(DEVICE)
        opt   = optim.Adam(model.parameters(), lr=LR)

        model.train()
        for _ in range(N_EPOCHS):
            for xb, disc_b, hgf_b in loader_train:
                xb, disc_b, hgf_b = xb.to(DEVICE), disc_b.to(DEVICE), hgf_b.to(DEVICE)
                out      = model(xb)
                logits   = out[:, :n_classes]
                cont_out = out[:, n_classes:n_classes+1]
                loss = (HYBRID_ALPHA * ce_fn(logits, disc_b) / ce_norm
                        + (1 - HYBRID_ALPHA) * mse_fn(cont_out, hgf_b))
                opt.zero_grad(); loss.backward(); opt.step()

        # Evaluate discrete head
        model.eval()
        with torch.no_grad():
            X_val_t   = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
            out_val   = model(X_val_t).cpu().numpy()
        disc_pred  = out_val[:, :n_classes].argmax(axis=1)
        disc_true  = y_disc_val.astype(np.int64)
        cont_pred  = out_val[:, n_classes]
        cont_true  = hgf_val.astype(np.float32)

        acc = accuracy_score(disc_true, disc_pred)
        f1  = f1_score(disc_true, disc_pred, average=None, labels=state_labels, zero_division=0)
        r2  = r2_score(cont_true, cont_pred)

        val_acc_list.append(acc)
        val_f1_list.append(f1)
        val_conf_sum += confusion_matrix(disc_true, disc_pred, labels=state_labels)
        val_preds_all.append(disc_pred)
        val_true_all.append(disc_true)
        val_r2_list.append(r2)
        print(f"    Run {run+1:2d}/{N_RUNS}  acc={acc:.3f}  R²(cont)={r2:.3f}")

    val_acc_list = np.array(val_acc_list)
    val_f1_list  = np.array(val_f1_list)
    val_r2_list  = np.array(val_r2_list)
    avg_conf     = val_conf_sum / N_RUNS

    print(f"  → {label} val acc: {val_acc_list.mean():.3f} ± {val_acc_list.std():.3f} "
          f"  R²: {val_r2_list.mean():.3f} ± {val_r2_list.std():.3f}")
    return dict(
        val_acc_list=val_acc_list,
        val_f1_list=val_f1_list,
        avg_conf=avg_conf,
        val_preds_all=val_preds_all,
        val_true_all=val_true_all,
        val_r2_list=val_r2_list,
    )


def run_hybrid_decoding(X_train, y_train, hgf_train,
                         X_val, y_val, hgf_val,
                         out_dir, session_dir, state_names_dict):
    """Joint discrete+continuous decoder on the BunDLeNet latent space.

    Uses α·CE_norm + (1-α)·MSE (identical to BunDLeNet training loss) and
    reports both:
      - Discrete metrics (accuracy, F1, confusion) — 2×3 summary PDF
      - Continuous R² from the shared linear head — saved as txt + in JSON
    """
    state_labels = np.array(sorted(np.unique(np.concatenate([y_train, y_val]))))
    n_classes    = len(state_labels)
    state_names  = [state_names_dict.get(int(s), f'S{s}') for s in state_labels]
    x_pos        = np.arange(n_classes)

    print(f"\n{'#'*60}")
    print(f"### HYBRID DECODING — LATENT SPACE (α={HYBRID_ALPHA}) ###")
    print(f"{'#'*60}")
    print(f"Train={len(X_train)}  Val={len(X_val)}  N_runs={N_RUNS}  Epochs={N_EPOCHS}")
    print(f"α·CE_norm + (1-α)·MSE  (α={HYBRID_ALPHA})")

    total_train = len(y_train)
    counts = {lbl: int((y_train == lbl).sum()) for lbl in state_labels}
    class_weights = {
        lbl: total_train / (n_classes * counts[lbl]) if counts[lbl] > 0 else 1.0
        for lbl in state_labels
    }
    weights_list         = [class_weights[l] for l in state_labels]
    class_weights_tensor = torch.FloatTensor(weights_list).to(DEVICE)

    res_uw = _run_hybrid_variant(
        X_train, y_train, hgf_train, X_val, y_val, hgf_val,
        state_labels, class_weights_tensor, use_weighted=False, label='UNWEIGHTED',
    )
    res_w  = _run_hybrid_variant(
        X_train, y_train, hgf_train, X_val, y_val, hgf_val,
        state_labels, class_weights_tensor, use_weighted=True,  label='WEIGHTED',
    )

    # Permutation chance baseline (discrete)
    print(f"\n  Estimating chance accuracy ({N_PERMUTATIONS} permutations)...")
    chance_acc = np.array([
        accuracy_score(y_val, np.random.choice(y_val, size=y_val.shape))
        for _ in range(N_PERMUTATIONS)
    ])
    print(f"  Chance: {chance_acc.mean():.3f} ± {chance_acc.std():.3f}")

    t_uw, p_uw = stats.ttest_ind(res_uw['val_acc_list'], chance_acc)
    t_w,  p_w  = stats.ttest_ind(res_w['val_acc_list'],  chance_acc)

    # ── Save data files ───────────────────────────────────────────────────────
    np.savetxt(out_dir / f'acc_list_hybrid_val_{session_dir}_unweighted.txt', res_uw['val_acc_list'])
    np.savetxt(out_dir / f'acc_list_hybrid_val_{session_dir}_weighted.txt',   res_w['val_acc_list'])
    np.savetxt(out_dir / f'acc_list_hybrid_chance_{session_dir}.txt',         chance_acc)
    np.save(out_dir / f'all_f1_scores_hybrid_val_{session_dir}_unweighted.npy', res_uw['val_f1_list'])
    np.save(out_dir / f'all_f1_scores_hybrid_val_{session_dir}_weighted.npy',   res_w['val_f1_list'])
    np.savetxt(out_dir / f'r2_hybrid_continuous_val_{session_dir}_unweighted.txt', res_uw['val_r2_list'])
    np.savetxt(out_dir / f'r2_hybrid_continuous_val_{session_dir}_weighted.txt',   res_w['val_r2_list'])

    # ── Derived quantities ────────────────────────────────────────────────────
    uw_f1_means = res_uw['val_f1_list'].mean(axis=0)
    uw_f1_std   = res_uw['val_f1_list'].std(axis=0)
    w_f1_means  = res_w['val_f1_list'].mean(axis=0)
    w_f1_std    = res_w['val_f1_list'].std(axis=0)

    norm_conf_uw = res_uw['avg_conf'] / (res_uw['avg_conf'].sum(axis=1, keepdims=True) + 1e-9) * 100
    norm_conf_w  = res_w['avg_conf']  / (res_w['avg_conf'].sum(axis=1, keepdims=True)  + 1e-9) * 100

    avg_prec, avg_rec = [], []
    for si in range(n_classes):
        precs, recs = [], []
        for pred, true in zip(res_uw['val_preds_all'], res_uw['val_true_all']):
            p, r, _, _ = precision_recall_fscore_support(
                true, pred, labels=state_labels, zero_division=0, average=None)
            precs.append(p[si]); recs.append(r[si])
        avg_prec.append(np.mean(precs)); avg_rec.append(np.mean(recs))

    # ── 2×3 Summary figure ───────────────────────────────────────────────────
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Latent Hybrid Decoder (α={HYBRID_ALPHA}) — {session_dir}',
                 fontsize=15, fontweight='bold')

    # (0,0) Accuracy boxplot
    ax = axs[0, 0]
    bp = ax.boxplot(
        [res_uw['val_acc_list'], res_w['val_acc_list'], chance_acc],
        positions=[1, 2, 3], widths=0.6, patch_artist=True, showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
    for patch, col in zip(bp['boxes'], ['#4C9BE8', '#6BBF6B', '#AAAAAA']):
        patch.set_facecolor(col); patch.set_alpha(0.8)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(['Unweighted', 'Weighted', 'Chance'])
    ax.set_ylabel('Validation Accuracy', fontsize=12)
    ax.set_title('Decoder Accuracy vs Chance', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y'); ax.set_ylim([0, 1.0])
    ax.text(0.5, 0.98,
            f'UW: t={t_uw:.1f}, p={p_uw:.1e}\nW:  t={t_w:.1f}, p={p_w:.1e}',
            transform=ax.transAxes, ha='center', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

    # (0,1) Confusion matrix — Unweighted
    ax = axs[0, 1]
    sns.heatmap(norm_conf_uw, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax,
                xticklabels=state_names, yticklabels=state_names,
                cbar_kws={'label': '%'}, vmin=0, vmax=100)
    ax.set_title('Confusion Matrix — Unweighted (%)', fontweight='bold')
    ax.tick_params(axis='x', rotation=45); ax.tick_params(axis='y', rotation=0)

    # (0,2) Confusion matrix — Weighted
    ax = axs[0, 2]
    sns.heatmap(norm_conf_w, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax,
                xticklabels=state_names, yticklabels=state_names,
                cbar_kws={'label': '%'}, vmin=0, vmax=100)
    ax.set_title('Confusion Matrix — Weighted (%)', fontweight='bold')
    ax.tick_params(axis='x', rotation=45); ax.tick_params(axis='y', rotation=0)

    # (1,0) Per-state F1 UW vs W
    ax = axs[1, 0]
    bw = 0.35
    ax.bar(x_pos - bw/2, uw_f1_means, bw, yerr=uw_f1_std, label='Unweighted',
           color='#4C9BE8', alpha=0.85, capsize=4, error_kw=dict(elinewidth=1.2))
    ax.bar(x_pos + bw/2, w_f1_means,  bw, yerr=w_f1_std,  label='Weighted',
           color='#6BBF6B', alpha=0.85, capsize=4, error_kw=dict(elinewidth=1.2))
    ax.set_xticks(x_pos); ax.set_xticklabels(state_names, rotation=45, ha='right')
    ax.set_ylabel('F1', fontsize=12)
    ax.set_title('Per-State F1 — UW vs W (mean ± std)', fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y'); ax.set_ylim([0, 1.05])

    # (1,1) Precision / Recall / F1 — Unweighted
    ax = axs[1, 1]
    pw = 0.25
    ax.bar(x_pos - pw,  avg_prec,    pw, label='Precision', color='#5B8DD9', alpha=0.85)
    ax.bar(x_pos,       avg_rec,     pw, label='Recall',    color='#E8864C', alpha=0.85)
    ax.bar(x_pos + pw,  uw_f1_means, pw, label='F1',        color='#6BBF6B', alpha=0.85)
    ax.set_xticks(x_pos); ax.set_xticklabels(state_names, rotation=45, ha='right')
    ax.set_title('Precision / Recall / F1 — Unweighted', fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y'); ax.set_ylim([0, 1.05])

    # (1,2) Summary text
    ax = axs[1, 2]; ax.axis('off')
    summary_t = (
        f"HYBRID LATENT DECODER — {session_dir}\n{'─'*36}\n"
        f"(BunDLeNet latent, α={HYBRID_ALPHA}  α·CE_norm + (1-α)·MSE)\n\n"
        f"Val Accuracy (discrete head)\n"
        f"  Unweighted : {res_uw['val_acc_list'].mean():.3f} ± {res_uw['val_acc_list'].std():.3f}\n"
        f"  Weighted   : {res_w['val_acc_list'].mean():.3f} ± {res_w['val_acc_list'].std():.3f}\n"
        f"  Chance     : {chance_acc.mean():.3f} ± {chance_acc.std():.3f}\n"
        f"  W − UW     : {res_w['val_acc_list'].mean() - res_uw['val_acc_list'].mean():+.3f}\n\n"
        f"Val R² (continuous head)\n"
        f"  Unweighted : {res_uw['val_r2_list'].mean():.3f} ± {res_uw['val_r2_list'].std():.3f}\n"
        f"  Weighted   : {res_w['val_r2_list'].mean():.3f} ± {res_w['val_r2_list'].std():.3f}\n\n"
        f"Macro F1 (val, mean)\n"
        f"  Unweighted : {uw_f1_means.mean():.3f}\n"
        f"  Weighted   : {w_f1_means.mean():.3f}\n\n"
        f"vs Chance (t-test)\n"
        f"  UW : t={t_uw:.2f}, p={p_uw:.2e}\n"
        f"  W  : t={t_w:.2f}, p={p_w:.2e}\n\n"
        f"N_runs={N_RUNS}  ·  Epochs={N_EPOCHS}\n"
        f"Permutations={N_PERMUTATIONS}"
    )
    ax.text(0.05, 0.97, summary_t, transform=ax.transAxes, fontsize=11,
            va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

    plt.tight_layout()
    fig_path = out_dir / f'summary_hybrid_{session_dir}.pdf'
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved hybrid summary figure → {fig_path}")

    return {
        'unweighted_val_acc':       float(res_uw['val_acc_list'].mean()),
        'unweighted_val_acc_std':   float(res_uw['val_acc_list'].std()),
        'weighted_val_acc':         float(res_w['val_acc_list'].mean()),
        'weighted_val_acc_std':     float(res_w['val_acc_list'].std()),
        'chance_acc':               float(chance_acc.mean()),
        't_weighted_vs_chance':     float(t_w),
        'p_weighted_vs_chance':     float(p_w),
        'unweighted_cont_r2_mean':  float(res_uw['val_r2_list'].mean()),
        'unweighted_cont_r2_std':   float(res_uw['val_r2_list'].std()),
        'weighted_cont_r2_mean':    float(res_w['val_r2_list'].mean()),
        'weighted_cont_r2_std':     float(res_w['val_r2_list'].std()),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(
        description='Behaviour decoding evaluation from a BunDLeNet run folder.'
    )
    ap.add_argument('run_folder', help='Path to the BunDLeNet run folder.')
    ap.add_argument(
        '--out', default=None,
        help='Output directory for decoding results '
             '(default: <run_folder>/data/decoding).',
    )
    args = ap.parse_args()

    run_dir = Path(args.run_folder)
    if not run_dir.is_dir():
        print(f"ERROR: not a directory: {run_dir}")
        sys.exit(1)

    # Load config
    with open(run_dir / 'config.json') as f:
        config = json.load(f)
    latent_dim  = config['latent_dim']
    session_dir = os.path.basename(config.get('data_path', '').rstrip('/'))
    if not session_dir:
        session_dir = run_dir.name[:40]

    print(f"Run folder  : {run_dir.name}")
    print(f"Session     : {session_dir}")
    print(f"Latent dim  : {latent_dim}  |  Device: {DEVICE}")

    # Auto-detect latent + label split files in the run `data/` folder
    data_dir = run_dir / 'data'
    try:
        # Try top-level data dir first
        train_split, val_split, X_train_path, y_train_path, X_val_path, y_val_path = _detect_train_validation_splits(data_dir)
    except RuntimeError as e:
        # Fallback: try per-fold directories (CV runs)
        fold_dirs = sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith('fold_')])
        if fold_dirs:
            print(f"No top-level latent/label splits found in {data_dir}; attempting per-fold decoding for {len(fold_dirs)} folds.")
            for fold_dir in fold_dirs:
                fold_data = fold_dir / 'data'
                try:
                    _process_decoding_for_data_dir(fold_data, f"{session_dir}_{fold_dir.name}", config, latent_dim)
                except Exception as e2:
                    print(f"  Skipping {fold_dir.name}: {e2}")
            sys.exit(0)
        else:
            print(f"ERROR: {e}")
            sys.exit(1)

    print(f"Detected splits: train='{train_split}'  val='{val_split}'")
    print(f"  X_train -> {X_train_path.name}")
    print(f"  y_train -> {y_train_path.name}")
    print(f"  X_val   -> {X_val_path.name}")
    print(f"  y_val   -> {y_val_path.name}")

    X_train = np.load(X_train_path)
    X_val   = np.load(X_val_path)

    # Load discrete labels
    y_train = np.load(y_train_path).astype(int)
    y_val   = np.load(y_val_path).astype(int)

    print(f"Samples     : train={len(X_train)}, val={len(X_val)}")
    print(f"Labels      : {sorted(np.unique(y_train).tolist())}")

    # Detect trial IDs (trial-based runs) if present
    trial_ids_train = None
    trial_ids_val = None
    for p in data_dir.glob('trial_ids*.npy'):
        if train_split and train_split in p.name:
            try:
                trial_ids_train = np.load(p)
            except Exception:
                print(f"Warning: could not load {p}")
        if val_split and val_split in p.name:
            try:
                trial_ids_val = np.load(p)
            except Exception:
                print(f"Warning: could not load {p}")
    if trial_ids_train is not None or trial_ids_val is not None:
        print("Trial IDs detected (trial-based run). Decoding will run per-sample; trial-aware metrics can be added if desired.")

    # Check for HGF files
    hgf_train_path = run_dir / 'data' / 'hgf_belief_train.npy'
    hgf_val_path   = run_dir / 'data' / 'hgf_belief_validation.npy'
    has_hgf = hgf_train_path.exists() and hgf_val_path.exists()
    if has_hgf:
        hgf_train = np.load(hgf_train_path).astype(np.float32)
        hgf_val   = np.load(hgf_val_path).astype(np.float32)
        print(f"HGF belief  : found  range=[{hgf_train.min():.3f}, {hgf_train.max():.3f}]")
    else:
        print("HGF belief  : not found — skipping HGF decoding")

    # Output directory
    out_dir = Path(args.out) if args.out else run_dir / 'data' / 'decoding'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build state name mapping from run config if available. The run's
    # `config.json` may include `b_labels` (ordered list of class names).
    # Use that to map label integers -> human-readable names; otherwise
    # fall back to the package default `_DEFAULT_STATE_NAMES`.
    if 'b_labels' in config and isinstance(config['b_labels'], list) and len(config['b_labels']) > 0:
        state_names_dict = {i: config['b_labels'][i] for i in range(len(config['b_labels']))}
    else:
        state_names_dict = _DEFAULT_STATE_NAMES

    metrics = {}

    # ── Discrete decoding ─────────────────────────────────────────────────────
    metrics['discrete'] = run_discrete_decoding(
        X_train, y_train, X_val, y_val,
        out_dir, session_dir, state_names_dict,
    )

    # ── HGF decoding ──────────────────────────────────────────────────────────
    if has_hgf:
        metrics['hgf'] = run_hgf_decoding(
            X_train, hgf_train, X_val, hgf_val,
            out_dir, session_dir,
        )

    # ── Hybrid decoding ───────────────────────────────────────────────────────
    if has_hgf:
        metrics['hybrid'] = run_hybrid_decoding(
            X_train, y_train, hgf_train,
            X_val, y_val, hgf_val,
            out_dir, session_dir, state_names_dict,
        )

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        'session_dir': session_dir,
        'n_runs':      N_RUNS,
        'n_epochs':    N_EPOCHS,
        'hybrid_alpha': HYBRID_ALPHA,
        'latent_dim':  latent_dim,
        'decoder':     'linear (fixed train/val split from BunDLeNet run)',
        'metrics':     metrics,
    }
    with open(out_dir / 'decoding_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # ── Final printout ────────────────────────────────────────────────────────
    d = metrics['discrete']
    print(f"\n{'='*60}")
    print(f"RESULTS saved to: {out_dir}")
    print(f"  Discrete acc (UW) : {d['unweighted_val_acc']:.3f} ± {d['unweighted_val_acc_std']:.3f}")
    print(f"  Discrete acc (W)  : {d['weighted_val_acc']:.3f} ± {d['weighted_val_acc_std']:.3f}")
    print(f"  Chance acc        : {d['chance_acc']:.3f}")
    if has_hgf:
        h = metrics['hgf']
        print(f"  HGF R²            : {h['val_r2_mean']:.3f} ± {h['val_r2_std']:.3f}")
        print(f"  HGF chance R²     : {h['chance_r2_mean']:.3f}")
        hy = metrics['hybrid']
        print(f"  Hybrid acc (UW)   : {hy['unweighted_val_acc']:.3f} ± {hy['unweighted_val_acc_std']:.3f}")
        print(f"  Hybrid acc (W)    : {hy['weighted_val_acc']:.3f} ± {hy['weighted_val_acc_std']:.3f}")
        print(f"  Hybrid R² (cont,UW): {hy['unweighted_cont_r2_mean']:.3f} ± {hy['unweighted_cont_r2_std']:.3f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

