"""
Unified Microvariable Evaluation for Two-Arm Bandit Task

Combines discrete (behavioral state classification) and continuous (HGF belief regression)
linear decodability evaluation in a single script.

Original discrete version adapted from:
  https://github.com/akshey-kumar/comparison-algorithms/.../microvariable_evaluation.py
  Original author: Akshey Kumar
Adapted and extended by: Kerim Atak
"""

import sys
sys.path.append(r'../../../')
import numpy as np
import os
import json
import datetime
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from ncmcm.data_loaders.bandit_task import BanditTaskNeuroPixelsDataset
from ncmcm.bundlenet.utils import (
    prep_data, prep_data_lazy,
    timeseries_train_test_split_cv, timeseries_train_test_split_cv_lazy,
    make_hybrid_b,
)

# ===========================================================================
# EXPERIMENT CONFIGURATION — edit everything here
# ===========================================================================

# --- Dataset (BanditTaskNeuroPixelsDataset constructor params) --------------
DOWNSAMPLE_FS       = 20            # target sampling frequency in Hz; None = no downsampling
DOWNSAMPLE_METHOD   = 'count'       # 'binary' | 'count' | 'rate' | 'mean' | 'gaussian'
GOOD_NEURONS_ONLY   = True
NORMALIZE_METHOD    = 'minmax_global'  # None | 'minmax' | 'minmax_global'
STATE_TRANSITIONS   = None          # e.g. BanditTaskNeuroPixelsDataset.CHOOSING_TO_CORRECTNESS_TRANSITIONS
CHOOSING_STATE_MODE = 'side'        # 'side' | 'correctness'
GAUSSIAN_SIGMA_MS   = 25.0          # only used when DOWNSAMPLE_METHOD='gaussian'
RECOMPUTE_CACHE     = False

# --- HGF -------------------------------------------------------------------
USE_HGF             = True          # set False to skip HGF loading (disables RUN_CONTINUOUS)
HGF_MODEL           = 'binary2'     # substring matching HGF pkl filename
HGF_COLUMN          = 'x_1_expected_mean'   # 'x_1_expected_mean' | 'x_0_expected_mean'
HGF_BELIEF_RANGE    = None          # None = use KNOWN_HGF_RANGES; or explicit (lo, hi)

# --- Evaluation modes -------------------------------------------------------
RUN_DISCRETE        = False          # classification: behavioral state → cross-entropy decoder
RUN_CONTINUOUS      = False          # regression: HGF belief → MSE decoder (requires USE_HGF=True)
RUN_HYBRID          = True          # joint decoder: discrete + continuous (requires USE_HGF=True)
HYBRID_ALPHA        = 0.5           # α * CE_norm + (1-α) * MSE  (matches BunDLeNet default)

# --- Data pipeline ----------------------------------------------------------
WINDOW_SIZE         = 60            # sliding window length (timesteps)
NUM_OF_SPLITS       = 9             # number of time-series CV folds
USE_LAZY_LOADING    = True           # True = memory-efficient (required for large datasets)
                                    # False = eager numpy (fast but may OOM)
NUM_WORKERS         = 4             # DataLoader worker processes for prefetching (USE_LAZY_LOADING only)

# --- Decoder training -------------------------------------------------------
NUM_DECODER_RUNS    = 10            # independent decoder runs per fold
TRAIN_EPOCHS        = 100           # epochs per decoder run
BATCH_SIZE          = 512           # mini-batch size (only used when USE_LAZY_LOADING=True)
NUM_PERMUTATIONS    = 200           # permutation baseline runs (discrete: chance acc; continuous: chance R²)

# ===========================================================================
# Validation
# ===========================================================================

if RUN_CONTINUOUS and not USE_HGF:
    raise ValueError(
        "RUN_CONTINUOUS=True requires USE_HGF=True. "
        "Either set USE_HGF=True or disable RUN_CONTINUOUS."
    )
if RUN_HYBRID and not USE_HGF:
    raise ValueError(
        "RUN_HYBRID=True requires USE_HGF=True. "
        "Either set USE_HGF=True or disable RUN_HYBRID."
    )

# ===========================================================================
# CLI
# ===========================================================================

data_path   = sys.argv[1]
session_dir = os.path.basename(data_path.rstrip('/'))

# ===========================================================================
# Data loading
# ===========================================================================

print(f"Loading data from: {data_path}")

_hgf_kwargs = dict(
    hgf_model=HGF_MODEL,
    hgf_column=HGF_COLUMN,
    hgf_belief_range=HGF_BELIEF_RANGE,
) if USE_HGF else dict(hgf_model=None)

data = BanditTaskNeuroPixelsDataset(
    data_path=data_path,
    downsample_fs=DOWNSAMPLE_FS,
    downsample_method=DOWNSAMPLE_METHOD,
    good_neurons_only=GOOD_NEURONS_ONLY,
    normalize_method=NORMALIZE_METHOD,
    state_transitions=STATE_TRANSITIONS,
    choosing_state_mode=CHOOSING_STATE_MODE,
    gaussian_sigma_ms=GAUSSIAN_SIGMA_MS,
    recompute_cache=RECOMPUTE_CACHE,
    **_hgf_kwargs,
)

X        = data.x.toarray().T          # (T, N)
B        = data.b.toarray().flatten()  # (T,) — discrete behavioral states
B_belief = data.hgf_beliefs            # (T,) float32 or None

b_labels_dict = data.b_labels_dict
b_labels      = data.b_labels

print(f"Data shape: X={X.shape}, B={B.shape}")
print(f"Behavioral labels: {b_labels_dict}")
if B_belief is not None:
    print(f"HGF beliefs shape: {B_belief.shape}, "
          f"range=[{B_belief.min():.3f}, {B_belief.max():.3f}]")

# Hybrid label array: col 0 = discrete class index (float), col 1 = HGF belief
# Shape (T, 2) — matches BunDLeNet make_hybrid_b convention
B_hybrid = make_hybrid_b(B, B_belief) if (USE_HGF and B_belief is not None) else None

# ===========================================================================
# Output folder (results/twoArmBandit/microvariable_evaluation/<session>_<ts>/)
# ===========================================================================

_ts      = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
run_dir  = os.path.join('results', 'twoArmBandit', 'microvariable_evaluation',
                        f'{session_dir}_{_ts}')
for _sub in ('discrete', 'continuous', 'hybrid'):
    os.makedirs(os.path.join(run_dir, _sub), exist_ok=True)

_config = dict(
    # dataset
    data_path=data_path, session_dir=session_dir,
    downsample_fs=DOWNSAMPLE_FS, downsample_method=DOWNSAMPLE_METHOD,
    good_neurons_only=GOOD_NEURONS_ONLY, normalize_method=NORMALIZE_METHOD,
    state_transitions=str(STATE_TRANSITIONS), choosing_state_mode=CHOOSING_STATE_MODE,
    gaussian_sigma_ms=GAUSSIAN_SIGMA_MS, recompute_cache=RECOMPUTE_CACHE,
    # HGF
    use_hgf=USE_HGF, hgf_model=HGF_MODEL, hgf_column=HGF_COLUMN,
    hgf_belief_range=str(HGF_BELIEF_RANGE),
    # modes
    run_discrete=RUN_DISCRETE, run_continuous=RUN_CONTINUOUS,
    run_hybrid=RUN_HYBRID, hybrid_alpha=HYBRID_ALPHA,
    # pipeline
    window_size=WINDOW_SIZE, num_of_splits=NUM_OF_SPLITS,
    use_lazy_loading=USE_LAZY_LOADING, num_workers=NUM_WORKERS,
    # training
    num_decoder_runs=NUM_DECODER_RUNS, train_epochs=TRAIN_EPOCHS,
    batch_size=BATCH_SIZE, num_permutations=NUM_PERMUTATIONS,
    # data info
    n_timesteps=int(X.shape[0]), n_neurons=int(X.shape[1]),
    start_timestamp=_ts,
)
with open(os.path.join(run_dir, 'config.json'), 'w') as _f:
    json.dump(_config, _f, indent=2)
print(f"Run folder: {run_dir}")
print(f"Config written → {run_dir}/config.json")

# ===========================================================================
# Data pipeline helpers
# ===========================================================================

def make_cv_splits(label_array):
    """Return CV splits for a given label array using the configured pipeline."""
    if USE_LAZY_LOADING:
        X_, B_ = prep_data_lazy(X, label_array, win=WINDOW_SIZE)
        return timeseries_train_test_split_cv_lazy(X_, B_, NUM_OF_SPLITS), B_
    else:
        X_, B_ = prep_data(X, label_array, win=WINDOW_SIZE)
        return timeseries_train_test_split_cv(X_, B_, NUM_OF_SPLITS), B_


n_neurons = X.shape[1]
input_dim = n_neurons * WINDOW_SIZE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
_config['device'] = str(device)
with open(os.path.join(run_dir, 'config.json'), 'w') as _f:
    json.dump(_config, _f, indent=2)
print(f"Config written \u2192 {run_dir}/config.json")

# ===========================================================================
# Shared utilities
# ===========================================================================

class FoldDataset(Dataset):
    """Extract channel-1 windows on demand, flatten, return tensor.

    dtype_str: 'long'   for discrete labels (CrossEntropy)  — returns (x, long scalar)
               'float'  for continuous labels (MSE)         — returns (x, float [1])
               'hybrid' for joint labels (CE + MSE)         — returns (x, float [2])
                        col 0 = class index (float), col 1 = belief
    """
    def __init__(self, subset, b_labels, dtype_str='long'):
        self.subset    = subset
        self.dtype_str = dtype_str
        if dtype_str == 'long':
            self.b_labels = b_labels.astype(np.int64)
        else:
            self.b_labels = b_labels.astype(np.float32)

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x_paired = self.subset[idx]   # (2, win, N)
        x1 = x_paired[1].flatten()   # (win*N,)
        x_t = torch.from_numpy(x1.astype(np.float32))
        if self.dtype_str == 'long':
            return x_t, torch.tensor(self.b_labels[idx], dtype=torch.long)
        elif self.dtype_str == 'float':
            return x_t, torch.tensor([self.b_labels[idx]], dtype=torch.float32)
        else:  # hybrid
            return x_t, torch.tensor(self.b_labels[idx], dtype=torch.float32)  # (2,)


def make_loaders(x_fold, b_fold, dtype_str):
    """Build train/val DataLoaders (lazy) or tensors (eager)."""
    if USE_LAZY_LOADING:
        ds = FoldDataset(x_fold, b_fold, dtype_str)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0)
    else:
        # Eager: x_fold shape (T, 2, win, N); channel 1
        X1 = x_fold[:, 1, :, :]  # (T, win, N)
        X1_flat = X1.reshape(X1.shape[0], -1)  # (T, win*N)
        x_t = torch.FloatTensor(X1_flat)
        if dtype_str == 'long':
            b_t = torch.LongTensor(b_fold)
        elif dtype_str == 'float':
            b_t = torch.FloatTensor(b_fold).unsqueeze(1)
        else:  # hybrid: b_fold is (T, 2)
            b_t = torch.FloatTensor(b_fold)
        return list(zip(
            [x_t[i:i+BATCH_SIZE] for i in range(0, len(x_t), BATCH_SIZE)],
            [b_t[i:i+BATCH_SIZE] for i in range(0, len(b_t), BATCH_SIZE)],
        ))


def predict_all(model, loader, squeeze_pred=False, squeeze_true=False):
    """Run model over loader, return (preds, trues) as numpy arrays."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            preds.append((out.squeeze(1) if squeeze_pred else out.argmax(dim=1)).cpu().numpy())
            trues.append((yb.squeeze(1) if squeeze_true else yb).numpy())
    return np.concatenate(preds), np.concatenate(trues)


def predict_all_hybrid(model, loader, n_classes):
    """Run hybrid model over loader, return (disc_preds, cont_preds, disc_true, cont_true)."""
    model.eval()
    disc_preds, cont_preds, disc_true, cont_true = [], [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)                               # (batch, n_classes + 1)
            disc_preds.append(out[:, :n_classes].argmax(dim=1).cpu().numpy())
            cont_preds.append(out[:, n_classes].cpu().numpy())
            disc_true.append(yb[:, 0].long().numpy())
            cont_true.append(yb[:, 1].numpy())
    return (np.concatenate(disc_preds), np.concatenate(cont_preds),
            np.concatenate(disc_true),  np.concatenate(cont_true))


def fold_size(fold_split):
    """Return number of samples, regardless of lazy/eager."""
    if USE_LAZY_LOADING:
        return len(fold_split)
    else:
        return fold_split.shape[0]

# ===========================================================================
# DISCRETE EVALUATION
# ===========================================================================

def run_discrete_evaluation():
    print(f"\n{'#'*70}")
    print(f"### DISCRETE EVALUATION — BEHAVIORAL STATE CLASSIFICATION ###")
    print(f"{'#'*70}")

    # ── CV splits ────────────────────────────────────────────────────────────
    cv_splits, B_ = make_cv_splits(B)
    num_folds  = len(cv_splits)
    state_labels = np.unique(B_)
    num_states   = len(state_labels)
    print(f"Prepared {num_folds}-fold CV. {num_states} behavioral states.")

    # ── Label distribution ────────────────────────────────────────────────────
    train_label_counts_raw = defaultdict(int)
    val_label_counts_raw   = defaultdict(int)
    total_train = total_val = 0

    for x_tr, x_val, b_tr, b_val in cv_splits:
        for lbl, cnt in zip(*np.unique(b_tr, return_counts=True)):
            train_label_counts_raw[lbl] += cnt
        for lbl, cnt in zip(*np.unique(b_val, return_counts=True)):
            val_label_counts_raw[lbl] += cnt
        total_train += len(b_tr)
        total_val   += len(b_val)

    train_label_counts = {l: train_label_counts_raw[l] / num_folds for l in state_labels}
    val_label_counts   = {l: val_label_counts_raw[l]   / num_folds for l in state_labels}
    total_train_avg = total_train / num_folds
    total_val_avg   = total_val   / num_folds

    print(f"\nTotal samples per fold (avg): Train={total_train_avg:.1f}, Val={total_val_avg:.1f}")
    print(f"\n{'State':<20} {'Train':>8} {'Train%':>8} {'Val':>8} {'Val%':>8}")
    print("-" * 60)
    for lbl in sorted(state_labels):
        sn = b_labels_dict.get(lbl, f'State {lbl}')
        tc = train_label_counts.get(lbl, 0)
        vc = val_label_counts.get(lbl, 0)
        print(f"{sn:<20} {tc:>8.1f} {100*tc/total_train_avg:>7.1f}% "
              f"{vc:>8.1f} {100*vc/total_val_avg:>7.1f}%")

    imbalance = max(train_label_counts.values()) / min(v for v in train_label_counts.values() if v > 0)
    print(f"\nClass imbalance ratio: {imbalance:.1f}x")

    class_weights = {
        lbl: total_train / (num_states * train_label_counts_raw[lbl])
        if train_label_counts_raw[lbl] > 0 else 1.0
        for lbl in state_labels
    }
    weights_list          = [class_weights[l] for l in sorted(state_labels)]
    class_weights_tensor  = torch.FloatTensor(weights_list).to(device)

    # ── Output dir ───────────────────────────────────────────────────────────
    output_dir = os.path.join(run_dir, 'discrete')

    # ── Class-weights plot ────────────────────────────────────────────────────
    x_pos = np.arange(len(state_labels))
    fig_w, ax_w = plt.subplots(figsize=(10, 6))
    bars = ax_w.bar(x_pos, weights_list, alpha=0.8, edgecolor='black')
    mn, mx = min(weights_list), max(weights_list)
    for bar, w in zip(bars, weights_list):
        nrm = (w - mn) / (mx - mn) if mx != mn else 0.5
        bar.set_color(plt.cm.RdYlGn_r(nrm))
    ax_w.axhline(1.0, color='black', linestyle='--', linewidth=1.5, label='weight=1.0')
    ax_w.set_xticks(x_pos)
    ax_w.set_xticklabels([b_labels_dict.get(s, f'S{s}') for s in state_labels], rotation=45, ha='right')
    ax_w.set_ylabel('Class Weight'); ax_w.set_xlabel('Behavioral State')
    ax_w.set_title('Class Weights for Weighted Loss', fontweight='bold')
    ax_w.legend(); ax_w.grid(True, alpha=0.3, axis='y')
    for bar, w in zip(bars, weights_list):
        ax_w.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                  f'{w:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'class_weights_{session_dir}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'class_weights_{session_dir}.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved class weights plot → {output_dir}/")

    # ── Inner training function ───────────────────────────────────────────────
    def _train(use_weighted_loss, suffix):
        loss_type = "WEIGHTED" if use_weighted_loss else "UNWEIGHTED"
        print(f"\n{'#'*60}")
        print(f"### {loss_type} LOSS ###")
        print(f"{'#'*60}")

        val_acc_list, val_all_predictions, val_all_f1_scores, val_true_labels       = [], [], [], []
        train_acc_list, train_all_predictions, train_all_f1_scores, train_true_labels = [], [], [], []
        val_conf_sum   = np.zeros((num_states, num_states))
        train_conf_sum = np.zeros((num_states, num_states))
        total_runs = 0

        for fold_idx, (x_tr, x_val, b_tr, b_val) in enumerate(cv_splits):
            print(f"\nFold {fold_idx+1}/{num_folds}: train={fold_size(x_tr)}, val={fold_size(x_val)}")
            tr_loader  = make_loaders(x_tr,  b_tr,  'long')
            val_loader = make_loaders(x_val, b_val, 'long')

            for _ in tqdm(range(NUM_DECODER_RUNS), desc=f'Decoders {loss_type} fold {fold_idx+1}', leave=False):
                model = nn.Sequential(nn.Linear(input_dim, num_states)).to(device)
                opt   = optim.Adam(model.parameters(), lr=0.01)
                crit  = nn.CrossEntropyLoss(weight=class_weights_tensor if use_weighted_loss else None)

                for epoch in range(TRAIN_EPOCHS):
                    model.train()
                    for xb, yb in tr_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        opt.zero_grad()
                        crit(model(xb), yb).backward()
                        opt.step()

                val_pred,   val_true   = predict_all(model, val_loader,  squeeze_pred=False)
                train_pred, train_true = predict_all(model, tr_loader,   squeeze_pred=False)

                val_acc_list.append(accuracy_score(val_true, val_pred))
                val_all_predictions.append(val_pred)
                val_true_labels.append(val_true)
                val_all_f1_scores.append(f1_score(val_true, val_pred, average=None, labels=state_labels, zero_division=0))
                val_conf_sum += confusion_matrix(val_true, val_pred, labels=state_labels)

                train_acc_list.append(accuracy_score(train_true, train_pred))
                train_all_predictions.append(train_pred)
                train_true_labels.append(train_true)
                train_all_f1_scores.append(f1_score(train_true, train_pred, average=None, labels=state_labels, zero_division=0))
                train_conf_sum += confusion_matrix(train_true, train_pred, labels=state_labels)
                total_runs += 1

        val_acc_list        = np.array(val_acc_list)
        val_all_f1_scores   = np.array(val_all_f1_scores)
        train_acc_list      = np.array(train_acc_list)
        train_all_f1_scores = np.array(train_all_f1_scores)
        avg_conf_val   = val_conf_sum   / total_runs
        avg_conf_train = train_conf_sum / total_runs

        print(f"\n{'='*60}")
        print(f"{loss_type} — Val acc: {val_acc_list.mean():.3f} ± {val_acc_list.std():.3f}  |  "
              f"Train acc: {train_acc_list.mean():.3f} ± {train_acc_list.std():.3f}")
        print(f"Train-Val gap: {train_acc_list.mean() - val_acc_list.mean():.3f}")
        print(f"{'='*60}")

        return dict(
            suffix=suffix, loss_type=loss_type,
            val_acc_list=val_acc_list, val_all_predictions=np.array(val_all_predictions, dtype=object),
            val_all_f1_scores=val_all_f1_scores, val_true_labels=np.array(val_true_labels, dtype=object),
            train_acc_list=train_acc_list, train_all_predictions=np.array(train_all_predictions, dtype=object),
            train_all_f1_scores=train_all_f1_scores, train_true_labels=np.array(train_true_labels, dtype=object),
            avg_conf_matrix_val=avg_conf_val, avg_conf_matrix_train=avg_conf_train,
        )

    def _save(results):
        suffix    = results['suffix']
        loss_type = results['loss_type']
        val_acc   = results['val_acc_list']
        tr_acc    = results['train_acc_list']
        val_f1    = results['val_all_f1_scores']
        tr_f1     = results['train_all_f1_scores']
        avg_conf  = results['avg_conf_matrix_val']
        tr_conf   = results['avg_conf_matrix_train']

        np.savetxt(os.path.join(output_dir, f'acc_list_val_{session_dir}{suffix}.txt'),   val_acc)
        np.savetxt(os.path.join(output_dir, f'acc_list_train_{session_dir}{suffix}.txt'), tr_acc)
        np.save(os.path.join(output_dir, f'all_f1_scores_val_{session_dir}{suffix}.npy'),   val_f1)
        np.save(os.path.join(output_dir, f'all_f1_scores_train_{session_dir}{suffix}.npy'), tr_f1)

        val_f1_means = val_f1.mean(axis=0)
        tr_f1_means  = tr_f1.mean(axis=0)
        f1_gap       = tr_f1_means - val_f1_means

        # ── Comprehensive train/val comparison (8 panels) ────────────────────
        fig = plt.figure(figsize=(24, 16))
        fig.suptitle(f'{loss_type} Loss Results — {session_dir}', fontsize=18, fontweight='bold', y=1.01)

        # 0: label distribution
        ax0 = plt.subplot(2, 4, 1)
        w   = 0.35
        tc  = [train_label_counts.get(s, 0) for s in state_labels]
        vc  = [val_label_counts.get(s, 0)   for s in state_labels]
        ax0.bar(x_pos - w/2, tc, w, label='Train',      color='lightgreen', alpha=0.8)
        ax0.bar(x_pos + w/2, vc, w, label='Validation', color='skyblue',    alpha=0.8)
        ax0.set_xticks(x_pos); ax0.set_xticklabels([b_labels_dict.get(s, f'S{s}') for s in state_labels], rotation=45, ha='right')
        ax0.set_title('Label Distribution', fontweight='bold'); ax0.legend(); ax0.grid(True, alpha=0.3, axis='y')

        # 1: accuracy boxplot
        ax1 = plt.subplot(2, 4, 2)
        bp  = ax1.boxplot([tr_acc, val_acc], positions=[1, 2], widths=0.6, patch_artist=True, showmeans=True,
                          meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
        for patch, col in zip(bp['boxes'], ['lightgreen', 'skyblue']): patch.set_facecolor(col)
        ax1.set_xticks([1, 2]); ax1.set_xticklabels(['Train', 'Validation'])
        ax1.set_ylabel('Accuracy'); ax1.set_title(f'Accuracy\nGap={tr_acc.mean()-val_acc.mean():.3f}', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y'); ax1.set_ylim([0, 1.05])

        # 2: per-state F1 boxplots
        ax2 = plt.subplot(2, 4, 3)
        f1_data = []
        for si, sl in enumerate(state_labels):
            for ri in range(val_f1.shape[0]):
                f1_data += [
                    {'State': b_labels_dict.get(sl, f'S{sl}'), 'F1': val_f1[ri, si],   'Set': 'Validation'},
                    {'State': b_labels_dict.get(sl, f'S{sl}'), 'F1': tr_f1[ri,  si],   'Set': 'Train'},
                ]
        sns.boxplot(data=pd.DataFrame(f1_data), x='State', y='F1', hue='Set', ax=ax2,
                    palette={'Train': 'lightgreen', 'Validation': 'skyblue'})
        ax2.set_title('Per-State F1', fontweight='bold'); ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')

        # 3: confusion matrix (val)
        ax3 = plt.subplot(2, 4, 4)
        sns.heatmap(avg_conf, annot=True, fmt='.1f', cmap='Blues', ax=ax3,
                    xticklabels=[b_labels_dict.get(s, f'S{s}') for s in state_labels],
                    yticklabels=[b_labels_dict.get(s, f'S{s}') for s in state_labels])
        ax3.set_title('Confusion Matrix (Val)', fontweight='bold')

        # 4: confusion matrix (train)
        ax4 = plt.subplot(2, 4, 5)
        sns.heatmap(tr_conf, annot=True, fmt='.1f', cmap='Greens', ax=ax4,
                    xticklabels=[b_labels_dict.get(s, f'S{s}') for s in state_labels],
                    yticklabels=[b_labels_dict.get(s, f'S{s}') for s in state_labels])
        ax4.set_title('Confusion Matrix (Train)', fontweight='bold')

        # 5: F1 bar
        ax5 = plt.subplot(2, 4, 6)
        ax5.bar(x_pos - w/2, tr_f1_means,  w, label='Train',      color='lightgreen', alpha=0.8)
        ax5.bar(x_pos + w/2, val_f1_means, w, label='Validation',  color='skyblue',    alpha=0.8)
        ax5.set_xticks(x_pos); ax5.set_xticklabels([b_labels_dict.get(s, f'S{s}') for s in state_labels], rotation=45, ha='right')
        ax5.set_title('F1 Score', fontweight='bold'); ax5.legend(); ax5.grid(True, alpha=0.3, axis='y'); ax5.set_ylim([0, 1.05])

        # 6: F1 gap
        ax6 = plt.subplot(2, 4, 7)
        ax6.bar(x_pos, f1_gap, color=['green' if g > 0 else 'red' for g in f1_gap], alpha=0.7)
        ax6.axhline(0, color='black', linewidth=0.5)
        ax6.axhline(f1_gap.mean(), color='red', linestyle='--', label=f'Mean: {f1_gap.mean():.3f}')
        ax6.set_xticks(x_pos); ax6.set_xticklabels([b_labels_dict.get(s, f'S{s}') for s in state_labels], rotation=45, ha='right')
        ax6.set_title('F1 Gap (Train−Val)', fontweight='bold'); ax6.legend(); ax6.grid(True, alpha=0.3, axis='y')

        # 7: F1 vs sample count
        ax7 = plt.subplot(2, 4, 8)
        vc_arr = np.array([val_label_counts.get(s, 0) for s in state_labels])
        ax7.scatter(vc_arr, val_f1_means, s=100, c='skyblue', edgecolors='blue', alpha=0.8)
        for i, (x_, y_) in enumerate(zip(vc_arr, val_f1_means)):
            ax7.annotate(b_labels_dict.get(state_labels[i], f'S{state_labels[i]}'), (x_, y_), xytext=(5,5), textcoords='offset points', fontsize=9)
        ax7.set_xlabel('Val sample count'); ax7.set_ylabel('F1'); ax7.set_title('F1 vs Sample Count', fontweight='bold'); ax7.grid(True, alpha=0.3)
        corr = np.corrcoef(vc_arr, val_f1_means)[0, 1]
        ax7.text(0.05, 0.95, f'r={corr:.3f}', transform=ax7.transAxes, va='top', fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'train_validation_comparison_{session_dir}{suffix}.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f'train_validation_comparison_{session_dir}{suffix}.pdf'), bbox_inches='tight')
        plt.close()
        print(f"  Saved train/validation comparison ({loss_type})")

        # ── Normalized confusion + precision/recall (detailed) ────────────────
        fig_d, axs_d = plt.subplots(2, 3, figsize=(20, 12))
        fig_d.suptitle(f'{loss_type} Loss — Detailed Analysis', fontsize=16, fontweight='bold', y=1.01)

        ax = axs_d[0, 0]; sns.boxplot(y=val_acc, ax=ax, color='skyblue')
        ax.axhline(val_acc.mean(), color='red', linestyle='--', label=f'Mean: {val_acc.mean():.3f}')
        ax.set_ylabel('Accuracy'); ax.set_title('Val Accuracy Distribution', fontweight='bold'); ax.legend(); ax.grid(True, alpha=0.3)

        ax = axs_d[0, 1]
        f1_df_val = pd.DataFrame([{'State': b_labels_dict.get(sl, f'S{sl}'), 'F1': val_f1[ri, si]}
                                   for si, sl in enumerate(state_labels) for ri in range(val_f1.shape[0])])
        sns.boxplot(data=f1_df_val, x='State', y='F1', ax=ax, palette='Set2')
        ax.set_title('Per-State F1 (Val)', fontweight='bold'); ax.tick_params(axis='x', rotation=45); ax.grid(True, alpha=0.3, axis='y')

        ax = axs_d[0, 2]; sns.heatmap(avg_conf, annot=True, fmt='.1f', cmap='Blues', ax=ax,
                                       xticklabels=[b_labels_dict.get(s, f'S{s}') for s in state_labels],
                                       yticklabels=[b_labels_dict.get(s, f'S{s}') for s in state_labels])
        ax.set_title('Average Confusion Matrix', fontweight='bold')

        ax = axs_d[1, 0]
        norm_conf = avg_conf / avg_conf.sum(axis=1, keepdims=True) * 100
        sns.heatmap(norm_conf, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax,
                    xticklabels=[b_labels_dict.get(s, f'S{s}') for s in state_labels],
                    yticklabels=[b_labels_dict.get(s, f'S{s}') for s in state_labels],
                    cbar_kws={'label': '%'})
        ax.set_title('Normalized Confusion Matrix (%)', fontweight='bold')

        ax = axs_d[1, 1]
        avg_prec, avg_rec, avg_f1_per = [], [], []
        for si, sl in enumerate(state_labels):
            precs, recs = [], []
            for pred, true in zip(results['val_all_predictions'], results['val_true_labels']):
                p, r, _, _ = precision_recall_fscore_support(
                    np.asarray(true, dtype=np.int64).ravel(),
                    np.asarray(pred, dtype=np.int64).ravel(),
                    labels=state_labels, zero_division=0, average=None)
                precs.append(p[si]); recs.append(r[si])
            avg_prec.append(np.mean(precs)); avg_rec.append(np.mean(recs))
            avg_f1_per.append(val_f1_means[si])
        w_ = 0.25
        ax.bar(x_pos - w_, avg_prec,  w_, label='Precision', alpha=0.8)
        ax.bar(x_pos,      avg_rec,   w_, label='Recall',    alpha=0.8)
        ax.bar(x_pos + w_, avg_f1_per,w_, label='F1',        alpha=0.8)
        ax.set_xticks(x_pos); ax.set_xticklabels([b_labels_dict.get(s, f'S{s}') for s in state_labels], rotation=45, ha='right')
        ax.set_title('Precision / Recall / F1 (Val)', fontweight='bold'); ax.legend(); ax.grid(True, alpha=0.3, axis='y'); ax.set_ylim([0, 1.05])

        ax = axs_d[1, 2]
        ax.errorbar(state_labels, val_f1_means, yerr=val_f1.std(axis=0), fmt='o-', capsize=5, capthick=2, markersize=8, linewidth=2)
        ax.set_title('F1 per State (mean ± std)', fontweight='bold'); ax.grid(True, alpha=0.3); ax.set_ylim([0, 1.05])

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'detailed_analysis_{session_dir}{suffix}.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f'detailed_analysis_{session_dir}{suffix}.pdf'), bbox_inches='tight')
        plt.close()
        print(f"  Saved detailed analysis ({loss_type})")

        return output_dir

    # ── Run both loss variants ───────────────────────────────────────────────
    res_uw = _train(use_weighted_loss=False, suffix='_unweighted')
    _save(res_uw)
    res_w  = _train(use_weighted_loss=True,  suffix='_weighted')
    _save(res_w)

    # ── Weighted vs unweighted comparison ────────────────────────────────────
    print("\nGenerating weighted vs unweighted comparison...")
    fig_c = plt.figure(figsize=(20, 12))
    fig_c.suptitle('Weighted vs Unweighted Loss Comparison', fontsize=18, fontweight='bold')

    uw_f1 = res_uw['val_all_f1_scores'].mean(axis=0)
    w_f1  = res_w['val_all_f1_scores'].mean(axis=0)
    f1_imp = w_f1 - uw_f1

    ax1c = plt.subplot(2, 3, 1)
    bp = ax1c.boxplot([res_uw['val_acc_list'], res_w['val_acc_list']], positions=[1, 2], widths=0.6,
                      patch_artist=True, showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
    for p, c in zip(bp['boxes'], ['lightcoral', 'lightgreen']): p.set_facecolor(c)
    ax1c.set_xticks([1, 2]); ax1c.set_xticklabels(['Unweighted', 'Weighted'])
    ax1c.set_ylabel('Val Accuracy'); ax1c.set_title('Val Accuracy Comparison', fontweight='bold')
    ax1c.grid(True, alpha=0.3, axis='y'); ax1c.set_ylim([0, 1.05])

    ax2c = plt.subplot(2, 3, 2)
    ax2c.bar(x_pos - 0.175, uw_f1, 0.35, label='Unweighted', color='lightcoral', alpha=0.8)
    ax2c.bar(x_pos + 0.175, w_f1,  0.35, label='Weighted',   color='lightgreen', alpha=0.8)
    ax2c.set_xticks(x_pos); ax2c.set_xticklabels([b_labels_dict.get(s, f'S{s}') for s in state_labels], rotation=45, ha='right')
    ax2c.set_title('Per-State Val F1', fontweight='bold'); ax2c.legend(); ax2c.grid(True, alpha=0.3, axis='y'); ax2c.set_ylim([0, 1.05])

    ax3c = plt.subplot(2, 3, 3)
    ax3c.bar(x_pos, f1_imp, color=['green' if v > 0 else 'red' for v in f1_imp], alpha=0.7)
    ax3c.axhline(0, color='black', linewidth=0.5)
    ax3c.axhline(f1_imp.mean(), color='blue', linestyle='--', label=f'Mean: {f1_imp.mean():.3f}')
    ax3c.set_xticks(x_pos); ax3c.set_xticklabels([b_labels_dict.get(s, f'S{s}') for s in state_labels], rotation=45, ha='right')
    ax3c.set_title('F1 Improvement (W − UW)', fontweight='bold'); ax3c.legend(); ax3c.grid(True, alpha=0.3, axis='y')

    ax4c = plt.subplot(2, 3, 4)
    tc_arr = np.array([train_label_counts.get(s, 0) for s in state_labels])
    ax4c.scatter(tc_arr, f1_imp, s=100, c='purple', edgecolors='black', alpha=0.8)
    for i, (x_, y_) in enumerate(zip(tc_arr, f1_imp)):
        ax4c.annotate(b_labels_dict.get(state_labels[i], f'S{state_labels[i]}'), (x_, y_), xytext=(5,5), textcoords='offset points', fontsize=9)
    ax4c.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax4c.set_xlabel('Train sample count'); ax4c.set_ylabel('F1 improvement')
    ax4c.set_title('F1 Improvement vs Frequency', fontweight='bold'); ax4c.grid(True, alpha=0.3)
    corr_c = np.corrcoef(tc_arr, f1_imp)[0, 1]
    ax4c.text(0.05, 0.95, f'r={corr_c:.3f}', transform=ax4c.transAxes, va='top', fontsize=11,
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax5c = plt.subplot(2, 3, 5)
    cmp_data = []
    for si, sl in enumerate(state_labels):
        sn = b_labels_dict.get(sl, f'S{sl}')
        for ri in range(res_uw['val_all_f1_scores'].shape[0]):
            cmp_data += [
                {'State': sn, 'F1': res_uw['val_all_f1_scores'][ri, si], 'Loss': 'Unweighted'},
                {'State': sn, 'F1': res_w['val_all_f1_scores'][ri,  si], 'Loss': 'Weighted'},
            ]
    sns.boxplot(data=pd.DataFrame(cmp_data), x='State', y='F1', hue='Loss', ax=ax5c,
                palette={'Unweighted': 'lightcoral', 'Weighted': 'lightgreen'})
    ax5c.set_title('Per-State F1 Distribution Comparison', fontweight='bold')
    ax5c.tick_params(axis='x', rotation=45); ax5c.grid(True, alpha=0.3, axis='y')

    ax6c = plt.subplot(2, 3, 6); ax6c.axis('off')
    summary_t = (
        f"SUMMARY\n{'='*40}\n\n"
        f"Val Accuracy\n"
        f"  Unweighted: {res_uw['val_acc_list'].mean():.3f} ± {res_uw['val_acc_list'].std():.3f}\n"
        f"  Weighted:   {res_w['val_acc_list'].mean():.3f}  ± {res_w['val_acc_list'].std():.3f}\n"
        f"  Diff:       {res_w['val_acc_list'].mean() - res_uw['val_acc_list'].mean():+.3f}\n\n"
        f"Macro F1\n"
        f"  Unweighted: {uw_f1.mean():.3f}\n"
        f"  Weighted:   {w_f1.mean():.3f}\n"
        f"  Diff:       {w_f1.mean() - uw_f1.mean():+.3f}\n"
    )
    ax6c.text(0.05, 0.95, summary_t, transform=ax6c.transAxes, fontsize=11,
              va='top', family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'weighted_vs_unweighted_comparison_{session_dir}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'weighted_vs_unweighted_comparison_{session_dir}.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved weighted vs unweighted comparison.")

    # ── Chance accuracy ───────────────────────────────────────────────────────
    print(f"\nEstimating chance accuracy ({NUM_PERMUTATIONS} permutations)...")
    all_val_labels_chance = np.concatenate([split[3] for split in cv_splits])
    chance_acc = np.array([
        accuracy_score(np.random.choice(all_val_labels_chance, size=all_val_labels_chance.shape),
                       all_val_labels_chance)
        for _ in tqdm(range(NUM_PERMUTATIONS), desc='Chance', leave=False)
    ])
    print(f"Chance accuracy: {chance_acc.mean():.3f} ± {chance_acc.std():.3f}")
    np.savetxt(os.path.join(output_dir, f'acc_list_chance_{session_dir}.txt'), chance_acc)

    fig_ch, ax_ch = plt.subplots(figsize=(12, 6))
    bp_ch = ax_ch.boxplot(
        [res_uw['val_acc_list'], res_w['val_acc_list'], chance_acc],
        positions=[1, 2, 3], widths=0.6, patch_artist=True, showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
    for p, c in zip(bp_ch['boxes'], ['lightcoral', 'lightgreen', 'lightgray']): p.set_facecolor(c)
    ax_ch.set_xticks([1, 2, 3]); ax_ch.set_xticklabels(['Unweighted', 'Weighted', 'Chance'])
    ax_ch.set_ylabel('Accuracy', fontsize=14); ax_ch.set_title('Decoder vs Chance', fontsize=16, fontweight='bold')
    ax_ch.grid(True, alpha=0.3, axis='y'); ax_ch.set_ylim([0, 1.0])
    t_uw, p_uw = stats.ttest_ind(res_uw['val_acc_list'], chance_acc)
    t_w,  p_w  = stats.ttest_ind(res_w['val_acc_list'],  chance_acc)
    ax_ch.text(0.5, 0.95, f'UW vs Chance: t={t_uw:.2f} p={p_uw:.2e}\nW vs Chance: t={t_w:.2f} p={p_w:.2e}',
               transform=ax_ch.transAxes, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'decoder_vs_chance_{session_dir}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'decoder_vs_chance_{session_dir}.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved decoder vs chance comparison.")

    print(f"\nDiscrete evaluation done. Results → {output_dir}/")
    return {
        'unweighted_val_acc': float(res_uw['val_acc_list'].mean()),
        'unweighted_val_acc_std': float(res_uw['val_acc_list'].std()),
        'weighted_val_acc': float(res_w['val_acc_list'].mean()),
        'weighted_val_acc_std': float(res_w['val_acc_list'].std()),
        'chance_acc': float(chance_acc.mean()),
        't_weighted_vs_chance': float(t_w), 'p_weighted_vs_chance': float(p_w),
    }


# ===========================================================================
# CONTINUOUS EVALUATION
# ===========================================================================

def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float32).ravel()
    mse  = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    pearson_r, _ = stats.pearsonr(y_true, y_pred)
    return dict(mse=mse, rmse=rmse, r2=r2, pearson_r=float(pearson_r))


def run_continuous_evaluation():
    print(f"\n{'#'*70}")
    print(f"### CONTINUOUS EVALUATION — HGF BELIEF REGRESSION ###")
    print(f"{'#'*70}")

    cv_splits, _ = make_cv_splits(B_belief)
    num_folds = len(cv_splits)
    print(f"Prepared {num_folds}-fold CV.")

    output_dir = os.path.join(run_dir, 'continuous')

    val_metrics_list   = []
    train_metrics_list = []
    val_preds_per_fold  = []
    val_true_per_fold   = []
    train_preds_per_fold = []
    train_true_per_fold  = []

    for fold_idx, (x_tr, x_val, b_tr, b_val) in enumerate(cv_splits):
        print(f"\nFold {fold_idx+1}/{num_folds}: train={fold_size(x_tr)}, val={fold_size(x_val)}")

        tr_loader  = make_loaders(x_tr,  b_tr,  'float')
        val_loader = make_loaders(x_val, b_val, 'float')

        fold_val_m, fold_tr_m = [], []
        best_r2 = -np.inf

        for _ in tqdm(range(NUM_DECODER_RUNS), desc=f'Decoders fold {fold_idx+1}', leave=False):
            model = nn.Linear(input_dim, 1).to(device)
            opt   = optim.Adam(model.parameters(), lr=0.01)
            crit  = nn.MSELoss()

            for epoch in range(TRAIN_EPOCHS):
                model.train()
                for xb, yb in tr_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    crit(model(xb), yb).backward()
                    opt.step()

            val_pred,   val_true   = predict_all(model, val_loader,  squeeze_pred=True, squeeze_true=True)
            train_pred, train_true = predict_all(model, tr_loader,   squeeze_pred=True, squeeze_true=True)

            vm = regression_metrics(val_true,   val_pred)
            tm = regression_metrics(train_true, train_pred)
            fold_val_m.append(vm); fold_tr_m.append(tm)

            if vm['r2'] > best_r2:
                best_r2 = vm['r2']
                best_vp, best_vt = val_pred,   val_true
                best_tp, best_tt = train_pred, train_true

        val_metrics_list.extend(fold_val_m)
        train_metrics_list.extend(fold_tr_m)
        val_preds_per_fold.append(best_vp);   val_true_per_fold.append(best_vt)
        train_preds_per_fold.append(best_tp); train_true_per_fold.append(best_tt)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    def _ms(key, lst):
        v = np.array([m[key] for m in lst]); return v.mean(), v.std(), v

    val_r2_mean,   val_r2_std,   val_r2_all   = _ms('r2',        val_metrics_list)
    val_pr_mean,   val_pr_std,   val_pr_all   = _ms('pearson_r', val_metrics_list)
    val_mse_mean,  val_mse_std,  val_mse_all  = _ms('mse',       val_metrics_list)
    val_rmse_mean, val_rmse_std, val_rmse_all = _ms('rmse',      val_metrics_list)
    tr_r2_mean,    tr_r2_std,    tr_r2_all    = _ms('r2',        train_metrics_list)
    tr_pr_mean,    tr_pr_std,    tr_pr_all    = _ms('pearson_r', train_metrics_list)

    print(f"\n{'='*60}")
    print(f"VALIDATION  R²={val_r2_mean:.4f}±{val_r2_std:.4f}  r={val_pr_mean:.4f}±{val_pr_std:.4f}  MSE={val_mse_mean:.6f}")
    print(f"TRAIN       R²={tr_r2_mean:.4f}±{tr_r2_std:.4f}  r={tr_pr_mean:.4f}±{tr_pr_std:.4f}")
    print(f"ΔR²={tr_r2_mean - val_r2_mean:+.4f}  ΔPearson={tr_pr_mean - val_pr_mean:+.4f}")
    print(f"{'='*60}")

    # ── Permutation baseline ──────────────────────────────────────────────────
    print(f"\nPermutation baseline ({NUM_PERMUTATIONS} permutations)...")
    all_val_beliefs = np.concatenate(val_true_per_fold)
    perm_r2_list = []
    for _ in tqdm(range(NUM_PERMUTATIONS), desc='Permutations', leave=False):
        perm = np.random.permutation(all_val_beliefs)
        ss_res = np.sum((all_val_beliefs - perm) ** 2)
        ss_tot = np.sum((all_val_beliefs - all_val_beliefs.mean()) ** 2)
        perm_r2_list.append(1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan'))
    perm_r2 = np.array(perm_r2_list)
    t_stat, p_value = stats.ttest_ind(val_r2_all, perm_r2)
    print(f"Permutation R²: {perm_r2.mean():.4f} ± {perm_r2.std():.4f}")
    print(f"t-test: t={t_stat:.3f}, p={p_value:.2e}")

    # ── Save raw ──────────────────────────────────────────────────────────────
    np.savetxt(os.path.join(output_dir, f'r2_val_{session_dir}.txt'),          val_r2_all)
    np.savetxt(os.path.join(output_dir, f'r2_train_{session_dir}.txt'),        tr_r2_all)
    np.savetxt(os.path.join(output_dir, f'pearson_r_val_{session_dir}.txt'),   val_pr_all)
    np.savetxt(os.path.join(output_dir, f'pearson_r_train_{session_dir}.txt'), tr_pr_all)
    np.savetxt(os.path.join(output_dir, f'mse_val_{session_dir}.txt'),         val_mse_all)
    np.savetxt(os.path.join(output_dir, f'r2_permutation_{session_dir}.txt'),  perm_r2)

    # ── Figure 1: Main summary (2×3) ──────────────────────────────────────────
    print("\nGenerating visualizations...")
    fig1, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig1.suptitle(f'HGF Belief Linear Decoder — {session_dir}', fontsize=16, fontweight='bold')

    ax = axes[0, 0]
    bp = ax.boxplot([tr_r2_all, val_r2_all, perm_r2], positions=[1, 2, 3], widths=0.6,
                    patch_artist=True, showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
    for p, c in zip(bp['boxes'], ['lightgreen', 'skyblue', 'lightgray']): p.set_facecolor(c)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(['Train', 'Validation', 'Permutation'])
    ax.set_ylabel('R²'); ax.set_title(f'R² Distribution\nVal={val_r2_mean:.3f}±{val_r2_std:.3f}  p={p_value:.2e}', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y'); ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    ax = axes[0, 1]
    bp2 = ax.boxplot([tr_pr_all, val_pr_all], positions=[1, 2], widths=0.6, patch_artist=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
    for p, c in zip(bp2['boxes'], ['lightgreen', 'skyblue']): p.set_facecolor(c)
    ax.set_xticks([1, 2]); ax.set_xticklabels(['Train', 'Validation'])
    ax.set_ylabel('Pearson r'); ax.set_title(f'Pearson r\nVal={val_pr_mean:.3f}±{val_pr_std:.3f}', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y'); ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    ax = axes[0, 2]
    ax.scatter(tr_r2_all, val_r2_all, alpha=0.5, s=30, color='steelblue')
    lims = [min(tr_r2_all.min(), val_r2_all.min()) - 0.05, max(tr_r2_all.max(), val_r2_all.max()) + 0.05]
    ax.plot(lims, lims, 'k--', linewidth=1, label='y=x'); ax.set_xlabel('Train R²'); ax.set_ylabel('Val R²')
    ax.set_title('Overfitting: Train vs Val R²', fontweight='bold'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    all_val_preds = np.concatenate(val_preds_per_fold)
    ax.scatter(all_val_beliefs, all_val_preds, alpha=0.2, s=5, color='steelblue', rasterized=True)
    lim_lo = min(all_val_beliefs.min(), all_val_preds.min())
    lim_hi = max(all_val_beliefs.max(), all_val_preds.max())
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], 'r--', linewidth=1.5, label='Perfect')
    ax.set_xlabel('True HGF Belief'); ax.set_ylabel('Predicted HGF Belief')
    ax.set_title('True vs Predicted (all folds, best run)', fontweight='bold'); ax.legend(); ax.grid(True, alpha=0.3)
    r_all, _ = stats.pearsonr(all_val_beliefs, all_val_preds)
    ax.text(0.05, 0.93, f'r={r_all:.3f}', transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    ax = axes[1, 1]
    sns.histplot(val_mse_all, ax=ax, color='skyblue', kde=True, bins=20)
    ax.axvline(val_mse_mean, color='red', linestyle='--', label=f'Mean: {val_mse_mean:.5f}')
    ax.set_xlabel('MSE'); ax.set_ylabel('Count'); ax.set_title('Val MSE Distribution', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]; ax.axis('off')
    summary = (
        f"SUMMARY — {session_dir}\n{'='*36}\n\n"
        f"Validation\n"
        f"  R²:        {val_r2_mean:.4f} ± {val_r2_std:.4f}\n"
        f"  Pearson r: {val_pr_mean:.4f} ± {val_pr_std:.4f}\n"
        f"  MSE:       {val_mse_mean:.6f} ± {val_mse_std:.6f}\n"
        f"  RMSE:      {val_rmse_mean:.6f} ± {val_rmse_std:.6f}\n\n"
        f"Train\n"
        f"  R²:        {tr_r2_mean:.4f} ± {tr_r2_std:.4f}\n"
        f"  Pearson r: {tr_pr_mean:.4f} ± {tr_pr_std:.4f}\n\n"
        f"Overfitting\n"
        f"  ΔR²:       {tr_r2_mean - val_r2_mean:+.4f}\n"
        f"  ΔPearson:  {tr_pr_mean - val_pr_mean:+.4f}\n\n"
        f"Permutation\n"
        f"  R²:        {perm_r2.mean():.4f} ± {perm_r2.std():.4f}\n"
        f"  t={t_stat:.3f}, p={p_value:.2e}\n"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10, va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'summary_{session_dir}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'summary_{session_dir}.pdf'), bbox_inches='tight')
    plt.close(); print("  Saved summary figure.")

    # ── Figure 2: Time-series overlay ─────────────────────────────────────────
    n_show = min(3, num_folds)
    fig2, axes2 = plt.subplots(n_show, 1, figsize=(18, 4 * n_show), sharex=False)
    if n_show == 1: axes2 = [axes2]
    fig2.suptitle(f'Time-Series Overlay: True vs Predicted HGF Belief\n{session_dir}', fontsize=14, fontweight='bold')
    for fi in range(n_show):
        ax = axes2[fi]
        t  = np.arange(len(val_true_per_fold[fi]))
        ax.plot(t, val_true_per_fold[fi],  color='steelblue', linewidth=1.2, label='True',      alpha=0.9)
        ax.plot(t, val_preds_per_fold[fi], color='tomato',    linewidth=1.0, linestyle='--', label='Predicted', alpha=0.85)
        ax.set_ylabel('HGF Belief', fontsize=11)
        ax.set_title(f'Fold {fi+1}  (R²={val_r2_all[fi * NUM_DECODER_RUNS]:.3f})', fontsize=11)
        ax.legend(loc='upper right', fontsize=9); ax.grid(True, alpha=0.3)
    axes2[-1].set_xlabel('Time (samples)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'timeseries_{session_dir}.png'), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'timeseries_{session_dir}.pdf'), bbox_inches='tight')
    plt.close(); print("  Saved time-series overlay.")

    # ── Figure 3: Per-fold R² bar chart ───────────────────────────────────────
    fold_val_r2_m  = np.array([np.mean([val_metrics_list[f*NUM_DECODER_RUNS+r]['r2']   for r in range(NUM_DECODER_RUNS)]) for f in range(num_folds)])
    fold_val_r2_s  = np.array([np.std( [val_metrics_list[f*NUM_DECODER_RUNS+r]['r2']   for r in range(NUM_DECODER_RUNS)]) for f in range(num_folds)])
    fold_tr_r2_m   = np.array([np.mean([train_metrics_list[f*NUM_DECODER_RUNS+r]['r2'] for r in range(NUM_DECODER_RUNS)]) for f in range(num_folds)])

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    xf = np.arange(num_folds)
    ax3.bar(xf - 0.175, fold_tr_r2_m, 0.35, label='Train R²',      color='lightgreen', alpha=0.85)
    ax3.bar(xf + 0.175, fold_val_r2_m, 0.35, yerr=fold_val_r2_s, capsize=4, label='Val R²', color='skyblue', alpha=0.85)
    ax3.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax3.axhline(perm_r2.mean(), color='gray', linestyle=':', linewidth=1.2, label=f'Permutation ({perm_r2.mean():.3f})')
    ax3.set_xticks(xf); ax3.set_xticklabels([f'F{i+1}' for i in range(num_folds)])
    ax3.set_xlabel('Fold'); ax3.set_ylabel('R²'); ax3.set_title(f'R² per Fold — {session_dir}', fontsize=13, fontweight='bold')
    ax3.legend(); ax3.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'per_fold_r2_{session_dir}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'per_fold_r2_{session_dir}.pdf'), bbox_inches='tight')
    plt.close(); print("  Saved per-fold R² figure.")

    print(f"\nContinuous evaluation done. Results → {output_dir}/")
    return {
        'val_r2': float(val_r2_mean), 'val_r2_std': float(val_r2_std),
        'val_pearson_r': float(val_pr_mean), 'val_pearson_r_std': float(val_pr_std),
        'val_mse': float(val_mse_mean),
        'train_r2': float(tr_r2_mean),
        'perm_r2': float(perm_r2.mean()),
        't_stat': float(t_stat), 'p_value': float(p_value),
    }


# ===========================================================================
# HYBRID EVALUATION
# ===========================================================================

def run_hybrid_evaluation():
    print(f"\n{'#'*70}")
    print(f"### HYBRID EVALUATION — JOINT DISCRETE + CONTINUOUS DECODER ###")
    print(f"{'#'*70}")
    print(f"Alpha={HYBRID_ALPHA}  (α·CE_norm + (1-α)·MSE, matching BunDLeNet)")

    import math

    cv_splits, B_hyb_ = make_cv_splits(B_hybrid)
    num_folds  = len(cv_splits)
    state_labels = np.unique(B_hyb_[:, 0].astype(np.int64))
    num_states   = len(state_labels)
    ce_norm      = math.log(num_states)          # normalise CE to ~[0,1] — same as BunDLeNet
    output_dim   = num_states + 1                # logits + 1 belief output
    print(f"Prepared {num_folds}-fold CV. {num_states} states, output_dim={output_dim}.")

    output_dir = os.path.join(run_dir, 'hybrid')

    val_acc_list,   train_acc_list   = [], []
    val_r2_list,    train_r2_list    = [], []
    val_pr_list,    train_pr_list    = [], []
    val_preds_per_fold,  val_true_per_fold   = [], []
    val_cpreds_per_fold, val_ctrue_per_fold  = [], []

    ce_loss_fn  = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()

    for fold_idx, (x_tr, x_val, b_tr, b_val) in enumerate(cv_splits):
        print(f"\nFold {fold_idx+1}/{num_folds}: train={fold_size(x_tr)}, val={fold_size(x_val)}")

        tr_loader  = make_loaders(x_tr,  b_tr,  'hybrid')
        val_loader = make_loaders(x_val, b_val, 'hybrid')

        fold_best_acc = -np.inf
        fold_best_r2  = -np.inf

        fold_val_acc, fold_val_r2, fold_val_pr     = [], [], []
        fold_tr_acc,  fold_tr_r2,  fold_tr_pr      = [], [], []

        for _ in tqdm(range(NUM_DECODER_RUNS), desc=f'Decoders hybrid fold {fold_idx+1}', leave=False):
            model = nn.Linear(input_dim, output_dim).to(device)
            opt   = optim.Adam(model.parameters(), lr=0.01)

            for epoch in range(TRAIN_EPOCHS):
                model.train()
                for xb, yb in tr_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out      = model(xb)                       # (batch, num_states+1)
                    logits   = out[:, :num_states]             # (batch, num_states)
                    cont_out = out[:, num_states:num_states+1] # (batch, 1)
                    disc_lbl = yb[:, 0].long()                 # (batch,)
                    cont_lbl = yb[:, 1:2]                      # (batch, 1)
                    loss = (HYBRID_ALPHA * ce_loss_fn(logits, disc_lbl) / ce_norm
                            + (1 - HYBRID_ALPHA) * mse_loss_fn(cont_out, cont_lbl))
                    opt.zero_grad(); loss.backward(); opt.step()

            dp, cp, dt, ct = predict_all_hybrid(model, val_loader, num_states)
            acc = accuracy_score(dt, dp)
            r2  = regression_metrics(ct, cp)['r2']
            pr  = regression_metrics(ct, cp)['pearson_r']
            fold_val_acc.append(acc); fold_val_r2.append(r2); fold_val_pr.append(pr)

            dp_tr, cp_tr, dt_tr, ct_tr = predict_all_hybrid(model, tr_loader, num_states)
            fold_tr_acc.append(accuracy_score(dt_tr, dp_tr))
            fold_tr_r2.append(regression_metrics(ct_tr, cp_tr)['r2'])
            fold_tr_pr.append(regression_metrics(ct_tr, cp_tr)['pearson_r'])

            if r2 > fold_best_r2:
                fold_best_r2  = r2
                fold_best_acc = acc
                best_dp, best_cp, best_dt, best_ct = dp, cp, dt, ct

        val_acc_list.extend(fold_val_acc);   train_acc_list.extend(fold_tr_acc)
        val_r2_list.extend(fold_val_r2);     train_r2_list.extend(fold_tr_r2)
        val_pr_list.extend(fold_val_pr);     train_pr_list.extend(fold_tr_pr)
        val_preds_per_fold.append(best_dp);  val_true_per_fold.append(best_dt)
        val_cpreds_per_fold.append(best_cp); val_ctrue_per_fold.append(best_ct)

    val_acc   = np.array(val_acc_list);   tr_acc  = np.array(train_acc_list)
    val_r2    = np.array(val_r2_list);    tr_r2   = np.array(train_r2_list)
    val_pr    = np.array(val_pr_list);    tr_pr   = np.array(train_pr_list)

    print(f"\n{'='*60}")
    print(f"VALIDATION  Acc={val_acc.mean():.4f}±{val_acc.std():.4f}  "
          f"R²={val_r2.mean():.4f}±{val_r2.std():.4f}  r={val_pr.mean():.4f}±{val_pr.std():.4f}")
    print(f"TRAIN       Acc={tr_acc.mean():.4f}±{tr_acc.std():.4f}  "
          f"R²={tr_r2.mean():.4f}±{tr_r2.std():.4f}  r={tr_pr.mean():.4f}±{tr_pr.std():.4f}")
    print(f"ΔAcc={tr_acc.mean()-val_acc.mean():+.4f}  ΔR²={tr_r2.mean()-val_r2.mean():+.4f}")
    print(f"{'='*60}")

    # ── Permutation baselines ─────────────────────────────────────────────────
    print(f"\nPermutation baselines ({NUM_PERMUTATIONS} permutations)...")
    all_val_disc  = np.concatenate(val_true_per_fold)
    all_val_cont  = np.concatenate(val_ctrue_per_fold)

    perm_acc_list, perm_r2_list = [], []
    for _ in tqdm(range(NUM_PERMUTATIONS), desc='Permutations', leave=False):
        perm_acc_list.append(
            accuracy_score(all_val_disc, np.random.choice(all_val_disc, size=all_val_disc.shape))
        )
        perm = np.random.permutation(all_val_cont)
        ss_res = np.sum((all_val_cont - perm) ** 2)
        ss_tot = np.sum((all_val_cont - all_val_cont.mean()) ** 2)
        perm_r2_list.append(1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan'))

    perm_acc = np.array(perm_acc_list)
    perm_r2  = np.array(perm_r2_list)
    t_acc, p_acc = stats.ttest_ind(val_acc, perm_acc)
    t_r2,  p_r2  = stats.ttest_ind(val_r2,  perm_r2)
    print(f"Perm Acc: {perm_acc.mean():.4f}±{perm_acc.std():.4f}  t={t_acc:.3f} p={p_acc:.2e}")
    print(f"Perm R²:  {perm_r2.mean():.4f}±{perm_r2.std():.4f}  t={t_r2:.3f}  p={p_r2:.2e}")

    # ── Save raw ──────────────────────────────────────────────────────────────
    np.savetxt(os.path.join(output_dir, f'acc_val_{session_dir}.txt'),        val_acc)
    np.savetxt(os.path.join(output_dir, f'acc_train_{session_dir}.txt'),      tr_acc)
    np.savetxt(os.path.join(output_dir, f'r2_val_{session_dir}.txt'),         val_r2)
    np.savetxt(os.path.join(output_dir, f'r2_train_{session_dir}.txt'),       tr_r2)
    np.savetxt(os.path.join(output_dir, f'pearson_r_val_{session_dir}.txt'),  val_pr)
    np.savetxt(os.path.join(output_dir, f'acc_permutation_{session_dir}.txt'),perm_acc)
    np.savetxt(os.path.join(output_dir, f'r2_permutation_{session_dir}.txt'), perm_r2)

    # ── Figure: 2×3 summary ───────────────────────────────────────────────────
    print("\nGenerating visualizations...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Hybrid Decoder (α={HYBRID_ALPHA}) — {session_dir}', fontsize=16, fontweight='bold')

    # [0,0] Accuracy: train / val / permutation
    ax = axes[0, 0]
    bp = ax.boxplot([tr_acc, val_acc, perm_acc], positions=[1, 2, 3], widths=0.6,
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
    for p, c in zip(bp['boxes'], ['lightgreen', 'skyblue', 'lightgray']): p.set_facecolor(c)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(['Train', 'Validation', 'Permutation'])
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Accuracy\nVal={val_acc.mean():.3f}±{val_acc.std():.3f}  p={p_acc:.2e}', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y'); ax.set_ylim([0, 1.05])

    # [0,1] R²: train / val / permutation
    ax = axes[0, 1]
    bp2 = ax.boxplot([tr_r2, val_r2, perm_r2], positions=[1, 2, 3], widths=0.6,
                     patch_artist=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
    for p, c in zip(bp2['boxes'], ['lightgreen', 'skyblue', 'lightgray']): p.set_facecolor(c)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(['Train', 'Validation', 'Permutation'])
    ax.set_ylabel('R²')
    ax.set_title(f'R² (Belief)\nVal={val_r2.mean():.3f}±{val_r2.std():.3f}  p={p_r2:.2e}', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y'); ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    # [0,2] Pearson r
    ax = axes[0, 2]
    bp3 = ax.boxplot([tr_pr, val_pr], positions=[1, 2], widths=0.6,
                     patch_artist=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
    for p, c in zip(bp3['boxes'], ['lightgreen', 'skyblue']): p.set_facecolor(c)
    ax.set_xticks([1, 2]); ax.set_xticklabels(['Train', 'Validation'])
    ax.set_ylabel('Pearson r')
    ax.set_title(f'Pearson r (Belief)\nVal={val_pr.mean():.3f}±{val_pr.std():.3f}', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y'); ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    # [1,0] True vs Predicted belief (all folds, best run)
    ax = axes[1, 0]
    all_ct = np.concatenate(val_ctrue_per_fold)
    all_cp = np.concatenate(val_cpreds_per_fold)
    ax.scatter(all_ct, all_cp, alpha=0.2, s=5, color='steelblue', rasterized=True)
    lim_lo = min(all_ct.min(), all_cp.min()); lim_hi = max(all_ct.max(), all_cp.max())
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], 'r--', linewidth=1.5, label='Perfect')
    r_all, _ = stats.pearsonr(all_ct, all_cp)
    ax.text(0.05, 0.93, f'r={r_all:.3f}', transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax.set_xlabel('True HGF Belief'); ax.set_ylabel('Predicted HGF Belief')
    ax.set_title('True vs Predicted Belief (best runs)', fontweight='bold'); ax.legend(); ax.grid(True, alpha=0.3)

    # [1,1] Per-fold R² and Acc bar
    ax = axes[1, 1]
    fold_val_acc_m = np.array([np.mean(val_acc_list[f*NUM_DECODER_RUNS:(f+1)*NUM_DECODER_RUNS]) for f in range(num_folds)])
    fold_val_r2_m  = np.array([np.mean(val_r2_list [f*NUM_DECODER_RUNS:(f+1)*NUM_DECODER_RUNS]) for f in range(num_folds)])
    xf = np.arange(num_folds)
    ax.bar(xf - 0.175, fold_val_acc_m, 0.35, label='Val Acc',  color='skyblue',  alpha=0.85)
    ax.bar(xf + 0.175, fold_val_r2_m,  0.35, label='Val R²',   color='tomato',   alpha=0.85)
    ax.axhline(perm_acc.mean(), color='blue',  linestyle=':', linewidth=1.2, label=f'Perm Acc ({perm_acc.mean():.3f})')
    ax.axhline(perm_r2.mean(),  color='red',   linestyle=':', linewidth=1.2, label=f'Perm R² ({perm_r2.mean():.3f})')
    ax.set_xticks(xf); ax.set_xticklabels([f'F{i+1}' for i in range(num_folds)])
    ax.set_xlabel('Fold'); ax.set_title('Per-Fold Val Acc & R²', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    # [1,2] Summary text
    ax = axes[1, 2]; ax.axis('off')
    summary = (
        f"HYBRID SUMMARY — {session_dir}\n{'='*36}\n\n"
        f"Alpha = {HYBRID_ALPHA}  (CE_norm + MSE)\n\n"
        f"Accuracy (Discrete)\n"
        f"  Val:   {val_acc.mean():.4f} ± {val_acc.std():.4f}\n"
        f"  Train: {tr_acc.mean():.4f} ± {tr_acc.std():.4f}\n"
        f"  Perm:  {perm_acc.mean():.4f}  p={p_acc:.2e}\n\n"
        f"R² (Belief Regression)\n"
        f"  Val:   {val_r2.mean():.4f} ± {val_r2.std():.4f}\n"
        f"  Train: {tr_r2.mean():.4f} ± {tr_r2.std():.4f}\n"
        f"  Perm:  {perm_r2.mean():.4f}  p={p_r2:.2e}\n\n"
        f"Pearson r (Belief)\n"
        f"  Val:   {val_pr.mean():.4f} ± {val_pr.std():.4f}\n"
        f"  Train: {tr_pr.mean():.4f} ± {tr_pr.std():.4f}\n"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10, va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'summary_{session_dir}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'summary_{session_dir}.pdf'), bbox_inches='tight')
    plt.close(); print("  Saved summary figure.")

    # ── Per-fold R² bar chart ─────────────────────────────────────────────────
    fold_tr_r2_m  = np.array([np.mean(train_r2_list[f*NUM_DECODER_RUNS:(f+1)*NUM_DECODER_RUNS]) for f in range(num_folds)])
    fold_val_r2_s = np.array([np.std( val_r2_list  [f*NUM_DECODER_RUNS:(f+1)*NUM_DECODER_RUNS]) for f in range(num_folds)])

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.bar(xf - 0.175, fold_tr_r2_m,  0.35, label='Train R²', color='lightgreen', alpha=0.85)
    ax2.bar(xf + 0.175, fold_val_r2_m, 0.35, yerr=fold_val_r2_s, capsize=4, label='Val R²', color='skyblue', alpha=0.85)
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax2.axhline(perm_r2.mean(), color='gray', linestyle=':', linewidth=1.2, label=f'Permutation ({perm_r2.mean():.3f})')
    ax2.set_xticks(xf); ax2.set_xticklabels([f'F{i+1}' for i in range(num_folds)])
    ax2.set_xlabel('Fold'); ax2.set_ylabel('R²')
    ax2.set_title(f'R² per Fold (Hybrid) — {session_dir}', fontsize=13, fontweight='bold')
    ax2.legend(); ax2.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'per_fold_r2_{session_dir}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'per_fold_r2_{session_dir}.pdf'), bbox_inches='tight')
    plt.close(); print("  Saved per-fold R² figure.")

    print(f"\nHybrid evaluation done. Results → {output_dir}/")
    return {
        'val_acc': float(val_acc.mean()), 'val_acc_std': float(val_acc.std()),
        'val_r2': float(val_r2.mean()), 'val_r2_std': float(val_r2.std()),
        'val_pearson_r': float(val_pr.mean()), 'val_pearson_r_std': float(val_pr.std()),
        'perm_acc': float(perm_acc.mean()), 'p_acc': float(p_acc),
        'perm_r2': float(perm_r2.mean()), 'p_r2': float(p_r2),
    }


# ===========================================================================
# MAIN
# ===========================================================================

if RUN_DISCRETE:
    run_discrete_evaluation()

if RUN_CONTINUOUS:
    run_continuous_evaluation()

_run_metrics = {}

if RUN_DISCRETE:
    _run_metrics['discrete'] = run_discrete_evaluation()

if RUN_CONTINUOUS:
    _run_metrics['continuous'] = run_continuous_evaluation()

if RUN_HYBRID:
    _run_metrics['hybrid'] = run_hybrid_evaluation()

_summary = dict(
    status='completed',
    completed_at=datetime.datetime.now().isoformat(),
    output_dir=run_dir,
    configuration=_config,
    metrics=_run_metrics,
)
with open(os.path.join(run_dir, 'run_summary.json'), 'w') as _f:
    json.dump(_summary, _f, indent=2)

print(f"\n{'='*60}")
print(f"All done.  Session: {session_dir}")
print(f"Results  → {run_dir}/")
print(f"{'='*60}")
