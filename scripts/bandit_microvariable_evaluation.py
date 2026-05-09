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
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, r2_score
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
# Allow overriding key dataset / pipeline parameters via environment
# variables so we can run comparable microvariable evals without editing code.
# Examples: MICRO_DOWNSAMPLE_FS, MICRO_DOWNSAMPLE_METHOD, MICRO_NORMALIZE_METHOD,
# MICRO_WINDOW_SIZE, MICRO_GOOD_NEURONS_ONLY, MICRO_USE_LAZY_LOADING, etc.
DOWNSAMPLE_FS       = int(os.getenv('MICRO_DOWNSAMPLE_FS', '20'))
DOWNSAMPLE_METHOD   = os.getenv('MICRO_DOWNSAMPLE_METHOD', 'count')
GOOD_NEURONS_ONLY   = os.getenv('MICRO_GOOD_NEURONS_ONLY', 'True').lower() in ('1', 'true', 'yes')
NORMALIZE_METHOD    = os.getenv('MICRO_NORMALIZE_METHOD', 'minmax_global')  # None | 'minmax' | 'minmax_global'
STATE_TRANSITIONS   = None          # e.g. BanditTaskNeuroPixelsDataset.CHOOSING_TO_CORRECTNESS_TRANSITIONS
CHOOSING_STATE_MODE = os.getenv('MICRO_CHOOSING_STATE_MODE', 'side')        # 'side' | 'correctness'
GAUSSIAN_SIGMA_MS   = float(os.getenv('MICRO_GAUSSIAN_SIGMA_MS', '25.0'))  # only used when DOWNSAMPLE_METHOD='gaussian'
RECOMPUTE_CACHE     = os.getenv('MICRO_RECOMPUTE_CACHE', 'False').lower() in ('1', 'true', 'yes')

# --- HGF -------------------------------------------------------------------
USE_HGF             = os.getenv('MICRO_USE_HGF', 'True').lower() in ('1', 'true', 'yes')
HGF_MODEL           = os.getenv('MICRO_HGF_MODEL', 'binary2')     # substring matching HGF pkl filename
HGF_COLUMN          = os.getenv('MICRO_HGF_COLUMN', 'x_1_expected_mean')   # 'x_1_expected_mean' | 'x_0_expected_mean'
HGF_BELIEF_RANGE    = None          # None = use KNOWN_HGF_RANGES; or explicit (lo, hi)

# --- Evaluation modes -------------------------------------------------------
RUN_DISCRETE        = os.getenv('MICRO_RUN_DISCRETE', 'True').lower() in ('1', 'true', 'yes')
RUN_HYBRID          = os.getenv('MICRO_RUN_HYBRID', 'True').lower() in ('1', 'true', 'yes')
RUN_CONTINUOUS      = os.getenv('MICRO_RUN_CONTINUOUS', 'True').lower() in ('1', 'true', 'yes')
HYBRID_ALPHA        = float(os.getenv('MICRO_HYBRID_ALPHA', '0.1'))           # α * CE_norm + (1-α) * MSE  (matches BunDLeNet default)

# --- Data pipeline ----------------------------------------------------------
WINDOW_SIZE         = int(os.getenv('MICRO_WINDOW_SIZE', '60'))            # sliding window length (timesteps)
NUM_OF_SPLITS       = int(os.getenv('MICRO_NUM_OF_SPLITS', '9'))          # number of time-series CV folds (only used when SPLIT_MODE='cv')
USE_LAZY_LOADING    = os.getenv('MICRO_USE_LAZY_LOADING', 'True').lower() in ('1', 'true', 'yes')
                                    # False = eager numpy (fast but may OOM)
NUM_WORKERS         = int(os.getenv('MICRO_NUM_WORKERS', '4'))            # DataLoader worker processes for prefetching (USE_LAZY_LOADING only)

# --- Split mode -------------------------------------------------------------
# 'cv'             : NUM_OF_SPLITS-fold time-series cross-validation (default, thorough)
# 'test_split'     : KFold(n_splits=7) fold-4 — legacy alias, hardcoded 7/4
# 'bundlenet_split': KFold single split matching bandit_gridsearch.py params
SPLIT_MODE               = 'cv'   # 'cv' | 'test_split' | 'bundlenet_split'
BUNDLENET_KFOLD_N_SPLITS = 7     # KFold n_splits when SPLIT_MODE='bundlenet_split'
                                  # matches bandit_gridsearch.py --kfold_n_splits default
BUNDLENET_KFOLD_FOLD_IDX = 4     # fold index used as val when SPLIT_MODE='bundlenet_split'
                                  # matches bandit_gridsearch.py --kfold_test_fold default

# --- Decoder training -------------------------------------------------------
NUM_DECODER_RUNS    = int(os.getenv('MICRO_NUM_DECODER_RUNS', '10'))    # independent decoder runs per fold
TRAIN_EPOCHS        = int(os.getenv('MICRO_TRAIN_EPOCHS', '100'))      # epochs per decoder run
BATCH_SIZE          = int(os.getenv('MICRO_BATCH_SIZE', '512'))        # mini-batch size (only used when USE_LAZY_LOADING=True)
NUM_PERMUTATIONS    = int(os.getenv('MICRO_NUM_PERMUTATIONS', '200'))   # permutation baseline runs (discrete: chance acc; continuous: chance R²)

# ===========================================================================
# Validation
# ===========================================================================

if RUN_HYBRID and not USE_HGF:
    raise ValueError(
        "RUN_HYBRID=True requires USE_HGF=True. "
        "Either set USE_HGF=True or disable RUN_HYBRID."
    )

if RUN_CONTINUOUS and not USE_HGF:
    raise ValueError(
        "RUN_CONTINUOUS=True requires USE_HGF=True. "
        "Either set USE_HGF=True or disable RUN_CONTINUOUS."
    )

# ===========================================================================
# CLI
# ===========================================================================

data_path   = sys.argv[1]
# Optional second positional arg overrides SPLIT_MODE from CLI
if len(sys.argv) > 2 and sys.argv[2] in ('cv', 'test_split', 'bundlenet_split'):
    SPLIT_MODE = sys.argv[2]
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

_active_split_mode = SPLIT_MODE
if _active_split_mode == 'cv':
    _split_suffix = 'cv'
    _split_label  = f'{NUM_OF_SPLITS}-fold CV'
elif _active_split_mode == 'bundlenet_split':
    _split_suffix = 'bundlenet'
    _split_label  = (f'KFold-{BUNDLENET_KFOLD_N_SPLITS} fold-{BUNDLENET_KFOLD_FOLD_IDX}'
                     ' (BunDLeNet-style)')
else:  # 'test_split' — legacy alias, hardcoded 7/4
    _split_suffix = 'testsplit'
    _split_label  = 'KFold-7 fold-4 (BunDLeNet-style)'
run_dir  = os.path.join('results', 'twoArmBandit', 'microvariable_evaluation',
                        f'{session_dir}_{_ts}_{_split_suffix}')
for _sub in ('discrete', 'hybrid', 'continuous'):
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
    run_discrete=RUN_DISCRETE,
    run_hybrid=RUN_HYBRID, hybrid_alpha=HYBRID_ALPHA,
    run_continuous=RUN_CONTINUOUS,
    # pipeline
    window_size=WINDOW_SIZE, num_of_splits=NUM_OF_SPLITS,
    split_mode=_active_split_mode,
    bundlenet_kfold_n_splits=BUNDLENET_KFOLD_N_SPLITS,
    bundlenet_kfold_fold_idx=BUNDLENET_KFOLD_FOLD_IDX,
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
    """Return CV splits for a given label array using the configured pipeline.

    Returns a list of (x_tr, x_val, b_tr, b_val) tuples and the processed B.
    In 'cv' mode: NUM_OF_SPLITS folds (TimeSeriesSplit).
    In 'bundlenet_split' mode: single KFold split using BUNDLENET_KFOLD_N_SPLITS /
        BUNDLENET_KFOLD_FOLD_IDX (matches bandit_gridsearch.py defaults).
    In 'test_split' mode: legacy alias — same as bundlenet_split but hardcoded
        to KFold(7) fold-4 regardless of BUNDLENET_* params.
    """
    if _active_split_mode == 'cv':
        if USE_LAZY_LOADING:
            X_, B_ = prep_data_lazy(X, label_array, win=WINDOW_SIZE)
            return timeseries_train_test_split_cv_lazy(X_, B_, NUM_OF_SPLITS), B_
        else:
            X_, B_ = prep_data(X, label_array, win=WINDOW_SIZE)
            return timeseries_train_test_split_cv(X_, B_, NUM_OF_SPLITS), B_
    else:
        # Single KFold split — 'bundlenet_split' uses BUNDLENET_* params,
        # 'test_split' (legacy) uses hardcoded 7/4.
        _n_splits = (BUNDLENET_KFOLD_N_SPLITS if _active_split_mode == 'bundlenet_split'
                     else 7)
        _fold_idx = (BUNDLENET_KFOLD_FOLD_IDX if _active_split_mode == 'bundlenet_split'
                     else 4)
        if USE_LAZY_LOADING:
            from torch.utils.data import Subset
            X_, B_ = prep_data_lazy(X, label_array, win=WINDOW_SIZE)
            kf = KFold(n_splits=_n_splits, shuffle=False)
            for i, (train_idx, test_idx) in enumerate(kf.split(range(len(X_)))):
                if i == _fold_idx:
                    x_tr  = Subset(X_, train_idx)
                    x_val = Subset(X_, test_idx)
                    b_tr  = B_[train_idx]
                    b_val = B_[test_idx]
                    return [(x_tr, x_val, b_tr, b_val)], B_
        else:
            X_, B_ = prep_data(X, label_array, win=WINDOW_SIZE)
            kf = KFold(n_splits=_n_splits, shuffle=False)
            for i, (train_idx, test_idx) in enumerate(kf.split(X_)):
                if i == _fold_idx:
                    x_tr  = X_[train_idx]
                    x_val = X_[test_idx]
                    b_tr  = B_[train_idx]
                    b_val = B_[test_idx]
                    return [(x_tr, x_val, b_tr, b_val)], B_


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

    # x_pos used in summary figure below
    x_pos = np.arange(len(state_labels))

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
        """Save data files only — figures are produced once after both variants finish."""
        suffix = results['suffix']
        np.savetxt(os.path.join(output_dir, f'acc_list_val_{session_dir}{suffix}.txt'),   results['val_acc_list'])
        np.savetxt(os.path.join(output_dir, f'acc_list_train_{session_dir}{suffix}.txt'), results['train_acc_list'])
        np.save(os.path.join(output_dir, f'all_f1_scores_val_{session_dir}{suffix}.npy'),   results['val_all_f1_scores'])
        np.save(os.path.join(output_dir, f'all_f1_scores_train_{session_dir}{suffix}.npy'), results['train_all_f1_scores'])

    # ── Run both loss variants ───────────────────────────────────────────────
    res_uw = _train(use_weighted_loss=False, suffix='_unweighted')
    _save(res_uw)
    res_w  = _train(use_weighted_loss=True,  suffix='_weighted')
    _save(res_w)

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
    t_uw, p_uw = stats.ttest_ind(res_uw['val_acc_list'], chance_acc)
    t_w,  p_w  = stats.ttest_ind(res_w['val_acc_list'],  chance_acc)

    # ── Derived quantities for figure ────────────────────────────────────────
    uw_f1_means = res_uw['val_all_f1_scores'].mean(axis=0)
    uw_f1_std   = res_uw['val_all_f1_scores'].std(axis=0)
    w_f1_means  = res_w['val_all_f1_scores'].mean(axis=0)
    w_f1_std    = res_w['val_all_f1_scores'].std(axis=0)

    uw_conf = res_uw['avg_conf_matrix_val']
    w_conf  = res_w['avg_conf_matrix_val']
    norm_conf_uw = uw_conf / uw_conf.sum(axis=1, keepdims=True) * 100
    norm_conf_w  = w_conf  / w_conf.sum(axis=1, keepdims=True)  * 100

    avg_prec, avg_rec = [], []
    for si in range(len(state_labels)):
        precs, recs = [], []
        for pred, true in zip(res_uw['val_all_predictions'], res_uw['val_true_labels']):
            p, r, _, _ = precision_recall_fscore_support(
                np.asarray(true, dtype=np.int64).ravel(),
                np.asarray(pred, dtype=np.int64).ravel(),
                labels=state_labels, zero_division=0, average=None)
            precs.append(p[si]); recs.append(r[si])
        avg_prec.append(np.mean(precs)); avg_rec.append(np.mean(recs))

    state_names = [b_labels_dict.get(s, f'S{s}') for s in state_labels]

    # ── Single summary figure (2 × 3, PDF only) ──────────────────────────────
    print("\nGenerating summary figure...")
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Discrete Decoder Summary — {session_dir}', fontsize=15, fontweight='bold')

    # (0,0) Accuracy: UW / W / Chance
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

    # (0,1) Normalised confusion matrix — Unweighted
    ax = axs[0, 1]
    sns.heatmap(norm_conf_uw, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax,
                xticklabels=state_names, yticklabels=state_names,
                cbar_kws={'label': '%'}, vmin=0, vmax=100)
    ax.set_title('Confusion Matrix — Unweighted (%)', fontweight='bold')
    ax.tick_params(axis='x', rotation=45); ax.tick_params(axis='y', rotation=0)

    # (0,2) Normalised confusion matrix — Weighted
    ax = axs[0, 2]
    sns.heatmap(norm_conf_w, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax,
                xticklabels=state_names, yticklabels=state_names,
                cbar_kws={'label': '%'}, vmin=0, vmax=100)
    ax.set_title('Confusion Matrix — Weighted (%)', fontweight='bold')
    ax.tick_params(axis='x', rotation=45); ax.tick_params(axis='y', rotation=0)

    # (1,0) Per-state F1 — UW vs W (mean ± std bars)
    ax = axs[1, 0]
    bw = 0.35
    ax.bar(x_pos - bw/2, uw_f1_means, bw, yerr=uw_f1_std, label='Unweighted',
           color='#4C9BE8', alpha=0.85, capsize=4, error_kw=dict(elinewidth=1.2))
    ax.bar(x_pos + bw/2, w_f1_means,  bw, yerr=w_f1_std,  label='Weighted',
           color='#6BBF6B', alpha=0.85, capsize=4, error_kw=dict(elinewidth=1.2))
    ax.set_xticks(x_pos); ax.set_xticklabels(state_names, rotation=45, ha='right')
    ax.set_ylabel('F1', fontsize=12); ax.set_title('Per-State F1 — UW vs W (mean ± std)', fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y'); ax.set_ylim([0, 1.05])

    # (1,1) Precision / Recall / F1 per state — Unweighted
    ax = axs[1, 1]
    pw = 0.25
    ax.bar(x_pos - pw,   avg_prec,    pw, label='Precision', color='#5B8DD9', alpha=0.85)
    ax.bar(x_pos,        avg_rec,     pw, label='Recall',    color='#E8864C', alpha=0.85)
    ax.bar(x_pos + pw,   uw_f1_means, pw, label='F1',        color='#6BBF6B', alpha=0.85)
    ax.set_xticks(x_pos); ax.set_xticklabels(state_names, rotation=45, ha='right')
    ax.set_title('Precision / Recall / F1 — Unweighted', fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y'); ax.set_ylim([0, 1.05])

    # (1,2) Summary text
    ax = axs[1, 2]; ax.axis('off')
    summary_t = (
        f"SUMMARY — {session_dir}\n{'─'*36}\n"
        f"Split: {_split_label}\n\n"
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
        f"Folds: {num_folds}  ×  Runs/fold: {NUM_DECODER_RUNS}\n"
        f"Window: {WINDOW_SIZE}  ·  Permutations: {NUM_PERMUTATIONS}"
    )
    ax.text(0.05, 0.97, summary_t, transform=ax.transAxes, fontsize=11,
            va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'summary_{session_dir}.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved summary → {output_dir}/summary_{session_dir}.pdf")

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
# HYBRID EVALUATION  (joint discrete+continuous decoder, discrete-style output)
# ===========================================================================

def _predict_disc(model, loader, n_classes):
    """Run hybrid model, return only (disc_preds, disc_true) as numpy arrays."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            preds.append(out[:, :n_classes].argmax(dim=1).cpu().numpy())
            trues.append(yb[:, 0].long().numpy())
    return np.concatenate(preds), np.concatenate(trues)





def run_hybrid_evaluation():
    """Joint discrete+continuous decoder — produces the same figures as discrete
    so decodability of behavioral labels can be compared directly.
    Trains with α·CE_norm + (1-α)·MSE (matching BunDLeNet) but reports
    accuracy / F1 / confusion matrix for easy comparison with run_discrete_evaluation."""
    print(f"\n{'#'*70}")
    print(f"### HYBRID EVALUATION — JOINT DISCRETE + CONTINUOUS DECODER ###")
    print(f"{'#'*70}")
    print(f"Alpha={HYBRID_ALPHA}  (α·CE_norm + (1-α)·MSE, matching BunDLeNet)")

    import math

    cv_splits, B_hyb_ = make_cv_splits(B_hybrid)
    num_folds    = len(cv_splits)
    state_labels = np.unique(B_hyb_[:, 0].astype(np.int64))
    num_states   = len(state_labels)
    ce_norm      = math.log(num_states)   # normalise CE to ~[0,1]
    output_dim   = num_states + 1          # logits + 1 belief output
    print(f"Prepared {num_folds}-fold CV. {num_states} behavioral states.")

    # ── Label distribution (from discrete column) ────────────────────────────
    train_label_counts_raw = defaultdict(int)
    val_label_counts_raw   = defaultdict(int)
    total_train = total_val = 0

    for x_tr, x_val, b_tr, b_val in cv_splits:
        b_tr_disc  = b_tr[:, 0].astype(np.int64)
        b_val_disc = b_val[:, 0].astype(np.int64)
        for lbl, cnt in zip(*np.unique(b_tr_disc,  return_counts=True)):
            train_label_counts_raw[lbl] += cnt
        for lbl, cnt in zip(*np.unique(b_val_disc, return_counts=True)):
            val_label_counts_raw[lbl] += cnt
        total_train += len(b_tr_disc)
        total_val   += len(b_val_disc)

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
    weights_list         = [class_weights[l] for l in sorted(state_labels)]
    class_weights_tensor = torch.FloatTensor(weights_list).to(device)

    output_dir = os.path.join(run_dir, 'hybrid')
    x_pos = np.arange(len(state_labels))

    # ── Inner training function ───────────────────────────────────────────────
    def _train(use_weighted_loss, suffix):
        loss_type = "WEIGHTED" if use_weighted_loss else "UNWEIGHTED"
        print(f"\n{'#'*60}")
        print(f"### HYBRID {loss_type} LOSS ###")
        print(f"{'#'*60}")

        val_acc_list, val_all_predictions, val_all_f1_scores, val_true_labels       = [], [], [], []
        train_acc_list, train_all_predictions, train_all_f1_scores, train_true_labels = [], [], [], []
        val_conf_sum   = np.zeros((num_states, num_states))
        train_conf_sum = np.zeros((num_states, num_states))
        total_runs = 0

        ce_loss_fn  = nn.CrossEntropyLoss(weight=class_weights_tensor if use_weighted_loss else None)
        mse_loss_fn = nn.MSELoss()

        for fold_idx, (x_tr, x_val, b_tr, b_val) in enumerate(cv_splits):
            print(f"\nFold {fold_idx+1}/{num_folds}: train={fold_size(x_tr)}, val={fold_size(x_val)}")
            tr_loader  = make_loaders(x_tr,  b_tr,  'hybrid')
            val_loader = make_loaders(x_val, b_val, 'hybrid')

            for _ in tqdm(range(NUM_DECODER_RUNS), desc=f'Decoders Hybrid {loss_type} fold {fold_idx+1}', leave=False):
                model = nn.Linear(input_dim, output_dim).to(device)
                opt   = optim.Adam(model.parameters(), lr=0.01)

                for epoch in range(TRAIN_EPOCHS):
                    model.train()
                    for xb, yb in tr_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        out      = model(xb)
                        logits   = out[:, :num_states]
                        cont_out = out[:, num_states:num_states+1]
                        disc_lbl = yb[:, 0].long()
                        cont_lbl = yb[:, 1:2]
                        loss = (HYBRID_ALPHA * ce_loss_fn(logits, disc_lbl) / ce_norm
                                + (1 - HYBRID_ALPHA) * mse_loss_fn(cont_out, cont_lbl))
                        opt.zero_grad(); loss.backward(); opt.step()

                val_pred,   val_true   = _predict_disc(model, val_loader,  num_states)
                train_pred, train_true = _predict_disc(model, tr_loader,   num_states)

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
        print(f"HYBRID {loss_type} — Val acc: {val_acc_list.mean():.3f} ± {val_acc_list.std():.3f}  |  "
              f"Train acc: {train_acc_list.mean():.3f} ± {train_acc_list.std():.3f}")
        print(f"Train-Val gap: {train_acc_list.mean() - val_acc_list.mean():.3f}")
        print(f"{'='*60}")

        return dict(
            suffix=suffix, loss_type=f'HYBRID {loss_type}',
            val_acc_list=val_acc_list, val_all_predictions=np.array(val_all_predictions, dtype=object),
            val_all_f1_scores=val_all_f1_scores, val_true_labels=np.array(val_true_labels, dtype=object),
            train_acc_list=train_acc_list, train_all_predictions=np.array(train_all_predictions, dtype=object),
            train_all_f1_scores=train_all_f1_scores, train_true_labels=np.array(train_true_labels, dtype=object),
            avg_conf_matrix_val=avg_conf_val, avg_conf_matrix_train=avg_conf_train,
        )

    def _save(results):
        """Save data files only — figure produced once after both variants finish."""
        suffix = results['suffix']
        np.savetxt(os.path.join(output_dir, f'acc_list_val_{session_dir}{suffix}.txt'),   results['val_acc_list'])
        np.savetxt(os.path.join(output_dir, f'acc_list_train_{session_dir}{suffix}.txt'), results['train_acc_list'])
        np.save(os.path.join(output_dir, f'all_f1_scores_val_{session_dir}{suffix}.npy'),   results['val_all_f1_scores'])
        np.save(os.path.join(output_dir, f'all_f1_scores_train_{session_dir}{suffix}.npy'), results['train_all_f1_scores'])

    # ── Run both loss variants ───────────────────────────────────────────────
    res_uw = _train(use_weighted_loss=False, suffix='_unweighted')
    _save(res_uw)
    res_w  = _train(use_weighted_loss=True,  suffix='_weighted')
    _save(res_w)

    # ── Chance accuracy ───────────────────────────────────────────────────────
    print(f"\nEstimating chance accuracy ({NUM_PERMUTATIONS} permutations)...")
    all_val_labels_chance = np.concatenate([split[3][:, 0].astype(np.int64) for split in cv_splits])
    chance_acc = np.array([
        accuracy_score(np.random.choice(all_val_labels_chance, size=all_val_labels_chance.shape),
                       all_val_labels_chance)
        for _ in tqdm(range(NUM_PERMUTATIONS), desc='Chance', leave=False)
    ])
    print(f"Chance accuracy: {chance_acc.mean():.3f} ± {chance_acc.std():.3f}")
    np.savetxt(os.path.join(output_dir, f'acc_list_chance_{session_dir}.txt'), chance_acc)
    t_uw, p_uw = stats.ttest_ind(res_uw['val_acc_list'], chance_acc)
    t_w,  p_w  = stats.ttest_ind(res_w['val_acc_list'],  chance_acc)

    # ── Derived quantities for figure ────────────────────────────────────────
    uw_f1_means = res_uw['val_all_f1_scores'].mean(axis=0)
    uw_f1_std   = res_uw['val_all_f1_scores'].std(axis=0)
    w_f1_means  = res_w['val_all_f1_scores'].mean(axis=0)
    w_f1_std    = res_w['val_all_f1_scores'].std(axis=0)

    uw_conf = res_uw['avg_conf_matrix_val']
    w_conf  = res_w['avg_conf_matrix_val']
    norm_conf_uw = uw_conf / uw_conf.sum(axis=1, keepdims=True) * 100
    norm_conf_w  = w_conf  / w_conf.sum(axis=1, keepdims=True)  * 100

    avg_prec, avg_rec = [], []
    for si in range(len(state_labels)):
        precs, recs = [], []
        for pred, true in zip(res_uw['val_all_predictions'], res_uw['val_true_labels']):
            p, r, _, _ = precision_recall_fscore_support(
                np.asarray(true, dtype=np.int64).ravel(),
                np.asarray(pred, dtype=np.int64).ravel(),
                labels=state_labels, zero_division=0, average=None)
            precs.append(p[si]); recs.append(r[si])
        avg_prec.append(np.mean(precs)); avg_rec.append(np.mean(recs))

    state_names = [b_labels_dict.get(s, f'S{s}') for s in state_labels]

    # ── Single summary figure (2 × 3, PDF only) ──────────────────────────────
    print("\nGenerating summary figure...")
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Hybrid Decoder Summary (α={HYBRID_ALPHA}) — {session_dir}', fontsize=15, fontweight='bold')

    # (0,0) Accuracy: UW / W / Chance
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

    # (0,1) Normalised confusion matrix — Unweighted
    ax = axs[0, 1]
    sns.heatmap(norm_conf_uw, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax,
                xticklabels=state_names, yticklabels=state_names,
                cbar_kws={'label': '%'}, vmin=0, vmax=100)
    ax.set_title('Confusion Matrix — Unweighted (%)', fontweight='bold')
    ax.tick_params(axis='x', rotation=45); ax.tick_params(axis='y', rotation=0)

    # (0,2) Normalised confusion matrix — Weighted
    ax = axs[0, 2]
    sns.heatmap(norm_conf_w, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax,
                xticklabels=state_names, yticklabels=state_names,
                cbar_kws={'label': '%'}, vmin=0, vmax=100)
    ax.set_title('Confusion Matrix — Weighted (%)', fontweight='bold')
    ax.tick_params(axis='x', rotation=45); ax.tick_params(axis='y', rotation=0)

    # (1,0) Per-state F1 — UW vs W (mean ± std bars)
    ax = axs[1, 0]
    bw = 0.35
    ax.bar(x_pos - bw/2, uw_f1_means, bw, yerr=uw_f1_std, label='Unweighted',
           color='#4C9BE8', alpha=0.85, capsize=4, error_kw=dict(elinewidth=1.2))
    ax.bar(x_pos + bw/2, w_f1_means,  bw, yerr=w_f1_std,  label='Weighted',
           color='#6BBF6B', alpha=0.85, capsize=4, error_kw=dict(elinewidth=1.2))
    ax.set_xticks(x_pos); ax.set_xticklabels(state_names, rotation=45, ha='right')
    ax.set_ylabel('F1', fontsize=12); ax.set_title('Per-State F1 — UW vs W (mean ± std)', fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y'); ax.set_ylim([0, 1.05])

    # (1,1) Precision / Recall / F1 per state — Unweighted
    ax = axs[1, 1]
    pw = 0.25
    ax.bar(x_pos - pw,   avg_prec,    pw, label='Precision', color='#5B8DD9', alpha=0.85)
    ax.bar(x_pos,        avg_rec,     pw, label='Recall',    color='#E8864C', alpha=0.85)
    ax.bar(x_pos + pw,   uw_f1_means, pw, label='F1',        color='#6BBF6B', alpha=0.85)
    ax.set_xticks(x_pos); ax.set_xticklabels(state_names, rotation=45, ha='right')
    ax.set_title('Precision / Recall / F1 — Unweighted', fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y'); ax.set_ylim([0, 1.05])

    # (1,2) Summary text
    ax = axs[1, 2]; ax.axis('off')
    summary_t = (
        f"HYBRID SUMMARY — {session_dir}\n{'─'*36}\n"
        f"Alpha = {HYBRID_ALPHA}  (α·CE_norm + (1-α)·MSE)\n"
        f"Split: {_split_label}\n\n"
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
        f"Folds: {num_folds}  ×  Runs/fold: {NUM_DECODER_RUNS}\n"
        f"Window: {WINDOW_SIZE}  ·  Permutations: {NUM_PERMUTATIONS}"
    )
    ax.text(0.05, 0.97, summary_t, transform=ax.transAxes, fontsize=11,
            va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'summary_{session_dir}.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved summary → {output_dir}/summary_{session_dir}.pdf")

    print(f"\nHybrid evaluation done. Results → {output_dir}/")
    return {
        'unweighted_val_acc': float(res_uw['val_acc_list'].mean()),
        'unweighted_val_acc_std': float(res_uw['val_acc_list'].std()),
        'weighted_val_acc': float(res_w['val_acc_list'].mean()),
        'weighted_val_acc_std': float(res_w['val_acc_list'].std()),
        'chance_acc': float(chance_acc.mean()),
        't_weighted_vs_chance': float(t_w), 'p_weighted_vs_chance': float(p_w),
    }


# ===========================================================================
# CONTINUOUS EVALUATION  (pure belief regression, no behavioral labels)
# ===========================================================================

def run_continuous_evaluation():
    """Linear MSE decoder on HGF belief trajectory only.

    Tests whether neuronal activity linearly encodes the animal's belief
    (as estimated by the HGF model) independently of any behavioral label.
    Reports R² on held-out validation folds with a permutation baseline.
    """
    print(f"\n{'#'*70}")
    print(f"### CONTINUOUS EVALUATION — HGF BELIEF REGRESSION ###")
    print(f"{'#'*70}")

    if B_belief is None:
        print("WARNING: B_belief is None — skipping continuous evaluation.")
        return {}

    output_dir = os.path.join(run_dir, 'continuous')

    # ── CV splits ────────────────────────────────────────────────────────────
    cv_splits, B_cont_ = make_cv_splits(B_belief)
    num_folds = len(cv_splits)
    print(f"Prepared {num_folds}-fold CV.  Belief range: [{B_cont_.min():.3f}, {B_cont_.max():.3f}]")

    # ── Training loop ────────────────────────────────────────────────────────
    val_r2_list, train_r2_list = [], []
    last_val_pred = last_val_true = None   # saved for predicted-vs-true scatter

    mse_loss_fn = nn.MSELoss()

    for fold_idx, (x_tr, x_val, b_tr, b_val) in enumerate(cv_splits):
        print(f"\nFold {fold_idx+1}/{num_folds}: train={fold_size(x_tr)}, val={fold_size(x_val)}")
        tr_loader  = make_loaders(x_tr,  b_tr,  'float')
        val_loader = make_loaders(x_val, b_val, 'float')

        for _ in tqdm(range(NUM_DECODER_RUNS), desc=f'Belief decoder fold {fold_idx+1}', leave=False):
            model = nn.Linear(input_dim, 1).to(device)
            opt   = optim.Adam(model.parameters(), lr=0.01)

            for epoch in range(TRAIN_EPOCHS):
                model.train()
                for xb, yb in tr_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    mse_loss_fn(model(xb), yb).backward()
                    opt.step()

            val_pred,   val_true   = predict_all(model, val_loader,   squeeze_pred=True, squeeze_true=True)
            train_pred, train_true = predict_all(model, tr_loader,    squeeze_pred=True, squeeze_true=True)

            val_r2_list.append(r2_score(val_true, val_pred))
            train_r2_list.append(r2_score(train_true, train_pred))

            last_val_pred = val_pred
            last_val_true = val_true

    val_r2_list   = np.array(val_r2_list)
    train_r2_list = np.array(train_r2_list)

    print(f"\n{'='*60}")
    print(f"Val R²:   {val_r2_list.mean():.3f} ± {val_r2_list.std():.3f}")
    print(f"Train R²: {train_r2_list.mean():.3f} ± {train_r2_list.std():.3f}")
    print(f"Train-Val gap: {train_r2_list.mean() - val_r2_list.mean():.3f}")
    print(f"{'='*60}")

    np.savetxt(os.path.join(output_dir, f'r2_val_{session_dir}.txt'),   val_r2_list)
    np.savetxt(os.path.join(output_dir, f'r2_train_{session_dir}.txt'), train_r2_list)

    # ── Permutation baseline ─────────────────────────────────────────────────
    print(f"\nEstimating chance R² ({NUM_PERMUTATIONS} permutations)...")
    all_val_beliefs = np.concatenate([split[3] for split in cv_splits])
    chance_r2 = np.array([
        r2_score(all_val_beliefs,
                 np.random.permutation(all_val_beliefs))
        for _ in tqdm(range(NUM_PERMUTATIONS), desc='Chance R²', leave=False)
    ])
    print(f"Chance R²: {chance_r2.mean():.3f} ± {chance_r2.std():.3f}")
    np.savetxt(os.path.join(output_dir, f'r2_chance_{session_dir}.txt'), chance_r2)

    t_stat, p_val = stats.ttest_ind(val_r2_list, chance_r2)
    print(f"t-test vs chance: t={t_stat:.2f}, p={p_val:.2e}")

    # ── Plot 1: R² distribution (train vs val) ────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    bp = ax1.boxplot([train_r2_list, val_r2_list],
                     positions=[1, 2], widths=0.6, patch_artist=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
    for patch, col in zip(bp['boxes'], ['lightgreen', 'skyblue']):
        patch.set_facecolor(col)
    ax1.set_xticks([1, 2]); ax1.set_xticklabels(['Train', 'Validation'])
    ax1.set_ylabel('R²', fontsize=13)
    ax1.set_title(f'Belief Decoder R² — {session_dir}', fontsize=15, fontweight='bold')
    ax1.text(0.98, 0.02,
             f'Val: {val_r2_list.mean():.3f}±{val_r2_list.std():.3f}\n'
             f'Train: {train_r2_list.mean():.3f}±{train_r2_list.std():.3f}\n'
             f'Gap: {train_r2_list.mean()-val_r2_list.mean():.3f}',
             transform=ax1.transAxes, ha='right', va='bottom', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax1.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'r2_distribution_{session_dir}.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved R² distribution plot.")

    # ── Plot 2: decoder R² vs chance ─────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    bp2 = ax2.boxplot([val_r2_list, chance_r2],
                      positions=[1, 2], widths=0.6, patch_artist=True, showmeans=True,
                      meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
    for patch, col in zip(bp2['boxes'], ['skyblue', 'lightgray']):
        patch.set_facecolor(col)
    ax2.set_xticks([1, 2]); ax2.set_xticklabels(['Belief Decoder', 'Chance (permutation)'])
    ax2.set_ylabel('R²', fontsize=13)
    ax2.set_title(f'Belief Decoder vs Chance — {session_dir}', fontsize=15, fontweight='bold')
    ax2.text(0.5, 0.95,
             f'Val R² vs Chance: t={t_stat:.2f}, p={p_val:.2e}',
             transform=ax2.transAxes, ha='center', va='top', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'decoder_vs_chance_{session_dir}.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved decoder vs chance plot.")

    # ── Plot 3: predicted vs true belief (last fold, last run) ───────────────
    if last_val_pred is not None:
        fig3, ax3 = plt.subplots(figsize=(8, 8))
        ax3.scatter(last_val_true, last_val_pred, alpha=0.3, s=10, color='steelblue', label='Predictions')
        lo = min(last_val_true.min(), last_val_pred.min())
        hi = max(last_val_true.max(), last_val_pred.max())
        ax3.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='Identity (perfect)')
        r2_last = r2_score(last_val_true, last_val_pred)
        ax3.set_xlabel('True HGF belief', fontsize=13)
        ax3.set_ylabel('Predicted HGF belief', fontsize=13)
        ax3.set_title(f'Predicted vs True Belief (last fold) — {session_dir}', fontsize=13, fontweight='bold')
        ax3.text(0.05, 0.95, f'R² = {r2_last:.3f}',
                 transform=ax3.transAxes, va='top', fontsize=13,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax3.legend(); ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'predicted_vs_true_{session_dir}.pdf'), bbox_inches='tight')
        plt.close()
        print(f"Saved predicted vs true belief plot.")

    print(f"\nContinuous evaluation done. Results → {output_dir}/")
    return {
        'val_r2_mean':    float(val_r2_list.mean()),
        'val_r2_std':     float(val_r2_list.std()),
        'train_r2_mean':  float(train_r2_list.mean()),
        'train_r2_std':   float(train_r2_list.std()),
        'chance_r2_mean': float(chance_r2.mean()),
        't_vs_chance':    float(t_stat),
        'p_vs_chance':    float(p_val),
    }


# ===========================================================================
# MAIN
# ===========================================================================

_run_metrics = {}

if RUN_DISCRETE:
    _run_metrics['discrete'] = run_discrete_evaluation()

if RUN_HYBRID:
    _run_metrics['hybrid'] = run_hybrid_evaluation()

if RUN_CONTINUOUS:
    _run_metrics['continuous'] = run_continuous_evaluation()

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

