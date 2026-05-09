#!/usr/bin/env python3
"""
Trial-based Microvariable Evaluation for Two-Arm Bandit Task

This script evaluates linear decodability of behavioural states (and
optionally HGF beliefs) on a trial-wise split. It mirrors the
`bandit_microvariable_evaluation.py` figures but performs train/test
splits at the trial level (no cross-trial leakage).

Usage:
    python scripts/bandit_microvariable_evaluation_trial_based.py <data_path>

Defaults are chosen to match typical BunDLeNet runs (downsample=30,
gaussian, normalize=minmax_global, window=50). Environment variables
can override key parameters (see constants below).
"""

import sys
import os
import json
import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from ncmcm.data_loaders.bandit_task import BanditTaskNeuroPixelsDataset
from ncmcm.bundlenet.utils import (
    segments_from_trial_starts, prep_data_trials, prep_data_trials_lazy,
    trial_train_test_split, trial_train_test_split_lazy, make_hybrid_b,
    torch_batch_prep,
)

# ---------------------------------------------------------------------------
# Config (defaults chosen to match BunDLeNet run). Hardcoded values.
# ---------------------------------------------------------------------------
DOWNSAMPLE_FS       = 30
DOWNSAMPLE_METHOD   = 'gaussian'
GOOD_NEURONS_ONLY   = False
NORMALIZE_METHOD    = 'minmax_global'
CHOOSING_STATE_MODE = 'side'
GAUSSIAN_SIGMA_MS   = 25.0
RECOMPUTE_CACHE     = False

USE_HGF             = True
HGF_MODEL           = 'binary2'
HGF_COLUMN          = 'x_1_expected_mean'

# Behavioural label mode: 'full' (per-timepoint) or 'decision' (one label per trial)
B_MODE              = 'decision'

RUN_DISCRETE        = True
RUN_HYBRID          = True
RUN_CONTINUOUS      = True
HYBRID_ALPHA        = 0.1

WINDOW_SIZE         = 50
USE_LAZY_LOADING    = True
NUM_WORKERS         = 4

NUM_DECODER_RUNS    = 10
TRAIN_EPOCHS        = 100
BATCH_SIZE          = 256
NUM_PERMUTATIONS    = 200 # For estimating chance accuracy in discrete decoding

TRIAL_TEST_RATIO    = 0.2
RANDOM_SEED         = 42

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloader_eager(x_paired, b_arr, batch_size, shuffle=True):
    # x_paired: (M, 2, win, N)
    X1 = x_paired[:, 1, :, :]
    X1_flat = X1.reshape(X1.shape[0], -1)
    x_t = torch.FloatTensor(X1_flat)
    b_t = torch.LongTensor(b_arr)
    batches = list(zip([x_t[i:i+batch_size] for i in range(0, len(x_t), batch_size)],
                       [b_t[i:i+batch_size] for i in range(0, len(b_t), batch_size)]))
    return batches


def main():
    if len(sys.argv) < 2:
        print("Usage: python bandit_microvariable_evaluation_trial_based.py <data_path>")
        sys.exit(1)

    data_path = sys.argv[1]
    session_dir = os.path.basename(data_path.rstrip('/'))
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('results', 'twoArmBandit', 'microvariable_evaluation_trial_based', f'{session_dir}_{ts}_trialbased')
    for sub in ('discrete', 'hybrid', 'continuous'):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)

    print(f"Loading data from: {data_path}")
    _hgf_kwargs = dict(hgf_model=HGF_MODEL, hgf_column=HGF_COLUMN) if USE_HGF else dict(hgf_model=None)
    data = BanditTaskNeuroPixelsDataset(
        data_path=data_path,
        downsample_fs=DOWNSAMPLE_FS,
        downsample_method=DOWNSAMPLE_METHOD,
        good_neurons_only=GOOD_NEURONS_ONLY,
        normalize_method=NORMALIZE_METHOD,
        choosing_state_mode=CHOOSING_STATE_MODE,
        b_mode=B_MODE,
        gaussian_sigma_ms=GAUSSIAN_SIGMA_MS,
        recompute_cache=RECOMPUTE_CACHE,
        **_hgf_kwargs,
    )

    X = data.x.toarray().T
    B_raw = data.b.toarray().flatten()
    b_labels_dict = data.b_labels_dict
    print(f"Data shape: X={X.shape}, B={B_raw.shape}")

    # Encode labels to integers (LabelEncoder keeps mapping)
    le = LabelEncoder()
    B_encoded = le.fit_transform(B_raw)

    # HGF beliefs if available
    B_belief = data.hgf_beliefs if USE_HGF and getattr(data, 'hgf_beliefs', None) is not None else None
    if B_belief is not None:
        print(f"HGF beliefs found: range=[{B_belief.min():.3f}, {B_belief.max():.3f}]")

    # Trial segmentation + windowing
    print("Segmenting trials and windowing (trial-based)...")
    trial_start_indices = data.trial_start_indices
    if trial_start_indices is None:
        raise RuntimeError('Dataset does not expose trial_start_indices; trial-based evaluation requires it.')

    if USE_LAZY_LOADING:
        dataset_lazy, B_pairs, trial_ids = prep_data_trials_lazy(X, B_encoded, win=WINDOW_SIZE, trial_start_indices=trial_start_indices)
        (train_subset, B_train), (test_subset, B_test) = trial_train_test_split_lazy(dataset_lazy, B_pairs, trial_ids, test_ratio=TRIAL_TEST_RATIO, random_state=RANDOM_SEED)
        n_neurons = X.shape[1]
        input_dim = WINDOW_SIZE * n_neurons
        num_train = len(train_subset)
        num_test = len(test_subset)
    else:
        trial_segments = segments_from_trial_starts(X, B_encoded, trial_start_indices)
        X_paired, B_pairs, trial_ids = prep_data_trials(trial_segments, win=WINDOW_SIZE)
        (X_train, B_train), (X_test, B_test) = trial_train_test_split(X_paired, B_pairs, trial_ids, test_ratio=TRIAL_TEST_RATIO, random_state=RANDOM_SEED)
        n_neurons = X.shape[1]
        input_dim = WINDOW_SIZE * n_neurons
        num_train = len(X_train)
        num_test = len(X_test)

    print(f"Prepared trial-based pairs: train={num_train}, test={num_test}, input_dim={input_dim}")

    # Labels and class weights
    state_labels = np.unique(B_encoded)
    n_states = len(state_labels)
    print(f"State labels: {n_states} classes -> {b_labels_dict}")

    # Train/eval discrete decoder
    output_dir = os.path.join(run_dir, 'discrete')

    def train_and_eval_discrete():
        print('\nRunning discrete decoders (trial-based)')
        val_accs = []
        val_f1s = []
        val_conf_sum = np.zeros((n_states, n_states))

        # Precompute class weights from training distribution
        if USE_LAZY_LOADING:
            counts = defaultdict(int)
            for lbl in B_train:
                counts[int(lbl)] += 1
            total_train = len(B_train)
        else:
            vals, cnts = np.unique(B_train, return_counts=True)
            counts = {int(v): int(c) for v, c in zip(vals, cnts)}
            total_train = len(B_train)

        class_weights = {lbl: total_train / (n_states * counts.get(lbl, 1)) for lbl in state_labels}
        weights_list = [class_weights[l] for l in sorted(state_labels)]
        class_weights_tensor = torch.FloatTensor(weights_list).to(device)

        for run in range(NUM_DECODER_RUNS):
            model = nn.Linear(input_dim, n_states).to(device)
            opt = optim.Adam(model.parameters(), lr=0.01)
            crit = nn.CrossEntropyLoss(weight=class_weights_tensor)

            # Build loaders
            if USE_LAZY_LOADING:
                tr_loader = torch_batch_prep(train_subset, B_train, device=device, batch_size=BATCH_SIZE, shuffle=True)
                val_loader = torch_batch_prep(test_subset, B_test, device=device, batch_size=BATCH_SIZE, shuffle=False)
            else:
                tr_batches = make_dataloader_eager(X_train, B_train, BATCH_SIZE, shuffle=True)
                val_batches = make_dataloader_eager(X_test, B_test, BATCH_SIZE, shuffle=False)

            # Training
            model.train()
            for epoch in range(TRAIN_EPOCHS):
                if USE_LAZY_LOADING:
                    for xb, yb in tr_loader:
                        xb_feat = xb[:, 1, :, :].reshape(xb.shape[0], -1)
                        yb_t = yb.long()
                        opt.zero_grad()
                        loss = crit(model(xb_feat), yb_t)
                        loss.backward(); opt.step()
                else:
                    for xb, yb in tr_batches:
                        xb = xb.to(device)
                        yb = yb.to(device)
                        opt.zero_grad()
                        loss = crit(model(xb), yb)
                        loss.backward(); opt.step()

            # Evaluation
            if USE_LAZY_LOADING:
                preds, trues = [], []
                model.eval()
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb_feat = xb[:, 1, :, :].reshape(xb.shape[0], -1)
                        out = model(xb_feat)
                        preds.append(out.argmax(dim=1).cpu().numpy())
                        trues.append(yb.cpu().numpy())
                pred = np.concatenate(preds); true = np.concatenate(trues)
            else:
                preds, trues = [], []
                model.eval()
                with torch.no_grad():
                    for xb, yb in val_batches:
                        xb = xb.to(device); yb = yb.to(device)
                        out = model(xb)
                        preds.append(out.argmax(dim=1).cpu().numpy())
                        trues.append(yb.cpu().numpy())
                pred = np.concatenate(preds); true = np.concatenate(trues)

            acc = accuracy_score(true, pred)
            f1 = f1_score(true, pred, average=None, labels=sorted(state_labels), zero_division=0)
            val_accs.append(acc); val_f1s.append(f1)
            val_conf_sum += confusion_matrix(true, pred, labels=sorted(state_labels))
            print(f"  Run {run+1}/{NUM_DECODER_RUNS}  acc={acc:.3f}")

        val_accs = np.array(val_accs)
        val_f1s = np.array(val_f1s)
        avg_conf = val_conf_sum / NUM_DECODER_RUNS

        # Permutation chance baseline
        print(f"Estimating chance accuracy ({NUM_PERMUTATIONS} permutations)...")
        all_val_labels = B_test if not USE_LAZY_LOADING else B_test
        chance_acc = np.array([
            accuracy_score(np.random.choice(all_val_labels, size=all_val_labels.shape), all_val_labels)
            for _ in range(NUM_PERMUTATIONS)
        ])
        print(f"Chance: {chance_acc.mean():.3f} ± {chance_acc.std():.3f}")

        # Save outputs
        np.savetxt(os.path.join(output_dir, f'acc_list_val_{session_dir}_unweighted.txt'), val_accs)
        np.savetxt(os.path.join(output_dir, f'acc_list_chance_{session_dir}.txt'), chance_acc)
        np.save(os.path.join(output_dir, f'all_f1_scores_val_{session_dir}_unweighted.npy'), val_f1s)

        # Summary JSON for discrete
        res = dict(
            unweighted_val_acc=float(val_accs.mean()),
            unweighted_val_acc_std=float(val_accs.std()),
            chance_acc=float(chance_acc.mean()),
        )
        with open(os.path.join(run_dir, 'discrete', 'summary_discrete.json'), 'w') as f:
            json.dump(res, f, indent=2)

        # Simple summary figure (accuracy boxplot)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.boxplot([val_accs, chance_acc], labels=['Val', 'Chance'], patch_artist=True)
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Discrete Decoder (trial-based) — {session_dir}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'summary_{session_dir}.pdf'))
        plt.close()

        return res

    metrics = {}
    if RUN_DISCRETE:
        metrics['discrete'] = train_and_eval_discrete()

    summary = dict(
        status='completed',
        completed_at=datetime.datetime.now().isoformat(),
        output_dir=run_dir,
        configuration=dict(
            data_path=data_path, downsample_fs=DOWNSAMPLE_FS, downsample_method=DOWNSAMPLE_METHOD,
            normalize_method=NORMALIZE_METHOD, window_size=WINDOW_SIZE, use_lazy_loading=USE_LAZY_LOADING,
            num_decoder_runs=NUM_DECODER_RUNS, train_epochs=TRAIN_EPOCHS, batch_size=BATCH_SIZE,
            b_mode=B_MODE,
        ),
        metrics=metrics,
    )
    with open(os.path.join(run_dir, 'run_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print('\nAll done. Results →', run_dir)


if __name__ == '__main__':
    main()
