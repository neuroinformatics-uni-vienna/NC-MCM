"""
Behaviour decoding evaluation from a BunDLeNet run folder.

Usage:
    python bandit_behaviour_decoding.py <path_to_run_folder>

Loads pre-saved latent trajectories and behaviour labels from the run folder,
trains linear decoders (N=10 independent runs), and saves results under
<run_folder>/data/decoding/.

Evaluates:
  - Discrete state classification (always)
  - HGF belief regression (only if hgf_belief_{train,validation}.npy exist)
"""

import sys
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, r2_score

# ── Config ────────────────────────────────────────────────────────────────────
N_RUNS = 10
N_EPOCHS = 200
BATCH_SIZE = 256
LR = 0.01
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_loader(X, y, dtype_y):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=dtype_y)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=True)


def predict_all(model, X, squeeze=False):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        out = model(X_t).cpu().numpy()
    if squeeze:
        return out.squeeze()
    return out


# ── Discrete decoder ──────────────────────────────────────────────────────────

def run_discrete_decoding(X_train, y_train, X_val, y_val, n_classes):
    print(f"\n=== Discrete decoding ({n_classes} classes, {N_RUNS} runs) ===")
    latent_dim = X_train.shape[1]
    state_labels = np.arange(n_classes)

    val_acc_list = []
    val_f1_list  = []

    loader_train = make_loader(X_train, y_train.astype(np.int64), torch.long)

    for run in range(N_RUNS):
        model = nn.Linear(latent_dim, n_classes).to(DEVICE)
        opt   = optim.Adam(model.parameters(), lr=LR)
        crit  = nn.CrossEntropyLoss()

        model.train()
        for _ in range(N_EPOCHS):
            for xb, yb in loader_train:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                crit(model(xb), yb).backward()
                opt.step()

        logits = predict_all(model, X_val)
        pred   = logits.argmax(axis=1)
        true   = y_val.astype(np.int64)

        acc = accuracy_score(true, pred)
        f1  = f1_score(true, pred, average=None, labels=state_labels, zero_division=0)

        val_acc_list.append(acc)
        val_f1_list.append(f1)
        print(f"  Run {run+1:2d}/{N_RUNS}  acc={acc:.3f}")

    val_acc_list = np.array(val_acc_list)
    val_f1_list  = np.array(val_f1_list)

    print(f"\nDiscrete val acc: {val_acc_list.mean():.3f} ± {val_acc_list.std():.3f}")
    return val_acc_list, val_f1_list


# ── HGF decoder ───────────────────────────────────────────────────────────────

def run_hgf_decoding(X_train, hgf_train, X_val, hgf_val):
    print(f"\n=== HGF belief decoding ({N_RUNS} runs) ===")
    latent_dim = X_train.shape[1]
    mse_fn = nn.MSELoss()

    val_r2_list = []

    loader_train = make_loader(
        X_train, hgf_train.astype(np.float32).reshape(-1, 1), torch.float32
    )

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

        pred = predict_all(model, X_val, squeeze=True)
        r2   = r2_score(hgf_val, pred)

        val_r2_list.append(r2)
        print(f"  Run {run+1:2d}/{N_RUNS}  R²={r2:.3f}")

    val_r2_list = np.array(val_r2_list)
    print(f"\nHGF val R²: {val_r2_list.mean():.3f} ± {val_r2_list.std():.3f}")
    return val_r2_list


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python bandit_behaviour_decoding.py <run_folder>")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    if not run_dir.is_dir():
        print(f"ERROR: not a directory: {run_dir}")
        sys.exit(1)

    # Load config
    with open(run_dir / 'config.json') as f:
        config = json.load(f)
    latent_dim = config['latent_dim']
    print(f"Run folder : {run_dir.name}")
    print(f"Latent dim : {latent_dim}  |  Device: {DEVICE}")

    # Load latent trajectories
    X_train = np.load(run_dir / 'data' / 'latent_trajectories_train.npy')
    X_val   = np.load(run_dir / 'data' / 'latent_trajectories_validation.npy')

    # Load discrete labels
    y_train = np.load(run_dir / 'data' / 'behaviour_labels_train.npy').astype(int)
    y_val   = np.load(run_dir / 'data' / 'behaviour_labels_validation.npy').astype(int)

    n_classes = int(max(y_train.max(), y_val.max())) + 1
    print(f"Samples    : train={len(X_train)}, val={len(X_val)}")
    print(f"Classes    : {n_classes}  labels={sorted(np.unique(y_train).tolist())}")

    # Check for HGF files
    hgf_train_path = run_dir / 'data' / 'hgf_belief_train.npy'
    hgf_val_path   = run_dir / 'data' / 'hgf_belief_validation.npy'
    has_hgf = hgf_train_path.exists() and hgf_val_path.exists()
    if has_hgf:
        hgf_train = np.load(hgf_train_path).astype(np.float32)
        hgf_val   = np.load(hgf_val_path).astype(np.float32)
        print(f"HGF belief : found  range=[{hgf_train.min():.3f}, {hgf_train.max():.3f}]")
    else:
        print("HGF belief : not found — skipping HGF decoding")

    # Output directory
    out_dir = run_dir / 'data' / 'decoding'
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Discrete decoding ─────────────────────────────────────────────────────
    val_acc, val_f1 = run_discrete_decoding(X_train, y_train, X_val, y_val, n_classes)

    np.save(out_dir / 'discrete_accuracies.npy', val_acc)
    np.save(out_dir / 'discrete_f1_scores.npy',  val_f1)

    discrete_summary = {
        'n_runs': N_RUNS,
        'n_epochs': N_EPOCHS,
        'n_classes': n_classes,
        'val_acc_mean': float(val_acc.mean()),
        'val_acc_std':  float(val_acc.std()),
        'val_f1_mean_per_class': val_f1.mean(axis=0).tolist(),
    }
    with open(out_dir / 'discrete_summary.json', 'w') as f:
        json.dump(discrete_summary, f, indent=2)

    # ── HGF decoding ──────────────────────────────────────────────────────────
    if has_hgf:
        val_r2 = run_hgf_decoding(X_train, hgf_train, X_val, hgf_val)

        np.save(out_dir / 'hgf_r2_scores.npy', val_r2)

        hgf_summary = {
            'n_runs': N_RUNS,
            'n_epochs': N_EPOCHS,
            'val_r2_mean': float(val_r2.mean()),
            'val_r2_std':  float(val_r2.std()),
        }
        with open(out_dir / 'hgf_summary.json', 'w') as f:
            json.dump(hgf_summary, f, indent=2)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"RESULTS saved to: {out_dir}")
    print(f"  Discrete acc : {val_acc.mean():.3f} ± {val_acc.std():.3f}")
    if has_hgf:
        print(f"  HGF R²       : {val_r2.mean():.3f} ± {val_r2.std():.3f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

