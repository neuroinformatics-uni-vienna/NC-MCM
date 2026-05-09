"""Shared utilities for microvariable evaluation scripts.

This module centralises duplicated dataset/prediction helpers used by
`scripts/bandit_microvariable_evaluation.py` and
`scripts/bandit_microvariable_evaluation_trial_based.py`.

Set module-level config attributes from the caller scripts after import:

  import ncmcm.microvariable_common as microcommon
  microcommon.USE_LAZY_LOADING = USE_LAZY_LOADING
  microcommon.BATCH_SIZE = BATCH_SIZE
  microcommon.NUM_WORKERS = NUM_WORKERS
  microcommon.device = device

Defaults are provided so the functions work even if the caller doesn't
explicitly configure all attributes (useful for tests).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Configurable by caller scripts (set after import)
USE_LAZY_LOADING: bool = True
BATCH_SIZE: int = 256
NUM_WORKERS: int = 0
device = torch.device('cpu')


class FoldDataset(Dataset):
    """Extract channel-1 windows on demand, flatten, return tensor.

    dtype_str: 'long'   for discrete labels (CrossEntropy)  — returns (x, long scalar)
               'float'  for continuous labels (MSE)         — returns (x, float [1])
               'hybrid' for joint labels (CE + MSE)         — returns (x, float [2])
                        col 0 = class index (float), col 1 = belief
    """
    def __init__(self, subset, b_labels, dtype_str='long'):
        self.subset = subset
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
    """Build DataLoader (lazy) or list-of-batches (eager).

    Callers should set `USE_LAZY_LOADING`, `BATCH_SIZE` and `NUM_WORKERS`
    on this module before using the function if the defaults are not
    appropriate.
    """
    if USE_LAZY_LOADING:
        ds = FoldDataset(x_fold, b_fold, dtype_str)
        return DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0,
        )
    else:
        # Eager: x_fold shape (T, 2, win, N); use channel 1
        X1 = x_fold[:, 1, :, :]          # (T, win, N)
        X1_flat = X1.reshape(X1.shape[0], -1)
        x_t = torch.FloatTensor(X1_flat)
        if dtype_str == 'long':
            b_t = torch.LongTensor(b_fold)
        elif dtype_str == 'float':
            b_t = torch.FloatTensor(b_fold).unsqueeze(1)
        else:  # hybrid
            b_t = torch.FloatTensor(b_fold)
        return list(zip(
            [x_t[i:i + BATCH_SIZE] for i in range(0, len(x_t), BATCH_SIZE)],
            [b_t[i:i + BATCH_SIZE] for i in range(0, len(b_t), BATCH_SIZE)],
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
    """Run hybrid model; return (disc_preds, cont_preds, disc_true, cont_true)."""
    model.eval()
    disc_preds, cont_preds, disc_true, cont_true = [], [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            disc_preds.append(out[:, :n_classes].argmax(dim=1).cpu().numpy())
            cont_preds.append(out[:, n_classes].cpu().numpy())
            disc_true.append(yb[:, 0].long().numpy())
            cont_true.append(yb[:, 1].numpy())
    return (np.concatenate(disc_preds), np.concatenate(cont_preds),
            np.concatenate(disc_true),  np.concatenate(cont_true))


def fold_size(fold_split):
    """Return number of samples regardless of lazy/eager."""
    return len(fold_split) if USE_LAZY_LOADING else fold_split.shape[0]


def _predict_disc(model, loader, n_classes):
    """Run hybrid model, return only (disc_preds, disc_true)."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            preds.append(out[:, :n_classes].argmax(dim=1).cpu().numpy())
            trues.append(yb[:, 0].long().numpy())
    return np.concatenate(preds), np.concatenate(trues)
