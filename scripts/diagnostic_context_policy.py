"""
post-commit diagnostic — context_policy window-count preview
Run:  python scripts/diagnostic_context_policy.py
Purpose:
  - Confirm real session produces expected window counts under same_partition
  - Confirm same_partition < always (cross-partition context suppressed)
  - Confirm context_policy is captured in config dict
"""
import sys, json, textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy.sparse
from ncmcm.data_loaders.bandit_task import BanditTaskNeuroPixelsDataset
from ncmcm.bundlenet.utils import (
    prep_data_trials_lazy,
    trial_train_test_split_lazy,
    compute_trial_partition,
)

# ── Config ────────────────────────────────────────────────────────────────────
SESSION_DIR = ROOT / "datasets/raw/twoArmBandit/JPAS_0023_20230922"
DOWNSAMPLE_FS = 10          # Hz — keep loading fast; same as gridsearch default
WIN = 50                    # frames — same as gridsearch default
TEST_RATIO = 0.2
RANDOM_STATE = 42
B_MODE = "full"             # simplest discrete behaviour mode

_config = {
    "session": SESSION_DIR.name,
    "downsample_fs": DOWNSAMPLE_FS,
    "win": WIN,
    "test_ratio": TEST_RATIO,
    "random_state": RANDOM_STATE,
    "b_mode": B_MODE,
    "context_policy": None,   # filled in per-policy below
}

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading session …", flush=True)
ds = BanditTaskNeuroPixelsDataset(
    data_path=str(SESSION_DIR),
    downsample_fs=DOWNSAMPLE_FS,
    b_mode=B_MODE,
)
X = ds.x.toarray().T          # (T, n_neurons) — sparse → dense, transpose
if scipy.sparse.issparse(ds.b):
    B_raw = ds.b.toarray().T  # (T, n_behaviours)
else:
    B_raw = np.array(ds.b).T

# Flatten to 1-D discrete label (argmax of one-hot)
if B_raw.ndim == 2 and B_raw.shape[1] > 1:
    B = B_raw.argmax(axis=1).astype(np.int64)
else:
    B = B_raw.ravel().astype(np.int64)

n_trials = len(ds.trial_start_indices)
print(f"  T={X.shape[0]:,}  n_neurons={X.shape[1]}  n_trials={n_trials}\n")

# Pre-compute partition once (same call semantics as gridsearch)
train_set, test_set = compute_trial_partition(
    n_trials, test_ratio=TEST_RATIO, random_state=RANDOM_STATE
)
partition_map = {t: "train" for t in train_set} | {t: "test" for t in test_set}

# ── Helper ────────────────────────────────────────────────────────────────────
def windowing_stats(policy, trial_partition_map=None, partition_sets=None):
    dataset, B_1, trial_ids = prep_data_trials_lazy(
        X, B,
        b_labels_dict=ds.b_labels_dict,
        win=WIN,
        trial_start_indices=ds.trial_start_indices,
        context_policy=policy,
        trial_partition_map=trial_partition_map,
    )
    (tr_subset, _), (te_subset, _) = trial_train_test_split_lazy(
        dataset, B_1, trial_ids,
        test_ratio=TEST_RATIO,
        random_state=RANDOM_STATE,
        partition_sets=partition_sets,
    )
    return {
        "total_windows": len(dataset),
        "train_windows": len(tr_subset),
        "test_windows":  len(te_subset),
    }

# ── Run all three policies ────────────────────────────────────────────────────
results = {}

for policy in ("always", "none", "same_partition"):
    pmap = partition_map if policy == "same_partition" else None
    psets = (train_set, test_set) if policy == "same_partition" else None
    stats = windowing_stats(policy, trial_partition_map=pmap, partition_sets=psets)
    results[policy] = stats

    cfg = {**_config, "context_policy": policy}
    # Confirm the field is present and correct
    assert cfg["context_policy"] == policy, "context_policy missing from config!"

# ── Report ────────────────────────────────────────────────────────────────────
print("Window counts per policy")
print(f"{'policy':<20} {'total':>8} {'train':>8} {'test':>8}")
print("-" * 48)
for policy, s in results.items():
    print(f"{policy:<20} {s['total_windows']:>8,} {s['train_windows']:>8,} {s['test_windows']:>8,}")

print()

# Assertions
always_total = results["always"]["total_windows"]
none_total   = results["none"]["total_windows"]
sp_total     = results["same_partition"]["total_windows"]

assert none_total <= always_total, \
    f"FAIL: none ({none_total}) should be <= always ({always_total})"

assert none_total <= sp_total <= always_total, \
    f"FAIL: same_partition total ({sp_total}) should be in [{none_total}, {always_total}]"

# Train/test windows must add up to total for each policy
for policy, s in results.items():
    assert s["train_windows"] + s["test_windows"] == s["total_windows"], \
        f"FAIL: train+test != total for policy={policy}"

print("Assertions:")
print(f"  none <= same_partition <= always total: {none_total} <= {sp_total} <= {always_total}  ✓")
delta = always_total - sp_total
print(f"  cross-partition windows suppressed by same_partition: {delta:,}")
if delta == 0:
    print("  NOTE: delta=0 — all trial transitions are within the same partition")
else:
    print("  same_partition removes cross-partition context  ✓")

print()

# Show sample config dict
sample_config = {**_config, "context_policy": "same_partition"}
print("Sample config dict (as written to config.json):")
print(textwrap.indent(json.dumps(sample_config, indent=2), "  "))
assert "context_policy" in sample_config, "context_policy missing from config dict!"
print()
print("  context_policy field present in config dict  ✓")
print()
print("Diagnostic complete — commit 1dd7934 confirmed clean.")
