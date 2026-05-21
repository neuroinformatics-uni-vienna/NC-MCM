#!/usr/bin/env python3
"""
preflight_reward_to_choice.py
------------------------------
Preflight verification before launching BunDLe-Net training with
b_mode='reward_to_choice'.

Checks:
  1. Git commit and status
  2. Dataset loads correctly with b_mode='reward_to_choice'
  3. Segment boundaries align to t_chosen (via verify_reward_to_choice logic)
  4. No post-t_chosen frames in any segment
  5. Train/val split stats (trial_random_state=42, trial_test_ratio=0.2)
  6. Label counts per partition

Saves report to:
  logs/preflight_reward_to_choice_{timestamp}.txt

Exits with code 0 if all checks pass, code 1 on failure.

Usage:
    python scripts/preflight_reward_to_choice.py
"""
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path
from io import StringIO

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH         = str(REPO_ROOT / 'datasets/raw/twoArmBandit/JPAS_0023_20230922')
DOWNSAMPLE_FS     = 30
METHOD            = 'gaussian'
GOOD_ONLY         = False
B_MODE            = 'reward_to_choice'
WINDOW            = 50
LATENT_DIM        = 3
BATCH_SIZE        = 50
N_EPOCHS          = 500
LR                = 0.0001
GAMMA             = 0.75
TRIAL_TEST_RATIO  = 0.2
TRIAL_RANDOM_STATE = 42
CONTEXT_POLICY    = 'same_partition'
HGF_MODEL         = 'binary2'
HGF_COLUMN        = 'x_1_expected_mean'
NORMALIZE_METHOD  = 'minmax_global'
ALPHA_HYBRID      = 0.5

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
LOGS_DIR  = REPO_ROOT / 'logs'
LOGS_DIR.mkdir(exist_ok=True)
REPORT_PATH = LOGS_DIR / f'preflight_reward_to_choice_{TIMESTAMP}.txt'

OUTPUT_HYBRID_NAME   = f'grid_search_{TIMESTAMP}_same_partition_reward_to_choice_hybrid_alpha_050'
OUTPUT_DISCRETE_NAME = f'grid_search_{TIMESTAMP}_same_partition_reward_to_choice_discrete_only'


class Tee:
    """Write to both stdout and a buffer simultaneously."""
    def __init__(self):
        self._buf = StringIO()

    def write(self, s):
        sys.stdout.write(s)
        self._buf.write(s)

    def flush(self):
        sys.stdout.flush()

    def getvalue(self):
        return self._buf.getvalue()


tee = Tee()


def p(*args, **kwargs):
    kwargs.setdefault('file', tee)
    print(*args, **kwargs)


# ── Step 1: Git info ──────────────────────────────────────────────────────────
p("=" * 72)
p("PREFLIGHT: reward_to_choice BunDLe-Net training")
p(f"Timestamp: {TIMESTAMP}")
p("=" * 72)

p("\n-- Git info --")
try:
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip()
    commit_msg = subprocess.check_output(
        ['git', 'log', '-1', '--pretty=%s'], cwd=REPO_ROOT, text=True).strip()
    branch = subprocess.check_output(
        ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=REPO_ROOT, text=True).strip()
    status = subprocess.check_output(
        ['git', 'status', '--short'], cwd=REPO_ROOT, text=True).strip()
    p(f"  Branch:  {branch}")
    p(f"  Commit:  {commit}")
    p(f"  Message: {commit_msg}")
    if status:
        p(f"  Status (dirty):\n{status}")
    else:
        p("  Status:  clean")
    git_ok = True
except Exception as e:
    p(f"  ERROR reading git info: {e}")
    git_ok = False

# ── Step 2: Load dataset ──────────────────────────────────────────────────────
p("\n-- Dataset loading --")
from ncmcm.data_loaders.bandit_task import BanditTaskNeuroPixelsDataset

ds = BanditTaskNeuroPixelsDataset(
    data_path=DATA_PATH,
    downsample_fs=DOWNSAMPLE_FS,
    downsample_method=METHOD,
    good_neurons_only=GOOD_ONLY,
    b_mode=B_MODE,
    hgf_model=HGF_MODEL,
    hgf_column=HGF_COLUMN,
    recompute_cache=False,
)

x_shape = ds.x.shape
b_shape = ds.b.shape
n_segments = len(ds.trial_start_indices)
frame_ms = 1000.0 / ds.fs

p(f"  b_mode:              {ds.b_mode}")
p(f"  x shape (n,T):       {x_shape}")
p(f"  b shape (1,T):       {b_shape}")
p(f"  b_labels:            {ds.b_labels}")
p(f"  fs:                  {ds.fs:.4f} Hz  ->  frame = {frame_ms:.2f} ms")
p(f"  trial_start_indices: {n_segments} segments")
p(f"  hgf_beliefs:         {'present, shape=' + str(ds.hgf_beliefs.shape) if ds.hgf_beliefs is not None else 'None'}")

# ── Step 3: Segment boundary verification ────────────────────────────────────
p("\n-- Segment boundary verification --")

with open(Path(DATA_PATH) / 'metrics.json') as f:
    metrics = json.load(f)

all_trials = metrics['metrics']['trials']
all_sorted = sorted(all_trials, key=lambda t: t.get('start', 0))
all_sorted_idx_by_start = {int(t.get('start', 0)): i for i, t in enumerate(all_sorted)}

usable = [
    t for t in all_trials
    if t.get('start') is not None
    and t.get('t chosen') is not None
    and t.get('choice', '').lower() in ['l', 'r']
]
usable.sort(key=lambda t: t['start'])
p(f"  Total usable trial pairs (after dropping trial 0): {len(usable) - 1}")

rec_end_ms = int(ds.behavioral_time[-1])
p(f"  Recording end: {rec_end_ms} ms")

btime_int   = ds.behavioral_time.astype(int)
b_dense     = ds.b.toarray().flatten()
trial_idx_arr = ds.trial_indices

max_start_err = 0.0
max_end_err   = 0.0
n_post_contaminated  = 0
n_label_contaminated = 0
n_full      = 0
n_truncated = 0

for i in range(1, len(usable)):
    prev_tc   = int(usable[i - 1]['t chosen'])
    curr_tc   = int(usable[i]['t chosen'])
    trial_idx = all_sorted_idx_by_start.get(int(usable[i]['start']), -1)
    mask      = (trial_idx_arr == trial_idx)
    n_frames  = int(mask.sum())
    if n_frames == 0:
        continue
    is_truncated = (curr_tc > rec_end_ms)
    obs_start = int(btime_int[np.argmax(mask)])
    obs_end   = int(btime_int[len(mask) - 1 - np.argmax(mask[::-1])])
    exp_start = prev_tc + 1
    exp_end   = curr_tc if not is_truncated else rec_end_ms
    start_err = abs(obs_start - exp_start)
    end_err   = abs(obs_end - exp_end)
    n_post    = int(np.sum(btime_int[mask] > curr_tc))
    labels_ok = bool(len(b_dense[mask]) > 0 and np.all(b_dense[mask] == b_dense[mask][0]))
    if not labels_ok:
        n_label_contaminated += 1
    if n_post > 0:
        n_post_contaminated += 1
    if not is_truncated:
        max_start_err = max(max_start_err, start_err)
        max_end_err   = max(max_end_err, end_err)
        n_full += 1
    else:
        n_truncated += 1

c_start  = max_start_err <= frame_ms
c_end    = max_end_err   <= frame_ms
c_post   = n_post_contaminated == 0
c_labels = n_label_contaminated == 0

p(f"  Full segments: {n_full}  |  Truncated (recording boundary): {n_truncated}")
p(f"  Max start offset (full): {max_start_err:.1f} ms  [<= 1 frame = {frame_ms:.2f} ms]  {'PASS' if c_start else 'FAIL'}")
p(f"  Max end offset   (full): {max_end_err:.1f} ms  [<= 1 frame]  {'PASS' if c_end else 'FAIL'}")
p(f"  Post-t_chosen frames:    {n_post_contaminated} segments  {'PASS' if c_post else 'FAIL'}")
p(f"  Label contamination:     {n_label_contaminated} segments  {'PASS' if c_labels else 'FAIL'}")
seg_ok = c_start and c_end and c_post and c_labels

# ── Step 4: Train/val split stats ────────────────────────────────────────────
p("\n-- Train/validation split stats --")
p(f"  trial_test_ratio:    {TRIAL_TEST_RATIO}")
p(f"  trial_random_state:  {TRIAL_RANDOM_STATE}")
p(f"  context_policy:      {CONTEXT_POLICY}")

from ncmcm.bundlenet.utils import (
    segments_from_trial_starts, prep_data_trials, trial_train_test_split,
    compute_trial_partition,
)

x_dense = ds.x.T.toarray().astype(np.float32)
b_flat  = b_dense.copy()

segments = segments_from_trial_starts(x_dense, b_flat, ds.trial_start_indices)
p(f"  Segments loaded: {len(segments)}")

train_set, test_set = compute_trial_partition(len(segments), TRIAL_TEST_RATIO, TRIAL_RANDOM_STATE)
partition_map = {tid: ('train' if tid in train_set else 'test') for tid in range(len(segments))}

X_paired, B_1, trial_ids = prep_data_trials(
    segments, win=WINDOW,
    context_policy=CONTEXT_POLICY,
    trial_partition_map=partition_map,
)
p(f"  Total windows after prep_data_trials: {X_paired.shape[0]}")
p(f"  X_paired shape: {X_paired.shape}")
p(f"  B_1 shape:      {B_1.shape}")

(x_train, b_train), (x_test, b_test) = trial_train_test_split(
    X_paired, B_1, trial_ids,
    test_ratio=TRIAL_TEST_RATIO,
    random_state=TRIAL_RANDOM_STATE,
    partition_sets=(train_set, test_set),
)
train_ids = trial_ids[np.isin(trial_ids, list(train_set))]
test_ids  = trial_ids[np.isin(trial_ids, list(test_set))]
n_train_trials = len(np.unique(train_ids))
n_test_trials  = len(np.unique(test_ids))

p(f"  Train windows: {x_train.shape[0]}  ({n_train_trials} trials)")
p(f"  Test  windows: {x_test.shape[0]}   ({n_test_trials} trials)")

# Label counts
for split_name, b_split, ids_split in [
    ('train', b_train, train_ids),
    ('test',  b_test,  test_ids),
]:
    unique_ids, counts = np.unique(b_split, return_counts=True)
    label_str = ', '.join(
        f"'{ds.b_labels[int(u)]}'={c}" for u, c in zip(unique_ids, counts)
    )
    p(f"  {split_name} label counts: {label_str}")

# First/last 5 train trial IDs
train_unique = sorted(train_set)
test_unique  = sorted(test_set)
p(f"  Train trial IDs (first 5): {train_unique[:5]}")
p(f"  Train trial IDs (last  5): {train_unique[-5:]}")
p(f"  Test  trial IDs (first 5): {test_unique[:5]}")
p(f"  Test  trial IDs (last  5): {test_unique[-5:]}")

split_ok = (x_train.shape[0] > 0 and x_test.shape[0] > 0)

# ── Step 5: Print exact training commands ────────────────────────────────────
BASE_ARGS = (
    f'  --data_path "{DATA_PATH}" \\\n'
    f'  --downsample_fs {DOWNSAMPLE_FS} --downsample_method {METHOD} \\\n'
    f'  --good_neurons_only false \\\n'
    f'  --normalize_method {NORMALIZE_METHOD} \\\n'
    f'  --window {WINDOW} --latent_dim {LATENT_DIM} \\\n'
    f'  --batch_size {BATCH_SIZE} --n_epochs {N_EPOCHS} \\\n'
    f'  --learning_rate {LR} --gamma {GAMMA} \\\n'
    f'  --device cuda --lazy_loading --pca_init \\\n'
    f'  --choosing_state_mode side --gaussian_sigma_ms 25.0 \\\n'
    f'  --b_mode {B_MODE} \\\n'
    f'  --trial_based --trial_test_ratio {TRIAL_TEST_RATIO} '
    f'--trial_random_state {TRIAL_RANDOM_STATE} \\\n'
    f'  --context_policy {CONTEXT_POLICY} \\\n'
    f'  --hgf_model {HGF_MODEL} --hgf_column {HGF_COLUMN}'
)

CMD_HYBRID = (
    f'python scripts/bandit_gridsearch.py \\\n'
    f'{BASE_ARGS} \\\n'
    f'  --b_type hybrid --alpha {ALPHA_HYBRID} \\\n'
    f'  --output_dir "./results/{OUTPUT_HYBRID_NAME}"'
)

CMD_DISCRETE = (
    f'python scripts/bandit_gridsearch.py \\\n'
    f'{BASE_ARGS} \\\n'
    f'  --b_type discrete \\\n'
    f'  --output_dir "./results/{OUTPUT_DISCRETE_NAME}"'
)

p("\n-- Exact training commands --")
p("\n[1] Hybrid (alpha=0.5):")
p(CMD_HYBRID)
p("\n[2] Discrete-only control:")
p(CMD_DISCRETE)

# ── Step 6: Overall pass/fail ─────────────────────────────────────────────────
p("\n" + "=" * 72)
p("PREFLIGHT SUMMARY")
p("=" * 72)
all_ok = git_ok and seg_ok and split_ok
checks = [
    (git_ok,   "Git commit readable"),
    (c_start,  "Segment start offset <= 1 frame"),
    (c_end,    "Segment end offset   <= 1 frame (full segs)"),
    (c_post,   "No post-t_chosen frames"),
    (c_labels, "No mid-segment label contamination"),
    (split_ok, "Train/test split non-empty"),
]
for ok, label in checks:
    p(f"  [{'PASS' if ok else 'FAIL'}]  {label}")

p(f"\n  OVERALL: {'ALL CHECKS PASSED — safe to launch training' if all_ok else 'PREFLIGHT FAILED — do NOT launch training'}")
p(f"\n  Report saved to: {REPORT_PATH}")

# ── Save report ───────────────────────────────────────────────────────────────
REPORT_PATH.write_text(tee.getvalue())

sys.exit(0 if all_ok else 1)
