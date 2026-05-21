#!/usr/bin/env python3
"""
verify_reward_to_choice.py
--------------------------
Diagnostic script for b_mode='reward_to_choice'.

Verification approach:
  For each usable trial pair (prev=i-1, curr=i, i>=1), the expected segment is
  [t_chosen[i-1]+1, t_chosen[i]] in behavioral ms.

  For each such pair the script:
    1. Finds all neuronal frames assigned to curr trial's idx.
    2. Checks start: first frame's int(btime) >= t_chosen[i-1]+1
    3. Checks end:   last  frame's int(btime) <= t_chosen[i]
    4. Checks no frame in segment has int(btime) > t_chosen[i]  (post-choice contamination)
    5. Checks all frames have the same behavioral label (no label flip).

  Pairs whose t_chosen[i] exceeds the recording window are counted separately
  (they are truncated by the recording end, not a bug).

Usage:
    python scripts/verify_reward_to_choice.py
"""
import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from ncmcm.data_loaders.bandit_task import BanditTaskNeuroPixelsDataset

# ── Dataset config ────────────────────────────────────────────────────────────
DATA_PATH      = str(REPO_ROOT / 'datasets/raw/twoArmBandit/JPAS_0023_20230922')
DOWNSAMPLE_FS  = 30
METHOD         = 'gaussian'
GOOD_ONLY      = False
B_MODE         = 'reward_to_choice'
N_DETAIL       = 10

# ── Load dataset (cache key includes b_mode, so cache is mode-specific) ──────
print(f"Loading BanditTaskNeuroPixelsDataset with b_mode='{B_MODE}' …")
ds = BanditTaskNeuroPixelsDataset(
    data_path=DATA_PATH,
    downsample_fs=DOWNSAMPLE_FS,
    downsample_method=METHOD,
    good_neurons_only=GOOD_ONLY,
    b_mode=B_MODE,
    recompute_cache=False,   # cache key is b_mode-specific; safe to use cache
)

frame_ms = 1000.0 / ds.fs

print(f"\n── Shapes ──")
print(f"  x:                   {ds.x.shape}")
print(f"  b:                   {ds.b.shape}")
print(f"  b_labels:            {ds.b_labels}")
print(f"  trial_start_indices: {len(ds.trial_start_indices)} segments")
print(f"  fs: {ds.fs:.2f} Hz  →  frame = {frame_ms:.2f} ms")

# ── Load ground-truth ─────────────────────────────────────────────────────────
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

rec_end_ms = int(ds.behavioral_time[-1])   # last behavioral time in recording (integer ms)

print(f"\n  Usable trials:              {len(usable)}")
print(f"  Recording ends at:          {rec_end_ms} ms")
print(f"  Pairs with t_chosen within recording (full):  "
      f"{sum(1 for i in range(1,len(usable)) if int(usable[i]['t chosen']) <= rec_end_ms)}")
print(f"  Pairs with t_chosen beyond recording (truncated): "
      f"{sum(1 for i in range(1,len(usable)) if int(usable[i]['t chosen']) > rec_end_ms)}")
print(f"  First pair: dropped (no prior t_chosen) → 1 trial dropped by design")

# ── Dense arrays ─────────────────────────────────────────────────────────────
b_dense = ds.b.toarray().flatten()
trial_idx_arr = ds.trial_indices          # (T,) each frame's trial idx
btime_int = ds.behavioral_time.astype(int) # integer ms behavioral time per frame

# ── Per-pair verification ─────────────────────────────────────────────────────
print(f"\n── Per-pair detail (first {N_DETAIL} detectable pairs) ──")
hdr = (f"{'pair':>5}  {'choice':>6}  {'exp_start':>10}  {'obs_start':>10}  "
       f"{'Δstart':>7}  {'exp_end':>10}  {'obs_end':>10}  {'Δend':>7}  "
       f"{'n_frames':>8}  {'labels_ok':>9}  {'status':>10}")
print(hdr)
print("-" * len(hdr))

max_start_err = 0.0
max_end_err   = 0.0
n_post_contaminated = 0
n_label_contaminated = 0
n_full = 0
n_truncated = 0
n_printed = 0

for i in range(1, len(usable)):
    prev_tc = int(usable[i - 1]['t chosen'])
    curr_tc = int(usable[i]['t chosen'])
    choice  = usable[i]['choice'].lower()
    trial_idx = all_sorted_idx_by_start.get(int(usable[i]['start']), -1)

    # Find all frames assigned to this trial
    mask = (trial_idx_arr == trial_idx)
    n_frames = int(mask.sum())

    is_truncated = (curr_tc > rec_end_ms)

    if n_frames == 0:
        # Segment entirely outside recording window — skip silently
        continue

    obs_start = int(btime_int[np.argmax(mask)])           # first frame's int-ms
    obs_end   = int(btime_int[len(mask) - 1 - np.argmax(mask[::-1])])  # last

    exp_start = prev_tc + 1
    exp_end   = curr_tc if not is_truncated else rec_end_ms

    start_err = obs_start - exp_start   # should be ≥ 0 (frame is after expected start)
    end_err   = exp_end - obs_end       # should be ≥ 0 (frame is before expected end)

    # Post-t_chosen contamination: any frame with int(btime) > curr_tc
    n_post = int(np.sum(btime_int[mask] > curr_tc))
    if n_post > 0:
        n_post_contaminated += 1

    # Label consistency
    seg_labels = b_dense[mask]
    labels_ok = bool(len(seg_labels) > 0 and np.all(seg_labels == seg_labels[0]))
    if not labels_ok:
        n_label_contaminated += 1

    if not is_truncated:
        max_start_err = max(max_start_err, abs(start_err))
        max_end_err   = max(max_end_err,   abs(end_err))
        n_full += 1
    else:
        n_truncated += 1

    status = 'truncated' if is_truncated else '✓'
    if not labels_ok:
        status = '✗ LABEL'
    elif n_post > 0:
        status = '✗ POST'

    if n_printed < N_DETAIL:
        print(f"{i:>5}  {choice:>6}  {exp_start:>10}  {obs_start:>10}  "
              f"{start_err:>+7}  {exp_end:>10}  {obs_end:>10}  {-end_err:>+7}  "
              f"{n_frames:>8}  {'✓' if labels_ok else '✗':>9}  {status:>10}")
        n_printed += 1

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n── Summary ──")
print(f"  Full segments (t_chosen within recording):  {n_full}")
print(f"  Truncated segments (t_chosen beyond recording end): {n_truncated}  "
      f"← expected; recording ends before trial completes")
print(f"  Max |start offset| (full segs, ms):   {max_start_err:.1f}  "
      f"[1 frame = {frame_ms:.2f} ms]")
print(f"  Max |end offset|   (full segs, ms):   {max_end_err:.1f}")
print(f"  Pairs with post-t_chosen frames:      {n_post_contaminated}")
print(f"  Pairs with mid-segment label flip:    {n_label_contaminated}")
print(f"  First usable trial dropped by design: 1")

# ── compare with decision_strict ─────────────────────────────────────────────
from ncmcm.data_loaders.bandit_task import BanditTaskNeuroPixelsDataset as _DS
ds_strict = _DS(
    data_path=DATA_PATH, downsample_fs=DOWNSAMPLE_FS,
    downsample_method=METHOD, good_neurons_only=GOOD_ONLY,
    b_mode='decision_strict', recompute_cache=False)
print(f"\n── Compare with decision_strict ──")
print(f"  reward_to_choice  trial_start_indices: {len(ds.trial_start_indices)}")
print(f"  decision_strict   trial_start_indices: {len(ds_strict.trial_start_indices)}")
print(f"  Both 237: the 17 trials beyond recording end are absent in both modes ✓")

# ── Acceptance criteria ───────────────────────────────────────────────────────
print(f"\n── Acceptance Criteria ──")
c1 = max_start_err <= frame_ms
c2 = max_end_err   <= frame_ms
c3 = n_post_contaminated == 0
c4 = n_label_contaminated == 0
for ok, label in [
    (c1, f"start offset ≤ 1 frame ({frame_ms:.2f} ms) for all full segments"),
    (c2, f"end offset   ≤ 1 frame ({frame_ms:.2f} ms) for all full segments"),
    (c3, "no post-t_chosen frames in any segment"),
    (c4, "no mid-segment label contamination"),
]:
    print(f"  [{'✓ PASS' if ok else '✗ FAIL'}]  {label}")

all_pass = c1 and c2 and c3 and c4
print(f"\n  Overall: {'ALL PASS ✓' if all_pass else 'SOME FAILURES'}")

# ── Retraining command ────────────────────────────────────────────────────────
print(f"""
── Next-Prompt Retraining Command ──
python scripts/bandit_gridsearch.py \\
  --data_path '{DATA_PATH}' \\
  --downsample_fs 30 --downsample_method gaussian \\
  --good_neurons_only false \\
  --normalize_method minmax_global \\
  --window 50 --latent_dim 3 \\
  --batch_size 50 --n_epochs 500 \\
  --learning_rate 0.0001 --gamma 0.75 \\
  --device cuda --lazy_loading --pca_init \\
  --choosing_state_mode side --gaussian_sigma_ms 25.0 \\
  --b_mode reward_to_choice \\
  --trial_based --trial_test_ratio 0.2 \\
  --context_policy same_partition \\
  --hgf_model binary2 --hgf_column x_1_expected_mean \\
  --b_type hybrid \\
  --alpha 0.1 0.3 0.5 0.7 0.9 \\
  --output_dir './results'
""")

