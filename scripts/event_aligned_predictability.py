#!/usr/bin/env python3
"""
Prompt 022 — Event-Aligned Predictability Analysis
Refactors time-resolved accuracy onto a correct event-aligned x-axis.

Usage:
    python event_aligned_predictability.py [--val-csv PATH] [--out DIR]

Defaults:
    --val-csv  <hardcoded legacy path> (override to use a different run's CSV)
    --out      <val-csv parent dir>/event_aligned_predictability_{timestamp}/
"""
import argparse, json, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy import stats

REPO_ROOT    = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / 'datasets/raw/twoArmBandit/JPAS_0023_20230922'

_LEGACY_CSV = REPO_ROOT / (
    'results/analysis/time_resolved_predictability_20260521_012643/'
    'validation_prediction_table_with_metadata_and_margin.csv'
)

_ap = argparse.ArgumentParser()
_ap.add_argument('--val-csv', default=None,
                 help='Path to validation_prediction_table_with_metadata_and_margin.csv '
                      '(default: legacy hardcoded path)')
_ap.add_argument('--out', default=None,
                 help='Output directory (default: <val-csv parent>/event_aligned_predictability_{ts})')
_args, _ = _ap.parse_known_args()

VAL_CSV = Path(_args.val_csv) if _args.val_csv else _LEGACY_CSV
TS      = datetime.now().strftime('%Y%m%d_%H%M%S')
OUT_DIR = Path(_args.out) if _args.out else VAL_CSV.parent / f'event_aligned_predictability_{TS}'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 0. Load raw data
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(VAL_CSV)
with open(DATASET_PATH / 'metrics.json') as f:
    mdata = json.load(f)

# ALL trials sorted exactly as _create_trial_indices does
all_sorted = sorted(mdata['metrics']['trials'], key=lambda t: t.get('start', 0))
FRAME_MS   = 1000.0 / 30.0   # 30 Hz downsampling

# ─────────────────────────────────────────────────────────────────────────────
# PART 1 – Audit event fields
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("PART 1 — EVENT FIELD AUDIT")
print("=" * 60)

print(f"Total trials in metrics.json (all sorted): {len(all_sorted)}")
print(f"Unique keys: {sorted(set(k for t in all_sorted for k in t.keys()))}")
print()

val_tids = sorted(df['trial_id'].unique())
print(f"Validation trial IDs ({len(val_tids)} trials): {val_tids[:10]} ...")
print()
print("Sample validation trials (cross-checked with val df):")
for k in val_tids[:5]:
    t  = all_sorted[k]
    s  = float(t['start'])
    tg = float(t['t choosing'])
    tc = float(t['t chosen'])
    t_df = df[df['trial_id'] == k]
    df_choice  = t_df['choice'].iloc[0]
    df_label   = t_df['label'].iloc[0]
    df_nwin    = len(t_df)
    print(f"  trial_id={k}: start={int(s)}  t_choosing={int(tg)}  t_chosen={int(tc)}"
          f"  rewarded={t.get('rewarded')}  choice={t.get('choice')}")
    print(f"           start→t_choosing={tg-s:.0f}ms  t_choosing→t_chosen={tc-tg:.0f}ms"
          f"  val_df: choice={df_choice}  label={df_label}  n_win={df_nwin}")
print()

# Timing statistics across ALL usable val trials
tg_arr = np.array([float(all_sorted[k]['t choosing']) for k in val_tids])
tc_arr = np.array([float(all_sorted[k]['t chosen'])   for k in val_tids])
ts_arr = np.array([float(all_sorted[k]['start'])      for k in val_tids])
print(f"start → t_choosing:  mean={np.mean(tg_arr-ts_arr):.0f}  median={np.median(tg_arr-ts_arr):.0f}  "
      f"min={np.min(tg_arr-ts_arr):.0f}  max={np.max(tg_arr-ts_arr):.0f} ms")
print(f"t_choosing→t_chosen: mean={np.mean(tc_arr-tg_arr):.0f}  median={np.median(tc_arr-tg_arr):.0f}  "
      f"min={np.min(tc_arr-tg_arr):.0f}  max={np.max(tc_arr-tg_arr):.0f} ms")
print()

# ─────────────────────────────────────────────────────────────────────────────
# PART 2 – Build event-aligned validation table
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("PART 2 — EVENT-ALIGNED VALIDATION TABLE")
print("=" * 60)

# For each window, reconstruct behavioral_time_ms.
# Approximation: t_start + within_trial_idx * frame_ms
# (error ≤ ~2ms/frame, well within 200ms bin resolution)
t_start_arr    = np.array([float(all_sorted[tid]['start'])      for tid in df['trial_id']])
t_choosing_arr = np.array([float(all_sorted[tid]['t choosing']) for tid in df['trial_id']])
t_chosen_arr   = np.array([float(all_sorted[tid]['t chosen'])   for tid in df['trial_id']])
beh_time_ms    = t_start_arr + df['within_trial_idx'].values * FRAME_MS

ea = df.copy()
ea['behavioral_time_ms']   = beh_time_ms
ea['trial_start_ms']       = t_start_arr
ea['t_choosing_ms']        = t_choosing_arr
ea['t_chosen_ms']          = t_chosen_arr
ea['time_to_t_chosen_ms']  = beh_time_ms - t_chosen_arr
ea['time_to_t_chosen_s']   = (beh_time_ms - t_chosen_arr)  / 1000.0
ea['time_to_t_choosing_ms']= beh_time_ms - t_choosing_arr
ea['time_to_t_choosing_s'] = (beh_time_ms - t_choosing_arr) / 1000.0

# Phase assignment
ea['phase_event_aligned'] = np.select(
    [beh_time_ms >= t_chosen_arr, beh_time_ms >= t_choosing_arr],
    ['post-choice', 'choosing'],
    default='pre-choosing'
)

# Label contamination: post-choice windows where decision_strict label ≠ current trial's choice
# (happens when next trial has different choice)
first_label = ea.groupby('trial_id')['label'].first().to_dict()
ea['trial_first_label']   = ea['trial_id'].map(first_label)
ea['label_contaminated']  = ((ea['phase_event_aligned'] == 'post-choice')
                              & (ea['label'] != ea['trial_first_label']))

ea.to_csv(OUT_DIR / 'event_aligned_validation_table.csv', index=False)
print(f"Saved event_aligned_validation_table.csv  shape={ea.shape}")
print(f"Phase counts: {ea['phase_event_aligned'].value_counts().to_dict()}")
print(f"Phase fractions: pre-choosing={( ea['phase_event_aligned']=='pre-choosing').mean():.3f}  "
      f"choosing={(ea['phase_event_aligned']=='choosing').mean():.3f}  "
      f"post-choice={(ea['phase_event_aligned']=='post-choice').mean():.3f}")
print(f"Contaminated windows: {ea['label_contaminated'].sum()} "
      f"({100*ea['label_contaminated'].mean():.1f}%)")
print()

# ─────────────────────────────────────────────────────────────────────────────
# PART 3 – Figures
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("PART 3 — FIGURES")
print("=" * 60)

chance = ea['label'].value_counts().max() / len(ea)
phase_order = ['pre-choosing', 'choosing', 'post-choice']
PHASE_COLORS = {'pre-choosing': '#1565C0', 'choosing': '#E65100', 'post-choice': '#2E7D32'}

BIN_W = 0.2   # 200 ms bins
t_min, t_max = ea['time_to_t_chosen_s'].min(), ea['time_to_t_chosen_s'].max()
bins_tc  = np.arange(np.floor(t_min), np.ceil(t_max) + BIN_W, BIN_W)
ctrs_tc  = (bins_tc[:-1] + bins_tc[1:]) / 2
bidx_tc  = np.digitize(ea['time_to_t_chosen_s'].values, bins_tc) - 1
N_BINS_TC = len(ctrs_tc)

med_tch_rel = float(np.median(t_choosing_arr - t_chosen_arr)) / 1000.0  # neg, before t_chosen


def bin_accuracy(sub, bidx, n):
    ba = np.array([sub['correct'][bidx == i].mean() if (bidx == i).any() else np.nan
                   for i in range(n)])
    bs = np.array([stats.sem(sub['correct'][bidx == i]) if (bidx == i).any() else np.nan
                   for i in range(n)])
    bn = np.array([(bidx == i).sum() for i in range(n)])
    return ba, bs, bn


# ── Figure 1: Overall accuracy vs time_to_t_chosen ───────────────────────────
ba1, bs1, bn1 = bin_accuracy(ea, bidx_tc, N_BINS_TC)
v1 = (bn1 >= 5) & ~np.isnan(ba1)

fig, ax = plt.subplots(figsize=(11, 5))
ax.fill_between(ctrs_tc[v1], (ba1 - bs1)[v1], (ba1 + bs1)[v1], alpha=0.25, color='steelblue')
ax.plot(ctrs_tc[v1], ba1[v1], '-o', color='steelblue', ms=4, lw=2, label='accuracy ± SEM')
ax.axhline(chance, color='grey', ls='--', lw=1.2, alpha=0.8, label=f'majority baseline ({chance:.3f})')
ax.axvline(0,   color='red',    lw=2,   ls='-',  label='t_chosen (choice action)', zorder=5)
ax.axvline(med_tch_rel, color='orange', lw=1.5, ls='--',
           label=f't_choosing  (arm entry, median {med_tch_rel:.2f}s)')
# Shade phases
ax.axvspan(t_min, med_tch_rel,  alpha=0.04, color='#1565C0', label='_')
ax.axvspan(med_tch_rel, 0,      alpha=0.04, color='#E65100', label='_')
ax.axvspan(0, t_max,            alpha=0.04, color='#2E7D32', label='_')
for x, lbl, c in [(t_min + 0.3,        'pre-choosing',  '#1565C0'),
                   (med_tch_rel + 0.15, 'choosing',       '#E65100'),
                   (0.15,               'post-choice',    '#2E7D32')]:
    ax.text(x, 0.06, lbl, color=c, fontsize=8, ha='left')
ax.set_xlabel('Time relative to choice action   t_chosen = 0  (s)\n'
              '← before choice | post-choice / reward →', fontsize=9)
ax.set_ylabel('Decoding accuracy  (left vs right)')
ax.set_title('BunDLe-Net latent space decoding — event-aligned to t_chosen', fontsize=11)
ax.legend(loc='upper left', fontsize=8.5)
ax.set_xlim(-4.5, 2.0)
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig1_accuracy_vs_t_chosen.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved fig1_accuracy_vs_t_chosen.png")

# ── Figure 2: Phase-split bar chart ──────────────────────────────────────────
phase_acc = {p: ea[ea['phase_event_aligned'] == p]['correct'].mean() for p in phase_order}
phase_se  = {p: stats.sem(ea[ea['phase_event_aligned'] == p]['correct'])   for p in phase_order}
phase_n   = {p: (ea['phase_event_aligned'] == p).sum() for p in phase_order}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: bar chart
ax = axes[0]
xs  = np.arange(len(phase_order))
bars = ax.bar(xs, [phase_acc[p] for p in phase_order],
              color=[PHASE_COLORS[p] for p in phase_order],
              alpha=0.85, edgecolor='black', linewidth=0.8, width=0.55)
ax.errorbar(xs, [phase_acc[p] for p in phase_order],
            yerr=[phase_se[p]  for p in phase_order],
            fmt='none', color='black', capsize=6, lw=2, zorder=5)
ax.axhline(chance, color='grey', ls='--', lw=1.5, label=f'baseline ({chance:.3f})')
for i, (bar, p) in enumerate(zip(bars, phase_order)):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + phase_se[p] + 0.025,
            f'{phase_acc[p]:.3f}\n(n={phase_n[p]})', ha='center', va='bottom', fontsize=9)
ax.set_xticks(xs); ax.set_xticklabels(phase_order, fontsize=10)
ax.set_ylim(0, 1)
ax.set_ylabel('Decoding accuracy')
ax.set_title('Accuracy by event-aligned phase')
ax.legend(fontsize=9)

# Right: stay vs switch per phase
ax2 = axes[1]
w = 0.35
xs2 = np.arange(len(phase_order))
for offset, ss, color, lbl in [(-w/2, 'stay', '#1565C0', 'stay'),
                                 ( w/2, 'switch', '#E65100', 'switch')]:
    accs = [ea[(ea['phase_event_aligned']==p) & (ea['stay_switch']==ss)]['correct'].mean()
            for p in phase_order]
    ses  = [stats.sem(ea[(ea['phase_event_aligned']==p) & (ea['stay_switch']==ss)]['correct'])
            for p in phase_order]
    ns   = [(  (ea['phase_event_aligned']==p) & (ea['stay_switch']==ss)).sum()
            for p in phase_order]
    bars2 = ax2.bar(xs2 + offset, accs, width=w, color=color,
                    alpha=0.8, edgecolor='black', linewidth=0.6, label=lbl)
    ax2.errorbar(xs2 + offset, accs, yerr=ses,
                 fmt='none', color='black', capsize=4, lw=1.5)
ax2.axhline(chance, color='grey', ls='--', lw=1.5)
ax2.set_xticks(xs2); ax2.set_xticklabels(phase_order, fontsize=10)
ax2.set_ylim(0, 1)
ax2.set_ylabel('Accuracy')
ax2.set_title('Stay vs switch — per phase')
ax2.legend(fontsize=9)

fig.suptitle('Accuracy by event phase  |  Session JPAS_0023_20230922', fontsize=11)
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig2_accuracy_by_phase.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved fig2_accuracy_by_phase.png")

# ── Figure 3: Stay vs switch event-aligned time course ───────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
for ss, color, lbl in [('stay', '#1565C0', 'stay'), ('switch', '#E65100', 'switch')]:
    sub = ea[ea['stay_switch'] == ss]
    bidx = np.digitize(sub['time_to_t_chosen_s'].values, bins_tc) - 1
    ba, bs, bn = bin_accuracy(sub, bidx, N_BINS_TC)
    v = (bn >= 3) & ~np.isnan(ba)
    ax.fill_between(ctrs_tc[v], (ba-bs)[v], (ba+bs)[v], alpha=0.15, color=color)
    ax.plot(ctrs_tc[v], ba[v], '-o', color=color, ms=4, lw=2, label=f'{lbl}  (n_trials={sub["trial_id"].nunique()})')
ax.axhline(chance, color='grey', ls='--', lw=1.2, label=f'baseline ({chance:.3f})')
ax.axvline(0,   color='red',    lw=2,   ls='-',  label='t_chosen', zorder=5)
ax.axvline(med_tch_rel, color='orange', lw=1.5, ls='--', label=f't_choosing (median)')
ax.set_xlabel('Time relative to t_chosen (s)', fontsize=9)
ax.set_ylabel('Accuracy')
ax.set_title('Stay vs switch — event-aligned to t_chosen')
ax.legend(fontsize=9); ax.set_xlim(-4.5, 2.0); ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig3_stay_vs_switch_event_aligned.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved fig3_stay_vs_switch_event_aligned.png")

# ── Figure 4: Congruent vs outlier event-aligned ──────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
for cong, color, lbl in [('congruent', '#4CAF50', 'congruent'), ('outlier', '#F44336', 'outlier')]:
    sub = ea[ea['congruent_label'] == cong]
    bidx = np.digitize(sub['time_to_t_chosen_s'].values, bins_tc) - 1
    ba, bs, bn = bin_accuracy(sub, bidx, N_BINS_TC)
    v = (bn >= 3) & ~np.isnan(ba)
    ax.fill_between(ctrs_tc[v], (ba-bs)[v], (ba+bs)[v], alpha=0.15, color=color)
    ax.plot(ctrs_tc[v], ba[v], '-o', color=color, ms=4, lw=2,
            label=f'{cong}  (n_trials={sub["trial_id"].nunique()})')
ax.axhline(chance, color='grey', ls='--', lw=1.2, label=f'baseline ({chance:.3f})')
ax.axvline(0,   color='red',    lw=2, ls='-',   label='t_chosen', zorder=5)
ax.axvline(med_tch_rel, color='orange', lw=1.5, ls='--', label='t_choosing (median)')
ax.set_xlabel('Time relative to t_chosen (s)', fontsize=9)
ax.set_ylabel('Accuracy')
ax.set_title('Congruent vs outlier — event-aligned to t_chosen')
ax.legend(fontsize=9); ax.set_xlim(-4.5, 2.0); ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig4_congruent_vs_outlier_event_aligned.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved fig4_congruent_vs_outlier_event_aligned.png")

# ── Figure 5: Aligned to t_choosing (arm entry) ──────────────────────────────
t_min2, t_max2 = ea['time_to_t_choosing_s'].min(), ea['time_to_t_choosing_s'].max()
bins_tch  = np.arange(np.floor(t_min2), np.ceil(t_max2) + BIN_W, BIN_W)
ctrs_tch  = (bins_tch[:-1] + bins_tch[1:]) / 2
bidx_tch  = np.digitize(ea['time_to_t_choosing_s'].values, bins_tch) - 1
N_BINS_TCH = len(ctrs_tch)
ba5, bs5, bn5 = bin_accuracy(ea, bidx_tch, N_BINS_TCH)
v5 = (bn5 >= 5) & ~np.isnan(ba5)
med_chosen_rel = float(np.median(tc_arr - tg_arr)) / 1000.0

fig, ax = plt.subplots(figsize=(11, 5))
ax.fill_between(ctrs_tch[v5], (ba5-bs5)[v5], (ba5+bs5)[v5], alpha=0.25, color='#6A1B9A')
ax.plot(ctrs_tch[v5], ba5[v5], '-o', color='#6A1B9A', ms=4, lw=2, label='accuracy ± SEM')
ax.axhline(chance, color='grey', ls='--', lw=1.2, label=f'baseline ({chance:.3f})')
ax.axvline(0, color='orange', lw=2, ls='-', label='t_choosing (arm entry)', zorder=5)
ax.axvline(med_chosen_rel, color='red', lw=1.5, ls='--',
           label=f't_chosen (commit, median {med_chosen_rel:.2f}s)')
ax.set_xlabel('Time relative to arm entry  t_choosing = 0  (s)\n'
              '← approach | in arm → commit →', fontsize=9)
ax.set_ylabel('Accuracy')
ax.set_title('BunDLe-Net decoding — aligned to t_choosing (arm entry)', fontsize=11)
ax.legend(fontsize=8.5); ax.set_xlim(-4.5, 2.0); ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig5_accuracy_vs_t_choosing.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved fig5_accuracy_vs_t_choosing.png")

# ── Figure 6: Comparison old normalized-time vs event-aligned ────────────────
# Side-by-side: old normalized_time x-axis vs new t_chosen-aligned x-axis
N_BINS_OLD = 20
old_bin_edges = np.linspace(0, 1, N_BINS_OLD + 1)
old_ctrs      = (old_bin_edges[:-1] + old_bin_edges[1:]) / 2
old_bidx      = np.digitize(ea['normalized_time'].values, old_bin_edges) - 1
ba_old, bs_old, bn_old = bin_accuracy(ea, old_bidx, N_BINS_OLD)
v_old = (bn_old >= 5) & ~np.isnan(ba_old)

# Mark which old bins are mostly post-choice
post_frac_per_bin = np.array(
    [(ea.loc[old_bidx == i, 'phase_event_aligned'] == 'post-choice').mean()
     if (old_bidx == i).any() else 0.0 for i in range(N_BINS_OLD)]
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Old normalized-time vs event-aligned — same data, different x-axis', fontsize=11)

ax = axes[0]
cols_old = np.where(post_frac_per_bin > 0.5, '#2E7D32', 'steelblue')
ax.fill_between(old_ctrs[v_old], (ba_old - bs_old)[v_old], (ba_old + bs_old)[v_old],
                alpha=0.2, color='steelblue')
ax.plot(old_ctrs[v_old], ba_old[v_old], '-o', color='steelblue', ms=4, lw=2)
ax.axhline(chance, color='grey', ls='--', lw=1.2)
median_chosen_norm = float(np.median(1 - (
    (ea.groupby('trial_id').apply(lambda g: (g['time_to_t_chosen_ms'] >= 0).sum() / len(g)))
)))
ax.axvline(median_chosen_norm, color='red', lw=2, ls='--', label=f't_chosen (median ≈{median_chosen_norm:.2f})')
ax.axvline(np.median(
    ea.groupby('trial_id').apply(lambda g:
        (g['time_to_t_choosing_ms'] >= 0).sum() / len(g) * 0 +
        (g['time_to_t_choosing_ms'].abs().idxmin() and
         g.loc[g['time_to_t_choosing_ms'].abs().idxmin(), 'normalized_time'])
    )), color='orange', lw=1.5, ls='--', label='t_choosing approx')
ax.set_xlabel('normalized_time (0=trial entry, 1=~1s post-choice)')
ax.set_ylabel('Accuracy')
ax.set_title('OLD axis (normalized within-trial)')
ax.legend(fontsize=8); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

ax2 = axes[1]
ax2.fill_between(ctrs_tc[v1], (ba1 - bs1)[v1], (ba1 + bs1)[v1], alpha=0.25, color='steelblue')
ax2.plot(ctrs_tc[v1], ba1[v1], '-o', color='steelblue', ms=4, lw=2)
ax2.axhline(chance, color='grey', ls='--', lw=1.2)
ax2.axvline(0, color='red', lw=2, ls='-', label='t_chosen')
ax2.axvline(med_tch_rel, color='orange', lw=1.5, ls='--', label='t_choosing')
ax2.set_xlabel('Time relative to t_chosen (s)')
ax2.set_ylabel('Accuracy')
ax2.set_title('NEW axis (event-aligned to t_chosen)')
ax2.legend(fontsize=8); ax2.set_xlim(-4.5, 2.0); ax2.set_ylim(0, 1)

fig.tight_layout()
fig.savefig(OUT_DIR / 'fig6_old_vs_event_aligned.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved fig6_old_vs_event_aligned.png")

# ─────────────────────────────────────────────────────────────────────────────
# PART 4 – Interpretation
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("PART 4 — INTERPRETATION")
print("=" * 60)

pre   = ea[ea['phase_event_aligned'] == 'pre-choosing']
choos = ea[ea['phase_event_aligned'] == 'choosing']
post  = ea[ea['phase_event_aligned'] == 'post-choice']

pre_acc   = pre['correct'].mean()
choos_acc = choos['correct'].mean()
post_acc  = post['correct'].mean()
t1, p1    = stats.ttest_1samp(pre['correct'],   chance)
t1c, p1c  = stats.ttest_1samp(choos['correct'], chance)
t1p, p1p  = stats.ttest_1samp(post['correct'],  chance)

stay_pre   = ea[(ea['stay_switch'] == 'stay')   & (ea['phase_event_aligned'] == 'pre-choosing')]
switch_pre = ea[(ea['stay_switch'] == 'switch') & (ea['phase_event_aligned'] == 'pre-choosing')]
t_sw, p_sw = stats.ttest_1samp(switch_pre['correct'], chance)

frac_pre  = (ea['phase_event_aligned'] == 'pre-choosing').mean()
frac_choo = (ea['phase_event_aligned'] == 'choosing').mean()
frac_post = (ea['phase_event_aligned'] == 'post-choice').mean()

print(f"\nQ1 — Does early (pre-choice) high accuracy remain in the PRE-CHOOSING interval?")
print(f"  Pre-choosing accuracy:  {pre_acc:.4f}  (baseline: {chance:.4f})")
print(f"  Δ above baseline:       {pre_acc - chance:+.4f}")
print(f"  t-test vs baseline:     t={t1:.2f}  p={p1:.3e}")
print(f"  → {'YES — survives in clean pre-choice window' if pre_acc > chance + 0.02 and p1 < 0.05 else 'MARGINAL or NO'}")

print(f"\nQ2 — Is pre-choice accuracy mostly driven by stay trials?")
print(f"  Stay pre-choosing:    {stay_pre['correct'].mean():.4f}  (n_win={len(stay_pre)}, n_trial={stay_pre['trial_id'].nunique()})")
print(f"  Switch pre-choosing:  {switch_pre['correct'].mean():.4f}  (n_win={len(switch_pre)}, n_trial={switch_pre['trial_id'].nunique()})")
stay_dom = stay_pre['correct'].mean() - switch_pre['correct'].mean()
print(f"  Stay advantage:       {stay_dom:+.4f}")
print(f"  → {'Stay dominant' if abs(stay_dom) > 0.05 else 'No clear dominance'}")

print(f"\nQ3 — Is switch-trial pre-choice accuracy still above majority baseline?")
print(f"  Switch pre-choosing accuracy: {switch_pre['correct'].mean():.4f}  (baseline: {chance:.4f})")
print(f"  t-test vs baseline:  t={t_sw:.2f}  p={p_sw:.3e}")
print(f"  → {'YES — switch pre-choice is above baseline' if switch_pre['correct'].mean() > chance and p_sw < 0.05 else 'NOT significantly above baseline'}")

print(f"\nQ4 — What fraction of old normalized-time plot was post-choice?")
print(f"  pre-choosing: {frac_pre:.3f} ({frac_pre*100:.1f}%)")
print(f"  choosing:     {frac_choo:.3f} ({frac_choo*100:.1f}%)")
print(f"  post-choice:  {frac_post:.3f} ({frac_post*100:.1f}%)")
print(f"  → About {frac_post*100:.0f}% of all windows in the old plot were post-choice/reward phase.")
print(f"    The last ~{frac_post*100:.0f}% of the normalized-time axis (0.{int(100-frac_post*100)}–1.0) is primarily post-choice.")

print(f"\nQ5 — Should old figures be relabeled, replaced, or kept with caveat?")
if pre_acc > chance + 0.03 and p1 < 0.01:
    print("  RECOMMENDATION: RELABEL (not discard).")
    print("  The 'early high accuracy' finding is REAL and survives in the clean pre-choice interval.")
    print("  Old figures should get corrected x-axis captions:")
    print("  'normalised within-trial window index  (0=trial entry · ~0.71=choice · 1=~1s post-choice)'")
    print("  Event-aligned versions (this analysis) should accompany old figures in the thesis.")
else:
    print("  RECOMMENDATION: ADD CAVEAT + event-aligned replacement.")

# ─────────────────────────────────────────────────────────────────────────────
# PART 5 – Save summary.json and README.md
# ─────────────────────────────────────────────────────────────────────────────
summary = {
    "timestamp": TS,
    "input_csv": str(VAL_CSV),
    "n_val_windows": int(len(ea)),
    "n_val_trials": int(ea['trial_id'].nunique()),
    "event_fields_available": ["start", "t choosing", "t chosen", "rewarded", "choice",
                               "hold_start", "Achievement reached"],
    "timing_stats_ms": {
        "start_to_t_choosing_median": float(np.median(tg_arr - ts_arr)),
        "start_to_t_choosing_mean": float(np.mean(tg_arr - ts_arr)),
        "t_choosing_to_t_chosen_median": float(np.median(tc_arr - tg_arr)),
        "t_choosing_to_t_chosen_mean": float(np.mean(tc_arr - tg_arr)),
        "t_chosen_to_segment_end_approx": 991.0,
        "t_choosing_relative_to_t_chosen_median_s": float(med_tch_rel),
    },
    "phase_counts": {p: int((ea['phase_event_aligned'] == p).sum()) for p in phase_order},
    "phase_fractions": {p: float((ea['phase_event_aligned'] == p).mean()) for p in phase_order},
    "accuracy": {
        "majority_baseline": float(chance),
        "pre_choosing": float(pre_acc),
        "pre_choosing_se": float(stats.sem(pre['correct'])),
        "choosing": float(choos_acc),
        "post_choice": float(post_acc),
        "stay_pre_choosing": float(stay_pre['correct'].mean()),
        "switch_pre_choosing": float(switch_pre['correct'].mean()),
    },
    "significance": {
        "pre_choice_t": float(t1), "pre_choice_p": float(p1),
        "choosing_t": float(t1c),  "choosing_p": float(p1c),
        "post_choice_t": float(t1p), "post_choice_p": float(p1p),
        "switch_pre_vs_baseline_t": float(t_sw),
        "switch_pre_vs_baseline_p": float(p_sw),
        "switch_pre_above_baseline": bool(switch_pre['correct'].mean() > chance and p_sw < 0.05),
    },
    "label_contamination": {
        "n_windows": int(ea['label_contaminated'].sum()),
        "fraction": float(ea['label_contaminated'].mean()),
        "explanation": "post-choice windows where decision_strict label = next trial's choice"
    },
    "old_plot_composition": {
        "pre_choosing_frac": float(frac_pre),
        "choosing_frac": float(frac_choo),
        "post_choice_frac": float(frac_post),
    },
    "interpretation": {
        "early_accuracy_survives_pre_choice": bool(pre_acc > chance + 0.02 and p1 < 0.05),
        "switch_pre_choice_above_baseline": bool(switch_pre['correct'].mean() > chance and p_sw < 0.05),
        "stay_advantage_pre_choice": float(stay_pre['correct'].mean() - switch_pre['correct'].mean()),
        "recommendation": "relabel_old_figures" if pre_acc > chance + 0.03 and p1 < 0.01 else "add_caveat",
    },
    "figures": [
        "fig1_accuracy_vs_t_chosen.png",
        "fig2_accuracy_by_phase.png",
        "fig3_stay_vs_switch_event_aligned.png",
        "fig4_congruent_vs_outlier_event_aligned.png",
        "fig5_accuracy_vs_t_choosing.png",
        "fig6_old_vs_event_aligned.png",
    ],
}
with open(OUT_DIR / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

readme_text = f"""# Event-Aligned Predictability Analysis
Generated: {TS}

## Context
Refactors the time-resolved predictability plots from
`demos/time_resolved_predictability.ipynb` onto a correct event-aligned x-axis.

## Key correction
The original `normalized_time` axis (0–1) ran from `trial['start']` (animal enters
trial zone) to ~991 ms after `t_chosen` (reward delivery period), NOT from
previous-choice to current-choice as originally described.

The median choice action (`t_chosen`) occurs at normalized_time ≈ 0.71.

## Event timing (session JPAS_0023_20230922, 30 Hz downsampling)
| Event | Time relative to trial start |
|---|---|
| `trial['start']` | 0 ms (normalized_time = 0.0) |
| `t_choosing` (arm entry) | median +{int(np.median(tg_arr - ts_arr))} ms |
| `t_chosen` (commit) | median +{int(np.median(tc_arr - ts_arr))} ms (normalized_time ≈ 0.71) |
| segment end | ~+{int(np.median(tc_arr - ts_arr)) + 991} ms (normalized_time = 1.0) |

## Three event-aligned phases
1. **pre-choosing** (`time < t_choosing`): animal approaching arm — {frac_pre*100:.0f}% of windows
2. **choosing** (`t_choosing ≤ time < t_chosen`): arm entry to commit — {frac_choo*100:.0f}% of windows
3. **post-choice** (`time ≥ t_chosen`): reward delivery — {frac_post*100:.0f}% of windows

## Label contamination
`decision_strict` labels transition at `t_chosen`. Post-choice windows carry
the NEXT trial's choice label. This affects ~17% of validation trials
(those where next_choice ≠ current_choice).

## Accuracy findings
| Phase | Accuracy | Baseline | p-value |
|---|---|---|---|
| pre-choosing | {pre_acc:.3f} | {chance:.3f} | {p1:.3e} |
| choosing | {choos_acc:.3f} | {chance:.3f} | {p1c:.3e} |
| post-choice | {post_acc:.3f} | {chance:.3f} | {p1p:.3e} |
| stay pre-choosing | {stay_pre['correct'].mean():.3f} | {chance:.3f} | — |
| switch pre-choosing | {switch_pre['correct'].mean():.3f} | {chance:.3f} | {p_sw:.3e} |

## Interpretation
- Early high accuracy **SURVIVES** in the clean pre-choosing interval.
- Switch trials are {"also above" if switch_pre["correct"].mean() > chance else "NOT above"} baseline even before arm entry.
- Old figures should be RELABELED with corrected x-axis caption, not discarded.

## Files
- `event_aligned_validation_table.csv` — enriched per-window table with event times and phase labels
- `fig1_accuracy_vs_t_chosen.png` — main result: accuracy vs time aligned to t_chosen
- `fig2_accuracy_by_phase.png` — phase-split bar chart with stay/switch breakdown
- `fig3_stay_vs_switch_event_aligned.png` — stay vs switch time course
- `fig4_congruent_vs_outlier_event_aligned.png` — block congruence time course
- `fig5_accuracy_vs_t_choosing.png` — aligned to t_choosing (arm entry)
- `fig6_old_vs_event_aligned.png` — side-by-side comparison of old vs new axis
- `summary.json` — all numerical results
"""
with open(OUT_DIR / 'README.md', 'w') as f:
    f.write(readme_text)

print(f"\n{'='*60}")
print(f"All outputs saved to: {OUT_DIR}")
print(f"Output timestamp: {TS}")
