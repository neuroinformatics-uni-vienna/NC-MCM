# Bandit Loader Refactor Plan

**Document version:** 2026-05-21  
**Status:** Scaffolding complete (Prompt 036). No behavior changes yet.  
**Relevant code:** `ncmcm/data_loaders/bandit_task.py`, `ncmcm/data_loaders/bandit_specs.py`

---

## Executive Summary

The current `BanditTaskNeuroPixelsDataset` conflates two orthogonal design dimensions inside the `b_mode` parameter:

1. **Segment policy** — where each training window starts and ends (controls `trial_indices`, `trial_start_indices`, and thus all `prep_data_trials` windows).
2. **Target policy** — what behavioral label is attached to each timepoint (controls `b`, `b_labels_dict`, `b_labels`).

This coupling causes **silent mismatches** between label coverage and training window coverage. Only `b_mode='reward_to_choice'` is internally consistent. All other modes suffer from some form of label/window misalignment.

This document catalogues the coupling points, formalizes the mismatch, proposes a two-layer architecture, and provides a phased migration path that preserves full backward compatibility.

---

## 1. Coupling Point Catalogue

| # | Location | Coupling |
|---|----------|---------|
| CP-1 | `_create_behavioral_data_matrix` (line 721) | Dispatches on `b_mode` to set both segment boundaries AND label vocabulary in one pass. Single parameter controls two independent dimensions. |
| CP-2 | `_create_trial_indices` (line 1285) | Uses `[trial.start, t_chosen]` for ALL non-`reward_to_choice` modes, regardless of what segment boundaries the labels actually use. |
| CP-3 | `_apply_state_transitions` (line 355) | Post-hoc state fusion via pattern matching. Separation from label creation is correct, but fused states are not registered in `DEFAULT_TRANSITION_MAP`, causing false negatives in `check_state_transitions()`. |
| CP-4 | `choosing_state_mode` woven into four methods | `_create_behavioral_data_matrix`, `_create_decision_behavioral_data_matrix`, `_create_decision_strict_behavioral_data_matrix`, `_create_reward_to_choice_behavioral_data_matrix` each contain separate `choosing_state_mode` logic. |
| CP-5 | `b_mode` controls applicability of `state_transitions` | Whether `state_transitions` is meaningful depends on `b_mode` (only `'full'` uses per-event labels). For `'decision'`/`'decision_strict'`/`'reward_to_choice'`, passing non-empty `state_transitions` has no effect but is silently ignored. |
| CP-6 | `bandit_gridsearch.py:load_data()` uses string → dict translation | `apply_hold_transitions` string is mapped to a `state_transitions` dict via a hardcoded `transition_lookup`. This lookup is incomplete (does not cover T2/T2b state names). |

---

## 2. Label–Window Mismatch Table

The critical issue: `_create_trial_indices` does not always use the same boundaries as `_create_behavioral_data_matrix`.

| `b_mode` | Label boundary | `trial_indices` boundary | Consistent? | Consequence |
|----------|---------------|--------------------------|------------|-------------|
| `'full'` | per-event (reward/no-reward labeled at all T) | `[trial.start, t_chosen]` | ❌ | reward/no-reward timepoints are labeled but have `trial_index=-1`. They never appear in `prep_data_trials` windows. |
| `'decision'` | `[trial.start, next_trial.start − 1]` (full lifecycle) | `[trial.start, t_chosen]` | ❌ | Reward/no-reward period gets a choice label but `trial_index=-1`. They are excluded from training windows. The model never sees reward-period activity during training. |
| `'decision_strict'` | trial 0: `[start, t_chosen]`; trial i>0: `[prev_t_chosen+1, t_chosen]` | always `[trial.start, t_chosen]` | ❌ (for i>0) | For trial i>0: reward/no-reward period of trial i-1 is in the label window but `trial_index[i-1]` — excluded from trial-i windows in `prep_data_trials`. |
| `'reward_to_choice'` | `[prev_t_chosen+1, t_chosen]` for i≥1 | `[prev_t_chosen+1, t_chosen]` for i≥1 (via `_create_reward_to_choice_trial_indices`) | ✅ | **Fully consistent.** Current production configuration. |

**Quantitative impact (JPAS_0023_20230922, 30 Hz gaussian, T0 config):**

| Metric | Value |
|--------|-------|
| Total session timepoints | 38,794 |
| Reward timepoints (full session) | 4,709 (12.1%) |
| No-reward timepoints (full session) | 11,955 (30.8%) |
| Reward timepoints in trial windows | ~4 (0.0%) |
| No-reward timepoints in trial windows | ~1 (0.0%) |
| Imbalance ratio (hold vs no-reward, T0) | 11.0× SEVERE |
| Imbalance ratio (T1) | 2.5× |
| Imbalance ratio (T2) | 3.7× |
| Imbalance ratio (T2b) | 2.1× |

The reward/no-reward states account for ~43% of the full session but are **effectively absent** from training windows in the T0 baseline configuration.

---

## 3. Proposed Architecture

### Three independent dimensions

```
           ┌─────────────────────────────┐
           │  BanditTrainingConfig        │
           │                             │
           │  SegmentPolicy  (WHERE)      │  ← controls trial_indices, prep_data windows
           │  TargetPolicy   (WHAT)       │  ← controls b, b_labels
           │  WindowPolicy   (HOW)        │  ← controls context_policy, seed (future)
           └─────────────────────────────┘
```

**SegmentPolicy** answers: *Which timepoints form a training segment, and what is the trial boundary?*

**TargetPolicy** answers: *What is the behavioral prediction target for each timepoint?*

**WindowPolicy** (future) answers: *How are sliding windows sampled from segments?*

### Segment Policies (formal definitions)

| Name | Start event | End event | Trial 0 | Reward period included | Current implementation |
|------|------------|-----------|---------|----------------------|----------------------|
| `start_to_choice` | `trial.start` | `t_chosen` | include | No (stops at t_chosen) | `_create_trial_indices` default (non-rtc) |
| `lifecycle_start_to_next_start` | `trial.start` | `next_trial.start − 1` | include | Yes | Labels only (`b_mode='decision'`). **Not used for trial_indices anywhere.** |
| `prev_choice_to_choice_mixed` | trial 0: `trial.start`; trial i>0: `prev_t_chosen+1` | `t_chosen` | include | Yes (leading context of next) | Labels only (`b_mode='decision_strict'`). trial_indices use `start_to_choice`. |
| `reward_to_choice` | `prev_t_chosen + 1` | `t_chosen` | drop | No (reward of prev trial = leading context) | `_create_reward_to_choice_trial_indices` + `_create_reward_to_choice_behavioral_data_matrix` |
| `post_choice_reward` | `t_chosen` | `next_trial.start − 1` | include | Yes (entire period) | **Not implemented** |
| `choosing_only` | `t_choosing` | `t_chosen` | include | No | **Not implemented** |

### Target Policies (formal definitions)

| Name | Type | Labels | Current `b_mode` | `choosing_state_mode` | `state_transitions` |
|------|------|--------|-----------------|----------------------|---------------------|
| `phase_full_side` | discrete | intertrial, hold, choosing L/R, reward, no reward | `full` | `side` | none |
| `phase_hold_choice_fused` | discrete | intertrial, reward, no reward, hold→choosing L/R | `full` | `side` | HOLD_TO_CHOOSING |
| `choice_outcome_fused` | discrete | intertrial, hold, choosing L/R → reward/no reward | `full` | `side` | CHOOSING_TO_OUTCOME |
| `choice_correctness_fused` | discrete | intertrial, hold, choosing reward/no reward | `full` | `side` | CHOOSING_TO_CORRECTNESS |
| `choice_side` | discrete | choosing left, choosing right | `reward_to_choice` | `side` | none |
| `choice_correctness` | discrete | choosing correct, choosing wrong | `reward_to_choice` | `correctness` | none |
| `hgf_belief_signed` | continuous | continuous in [-1, 1] | (hgf_beliefs array) | — | — |

### BanditSpec Compatibility Matrix

| Spec name | Segment policy | Target policy | Consistent? |
|-----------|---------------|---------------|------------|
| `T0_phase_full_side` | `start_to_choice` | `phase_full_side` | ❌ |
| `T1_hold_choice_fused` | `start_to_choice` | `phase_hold_choice_fused` | ❌ |
| `T2_choice_outcome_fused` | `start_to_choice` | `choice_outcome_fused` | ❌ |
| `T2b_choice_correctness_fused` | `start_to_choice` | `choice_correctness_fused` | ❌ |
| `decision_lifecycle` | `lifecycle_start_to_next_start` | `choice_side` | ❌ (trial_indices don't match) |
| `decision_strict` | `prev_choice_to_choice_mixed` | `choice_side` | ❌ (trial_indices don't match for i>0) |
| `reward_to_choice` | `reward_to_choice` | `choice_side` | ✅ |
| `reward_to_choice_hybrid` | `reward_to_choice` | `hgf_belief_signed` (+`choice_side`) | ✅ |
| `T0_phase_full_side_lifecycle` (proposed) | `lifecycle_start_to_next_start` | `phase_full_side` | ✅ (if trial_indices fixed) |

---

## 4. Migration Path

### Phase 1 — Description layer (CURRENT, Prompt 036)

**Status:** Complete. No behavior changes.

- `ncmcm/data_loaders/bandit_specs.py` provides `SegmentPolicy`, `TargetPolicy`, `BanditSpec` dataclasses.
- All existing specs are registered with their known issues.
- `old_params_to_spec()` and `spec_to_old_params()` enable bidirectional translation.
- `scripts/diagnose_bandit_segment_target_specs.py` validates specs against actual data.

### Phase 2 — Separate `segment_policy` from `b_mode`

**Status:** Proposed.

Add a `segment_policy: str` parameter to `BanditTaskNeuroPixelsDataset.__init__`:

```python
def __init__(
    self,
    ...,
    b_mode: str = 'full',
    segment_policy: Optional[str] = None,  # NEW; overrides trial_indices boundary
    ...
):
```

When `segment_policy` is provided:
- `_create_trial_indices` uses the specified segment boundary instead of `b_mode`-derived default.
- This fixes the mismatch for all modes without breaking existing behavior (default `segment_policy=None` = current behavior).

Changes required:
1. Add `segment_policy` param to `__init__`, `_get_cache_filename`.
2. Modify `_create_trial_indices` to dispatch on `segment_policy` when set.
3. Add `segment_policy` to cache key.
4. Update `bandit_gridsearch.py` to pass `segment_policy` from CLI.

### Phase 3 — Introduce explicit `TargetPolicy` param (future)

**Status:** Not yet designed.

Replace `b_mode` / `choosing_state_mode` / `state_transitions` with a single `target_policy: str` parameter referencing `TARGET_POLICIES`.

### Phase 4 — Deprecate `b_mode`

**Status:** Long-term.

Once `segment_policy` + `target_policy` cover all existing configurations, `b_mode` becomes a deprecated compatibility shim resolved via `old_params_to_spec()`.

---

## 5. Recommended Training Configurations

| Use case | Recommended spec | Reasoning |
|----------|-----------------|-----------|
| Upcoming choice prediction (production) | `reward_to_choice` | Only fully consistent; segment and labels fully aligned |
| Upcoming choice + HGF (production) | `reward_to_choice_hybrid` | Adds continuous belief target |
| Full behavioral state decoding | `T0_phase_full_side_lifecycle` (proposed) | Would include reward/no-reward in training windows |
| Hold vs choice discrimination | `T1_hold_choice_fused` | Reduces imbalance but still excludes reward period |
| Outcome discrimination | `T2_choice_outcome_fused` | Straddle issue — use only if outcome-period activity is not needed |
| Correctness discrimination | `reward_to_choice` with `choosing_state_mode='correctness'` | Clean, consistent |

**Critical recommendation:** Any experiment intending to include reward/no-reward activity in training windows should:
1. Wait for Phase 2 (`segment_policy` parameter) to be implemented, OR
2. Use a session-level filter: load `b_mode='full'` and create custom trial windows that extend to `next_trial.start`.
3. Do NOT use `b_mode='decision'` as a workaround — the trial_indices mismatch means reward-period frames are still excluded.

---

## 6. Impact on `check_state_transitions()`

The `DEFAULT_TRANSITION_MAP` in `bandit_task.py` was designed for `b_mode='full'` with no transitions. It maps transition pairs to expected fused state names for the currently known simple transitions (hold→choosing).

**Known false negatives:**
- T2 fused states (`'choosing left --> reward'` etc.) are NOT in `DEFAULT_TRANSITION_MAP`.
- T2b fused states (`'choosing reward'`, `'choosing no reward'`) are NOT in `DEFAULT_TRANSITION_MAP`.
- `check_state_transitions()` returns `False` for T2/T2b — but the data is correct.

**Fix:** Update `DEFAULT_TRANSITION_MAP` to include CHOOSING_TO_OUTCOME and CHOOSING_TO_CORRECTNESS fused state names. This is a one-line dict extension, not a behavior change.

---

## 7. Files

| File | Status | Purpose |
|------|--------|---------|
| `ncmcm/data_loaders/bandit_specs.py` | New (Prompt 036) | Pure description layer |
| `scripts/diagnose_bandit_segment_target_specs.py` | New (Prompt 036) | Per-spec diagnostics |
| `scripts/target_mode_diagnostics.py` | Existing | T0/T1/T2/T2b counts and plots |
| `scripts/verify_reward_to_choice.py` | Existing | 4-criteria reward_to_choice validation |
| `ncmcm/data_loaders/bandit_task.py` | Unchanged (Prompt 036) | Core loader (to be extended in Phase 2) |

---

## Appendix: `_create_trial_indices` current dispatch logic

```python
def _create_trial_indices(self, metrics, neuronal_length, translation_indices):
    if self.b_mode == 'reward_to_choice':
        return self._create_reward_to_choice_trial_indices(...)
    # ALL other modes use:
    for trial in usable_trials:
        start_time = int(trial['start'])    # ← always trial.start
        end_time   = int(trial['t chosen']) # ← always t_chosen
        trial_indices_ms[start_time:end_time + 1] = trial_idx
```

The Phase 2 change would add:
```python
    elif self.segment_policy == 'lifecycle_start_to_next_start':
        return self._create_lifecycle_trial_indices(...)
    elif self.segment_policy == 'reward_to_choice':
        return self._create_reward_to_choice_trial_indices(...)
    else:  # 'start_to_choice' (default)
        ...  # current default logic
```
