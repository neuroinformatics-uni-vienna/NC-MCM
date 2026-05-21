"""
bandit_specs.py — Formal descriptions of segment and target policies for the
bandit-task loader.

This is a **pure description module**.  It does not change any existing
behavior.  Its purpose is to:

  1. Give canonical names to the segment / target combinations that the
     current loader already implements (or should implement).
  2. Record the compatibility mapping between old b_mode / state_transitions
     parameters and the new policy names.
  3. Document known issues (primarily: where segment boundaries used for
     trial_indices differ from segment boundaries used for labels).

Use ``diagnose_bandit_segment_target_specs.py`` to validate these specs
against actual dataset outputs.

Architecture overview
---------------------
Three orthogonal dimensions describe a training configuration:

  SegmentPolicy
      Where does each training segment start and end?
      Controls: trial_indices, trial_start_indices, prep_data_trials windows.

  TargetPolicy
      What behavioral label / value is attached to each timepoint?
      Controls: b (sparse label array), b_labels_dict, b_labels.

  WindowPolicy  (future; not yet formalized)
      How are sliding windows extracted from segments?
      Controls: window size, context_policy, train/val split seed.

The current loader conflates SegmentPolicy and TargetPolicy inside ``b_mode``.
``b_mode='full'`` is a TargetPolicy (per-event labels) but uses one fixed
SegmentPolicy (start_to_choice).  ``b_mode='reward_to_choice'`` is both at
once (segment AND target match).

Known critical coupling bug
---------------------------
For ``b_mode='full'``, ``'decision'``, and ``'decision_strict'``, the
``_create_trial_indices`` method always uses the ``start_to_choice`` boundary
``[trial.start, t_chosen]``.  But the *label* coverage differs:

  b_mode='full'             labels: per-event (reward/no-reward included everywhere)
                            trial_indices: [start, t_chosen]
                            → reward/no-reward timepoints at trial_index=-1

  b_mode='decision'         labels: [start, next_start-1] (full lifecycle)
                            trial_indices: [start, t_chosen]
                            → reward/no-reward timepoints labeled as 'choosing X'
                              but still at trial_index=-1

  b_mode='decision_strict'  labels: trial 0: [start, t_chosen],
                                    trial i>0: [prev_t_chosen+1, t_chosen]
                            trial_indices: always [start, t_chosen]
                            → for trial i>0, reward period of prev trial is
                              labeled (included in label window) but at
                              trial_index=-1 (excluded from trial window)

  b_mode='reward_to_choice' labels: [prev_t_chosen+1, t_chosen] for i>=1
                            trial_indices: same (delegates to
                              _create_reward_to_choice_trial_indices)
                            → FULLY CONSISTENT ✓
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Segment policies
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SegmentPolicy:
    """Describes where each training segment starts and ends.

    Attributes
    ----------
    name : str
        Canonical name for this policy.
    description : str
        Human-readable description.
    segment_start_event : str
        Which event marks the start of each segment (informal notation).
    segment_end_event : str
        Which event marks the end of each segment (inclusive).
    first_trial_policy : str
        What happens to trial 0: ``'include'`` or ``'drop'``.
    includes_reward_period : bool
        True if the segment window extends past t_chosen into the reward/
        no-reward period of the *current* trial.
    trial_index_method : str
        Which internal method currently provides trial_indices for this policy.
        ``'start_to_choice'`` means ``_create_trial_indices`` default (non-rtc).
        ``'reward_to_choice'`` means ``_create_reward_to_choice_trial_indices``.
        ``'not_implemented'`` means no current method matches.
    current_b_modes_for_labels : tuple[str, ...]
        Which ``b_mode`` values produce labels whose coverage matches this
        segment policy.
    current_b_modes_for_trial_indices : tuple[str, ...]
        Which ``b_mode`` values make trial_indices match this segment policy.
    known_mismatch : str
        Non-empty if label coverage does not match trial_indices coverage.
    """
    name: str
    description: str
    segment_start_event: str
    segment_end_event: str
    first_trial_policy: str
    includes_reward_period: bool
    trial_index_method: str
    current_b_modes_for_labels: Tuple[str, ...] = field(default=())
    current_b_modes_for_trial_indices: Tuple[str, ...] = field(default=())
    known_mismatch: str = field(default='')


# Predefined segment policies

SP_START_TO_CHOICE = SegmentPolicy(
    name='start_to_choice',
    description=(
        'Trial window spans [trial.start, t_chosen] (inclusive). '
        'The reward/no-reward period (after t_chosen) is NOT covered. '
        'This is the default trial_indices boundary for b_mode=full, '
        'decision, and decision_strict.'
    ),
    segment_start_event='trial.start',
    segment_end_event='t_chosen',
    first_trial_policy='include',
    includes_reward_period=False,
    trial_index_method='start_to_choice',
    current_b_modes_for_labels=(),           # no b_mode labels cover only this range
    current_b_modes_for_trial_indices=('full', 'decision', 'decision_strict'),
)

SP_LIFECYCLE = SegmentPolicy(
    name='lifecycle_start_to_next_start',
    description=(
        'Full trial lifecycle: [trial.start, next_trial.start − 1]. '
        'Last trial extends to last_timestamp_ms. '
        'Covers intertrial + hold + choosing + reward/no-reward. '
        'Used by b_mode=decision for LABELS but NOT for trial_indices. '
        'CRITICAL: There is currently no b_mode that uses this boundary '
        'for trial_indices. If used as segment policy, reward/no-reward '
        'frames would be included in prep_data_trials windows.'
    ),
    segment_start_event='trial.start',
    segment_end_event='next_trial.start - 1  (or last_timestamp_ms)',
    first_trial_policy='include',
    includes_reward_period=True,
    trial_index_method='not_implemented',
    current_b_modes_for_labels=('decision',),
    current_b_modes_for_trial_indices=(),    # NOT used for trial_indices anywhere
    known_mismatch=(
        'b_mode=decision labels use this boundary but trial_indices use '
        'start_to_choice. Reward/no-reward timepoints are labeled as '
        'choosing-X but have trial_index=-1 — excluded from prep_data_trials.'
    ),
)

SP_DECISION_STRICT = SegmentPolicy(
    name='prev_choice_to_choice_mixed',
    description=(
        'Trial 0: [trial.start, t_chosen]. '
        'Trial i>0: [t_chosen[i-1]+1, t_chosen[i]]. '
        'Used by b_mode=decision_strict for LABELS. '
        'trial_indices always use start_to_choice, so for trial i>0 the '
        'reward period of trial i-1 is in the label window but outside '
        'the trial window.'
    ),
    segment_start_event='trial 0: trial.start; trial i>0: prev_t_chosen + 1',
    segment_end_event='t_chosen',
    first_trial_policy='include (trial 0 uses trial.start)',
    includes_reward_period=True,   # includes reward of prev trial as leading context
    trial_index_method='start_to_choice',
    current_b_modes_for_labels=('decision_strict',),
    current_b_modes_for_trial_indices=(),    # trial_indices don't match
    known_mismatch=(
        'b_mode=decision_strict labels use [prev_t_chosen+1, t_chosen] for '
        'trial i>0, but trial_indices use [trial.start, t_chosen]. '
        'The reward/no-reward period of trial i-1 is in the label window '
        'but at trial_index[trial_i-1].start..trial_i-1 — it carries '
        'trial i-1 index, not trial i index. So prep_data_trials will '
        'not include it in trial i windows.'
    ),
)

SP_REWARD_TO_CHOICE = SegmentPolicy(
    name='reward_to_choice',
    description=(
        'Trial i >= 1: [t_chosen[i-1]+1, t_chosen[i]]. '
        'Trial 0 is dropped (no prior t_chosen). '
        'Labels AND trial_indices use identical boundaries. '
        'FULLY CONSISTENT. Current production configuration.'
    ),
    segment_start_event='prev_t_chosen + 1',
    segment_end_event='t_chosen',
    first_trial_policy='drop',
    includes_reward_period=False,  # ends exactly at t_chosen
    trial_index_method='reward_to_choice',
    current_b_modes_for_labels=('reward_to_choice',),
    current_b_modes_for_trial_indices=('reward_to_choice',),
    known_mismatch='',             # Fully consistent ✓
)

SP_POST_CHOICE_REWARD = SegmentPolicy(
    name='post_choice_reward',
    description=(
        'Segment spans [t_chosen, next_trial.start − 1]: the reward/no-reward '
        'period immediately after the decision. '
        'NOT currently implemented in the loader.'
    ),
    segment_start_event='t_chosen',
    segment_end_event='next_trial.start - 1',
    first_trial_policy='include',
    includes_reward_period=True,
    trial_index_method='not_implemented',
    current_b_modes_for_labels=(),
    current_b_modes_for_trial_indices=(),
    known_mismatch='Not implemented.',
)

SP_CHOOSING_ONLY = SegmentPolicy(
    name='choosing_to_chosen',
    description=(
        'Pure choosing interval: [t_choosing, t_chosen]. '
        'NOT currently implemented in the loader.'
    ),
    segment_start_event='t_choosing',
    segment_end_event='t_chosen',
    first_trial_policy='include',
    includes_reward_period=False,
    trial_index_method='not_implemented',
    current_b_modes_for_labels=(),
    current_b_modes_for_trial_indices=(),
    known_mismatch='Not implemented.',
)

SEGMENT_POLICIES: dict[str, SegmentPolicy] = {
    sp.name: sp for sp in [
        SP_START_TO_CHOICE,
        SP_LIFECYCLE,
        SP_DECISION_STRICT,
        SP_REWARD_TO_CHOICE,
        SP_POST_CHOICE_REWARD,
        SP_CHOOSING_ONLY,
    ]
}


# ---------------------------------------------------------------------------
# Target policies
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TargetPolicy:
    """Describes what behavioral label is attached to each timepoint.

    Attributes
    ----------
    name : str
        Canonical name for this policy.
    target_type : str
        ``'discrete'``, ``'continuous'``, or ``'hybrid'``.
    description : str
        Human-readable description.
    label_vocabulary : tuple[str, ...]
        Full set of possible labels for discrete targets.
    current_b_mode : Optional[str]
        Which ``b_mode`` currently produces this target, or None.
    current_choosing_state_mode : Optional[str]
        Required ``choosing_state_mode`` for this target, or None.
    current_state_transitions_key : Optional[str]
        Required ``apply_hold_transitions`` key for this target, or None.
    notes : str
        Additional notes or caveats.
    """
    name: str
    target_type: str
    description: str
    label_vocabulary: Tuple[str, ...]
    current_b_mode: Optional[str] = field(default=None)
    current_choosing_state_mode: Optional[str] = field(default=None)
    current_state_transitions_key: Optional[str] = field(default=None)
    notes: str = field(default='')


# Predefined target policies

TP_PHASE_FULL_SIDE = TargetPolicy(
    name='phase_full_side',
    target_type='discrete',
    description=(
        'Per-timepoint behavioral phase labels, split by choice side. '
        'Full vocabulary: intertrial, hold, choosing left, choosing right, '
        'reward, no reward. Historical DAP-style baseline.'
    ),
    label_vocabulary=('intertrial', 'hold', 'choosing left', 'choosing right',
                      'reward', 'no reward'),
    current_b_mode='full',
    current_choosing_state_mode='side',
    current_state_transitions_key='none',
    notes=(
        'For b_mode=full with trial-based training: reward and no-reward '
        'timepoints are in the label vocabulary but have trial_index=-1 '
        '(after t_chosen). They will NOT appear in prep_data_trials windows.'
    ),
)

TP_PHASE_HOLD_CHOICE_FUSED = TargetPolicy(
    name='phase_hold_choice_fused',
    target_type='discrete',
    description=(
        'Hold and choosing phases are merged into a single combined state. '
        'Vocabulary: intertrial, reward, no reward, '
        'hold --> choosing left, hold --> choosing right.'
    ),
    label_vocabulary=('intertrial', 'reward', 'no reward',
                      'hold --> choosing left', 'hold --> choosing right'),
    current_b_mode='full',
    current_choosing_state_mode='side',
    current_state_transitions_key='HOLD_TO_CHOOSING_TRANSITIONS',
    notes=(
        'Same reward/no-reward exclusion issue as phase_full_side: '
        'these states are outside trial windows. '
        'The fused states (hold+choosing) ARE within trial windows.'
    ),
)

TP_CHOICE_OUTCOME_FUSED = TargetPolicy(
    name='choice_outcome_fused',
    target_type='discrete',
    description=(
        'Choosing phase and outcome are merged. '
        'Vocabulary: intertrial, hold, '
        'choosing left --> reward, choosing left --> no reward, '
        'choosing right --> reward, choosing right --> no reward.'
    ),
    label_vocabulary=('intertrial', 'hold',
                      'choosing left --> reward', 'choosing left --> no reward',
                      'choosing right --> reward', 'choosing right --> no reward'),
    current_b_mode='full',
    current_choosing_state_mode='side',
    current_state_transitions_key='CHOOSING_TO_OUTCOME_TRANSITIONS',
    notes=(
        'The fused state straddles t_chosen: the choosing half is within '
        '[start, t_chosen] and the reward half is after t_chosen. '
        'In trial-based training with current segment policy, the reward '
        'half of each fused state has trial_index=-1. '
        'Practically: low but non-zero counts of fused states appear in '
        'train/val because the choosing half is within trial windows. '
        'DEFAULT_TRANSITION_MAP does not cover these states — '
        'check_state_transitions() will report false invalids.'
    ),
)

TP_CHOICE_CORRECTNESS_FUSED = TargetPolicy(
    name='choice_correctness_fused',
    target_type='discrete',
    description=(
        'Choosing + outcome merged, side collapsed to correctness. '
        'Vocabulary: intertrial, hold, choosing reward, choosing no reward.'
    ),
    label_vocabulary=('intertrial', 'hold', 'choosing reward', 'choosing no reward'),
    current_b_mode='full',
    current_choosing_state_mode='side',
    current_state_transitions_key='CHOOSING_TO_CORRECTNESS_TRANSITIONS',
    notes=(
        'Same straddle issue as choice_outcome_fused. '
        'DEFAULT_TRANSITION_MAP does not cover these states.'
    ),
)

TP_CHOICE_SIDE = TargetPolicy(
    name='choice_side',
    target_type='discrete',
    description=(
        'Each timepoint is labeled with the upcoming/current trial choice: '
        '"choosing left" or "choosing right". '
        'The full-session label covers the segment window, not individual events.'
    ),
    label_vocabulary=('choosing left', 'choosing right'),
    current_b_mode='reward_to_choice',
    current_choosing_state_mode='side',
    current_state_transitions_key='none',
    notes=(
        'b_mode=decision and b_mode=decision_strict also produce choice_side '
        'targets but with different segment boundaries. '
        'Only reward_to_choice has fully consistent label/trial_index alignment.'
    ),
)

TP_CHOICE_CORRECTNESS = TargetPolicy(
    name='choice_correctness',
    target_type='discrete',
    description=(
        'Two-label target: "choosing correct" / "choosing wrong" based on '
        'which side is better during the current block.'
    ),
    label_vocabulary=('choosing correct', 'choosing wrong'),
    current_b_mode='reward_to_choice',
    current_choosing_state_mode='correctness',
    current_state_transitions_key='none',
)

TP_HGF_BELIEF_SIGNED = TargetPolicy(
    name='hgf_belief_signed',
    target_type='continuous',
    description=(
        'Signed HGF belief trajectory (x_1_expected_mean), rescaled to [-1, 1]. '
        'Used as continuous target in hybrid b_type.'
    ),
    label_vocabulary=(),
    current_b_mode=None,         # Stored in hgf_beliefs, not in b
    current_state_transitions_key=None,
    notes=(
        'hgf_beliefs is a separate array, not part of b. '
        'Combined with a discrete target via make_hybrid_b() '
        'when b_type=hybrid.'
    ),
)

TP_HGF_CONFIDENCE_ABS = TargetPolicy(
    name='hgf_confidence_abs',
    target_type='continuous',
    description=(
        'Absolute HGF belief (|x_1_expected_mean|), representing confidence '
        'regardless of direction. NOT currently implemented in the loader — '
        'would require a new hgf_column or post-processing.'
    ),
    label_vocabulary=(),
    current_b_mode=None,
    notes='Not yet implemented.',
)

TARGET_POLICIES: dict[str, TargetPolicy] = {
    tp.name: tp for tp in [
        TP_PHASE_FULL_SIDE,
        TP_PHASE_HOLD_CHOICE_FUSED,
        TP_CHOICE_OUTCOME_FUSED,
        TP_CHOICE_CORRECTNESS_FUSED,
        TP_CHOICE_SIDE,
        TP_CHOICE_CORRECTNESS,
        TP_HGF_BELIEF_SIGNED,
        TP_HGF_CONFIDENCE_ABS,
    ]
}


# ---------------------------------------------------------------------------
# BanditSpec: (segment, target) pairs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BanditSpec:
    """A named combination of segment policy and target policy.

    Attributes
    ----------
    name : str
        Canonical name for this specification.
    segment_policy : SegmentPolicy
    target_policy : TargetPolicy
    description : str
    old_params : dict
        Mapping to current constructor / CLI arguments that approximate this
        spec. Used for backward compatibility.
    is_consistent : bool
        True if segment_policy boundaries match trial_indices boundaries.
    known_issues : tuple[str, ...]
        List of known problems with the current implementation.
    """
    name: str
    segment_policy: SegmentPolicy
    target_policy: TargetPolicy
    description: str
    old_params: dict = field(default_factory=dict)
    is_consistent: bool = field(default=True)
    known_issues: Tuple[str, ...] = field(default=())


# Predefined specs

SPEC_T0 = BanditSpec(
    name='T0_phase_full_side',
    segment_policy=SP_START_TO_CHOICE,
    target_policy=TP_PHASE_FULL_SIDE,
    description=(
        'Old full multi-state baseline. '
        'Per-event phase labels (intertrial / hold / choosing left/right / '
        'reward / no reward) with trial windows [trial.start, t_chosen].'
    ),
    old_params={
        'b_mode': 'full',
        'choosing_state_mode': 'side',
        'apply_hold_transitions': 'none',
    },
    is_consistent=False,
    known_issues=(
        'reward and no reward timepoints are labeled correctly but have '
        'trial_index=-1 (outside [trial.start, t_chosen] windows). '
        'They will not appear in prep_data_trials training windows.',
    ),
)

SPEC_T1 = BanditSpec(
    name='T1_hold_choice_fused',
    segment_policy=SP_START_TO_CHOICE,
    target_policy=TP_PHASE_HOLD_CHOICE_FUSED,
    description=(
        'Hold + choice-side fusion. '
        'Hold and choosing segments are merged into "hold --> choosing left/right". '
        'reward/no-reward remain separate states but outside trial windows.'
    ),
    old_params={
        'b_mode': 'full',
        'choosing_state_mode': 'side',
        'apply_hold_transitions': 'HOLD_TO_CHOOSING_TRANSITIONS',
    },
    is_consistent=False,
    known_issues=(
        'reward and no reward timepoints have trial_index=-1 (same as T0). '
        'The fused hold+choosing states DO appear in trial windows.',
    ),
)

SPEC_T2 = BanditSpec(
    name='T2_choice_outcome_fused',
    segment_policy=SP_START_TO_CHOICE,
    target_policy=TP_CHOICE_OUTCOME_FUSED,
    description=(
        'Choice-side × outcome fusion (4 states). '
        'Choosing + reward/no-reward segments are merged. '
        'The fused state straddles t_chosen: the choosing half is in trial '
        'windows; the reward half is outside.'
    ),
    old_params={
        'b_mode': 'full',
        'choosing_state_mode': 'side',
        'apply_hold_transitions': 'CHOOSING_TO_OUTCOME_TRANSITIONS',
    },
    is_consistent=False,
    known_issues=(
        'Fused state "choosing X --> reward/no reward" straddles the t_chosen '
        'boundary. The choosing half (pre-t_chosen) has trial_index>=0; the '
        'reward half (post-t_chosen) has trial_index=-1. '
        'DEFAULT_TRANSITION_MAP does not cover fused states — '
        'check_state_transitions() reports 8 false invalids.',
    ),
)

SPEC_T2B = BanditSpec(
    name='T2b_choice_correctness_fused',
    segment_policy=SP_START_TO_CHOICE,
    target_policy=TP_CHOICE_CORRECTNESS_FUSED,
    description=(
        'Correctness × outcome fusion (2 states). '
        'Side collapsed; only rewarded/unrewarded distinction kept.'
    ),
    old_params={
        'b_mode': 'full',
        'choosing_state_mode': 'side',
        'apply_hold_transitions': 'CHOOSING_TO_CORRECTNESS_TRANSITIONS',
    },
    is_consistent=False,
    known_issues=(
        'Same straddle issue as T2. '
        'DEFAULT_TRANSITION_MAP does not cover "choosing reward/no reward".',
    ),
)

SPEC_DECISION_LIFECYCLE = BanditSpec(
    name='decision_lifecycle',
    segment_policy=SP_LIFECYCLE,
    target_policy=TP_CHOICE_SIDE,
    description=(
        'Full trial lifecycle labeled with upcoming choice. '
        'Labels cover [trial.start, next_trial.start-1] (full lifecycle). '
        'MISMATCH: trial_indices use [trial.start, t_chosen] so reward/no-reward '
        'timepoints (labeled as "choosing X") are at trial_index=-1.'
    ),
    old_params={
        'b_mode': 'decision',
        'choosing_state_mode': 'side',
        'apply_hold_transitions': 'none',
    },
    is_consistent=False,
    known_issues=(
        'trial_indices do NOT match label boundaries: labels cover the full '
        'lifecycle but trial_indices end at t_chosen. '
        'Reward/no-reward timepoints are labeled "choosing X" but have '
        'trial_index=-1 and are excluded from prep_data_trials windows.',
    ),
)

SPEC_DECISION_STRICT = BanditSpec(
    name='decision_strict',
    segment_policy=SP_DECISION_STRICT,
    target_policy=TP_CHOICE_SIDE,
    description=(
        'decision_strict: trial 0 uses [start, t_chosen]; '
        'trial i>0 uses [prev_t_chosen+1, t_chosen]. '
        'PARTIAL MISMATCH: trial_indices always use [start, t_chosen].'
    ),
    old_params={
        'b_mode': 'decision_strict',
        'choosing_state_mode': 'side',
        'apply_hold_transitions': 'none',
    },
    is_consistent=False,
    known_issues=(
        'trial_indices use [trial.start, t_chosen] for ALL trials. '
        'For trial i>0, the label window starts at prev_t_chosen+1 but '
        'the trial_index window starts at trial[i].start. '
        'The gap [prev_t_chosen+1, trial[i].start-1] (reward of prev trial) '
        'is in the label window but at trial_index=i-1 — excluded from '
        'trial-i windows in prep_data_trials.',
    ),
)

SPEC_REWARD_TO_CHOICE = BanditSpec(
    name='reward_to_choice',
    segment_policy=SP_REWARD_TO_CHOICE,
    target_policy=TP_CHOICE_SIDE,
    description=(
        'Corrected upcoming-choice baseline. '
        'Labels AND trial_indices use [prev_t_chosen+1, t_chosen]. '
        'FULLY CONSISTENT. Current production configuration (Prompt 029+).'
    ),
    old_params={
        'b_mode': 'reward_to_choice',
        'choosing_state_mode': 'side',
        'apply_hold_transitions': 'none',
    },
    is_consistent=True,
    known_issues=(),
)

SPEC_REWARD_TO_CHOICE_HYBRID = BanditSpec(
    name='reward_to_choice_hybrid',
    segment_policy=SP_REWARD_TO_CHOICE,
    target_policy=TP_HGF_BELIEF_SIGNED,
    description=(
        'reward_to_choice segment with HGF belief as continuous target '
        '(combined with choice_side as discrete component via make_hybrid_b). '
        'Current production training spec (Prompt 029+).'
    ),
    old_params={
        'b_mode': 'reward_to_choice',
        'b_type': 'hybrid',
        'choosing_state_mode': 'side',
        'apply_hold_transitions': 'none',
        'hgf_model': 'binary2',
        'hgf_column': 'x_1_expected_mean',
    },
    is_consistent=True,
    known_issues=(),
)

# Proposed future spec (not yet implemented):
SPEC_T0_FULL_LIFECYCLE = BanditSpec(
    name='T0_phase_full_side_lifecycle',
    segment_policy=SP_LIFECYCLE,
    target_policy=TP_PHASE_FULL_SIDE,
    description=(
        '[PROPOSED — NOT IMPLEMENTED] '
        'Full multi-state labels with lifecycle segment windows. '
        'This is what T0 SHOULD be if reward/no-reward activity is meant to '
        'be included in training. Requires a new _create_trial_indices that '
        'uses [trial.start, next_trial.start-1] as the trial boundary.'
    ),
    old_params={},
    is_consistent=True,   # Would be consistent once trial_indices are fixed
    known_issues=(
        'NOT CURRENTLY IMPLEMENTABLE without modifying _create_trial_indices '
        'to use lifecycle boundaries. Requires a new segment_policy parameter '
        'separate from b_mode.',
    ),
)

BANDIT_SPECS: dict[str, BanditSpec] = {
    spec.name: spec for spec in [
        SPEC_T0,
        SPEC_T1,
        SPEC_T2,
        SPEC_T2B,
        SPEC_DECISION_LIFECYCLE,
        SPEC_DECISION_STRICT,
        SPEC_REWARD_TO_CHOICE,
        SPEC_REWARD_TO_CHOICE_HYBRID,
        SPEC_T0_FULL_LIFECYCLE,
    ]
}


# ---------------------------------------------------------------------------
# Compatibility translation functions
# ---------------------------------------------------------------------------

def old_params_to_spec(
    b_mode: str,
    choosing_state_mode: str = 'side',
    apply_hold_transitions: str = 'none',
    b_type: str = 'discrete',
) -> Optional[BanditSpec]:
    """Map old constructor / CLI parameters to the nearest BanditSpec.

    Returns None if no registered spec matches.
    """
    key = (
        b_mode,
        choosing_state_mode,
        (apply_hold_transitions or 'none').lower(),
        b_type,
    )
    _lookup = {
        ('full', 'side', 'none', 'discrete'):             SPEC_T0,
        ('full', 'side', 'hold_to_choosing_transitions', 'discrete'): SPEC_T1,
        ('full', 'side', 'choosing_to_outcome_transitions', 'discrete'): SPEC_T2,
        ('full', 'side', 'choosing_to_correctness_transitions', 'discrete'): SPEC_T2B,
        ('decision', 'side', 'none', 'discrete'):         SPEC_DECISION_LIFECYCLE,
        ('decision_strict', 'side', 'none', 'discrete'):  SPEC_DECISION_STRICT,
        ('reward_to_choice', 'side', 'none', 'discrete'): SPEC_REWARD_TO_CHOICE,
        ('reward_to_choice', 'side', 'none', 'hybrid'):   SPEC_REWARD_TO_CHOICE_HYBRID,
    }
    return _lookup.get(key)


def spec_to_old_params(spec: BanditSpec) -> dict:
    """Return the old constructor / CLI parameters for a BanditSpec.

    Returns the ``spec.old_params`` dict directly.
    """
    return dict(spec.old_params)
