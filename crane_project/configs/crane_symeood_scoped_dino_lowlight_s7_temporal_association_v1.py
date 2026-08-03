"""Unified source-only causal association on fixed native/S7 candidates.

This is stage one of the retained unified plan.  It does not add a candidate
quality head and does not route by dark/small sequence identity.  Deployment
is authorized only when the stored source gate selects an epoch greater than
zero; epoch zero is the native S14 fallback.
"""

_base_ = ['./crane_symeood_scoped_dino_lowlight_s7_retention_merge_v1.py']

_temporal_head = dict(
    s7_protected_merge=True,
    s7_lane_arbitration=False,
    s7_quality_suppression=False,
    s7_temporal_association=True,
    s7_temporal_max_candidates=100,
    s7_temporal_min_confirmations=2,
    s7_temporal_override_margin=0.25,
    s7_temporal_max_center_distance=3.0,
    s7_temporal_min_riou=0.05,
    s7_temporal_min_appearance=0.20)

model = dict(
    dino_rescue=dict(head=_temporal_head),
    dino_head_checkpoint=(
        'work_dirs/dino_teacher_s7_temporal_association_v1/'
        'labeller_best_source_only.pth'),
    # Unified inference applies one causal policy to every incoming sequence.
    scope_manifest=None,
    scope_policy='all_frames',
    temporal_association=dict(
        enabled=True, source_selected=True, target_used_for_selection=False,
        source_gate=dict(min_full_top1=688, min_small_top1=311,
                         max_mcml=3)))

dino_rescue = dict(head=_temporal_head)

s7_temporal_training = dict(
    base_checkpoint=(
        'work_dirs/dino_teacher_s7_retention_merge_v1/'
        'labeller_epoch_01_source_only.pth'),
    frozen_components=[
        'DINOv2', 'native_s14_rpn', 's7_readout_rpn',
        'roi_classifier_regressor', 'global_affine_calibrator'],
    trainable_parameters='six_non_negative_multi_cue_weights',
    cues=[
        'calibrated_score_logit',
        'negative_normalized_center_distance',
        'rotated_iou',
        'negative_log_scale_change',
        'periodic_angle_similarity',
        'dino_roi_appearance_similarity'],
    causal=True,
    min_confirmations=2,
    native_fallback=True,
    reset_on=['sequence_change', 'frame_gap', 'continuity_failure'],
    candidate_quality_head=False,
    source_only=True,
    source_gate=dict(
        exact_retention=True, min_full_top1=688,
        min_small_top1=311, max_mcml=3,
        dfr_nonregression=True, aci_nonregression=True,
        aci_angle_limit_deg=35.0),
    target_gate='one_fixed_diagnosis_only_after_formal_source_gate')
