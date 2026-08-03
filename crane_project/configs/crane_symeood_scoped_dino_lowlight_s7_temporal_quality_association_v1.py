"""Source-only continuous candidate-quality cue on the fixed affine pool.

The native S14 lane remains the fallback and the candidate quality head is
used only as a causal temporal cue.  It is trained densely against each
candidate's source GT max-RIoU; it does not use target slices, sequence-name
routing, positive-lane promotion, or gain replay.
"""

_base_ = ['./crane_symeood_scoped_dino_lowlight_s7_retention_merge_v1.py']

_quality_temporal_head = dict(
    s7_protected_merge=True,
    s7_lane_arbitration=False,
    s7_quality_suppression=False,
    s7_temporal_association=True,
    s7_temporal_quality_head=True,
    s7_temporal_quality_hidden=128,
    s7_temporal_quality_loss_weight=1.0,
    s7_temporal_max_candidates=100,
    s7_temporal_min_confirmations=2,
    s7_temporal_override_margin=0.25,
    s7_temporal_max_center_distance=3.0,
    s7_temporal_min_riou=0.05,
    s7_temporal_min_appearance=0.20)

model = dict(
    dino_rescue=dict(head=_quality_temporal_head),
    dino_head_checkpoint=(
        'work_dirs/dino_teacher_s7_temporal_quality_association_v1/'
        'labeller_best_source_only.pth'),
    # One causal policy is applied to every sequence.  No target-derived or
    # manually selected slice is part of deployment.
    scope_manifest=None,
    scope_policy='all_frames',
    temporal_association=dict(
        enabled=True, source_selected=True, target_used_for_selection=False,
        source_gate=dict(min_full_top1=688, min_small_top1=311,
                         max_mcml=3)))

dino_rescue = dict(head=_quality_temporal_head)

s7_temporal_quality_training = dict(
    base_checkpoint=(
        'work_dirs/dino_teacher_s7_retention_merge_v1/'
        'labeller_epoch_01_source_only.pth'),
    base_epoch=1,
    frozen_components=[
        'DINOv2', 'native_s14_rpn', 's7_readout_rpn',
        'roi_classifier_regressor', 'global_affine_calibrator',
        'temporal_cue_weights'],
    trainable_parameters='candidate_quality_head_only',
    supervision='dense_source_candidate_max_riou',
    quality_target='max_rotated_iou(candidate,source_gt)',
    quality_loss='weighted_smooth_l1(sigmoid(logit),target),weight=1+3*target',
    candidate_pool='fixed_affine_post_nms_top100',
    temporal_cue='candidate_quality_logit_as_seventh_cue',
    causal=True,
    min_confirmations=2,
    native_fallback=True,
    positive_promotion=False,
    gain_replay=False,
    source_only=True,
    target_read=False,
    source_gate=dict(
        exact_retention=True, min_full_top1=688,
        min_small_top1=311, max_mcml=3,
        dfr_nonregression=True, aci_nonregression=True,
        aci_angle_limit_deg=35.0),
    target_gate='one_fixed_diagnosis_only_after_formal_source_gate')
