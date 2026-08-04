"""Phase 2B source-only relative candidate-quality training.

This stage starts from the source-gated pointwise quality-head epoch 4
checkpoint.  It adds only same-frame relative ranking supervision; all
detector components and the six original temporal cue weights remain fixed.
"""

_base_ = ['./crane_symeood_scoped_dino_lowlight_s7_retention_merge_v1.py']

_relative_quality_temporal_head = dict(
    s7_protected_merge=True,
    s7_lane_arbitration=False,
    s7_quality_suppression=False,
    s7_temporal_association=True,
    s7_temporal_quality_head=True,
    s7_temporal_quality_hidden=128,
    s7_temporal_quality_loss_weight=1.0,
    s7_temporal_relative_quality=True,
    s7_temporal_relative_quality_weight=0.5,
    s7_temporal_relative_quality_margin=0.25,
    s7_temporal_relative_quality_min_gap=0.10,
    s7_temporal_relative_quality_max_pairs=128,
    s7_temporal_relative_base_epoch=4,
    s7_temporal_max_candidates=100,
    s7_temporal_min_confirmations=1,
    s7_temporal_override_margin=0.25,
    s7_temporal_max_center_distance=3.0,
    s7_temporal_min_riou=0.05,
    s7_temporal_min_appearance=0.20)

model = dict(
    dino_rescue=dict(head=_relative_quality_temporal_head),
    dino_head_checkpoint=(
        'work_dirs/dino_teacher_s7_temporal_relative_quality_v1/'
        'labeller_best_source_only.pth'),
    scope_manifest=None,
    scope_policy='all_frames',
    temporal_association=dict(
        enabled=True, source_selected=True, target_used_for_selection=False,
        source_gate=dict(min_full_top1=688, min_small_top1=311,
                         max_mcml=3)))

dino_rescue = dict(head=_relative_quality_temporal_head)

s7_temporal_quality_training = dict(
    base_checkpoint=(
        'work_dirs/dino_teacher_s7_temporal_quality_association_v1/'
        'labeller_epoch_04_source_only.pth'),
    base_epoch=4,
    frozen_components=[
        'DINOv2', 'native_s14_rpn', 's7_readout_rpn',
        'roi_classifier_regressor', 'global_affine_calibrator',
        'temporal_cue_weights'],
    trainable_parameters='candidate_quality_head_only',
    supervision='dense_source_candidate_max_riou_plus_same_frame_relative',
    quality_target='max_rotated_iou(candidate,source_gt)',
    quality_loss=(
        'weighted_smooth_l1(sigmoid(logit),target) + '
        '0.5*pairwise_hinge(logit_pos-logit_neg,margin=0.25)'),
    relative_quality=True,
    relative_quality_weight=0.5,
    relative_quality_margin=0.25,
    relative_quality_min_gap=0.10,
    relative_quality_max_pairs=128,
    pair_source='same-frame source candidates with max-RIoU gap >= 0.10',
    candidate_pool='fixed_affine_post_nms_top100',
    temporal_cue='candidate_quality_logit_as_seventh_cue',
    causal=True,
    min_confirmations=1,
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
    target_gate='one fixed diagnosis only after formal source gate')
