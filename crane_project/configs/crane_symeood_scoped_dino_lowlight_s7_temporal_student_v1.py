"""Stage-3 source-only distilled temporal candidate student."""

_base_ = ['./crane_symeood_scoped_dino_lowlight_s7_temporal_relative_quality_v1.py']

_student_temporal_head = dict(
    s7_protected_merge=True,
    s7_lane_arbitration=False,
    s7_quality_suppression=False,
    s7_temporal_association=True,
    s7_temporal_quality_head=True,
    s7_temporal_quality_hidden=128,
    s7_temporal_relative_quality=True,
    s7_temporal_max_candidates=100,
    s7_temporal_min_confirmations=1,
    s7_temporal_override_margin=0.25,
    s7_temporal_max_center_distance=3.0,
    s7_temporal_min_riou=0.05,
    s7_temporal_min_appearance=0.20,
    s7_temporal_student=True,
    s7_student_hidden=128)

model = dict(
    dino_rescue=dict(head=_student_temporal_head),
    dino_head_checkpoint=(
        'work_dirs/dino_teacher_s7_temporal_student_v1/'
        'labeller_best_source_only.pth'),
    scope_manifest=None,
    scope_policy='all_frames',
    temporal_association=dict(
        enabled=True, source_selected=True, target_used_for_selection=False,
        source_gate=dict(min_full_top1=688, min_small_top1=311,
                         max_mcml=3)))

dino_rescue = dict(head=_student_temporal_head)

s7_temporal_student_training = dict(
    stage=3,
    base_checkpoint=(
        'work_dirs/dino_teacher_s7_temporal_relative_quality_v1/'
        'labeller_best_source_only.pth'),
    base_epoch=4,
    frozen_components=[
        'DINOv2', 'native_s14_rpn', 's7_readout_rpn',
        'roi_classifier_regressor', 'global_affine_calibrator',
        'temporal_cue_weights', 'phase2_candidate_quality_teacher'],
    trainable_parameters='student_candidate_quality_head_only',
    initialization='exact_copy_of_phase2_teacher',
    objective=(
        'source_continuous_max_riou + 0.5*same_frame_relative_quality + '
        'teacher_bernoulli_distillation'),
    small_object_training_weight=dict(
        short_side_tokens_at_most=4.0, multiplier=2.0,
        inference_feature=False),
    candidate_pool='fixed_affine_post_nms_top100',
    causal=True,
    min_confirmations=1,
    native_fallback=True,
    inference_slice_routing=False,
    source_only=True,
    target_read=False,
    pseudo_label_training=False,
    source_gate=dict(
        exact_retention=True, min_full_top1=688,
        min_small_top1=311, max_mcml=3,
        dfr_nonregression=True, aci_nonregression=True),
    next_gate=(
        'target_dev only when a positive student epoch beats the copied '
        'phase2 teacher and passes every source gate'))
