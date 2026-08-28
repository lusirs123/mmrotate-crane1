"""Frozen native-S14 DINO component for detector-to-depth validation.

This config isolates the formally source-gated DINO component: DINOv2
ViT-L/14 is frozen, the native S14 RPN/ROI head uses the source-only
alpha=0.5 classifier interpolation checkpoint, and S7 is disabled.  It does
not construct SymEOOD/BrightAug, use target scope, route by sequence identity,
or apply temporal/geometry post-processing.

The positional checkpoint accepted by the Webots inference entry is the same
``source_safe_interpolated_head.pth`` loaded and provenance-checked by the
constructor.  This is a component diagnostic, not a newly selected deployment
model.
"""

_base_ = ['./crane_symeood_k1.py']

custom_imports = dict(
    imports=[
        'mmrotate.datasets.crane_custom_dota',
        'mmrotate.models.detectors.scoped_dino_lowlight_detector',
    ],
    allow_failed_imports=False)

formal_dino_checkpoint = (
    'work_dirs/dino_teacher_fc_cls_interpolation_v1/'
    'source_safe_interpolated_head.pth')

dino_rescue = dict(
    protocol_name='Formal Frozen DINO Native-S14 Alpha05 Component V1',
    dinov2=dict(
        repo='third_party/dinov2',
        checkpoint='pretrained/dinov2_vitl14_pretrain.pth',
        model='dinov2_vitl14',
        gpus=[1, 2],
        legacy_sdpa_query_chunk=256,
        height=600,
        max_long_side=1333,
        patch_size=14),
    head=dict(
        gpu=0,
        rpn_feat_channels=256,
        roi_fc_channels=1024,
        roi_samples=256,
        proposal_count=2000,
        max_detections=2000,
        roi_nms_iou_thr=0.5,
        feature_strides=[14],
        s7_residual=False,
        s7_protected_merge=False,
        s7_lane_arbitration=False,
        s7_quality_suppression=False,
        s7_temporal_association=False))

dino_checkpoint_contract = dict(
    selector='Frozen DINO ROI Classifier Source Interpolation Selector V1',
    protocol_version=1,
    alpha=0.5,
    require_target_unread=True,
    require_source_gate=True,
    require_s7_disabled=True,
    source_full=dict(top1_hits=677, top1_mcml=3),
    source_small=dict(top1_hits=303, top1_mcml=3))

model = dict(
    _delete_=True,
    type='FrozenDinoNativeS14Detector',
    dino_rescue=dino_rescue,
    dino_head_checkpoint=formal_dino_checkpoint,
    dino_checkpoint_contract=dino_checkpoint_contract,
    runtime_checkpoint_in_constructor=True,
    scope_manifest=None,
    scope_policy='all_frames',
    scope_split='fixed_dev',
    stabilizer=dict(enabled=False, alpha=1.0),
    temporal_association=dict(enabled=False, source_selected=False),
    test_cfg=dict(score_thr=0.05, max_per_img=1))

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=2, shuffle=False))

formal_detection_contract = dict(
    component='frozen_dino_native_s14_alpha05',
    checkpoint_role='source_safe_constructor_loaded_dino_head',
    all_frames=True,
    symeood_enabled=False,
    brightaug_enabled=False,
    s7_enabled=False,
    target_scope=False,
    sequence_identity_routing=False,
    temporal_takeover=False,
    box_stabilizer=False,
    dino_silence='explicit_missing_observation',
    depth_interface='single_top1_obb_and_score')

work_dir = 'work_dirs/crane_symeood_formal_dino_native_s14_v1'
