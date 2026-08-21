"""Formal all-frame SymEOOD + frozen-DINO unified detector.

SymEOOD(K=1) remains the paper detector and contributes its source-trained
top-1 OBB as an external proposal.  That proposal is merged with native-S14
DINO RPN proposals before one shared frozen DINO ROI classifier/regressor.
The final OBB is therefore selected in one score space; raw SymEOOD and DINO
scores are never compared.  No BrightAug, target scope, sequence routing, S7,
temporal takeover, or target-derived threshold is used.

The positional checkpoint passed to ``tools/test.py`` is the SymEOOD(K=1)
checkpoint.  The frozen DINO head checkpoint is loaded by the wrapper.
"""

_base_ = ['./crane_symeood_k1.py']

custom_imports = dict(
    imports=[
        'mmrotate.datasets.crane_custom_dota',
        'mmrotate.models.detectors.sym_eood_detector',
        'mmrotate.models.detectors.scoped_dino_lowlight_detector',
        'mmrotate.models.dense_heads.sym_eood_head',
        'mmrotate.models.losses.sym_nfl_loss',
        'mmrotate.models.losses.sym_kld_loss',
        'mmrotate.core.bbox.assigners.sym_pola',
    ],
    allow_failed_imports=False)

sym_eood_config = 'crane_project/configs/crane_symeood_k1.py'
formal_sym_eood_checkpoint = (
    'work_dirs/crane_symeood_k1/best_Weighted_R_center_epoch_12.pth')
formal_dino_checkpoint = (
    'work_dirs/dino_teacher_fc_cls_interpolation_v1/'
    'source_safe_interpolated_head.pth')

dino_rescue = dict(
    protocol_name='Formal SymEOOD Proposal + Frozen DINO ROI Union V1',
    dinov2=dict(
        repo='third_party/dinov2',
        checkpoint='pretrained/dinov2_vitl14_pretrain.pth',
        model='dinov2_vitl14',
        gpus=[1, 2],
        legacy_sdpa_query_chunk=512,
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
    type='SymEOODDinoUnifiedDetector',
    baseline_config=sym_eood_config,
    dino_rescue=dino_rescue,
    dino_head_checkpoint=formal_dino_checkpoint,
    dino_checkpoint_contract=dino_checkpoint_contract,
    fusion_policy='sym_eood_proposal_dino_roi_union',
    scope_manifest=None,
    scope_policy='all_frames',
    scope_split='test',
    stabilizer=dict(
        enabled=False,
        alpha=1.0,
        target_used_for_selection=False),
    temporal_association=dict(
        enabled=False,
        source_selected=False,
        target_used_for_selection=False),
    test_cfg=dict(score_thr=0.05, max_per_img=1))

# SymEOOD runs first on logical GPU 0.  Its activations are released before
# native-S14 DINO uses logical GPUs 1/2 and the frozen ROI head returns to GPU
# 0.  Batch size one is required for deterministic per-frame fusion.
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=2, shuffle=False))

evaluation = dict(
    interval=1,
    metric='mAP',
    thresh_sim=10.0,
    thresh_real=25.0,
    weight_sim=0.7,
    weight_real=0.3,
    paper_temporal=True,
    temporal_center_thresh_px=15.0,
    temporal_ekf_window=10,
    temporal_mcml_limit=5,
    temporal_iou_thresh=0.5)

formal_detection_contract = dict(
    sym_eood_checkpoint=formal_sym_eood_checkpoint,
    sym_eood_selection_metric='source_val_Weighted_R_center',
    proposal_sources=['symeood_k1_top1', 'frozen_dino_native_s14_rpn'],
    common_ranker='frozen_dino_roi_classifier_alpha05',
    final_output='single_top1_obb',
    invalid_dino_fallback='symeood_k1_top1',
    raw_cross_model_score_comparison=False,
    target_scope=False,
    sequence_identity_routing=False,
    brightaug=False,
    s7_enabled=False,
    temporal_takeover=False,
    detector_training_required=False,
    joint_source_gate_required=True,
    depth_interface='top1_obb_score_then_optional_frozen_roi_feature')

work_dir = 'work_dirs/crane_symeood_dino_unified_v1'
