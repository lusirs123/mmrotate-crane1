"""One-time fixed TEST evaluation for source-promoted Dual-Tower V2.1.

The runtime consumes the already-computed 992-frame native-DINO all-lane
audit.  It has one domain-independent forward, no sequence/frame router and no
temporal inference state.  The promotion report, rather than TEST results,
selects and hashes the checkpoint.
"""

import json
import os


_base_ = ['./crane_symeood_dino_geometry_refiner_full_source_v1.py']

promotion_report_path = (
    'work_dirs/crane_symeood_dino_geometry_refiner_'
    'dual_tower_size_source_v21_seed3407/'
    'epoch7_source_promotion.json')
with open(promotion_report_path, 'r', encoding='utf-8') as _handle:
    promotion_report = json.load(_handle)
# MMCV 1.x deep-copies every non-module config global.  A closed TextIOWrapper
# is still unpicklable, so it must not survive in the config namespace.
del _handle
if promotion_report.get('decision') != (
        'ALLOW_ONE_DUAL_TOWER_V21_FIXED_TEST'):
    raise RuntimeError('Dual-Tower V2.1 promotion does not authorize TEST')
if promotion_report.get('eligible_for_one_fixed_test') is not True:
    raise RuntimeError('Promotion report is not fixed-TEST eligible')
if promotion_report.get('target_data_read') is not False:
    raise RuntimeError('Promotion report has target-read provenance')

promoted = dict(promotion_report.get('output') or {})
promoted_checkpoint = os.fspath(promoted['checkpoint'])
promoted_checkpoint_sha256 = str(promoted['checkpoint_sha256'])

geometry_refiner = dict(
    _delete_=True,
    type='DinoConditionedDualTowerGeometryRefiner',
    roi_output_size=7,
    in_channels=256,
    fc_channels=256,
    num_fcs=2,
    refine_center=True,
    refine_size=True,
    refine_angle=True,
    zero_init_output=False,
    train_size_tower=False,
    train_pose_tower=False,
    train_roi_extractor=False,
    evaluation_only=True,
    center_loss_weight=0.0,
    size_loss_weight=0.0,
    angle_loss_weight=0.0,
    decoded_geometry_loss_weight=0.0,
    temporal_size_loss_weight=0.0,
    bbox_coder=dict(
        type='DeltaXYWHAOBBoxCoder',
        angle_range='le90', edge_swap=True, proj_xy=True,
        target_means=(0., 0., 0., 0., 0.),
        target_stds=(1., 1., 1., 1., 1.)))

model = dict(
    geometry_refiner=geometry_refiner,
    geometry_refiner_checkpoint=promoted_checkpoint,
    geometry_refiner_checkpoint_sha256=promoted_checkpoint_sha256,
    geometry_refiner_checkpoint_contract=dict(
        protocol='source_gated_dual_tower_v21_promotion_v1',
        source_gate_passed=True,
        selected_source_epoch=7,
        promotion_before_fixed_test=True,
        domain_routing=False,
        sequence_frame_routing=False,
        temporal_state=False),
    evaluation_only=True)

fixed_test_audit = (
    'work_dirs/crane_symeood_dino_conservative_takeover_v2/'
    'full_test_metric_v2/conservative_takeover_fusion_audit.json')

fixed_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadDinoProposalFromAudit',
        audit_json=fixed_test_audit,
        expected_frame_count=992,
        expected_split='test'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='Pad', size=(1024, 1024),
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='FormatDinoProposal'),
            dict(type='Collect', keys=['img', 'dino_proposals']),
        ])
]

fixed_test_dataset = dict(
    type='CraneDataset',
    data_root='crane_project/data/crane_grab/',
    ann_file='test/annfiles/',
    img_prefix='test/images/',
    pipeline=fixed_test_pipeline,
    version='le90')

data = dict(
    val=fixed_test_dataset,
    test=fixed_test_dataset,
    val_dataloader=dict(
        samples_per_gpu=1, workers_per_gpu=2, shuffle=False),
    test_dataloader=dict(
        samples_per_gpu=1, workers_per_gpu=2, shuffle=False))

evaluation = dict(
    _delete_=True,
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

load_from = None
resume_from = None
work_dir = (
    'work_dirs/crane_symeood_dino_geometry_refiner_'
    'dual_tower_v21_fixed_test')
