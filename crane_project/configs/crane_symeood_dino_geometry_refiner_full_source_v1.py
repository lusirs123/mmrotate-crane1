"""Source-only Full single-frame DINO geometry-refiner training.

This config never runs DINO and never reads fixed TEST.  It consumes cached
native-DINO OBBs from complete all-lane source audits, freezes the formal
BrightAug SymEOOD checkpoint, and trains only the shared five-delta refiner.
"""

_base_ = ['./crane_symeood_k1_brightaug.py']

custom_imports = dict(
    imports=[
        'mmrotate.datasets.crane_custom_dota',
        'mmrotate.datasets.pipelines.loading',
        'mmrotate.models.detectors.sym_eood_detector',
        'mmrotate.models.detectors.symeood_dino_geometry_refiner_trainer',
        'mmrotate.models.roi_heads.dino_conditioned_geometry_refiner',
        'mmrotate.models.dense_heads.sym_eood_head',
        'mmrotate.models.losses.sym_nfl_loss',
        'mmrotate.models.losses.sym_kld_loss',
        'mmrotate.core.bbox.assigners.sym_pola',
        'mmrotate.core.hooks.geometry_refiner_contract_hook',
    ],
    allow_failed_imports=False)

source_train_audit = (
    'work_dirs/crane_symeood_dino_distill_support_v1/source_collect/'
    'source_train_all_lane_audit.json')
source_val_audit = (
    'work_dirs/crane_symeood_dino_conservative_takeover_v2/'
    'source_calibration_collect/source_val_fusion_source_audit.json')
formal_sym_eood_checkpoint = (
    'work_dirs/crane_symeood_k1_brightaug/epoch_20.pth')
sym_eood_config = 'crane_project/configs/crane_symeood_k1_brightaug.py'

geometry_refiner = dict(
    type='DinoConditionedGeometryRefiner',
    roi_output_size=7,
    in_channels=256,
    fc_channels=256,
    num_fcs=2,
    refine_center=True,
    refine_size=True,
    refine_angle=True,
    zero_init_output=True,
    center_loss_weight=1.0,
    size_loss_weight=1.0,
    angle_loss_weight=1.0,
    bbox_coder=dict(
        type='DeltaXYWHAOBBoxCoder',
        angle_range='le90', edge_swap=True, proj_xy=True,
        target_means=(0., 0., 0., 0., 0.),
        target_stds=(1., 1., 1., 1., 1.)))

evidence_contract = dict(
    source_train_frames=2781,
    source_val_frames=738,
    target_data_read=False,
    detector_forward_during_training=False,
    domain_routing=False,
    sequence_frame_routing=False,
    temporal_state=False)

model = dict(
    _delete_=True,
    type='SymEOODDinoGeometryRefinerTrainer',
    baseline_config=sym_eood_config,
    baseline_checkpoint=formal_sym_eood_checkpoint,
    geometry_refiner=geometry_refiner,
    evidence_contract=evidence_contract,
    train_cfg=dict(),
    test_cfg=dict(max_per_img=1))

dataset_type = 'CraneDataset'
data_root = 'crane_project/data/crane_grab/'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='LoadDinoProposalFromAudit',
        audit_json=source_train_audit,
        expected_frame_count=2781,
        expected_split='source-train'),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version='le90'),
    dict(
        type='RandomBrightnessContrast',
        brightness_range=(0.4, 1.0),
        contrast_range=(1.0, 1.0),
        noise_std_range=(0, 0),
        prob=0.5),
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
    dict(type='Collect',
         keys=['img', 'gt_bboxes', 'gt_labels', 'dino_proposals']),
]

source_val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadDinoProposalFromAudit',
        audit_json=source_val_audit,
        expected_frame_count=738,
        expected_split='val'),
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

source_val_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='val/annfiles/',
    img_prefix='val/images/',
    pipeline=source_val_pipeline,
    version='le90')

data = dict(
    _delete_=True,
    train=[
        dict(
            type=dataset_type, data_root=data_root,
            ann_file='train/annfiles/', img_prefix='train/images/',
            pipeline=train_pipeline, version='le90'),
        dict(
            type=dataset_type, data_root=data_root,
            ann_file='train_sim/annfiles/', img_prefix='train/images/',
            pipeline=train_pipeline, version='le90'),
    ],
    val=source_val_dataset,
    test=source_val_dataset,
    train_dataloader=dict(samples_per_gpu=2, workers_per_gpu=2,
                          shuffle=True),
    val_dataloader=dict(samples_per_gpu=1, workers_per_gpu=2, shuffle=False),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=2, shuffle=False))

optimizer = dict(
    _delete_=True,
    type='AdamW', constructor='GeometryRefinerOptimizerConstructor',
    lr=1e-4, weight_decay=1e-4)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step', warmup='linear', warmup_iters=100,
                 warmup_ratio=0.1, step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)

checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=12,
    meta=dict(
        geometry_refiner_checkpoint_contract=dict(
            source_train_frames=2781,
            source_val_frames=738,
            target_data_read=False,
            source_gate_passed=False,
            domain_routing=False,
            sequence_frame_routing=False,
            temporal_state=False,
            representation='five_delta_xywha',
            angle_range='le90', edge_swap=True, proj_xy=True,
            refine_center=True, refine_size=True, refine_angle=True)))

evaluation = dict(
    _delete_=True,
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

seed = 3407
gpu_ids = [0]
custom_hooks = [dict(type='GeometryRefinerContractHook')]
load_from = None
resume_from = None
work_dir = 'work_dirs/crane_symeood_dino_geometry_refiner_full_source_v1'
