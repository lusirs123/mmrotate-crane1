"""Source-only current-anchored causal history refiner V1.

The experiment consumes the current frame plus four strictly preceding source
frames.  Cached all-lane DINO proposals are reused; DINO is never invoked.
The same causal forward is used for real and simulation data.  Sequence/frame
identity only establishes chronology in the loader and is not a network input,
router feature, or output rule.  Fixed TEST is absent from this config.
"""

_base_ = ['./crane_symeood_dino_geometry_refiner_full_source_v1.py']

history_horizon = 4
source_train_audit = (
    'work_dirs/crane_symeood_dino_distill_support_v1/source_collect/'
    'source_train_all_lane_audit.json')
source_val_audit = (
    'work_dirs/crane_symeood_dino_conservative_takeover_v2/'
    'source_calibration_collect/source_val_fusion_source_audit.json')
dataset_type = 'CraneDataset'
data_root = 'crane_project/data/crane_grab/'
normalization = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

geometry_refiner = dict(
    _delete_=True,
    type='DinoConditionedCausalHistoryRefiner',
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
    decoded_geometry_loss_weight=0.25,
    temporal_size_loss_weight=0.0,
    history_horizon=history_horizon,
    max_history_center_delta=0.35,
    max_history_log_size_delta=0.45,
    max_history_angle_delta_deg=20.0,
    history_gate_bias=-4.0,
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
    dino_detector_forward_during_training=False,
    frozen_symeood_feature_forward=True,
    cached_dino_proposals_only=True,
    domain_routing=False,
    sequence_frame_routing=False,
    temporal_state=False,
    causal_history_input=True,
    history_horizon=history_horizon,
    history_identity_model_input=False,
    fixed_target_parameter_selection=False,
    source_only_proposal_corruption=True)

model = dict(
    geometry_refiner=geometry_refiner,
    evidence_contract=evidence_contract,
    geometry_refiner_checkpoint=None,
    geometry_refiner_checkpoint_sha256=None,
    geometry_refiner_checkpoint_contract=None,
    evaluation_only=False)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='LoadDinoProposalFromAudit',
        audit_json=source_train_audit,
        expected_frame_count=2781,
        expected_split='source-train'),
    dict(
        type='LoadCausalHistoryFromAudit',
        audit_json=source_train_audit,
        history_horizon=history_horizon,
        expected_frame_count=2781,
        expected_split='source-train'),
    dict(type='RResize', img_scale=(1024, 1024)),
    # Do not rely on RandomFlip(flip_ratio=0) implementation details: older
    # MMDetection versions may omit the required metadata keys entirely.
    dict(type='SetNoFlipMetadata'),
    dict(
        type='RandomBrightnessContrast',
        brightness_range=(0.4, 1.0),
        contrast_range=(1.0, 1.0),
        noise_std_range=(0, 0),
        prob=0.5),
    dict(type='Normalize', **normalization),
    dict(
        type='Pad', size=(1024, 1024),
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='PrepareCausalHistoryInputs', **normalization),
    dict(
        type='CausalHistoryProposalAugment',
        current_probability=0.5,
        history_probability=0.35,
        history_dropout_probability=0.25,
        center_fraction=0.20,
        log_size=0.30,
        angle_deg=12.0),
    dict(type='DefaultFormatBundle'),
    dict(type='FormatDinoProposal'),
    dict(type='FormatCausalHistoryInputs'),
    dict(
        type='Collect',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'dino_proposals',
            'causal_history_images', 'causal_history_proposals',
            'causal_history_valid_mask', 'causal_history_ages']),
]

source_val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadDinoProposalFromAudit',
        audit_json=source_val_audit,
        expected_frame_count=738,
        expected_split='val'),
    dict(
        type='LoadCausalHistoryFromAudit',
        audit_json=source_val_audit,
        history_horizon=history_horizon,
        expected_frame_count=738,
        expected_split='val'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **normalization),
            dict(
                type='Pad', size=(1024, 1024),
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='PrepareCausalHistoryInputs', **normalization),
            dict(type='DefaultFormatBundle'),
            dict(type='FormatDinoProposal'),
            dict(type='FormatCausalHistoryInputs'),
            dict(
                type='Collect',
                keys=[
                    'img', 'dino_proposals',
                    'causal_history_images',
                    'causal_history_proposals',
                    'causal_history_valid_mask',
                    'causal_history_ages']),
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
    train_dataloader=dict(
        samples_per_gpu=1, workers_per_gpu=2, shuffle=True),
    val_dataloader=dict(
        samples_per_gpu=1, workers_per_gpu=2, shuffle=False),
    test_dataloader=dict(
        samples_per_gpu=1, workers_per_gpu=2, shuffle=False))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    constructor='GeometryRefinerOptimizerConstructor',
    lr=5e-5,
    weight_decay=1e-4)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step', warmup='linear', warmup_iters=200,
    warmup_ratio=0.1, step=[6, 9])
runner = dict(type='EpochBasedRunner', max_epochs=10)

checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=10,
    meta=dict(
        geometry_refiner_checkpoint_contract=dict(
            protocol='source_only_causal_history_refiner_v1',
            architecture='current_anchored_causal_history_refiner_v1',
            source_train_frames=2781,
            source_val_frames=738,
            target_data_read=False,
            fixed_test_read=False,
            source_gate_passed=False,
            detector_forward_during_training=False,
            dino_detector_forward_during_training=False,
            frozen_symeood_feature_forward=True,
            cached_dino_proposals_only=True,
            domain_routing=False,
            sequence_frame_routing=False,
            temporal_state=False,
            causal_history_input=True,
            history_horizon=history_horizon,
            history_identity_model_input=False,
            current_frame_anchored=True,
            bounded_history_residual=True,
            rejectable_history_gate=True,
            exact_current_only_when_no_history=True,
            source_only_proposal_corruption=True,
            fixed_target_parameter_selection=False,
            representation='five_delta_xywha',
            angle_range='le90', edge_swap=True, proj_xy=True,
            refine_center=True, refine_size=True, refine_angle=True)))

seed = 3407
gpu_ids = [0]
load_from = None
resume_from = None
work_dir = (
    'work_dirs/crane_symeood_dino_causal_history_refiner_'
    'source_v1_seed3407')

custom_hooks = [
    dict(type='GeometryRefinerContractHook'),
    dict(type='CudaPeakMemoryContractHook')]
