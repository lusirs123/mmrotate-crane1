"""Collect both frozen detection lanes on source train data only.

This config is deliberately inference-only.  It runs the existing BrightAug
SymEOOD lane and the frozen native-S14 DINO lane on ``train`` and
``train_sim`` so a later CPU audit can measure teacher complementarity.  It
does not train either detector and must never point at ``test``.
"""

_base_ = ['./crane_symeood_dino_unified_v1.py']

dataset_type = 'CraneDataset'
data_root = 'crane_project/data/crane_grab/'

source_test_pipeline = [
    dict(type='LoadImageFromFile'),
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
                type='Pad',
                size=(1024, 1024),
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ]),
]

model = dict(
    scope_split='source_train',
    # Keep both lanes observable on every source frame.  Explicit values are
    # required here so the collection contract can be checked from MMConfig
    # instead of relying on constructor defaults.
    conditional_dino=dict(enabled=False),
    conservative_takeover=dict(enabled=False))

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    test=[
        dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='train/annfiles/',
            img_prefix='train/images/',
            pipeline=source_test_pipeline,
            version='le90'),
        dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='train_sim/annfiles/',
            img_prefix='train/images/',
            pipeline=source_test_pipeline,
            version='le90'),
    ],
    test_dataloader=dict(
        samples_per_gpu=1, workers_per_gpu=2, shuffle=False))

formal_distillation_support_collection_contract = dict(
    protocol='source_only_symeood_dino_distillation_support_collection_v1',
    source_splits=['train', 'train_sim'],
    expected_frame_count=2781,
    target_data_read=False,
    optimizer_steps=0,
    sym_eood_frozen=True,
    dino_frozen=True,
    all_lane_outputs_required=True,
    sequence_identity_routing=False,
    test_parameter_search=False)

work_dir = 'work_dirs/crane_symeood_dino_distill_support_v1/source_collect'
fusion_audit_file = 'source_train_all_lane_audit.json'
