"""Read-only all-lane collection for an additional source dataset.

Set ``SYMEOOD_INVENTORY_SPLIT`` to a directory below
``crane_project/data/crane_grab``.  The split must contain ``images`` and
``annfiles`` and must not be named test/val.  This config never enables a
runtime router and exists only to collect SymEOOD and frozen-DINO outputs for
the CPU support inventory.
"""

import os

_base_ = ['./crane_symeood_dino_unified_source_train_distill_support_v1.py']

dataset_type = 'CraneDataset'
data_root = os.environ.get(
    'SYMEOOD_INVENTORY_DATA_ROOT',
    'crane_project/data/crane_grab').strip().rstrip('/') + '/'
if os.path.isabs(data_root):
    raise ValueError('SYMEOOD_INVENTORY_DATA_ROOT must be project-relative')
source_inventory_pipeline = [
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
        ])]

source_inventory_split = os.environ.get(
    'SYMEOOD_INVENTORY_SPLIT', 'extra_source').strip().strip('/')
_parts = [part.lower() for part in source_inventory_split.split('/') if part]
if not source_inventory_split or any(
        part == 'test' or part.startswith('val') for part in _parts):
    raise ValueError(
        'SYMEOOD_INVENTORY_SPLIT must name a source-only split, not test/val')

model = dict(
    scope_split='source_inventory',
    conditional_dino=dict(enabled=False),
    conservative_takeover=dict(enabled=False))

data = dict(
    _delete_=True,
    samples_per_gpu=1,
    workers_per_gpu=2,
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=source_inventory_split + '/annfiles/',
        img_prefix=source_inventory_split + '/images/',
        pipeline=source_inventory_pipeline,
        version='le90'),
    test_dataloader=dict(
        samples_per_gpu=1, workers_per_gpu=2, shuffle=False))

formal_distillation_support_collection_contract = dict(
    protocol='source_only_additional_dataset_inventory_collection_v1',
    source_splits=[source_inventory_split],
    target_data_read=False,
    optimizer_steps=0,
    sym_eood_frozen=True,
    dino_frozen=True,
    all_lane_outputs_required=True,
    sequence_identity_routing=False,
    test_parameter_search=False)

work_dir = os.environ.get(
    'SYMEOOD_INVENTORY_WORK_DIR',
    'work_dirs/crane_symeood_dino_source_inventory_v1')
fusion_audit_file = 'source_inventory_all_lane_audit.json'
