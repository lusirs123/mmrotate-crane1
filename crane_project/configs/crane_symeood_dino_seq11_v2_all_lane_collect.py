"""Inference-only all-lane collection for all 251 labelled seq11 frames.

This stage runs before the 203/48 audit split is consumed by training.  It
uses the formal ordinary K1 checkpoint (not BrightAug), invokes both frozen
lanes on every frame, writes no optimizer state, and never reads fixed TEST.
"""

_base_ = ['./crane_symeood_dino_unified_v1.py']

dataset_type = 'CraneDataset'
data_root = 'crane_project/data/crane_grab/'
seq11_split = 'extra_source_real_seq11_pilot_k1p9_v2'

source_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug', img_scale=(1024, 1024), flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize',
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375], to_rgb=True),
            dict(type='Pad', size=(1024, 1024),
                 pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])])]

model = dict(
    baseline_config='crane_project/configs/crane_symeood_k1.py',
    scope_manifest=None,
    scope_policy='all_frames',
    scope_split=seq11_split,
    conditional_dino=dict(enabled=False),
    conservative_takeover=dict(enabled=False),
    fusion_audit_enabled=True)

formal_detection_contract = dict(
    _delete_=True,
    sym_eood_checkpoint='work_dirs/crane_symeood_k1/epoch_24.pth',
    sym_eood_variant_selection='formal_ordinary_k1_source_checkpoint',
    proposal_sources=['symeood_k1_top1', 'frozen_dino_native_s14_rpn'],
    common_ranker='frozen_dino_roi_classifier_alpha05',
    source_owned_geometry=True,
    final_output='single_top1_obb',
    invalid_dino_fallback='symeood_k1_top1',
    raw_cross_model_score_comparison=False,
    target_scope=False,
    sequence_identity_routing=False,
    brightaug=False,
    detector_training_required=False)

seq11_dataset = dict(
    type=dataset_type, data_root=data_root,
    ann_file=seq11_split + '/annfiles/',
    img_prefix=seq11_split + '/images/',
    pipeline=source_pipeline, version='le90')

data = dict(
    _delete_=True,
    val=seq11_dataset,
    test=seq11_dataset,
    val_dataloader=dict(
        samples_per_gpu=1, workers_per_gpu=2, shuffle=False),
    test_dataloader=dict(
        samples_per_gpu=1, workers_per_gpu=2, shuffle=False))

seq11_all_lane_collection_contract = dict(
    protocol='seq11_v2_source_only_all_lane_collection_v1',
    source_split=seq11_split,
    expected_frame_count=251,
    frozen_symeood_variant='symeood_k1_epoch24',
    frozen_symeood_checkpoint='work_dirs/crane_symeood_k1/epoch_24.pth',
    frozen_dino_variant='native_s14_source_safe_interpolated_head',
    both_lanes_required_every_frame=True,
    optimizer_steps=0,
    target_data_read=False,
    fixed_test_read=False,
    split_after_collection=True)

work_dir = (
    'work_dirs/crane_symeood_dino_source_inventory_v2/'
    'real_seq11_k1p9_v2/all_lane_collect')
fusion_audit_file = 'source_inventory_all_lane_audit.json'
