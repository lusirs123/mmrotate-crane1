"""Read-only formal ordinary-K1 inference on all 251 seq11-v2 frames."""


_base_ = ['./crane_symeood_k1.py']

data_root = 'crane_project/data/crane_grab/'
source_split = 'extra_source_real_seq11_pilot_k1p9_v2'
expected_checkpoint = 'work_dirs/crane_symeood_k1/epoch_24.pth'

test_pipeline = [
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

full251_dataset = dict(
    type='CraneDataset', data_root=data_root,
    ann_file=source_split + '/annfiles/',
    img_prefix=source_split + '/images/',
    pipeline=test_pipeline, version='le90')

data = dict(
    _delete_=True,
    val=full251_dataset,
    test=full251_dataset,
    val_dataloader=dict(samples_per_gpu=1, workers_per_gpu=2, shuffle=False),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=2, shuffle=False))

evaluation = dict(
    _delete_=True, metric='mAP', paper_temporal=False,
    thresh_sim=10.0, thresh_real=25.0)

formal_k1_full251_contract = dict(
    protocol='formal_k1_seq11_v2_full251_inference_v1',
    expected_frame_count=251,
    checkpoint_role='ordinary_k1_epoch24',
    source_split=source_split,
    prediction_coordinate_system='original_image_pixels',
    obb_convention='le90',
    annotation_k0=1.9,
    target_geometry='top_beam_only',
    optimizer_steps=0,
    target_data_read=False,
    fixed_test_read=False)

load_from = None
resume_from = None
work_dir = (
    'work_dirs/crane_symeood_dino_source_inventory_v2/'
    'real_seq11_k1p9_v2/formal_k1_full251')
