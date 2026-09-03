"""Frozen ordinary-K1 reference on the 11 held-out seq11 source frames."""

_base_ = ['./crane_symeood_k1.py']

aux_val_split = 'extra_source_real_seq11_pilot_k1p9_val_v1'
aux_val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug', img_scale=(1024, 1024), flip=False,
         transforms=[
             dict(type='RResize'),
             dict(type='Normalize',
                  mean=[123.675, 116.28, 103.53],
                  std=[58.395, 57.12, 57.375], to_rgb=True),
             dict(type='Pad', size=(1024, 1024),
                  pad_val=dict(img=(114.0, 114.0, 114.0))),
             dict(type='DefaultFormatBundle'),
             dict(type='Collect', keys=['img'])])]

aux_val_dataset = dict(
    type='CraneDataset', data_root='crane_project/data/crane_grab/',
    ann_file=aux_val_split + '/annfiles/',
    img_prefix=aux_val_split + '/images/',
    pipeline=aux_val_pipeline, version='le90')

data = dict(
    _delete_=True,
    val=aux_val_dataset,
    test=aux_val_dataset,
    val_dataloader=dict(samples_per_gpu=1, workers_per_gpu=2, shuffle=False),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=2, shuffle=False))
evaluation = dict(
    _delete_=True, metric='mAP', paper_temporal=False,
    thresh_sim=10.0, thresh_real=25.0)
load_from = None
resume_from = None

