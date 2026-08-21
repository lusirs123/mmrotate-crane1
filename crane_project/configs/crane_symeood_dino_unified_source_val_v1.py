"""Source-validation gate for the source-owned-geometry unified detector."""

_base_ = ['./crane_symeood_dino_unified_v1.py']

project_root = '/media/omnisky/personal_files/ljj/symEOOD'

data = dict(
    test=dict(
        type='CraneDataset',
        data_root=project_root + '/crane_project/data/crane_grab/',
        ann_file='val/annfiles/',
        img_prefix='val/images/',
        pipeline=[
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
        ],
        version='le90'))

work_dir = project_root + '/work_dirs/crane_symeood_dino_unified_v2/source_val'
fusion_audit_file = 'source_val_fusion_source_audit.json'
