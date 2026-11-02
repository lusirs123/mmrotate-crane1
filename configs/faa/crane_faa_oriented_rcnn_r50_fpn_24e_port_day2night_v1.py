"""FAA / Oriented R-CNN R50: single-class port-grab external comparison.

Official model is inherited locally. Adaptations are documented in README.md.
One GPU, one image/microbatch, two microbatches/optimizer update. No TEST tuning.
"""
_base_ = ['./oriented_rcnn_r50_fpn_1x_dota_le90_faa.py']

custom_imports = dict(
    imports=[
        'mmrotate.models.necks.faafusion',
        'mmrotate.models.roi_heads.bbox_heads.faa_head',
        'mmrotate.datasets.crane_custom_dota',
        'mmrotate.datasets.pipelines.port_train_augment',
    ],
    allow_failed_imports=False)

model = dict(roi_head=dict(bbox_head=dict(num_classes=1)))
dataset_type = 'CraneDataset'
classes = ('grab',)
data_root = 'crane_project/data/crane_grab_port_day2night_v1/'
angle_version = 'le90'

# Same preprocessing/augmentation as the scale-augmented EOOD control.
# DOTA PolyRandomRotate/rect_classes=[9, 11] is deliberately replaced here.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(type='PortIsotropicShrink', prob=0.5, scale_range=(0.5, 1.0)),
    dict(type='RRandomFlip', flip_ratio=[0.25, 0.25, 0.25],
         direction=['horizontal', 'vertical', 'diagonal'], version='le90'),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='Pad', size=(1024, 1024),
         pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug', img_scale=(1024, 1024), flip=False,
         transforms=[
             dict(type='RResize'),
             dict(type='Normalize', mean=[123.675, 116.28, 103.53],
                  std=[58.395, 57.12, 57.375], to_rgb=True),
             dict(type='Pad', size=(1024, 1024),
                  pad_val=dict(img=(114.0, 114.0, 114.0))),
             dict(type='DefaultFormatBundle'),
             dict(type='Collect', keys=['img']),
         ]),
]

data = dict(
    _delete_=True,
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=[
        dict(type=dataset_type, classes=classes, data_root=data_root,
             ann_file='train/annfiles/', img_prefix='train/images/',
             pipeline=train_pipeline, version=angle_version),
        dict(type=dataset_type, classes=classes, data_root=data_root,
             ann_file='train_sim/annfiles/', img_prefix='train_sim/images/',
             pipeline=train_pipeline, version=angle_version),
    ],
    val=dict(type=dataset_type, classes=classes, data_root=data_root,
             ann_file='val/annfiles/', img_prefix='val/images/',
             pipeline=test_pipeline, version=angle_version),
    test=dict(type=dataset_type, classes=classes, data_root=data_root,
              ann_file='test/annfiles/', img_prefix='test/images/',
              pipeline=test_pipeline, version=angle_version),
)

# Preserve FAA's AdamW / loss / assigners / proposal sampling / native NMS.
# Do not replace AdamW with the EOOD optimizer or scale LR to its SGD LR.
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001,
                 betas=(0.9, 0.999), weight_decay=0.05)
optimizer_config = dict(
    _delete_=True, type='GradientCumulativeOptimizerHook', cumulative_iters=2,
    grad_clip=dict(max_norm=35, norm_type=2))
# 1000 microsteps = 500 optimizer updates for the declared single-GPU batch.
lr_config = dict(policy='step', warmup='linear', warmup_iters=1000,
                 warmup_ratio=1.0 / 3, step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(interval=2, max_keep_ckpts=12)
# Logging only; the official best-mAP hook must not silently become our selector.
evaluation = dict(_delete_=True, interval=2, metric='mAP', save_best=None,
                  thresh_sim=10.0, thresh_real=25.0,
                  weight_sim=0.7, weight_real=0.3)
seed = 0
gpu_ids = [0]
load_from = None
resume_from = None
work_dir = 'work_dirs/crane_faa_oriented_rcnn_r50_fpn_24e_port_day2night_v1'
