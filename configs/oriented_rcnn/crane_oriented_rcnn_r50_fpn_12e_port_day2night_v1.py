"""Standard Oriented R-CNN R50/FPN, single-class port task adaptation.

No PortIsotropicShrink, FAA, midpoint or LVSR. Retain native losses, assigners,
samplers, flips, normalization, SGD and NMS. Fixed canvas, split, class and
the common evaluator are task adaptations, not original DOTA results.
"""
_base_ = ['./oriented_rcnn_r50_fpn_1x_dota_le90.py']

custom_imports = dict(imports=[
    'mmrotate.datasets.crane_custom_dota',
    'crane_project.utils.external_training_observer_v1',
], allow_failed_imports=False)
model = dict(roi_head=dict(bbox_head=dict(num_classes=1)))
dataset_type = 'CraneDataset'
classes = ('grab',)
data_root = 'crane_project/data/crane_grab_port_day2night_v1/'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(type='RRandomFlip', flip_ratio=[0.25, 0.25, 0.25],
         direction=['horizontal', 'vertical', 'diagonal'], version='le90'),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375], to_rgb=True),
    # Keep the native default normalized padding value (0). A fixed canvas
    # makes the input/timing size explicit; this adds no random scale change.
    dict(type='Pad', size=(1024, 1024)),
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
             dict(type='Pad', size=(1024, 1024)),
             dict(type='DefaultFormatBundle'),
             dict(type='Collect', keys=['img']),
         ]),
]
data = dict(
    _delete_=True, samples_per_gpu=1, workers_per_gpu=2,
    train=[
        dict(type=dataset_type, classes=classes, data_root=data_root,
             ann_file='train/annfiles/', img_prefix='train/images/',
             pipeline=train_pipeline, version='le90'),
        dict(type=dataset_type, classes=classes, data_root=data_root,
             ann_file='train_sim/annfiles/', img_prefix='train_sim/images/',
             pipeline=train_pipeline, version='le90'),
    ],
    val=dict(type=dataset_type, classes=classes, data_root=data_root,
             ann_file='val/annfiles/', img_prefix='val/images/',
             pipeline=test_pipeline, version='le90'),
    test=dict(type=dataset_type, classes=classes, data_root=data_root,
              ann_file='test/annfiles/', img_prefix='test/images/',
              pipeline=test_pipeline, version='le90'),
)
# Paper section 4.2: one GPU x 2 images, LR 0.005. Two GPUs x 1 image
# preserves that global batch and LR. No accumulation or mixed precision.
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# Author OBBDetection schedule_1x.py uses warmup_ratio=0.001;
# MMRotate v0.3.4's generic schedule uses 1/3. Follow the author here.
lr_config = dict(policy='step', warmup='linear', warmup_iters=500,
                 warmup_ratio=0.001, step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1, max_keep_ckpts=12)
evaluation = dict(_delete_=True, interval=2, metric='mAP', save_best=None,
                  thresh_sim=10.0, thresh_real=25.0,
                  weight_sim=0.7, weight_real=0.3)
custom_hooks = [dict(type='ExternalTrainingObserverV1', priority='LOWEST')]
log_config = dict(interval=20)
seed = 0
find_unused_parameters = False
gpu_ids = [0, 1]
load_from = None
resume_from = None
cudnn_benchmark = False
work_dir = 'work_dirs/oriented_rcnn_no_port_shrink_v1/attempt_001/oriented_rcnn/train'
