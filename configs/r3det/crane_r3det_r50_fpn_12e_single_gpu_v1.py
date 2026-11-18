"""MMRotate standard R3Det R50/FPN, one-GPU task adaptation, no paper shrink.

The MMRotate recipe is the reference, not a claim of exact TensorFlow-paper
reproduction. One refinement stage, native oc/loss/anchors/assignment/NMS.
"""
_base_ = ['./r3det_r50_fpn_1x_dota_oc.py']
custom_imports = dict(imports=[
    'crane_project.utils.r3det_native_retina_head_v1',
    'crane_project.utils.r3det_native_refine_head_v1',
    'crane_project.utils.crane_dataset_oc_v1',
    'crane_project.utils.external_training_observer_v1',
], allow_failed_imports=False)
model = dict(bbox_head=dict(type='R3DetNativeRetinaHeadV1', num_classes=1), refine_heads=[
    dict(type='R3DetNativeRetinaRefineHeadV1', num_classes=1, in_channels=256,
         stacked_convs=4, feat_channels=256, assign_by_circumhbbox=None,
         anchor_generator=dict(type='PseudoAnchorGenerator', strides=[8, 16, 32, 64, 128]),
         bbox_coder=dict(type='DeltaXYWHAOBBoxCoder', angle_range='oc', norm_factor=None,
             edge_swap=False, proj_xy=False, target_means=(0., 0., 0., 0., 0.),
             target_stds=(1., 1., 1., 1., 1.)),
         loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2., alpha=.25, loss_weight=1.),
         loss_bbox=dict(type='SmoothL1Loss', beta=.11, loss_weight=1.))])
dataset_type = 'CraneDatasetOCV1'
classes = ('grab',)
data_root = 'crane_project/data/crane_grab_port_day2night_v1/'
train_pipeline = [
    dict(type='LoadImageFromFile'), dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(type='RRandomFlip', flip_ratio=[.25, .25, .25],
         direction=['horizontal', 'vertical', 'diagonal'], version='oc'),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='Pad', size=(1024, 1024)),
    dict(type='DefaultFormatBundle'), dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])]
test_pipeline = [dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug', img_scale=(1024, 1024), flip=False, transforms=[
        dict(type='RResize'),
        dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
        dict(type='Pad', size=(1024, 1024)), dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img'])])]
data = dict(_delete_=True, samples_per_gpu=1, workers_per_gpu=2, train=[
    dict(type=dataset_type, classes=classes, data_root=data_root, ann_file='train/annfiles/',
         img_prefix='train/images/', pipeline=train_pipeline, version='oc'),
    dict(type=dataset_type, classes=classes, data_root=data_root, ann_file='train_sim/annfiles/',
         img_prefix='train_sim/images/', pipeline=train_pipeline, version='oc')],
    val=dict(type=dataset_type, classes=classes, data_root=data_root, ann_file='val/annfiles/',
         img_prefix='val/images/', pipeline=test_pipeline, version='oc'),
    test=dict(type=dataset_type, classes=classes, data_root=data_root, ann_file='test/annfiles/',
         img_prefix='test/images/', pipeline=test_pipeline, version='oc'))
# Same native optimizer and effective batch 2; change only execution batching.
optimizer_config = dict(type='GradientCumulativeOptimizerHook', cumulative_iters=2,
                        grad_clip=dict(max_norm=35, norm_type=2))
# Native warmup is 500 batch-2 optimizer steps: 1000 batch-1 microbatches.
lr_config = dict(warmup_iters=1000)
checkpoint_config = dict(interval=1, max_keep_ckpts=12)
evaluation = dict(_delete_=True, interval=2, metric='mAP', save_best=None)
custom_hooks = [dict(type='ExternalTrainingObserverV1', priority='LOWEST')]
log_config = dict(interval=40)
gpu_ids = [0]
seed = 0
load_from = None
resume_from = None
cudnn_benchmark = False
work_dir = 'work_dirs/r3det_no_port_shrink_v1/attempt_001/r3det/train'
