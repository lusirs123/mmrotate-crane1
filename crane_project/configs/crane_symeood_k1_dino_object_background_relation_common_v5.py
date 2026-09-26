"""Common 24-epoch K1 warm-start configuration for relation A/C."""

_base_ = ['./crane_symeood_k1_dino_semantic_distill_v1.py']

# Both arms start from the same source-VAL-selected K1 checkpoint.
load_from = 'work_dirs/crane_symeood_k1/epoch_20.pth'
resume_from = None

# Keep K1's full training schedule and optimizer settings.
runner = dict(type='EpochBasedRunner', max_epochs=24)
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9,
                 weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step', warmup='linear', warmup_iters=1000,
                 warmup_ratio=0.001, step=[16, 22])
checkpoint_config = dict(interval=2, max_keep_ckpts=24)
evaluation = dict(
    interval=2,
    metric='mAP',
    save_best='Weighted_R_center',
    rule='greater',
    thresh_sim=10.0,
    thresh_real=25.0,
    weight_sim=0.7,
    weight_real=0.3)

# The new relation loss is the only planned difference between A and C.
model = dict(
    bbox_head=dict(use_semantic_cls_adapter=True),
    semantic_distillation=None)
