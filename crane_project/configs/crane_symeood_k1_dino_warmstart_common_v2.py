"""Common source-only warm start for matched K1 student experiments.

Both arms load the source-VAL-selected ordinary K1 epoch 20.  The training
pipeline intentionally loads the same frozen teacher cache in both arms, so
the extra DINO loss is the only planned difference during optimization.
"""

_base_ = ['./crane_symeood_k1_dino_semantic_distill_v1.py']

# This checkpoint was selected on source VAL with metric protocol v2.
# Run the paired warm-start preflight before training; it checks its SHA256
# against the saved sweep selection and final TEST identity report.
load_from = 'work_dirs/crane_symeood_k1/epoch_20.pth'
resume_from = None

# Short, identical source-only fine-tuning budget for both arms.  Checkpoint
# choice still uses the existing offline source-VAL selection rule.
runner = dict(type='EpochBasedRunner', max_epochs=4)
optimizer = dict(type='SGD', lr=0.00025, momentum=0.9,
                 weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step', warmup='linear', warmup_iters=100,
                 warmup_ratio=0.1, step=[3])
checkpoint_config = dict(interval=1, max_keep_ckpts=4)
evaluation = dict(interval=1)

