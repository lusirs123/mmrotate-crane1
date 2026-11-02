"""Formal FAA GPU2/3 run: microbatch1/GPU, accumulation2, global batch4.

Training uses the existing tools/train.py; the memory observer binds GPUs 2/3.
The earlier single-GPU adaptation and its validation receipts remain intact.
"""
_base_ = ['./crane_faa_oriented_rcnn_r50_fpn_24e_port_day2night_v1.py']

custom_imports = dict(
    imports=[
        'mmrotate.models.necks.faafusion',
        'mmrotate.models.roi_heads.bbox_heads.faa_head',
        'mmrotate.datasets.crane_custom_dota',
        'mmrotate.datasets.pipelines.port_train_augment',
        'crane_project.utils.faa_training_memory_v1',
    ], allow_failed_imports=False)
custom_hooks = [dict(type='FAATrainingMemoryHookV1', priority='LOWEST')]
gpu_ids = [0, 1]
log_config = dict(interval=20)
cudnn_benchmark = False
work_dir = 'work_dirs/faa_gpu23_20261101_v1/attempt_001/train'
