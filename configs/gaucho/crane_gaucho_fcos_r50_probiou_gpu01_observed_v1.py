"""Formal native dual-GPU entry, adding only resource/progress observation."""
_base_ = ['./crane_gaucho_fcos_r50_probiou_port_day2night_v1.py']
custom_imports = dict(imports=[
    'mmrotate.models.losses.gaucho_gaussian_dist_loss_v1',
    'mmrotate.models.dense_heads.gaucho_rotated_fcos_head',
    'mmrotate.models.dense_heads.gaussian_fcos_head',
    'mmrotate.datasets.crane_custom_dota',
    'mmrotate.datasets.pipelines.port_train_augment',
    'crane_project.utils.external_training_observer_v1',
], allow_failed_imports=False)
custom_hooks = [dict(type='ExternalTrainingObserverV1', priority='LOWEST')]
log_config = dict(interval=20)
