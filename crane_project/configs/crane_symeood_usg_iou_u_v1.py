"""Frozen U/S/G experiment; fresh initialization, TRAIN/VAL only."""
_base_ = ['./crane_symeood_k1_ablation_100_v1.py']
custom_imports = dict(imports=['mmrotate.datasets.crane_custom_dota', 'mmrotate.models.detectors.sym_eood_detector', 'mmrotate.models.dense_heads.sym_eood_head', 'mmrotate.models.losses.sym_nfl_loss', 'mmrotate.models.losses.sym_kld_loss', 'mmrotate.core.bbox.assigners.sym_pola', 'mmrotate.datasets.pipelines.port_train_augment', 'crane_project.utils.symeood_usg_v1'], allow_failed_imports=False)
model = dict(bbox_head=dict(type='SymEOODUSGHeadV1', loss_cls=dict(type='SymNFLUSGV1', objective='U', use_geometry_weight=False)))
custom_hooks = [dict(type='USGContractHookV1', priority='LOWEST', probe_paths=[
    'work_dirs/symeood_ablation_v1/c_frozen_diagnostic_20261024_v1/inputs/batch_000.pt',
    'work_dirs/symeood_ablation_v1/c_frozen_diagnostic_20261024_v1/inputs/batch_001.pt',
    'work_dirs/symeood_ablation_v1/c_frozen_diagnostic_20261024_v1/inputs/batch_002.pt',
    'work_dirs/symeood_ablation_v1/c_frozen_diagnostic_20261024_v1/inputs/batch_003.pt',
], probe_epochs=[1,12,24])]
work_dir = 'work_dirs/symeood_usg_v1/iou_u'
