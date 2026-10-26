"""Frozen MAL pair: inherit augmented B through C-off 110; main head only."""
_base_ = ['./crane_symeood_k1_ablation_110_v1.py']
custom_imports = dict(imports=[
    'mmrotate.datasets.crane_custom_dota',
    'mmrotate.models.detectors.sym_eood_detector',
    'mmrotate.models.dense_heads.sym_eood_head',
    'mmrotate.models.losses.sym_nfl_loss',
    'mmrotate.models.losses.sym_kld_loss',
    'mmrotate.core.bbox.assigners.sym_pola',
    'mmrotate.datasets.pipelines.port_train_augment',
    'crane_project.utils.symeood_mal_v1'], allow_failed_imports=False)
model = dict(bbox_head=dict(type='SymEOODMALHeadV1',
    loss_cls=dict(type='SymEOODMALLossV1', objective='MAL', mal_gamma=1.5,
                  mal_alpha=None, use_geometry_weight=False, strict_finite=True)))
custom_hooks = [dict(type='MALContractHookV1',priority='LOWEST',
    probe_paths=[
      'work_dirs/symeood_ablation_v1/c_frozen_diagnostic_20261024_v1/inputs/batch_000.pt',
      'work_dirs/symeood_ablation_v1/c_frozen_diagnostic_20261024_v1/inputs/batch_001.pt',
    ],probe_epochs=[1,12,24])]
