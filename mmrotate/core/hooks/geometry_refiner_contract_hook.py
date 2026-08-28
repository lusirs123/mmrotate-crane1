"""Runtime freeze/hash guard for source-only geometry-refiner training."""

import json
import os

from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook
from mmcv.runner.optimizer.builder import OPTIMIZER_BUILDERS, OPTIMIZERS


def _unwrap(model):
    return model.module if is_module_wrapper(model) else model


def _optimizer_parameter_ids(optimizer):
    optimizers = optimizer.values() if isinstance(optimizer, dict) else (
        optimizer, )
    identifiers = []
    for item in optimizers:
        for group in item.param_groups:
            identifiers.extend(id(parameter) for parameter in group['params'])
    return identifiers


@OPTIMIZER_BUILDERS.register_module()
class GeometryRefinerOptimizerConstructor:
    """Build an optimizer containing only trainable refiner parameters."""

    def __init__(self, optimizer_cfg, paramwise_cfg=None):
        if paramwise_cfg:
            raise ValueError(
                'Geometry-refiner optimizer does not support paramwise_cfg')
        self.optimizer_cfg = dict(optimizer_cfg)

    def __call__(self, model):
        model = _unwrap(model)
        config = dict(self.optimizer_cfg)
        optimizer_type = config.pop('type')
        parameters = [
            parameter for parameter in model.geometry_refiner.parameters()
            if parameter.requires_grad
        ]
        if not parameters:
            raise RuntimeError('Geometry refiner has no trainable parameters')
        optimizer_class = OPTIMIZERS.get(optimizer_type)
        if optimizer_class is None:
            raise KeyError('Unknown optimizer type: ' + optimizer_type)
        return optimizer_class(parameters, **config)


@HOOKS.register_module()
class GeometryRefinerContractHook(Hook):
    """Fail training if the frozen SymEOOD branch changes or enters train."""

    def before_run(self, runner):
        model = _unwrap(runner.model)
        report = model.verify_frozen_contract()
        if not report['baseline_eval']:
            raise RuntimeError('Frozen SymEOOD baseline entered train mode')
        if report['baseline_trainable_parameter_count'] != 0:
            raise RuntimeError('Frozen SymEOOD has trainable parameters')
        if report['refiner_trainable_parameter_count'] <= 0:
            raise RuntimeError('Geometry refiner has no trainable parameters')
        if not report.get('public_init_completed', False):
            raise RuntimeError(
                'Geometry refiner trainer did not complete public init')
        if not report['frozen_hash_unchanged']:
            raise RuntimeError('Frozen SymEOOD hash changed before training')
        if not report.get('frozen_refiner_hash_unchanged', True):
            raise RuntimeError(
                'Frozen geometry-refiner components changed before training')
        optimizer_ids = _optimizer_parameter_ids(runner.optimizer)
        refiner_ids = [
            id(parameter) for parameter in model.geometry_refiner.parameters()
            if parameter.requires_grad
        ]
        if (len(optimizer_ids) != len(set(optimizer_ids)) or
                set(optimizer_ids) != set(refiner_ids)):
            raise RuntimeError(
                'Optimizer parameters are not exactly the trainable geometry '
                'refiner parameters')

    def before_train_iter(self, runner):
        model = _unwrap(runner.model)
        if model.baseline.training:
            raise RuntimeError('Frozen SymEOOD baseline entered train mode')
        if any(parameter.grad is not None for parameter in
               model.baseline.parameters()):
            raise RuntimeError('Frozen SymEOOD received a gradient')
        if any(parameter.grad is not None for parameter in
               model.geometry_refiner.parameters()
               if not parameter.requires_grad):
            raise RuntimeError(
                'Frozen geometry-refiner component received a gradient')

    def after_run(self, runner):
        model = _unwrap(runner.model)
        report = model.verify_frozen_contract()
        if not report['frozen_hash_unchanged']:
            raise RuntimeError('Frozen SymEOOD parameters/buffers changed')
        if not report.get('frozen_refiner_hash_unchanged', True):
            raise RuntimeError(
                'Frozen geometry-refiner parameters/buffers changed')
        path = os.path.join(
            runner.work_dir, 'geometry_refiner_frozen_contract.json')
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write('\n')
