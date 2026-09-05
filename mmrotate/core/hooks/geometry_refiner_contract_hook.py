"""Runtime freeze/hash guard for source-only geometry-refiner training."""

import json
import os

import torch
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
        if report.get('teacher_refiner_enabled', False):
            if not report.get('teacher_refiner_eval', False):
                raise RuntimeError('Base-V3 teacher entered train mode')
            if report.get('teacher_refiner_trainable_parameter_count') != 0:
                raise RuntimeError('Base-V3 teacher has trainable parameters')
            if not report.get('teacher_refiner_hash_unchanged', False):
                raise RuntimeError('Base-V3 teacher changed before training')
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
        teacher = getattr(model, 'teacher_geometry_refiner', None)
        if teacher is not None:
            if teacher.training:
                raise RuntimeError('Base-V3 teacher entered train mode')
            if any(parameter.grad is not None
                   for parameter in teacher.parameters()):
                raise RuntimeError('Base-V3 teacher received a gradient')

    def after_run(self, runner):
        model = _unwrap(runner.model)
        report = model.verify_frozen_contract()
        if not report['frozen_hash_unchanged']:
            raise RuntimeError('Frozen SymEOOD parameters/buffers changed')
        if not report.get('frozen_refiner_hash_unchanged', True):
            raise RuntimeError(
                'Frozen geometry-refiner parameters/buffers changed')
        if not report.get('teacher_refiner_hash_unchanged', True):
            raise RuntimeError('Base-V3 teacher parameters/buffers changed')
        path = os.path.join(
            runner.work_dir, 'geometry_refiner_frozen_contract.json')
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write('\n')


@HOOKS.register_module()
class CudaPeakMemoryContractHook(Hook):
    """Record per-rank CUDA peaks without claiming zero memory overhead."""

    def before_run(self, runner):
        del runner
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())

    def after_run(self, runner):
        if not torch.cuda.is_available():
            return
        device = torch.cuda.current_device()
        report = dict(
            rank=int(getattr(runner, 'rank', 0)),
            visible_device_index=int(device),
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            device_name=torch.cuda.get_device_name(device),
            peak_allocated_bytes=int(
                torch.cuda.max_memory_allocated(device)),
            peak_reserved_bytes=int(
                torch.cuda.max_memory_reserved(device)))
        model = _unwrap(runner.model)
        counter = getattr(model, 'runtime_forward_counts', None)
        report['forward_counts'] = (
            counter() if callable(counter) else None)
        path = os.path.join(
            runner.work_dir,
            'cuda_peak_memory_rank{}.json'.format(report['rank']))
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write('\n')


@HOOKS.register_module()
class FixedRatioReplayEpochHook(Hook):
    """Rotate deterministic replay offsets between source epochs."""

    def before_run(self, runner):
        self.epoch_rows = []

    def before_train_epoch(self, runner):
        dataset = getattr(runner.data_loader, 'dataset', None)
        if dataset is None or not hasattr(dataset, 'set_epoch'):
            raise RuntimeError(
                'Fixed-ratio replay training requires set_epoch dataset')
        dataset.set_epoch(int(runner.epoch))
        self.current = dict(epoch=int(runner.epoch), original_steps=0,
                            auxiliary_steps=0, optimizer_steps=0)

    def after_train_iter(self, runner):
        dataset = runner.data_loader.dataset
        route = dataset.replay_route_for_optimizer_step(int(runner.inner_iter))
        key = 'auxiliary_steps' if route['auxiliary'] else 'original_steps'
        self.current[key] += 1
        self.current['optimizer_steps'] += 1

    def after_train_epoch(self, runner):
        dataset = runner.data_loader.dataset
        contract = dataset.replay_contract()
        expected = dict(
            original_steps=contract.get(
                'enumerated_original_steps',
                contract['scheduled_original_steps']),
            auxiliary_steps=contract.get(
                'enumerated_auxiliary_steps',
                contract['scheduled_auxiliary_steps']),
            optimizer_steps=contract['optimizer_steps_per_epoch'])
        if any(self.current[key] != value for key, value in expected.items()):
            raise RuntimeError(
                'Observed replay route differs from schedule contract: {} != {}'
                .format(self.current, expected))
        self.epoch_rows.append(dict(self.current))
        report = dict(
            protocol='fixed_ratio_replay_runtime_audit_v2',
            replay_schedule_protocol=contract['protocol'],
            schedule_sha256=contract['schedule_sha256'],
            offset_contract_steps=dict(
                original=contract['scheduled_original_steps'],
                auxiliary=contract['scheduled_auxiliary_steps']),
            epochs=list(self.epoch_rows),
            target_data_read=False, fixed_test_read=False)
        path = os.path.join(runner.work_dir,
                            'replay_schedule_runtime_audit.json')
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write('\n')
