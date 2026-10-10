"""Auxiliary-only repair. Historical B configuration and inference stay intact."""
REVISION = 'normalized_padding_per_image_gt_strict_v2'
HOOK_MODULE = 'mmrotate.core.hooks.port_strict_optimizer_hook'
MEMORY_GUARD = dict(warmup_steps=32, allocated_tolerance_bytes=16 * 2**20,
                    reserved_tolerance_bytes=0, absolute_limit_bytes=4 * 2**30)


class MemoryPlateau:
    """CPU scalar state only: stop if post-warmup CUDA storage exceeds its plateau."""
    def __init__(self, **settings):
        self.settings = dict(MEMORY_GUARD, **settings)
        self.steps = 0
        self.allocated = self.reserved = 0

    def observe(self, allocated, reserved):
        self.steps += 1
        allocated, reserved = int(allocated), int(reserved)
        record = dict(step=self.steps, allocated_bytes=allocated, reserved_bytes=reserved,
                      warmup=self.steps <= self.settings['warmup_steps'])
        if reserved > self.settings['absolute_limit_bytes']:
            raise RuntimeError('CUDA memory exceeds fixed 4GiB resource limit: %s' % record)
        if record['warmup']:
            self.allocated = max(self.allocated, allocated)
            self.reserved = max(self.reserved, reserved)
        elif (allocated > self.allocated + self.settings['allocated_tolerance_bytes'] or
              reserved > self.reserved + self.settings['reserved_tolerance_bytes']):
            raise RuntimeError('CUDA memory grows after warmup; stop training: %s; plateau=%s' %
                               (record, dict(allocated_bytes=self.allocated, reserved_bytes=self.reserved)))
        record.update(plateau_allocated_bytes=self.allocated, plateau_reserved_bytes=self.reserved)
        return record


def repair_config(cfg):
    """Normalize raw grey padding; isolate per-image GT; fail on nonfinite training."""
    for dataset in cfg.data.train:
        pipeline = dataset['pipeline']
        norm_index = next(i for i, op in enumerate(pipeline) if op['type'] == 'Normalize')
        pad_index = next(i for i, op in enumerate(pipeline) if op['type'] == 'Pad')
        if norm_index >= pad_index:
            raise ValueError('Expected the frozen Normalize -> Pad pipeline')
        norm, pad = pipeline[norm_index], pipeline[pad_index]
        if tuple(pad['pad_val']['img']) != (114., 114., 114.):
            raise ValueError('Unexpected historical padding; refuse double normalization')
        pad['pad_val']['img'] = tuple((114. - mean) / std
                                     for mean, std in zip(norm['mean'], norm['std']))
    cfg.model.strict_training_checks = True
    cfg.model.bbox_head.strict_training_checks = True
    cfg.model.bbox_head.loss_cls.update(isolate_images=True, strict_finite=True)
    cfg.model.bbox_head.loss_bbox.strict_finite = True
    cfg.model.train_cfg.assigner.strict_finite = True
    cfg.optimizer_config.type = 'PortStrictOptimizerHook'
    cfg.optimizer_config.memory_guard = dict(MEMORY_GUARD)
    if HOOK_MODULE not in cfg.custom_imports['imports']:
        cfg.custom_imports['imports'].append(HOOK_MODULE)
    cfg.auxiliary_training_revision = REVISION
    return cfg


def validate_config(cfg):
    """Reject old run configurations instead of continuing old checkpoints."""
    if cfg.get('auxiliary_training_revision') != REVISION:
        raise ValueError('Fresh repaired auxiliary configuration required')
    if not cfg.model.strict_training_checks or not cfg.model.bbox_head.strict_training_checks:
        raise ValueError('Strict detector/head checks required')
    if not cfg.model.bbox_head.loss_cls.isolate_images:
        raise ValueError('Per-image GT isolation required')
    if not all((cfg.model.bbox_head.loss_cls.strict_finite,
                cfg.model.bbox_head.loss_bbox.strict_finite,
                cfg.model.train_cfg.assigner.strict_finite)):
        raise ValueError('Strict assignment/classification/regression checks required')
    if cfg.optimizer_config.type != 'PortStrictOptimizerHook' or HOOK_MODULE not in cfg.custom_imports['imports']:
        raise ValueError('Strict optimizer hook required')
    if cfg.optimizer_config.get('memory_guard') != MEMORY_GUARD:
        raise ValueError('Fixed CUDA memory plateau guard required')
    for dataset in cfg.data.train:
        norm = next(op for op in dataset['pipeline'] if op['type'] == 'Normalize')
        pad = next(op for op in dataset['pipeline'] if op['type'] == 'Pad')
        expected = tuple((114. - m) / s for m, s in zip(norm['mean'], norm['std']))
        if tuple(pad['pad_val']['img']) != expected:
            raise ValueError('Normalized grey padding required')
    if cfg.load_from is not None or cfg.resume_from is not None or cfg.get('auto_resume', False):
        raise ValueError('Repaired auxiliaries must start fresh, never resume old weights')
