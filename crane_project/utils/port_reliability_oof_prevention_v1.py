"""Fixed auxiliary-A prevention window; not an OOF source or reliability result."""
from copy import deepcopy
from collections import deque
import math
from crane_project.utils.port_reliability_oof_followup_v1 import TRACE, anomaly
from crane_project.utils.port_reliability_oof_training_v2 import validate_config

VERSION = 'port_reliability_oof_prevention_v1'
SETTINGS = dict(max_updates=7000, gradient_limit=1000., main_cls_limit=5.,
                main_reg_limit=10., regression_budget=10., seed=1701)
HOOK = 'mmrotate.core.hooks.port_oof_prevention_hook'
ALLOCATION_REVISION = 'scalar_norm_finite_checks_v1'
MEMORY_REVISION = 'cache_aware_resource_v2'
MEMORY_SETTINGS = dict(warmup_steps=32, allocated_tolerance_bytes=16*2**20,
                       absolute_limit_bytes=4*2**30, trend_window_steps=64,
                       trend_check_interval=32, trend_tolerance_bytes=4*2**20,
                       sustained_growth_checks=2)


class CacheAwareMemoryGuard:
    """Bounded resources, not a claim that every cache increase is a leak.

    Cache growth below4GiB is recorded. End-of-backward active allocation is
    compared at the same point each step. A stable higher plateau or transient
    spike is allowed; two consecutive32-step checks must show both elevated
    active memory and an increasing64-step median trend to stop. Slow/small
    increases below the configured trend resolution remain subject to the cap.
    """
    def __init__(self, **settings):
        self.settings = dict(MEMORY_SETTINGS, **settings)
        if self.settings != MEMORY_SETTINGS:
            raise ValueError('Fixed cache-aware resource contract required')
        self.steps = self.warmup_peak = self.last_reserved = 0
        self.samples = deque(maxlen=self.settings['trend_window_steps'])
        self.growth_checks = self.cache_growth_events = 0
        self.max_allocated = self.max_reserved = 0
        self.max_peak_allocated = self.max_peak_reserved = 0

    def observe(self, allocated, reserved, peak_allocated=None, peak_reserved=None):
        self.steps += 1
        allocated, reserved = int(allocated), int(reserved)
        pa = allocated if peak_allocated is None else int(peak_allocated)
        pr = reserved if peak_reserved is None else int(peak_reserved)
        if min(allocated, reserved, pa, pr) < 0 or reserved < allocated:
            raise ValueError('Invalid allocator counters')
        warmup = self.steps <= self.settings['warmup_steps']
        growth = max(0, reserved-self.last_reserved)
        self.last_reserved = reserved
        if growth and not warmup:
            self.cache_growth_events += 1
        self.max_allocated = max(self.max_allocated, allocated)
        self.max_reserved = max(self.max_reserved, reserved)
        self.max_peak_allocated = max(self.max_peak_allocated, pa)
        self.max_peak_reserved = max(self.max_peak_reserved, pr)
        record = dict(step=self.steps, warmup=warmup, allocated_bytes=allocated,
            reserved_bytes=reserved, peak_allocated_bytes=pa, peak_reserved_bytes=pr,
            reserved_growth_bytes=growth, cache_growth_events=self.cache_growth_events,
            memory_revision=MEMORY_REVISION, sustained_growth_checks=self.growth_checks)
        if max(allocated, reserved, pa, pr) > self.settings['absolute_limit_bytes']:
            raise RuntimeError('CUDA memory exceeds fixed4GiB resource limit: %s'%record)
        if warmup:
            self.warmup_peak = max(self.warmup_peak, allocated)
        self.samples.append(allocated)
        if (not warmup and len(self.samples)==self.samples.maxlen and
                self.steps % self.settings['trend_check_interval']==0):
            values = list(self.samples);half=len(values)//2
            # Upper median makes a single stable upward jump cease to count
            # as increasing at the next non-overlapping32-step evaluation.
            early=sorted(values[:half])[half//2]
            late=sorted(values[half:])[half//2]
            increasing=(late > self.warmup_peak+self.settings['allocated_tolerance_bytes'] and
                        late-early >= self.settings['trend_tolerance_bytes'])
            self.growth_checks = self.growth_checks+1 if increasing else 0
            record.update(early_active_median_bytes=early, late_active_median_bytes=late,
                          active_median_growth_bytes=late-early,
                          sustained_growth_checks=self.growth_checks)
            if self.growth_checks >= self.settings['sustained_growth_checks']:
                raise RuntimeError('Persistent active CUDA allocation growth: %s'%record)
        record['warmup_peak_allocated_bytes'] = self.warmup_peak
        return record

    def summary(self):
        return dict(memory_revision=MEMORY_REVISION, observations=self.steps,
            cache_growth_events=self.cache_growth_events, max_allocated_bytes=self.max_allocated,
            max_reserved_bytes=self.max_reserved, max_peak_allocated_bytes=self.max_peak_allocated,
            max_peak_reserved_bytes=self.max_peak_reserved, sustained_growth_checks=self.growth_checks)


def finite_gradient_norm(gradients):
    """Reject NaN/Inf (including norm overflow) without gradient-sized masks."""
    import torch
    norms = [g.detach().norm() for g in gradients if g is not None]
    if not norms:
        raise FloatingPointError('Missing gradients')
    norm = torch.stack(norms).norm()
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError('Nonfinite gradient norm')
    return norm


def reasons(losses, mixed_norm):
    """Only the gradient actually delivered to global clip is a stop statistic."""
    result = anomaly(losses, mixed_norm, dict(TRACE, **SETTINGS))
    if any(not math.isfinite(float(v)) for v in losses.values()) and 'nonfinite' not in result:
        result.append('nonfinite')
    return result


def configure(cfg, directory):
    validate_config(cfg)
    cfg.checkpoint_config = None
    cfg.optimizer_config.type = 'PortOOFPreventionHook'
    cfg.optimizer_config.directory = str(directory)
    cfg.optimizer_config.settings = deepcopy(SETTINGS)
    cfg.optimizer_config.memory_guard = deepcopy(MEMORY_SETTINGS)
    cfg.custom_imports['imports'] = list(cfg.custom_imports['imports']) + [HOOK]
    cfg.custom_hooks = []
    check_config(cfg)
    return cfg


def check_config(cfg):
    if cfg.optimizer_config.type != 'PortOOFPreventionHook' or cfg.optimizer_config.settings != SETTINGS:
        raise ValueError('Unregistered prevention optimizer/limits')
    if HOOK not in cfg.custom_imports['imports'] or cfg.custom_hooks or cfg.checkpoint_config is not None:
        raise ValueError('No extra trace/checkpoint/selection hooks allowed')
    if cfg.data.val or cfg.data.test or cfg.evaluation is not None or cfg.workflow != [('train', 1)]:
        raise ValueError('TRAIN-only workflow required')
    if cfg.seed != SETTINGS['seed'] or cfg.runner.max_epochs != 24:
        raise ValueError('Preserve original seed and 24-epoch LR schedule')
    if cfg.optimizer_config.grad_clip != dict(max_norm=10, norm_type=2):
        raise ValueError('Original global clip10 required')
    if cfg.optimizer_config.memory_guard != MEMORY_SETTINGS:
        raise ValueError('Fixed cache-aware memory contract required')
    copied = deepcopy(cfg)
    copied.optimizer_config.type = 'PortStrictOptimizerHook'
    copied.optimizer_config.pop('directory')
    copied.optimizer_config.pop('settings')
    from crane_project.utils.port_reliability_oof_training_v2 import MEMORY_GUARD
    copied.optimizer_config.memory_guard = deepcopy(MEMORY_GUARD)
    validate_config(copied)


def measured_budget_backward(total, regression, parameters):
    """Same fixed repair, additionally measure raw and delivered mixed gradients.

    No extra total-loss backward is needed. Raw = other + regression; delivered
    = other + alpha*regression. Explicit addition preserves FP32 budget precision.
    """
    import torch
    if any(p.grad is not None for p in parameters):
        raise ValueError('Clear gradients before prevention backward')
    gradients = torch.autograd.grad(regression, parameters, retain_graph=True, allow_unused=True)
    norm = finite_gradient_norm(gradients)
    scale = min(1., SETTINGS['regression_budget'] / (float(norm) + 1e-12))
    (total - regression).backward()
    raw_norms = []
    for p, g in zip(parameters, gradients):
        other = p.grad
        if other is not None and not bool(torch.isfinite(other.detach().norm())):
            raise FloatingPointError('Nonfinite other gradients')
        raw = g.detach() if other is None and g is not None else other
        if other is not None and g is not None:
            raw = other.detach() + g.detach()
        if raw is not None:
            raw_norms.append(raw.norm())
        if g is not None:
            if other is None:
                p.grad = g.detach().mul(scale)
            else:
                p.grad.add_(g.detach(), alpha=scale)
        del raw
    mixed = finite_gradient_norm(p.grad for p in parameters)
    raw_norm = torch.stack(raw_norms).norm()
    if not torch.isfinite(mixed) or not torch.isfinite(raw_norm):
        raise FloatingPointError('Nonfinite gradient norm')
    return dict(raw_gradient_before_budget=float(raw_norm),
                gradient_before_clip=float(mixed), regression_gradient_norm=float(norm),
                regression_gradient_scale=scale, regression_gradient_budgeted_norm=float(norm)*scale,
                regression_budget=SETTINGS['regression_budget'])
