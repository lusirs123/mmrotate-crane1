# Copyright (c) OpenMMLab. All rights reserved.
"""SetEpochInfoHook：在 MMRotate 0.x 中复刻 MMRotate 1.x 的 epoch 注入机制。

EOOD 的 MaxIoU→Pola 切换严格依赖 detector 内部的 self.epoch 字段。
原版 MMRotate 1.x 通过 SetEpochInfoHook 在每个 epoch 开始前调用
detector.set_epoch(epoch)，让网络感知当前训练阶段。

本 Hook 是该机制的最小化复刻，兼容 MMCV 1.x 的 Hook 接口。
"""
from mmcv.runner.hooks import HOOKS, Hook
from mmcv.parallel import is_module_wrapper


@HOOKS.register_module(force=True)
class SetEpochInfoHook(Hook):
    """Inject current epoch number into the detector at the start of every epoch.

    This hook calls ``detector.set_epoch(epoch)`` before each training epoch,
    enabling assigner switching strategies (e.g. EOOD's MaxIoU→Pola transition
    at ``init_epoch``).

    Notes:
        - The detector must implement a ``set_epoch(epoch: int)`` method.
        - DistributedDataParallel/DataParallel wrappers are unwrapped automatically.
    """

    def before_train_epoch(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        if hasattr(model, 'set_epoch'):
            model.set_epoch(runner.epoch)
