# Copyright (c) OpenMMLab. All rights reserved.
"""Compatibility registries for local OpenMMLab 2.x style extensions.

This repository uses the MMRotate 0.x/MMDetection 2.x builder layout, where
registries live under ``mmrotate.models.builder`` and ``mmrotate.core.bbox``.
Some project-local modules import ``mmrotate.registry`` following the newer
OpenMMLab 2.x layout; this module keeps those imports resolvable while mapping
them back to the registries used by this codebase.
"""

from mmcv.utils import Registry
from mmdet.models.builder import MODELS

from mmrotate.core.bbox.builder import ROTATED_BBOX_ASSIGNERS

TASK_UTILS = ROTATED_BBOX_ASSIGNERS

try:
    from mmengine.registry import METRICS  # type: ignore
except Exception:
    METRICS = Registry('metric')

__all__ = ['METRICS', 'MODELS', 'TASK_UTILS']
