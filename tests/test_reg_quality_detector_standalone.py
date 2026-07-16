"""CPU-only tests for quality-primary selection and detached targets."""

import importlib.util
import pathlib
import sys
import types
import unittest

import torch
import torch.nn as nn


class _Registry:
    def register_module(self, force=False):
        def decorate(cls):
            return cls
        return decorate


class _SingleStageDetector(nn.Module):
    pass


class _RotatedATSSHead:
    pass


class _FakeOverlaps:
    def __call__(self, boxes, gt_boxes):
        # Deterministic continuous qualities based on x centre.
        values = (boxes[:, 0] / 10.0).clamp(0.0, 1.0)
        return values[:, None].expand(-1, gt_boxes.shape[0])


def _load_detector_class():
    module_names = {
        'mmdet': types.ModuleType('mmdet'),
        'mmdet.models': types.ModuleType('mmdet.models'),
        'mmdet.models.detectors': types.ModuleType('mmdet.models.detectors'),
        'mmdet.models.detectors.single_stage': types.ModuleType(
            'mmdet.models.detectors.single_stage'),
        'mmrotate': types.ModuleType('mmrotate'),
        'mmrotate.models': types.ModuleType('mmrotate.models'),
        'mmrotate.models.builder': types.ModuleType('mmrotate.models.builder'),
        'mmrotate.models.dense_heads': types.ModuleType(
            'mmrotate.models.dense_heads'),
        'mmrotate.models.dense_heads.rotated_atss_head': types.ModuleType(
            'mmrotate.models.dense_heads.rotated_atss_head'),
        'mmrotate.core': types.ModuleType('mmrotate.core'),
        'mmrotate.core.bbox': types.ModuleType('mmrotate.core.bbox'),
        'mmrotate.core.bbox.iou_calculators': types.ModuleType(
            'mmrotate.core.bbox.iou_calculators'),
    }
    module_names[
        'mmdet.models.detectors.single_stage'].SingleStageDetector = (
            _SingleStageDetector)
    builder = module_names['mmrotate.models.builder']
    builder.ROTATED_DETECTORS = _Registry()
    builder.build_head = lambda cfg: cfg
    builder.build_loss = lambda cfg: cfg
    module_names[
        'mmrotate.models.dense_heads.rotated_atss_head'].RotatedATSSHead = (
            _RotatedATSSHead)
    core = module_names['mmrotate.core']
    core.build_assigner = lambda cfg: cfg
    core.rbbox2result = lambda bboxes, labels, classes: (bboxes, labels)
    module_names[
        'mmrotate.core.bbox.iou_calculators'].RBboxOverlaps2D = _FakeOverlaps

    previous = {name: sys.modules.get(name) for name in module_names}
    sys.modules.update(module_names)
    try:
        root = pathlib.Path(__file__).resolve().parents[1]
        path = root / 'mmrotate/models/detectors/sym_eood_detector.py'
        spec = importlib.util.spec_from_file_location(
            '_sym_eood_detector_standalone', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.SymEOOD
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


SymEOOD = _load_detector_class()


class _FakeCoder:
    def decode(self, anchors, deltas, max_shape=None):
        return anchors + deltas


class _FakeAnchorGenerator:
    def __init__(self, anchors):
        self.anchors = anchors

    def grid_priors(self, featmap_sizes, device=None):
        return [self.anchors.to(device)]


class _FakeBBoxHead:
    use_sigmoid_cls = True
    cls_out_channels = 1
    num_classes = 1

    def __init__(self, anchors):
        self.num_anchors = anchors.shape[0]
        self.anchor_generator = _FakeAnchorGenerator(anchors)
        self.bbox_coder = _FakeCoder()

    def get_anchors(self, featmap_sizes, img_metas, device=None):
        anchors = self.anchor_generator.anchors.to(device)
        return ([[anchors] for _ in img_metas],
                [[torch.ones(anchors.shape[0], dtype=torch.bool,
                             device=device)] for _ in img_metas])


def _make_detector(pre_topk=3):
    anchors = torch.tensor([
        [1.0, 1.0, 2.0, 2.0, 0.0],
        [3.0, 1.0, 2.0, 2.0, 0.0],
        [8.0, 1.0, 2.0, 2.0, 0.0],
        [9.0, 1.0, 2.0, 2.0, 0.0],
    ])
    detector = SymEOOD.__new__(SymEOOD)
    nn.Module.__init__(detector)
    detector.bbox_head = _FakeBBoxHead(anchors)
    detector.reg_quality_pre_topk = pre_topk
    detector.reg_quality_focal_gamma = 2.0
    detector.reg_quality_min_target_iou = 0.1
    detector.reg_quality_loss_weight = 1.0
    detector.test_cfg = dict(max_per_img=1)
    return detector


class TestRegQualityDetector(unittest.TestCase):

    def test_quality_ranks_only_inside_cls_pool(self):
        detector = _make_detector(pre_topk=3)
        cls_scores = (torch.tensor([[[[5.0]], [[4.0]], [[3.0]], [[-8.0]]]]),)
        bbox_preds = (torch.zeros(1, 20, 1, 1),)
        # Anchor 3 has the highest quality but is outside cls-top3. Anchor 2
        # must therefore win, proving cls is a pool gate rather than fusion.
        quality = (torch.tensor([[[[-2.0]], [[0.0]], [[2.0]], [[8.0]]]]),)
        meta = [dict(
            img_shape=(16, 16, 3), pad_shape=(16, 16, 3),
            scale_factor=[1.0, 1.0, 1.0, 1.0])]
        result = detector._reg_quality_primary_get_bboxes(
            cls_scores, bbox_preds, quality, meta)[0]
        det_bboxes, labels = result
        self.assertEqual(int(det_bboxes[0, 0].item()), 8)
        self.assertAlmostEqual(
            float(det_bboxes[0, 5].item()),
            float(torch.sigmoid(torch.tensor(2.0)).item()), places=6)
        self.assertEqual(int(labels[0].item()), 0)

    def test_quality_target_does_not_backprop_to_cls_or_bbox(self):
        detector = _make_detector(pre_topk=3)
        quality = torch.zeros(1, 4, 1, 1, requires_grad=True)
        cls = torch.tensor(
            [[[[5.0]], [[4.0]], [[3.0]], [[-8.0]]]],
            requires_grad=True)
        bbox = torch.zeros(1, 20, 1, 1, requires_grad=True)
        meta = [dict(img_shape=(16, 16, 3), pad_shape=(16, 16, 3))]
        gt = [torch.tensor([[8.0, 1.0, 2.0, 2.0, 0.0]])]
        loss, stats = detector._compute_reg_quality_loss(
            (quality,), (cls,), (bbox,), meta, gt)
        loss.backward()
        self.assertIsNotNone(quality.grad)
        self.assertIsNone(cls.grad)
        self.assertIsNone(bbox.grad)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(stats['reg_quality_positive']), 0.0)


if __name__ == '__main__':
    unittest.main()
