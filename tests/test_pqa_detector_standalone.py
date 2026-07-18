"""CPU-only selection test for the detector-side PQA inference path."""

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
    def __call__(self, boxes, gt_bboxes):
        distance = (boxes[:, None, 0] - gt_bboxes[None, :, 0]).abs()
        return (1.0 - distance / 10.0).clamp(0.0, 1.0)


def _load_detector_class():
    modules = {
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
    modules[
        'mmdet.models.detectors.single_stage'].SingleStageDetector = (
            _SingleStageDetector)
    builder = modules['mmrotate.models.builder']
    builder.ROTATED_DETECTORS = _Registry()
    builder.build_head = lambda cfg: cfg
    builder.build_loss = lambda cfg: cfg
    modules[
        'mmrotate.models.dense_heads.rotated_atss_head'].RotatedATSSHead = (
            _RotatedATSSHead)
    core = modules['mmrotate.core']
    core.build_assigner = lambda cfg: cfg
    core.rbbox2result = lambda bboxes, labels, classes: (bboxes, labels)
    modules[
        'mmrotate.core.bbox.iou_calculators'].RBboxOverlaps2D = _FakeOverlaps
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        root = pathlib.Path(__file__).resolve().parents[1]
        path = root / 'mmrotate/models/detectors/sym_eood_detector.py'
        spec = importlib.util.spec_from_file_location(
            '_sym_eood_detector_pqa_standalone', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.SymEOOD
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


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
        self.anchor_generator = _FakeAnchorGenerator(anchors)
        self.bbox_coder = _FakeCoder()


class _FakePQAHead(nn.Module):
    @staticmethod
    def quality_from_boxes(heatmaps, boxes, levels, pad_shape,
                           grid_size=9, batch_size=512,
                           canonical_level=None):
        del heatmaps, levels, pad_shape, grid_size, batch_size, canonical_level
        return boxes[:, 0] / 10.0


def _make_detector(score_mode='quality'):
    anchors = torch.tensor([
        [1.0, 1.0, 2.0, 2.0, 0.0],
        [3.0, 1.0, 2.0, 2.0, 0.0],
        [8.0, 1.0, 2.0, 2.0, 0.0],
        [9.0, 1.0, 2.0, 2.0, 0.0],
    ])
    detector = SymEOOD.__new__(SymEOOD)
    nn.Module.__init__(detector)
    detector.bbox_head = _FakeBBoxHead(anchors)
    detector.pqa_head = _FakePQAHead()
    detector.pqa_pre_topk = 3
    detector.pqa_score_mode = score_mode
    detector.pqa_grid_size = 9
    detector.pqa_quality_batch_size = 32
    detector.pqa_canonical_heatmap_level = None
    detector.pqa_rank_samples = 3
    detector.pqa_rank_mining_grid_size = 5
    detector.test_cfg = dict(max_per_img=1)
    return detector


class TestPQADetector(unittest.TestCase):

    def test_quality_only_ranks_inside_cls_pool(self):
        detector = _make_detector('quality')
        cls_scores = (torch.tensor([[[[5.0]], [[4.0]], [[3.0]], [[-8.0]]]]),)
        bbox_preds = (torch.zeros(1, 20, 1, 1),)
        heatmaps = (torch.zeros(1, 1, 1, 1),)
        meta = [dict(
            img_shape=(16, 16, 3), pad_shape=(16, 16, 3),
            scale_factor=[1.0, 1.0, 1.0, 1.0])]
        det_bboxes, labels = detector._pqa_get_bboxes(
            cls_scores, bbox_preds, heatmaps, meta)[0]
        # x=9 has the best Q but is excluded by cls-top3; x=8 must win.
        self.assertEqual(int(det_bboxes[0, 0]), 8)
        self.assertAlmostEqual(float(det_bboxes[0, 5]), 0.8, places=6)
        self.assertEqual(int(labels[0]), 0)

    def test_faithful_cls_times_quality_is_switchable(self):
        detector = _make_detector('cls_x_quality')
        cls_scores = (torch.tensor([[[[5.0]], [[4.0]], [[3.0]], [[-8.0]]]]),)
        bbox_preds = (torch.zeros(1, 20, 1, 1),)
        heatmaps = (torch.zeros(1, 1, 1, 1),)
        meta = [dict(
            img_shape=(16, 16, 3), pad_shape=(16, 16, 3),
            scale_factor=[1.0, 1.0, 1.0, 1.0])]
        det_bboxes, _ = detector._pqa_get_bboxes(
            cls_scores, bbox_preds, heatmaps, meta)[0]
        self.assertEqual(int(det_bboxes[0, 0]), 8)
        expected = float(torch.sigmoid(torch.tensor(3.0)) * 0.8)
        self.assertAlmostEqual(float(det_bboxes[0, 5]), expected, places=6)

    def test_rank_pool_contains_oracle_hard_and_low_iou_candidates(self):
        detector = _make_detector('quality')
        cls_scores = (torch.tensor([[[[5.0]], [[4.0]], [[3.0]], [[-8.0]]]]),)
        bbox_preds = (torch.zeros(1, 20, 1, 1),)
        meta = [dict(img_shape=(16, 16, 3), pad_shape=(16, 16, 3))]
        gt = [torch.tensor([[1.0, 1.0, 2.0, 2.0, 0.0]])]
        batch = detector._build_pqa_rank_batches(
            cls_scores, bbox_preds, (torch.zeros(1, 1, 1, 1),),
            meta, gt)[0]
        selected_x = set(float(value) for value in batch['boxes'][:, 0])
        # x=1 is the best-IoU/strongest-cls candidate, x=8 is the current PQA
        # false maximum, and x=3 fills the remaining slot.
        self.assertEqual(selected_x, {1.0, 3.0, 8.0})
        self.assertAlmostEqual(float(batch['target_ious'].max()), 1.0)


if __name__ == '__main__':
    unittest.main()
