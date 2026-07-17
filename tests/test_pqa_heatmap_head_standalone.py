"""CPU-only tests for PQAHeatmapHead without an MMCV installation."""

import importlib.util
import math
import pathlib
import sys
import types
import unittest

import torch


class _Registry:
    def register_module(self, force=False):
        def decorate(cls):
            return cls
        return decorate


def _load_head_class():
    fake_mmcv = types.ModuleType('mmcv')
    fake_mmcv_cnn = types.ModuleType('mmcv.cnn')
    fake_mmcv_cnn.bias_init_with_prob = (
        lambda probability: float(-math.log((1.0 - probability) / probability)))
    fake_mmrotate = types.ModuleType('mmrotate')
    fake_models = types.ModuleType('mmrotate.models')
    fake_builder = types.ModuleType('mmrotate.models.builder')
    fake_builder.ROTATED_HEADS = _Registry()
    modules = {
        'mmcv': fake_mmcv,
        'mmcv.cnn': fake_mmcv_cnn,
        'mmrotate': fake_mmrotate,
        'mmrotate.models': fake_models,
        'mmrotate.models.builder': fake_builder,
    }
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        root = pathlib.Path(__file__).resolve().parents[1]
        path = root / 'mmrotate/models/dense_heads/pqa_heatmap_head.py'
        spec = importlib.util.spec_from_file_location(
            '_pqa_heatmap_head_standalone', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.PQAHeatmapHead
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


PQAHeatmapHead = _load_head_class()


class TestPQAHeatmapHead(unittest.TestCase):

    def test_shapes_prior_and_detached_gradient(self):
        head = PQAHeatmapHead(in_channels=4, prior_prob=0.01)
        prior_output = head((torch.zeros(2, 4, 16, 12),))[0]
        self.assertTrue(torch.allclose(
            prior_output.sigmoid().mean(), torch.tensor(0.01), atol=1e-5))
        feature = torch.randn(2, 4, 16, 12, requires_grad=True)
        output = head((feature.detach(),))[0]
        self.assertEqual(tuple(output.shape), (2, 1, 16, 12))
        output.sum().backward()
        self.assertIsNone(feature.grad)
        self.assertIsNotNone(head.heatmap_pred.weight.grad)

    def test_private_localization_tower(self):
        head = PQAHeatmapHead(
            in_channels=4, feat_channels=6, stacked_convs=2,
            prior_prob=0.01)
        output = head((torch.zeros(1, 4, 10, 8),))[0]
        self.assertEqual(tuple(output.shape), (1, 1, 10, 8))
        self.assertEqual(len(head.localization_tower), 4)
        self.assertTrue(torch.allclose(
            output.sigmoid().mean(), torch.tensor(0.01), atol=1e-5))

    def test_oriented_gaussian_target_and_ld_loss(self):
        logits = (torch.zeros(1, 1, 32, 32, requires_grad=True),)
        meta = [dict(img_shape=(32, 32, 3))]
        gt = [torch.tensor([[16.0, 16.0, 12.0, 6.0, 0.3]])]
        targets, valid = PQAHeatmapHead.build_targets(
            logits, meta, gt, strides=[1])
        self.assertGreater(float(targets[0].max()), 0.8)
        self.assertEqual(float(valid[0].min()), 1.0)
        loss, stats = PQAHeatmapHead.ld_loss(
            logits, targets, valid, gamma=2.0, loss_weight=1.5)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(stats['pqa_positive']), 0.0)
        loss.backward()
        self.assertIsNotNone(logits[0].grad)

    def test_volume_iou_prefers_aligned_candidate(self):
        empty_logits = (torch.zeros(1, 1, 32, 32),)
        meta = [dict(img_shape=(32, 32, 3))]
        gt = [torch.tensor([[16.0, 16.0, 12.0, 8.0, 0.0]])]
        targets, _ = PQAHeatmapHead.build_targets(
            empty_logits, meta, gt, strides=[1])
        probability = targets[0].clamp(1e-4, 1.0 - 1e-4)
        heatmap_logits = (torch.logit(probability),)
        boxes = torch.tensor([
            [16.0, 16.0, 12.0, 8.0, 0.0],
            [25.0, 25.0, 12.0, 8.0, 0.0],
        ])
        levels = torch.zeros(2, dtype=torch.long)
        quality = PQAHeatmapHead.quality_from_boxes(
            heatmap_logits, boxes, levels, (32, 32, 3),
            grid_size=15, batch_size=2)
        self.assertTrue(torch.isfinite(quality).all())
        self.assertGreater(float(quality[0]), float(quality[1]) + 0.3)

    def test_canonical_heatmap_scores_all_fpn_candidates(self):
        high_res = torch.full((1, 1, 16, 16), -10.0)
        high_res[:, :, 6:10, 6:10] = 10.0
        low_res = torch.full((1, 1, 8, 8), -10.0)
        boxes = torch.tensor([
            [8.0, 8.0, 4.0, 4.0, 0.0],
            [8.0, 8.0, 4.0, 4.0, 0.0],
        ])
        levels = torch.tensor([0, 1], dtype=torch.long)
        quality = PQAHeatmapHead.quality_from_boxes(
            (high_res, low_res), boxes, levels, (16, 16, 3),
            grid_size=9, batch_size=2, canonical_level=0)
        self.assertAlmostEqual(float(quality[0]), float(quality[1]), places=6)

    def test_consistency_updates_dark_only(self):
        clean = (torch.randn(1, 1, 8, 8, requires_grad=True),)
        dark = (torch.randn(1, 1, 8, 8, requires_grad=True),)
        target = (torch.zeros(1, 1, 8, 8),)
        target[0][:, :, 2:6, 2:6] = 0.8
        valid = (torch.ones_like(target[0]),)
        loss = PQAHeatmapHead.consistency_loss(
            clean, dark, target, valid, loss_weight=0.1)
        loss.backward()
        self.assertIsNone(clean[0].grad)
        self.assertIsNotNone(dark[0].grad)


if __name__ == '__main__':
    unittest.main()
