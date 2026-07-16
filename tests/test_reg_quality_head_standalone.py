"""CPU-only structural tests for RegQualityHead without an MMCV install."""

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

    module_names = {
        'mmcv': fake_mmcv,
        'mmcv.cnn': fake_mmcv_cnn,
        'mmrotate': fake_mmrotate,
        'mmrotate.models': fake_models,
        'mmrotate.models.builder': fake_builder,
    }
    previous = {name: sys.modules.get(name) for name in module_names}
    sys.modules.update(module_names)
    try:
        root = pathlib.Path(__file__).resolve().parents[1]
        path = root / 'mmrotate/models/dense_heads/reg_quality_head.py'
        spec = importlib.util.spec_from_file_location(
            '_reg_quality_head_standalone', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.RegQualityHead
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


RegQualityHead = _load_head_class()


class TestRegQualityHead(unittest.TestCase):

    def test_multilevel_shapes_and_prior(self):
        head = RegQualityHead(
            in_channels=8, feat_channels=4, stacked_convs=2,
            num_anchors=3, prior_prob=0.01)
        outputs = head((
            torch.zeros(2, 8, 16, 12),
            torch.zeros(2, 8, 8, 6),
        ))
        self.assertEqual(tuple(outputs[0].shape), (2, 3, 16, 12))
        self.assertEqual(tuple(outputs[1].shape), (2, 3, 8, 6))
        self.assertTrue(torch.allclose(
            outputs[0].sigmoid().mean(), torch.tensor(0.01), atol=1e-5))

    def test_detached_input_is_gradient_isolated(self):
        head = RegQualityHead(
            in_channels=4, feat_channels=4, stacked_convs=1,
            num_anchors=1)
        feature = torch.randn(1, 4, 5, 5, requires_grad=True)
        loss = head((feature.detach(),))[0].sum()
        loss.backward()
        self.assertIsNone(feature.grad)
        self.assertTrue(any(
            parameter.grad is not None for parameter in head.parameters()))

    def test_invalid_shape_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            RegQualityHead(stacked_convs=0)
        with self.assertRaises(ValueError):
            RegQualityHead(num_anchors=0)


if __name__ == '__main__':
    unittest.main()
