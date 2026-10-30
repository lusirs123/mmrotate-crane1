"""Geometry-only changes: CPU protocol, assigner and native EOOD head checks."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch

from crane_project.utils.eood_A_geometry_protocol_v1 import (
    candidate_model, check_model_delta, assess_detection)

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'work_dirs/crane_eood_k1_port_day2night_aug_b_v1/crane_eood_k1_port_day2night_aug_b_v1.py'


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.reference = runpy.run_path(str(REFERENCE))['model']

    def test_only_POLA_geometry_changes(self):
        old = deepcopy(self.reference)
        value = candidate_model(old)
        check_model_delta(old, value)
        self.assertEqual(self.reference, old)
        head = value['bbox_head']
        self.assertEqual(head['stacked_convs'], 4)
        self.assertEqual(head['predictors'][0]['train_cfg']['init_epoch'], 4)
        self.assertEqual(head['predictors'][1], old['bbox_head']['predictors'][1])

    def test_extra_loss_tower_schedule_or_inference_changes_rejected(self):
        for key in ('loss_cls', 'loss_iou', 'test_cfg'):
            value = candidate_model(self.reference)
            value['bbox_head']['predictors'][0][key] = {'changed': True}
            with self.assertRaises(ValueError):
                check_model_delta(self.reference, value)
        value = candidate_model(self.reference)
        value['bbox_head']['predictors'][0]['train_cfg']['init_epoch'] = 0
        with self.assertRaises(ValueError):
            check_model_delta(self.reference, value)

    def test_detection_gate_rejects_geometry_tradeoff_and_incomplete_VAL(self):
        base = dict(frames=887, AP75=88.3318, Recall_pct=99.4363,
                    center_MAE_px=2.13066, mean_RIoU=.873874)
        good = dict(base, AP75=base['AP75']+1.1)
        self.assertTrue(assess_detection(good, base)['eligible'])
        self.assertFalse(assess_detection(dict(good, center_MAE_px=2.2), base)['eligible'])
        with self.assertRaises(ValueError):
            assess_detection(dict(good, frames=886), base)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Native torch/MMCV CPU environment required')
class NativeCPUTests(unittest.TestCase):
    def inputs(self):
        import torch
        logits = torch.tensor([[-1.], [1.], [3.]], requires_grad=True)
        boxes = torch.tensor([[32., 32., 20., 10., 0.],
                              [36., 32., 20., 10., .2],
                              [80., 70., 30., 12., -.4]], requires_grad=True)
        gt = torch.tensor([[33., 32., 20., 10., .1]])
        return logits, boxes, torch.zeros(1, dtype=torch.long), gt, {}

    def test_neutral_replay_exact_and_empty_contract(self):
        import torch
        from mmrotate.core.bbox.assigners.pola import PolaAssigner
        from mmrotate.core.bbox.assigners.pola_kld_geometry_v1 import PolaKLDGeometryV1
        old, neutral = PolaAssigner(), PolaKLDGeometryV1(geometry_mode='iou')
        args = self.inputs()
        for values in (args, (args[0], args[1], args[2][:0], args[3][:0], {}),
                       (args[0][:0], args[1][:0], args[2], args[3], {})):
            a, b = old.assign(*values), neutral.assign(*values)
            for key in ('gt_inds', 'labels', 'max_overlaps'):
                self.assertTrue(torch.equal(getattr(a, key), getattr(b, key)))

    def test_kld_matches_explicit_cost_and_preserves_iou_metadata(self):
        import torch
        from mmrotate.core.bbox.assigners.pola import PolaAssigner
        from mmrotate.core.bbox.assigners.pola_kld_geometry_v1 import PolaKLDGeometryV1
        from mmrotate.models.losses.sym_kld_calculator import sym_kld
        logits, boxes, labels, gt, meta = self.inputs()
        original = PolaAssigner().assign(logits, boxes, labels, gt, meta)
        assigner = PolaKLDGeometryV1()
        result = assigner.assign(logits, boxes, labels, gt, meta)
        p = logits.detach().sigmoid()
        cost = (.25 * (1-p)**2 * -(p+1e-8).log()).reshape(-1) + sym_kld(boxes.detach(), gt)
        expected = int(cost.argmin())
        self.assertEqual(torch.nonzero(result.gt_inds > 0).reshape(-1).tolist(), [expected])
        self.assertTrue(torch.equal(original.max_overlaps, result.max_overlaps))
        self.assertFalse(result.max_overlaps.requires_grad)
        self.assertFalse(hasattr(assigner, '_local_call_count'))
        self.assertFalse(hasattr(assigner, 'tau_init'))
        with self.assertRaises(ValueError):
            PolaKLDGeometryV1(o2m=True)
        invalid = boxes.detach().clone()
        invalid[0, 0] = float('nan')
        with self.assertRaises(FloatingPointError):
            assigner.assign(logits, invalid, labels, gt, meta)

    def test_full_skeleton_forward_and_epoch_switch_unchanged(self):
        import torch
        from mmcv import Config
        from mmrotate.models import build_detector
        import mmrotate.core.bbox.assigners.pola_kld_geometry_v1
        old_config = Config.fromfile(str(REFERENCE)).model
        new_config = Config(dict(model=candidate_model(old_config.to_dict()))).model
        for value in (old_config, new_config):
            value['backbone']['init_cfg'] = None
        old = build_detector(old_config).cpu().eval()
        new = build_detector(new_config).cpu().eval()
        new.load_state_dict(old.state_dict(), strict=True)
        self.assertEqual(sum(p.numel() for p in old.parameters()), 36281432)
        self.assertEqual(sum(p.numel() for p in new.parameters()), 36281432)
        self.assertEqual(len(new.bbox_head.cls_convs), 4)
        self.assertEqual(len(new.bbox_head.reg_convs), 4)
        features = [torch.randn(1, 256, size, size) for size in (8, 4, 2, 1, 1)]
        with torch.no_grad():
            a, b = old.bbox_head(features), new.bbox_head(features)
        self.assertTrue(all(torch.equal(x, y) for group_a, group_b in zip(a, b)
                            for x, y in zip(group_a, group_b)))
        head = new.bbox_head.predictors[0]
        logits, anchors, labels, gt, _ = self.inputs()
        deltas = torch.zeros_like(anchors)
        args = (logits.detach(), deltas, anchors.detach(), torch.ones(3, dtype=torch.bool),
                gt, None, labels, dict(img_shape=(128,128,3)))
        new.set_epoch(3)
        with patch.object(head.pola, 'assign', side_effect=AssertionError('POLA called before epoch4')):
            head._get_targets_single(*args)
        new.set_epoch(4)
        with patch.object(head.assigner, 'assign', side_effect=AssertionError('MaxIoU called after epoch4')):
            result = head._get_targets_single(*args)
        self.assertEqual(result[4].numel(), 1)


if __name__ == '__main__':
    unittest.main()
