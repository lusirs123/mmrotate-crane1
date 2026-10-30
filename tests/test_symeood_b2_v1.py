"""B retained, isolated architecture, overlap identity and geometry gradients."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

from crane_project.utils.symeood_b2_protocol_v1 import candidate_model, check_model_delta, assess_detection

ROOT = Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.reference = dict(bbox_head=dict(type='SymEOODHead', loss_cls=dict(use_geometry_weight=False),
            loss_bbox=dict(type='SymKLDLoss')), train_cfg={'assigner': 'unchanged'}, auxiliary={'weight': .01})

    def test_B_retained_and_only_regression_tower_changes_between_arms(self):
        a = candidate_model(self.reference, 'B2R0')
        b = candidate_model(self.reference, 'B2R4')
        self.assertGreater(a['bbox_head']['loss_bbox']['geometry_weight'], 0)
        b['bbox_head']['reg_tower_layers'] = 0
        self.assertEqual(a, b)
        self.assertEqual(self.reference['bbox_head']['loss_bbox']['type'], 'SymKLDLoss')

    def test_A_auxiliary_and_classifier_changes_are_rejected(self):
        for field in ('train_cfg', 'auxiliary'):
            value = candidate_model(self.reference, 'B2R0')
            value[field] = {'changed': True}
            with self.assertRaises(ValueError):
                check_model_delta(self.reference, value, 'B2R0')
        value = candidate_model(self.reference, 'B2R0')
        value['bbox_head']['loss_cls']['use_geometry_weight'] = True
        with self.assertRaises(ValueError):
            check_model_delta(self.reference, value, 'B2R0')

    def test_size_gains_cannot_hide_detection_failure(self):
        e = dict(frames=887, AP75=88.3318, Recall_pct=99.4363, center_MAE_px=2.13066, mean_RIoU=.873874)
        r = dict(e, AP75=e['AP75'] + 1.1)
        self.assertTrue(assess_detection(r, e)['eligible'])
        for field, value in [('center_MAE_px', 2.2), ('Recall_pct', 99.3), ('mean_RIoU', .87)]:
            bad = dict(r, **{field: value}, joint10=887)
            self.assertFalse(assess_detection(bad, e)['eligible'])


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Requires actual mmrotljj torch/MMCV runtime')
class NativeLossTests(unittest.TestCase):
    def test_head_matches_original_zero_and_four_layer_forward_paths(self):
        import torch
        from mmcv import Config
        from mmcv.utils import import_modules_from_strings
        from mmrotate.models import build_head
        import mmrotate.models.losses.sym_iou_hybrid_loss_v1
        import mmrotate.models.dense_heads.sym_eood_b2_head_v1
        import mmrotate.models.dense_heads.sym_eood_reg_tower_head_v1
        cfg = Config.fromfile(str(ROOT / 'work_dirs/symeood_110_retrain_seed0_20261026_v1/configs/Coff_seed0.py'))
        import_modules_from_strings(**cfg.custom_imports)
        torch.set_num_threads(1)
        old0 = build_head(deepcopy(cfg.model.bbox_head))
        reference = cfg.model.to_dict()
        new0 = build_head(candidate_model(reference, 'B2R0')['bbox_head'])
        new0.load_state_dict(old0.state_dict(), strict=True)
        old4cfg = deepcopy(cfg.model.bbox_head)
        old4cfg.type = 'SymEOODRegTowerHeadV1'
        old4 = build_head(old4cfg)
        new4 = build_head(candidate_model(reference, 'B2R4')['bbox_head'])
        new4.load_state_dict(old4.state_dict(), strict=True)
        inputs = torch.randn(2, 256, 8, 8)
        for old, new, count in ((old0, new0, 0), (old4, new4, 4)):
            expected, observed = old.forward_single(inputs), new.forward_single(inputs)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, observed)))
            self.assertEqual(len(new.reg_convs), count)
            self.assertFalse(hasattr(new, 'cls_convs') and len(new.cls_convs))
        self.assertEqual(sum(p.numel() for p in new4.parameters()) - sum(p.numel() for p in new0.parameters()), 2360320)

    def test_lambda_zero_replays_IoU_and_nonzero_geometry_has_gradients(self):
        import torch
        from mmrotate.models.losses.rotated_iou_loss import RotatedIoULoss
        from mmrotate.models.losses.sym_iou_hybrid_loss_v1 import SymIoUHybridLossV1
        # The differentiable rotated intersection operator requires CUDA in
        # this project runtime; stop rather than substitute another IoU.
        if not torch.cuda.is_available():
            self.skipTest('Native rotated IoU gradient requires CUDA')
        pred = torch.tensor([[22., 18., 16., 8., .17], [48., 54., 20., 9., -.12]], device='cuda', requires_grad=True)
        target = torch.tensor([[20., 20., 15., 7., .11], [50., 52., 19., 10., -.18]], device='cuda')
        weights = torch.tensor([1., .5], device='cuda')
        original = RotatedIoULoss(mode='linear', diou=True, loss_weight=2.)
        neutral = SymIoUHybridLossV1(geometry_weight=0.)
        self.assertTrue(torch.allclose(original(pred, target, weights, avg_factor=2.), neutral(pred, target, weights, avg_factor=2.), atol=1e-7, rtol=0))
        loss = SymIoUHybridLossV1(geometry_weight=.25)
        terms = loss.components(pred, target, weights, avg_factor=2.)
        gradients = {name: torch.autograd.grad(value, pred, retain_graph=True)[0] for name, value in terms.items()}
        for value in gradients.values():
            self.assertTrue(torch.isfinite(value).all())
            self.assertGreater(float(value.abs().sum()), 0.)
        loss(pred, target, weights, avg_factor=2.).backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(float(pred.grad.abs().sum()), 0.)

    def test_empty_and_zero_weight_samples_keep_zero_finite_gradients(self):
        import torch
        from mmrotate.models.losses.sym_iou_hybrid_loss_v1 import SymIoUHybridLossV1
        loss = SymIoUHybridLossV1()
        for count in (0, 2):
            pred = torch.ones((count, 5), requires_grad=True)
            value = loss(pred, pred.detach().clone(), torch.zeros(count), avg_factor=1.)
            self.assertEqual(float(value), 0.)
            value.backward()
            self.assertTrue(torch.isfinite(pred.grad).all())
            self.assertEqual(float(pred.grad.abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
