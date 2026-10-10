"""Correctness of auxiliary-only training repairs; no data/model selection."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from crane_project.utils import port_reliability_oof_training_v2 as repair

ROOT = Path(__file__).resolve().parents[1]
try:
    import torch
    from mmcv import Config
    from mmrotate.models.losses.sym_nfl_loss import SymNFLLoss
    from mmrotate.models.losses.sym_kld_loss import SymKLDLoss
    from mmrotate.core.bbox.assigners.sym_pola import SymPOLAAssigner
    from mmrotate.core.hooks.port_strict_optimizer_hook import PortStrictOptimizerHook
    from mmrotate.models.builder import build_head
    TORCH_RUNTIME = True
except ImportError:
    TORCH_RUNTIME = False


class Attr(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def fixture_config():
    pipeline = [dict(type='Normalize', mean=[123.675, 116.28, 103.53],
                     std=[58.395, 57.12, 57.375], to_rgb=True),
                dict(type='Pad', size=(1024, 1024), pad_val=dict(img=(114.,)*3))]
    return Attr(data=Attr(train=[dict(pipeline=deepcopy(pipeline)) for _ in range(2)]),
        model=Attr(bbox_head=Attr(loss_cls=Attr(), loss_bbox=Attr()),
                   train_cfg=Attr(assigner=Attr())),
        custom_imports=dict(imports=[]), optimizer_config=Attr(grad_clip=dict(max_norm=10, norm_type=2)),
        load_from=None, resume_from=None, auto_resume=False)


class ConfigTests(unittest.TestCase):
    def test_memory_guard_accepts_stable_storage_and_cpu_scalar_state(self):
        guard = repair.MemoryPlateau(warmup_steps=2, allocated_tolerance_bytes=0)
        guard.observe(10, 20); guard.observe(15, 30)
        for _ in range(50): guard.observe(15, 30)
        self.assertEqual(guard.steps, 52)
        self.assertEqual(guard.reserved, 30)

    def test_memory_guard_rejects_growth_after_warmup(self):
        for allocated, reserved in ((11, 20), (10, 21)):
            guard = repair.MemoryPlateau(warmup_steps=1, allocated_tolerance_bytes=0)
            guard.observe(10, 20)
            with self.assertRaises(RuntimeError): guard.observe(allocated, reserved)

    def test_memory_guard_rejects_absolute_resource_overflow(self):
        guard = repair.MemoryPlateau(absolute_limit_bytes=20)
        with self.assertRaises(RuntimeError): guard.observe(10, 21)

    def test_raw_grey_padding_is_normalized_each_channel(self):
        cfg = repair.repair_config(fixture_config())
        repair.validate_config(cfg)
        for dataset in cfg.data.train:
            norm, pad = dataset['pipeline']
            for value, mean, std in zip(pad['pad_val']['img'], norm['mean'], norm['std']):
                self.assertAlmostEqual(value * std + mean, 114.)
                self.assertLess(abs(value), 1.)

    def test_no_double_normalization(self):
        cfg = repair.repair_config(fixture_config())
        with self.assertRaises(ValueError): repair.repair_config(cfg)

    def test_wrong_pipeline_order_rejected(self):
        cfg = fixture_config(); cfg.data.train[0]['pipeline'].reverse()
        with self.assertRaises(ValueError): repair.repair_config(cfg)

    def test_old_config_or_resume_cannot_enter_new_training(self):
        with self.assertRaises(ValueError): repair.validate_config(fixture_config())
        for key in ('load_from', 'resume_from', 'auto_resume'):
            cfg = repair.repair_config(fixture_config()); cfg[key] = True
            with self.assertRaises(ValueError): repair.validate_config(cfg)

    def test_disabled_guard_is_rejected(self):
        cfg = repair.repair_config(fixture_config()); cfg.model.bbox_head.loss_cls.isolate_images = False
        with self.assertRaises(ValueError): repair.validate_config(cfg)


@unittest.skipUnless(TORCH_RUNTIME, 'Actual Torch/MMCV runtime required')
class TrainingTests(unittest.TestCase):
    def test_actual_normalize_pad_has_no_large_background(self):
        import numpy as np
        from mmdet.datasets.pipelines import Compose
        cfg = repair.repair_config(fixture_config())
        ops = deepcopy(cfg.data.train[0]['pipeline']); ops[1]['size'] = (4,5)
        data = dict(img=np.full((2,3,3),114,dtype=np.uint8), img_fields=['img'])
        output = Compose(ops)(data)['img']
        expected = np.asarray(ops[1]['pad_val']['img'],dtype=np.float32)
        np.testing.assert_allclose(output, np.broadcast_to(expected,output.shape),rtol=0.,atol=1e-6)

    def boxes(self):
        return torch.tensor([[20.,20.,12.,4.,0.], [200.,200.,12.,4.,0.],
                             [300.,300.,12.,4.,0.], [20.,20.,12.,4.,0.],
                             [200.,200.,12.,4.,0.], [300.,300.,12.,4.,0.]])

    def test_spatial_gt_and_topk_are_independent_per_image(self):
        loss = SymNFLLoss(isolate_images=True, strict_finite=True, spatial_topk=1)
        boxes = self.boxes(); ids = torch.tensor([0,0,0,1,1,1])
        gt = boxes[[1,3]].clone(); gids = torch.tensor([0,1])
        mu, _ = loss._image_spatial_weights(boxes, gt, ids, gids, 1.)
        self.assertEqual(mu.tolist(), [1.,2.,1.,2.,1.,1.])
        gt[1] = boxes[5]
        changed, _ = loss._image_spatial_weights(boxes, gt, ids, gids, 1.)
        torch.testing.assert_allclose(mu[:3], changed[:3], rtol=0., atol=0.)
        self.assertEqual(changed[5].item(), 2.)

    def test_image_without_gt_uses_plain_focal_weights(self):
        loss = SymNFLLoss(isolate_images=True, spatial_topk=1)
        mu, q = loss._image_spatial_weights(self.boxes(), self.boxes()[[1]],
            torch.tensor([0,0,0,1,1,1]), torch.tensor([0]), 1.)
        self.assertEqual(mu[3:].tolist(), [1.,1.,1.])
        self.assertEqual(q[3:].tolist(), [1.,1.,1.])

    def test_empty_gt_and_invalid_ids(self):
        loss = SymNFLLoss(isolate_images=True)
        boxes = self.boxes(); ids = torch.zeros(6, dtype=torch.long)
        mu, _ = loss._image_spatial_weights(boxes, boxes[:0], ids, ids[:0], 1.)
        self.assertEqual(mu.tolist(), [1.]*6)
        with self.assertRaises(ValueError): loss._image_spatial_weights(boxes, boxes[:1], None, ids[:1], 1.)
        with self.assertRaises(ValueError): loss._image_spatial_weights(boxes, boxes[:1], ids, torch.tensor([2]), 1.)

    def test_single_image_equals_legacy_value_and_logit_gradient(self):
        for quality in (False, True):
            old = SymNFLLoss(use_quality_target=quality, spatial_topk=2).eval()
            new = SymNFLLoss(use_quality_target=quality, spatial_topk=2, isolate_images=True).eval()
            boxes = self.boxes()[:3]; gt = boxes[[1]]
            logits = torch.tensor([[-2.],[-1.],[-3.]], requires_grad=True)
            labels = torch.tensor([1,0,1]); ids = torch.zeros(3, dtype=torch.long)
            a = old(logits, labels, boxes, gt, avg_factor=1.)
            b = new(logits, labels, boxes, gt, avg_factor=1., pred_image_ids=ids, gt_image_ids=ids[:1])
            torch.testing.assert_allclose(a, b, rtol=0., atol=0.)
            torch.testing.assert_allclose(torch.autograd.grad(a, logits)[0],
                                          torch.autograd.grad(b, logits)[0], rtol=0., atol=0.)
            self.assertEqual(set(old.state_dict()), set(new.state_dict()))

    def test_batch_loss_equals_sum_of_separate_images(self):
        loss = SymNFLLoss(isolate_images=True, spatial_topk=1).eval()
        boxes = self.boxes(); gt = boxes[[1,3]]; ids = torch.tensor([0,0,0,1,1,1])
        logits = torch.zeros(6,1, requires_grad=True); labels = torch.tensor([1,0,1,0,1,1])
        combined = loss(logits, labels, boxes, gt, avg_factor=2., pred_image_ids=ids, gt_image_ids=torch.tensor([0,1]))
        parts = [loss(logits[i*3:(i+1)*3], labels[i*3:(i+1)*3], boxes[i*3:(i+1)*3], gt[i:i+1],
                      avg_factor=2., pred_image_ids=torch.zeros(3,dtype=torch.long), gt_image_ids=torch.zeros(1,dtype=torch.long))
                 for i in range(2)]
        torch.testing.assert_allclose(combined, sum(parts), rtol=1e-6, atol=1e-8)

    def test_strict_kld_does_not_silently_replace_nan(self):
        loss = SymKLDLoss(strict_finite=True)
        with patch('mmrotate.models.losses.sym_kld_loss.sym_kld', return_value=torch.tensor([float('nan')])):
            with self.assertRaises(FloatingPointError): loss(self.boxes()[:1], self.boxes()[:1])

    def test_strict_assigner_rejects_invalid_cost(self):
        assigner = SymPOLAAssigner(strict_finite=True)
        with patch('mmrotate.models.losses.sym_kld_calculator.sym_kld', return_value=torch.tensor([[float('nan')]])):
            with self.assertRaises(FloatingPointError):
                assigner.assign(torch.zeros(1,1), self.boxes()[:1], torch.tensor([0]), self.boxes()[:1])

    def runner(self, bad=False):
        model = torch.nn.Linear(2,1, bias=False)
        loss = model.weight.sum() * (float('nan') if bad else 100.)
        logs = {}
        buffer = SimpleNamespace(update=lambda x, n: logs.update(x))
        return SimpleNamespace(model=model, optimizer=torch.optim.SGD(model.parameters(), lr=.1),
                               outputs=dict(loss=loss, num_samples=1), log_buffer=buffer), logs

    def test_optimizer_reports_clip_and_applies_only_finite_gradients(self):
        runner, logs = self.runner(); hook = PortStrictOptimizerHook(grad_clip=dict(max_norm=10, norm_type=2))
        before = runner.model.weight.detach().clone(); hook.after_train_iter(runner)
        self.assertLessEqual(logs['gradient_after_clip'], 10.00001)
        self.assertGreater(logs['gradient_before_clip'], 10.)
        self.assertLess(logs['gradient_clip_fraction'], 1.)
        self.assertFalse(torch.equal(before, runner.model.weight))

    def test_actual_optimizer_memory_guard_stops_on_storage_growth(self):
        hook = PortStrictOptimizerHook(grad_clip=dict(max_norm=10, norm_type=2),
            memory_guard=dict(warmup_steps=1, allocated_tolerance_bytes=0))
        with patch('torch.cuda.synchronize'), patch('torch.cuda.memory_allocated', return_value=10), \
             patch('torch.cuda.memory_reserved', return_value=20):
            runner, logs = self.runner(); hook.after_train_iter(runner)
            self.assertIn('cuda_reserved_mib', logs)
        with patch('torch.cuda.synchronize'), patch('torch.cuda.memory_allocated', return_value=10), \
             patch('torch.cuda.memory_reserved', return_value=21):
            with self.assertRaises(RuntimeError): hook.after_train_iter(self.runner()[0])
        self.assertIsNone(hook._runner)

    def test_nonfinite_gradient_stops_before_parameter_update(self):
        runner, _ = self.runner(bad=True); before = runner.model.weight.detach().clone()
        hook = PortStrictOptimizerHook(grad_clip=dict(max_norm=10, norm_type=2))
        with self.assertRaises(FloatingPointError): hook.after_train_iter(runner)
        torch.testing.assert_allclose(before, runner.model.weight, rtol=0., atol=0.)
        self.assertEqual(hook.nonfinite_gradient_events, 1)

    def head(self, repaired=True):
        cfg = Config.fromfile(str(ROOT/'crane_project/configs/crane_symeood_k1_port_day2night_aug_b_v1.py'))
        if repaired: repair.repair_config(cfg)
        spec = deepcopy(cfg.model.bbox_head)
        spec.update(in_channels=16, feat_channels=16, stacked_convs=1,
                    train_cfg=cfg.model.train_cfg, test_cfg=cfg.model.test_cfg)
        return build_head(spec).train()

    def head_inputs(self):
        cls = [torch.full((2,3,n,n), -2., requires_grad=True) for n in (8,4,2,1,1)]
        bbox = [torch.zeros(2,15,n,n, requires_grad=True) for n in (8,4,2,1,1)]
        gt = [torch.tensor([[20.,20.,12.,4.,0.]]), torch.empty(0,5)]
        metas = [dict(img_shape=(64,64,3), pad_shape=(64,64,3)) for _ in range(2)]
        return cls, bbox, gt, [torch.tensor([0]),torch.empty(0,dtype=torch.long)], metas

    def test_actual_head_loss_routes_gt_ids_and_reports_positives(self):
        head = self.head(); result = head.loss(*self.head_inputs())
        self.assertEqual(result['main_positive_per_image_min'].item(), 0.)
        self.assertGreater(result['main_positive_count'].item(), 0.)
        total = sum(result['loss_cls']) + sum(result['loss_bbox']); total.backward()
        self.assertTrue(torch.isfinite(total))

    def test_actual_head_stops_on_bad_assignment_input(self):
        head = self.head(); args = self.head_inputs()
        with torch.no_grad(): args[0][0][0,0,0,0] = float('nan')
        with self.assertRaises(FloatingPointError): head.loss(*args)
        self.assertEqual(head._nonfinite_training_events, 1)

    def test_historical_config_bytes_and_defaults_remain_unchanged(self):
        old = Config.fromfile(str(ROOT/'crane_project/configs/crane_symeood_k1_port_day2night_aug_b_v1.py'))
        before = deepcopy(old.to_dict()); repaired = repair.repair_config(deepcopy(old))
        self.assertEqual(before, old.to_dict())
        old_head, new_head = self.head(False), self.head(True)
        self.assertFalse(old_head.strict_training_checks)
        self.assertFalse(old_head.loss_cls.isolate_images)
        self.assertEqual(set(old_head.state_dict()), set(new_head.state_dict()))
        self.assertEqual(old.model.test_cfg, repaired.model.test_cfg)
        self.assertEqual(old.optimizer, repaired.optimizer)
        self.assertEqual(old.lr_config, repaired.lr_config)


if __name__ == '__main__': unittest.main()
