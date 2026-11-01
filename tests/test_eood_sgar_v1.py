"""SGAR combined wiring, original EOOD schedule, and geometry contracts.

Defined for later execution; not run during the code-only implementation.
"""
from copy import deepcopy
import importlib.util
from pathlib import Path
import runpy
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from crane_project.utils.eood_sgar_protocol_v1 import (
    candidate_model, check_model_delta, candidate_config, check_config_delta, describe_change)

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'work_dirs/crane_eood_k1_port_day2night_aug_b_v1/crane_eood_k1_port_day2night_aug_b_v1.py'


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.old=runpy.run_path(str(REFERENCE))['model']

    def test_only_two_geometry_slots_change(self):
        saved=deepcopy(self.old);new=candidate_model(self.old)
        check_model_delta(self.old,new)
        self.assertEqual(self.old,saved)
        main=new['bbox_head']['predictors'][0]
        original=saved['bbox_head']['predictors'][0]
        restored=deepcopy(new)
        restored_main=restored['bbox_head']['predictors'][0]
        restored_main['train_cfg']['pola']=deepcopy(original['train_cfg']['pola'])
        restored_main['loss_iou']=deepcopy(original['loss_iou'])
        self.assertEqual(restored,saved)
        self.assertEqual(main['train_cfg']['pola']['type'],'PolaKLDGeometryV1')
        self.assertEqual(main['loss_iou']['type'],'EOODSymKLDPositiveLossV1')
        self.assertEqual(main['train_cfg']['init_epoch'],4)
        self.assertEqual(new['bbox_head']['predictors'][1],saved['bbox_head']['predictors'][1])
        self.assertEqual(new['bbox_head']['predictors'][0]['loss_bbox'],saved['bbox_head']['predictors'][0]['loss_bbox'])

    def test_assignment_auxiliary_tower_and_inference_changes_rejected(self):
        for location,key in [('head','stacked_convs'),('main','train_cfg'),('main','test_cfg'),('main','loss_cls')]:
            new=candidate_model(self.old)
            obj=new['bbox_head'] if location=='head' else new['bbox_head']['predictors'][0]
            obj[key]={'changed':True}
            with self.assertRaises(ValueError):check_model_delta(self.old,new)
        for change in ('auxiliary', 'schedule', 'coefficient'):
            new=candidate_model(self.old)
            if change=='auxiliary':
                new['bbox_head']['predictors'][1]['loss_cls']['loss_weight']=.01
            elif change=='schedule':
                new['bbox_head']['predictors'][0]['train_cfg']['init_epoch']=0
            else:
                new['bbox_head']['predictors'][0]['train_cfg']['pola']['cost_riou']=2.
            with self.assertRaises(ValueError):check_model_delta(self.old,new)

    def test_full_config_preserved_and_tradeoffs_only_described(self):
        old={k:v for k,v in runpy.run_path(str(REFERENCE)).items() if not k.startswith('__')}
        new=candidate_config(old,'work_dirs/eood_sgar_test/train')
        check_config_delta(old,new)
        self.assertEqual(new['seed'],0)
        changed=deepcopy(new);changed['optimizer']['lr']*=2
        with self.assertRaises(ValueError):check_config_delta(old,changed)
        baseline=dict(frames=887,AP75=88.3318,Recall_pct=99.4363,
                      center_MAE_px=2.13066,mean_RIoU=.873874)
        worse=dict(baseline,AP75=80.,center_MAE_px=3.)
        result=describe_change(worse,baseline)
        self.assertLess(result['candidate_minus_baseline']['AP75'],0.)
        self.assertGreater(result['candidate_minus_baseline']['center_MAE_px'],0.)
        self.assertIsNone(result['success_threshold'])
        self.assertNotIn('eligible',result)
        with self.assertRaises(ValueError):describe_change(dict(worse,frames=886),baseline)


@unittest.skipUnless(importlib.util.find_spec('torch'),'Native torch/MMCV CPU runtime required')
class NativeCPUTests(unittest.TestCase):
    def test_regression_value_and_gradient_match_positive_reference(self):
        import torch
        from mmrotate.models.losses.sym_kld_loss import SymKLDLoss
        from mmrotate.models.losses.eood_sym_kld_positive_loss_v1 import EOODSymKLDPositiveLossV1
        pred=torch.tensor([[22.,18.,16.,8.,.17],[48.,54.,20.,9.,-.12],[90.,80.,4.,3.,0.]],requires_grad=True)
        target=torch.tensor([[20.,20.,15.,7.,.11],[50.,52.,19.,10.,-.18],[0.,0.,0.,0.,0.]])
        weights=torch.tensor([1.,.5,0.]);loss=EOODSymKLDPositiveLossV1(loss_weight=2.,strict_finite=True)
        got=loss(pred,target,weights[:,None].expand(-1,5),avg_factor=2.)
        expected=SymKLDLoss(loss_weight=2.,strict_finite=True)(pred[:2],target[:2],weights[:2],avg_factor=2.)
        self.assertTrue(torch.equal(got,expected))
        a=torch.autograd.grad(got,pred,retain_graph=True)[0]
        b=torch.autograd.grad(expected,pred)[0]
        self.assertTrue(torch.equal(a,b));self.assertTrue(torch.isfinite(a).all())
        self.assertGreater(float(a[:2].abs().sum()),0.)
        self.assertEqual(float(a[2].abs().sum()),0.)
        self.assertEqual(sum(p.numel() for p in loss.parameters()),0)

    def test_empty_and_all_negative_gradients_are_finite_zero(self):
        import torch
        from mmrotate.models.losses.eood_sym_kld_positive_loss_v1 import EOODSymKLDPositiveLossV1
        loss=EOODSymKLDPositiveLossV1(loss_weight=2.,strict_finite=True)
        for n in (0,3):
            for reduction in ('mean','none'):
                pred=torch.ones((n,5),requires_grad=True)
                value=loss(pred,torch.zeros_like(pred),torch.zeros(n),avg_factor=1.,reduction_override=reduction)
                value.sum().backward()
                self.assertEqual(float(value.sum()),0.)
                self.assertTrue(torch.isfinite(pred.grad).all())
                self.assertEqual(float(pred.grad.abs().sum()),0.)

    def test_native_EOOD_forward_parameters_and_loss_wiring(self):
        import torch
        from mmcv import Config
        from mmrotate.models import build_detector
        from mmrotate.core.bbox.assigners.pola_kld_geometry_v1 import PolaKLDGeometryV1
        from mmrotate.models.losses.eood_sym_kld_positive_loss_v1 import EOODSymKLDPositiveLossV1
        old_cfg=Config.fromfile(str(REFERENCE)).model
        new_cfg=Config(dict(model=candidate_model(old_cfg.to_dict()))).model
        for cfg in (old_cfg,new_cfg):cfg.backbone.init_cfg=None
        old=build_detector(old_cfg).cpu().eval();new=build_detector(new_cfg).cpu().eval()
        new.load_state_dict(old.state_dict(),strict=True)
        self.assertEqual(sum(p.numel() for p in old.parameters()),36281432)
        self.assertEqual(sum(p.numel() for p in new.parameters()),36281432)
        main=new.bbox_head.predictors[0]
        self.assertIs(type(main.pola),PolaKLDGeometryV1)
        self.assertEqual(main.pola.geometry_mode,'kld')
        self.assertFalse(hasattr(main.pola,'tau_init'))
        self.assertIsInstance(main.loss_iou,EOODSymKLDPositiveLossV1)
        self.assertEqual(len(new.bbox_head.cls_convs),4);self.assertEqual(len(new.bbox_head.reg_convs),4)
        features=[torch.randn(2,256,n,n)*.05 for n in (8,4,2,1,1)]
        with torch.no_grad():a=old.bbox_head(features);b=new.bbox_head(features)
        self.assertTrue(all(torch.equal(x,y) for aa,bb in zip(a,b) for x,y in zip(aa,bb)))
        gt=[torch.tensor([[24.,24.,32.,16.,.05]]) for _ in range(2)]
        labels=[torch.zeros(1,dtype=torch.long) for _ in range(2)]
        metas=[dict(img_shape=(64,64,3),pad_shape=(64,64,3),scale_factor=1.,flip=False) for _ in range(2)]
        new.bbox_head.train()
        with ExitStack() as stack:
            # The unchanged legacy get_anchors passes device positionally to
            # an API whose second argument became dtype. Adapt device routing
            # only in this CPU test; the real GPU gate uses unmodified code.
            for predictor in new.bbox_head.predictors:
                generator=predictor.anchor_generator
                original=generator.grid_priors
                def cpu_grid(sizes,*args,_original=original,**kwargs):
                    kwargs.update(device='cpu',dtype=torch.float32)
                    return _original(sizes,**kwargs)
                stack.enter_context(patch.object(generator,'grid_priors',side_effect=cpu_grid))
            for epoch in (0,4):
                new.set_epoch(epoch);new.zero_grad(set_to_none=True)
                result=new.bbox_head(features)
                # Exercise the changed main predictor's actual loss return.
                # The untouched ATSS auxiliary head requires full-resolution
                # levels for top-k=9 and is covered by the real GPU gate.
                inactive=main.pola if epoch==0 else main.assigner
                with patch.object(inactive,'assign',
                        side_effect=AssertionError('Inactive assignment phase executed')):
                    losses=main.loss(*result[0][0],gt,labels,metas)
                self.assertIn('loss_iou',losses)
                self.assertNotIn('loss_bbox',losses)
                value=sum(losses['loss_iou'])
                self.assertTrue(torch.isfinite(value));value.backward()
                gradient=main.retina_reg.weight.grad
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(float(gradient.abs().sum()),0.)

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
            if not len(values[1]) or not len(values[3]):
                kld=PolaKLDGeometryV1().assign(*values)
                for key in ('gt_inds', 'labels', 'max_overlaps'):
                    self.assertTrue(torch.equal(getattr(a,key),getattr(kld,key)))

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



if __name__=='__main__':unittest.main()
