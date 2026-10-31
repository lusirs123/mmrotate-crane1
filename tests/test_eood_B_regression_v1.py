"""B formula, negative masking, native loss wiring, unchanged EOOD forward."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import runpy
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from crane_project.utils.eood_B_regression_protocol_v1 import candidate_model, check_model_delta, assess_detection

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'work_dirs/crane_eood_k1_port_day2night_aug_b_v1/crane_eood_k1_port_day2night_aug_b_v1.py'


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.old=runpy.run_path(str(REFERENCE))['model']

    def test_only_effective_regression_changes(self):
        saved=deepcopy(self.old);new=candidate_model(self.old)
        check_model_delta(self.old,new)
        self.assertEqual(self.old,saved)
        self.assertEqual(new['bbox_head']['predictors'][0]['train_cfg'],saved['bbox_head']['predictors'][0]['train_cfg'])
        self.assertEqual(new['bbox_head']['predictors'][1],saved['bbox_head']['predictors'][1])
        self.assertEqual(new['bbox_head']['predictors'][0]['loss_bbox'],saved['bbox_head']['predictors'][0]['loss_bbox'])

    def test_assignment_auxiliary_tower_and_inference_changes_rejected(self):
        for location,key in [('head','stacked_convs'),('main','train_cfg'),('main','test_cfg'),('main','loss_cls')]:
            new=candidate_model(self.old)
            obj=new['bbox_head'] if location=='head' else new['bbox_head']['predictors'][0]
            obj[key]={'changed':True}
            with self.assertRaises(ValueError):check_model_delta(self.old,new)

    def test_detection_gate_does_not_use_size_or_TEST(self):
        base=dict(frames=887,AP75=88.3318,Recall_pct=99.4363,center_MAE_px=2.13066,mean_RIoU=.873874)
        good=dict(base,AP75=89.5)
        self.assertTrue(assess_detection(good,base)['eligible'])
        self.assertFalse(assess_detection(dict(good,mean_RIoU=.87,joint10=887),base)['eligible'])
        with self.assertRaises(ValueError):assess_detection(dict(good,frames=886),base)


@unittest.skipUnless(importlib.util.find_spec('torch'),'Native torch/MMCV CPU runtime required')
class NativeCPUTests(unittest.TestCase):
    def test_B_value_and_gradient_match_positive_reference(self):
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
        from mmrotate.core.bbox.assigners.pola import PolaAssigner
        from mmrotate.models.losses.eood_sym_kld_positive_loss_v1 import EOODSymKLDPositiveLossV1
        old_cfg=Config.fromfile(str(REFERENCE)).model
        new_cfg=Config(dict(model=candidate_model(old_cfg.to_dict()))).model
        for cfg in (old_cfg,new_cfg):cfg.backbone.init_cfg=None
        old=build_detector(old_cfg).cpu().eval();new=build_detector(new_cfg).cpu().eval()
        new.load_state_dict(old.state_dict(),strict=True)
        self.assertEqual(sum(p.numel() for p in old.parameters()),36281432)
        self.assertEqual(sum(p.numel() for p in new.parameters()),36281432)
        main=new.bbox_head.predictors[0]
        self.assertIs(type(main.pola),PolaAssigner)
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
                losses=main.loss(*result[0][0],gt,labels,metas)
                self.assertIn('loss_iou',losses)
                self.assertNotIn('loss_bbox',losses)
                value=sum(losses['loss_iou'])
                self.assertTrue(torch.isfinite(value));value.backward()
                gradient=main.retina_reg.weight.grad
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(float(gradient.abs().sum()),0.)


if __name__=='__main__':unittest.main()
