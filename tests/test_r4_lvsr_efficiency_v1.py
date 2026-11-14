"""Compare LB single-pass timing route with the audited accuracy route on CPU."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from crane_project.tools import benchmark_r4_lvsr_efficiency_v1 as t
from crane_project.utils import port_geometry_refine_g_v1 as g
from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead


class LBEfficiencyChecks(unittest.TestCase):
    def test_scope_and_completed_identity(self):
        c=t.checked_config();self.assertEqual((c['warmup_frames'],c['frames'],c['repeats']),(50,1440,3))
        self.assertFalse(c['old_models_remeasure']);self.assertFalse(c['accuracy_evaluation'])
        cp=dict(path='head_epoch_03.pth',epoch=3,sha256=c['checkpoint_sha256'])
        done=dict(passed=True,frames=1440,checkpoint=cp,state_before={},state_after={},training_updates=0,
            midpoint_model_built=False,midpoint_forwards=0,selection_on_test=False)
        summary=dict(frames=1440,checkpoint=cp,automatic_adoption=False);terminal=dict(passed=True,frames=1440,checkpoint=cp)
        self.assertEqual(t.validate_TEST(done,summary,terminal),cp)
        for k,v in [('frames',100),('training_updates',1),('midpoint_forwards',1),('selection_on_test',True)]:
            with self.assertRaises(ValueError):t.validate_TEST(dict(done,**{k:v}),summary,terminal)
        wrong=deepcopy(terminal);wrong['checkpoint']['epoch']=5
        with self.assertRaises(ValueError):t.validate_TEST(done,summary,wrong)

    def test_fast_real_head_matches_audited_B_condition_and_single_call(self):
        torch.set_num_threads(1)
        class Detector(torch.nn.Module):
            def __init__(self,empty,swapped):
                super().__init__();self.empty=empty;self.swapped=swapped;self.f=0;self.h=0
            def extract_feat(self,image):self.f+=1;return (image.new_zeros(1,256,128,128),)
            def simple_test_from_features(self,features,metas,rescale=False):
                self.h+=1
                a=np.empty((0,6),dtype=np.float32) if self.empty else np.asarray([[30.,30.,10. if self.swapped else 20.,20. if self.swapped else 10.,.1,.9]],dtype=np.float32)
                if rescale:a[:,:4]/=np.asarray(metas[0]['scale_factor'],dtype=np.float32)
                return [[a]]
        meta=dict(ori_shape=(128,128,3),img_shape=(64,64,3),pad_shape=(64,64,3),
            scale_factor=np.asarray([.5,.5,.5,.5],dtype=np.float32),flip=False)
        image=torch.zeros(1,3,64,64)
        for empty,nonzero,swapped in ((False,False,False),(False,True,False),(False,True,True),(True,False,False)):
            with self.subTest(empty=empty,nonzero=nonzero,swapped=swapped),torch.no_grad():
                detector=Detector(empty,swapped).eval().requires_grad_(False);head=PairSpecificSizeHead().eval().requires_grad_(False)
                if nonzero:
                    head.terminal.branches[0].bias.copy_(torch.tensor([.003,0.,-.003,0.]))
                    head.terminal.branches[1].bias.copy_(torch.tensor([0.,.001,0.,-.001]))
                audited=t.fixed.capture_B(detector,head,image,[meta],dict(features=0,native_heads=0,LB_heads=0))
                detector.f=detector.h=0
                fast=t.LBForward(SimpleNamespace(detector=detector,lvsr=head,torch=torch,geometry=g))
                actual=t.b.top1(fast(image,[meta]));fast.verify_invariants();t.verify_counts(fast.calls,1)
                self.assertEqual(actual,audited['edge_residual']);self.assertEqual((detector.f,detector.h),(1,1))
                self.assertEqual(t.b.top1(fast.last[0].numpy()),audited['b'])

    def test_repeated_or_midpoint_stage_is_rejected(self):
        good=dict(features=4370,detector_head=4370,sampling=4370,midpoint=0,lvsr=4370);t.verify_counts(good,4370)
        for k,v in [('midpoint',4370),('detector_head',13110),('sampling',0)]:
            with self.assertRaises(ValueError):t.verify_counts(dict(good,**{k:v}),4370)


if __name__=='__main__':unittest.main()
