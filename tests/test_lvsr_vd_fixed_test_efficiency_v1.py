"""Locked D representative and D-only single-pass full-chain timing checks."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from crane_project.tools import eval_lvsr_vd_fixed_test_efficiency_v1 as t
from crane_project.tools import benchmark_crane_final_models_v1 as benchmark
from crane_project.utils.port_size_pair_direct_control_v1 import PairDirectSizeControlHead
from crane_project.utils.port_geometry_midpoint_sigma_v1 import SigmaMidpointHead
from crane_project.utils import port_geometry_refine_g_v1 as geometry


class FixedDChecks(unittest.TestCase):
    def test_D_only_frozen_scope(self):
        c=t.checked_config()
        self.assertEqual(c['efficiency_arms'],['D'])
        self.assertEqual((c['warmup_frames'],c['repeats'],c['TEST_frames']),(50,3,1440))
        self.assertEqual(c['D_sha256'],'b56ac54259edbbcc31d677dfc851d77aafae068b94afdfdd90d022209fffb4ce')

    def test_lock_rejects_different_epoch_hash_or_incomplete_training(self):
        from unittest.mock import patch
        cp=dict(path='D_epoch_02.pth',sha256=t.checked_config()['D_sha256'],epoch=2)
        s=dict(D_all24=[dict(epoch=i) for i in range(1,25)],representative_D=dict(checkpoint=cp),TEST=False)
        c=dict(status='DIRECT_CONTROL_TRAIN_VAL_COMPLETE',D_updates=15336,D_epochs=24,D_checkpoint=cp,TEST=False)
        selection=dict(representative_epoch=2,representative_checkpoint=cp,selected_epoch=None)
        with patch.object(t.training,'pick',return_value=dict(epoch=2)):
            self.assertEqual(t.validate_training(s,c,selection),cp)
            for field,value in [('D_updates',639),('D_epochs',2),('TEST',True)]:
                with self.assertRaises(ValueError):t.validate_training(s,dict(c,**{field:value}),selection)
            wrong=deepcopy(selection);wrong['representative_checkpoint']['epoch']=3
            with self.assertRaises(ValueError):t.validate_training(s,c,wrong)
            wrong=deepcopy(selection);wrong['representative_checkpoint']['sha256']='0'*64
            with self.assertRaises(ValueError):t.validate_training(s,c,wrong)

    def test_core_analysis_names_are_D_including_tuple(self):
        value=({'metrics':{'PartialVisualNorm':{'joint10':10}}},[dict(method='PartialVisualNorm')],[dict(PartialVisualNorm_MAE_pct=3.)])
        result=t.rename_D(value)
        self.assertEqual(result[0]['metrics']['D']['joint10'],10)
        self.assertEqual(result[1][0]['method'],'D')
        self.assertIn('D_MAE_pct',result[2][0])

    def test_actual_D_fast_path_matches_audited_chain_and_one_call(self):
        torch.set_num_threads(1)
        class Detector(torch.nn.Module):
            def __init__(self,empty):
                super().__init__();self.empty=empty;self.feature_calls=0;self.head_calls=0
            def extract_feat(self,image):
                self.feature_calls+=1
                return (image.new_zeros(1,256,8,8),)
            def simple_test_from_features(self,features,metas,rescale=False):
                self.head_calls+=1
                a=np.empty((0,6),dtype=np.float32) if self.empty else np.asarray([[30.,30.,20.,10.,.1,.9]],dtype=np.float32)
                if rescale:a[:,:4]/=np.asarray(metas[0]['scale_factor'],dtype=np.float32)
                return [[a]]
        meta=dict(ori_shape=(128,128,3),img_shape=(64,64,3),pad_shape=(64,64,3),
                  scale_factor=np.asarray([.5,.5,.5,.5],dtype=np.float32),flip=False)
        for empty in (False,True):
            with self.subTest(empty=empty),torch.no_grad():
                detector=Detector(empty);mid=SigmaMidpointHead(1.5);head=PairDirectSizeControlHead()
                head.terminal.branches[0].bias.fill_(.01);head.terminal.branches[1].bias.fill_(.005)
                for m in (detector,mid,head):m.eval().requires_grad_(False)
                p=SimpleNamespace(torch=torch,geometry=geometry,detector=detector,midpoint=mid,lvsr=head,
                    models=lambda:(detector,mid,head),counts=dict(feature_extractions=0,native_head_calls=0,midpoint_calls=0,lvsr_calls=0))
                image=torch.zeros(1,3,64,64)
                audited=benchmark.frozen.FrozenCranePipeline.predict(p,image,[meta])
                detector.feature_calls=detector.head_calls=0
                fast=benchmark.FinalForward(p,3);actual=fast(image,[meta]);fast.verify_invariants()
                benchmark.compare_top1(benchmark.top1(actual),audited['edge_residual'],score_atol=0.)
                self.assertEqual((detector.feature_calls,detector.head_calls),(1,1))
                benchmark.verify_single_pass(fast.calls,1,3)

    def test_latency_and_output_drift_guard(self):
        stats=benchmark.timing_summary([.01,.03]);self.assertAlmostEqual(stats['mean_ms'],20.)
        self.assertAlmostEqual(stats['fps'],50.)
        with self.assertRaises(ValueError):benchmark.timing_summary([0.])
        a=[1.,2.,20.,10.,.1,.9];b=a[:];b[5]+=.001
        with self.assertRaises(ValueError):benchmark.compare_top1(a,b,score_atol=0.)


if __name__=='__main__':unittest.main()
