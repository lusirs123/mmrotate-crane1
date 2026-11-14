"""B-conditioned frozen TEST: independent route, replay and reference-label guards."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np
import torch
from crane_project.tools import eval_r4_lvsr_fixed_test_v1 as t
from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead
from crane_project.utils import port_geometry_refine_g_v1 as g


class FixedLBChecks(unittest.TestCase):
    def test_fixed_scope_and_completed_training(self):
        c=t.checked_config();self.assertFalse(c['efficiency']);self.assertEqual(c['physical_gpu'],3)
        cp=dict(path='head_epoch_03.pth',epoch=3,sha256=c['checkpoint_sha256'])
        s=dict(epochs=[dict(epoch=i) for i in range(1,25)],representative=dict(checkpoint=cp),TEST=False)
        done=dict(status='R4_LB_TRAIN_VAL_COMPLETE',updates=15336,epochs=24,checkpoint=cp,selected_epoch=3,
            condition='B',midpoint_forwards=0,original_inputs_unchanged=True,derived_inputs_unchanged=True,
            actual_order_and_exposure_verified=True,TEST=False)
        selection=dict(representative_checkpoint=cp,representative_epoch=3,selected_epoch=3,gate_reference='R4 input B')
        with patch.object(t.training.io,'pick',return_value=dict(epoch=3)):
            self.assertEqual(t.validate_training(s,done,selection),cp)
            for k,v in [('condition','M'),('updates',1917),('midpoint_forwards',1),('TEST',True)]:
                with self.assertRaises(ValueError):t.validate_training(s,dict(done,**{k:v}),selection)
            wrong=deepcopy(selection);wrong['representative_checkpoint']['sha256']='0'*64
            with self.assertRaises(ValueError):t.validate_training(s,done,wrong)

    def test_real_head_B_only_neutral_nonzero_empty_and_coordinates(self):
        torch.set_num_threads(1)
        class Detector(torch.nn.Module):
            def __init__(self,empty=False,bad_restore=False):
                super().__init__();self.empty=empty;self.bad_restore=bad_restore;self.features=0;self.heads=0
            def extract_feat(self,image):
                self.features+=1;return (image.new_zeros(1,256,128,128),)
            def simple_test_from_features(self,features,metas,rescale=False):
                self.heads+=1
                a=np.empty((0,6),dtype=np.float32) if self.empty else np.asarray([[30.,30.,20.,10.,.1,.9]],dtype=np.float32)
                if rescale:a[:,:4]/=np.asarray(metas[0]['scale_factor'],dtype=np.float32)
                if rescale and self.bad_restore:a[:,0]+=1.
                return [[a]]
        meta=dict(ori_shape=(128,128,3),img_shape=(64,64,3),pad_shape=(64,64,3),
            scale_factor=np.asarray([.5,.5,.5,.5],dtype=np.float32),flip=False)
        image=torch.zeros(1,3,64,64)
        for empty,nonzero in ((False,False),(False,True),(True,False)):
            with self.subTest(empty=empty,nonzero=nonzero),torch.no_grad():
                detector=Detector(empty).eval().requires_grad_(False);head=PairSpecificSizeHead().eval().requires_grad_(False)
                if nonzero:
                    head.terminal.branches[0].bias.copy_(torch.tensor([.003,0.,-.003,0.]))
                    head.terminal.branches[1].bias.copy_(torch.tensor([0.,.001,0.,-.001]))
                counts=dict(features=0,native_heads=0,LB_heads=0)
                actual=t.capture_B(detector,head,image,[meta],counts)
                self.assertEqual(counts,dict(features=1,native_heads=3,LB_heads=1))
                self.assertEqual((detector.features,detector.heads),(1,3));self.assertEqual(actual['midpoint'],actual['b'])
                if empty:self.assertIsNone(actual['edge_residual']);continue
                b=torch.tensor([actual['b']]);raw=b[:,:5].clone();raw[:,:4]*=.5
                roi,support,_=g.sample_local(image.new_zeros(1,256,128,128),raw,meta,'aligned')
                direct=head(roi,support,b,raw,torch.tensor([[.5,.5]]),b)['boxes_original'][0].tolist()
                self.assertEqual(actual['edge_residual'],direct)
                for i in (0,1,4,5):self.assertEqual(actual['edge_residual'][i],actual['b'][i])
                if not nonzero:self.assertEqual(actual['b'],actual['edge_residual'])
                else:self.assertNotEqual(actual['b'][2:4],actual['edge_residual'][2:4])
        with self.assertRaises(ValueError):
            t.capture_B(Detector(bad_restore=True).eval(),PairSpecificSizeHead().eval().requires_grad_(False),image,[meta],dict(features=0,native_heads=0,LB_heads=0))

    def test_actual_M20_is_not_internal_B_alias(self):
        b=[1.,2.,20.,10.,.1,.9];m=[1.,2.,21.,11.,.1,.9];lm=[1.,2.,22.,12.,.1,.9]
        current=[];sealed=[]
        for i in range(1440):
            row=dict(image=str(i),domain='real',sequence='seq',frame_id=i,gt=b[:5],b=b,midpoint=b,edge_residual=b)
            current.append(row);sealed.append(dict(row,midpoint=m,edge_residual=lm))
        merged=t.merge_reference(current,sealed)
        self.assertEqual(merged[0]['midpoint'],b);self.assertEqual(merged[0]['M20'],m);self.assertEqual(merged[0]['LM'],lm)
        self.assertEqual(current[0]['midpoint'],b)
        bad=deepcopy(current);bad[0]['midpoint']=m
        with self.assertRaises(ValueError):t.merge_reference(bad,sealed)
        bad=deepcopy(current);bad[0]['b'][0]+=1.
        with self.assertRaises(ValueError):t.merge_reference(bad,sealed)


if __name__=='__main__':unittest.main()
