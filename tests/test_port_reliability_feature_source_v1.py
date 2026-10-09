import unittest
from copy import deepcopy
import numpy as np
from crane_project.utils import port_reliability_feature_source_v1 as core

try:
    import torch
except ImportError:
    torch=None


class NumericTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=np.random.RandomState(17).normal(size=(260,270))
        cls.pca=core.fit_pca(cls.raw,'TRAIN')

    def test_val_never_fits(self):
        with self.assertRaises(ValueError):core.fit_pca(self.raw,'VAL')

    def test_fixed_capacity(self):
        self.assertEqual(np.asarray(self.pca['components']).shape,(256,270))
        self.assertFalse(self.pca['whiten'])

    def test_pca_center_orthogonal(self):
        c=np.asarray(self.pca['components']);np.testing.assert_allclose(c@c.T,np.eye(256),atol=1e-12)
        np.testing.assert_allclose(core.project(self.raw,self.pca).mean(0),0.,atol=1e-12)

    def test_val_transform_does_not_refit(self):
        before=deepcopy(self.pca);v=core.project(self.raw[:2]+3.,self.pca)
        np.testing.assert_allclose(v,(self.raw[:2]+3.-np.array(before['mean']))@np.array(before['components']).T)
        self.assertEqual(before,self.pca)

    def test_descriptor_preserves_M(self):
        d=np.ones((260,3));x=core.features(d,self.raw,self.pca)
        np.testing.assert_array_equal(x[:,:3],d);self.assertEqual(x.shape,(260,259))

    def test_normalization_finite(self):
        x=core.features(np.ones((260,3)),self.raw,self.pca);n=core.normalization(x)
        self.assertTrue(np.isfinite(core.normalize(x,n)).all())
        self.assertEqual(n['scale'][:3],[1.,1.,1.])

    def test_shape_rejected(self):
        with self.assertRaises(ValueError):core.normalization(np.ones((3,291)))
        with self.assertRaises(ValueError):core.project(self.raw[:,:269],self.pca)

    def test_patch_zeros_channel_order(self):
        f=np.ones((256,4,5));p=core.patch_reference(f,0,0).reshape(256,3,3)
        np.testing.assert_array_equal(p[0],[[0,0,0],[0,1,1],[0,1,1]])
        f[7]=7.;self.assertEqual(core.patch_reference(f,2,2).reshape(256,3,3)[7,1,1],7.)

    def test_bad_patch_rejected(self):
        with self.assertRaises(ValueError):core.patch_reference(np.ones((32,9,9)),1,1)

    def test_balanced_objective(self):
        y=np.array([0.,0.,0.,1.]);w=core.class_weights(y)
        self.assertEqual(w[y==0].sum(),w[y==1].sum())

    def test_candidate_gate_control_identity(self):
        point=dict(actual=dict(states=dict(CR=95,FR=5,FA=3),runs=dict(correct_rejection=dict(longest=1))),
            control_tie_bounds={c:dict(bad_min_over_tie=4) for c in core.CONTROLS},
            matched_CR_controls={c:dict(states=dict(FA=4),exact_CR=True,runs=dict(correct_rejection=dict(longest=1))) for c in core.CONTROLS})
        self.assertEqual(core.gate(dict(all=point))['selected_arm'],'native')
        point['control_tie_bounds']['midpoint']['bad_min_over_tie']=3
        g=core.gate(dict(all=point));self.assertFalse(g['passed'])
        self.assertEqual(g['failures'][0]['control'],'midpoint')


@unittest.skipIf(torch is None,'Torch unavailable locally; execute on server')
class TorchTests(unittest.TestCase):
    def test_capacity_identity_and_fair_start(self):
        from crane_project.utils import port_reliability_feature_source_v1_torch as native
        m=native.make_models('cpu');x=torch.randn(8,259);z=torch.randn(8)
        self.assertEqual(sum(p.numel() for p in m['native'].parameters()),4290)
        self.assertEqual(native.exported(m['native']),native.exported(m['midpoint']))
        self.assertTrue(torch.equal(m['native'](x,z),-z))

    def test_numpy_replay_and_gradient(self):
        from crane_project.utils import port_reliability_feature_source_v1_torch as native
        m=native.make_models('cpu')['native'];optim=native.optimizer(m)
        x=torch.randn(8,259);z=x[:,0];y=torch.tensor([0.,1.]*4);w=torch.ones(8)
        native.update(m,optim,x,z,y,w);r=native.update(m,optim,x,z,y,w)
        self.assertGreater(r['gradient_by_parameter']['network.0.weight'],0.)
        reference=core.numpy_logits(native.exported(m),x.numpy(),dict(mean=[0.]*259,scale=[1.]*259))
        np.testing.assert_allclose(m(x,z).detach().numpy(),reference,atol=1e-6,rtol=1e-5)

    def test_observe_actual_topk_and_restore_trace(self):
        import sys
        from types import SimpleNamespace
        from crane_project.utils import port_reliability_feature_source_v1_torch as native
        class Config(dict):
            __getattr__=dict.__getitem__
        class Head(torch.nn.Module):
            def __init__(self):
                super().__init__();self.retina_reg=torch.nn.Conv2d(256,15,3,padding=1)
                self.cls_out_channels=1;self.use_sigmoid_cls=True;self.filter_padding_anchors=False
                self.test_cfg=Config(max_per_img=1,score_thr=.05,nms_pre=1)
            def _get_bboxes_single(self,cls_score_list,mlvl_anchors,img_shape,cfg):
                scores=torch.cat([v.permute(1,2,0).reshape(-1,1).sigmoid()[:,0] for v in cls_score_list])
                _,topk_inds=scores.topk(1)
                boxes=torch.zeros((1,6));boxes[0,5]=scores[topk_inds[0]]
                return boxes,torch.zeros(1,dtype=torch.long)
        class Detector:
            def __init__(self):self.bbox_head=Head()
            def simple_test(self,image,metas,rescale):
                self.bbox_head.retina_reg(image)
                cls=torch.zeros(3,2,2);cls[1,0,1]=2.;cls[2,1,0]=2.
                return self.bbox_head._get_bboxes_single([cls],[torch.zeros(12,5)],(2,2),self.bbox_head.test_cfg)
        d=Detector();image=torch.arange(1024,dtype=torch.float32).reshape(1,256,2,2)/1000
        result,p,t=native.trace_native(d,image,[dict(scale_factor=np.ones(4))])
        self.assertIsNone(sys.gettrace());self.assertEqual(t['tied_maximum'],2)
        self.assertIn(t['flat_anchor_index'],[4,8]);self.assertEqual(len(p),2304)
        np.testing.assert_array_equal(p,core.patch_reference(image[0].numpy(),t['y'],t['x']))

    def test_hooks_removed_on_failure(self):
        import sys
        from crane_project.utils import port_reliability_feature_source_v1_torch as native
        # Regression shape check fails before installing observers.
        class Detector:pass
        d=Detector();d.bbox_head=torch.nn.Module();d.bbox_head.retina_reg=torch.nn.Conv2d(256,15,1)
        with self.assertRaises(ValueError):native.trace_native(d,torch.zeros(1,256,2,2),[])
        self.assertIsNone(sys.gettrace());self.assertFalse(d.bbox_head.retina_reg._forward_hooks)


if __name__=='__main__':unittest.main()
