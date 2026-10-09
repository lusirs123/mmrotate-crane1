import unittest
import numpy as np
try:
    import torch
    from crane_project.utils import port_reliability_spatial_axis_v1_torch as h
except ImportError:torch=None
from crane_project.utils import port_reliability_spatial_axis_v1 as c

@unittest.skipIf(torch is None,'Torch unavailable locally; server must run these checks')
class TorchSpatialTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(3)
        self.x=torch.randn(4,256,9,9,requires_grad=True);self.s=torch.ones(4,1,9,9);self.d=torch.randn(4,2)
        self.a=torch.tensor([.2,.3,.4,.5],dtype=torch.float64);self.y=torch.tensor([0.,1.,0.,1.],dtype=torch.float64);self.w=torch.ones(4,dtype=torch.float64);self.t=torch.randn(4,2,dtype=torch.float64)
    def test_same_init_capacity_neutral(self):
        m=h.make_models('cpu');self.assertEqual(h.exported(m[c.ARMS[0]]),h.exported(m[c.ARMS[-1]]))
        for a,v in m.items():
            self.assertEqual(sum(p.numel() for p in v.parameters()),13115)
            self.assertTrue(torch.equal(v(self.x,self.s,self.d,self.a,a)['risk'],self.a))
    def test_coarse_numpy(self):
        a=torch.nn.functional.avg_pool2d(self.x,3,3).repeat_interleave(3,2).repeat_interleave(3,3)
        np.testing.assert_allclose(a.detach(),c.coarse(self.x.detach().numpy()),atol=2e-7)
    def test_actual_gradients_and_no_detector(self):
        m=h.make_models('cpu')['spatial_axis'];v=h.update(m,h.optimizer(m),self.x,self.s,self.d,self.a,self.y,self.w,self.t,'spatial_axis',True)
        self.assertGreater(v['gradient_by_parameter']['stem.0.weight'],0)
        self.assertGreater(v['component_gradients']['weighted_aux_norm'],0)
        self.assertIsNone(self.x.grad)
    def test_controls_no_aux_backward(self):
        for a in c.ARMS[:2]:
            m=h.make_models('cpu')[a];v=h.update(m,h.optimizer(m),self.x,self.s,self.d,self.a,self.y,self.w,self.t,a,True)
            self.assertFalse(v['auxiliary_applied']);self.assertNotIn('axis_head.weight',v['gradient_by_parameter'])
            self.assertAlmostEqual(v['loss'],v['bce_loss'],places=13)
    def test_candidate_objective(self):
        m=h.make_models('cpu')['spatial_axis'];v=h.update(m,h.optimizer(m),self.x,self.s,self.d,self.a,self.y,self.w,self.t,'spatial_axis')
        self.assertAlmostEqual(v['loss'],v['bce_loss']+v['weighted_auxiliary_loss'],places=12)
    def test_smooth_l1_units_and_derivative(self):
        t=torch.tensor([[2.,-.5]],dtype=torch.float64,requires_grad=True);l=.25*torch.nn.functional.smooth_l1_loss(t,torch.zeros_like(t),beta=1.);l.backward()
        np.testing.assert_allclose(t.grad,[[.125,-.0625]])
    def test_numpy_replay_after_update(self):
        for a in c.ARMS:
            m=h.make_models('cpu')[a];h.update(m,h.optimizer(m),self.x,self.s,self.d,self.a,self.y,self.w,self.t,a)
            v=m(self.x,self.s,self.d,self.a,a);nr,nd,na=c.numpy_forward(h.exported(m),self.x.detach().numpy(),self.s.numpy(),self.d.numpy(),self.a.numpy(),a)
            np.testing.assert_allclose(nr,v['risk'].detach(),atol=2e-5);np.testing.assert_allclose(na,v['axis'].detach(),atol=2e-3)
    def test_neutral_risk_gradient(self):
        m=h.make_models('cpu')['spatial_axis'];v=m(self.x,self.s,self.d,self.a,'spatial_axis');v['risk'].sum().backward()
        self.assertGreater(float(m.risk_head.weight.grad.norm()),0)
    def test_aux_grad_finite_difference(self):
        m=h.make_models('cpu')['spatial_axis'].double();x=self.x.double();s=self.s.double();d=self.d.double()
        def loss():return .25*torch.nn.functional.smooth_l1_loss(m(x,s,d,self.a,'spatial_axis')['axis'],self.t,beta=1.)
        loss().backward();g=float(m.axis_head.bias.grad[0]);eps=1e-5
        with torch.no_grad():
            m.axis_head.bias[0]+=eps;p=float(loss());m.axis_head.bias[0]-=2*eps;n=float(loss());m.axis_head.bias[0]+=eps
        self.assertAlmostEqual(g,(p-n)/(2*eps),places=8)
    def test_second_control_hidden_gradient(self):
        m=h.make_models('cpu')['spatial_bce'];opt=h.optimizer(m)
        h.update(m,opt,self.x,self.s,self.d,self.a,self.y,self.w,self.t,'spatial_bce')
        v=h.update(m,opt,self.x,self.s,self.d,self.a,self.y,self.w,self.t,'spatial_bce');self.assertGreater(v['gradient_by_parameter']['stem.0.weight'],0)

if __name__=='__main__':unittest.main()
