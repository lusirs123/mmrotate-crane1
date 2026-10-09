import copy
import unittest
import numpy as np
try:
    import torch
except ModuleNotFoundError as error:
    if error.name!='torch':
        raise
    torch=None
if torch is not None:
    from crane_project.utils import port_reliability_within_video_rank_v1_torch as head
from crane_project.utils import port_reliability_within_video_rank_v1 as core


@unittest.skipIf(torch is None,'Torch absent locally; required before server training')
class TorchRankingTests(unittest.TestCase):
    def setUp(self):
        self.models=head.make_models('cpu')
        torch.manual_seed(1702)
        self.x=torch.randn(8,259,dtype=torch.float64)
        self.anchor=torch.linspace(.05,.95,8,dtype=torch.float64)
        self.y=torch.tensor([0,1]*4,dtype=torch.float64)
        self.w=torch.ones(8,dtype=torch.float64)
        self.indices=torch.arange(8)
        self.plan={'real_seq01':dict(bad=[1,3],good=[0,2],pairs=4),
                   'sim_seq08':dict(bad=[5,7],good=[4,6],pairs=4)}
        self.pairs=head.tensor_pairs(self.plan,'cpu')

    def update(self,arm,diagnose=True):
        model=self.models[arm]
        return head.update(model,head.optimizer(model),self.x,self.anchor,self.y,self.w,
                           self.indices,self.pairs,arm,diagnose)

    def test_initial_same_exact_and_capacity(self):
        self.assertEqual(head.exported(self.models['bce_control']),head.exported(self.models['within_video_rank']))
        for model in self.models.values():
            self.assertEqual(sum(p.numel() for p in model.parameters()),4290)
            self.assertTrue(torch.equal(model.risk(self.x,self.anchor),self.anchor))

    def test_neutral_readout_gradient(self):
        model=self.models['within_video_rank']; model.risk(self.x,self.anchor).sum().backward()
        expected=float((.5*self.anchor*(1-self.anchor)).sum())
        self.assertAlmostEqual(float(model.beta.grad),expected,places=13)

    def test_rank_torch_numpy_and_shift_invariance(self):
        s=torch.tensor([2.,-1.,.3,.4,10.,9.,11.,8.],dtype=torch.float64)
        actual,terms=head.rank_loss(s,self.pairs)
        expected,each=core.numpy_rank_loss(s.numpy(),self.plan)
        self.assertAlmostEqual(float(actual),expected,places=13)
        for name in each: self.assertAlmostEqual(float(terms[name]),each[name],places=13)
        self.assertAlmostEqual(float(head.rank_loss(s-100,self.pairs)[0]),expected,places=13)

    def test_rank_gradient_direction_and_video_offset_cancel(self):
        s=torch.zeros(8,dtype=torch.float64,requires_grad=True)
        head.rank_loss(s,self.pairs)[0].backward()
        self.assertTrue(bool((s.grad[self.y==1]<0).all()))
        self.assertTrue(bool((s.grad[self.y==0]>0).all()))
        for v in self.plan.values():
            self.assertAlmostEqual(float(s.grad[v['bad']+v['good']].sum()),0.,places=14)

    def test_full_head_rank_gradient_finite_difference(self):
        model=self.models['within_video_rank']
        loss=head.rank_loss(model(self.x,self.anchor),self.pairs)[0]
        grad=torch.autograd.grad(loss,model.network[4].weight)[0]
        j=int(grad.abs().argmax()); self.assertGreater(float(grad.abs().max()),0)
        parameter=model.network[4].weight
        with torch.no_grad():
            eps=1e-6; parameter[0,j]+=eps
            plus=float(head.rank_loss(model(self.x,self.anchor),self.pairs)[0])
            parameter[0,j]-=2*eps
            minus=float(head.rank_loss(model(self.x,self.anchor),self.pairs)[0])
            parameter[0,j]+=eps
        self.assertAlmostEqual((plus-minus)/(2*eps),float(grad[0,j]),places=8)

    def test_control_update_has_no_rank_gradient(self):
        model=self.models['bce_control']; expected=copy.deepcopy(model)
        result=self.update('bce_control')
        opt=head.optimizer(expected); opt.zero_grad()
        logits=expected(self.x,self.anchor)
        loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,self.y,reduction='none')*self.w).mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(expected.parameters(),5.); opt.step()
        self.assertEqual(head.exported(model),head.exported(expected))
        self.assertFalse(result['rank_applied'])
        self.assertEqual(result['loss'],result['bce_loss'])

    def test_candidate_gradient_is_bce_plus_fixed_rank(self):
        model=self.models['within_video_rank']; expected=copy.deepcopy(model)
        result=self.update('within_video_rank')
        opt=head.optimizer(expected); opt.zero_grad()
        logits=expected(self.x,self.anchor)
        bce=torch.nn.functional.binary_cross_entropy_with_logits(logits,self.y).mean()
        loss=bce+.25*head.rank_loss(logits,self.pairs)[0]
        loss.backward(); torch.nn.utils.clip_grad_norm_(expected.parameters(),5.); opt.step()
        self.assertEqual(head.exported(model),head.exported(expected))
        self.assertAlmostEqual(result['loss'],result['bce_loss']+result['weighted_rank_loss'],places=14)
        self.assertGreater(result['component_gradients']['weighted_rank_norm'],0)

    def test_second_step_hidden_gradient_and_numpy_reload(self):
        model=self.models['within_video_rank']; opt=head.optimizer(model)
        for _ in range(2):
            result=head.update(model,opt,self.x,self.anchor,self.y,self.w,self.indices,self.pairs,'within_video_rank',True)
        self.assertGreater(result['gradient_by_parameter']['network.0.weight'],0)
        self.assertLessEqual(result['gradient_norm_after'],5.+1e-12)
        saved=head.exported(model)
        normalizer=dict(mean=np.zeros(259),scale=np.ones(259))
        expected,_=core.numpy_forward(saved,self.x.numpy(),normalizer,self.anchor.numpy())
        np.testing.assert_allclose(model.risk(self.x,self.anchor).detach().numpy(),expected,rtol=1e-14,atol=1e-15)
        reloaded=head.make_models('cpu')['within_video_rank']; reloaded.load_state_dict(model.state_dict())
        self.assertTrue(torch.equal(model.risk(self.x,self.anchor),reloaded.risk(self.x,self.anchor)))

    def test_bound_and_finite_diagnostics(self):
        for arm in core.ARMS:
            result=self.update(arm)
            self.assertLessEqual(result['saturation_fraction'],1.)
            self.assertTrue(np.isfinite(result['component_gradients']['relative_strength']))
            self.assertTrue(bool((self.models[arm].delta(self.x).abs()<=.5).all()))


if __name__=='__main__': unittest.main()
