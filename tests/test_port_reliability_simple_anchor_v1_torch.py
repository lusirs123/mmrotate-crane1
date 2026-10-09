import unittest
import numpy as np
try:
    import torch
    from crane_project.utils import port_reliability_simple_anchor_v1_torch as head
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Torch absent locally; required on server')
class TorchAnchorTests(unittest.TestCase):
    def setUp(self):
        self.models=head.make_models('cpu'); self.x=torch.randn(8,259,dtype=torch.float64)
        self.anchor=torch.linspace(.05,.95,8,dtype=torch.float64)

    def test_initial_same_exact(self):
        self.assertEqual(head.exported(self.models['score_anchor']),head.exported(self.models['simple_anchor']))
        for model in self.models.values():
            self.assertEqual(sum(p.numel() for p in model.parameters()),4290)
            self.assertTrue(torch.equal(model.risk(self.x,self.anchor),self.anchor))

    def test_neutral_readout_gradient(self):
        model=self.models['simple_anchor']; result=model.risk(self.x,self.anchor).sum(); result.backward()
        expected=float((.5*self.anchor*(1-self.anchor)).sum())
        self.assertAlmostEqual(float(model.beta.grad),expected,places=13)
        self.assertAlmostEqual(float(model.network[4].bias.grad),expected,places=13)

    def test_hidden_gradient_second_step(self):
        model=self.models['simple_anchor']; opt=head.optimizer(model)
        y=torch.tensor([0,1]*4,dtype=torch.float64); w=torch.ones(8,dtype=torch.float64)
        head.update(model,opt,self.x,self.anchor,y,w)
        result=head.update(model,opt,self.x,self.anchor,y,w)
        self.assertGreater(result['gradient_by_parameter']['network.0.weight'],0)
        self.assertLessEqual(result['gradient_norm_after'],5.)

    def test_risk_logit_equivalence(self):
        model=self.models['simple_anchor']
        with torch.no_grad(): model.beta.fill_(1.3)
        self.assertTrue(torch.allclose(model.risk(self.x,self.anchor),torch.sigmoid(model(self.x,self.anchor)),atol=1e-15,rtol=1e-14))
        self.assertTrue(bool((model.delta(self.x).abs()<=.5).all()))

    def test_numpy_replay(self):
        from crane_project.utils import port_reliability_simple_anchor_v1 as core
        model=self.models['simple_anchor']; opt=head.optimizer(model)
        head.update(model,opt,self.x,self.anchor,torch.tensor([0,1]*4,dtype=torch.float64),torch.ones(8,dtype=torch.float64))
        normalizer=dict(mean=np.zeros(259),scale=np.ones(259))
        expected,_=core.numpy_forward(head.exported(model),self.x.numpy(),normalizer,self.anchor.numpy())
        np.testing.assert_allclose(model.risk(self.x,self.anchor).detach().numpy(),expected,rtol=1e-14,atol=1e-15)


if __name__=='__main__': unittest.main()
