import copy
import unittest
import numpy as np
import torch
from crane_project.utils import port_reliability_capacity_v1 as c
from crane_project.utils.port_reliability_feature_source_v1_torch import ResidualRisk
from crane_project.utils import port_reliability_feature_source_v1 as feature

class CapacityTests(unittest.TestCase):
    def test_small_exact_initialization_and_counts(self):
        for seed in c.SEEDS:
            torch.manual_seed(seed);old=ResidualRisk();new=c.make_model('small',seed)
            self.assertTrue(all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items()))
        self.assertEqual(sum(p.numel() for p in new.parameters()),4290)
        self.assertEqual(sum(p.numel() for p in c.make_model('large',1701).parameters()),8834)
    def test_neutral_and_hidden_gradient(self):
        for arm in ('small','large'):
            model=c.make_model(arm,1701);x=torch.randn(8,259,requires_grad=True);z=torch.randn(8,requires_grad=True)
            self.assertTrue(torch.equal(model(x,z),-z))
            opt=c.training.optimizer(model);y=torch.tensor([0.,1.]*4);w=torch.ones(8)
            c.training.update(model,opt,x,z,y,w)
            v=c.training.update(model,opt,x,z,y,w)
            self.assertGreater(v['gradient_by_parameter']['network.0.weight'],0)
            self.assertIsNone(x.grad);self.assertIsNone(z.grad)
            self.assertTrue(np.isfinite(v['loss']))
    def test_numpy_replay(self):
        for arm in ('small','large'):
            m=c.make_model(arm,1701)
            with torch.no_grad():m.network[4].weight.fill_(.01)
            a=np.random.RandomState(1).normal(size=(8,259));norm=dict(mean=[0.]*259,scale=[1.]*259)
            expected=m(torch.tensor(a,dtype=torch.float32),torch.tensor(a[:,0],dtype=torch.float32)).detach().numpy()
            replay=feature.numpy_logits(c.training.exported(m),a,norm)
            np.testing.assert_allclose(expected,replay,atol=1e-6)
    def test_identity_labels(self):
        rows=[dict(image='good',pred=[0,0,10,5,0,.5],gt=[0,0,10,5,0]),dict(image='bad',pred=[0,0,20,5,0,.5],gt=[0,0,10,5,0]),dict(image='missing',pred=None,gt=[0,0,10,5,0])]
        norm=dict(mean=[0.]*259,scale=[1.]*259);a=np.zeros((2,259))
        x,z,y=c.prepare(rows,a,['bad','good'],norm);self.assertEqual(y.tolist(),[1.,0.])
        with self.assertRaises(ValueError):c.prepare(rows,a,['good','good'],norm)
        with self.assertRaises(ValueError):c.prepare(rows,a,['missing','good'],norm)
        a[0,1]=float('nan')
        with self.assertRaises(ValueError):c.prepare(rows,a,['bad','good'],norm)
    def test_gate_predeclared_all_seeds_and_real(self):
        baseline={'all':{'states':{'FA':190}},'domain:real':{'states':{'CR':189}}}
        def primary(fa,real=189):return {'all':{'states':{'FA':fa}},'domain:real':{'states':{'CR':real}}}
        results={str(seed):{'small':{'VAL':{'primary':primary(195)}},'large':{'VAL':{'primary':primary(185)}}} for seed in c.SEEDS}
        self.assertTrue(c.gate(results,baseline)['passed']);self.assertEqual(c.gate(results,baseline)['representative_seed'],1701)
        r=copy.deepcopy(results);r['1702']['large']['VAL']['primary']=primary(186,188)
        self.assertFalse(c.gate(r,baseline)['passed'])
        r=copy.deepcopy(results);r['1703']['small']['VAL']['primary']=primary(180)
        self.assertFalse(c.gate(r,baseline)['passed'])
    def test_budget_rejects_partial_train(self):
        with self.assertRaises(ValueError):c.fit(np.zeros((5,259)),np.zeros(5),np.zeros(5),1701)
if __name__=='__main__':unittest.main()
