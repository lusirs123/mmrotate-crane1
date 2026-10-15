import copy
import math
import unittest
import numpy as np
import torch
from crane_project.utils import port_reliability_response_warm_v1 as core
from crane_project.utils import port_reliability_response_tail_v1 as old
from crane_project.utils import port_reliability_response_stats_v1 as response
from crane_project.utils.port_reliability_response_stats_v1_torch import Risk
from crane_project.tools.review_port_reliability_response_warm_v1 import independent_model


def parent_fixture(raw):
    m=Risk(1701,19)
    with torch.no_grad():
        m.network[0].weight[:,3:]=torch.tensor(np.random.RandomState(2).normal(scale=.1,size=(8,16)))
    return dict(seed=1701,dimensions=19,objective_kind='bce',normalizer=response.normalize_fit(raw,'TRAIN'),
        weights={k:v.detach().tolist() for k,v in m.state_dict().items()},final_state_sha256=old.state_digest(m))


class WarmTests(unittest.TestCase):
    def test_soft_formula_direction_shift_stability(self):
        z=torch.tensor([-1.,.2,.8,-.4,1.],dtype=torch.float64); y=torch.tensor([0,0,0,1,1])
        t=.2; expected=0
        for b in z[3:].tolist():
            v=[math.log1p(math.exp(g-b))/t for g in z[:3].tolist()]
            a=max(v); expected+=t*(a+math.log(sum(math.exp(x-a) for x in v)/3))/2
        self.assertAlmostEqual(core.soft_loss(z,y,t).item(),expected,places=12)
        self.assertAlmostEqual(core.soft_loss(z+10,y,t).item(),expected,places=12)
        z2=z.clone(); z2[:3]-=1; z2[3:]+=1
        self.assertLess(core.soft_loss(z2,y,t),core.soft_loss(z,y,t))
        self.assertTrue(torch.isfinite(core.soft_loss(z*1000,y,1e-6)))
        with self.assertRaises(ValueError): core.soft_loss(z,y,0)

    def test_soft_gradient_finite_difference(self):
        z=torch.tensor([-.7,.1,.9,-.3,1.2],dtype=torch.float64,requires_grad=True)
        y=torch.tensor([0,0,0,1,1]); v=core.soft_loss(z,y,.3); v.backward()
        self.assertAlmostEqual(z.grad.sum().item(),0.,places=12)
        for i in range(len(z)):
            p=z.detach().clone(); p[i]+=1e-6; a=core.soft_loss(p,y,.3)
            p[i]-=2e-6; b=core.soft_loss(p,y,.3)
            self.assertAlmostEqual(z.grad[i].item(),((a-b)/2e-6).item(),places=8)

    def test_temperature_train_only_target_and_support_limits(self):
        raw=np.random.RandomState(3).normal(size=(120,19)); labels=np.r_[np.zeros(100),np.ones(20)]
        parent=parent_fixture(raw); v=core.temperature_fit(raw,labels,[parent]*3)
        self.assertAlmostEqual(v['initial_mean_ESS'],2.,places=9)
        self.assertTrue(v['TRAIN_only']); self.assertTrue(v['no_exact_FPR_equivalence'])
        p=torch.tensor([[0.,.5,1.]],dtype=torch.float64)
        self.assertLess(core.effective_support(p,.01),core.effective_support(p,1.))

    def test_warm_start_paired_three_arms_export(self):
        torch.set_num_threads(1)
        raw=np.random.RandomState(8).normal(size=(25,19)); labels=np.r_[np.zeros(18),np.ones(7)]
        parent=parent_fixture(raw); saved=copy.deepcopy(core.SETTINGS)
        try:
            core.SETTINGS['steps']=3; values=[]
            for kind in core.ARMS:
                obj,export,records=core.fit(raw,labels,parent,kind,.2,'cpu')
                values.append(export['initial_state_sha256'])
                self.assertEqual(export['normalizer'],parent['normalizer'])
                self.assertNotEqual(export['initial_state_sha256'],export['final_state_sha256'])
                self.assertTrue(export['optimizer_reset']); self.assertEqual(records[-1]['step'],3)
                np.testing.assert_allclose(response.numpy_logits(export,raw),obj(torch.tensor(response.normalize(raw,parent['normalizer']))).detach().numpy(),atol=1e-12)
                independent_model(export,raw,labels)
            self.assertEqual(len(set(values)),1)
            parent['normalizer']['mean'][0]+=1
            with self.assertRaises(ValueError):core.fit(raw,labels,parent,'soft',.2,'cpu')
        finally:
            core.SETTINGS.clear();core.SETTINGS.update(saved)

    def test_preserved_center_and_missing(self):
        value=dict(final_box_original=[1,2,3,4,0,.9],center_accepted=True,angle_accepted=False,size_accepted=True)
        other=response.size_decision(value,.8,.4)
        self.assertTrue(other['center_accepted']);self.assertFalse(other['size_accepted'])
        self.assertEqual(other['final_box_original'],value['final_box_original'])
        missing=dict(value,final_box_original=None,center_accepted=False,size_accepted=False)
        self.assertFalse(response.size_decision(missing,None,.4)['size_accepted'])

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA validation runs on server')
    def test_gpu_fit_and_independent_cpu_gradient(self):
        raw=np.random.RandomState(17).normal(size=(30,19));y=np.r_[np.zeros(22),np.ones(8)]
        p=parent_fixture(raw);saved=copy.deepcopy(core.SETTINGS)
        try:
            core.SETTINGS['steps']=3
            for kind in core.ARMS:
                _,v,_=core.fit(raw,y,p,kind,.2,'cuda:0')
                self.assertEqual(v['training_device'],'cuda:0')
                independent_model(v,raw,y)
        finally:
            core.SETTINGS.clear();core.SETTINGS.update(saved)


if __name__=='__main__':unittest.main()
