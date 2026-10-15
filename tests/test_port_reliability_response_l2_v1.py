import unittest
import numpy as np
import torch
from crane_project.utils import port_reliability_response_l2_v1 as core
from crane_project.utils import port_reliability_response_warm_v1 as warm
from crane_project.tools.review_port_reliability_response_warm_v1 import independent_model
from test_port_reliability_response_warm_v1 import parent_fixture

class L2ControlTests(unittest.TestCase):
    def test_penalty_only_change(self):
        raw=np.random.RandomState(1).normal(size=(20,19));y=torch.tensor([0.]*15+[1.]*5)
        m=core.Risk(1701,19);x=torch.tensor(raw)
        for arm in core.ARMS:
            before=warm.objective(m,x,y,arm,.2);after=core.objective(m,x,y,arm,.2)
            penalty=.05*sum(p.square().sum() for n,p in m.named_parameters() if n.endswith('weight'))
            self.assertAlmostEqual((before-after).item(),penalty.item(),places=12)
        self.assertEqual(warm.SETTINGS['l2'],.1)
    def test_train_rank_invariant_to_positive_scaling_and_whole_ties(self):
        z=np.array([0.,1.,1.,2.,2.,3.]);y=np.array([0,0,1,0,1,1])
        a=core.train_points(z,y);b=core.train_points(z*3,y)
        for key in a:
            self.assertEqual(a[key]['FA'],b[key]['FA']);self.assertEqual(a[key]['CR'],3)
    def check_fit(self,device):
        torch.set_num_threads(1)
        raw=np.random.RandomState(2).normal(size=(24,19));y=np.r_[np.zeros(18),np.ones(6)]
        parent=parent_fixture(raw)
        for arm in core.ARMS:
            _,m,records=core.fit(raw,y,parent,arm,.2,device,steps=3)
            self.assertEqual(m['l2'],0);self.assertEqual(m['initial_state_sha256'],parent['final_state_sha256'])
            self.assertEqual([r['step'] for r in records],[0,3])
            np.testing.assert_allclose(records[-1]['train_logits'],core.response.numpy_logits(m,raw),atol=1e-12)
            original=warm.SETTINGS['l2']
            try:
                warm.SETTINGS['l2']=0.
                independent_model(m,raw,y)
            finally: warm.SETTINGS['l2']=original
    def test_cpu_fit_and_numpy_gradient(self):self.check_fit('cpu')
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA checked on server')
    def test_gpu_fit_and_numpy_gradient(self):self.check_fit('cuda:0')
