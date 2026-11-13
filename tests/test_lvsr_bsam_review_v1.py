"""Density provenance, exact reversible perturbations and Adam compatibility."""
from copy import deepcopy
import unittest
import torch
from crane_project.utils import lvsr_bsam_review_v1 as r
from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead
from crane_project.tools.diagnose_lvsr_small_fit_v1 import step
from tests.test_lvsr_conr_v1 import inputs


class ReviewChecks(unittest.TestCase):
    def test_density_is_joint_not_independent_marginals_and_counts_self(self):
        t=torch.tensor([[0.,0.],[0.,0.],[.1,.1],[0.,.1]],dtype=torch.float64,requires_grad=True)
        c=r.target_density(t,window=.01)
        self.assertEqual(c.tolist(),[2,2,1,1]);self.assertFalse(c.requires_grad)
        w=r.density_weights(torch.tensor([1.]+[1000.]*99))
        self.assertAlmostEqual(float(w.mean()),1,places=12);self.assertLessEqual(float(w.max()),4.)
        self.assertGreater(float(w[0]),float(w[1]))

    def test_perturbation_restores_every_parameter_even_on_exception(self):
        p=torch.nn.Parameter(torch.tensor([1.,1e-9,1e6]));old=p.detach().clone()
        with self.assertRaisesRegex(RuntimeError,'forced'):
            with r.displaced([p],torch.tensor([.2,.2,.2])):raise RuntimeError('forced')
        self.assertTrue(torch.equal(p,old))
        with self.assertRaises(ValueError):
            with r.displaced([p],torch.tensor([float('nan')]*3)):pass
        self.assertTrue(torch.equal(p,old))

    def test_zero_radius_exact_original_adam_state(self):
        torch.manual_seed(1703);a=PairSpecificSizeHead().double();b=deepcopy(a)
        oa=torch.optim.Adam(a.parameters(),lr=.001);ob=torch.optim.Adam(b.parameters(),lr=.001)
        x=inputs()
        for _ in range(2):
            step(a,oa,x);r.review_update(b,ob,x,torch.ones(4),0.)
            for pa,pb in zip(a.parameters(),b.parameters()):
                self.assertTrue(torch.equal(pa,pb))
                for key in oa.state[pa]:self.assertTrue(torch.equal(oa.state[pa][key],ob.state[pb][key]))

    def test_nonzero_radius_uses_one_adam_step_and_preserves_upstream(self):
        torch.manual_seed(1703);h=PairSpecificSizeHead().double();x=inputs()
        for v in x:v.requires_grad_(True)
        initial=r.loss_edges(h,x)[1]
        self.assertTrue(torch.equal(initial['boxes_original'],x[-2]))
        opt=torch.optim.Adam(h.parameters(),lr=.001)
        r.review_update(h,opt,x,torch.tensor([1.,1.,2.,2.]),.001)
        self.assertEqual({int(opt.state[p]['step']) for p in h.parameters()},{1})
        self.assertTrue(all(v.grad is None for v in x))
        self.assertEqual(sum(p.numel() for p in h.parameters()),2480)
        out=r.loss_edges(h,[v.detach() for v in x])[1]
        self.assertTrue(torch.equal(out['boxes_original'][:,[0,1,4,5]],x[-2][:,[0,1,4,5]]))

    def test_raw_axes_restore_canonical_loss(self):
        v=torch.tensor([[1.,2.],[3.,4.]])
        bm=torch.tensor([[0.,0.,20.,10.,0.],[0.,0.,10.,20.,0.]])
        m=torch.tensor([[0.,0.,10.,20.,0.,.9],[0.,0.,10.,20.,0.,.9]])
        self.assertTrue(torch.equal(r.canonical_pair(v,bm,m),torch.tensor([[2.,1.],[3.,4.]])))


if __name__=='__main__':torch.set_num_threads(1);unittest.main()
