"""Mechanism tests only. Scalar toy Adam steps are not LVSR trial updates."""
from copy import deepcopy
import unittest
from crane_project.tools import run_symeood_110_lvsr_esh_short_v1 as e


class ESHTests(unittest.TestCase):
    def test_sixteen_edge_normalization_and_all_one_weights(self):
        import torch
        from crane_project.utils.port_size_qalign_v1 import size_loss
        m=torch.tensor([[0.,0.,12.,6.,0.,.8],[0.,0.,6.,12.,0.,.8]]*4)
        bm=torch.tensor([[0.,0.,12.,6.,0.]]*4+[[0.,0.,6.,12.,0.]]*4)
        gt=torch.tensor([[0.,0.,11.,5.,0.]]*8)
        delta=torch.arange(16,dtype=torch.float32).reshape(8,2).div(100).requires_grad_()
        out=dict(delta_roi=delta,vector_valid=torch.ones(8,dtype=torch.bool))
        points=e.edge_points(out,gt,m,bm,torch)
        G=e.edge_gradients(points,(delta,),torch)
        native=torch.autograd.grad(size_loss(out,gt,m,bm),delta)[0].reshape(-1)
        self.assertEqual(tuple(G.shape),(16,16))
        self.assertTrue(torch.allclose(G.sum(0),native,atol=2e-7,rtol=2e-5))
        self.assertTrue(torch.allclose(G.T@torch.ones(16),native,atol=2e-7,rtol=2e-5))
        self.assertIsNone(delta.grad)

    def test_adam_preview_state_purity_and_single_real_step(self):
        import torch
        p=torch.nn.Parameter(torch.tensor([.25,-.5,.75]));opt=torch.optim.Adam([p],lr=.0001)
        for i in range(3):
            g=torch.tensor([.1+i*.02,-.03,.07]);state=deepcopy(opt.state_dict());rng=torch.get_rng_state().clone()
            a=e.preview((p,),opt,g,torch)
            self.assertEqual(list(state['state']),list(opt.state_dict()['state']))
            for k,v in state['state'].items():
                for name,t in v.items():self.assertTrue(torch.equal(t,opt.state_dict()['state'][k][name]))
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            before=p.detach().clone();p.grad=g;opt.step()
            self.assertTrue(torch.allclose(p.detach()-before,a['delta'],atol=1e-7,rtol=1e-4))
            self.assertEqual(int(opt.state[p]['step']),i+1)
            self.assertTrue(torch.allclose(opt.state[p]['exp_avg_sq'],a['next_v']))

    def test_solver_bounds_identity_and_finite_output(self):
        import torch
        settings=dict(sum_grads=False,constraint='both',box_bound=2.,radius_bound=.2,
            allow_neg_weights=False,reg=0.,warm_start_iter=10,max_iter=50,tol=.001,
            mom_stats_Q=.98,max_raise_condition_Q=1.25,max_raise_dgdg=1.25,use_wandb=False)
        G=torch.zeros(16,3);z=torch.zeros(3);ema=[torch.tensor(-1.),torch.tensor(-1.)]
        w,_,_,_=e.solve(G,z,torch.ones(3),ema,settings,torch)
        self.assertTrue(torch.equal(w,torch.ones(16)))
        G=torch.tensor([[.1,.0,.0],[-.2,.0,.0],[.0,.1,.0],[.0,.0,.1]])
        w,logs,_,effective=e.solve(G,torch.tensor([.0001,-.0001,-.0001]),torch.ones(3),ema,settings,torch)
        self.assertTrue(bool(torch.isfinite(w).all()))
        self.assertTrue(bool(((w>=1)&(w<=3)).all()))
        self.assertLessEqual(float((w-1)@(effective['Q2']@(w-1))),float(effective['energy'])*(1+1e-5))
        with self.assertRaises(ValueError):e.solve(G*float('nan'),z,torch.ones(3),ema,settings,torch)


if __name__ == '__main__':unittest.main()
