"""Meaningful derivative, Adam, sampling and video-leakage checks."""
from copy import deepcopy
import math
import unittest
import torch
from crane_project.tools import diagnose_lvsr_direction_representation_v1 as d


class DirectionRepresentationChecks(unittest.TestCase):
    def test_canonical_order_handles_both_independent_axis_swaps(self):
        value=torch.tensor([[1.,2.]]*4)
        bm=torch.tensor([[0,0,10,5,0,1],[0,0,5,10,0,1],[0,0,10,5,0,1],[0,0,5,10,0,1.]])
        m=torch.tensor([[0,0,10,5,0,1],[0,0,10,5,0,1],[0,0,5,10,0,1],[0,0,5,10,0,1.]])
        self.assertTrue(torch.equal(d.canonical(value,bm,m),torch.tensor([[1.,2.],[2.,1.],[2.,1.],[1.,2.]])))

    def test_log_direction_thresholds_preserve_neutral_interval(self):
        t=torch.tensor([math.log(.98),0.,math.log(1.02)])
        self.assertEqual(d.direction(t).tolist(),[0,1,2])
        x=torch.tensor([[.02,-.04]],requires_grad=True)
        target=torch.tensor([[-.01,.06]])
        loss=torch.nn.functional.smooth_l1_loss(x,target,beta=.1)
        grad=torch.autograd.grad(loss,x)[0]
        torch.testing.assert_close(grad,((x-target)/.1).clamp(-1,1)/2)

    def test_functional_adam_matches_real_adam_without_mutating_state(self):
        p=torch.nn.Parameter(torch.tensor([.2,-.4,.7],dtype=torch.float64))
        opt=torch.optim.Adam([p],lr=.001,betas=(.9,.999),eps=1e-8)
        for i in range(3):
            p.grad=torch.tensor([.5,-.8,1.2],dtype=p.dtype)*(i+1);opt.step()
        snapshot=deepcopy(opt.state_dict());before=p.detach().clone()
        gradient=torch.tensor([20.,-7.,.3],dtype=p.dtype)
        expected,info=d.adam_direction([p],gradient,snapshot)
        p.grad=gradient.clone();torch.nn.utils.clip_grad_norm_([p],10.);opt.step()
        torch.testing.assert_close(p.detach()-before,expected,atol=1e-12,rtol=1e-10)
        self.assertLess(info['clip_coefficient'],1.)
        self.assertEqual(int(snapshot['state'][0]['step']),3)

    def test_parameter_jacobian_loss_gradient_reconstruction(self):
        p=torch.nn.Parameter(torch.tensor([[.3,-.2],[.7,.1]],dtype=torch.float64))
        x=torch.tensor([.4,-.8],dtype=p.dtype);target=torch.tensor([-.02,.04],dtype=p.dtype)
        prediction=p@x
        j=torch.stack([torch.autograd.grad(prediction[i],p,retain_graph=True)[0].flatten() for i in range(2)])
        derivative=((prediction-target)/.1).clamp(-1,1)
        reconstructed=(j*derivative[:,None]).sum(0)/2
        direct=torch.autograd.grad(torch.nn.functional.smooth_l1_loss(prediction,target,beta=.1),p)[0].flatten()
        torch.testing.assert_close(reconstructed,direct)
        self.assertIsNone(p.grad)

    def test_knn_holds_out_entire_video_and_domain_missing_is_explicit(self):
        meta=[];vectors=[]
        for video,domain,sign in [('A','real',-.1),('B','real',.1),('C','sim',-.1)]:
            for i in range(6):
                meta.append(dict(image=video+str(i),sequence=video,frame_id=i*40,domain=domain,split='train',scale=1.,target=[sign,sign]))
                # Same-video is closest, thus cross-video cannot be perfect on A/B.
                vectors.append([1.,0.] if video=='A' else [0.,1.] if video=='B' else [.99,.01])
        result=d.knn(torch.tensor(vectors),meta)
        pool={r['group']:r for r in result['metrics'] if r['split']=='train' and r['scale']==1. and r['pool']=='same_domain_cross_video'}
        self.assertEqual(pool['sim']['n'],0)
        self.assertEqual(pool['real']['edges']['short']['accuracy'],0.)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
