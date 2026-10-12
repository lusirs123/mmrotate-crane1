import unittest
from crane_project.utils import port_reliability_oof_gradient_chain_v1 as chain

try:
    import torch
except ImportError:
    torch = None


class BoundaryTests(unittest.TestCase):
    def test_only_exact_previous_and_anomaly_steps(self):
        completion=dict(completed_updates=6451,status='ANOMALY_QUARANTINED_BEFORE_OPTIMIZER')
        chain.validate_boundary(dict(iteration=6450,diagnostic_only=True),completion,'previous')
        chain.validate_boundary(dict(iteration=6451,diagnostic_only=True),completion,'anomaly')
        for name,iteration,diagnostic in [('previous',6451,True),('anomaly',6450,True),('other',6451,True),('anomaly',6451,False)]:
            with self.assertRaises(ValueError):
                chain.validate_boundary(dict(iteration=iteration,diagnostic_only=diagnostic),completion,name)


@unittest.skipIf(torch is None, 'Torch required')
class GaussianTests(unittest.TestCase):
    def test_reference_matches_original_value_and_derivative(self):
        from mmrotate.models.losses.sym_kld_calculator import sym_kld
        torch.manual_seed(17)
        p=torch.tensor([[20.,30.,60.,12.,.7],[21.,40.,20.,5.,-.9]],dtype=torch.float64,requires_grad=True)
        q=torch.tensor([[22.,32.,55.,15.,.9],[20.,39.,21.,8.,-.7]],dtype=torch.float64)
        original=sym_kld(p,q); reference=chain.analytic_kld_terms(p,q)['raw']
        torch.testing.assert_close(original,reference,atol=1e-10,rtol=1e-10)
        a=torch.autograd.grad(original.sum(),p,retain_graph=True)[0]
        b=torch.autograd.grad(reference.sum(),p)[0]
        torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)

    def test_long_short_swap_and_pi_are_equivalent(self):
        import math
        p=torch.tensor([[20.,30.,60.,12.,.7]],dtype=torch.float64)
        q=p.clone();q[:,2:4]=p[:,2:4].flip(1);q[:,4]+=math.pi/2
        self.assertLess(abs(float(chain.analytic_kld_terms(p,q)['raw'])),1e-12)
        q=p.clone();q[:,4]+=math.pi
        self.assertLess(abs(float(chain.analytic_kld_terms(p,q)['raw'])),1e-12)
        self.assertLess(chain.canonical_error(p,q)['canonical_axis_error_deg'][0],1e-10)

    def test_real_reduction_preserves_weight_and_avg_factor(self):
        p=torch.tensor([[20.,30.,60.,12.,.7],[21.,40.,20.,5.,-.9]],dtype=torch.float64)
        q=p.clone();q[:,0]+=1
        w=torch.tensor([1.,0.],dtype=torch.float64)
        expected=((1+chain.analytic_kld_terms(p,q)['raw']).sqrt()-1)[0]/2*2
        torch.testing.assert_close(chain.reference_loss(p,q,w,2,2),expected)

    def test_budget_bounds_regression_and_preserves_other_gradient(self):
        p=torch.tensor([1.,2.],requires_grad=True)
        regression=100*p[0]+200*p[1]
        other=3*p[0]-4*p[1]
        before=float((regression+other).detach())
        stats=chain.regression_budget_backward(regression+other,regression,[p])
        scale=10/(100**2+200**2)**.5
        torch.testing.assert_close(p.grad,torch.tensor([3.+100*scale,-4.+200*scale]))
        self.assertAlmostEqual(stats['regression_gradient_budgeted_norm'],10)
        self.assertEqual(float((regression+other).detach()),before)

    def test_small_regression_is_exactly_unchanged(self):
        p=torch.tensor([1.,2.],requires_grad=True);regression=p.square().sum();total=regression+3*p.sum()
        stats=chain.regression_budget_backward(total,regression,[p])
        torch.testing.assert_close(p.grad,torch.tensor([5.,7.]))
        self.assertEqual(stats['regression_gradient_scale'],1.)

    def test_nonfinite_regression_stops_before_backward(self):
        p=torch.tensor(1.,requires_grad=True);regression=p*float('inf')
        with self.assertRaises(FloatingPointError):chain.regression_budget_backward(regression,regression,[p])
        self.assertIsNone(p.grad)

    def test_actual_hook_rejects_nonfinite_and_preserves_normal_update(self):
        from types import SimpleNamespace
        from mmrotate.core.hooks.port_oof_regression_budget_hook import PortOOFRegressionBudgetHook
        class Tiny(torch.nn.Module):
            def __init__(self):super().__init__();self.p=torch.nn.Parameter(torch.tensor(1.))
            def forward_train(self,bad=False):return dict(loss_bbox=self.p*(float('inf') if bad else 2),loss_cls=self.p*3)
        class Log:
            def __init__(self):self.v={}
            def update(self,v,n):self.v.update(v)
        m=Tiny();optimizer=torch.optim.SGD(m.parameters(),lr=.1)
        runner=SimpleNamespace(model=m,optimizer=optimizer,log_buffer=Log())
        hook=PortOOFRegressionBudgetHook(grad_clip=dict(max_norm=10,norm_type=2));hook.before_run(runner)
        hook.before_train_iter(runner);terms=m.forward_train();runner.outputs=dict(loss=sum(terms.values()),num_samples=1)
        hook.after_train_iter(runner);self.assertAlmostEqual(float(m.p),.5,places=6)
        hook.before_train_iter(runner);terms=m.forward_train(True);runner.outputs=dict(loss=sum(terms.values()),num_samples=1)
        with self.assertRaises(FloatingPointError):hook.after_train_iter(runner)
        self.assertAlmostEqual(float(m.p),.5,places=6)


if __name__ == '__main__': unittest.main()
