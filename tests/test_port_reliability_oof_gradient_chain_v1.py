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


if __name__ == '__main__': unittest.main()
