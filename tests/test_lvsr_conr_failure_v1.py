"""Verify diagnostic accounting rather than testing training again."""
import unittest
import torch
from crane_project.tools import diagnose_lvsr_conr_failure_v1 as d


class FailureAuditChecks(unittest.TestCase):
    def test_pair_coverage_requires_positive_and_confused_negative(self):
        t=torch.tensor([[0.,0.],[.005,0.],[.1,0.]])
        p=torch.zeros_like(t)
        pos,neg,active=d.masks(t,p,.02)
        self.assertEqual(pos.sum().item(),2)
        self.assertEqual(neg.sum().item(),4)
        self.assertEqual(active.tolist(),[True,True,False])
        _,neg,active=d.masks(t,t,.02)
        self.assertFalse(active.any());self.assertEqual(neg.sum().item(),0)

    def test_gradient_norms_cosine_and_descent_loss_derivative(self):
        a=torch.tensor([3.,4.],dtype=torch.float64)
        b=torch.tensor([-6.,-8.],dtype=torch.float64)
        s=d.gradient_stats(a,b)
        self.assertEqual(s['aux_to_size_norm'],2.)
        self.assertEqual(s['cosine'],-1.)
        self.assertEqual(s['aux_unit_descent_size_loss_derivative'],5.)
        self.assertIsNone(d.gradient_stats(a*0,b)['cosine'])
        self.assertIsNone(d.gradient_stats(a*0,b)['aux_to_size_norm'])

    def test_error_transport_counts_direction_and_magnitude_separately(self):
        def row(i,width):
            return dict(image=str(i),scale=1.,domain='real',sequence='v',gt=[0,0,90,50,0],
                midpoint=[0,0,100,50,0,.9],b=[0,0,100,50,0,.9],edge_residual=[0,0,width,50,0,.9])
        old=[row(0,102),row(1,101)];new=[row(0,89),row(1,110)]
        result=d.error_transport(old,new)
        rr=[q for q in result if q['group']=='overall' and q['edge']=='long' and q['target_group']=='shrink']
        allv=next(q for q in rr if q['transition']=='all')
        self.assertEqual(allv['n'],2)
        self.assertAlmostEqual(allv['error_delta_sum'],(1+20-12-11)/90)
        self.assertEqual(next(q for q in rr if q['transition']=='wrong_to_toward')['n'],1)
        self.assertEqual(next(q for q in rr if q['transition']=='wrong_to_wrong')['n'],1)
        self.assertEqual(allv['overshoot_new'],1)

    def test_error_transport_rejects_changed_upstream(self):
        a=dict(image='a',scale=1.,domain='real',sequence='v',gt=[0,0,90,50,0],
               midpoint=[0,0,100,50,0,.9],b=[0,0,100,50,0,.9],edge_residual=[0,0,99,50,0,.9])
        b=dict(a,midpoint=[0,0,100,50,0,.8])
        with self.assertRaises(AssertionError):d.error_transport([a],[b])


if __name__=='__main__':unittest.main()
