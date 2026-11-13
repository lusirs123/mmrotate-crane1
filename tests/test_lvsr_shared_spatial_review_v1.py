"""Shared parameter semantics, zero reference and the unsupported-neighbor risk."""
from copy import deepcopy
import io
import unittest
import torch
from torch.nn import functional as F
from crane_project.utils.lvsr_shared_spatial_review_v1 import SharedSpatialProjection,SharedSpatialReviewHead,tap_geometry
from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead
from crane_project.tools.diagnose_lvsr_small_fit_v1 import step
from tests.test_lvsr_conr_v1 import inputs


class SpatialChecks(unittest.TestCase):
    def test_initial_tensor_objects_keys_rng_and_neutral_inherited(self):
        torch.manual_seed(1703);a=PairSpecificSizeHead();rng=torch.get_rng_state()
        torch.manual_seed(1703);b=SharedSpatialReviewHead()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in b.parameters()),2480)
        self.assertEqual(list(a.state_dict()),list(b.state_dict()))
        for k,v in a.state_dict().items():self.assertTrue(torch.equal(v,b.state_dict()[k]),k)
        original=torch.nn.Conv2d(4,4,3,padding=1);wrapped=SharedSpatialProjection(original)
        self.assertIs(original.weight,wrapped.weight);self.assertIs(original.bias,wrapped.bias)
        x=inputs();b=b.double();o=b(*x[:-1])
        self.assertTrue(torch.equal(o['boxes_original'],x[-2]))
        self.assertEqual(len(b(*[v[:0] for v in x[:-1]])['boxes_original']),0)

    def test_formula_bias_once_and_shared_gradient(self):
        original=torch.nn.Conv2d(4,4,3,padding=1).double();a=SharedSpatialProjection(original)
        x=torch.randn(2,4,9,9,dtype=torch.float64,requires_grad=True)
        y=a(x);expected=.5*F.conv2d(x,a.weight,None,padding=1)+.5*F.conv2d(x,a.weight,None,padding=2,dilation=2)+a.bias[None,:,None,None]
        torch.testing.assert_close(y,expected,atol=1e-14,rtol=1e-14)
        gy=torch.autograd.grad(y.square().sum(),(x,a.weight,a.bias),retain_graph=True)
        ge=torch.autograd.grad(expected.square().sum(),(x,a.weight,a.bias))
        for u,v in zip(gy,ge):torch.testing.assert_close(u,v,atol=1e-11,rtol=1e-12)
        with torch.no_grad():a.weight.zero_();a.bias.fill_(2.)
        self.assertTrue(torch.equal(a(x),torch.full_like(y,2.)))

    def test_disabled_exact_old_forward_gradients_and_adam(self):
        torch.manual_seed(1703);a=PairSpecificSizeHead().double()
        torch.manual_seed(1703);b=SharedSpatialReviewHead().double();b.set_spatial_review(False)
        oa=torch.optim.Adam(a.parameters(),lr=.001);ob=torch.optim.Adam(b.parameters(),lr=.001)
        x=inputs()
        for _ in range(2):
            step(a,oa,x);step(b,ob,x)
            for pa,pb in zip(a.parameters(),b.parameters()):
                self.assertTrue(torch.equal(pa,pb));self.assertTrue(torch.equal(pa.grad,pb.grad))
                for k in oa.state[pa]:self.assertTrue(torch.equal(oa.state[pa][k],ob.state[pb][k]))

    def test_far_neighbor_and_no_implicit_mask_change(self):
        a=SharedSpatialProjection(torch.nn.Conv2d(4,4,3,padding=1)).double()
        with torch.no_grad():a.weight.zero_();a.bias.zero_();a.weight[0,0,1,2]=1.
        x=torch.zeros(1,4,9,9,dtype=torch.float64);x[0,0,4,6]=2.
        a.enabled=False;self.assertEqual(float(a(x)[0,0,4,4]),0.)
        a.enabled=True;self.assertEqual(float(a(x)[0,0,4,4]),1.)
        # The operator receives hidden features, not W; no new masking is added.
        w=torch.zeros(1,1,9,9,dtype=torch.float64);w[0,0,4,4]=1.
        s=tap_geometry(w)
        self.assertEqual(float(s['new_far_valid_taps_mean']),0.)
        self.assertEqual(float(s['d2']['inside_zero_support_fraction']),8/9)

    def test_tap_accounting_full_and_single_pixel_support(self):
        w=torch.ones(1,1,9,9,dtype=torch.float64);s=tap_geometry(w)
        for key in ('d1','d2'):
            self.assertAlmostEqual(sum(float(x) for x in s[key].values()),1.,places=12)
        self.assertAlmostEqual(float(s['d1']['outside_ROI_fraction']),104/729,places=12)
        self.assertAlmostEqual(float(s['d2']['outside_ROI_fraction']),200/729,places=12)
        self.assertEqual(float(s['supported_outputs_with_any_new_valid_neighbor_fraction']),1.)
        empty=tap_geometry(torch.zeros_like(w))
        self.assertTrue(all(torch.isfinite(x).all() for d in empty.values() for x in (d.values() if isinstance(d,dict) else [d])))

    def test_actual_disposable_updates_reload_and_upstream_isolation(self):
        torch.manual_seed(1703);h=SharedSpatialReviewHead().double();x=inputs()
        for v in x:v.requires_grad_(True)
        opt=torch.optim.Adam(h.parameters(),lr=.001)
        step(h,opt,x);self.assertTrue(all(p.grad.abs().sum()==0 for p in h.stem.parameters()))
        step(h,opt,x);self.assertTrue(any(p.grad.abs().sum()>0 for p in h.stem.parameters()))
        self.assertTrue(all(v.grad is None for v in x))
        o=h(*x[:-1]);self.assertTrue(torch.equal(o['boxes_original'][:,[0,1,4,5]],x[-2][:,[0,1,4,5]]))
        b=io.BytesIO();torch.save(h.state_dict(),b);b.seek(0)
        clone=SharedSpatialReviewHead().double();clone.load_state_dict(torch.load(b,weights_only=True));torch.testing.assert_close(clone(*x[:-1])['delta_roi'],o['delta_roi'],rtol=0,atol=0)


if __name__=='__main__':torch.set_num_threads(1);unittest.main()
