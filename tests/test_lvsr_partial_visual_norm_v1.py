"""Weighted moments, retained amplitude/geometry, exact control and gradients."""
import unittest
import torch
from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead
from crane_project.utils.port_size_pair_partial_visual_norm_v1 import (
    PairPartialVisualNormSizeHead,normalize_supported,VARIANCE_EPSILON,
)


class PartialVisualNormChecks(unittest.TestCase):
    def test_weighted_moments_per_channel_and_invalid_position(self):
        v=torch.tensor([[[[2.,4.,999.]],[[6.,12.,-999.]]]],dtype=torch.float64)
        w=torch.tensor([[[[1.,.5,0.]]]],dtype=v.dtype)
        out=normalize_supported(v,w)
        mu=(v[...,:2]*w[...,:2]).sum((2,3),keepdim=True)/1.5
        variance=((v[...,:2]-mu).square()*w[...,:2]).sum((2,3),keepdim=True)/1.5
        torch.testing.assert_close(out[...,:2],(v[...,:2]-mu)/(variance+VARIANCE_EPSILON).sqrt())
        self.assertEqual(out[...,2].abs().sum().item(),0.)
        torch.testing.assert_close((out*w).sum((2,3)),torch.zeros(1,2,dtype=v.dtype),atol=1e-12,rtol=0.)

    def test_empty_constant_and_low_variance_support_finite_gradients(self):
        for support in (0.,1e-12,1.):
            v=torch.full((2,2,9,9),3.,dtype=torch.float64,requires_grad=True)
            w=torch.full((2,1,9,9),support,dtype=v.dtype,requires_grad=True)
            out=normalize_supported(v,w)
            self.assertTrue(torch.isfinite(out).all());out.square().sum().backward()
            self.assertTrue(torch.isfinite(v.grad).all());self.assertIsNone(w.grad)
            if support==0.:self.assertEqual(out.abs().sum().item(),0.)

    def test_projection_normalizes_only_first_two_visual_channels(self):
        h=PairPartialVisualNormSizeHead().double();layer=h.stem.branches[0][0]
        with torch.no_grad():
            layer.weight.zero_();layer.bias.copy_(torch.tensor([.1,.2,.3,.4],dtype=torch.float64))
            for k in range(4):layer.weight[k,k,0,0]=1.;layer.weight[k,257,0,0]=k+1
        x=torch.zeros(1,267,1,3,dtype=torch.float64)
        x[:,:4]=torch.tensor([[[[1.,2.,3.]],[[4.,5.,6.]],[[7.,8.,9.]],[[10.,11.,12.]]]])
        x[:,256]=1.;x[:,257]=2.
        y=layer(x);visual=x[:,:4]
        expected=torch.cat((normalize_supported(visual[:,:2],x[:,256:257]),visual[:,2:]),1)
        expected=expected+torch.arange(1,5,dtype=x.dtype)[None,:,None,None]*2+layer.bias[None,:,None,None]
        torch.testing.assert_close(y,expected)
        moved=x.clone();moved[:,257]+=1.
        torch.testing.assert_close(layer(moved)-y,torch.arange(1,5,dtype=x.dtype)[None,:,None,None].expand_as(y))
        shifted=x.clone();shifted[:,:4]+=4.
        diff=layer(shifted)-y
        torch.testing.assert_close(diff[:,:2],torch.zeros_like(diff[:,:2]),atol=1e-12,rtol=0.)
        torch.testing.assert_close(diff[:,2:],torch.full_like(diff[:,2:],4.))

    def test_moments_backpropagate_and_first_bias_receives_gradients(self):
        torch.manual_seed(7);h=PairPartialVisualNormSizeHead().double()
        x=torch.randn(2,267,9,9,dtype=torch.float64);x[:,256]=1.
        out=h.stem.branches[0][0](x)
        out.square().mean().backward()
        layer=h.stem.branches[0][0]
        self.assertTrue(torch.isfinite(layer.weight.grad).all())
        self.assertTrue((layer.weight.grad[:,:256].abs().flatten(1).sum(1)>0).all())
        self.assertTrue((layer.bias.grad.abs()>0).all())
        v=torch.randn(1,2,1,3,dtype=torch.float64,requires_grad=True)
        torch.autograd.gradcheck(lambda q:normalize_supported(q,torch.ones(1,1,1,3,dtype=q.dtype)),(v,))

    def test_parameter_rng_keys_and_terminal_initialization_identical(self):
        torch.manual_seed(1703);a=PairSpecificSizeHead();rng=torch.get_rng_state()
        torch.manual_seed(1703);b=PairPartialVisualNormSizeHead()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in b.parameters()),2480)
        self.assertEqual(list(a.state_dict()),list(b.state_dict()))
        for k,v in a.state_dict().items():self.assertTrue(torch.equal(v,b.state_dict()[k]),k)
        for q in b.terminal.branches:
            self.assertEqual(q.weight.abs().sum().item(),0.);self.assertEqual(q.bias.abs().sum().item(),0.)

    def test_disabled_full_stem_matches_original_output_and_gradients(self):
        torch.manual_seed(1703);a=PairSpecificSizeHead().double()
        torch.manual_seed(1703);b=PairPartialVisualNormSizeHead().double();b.set_normalization(False)
        x=torch.randn(2,267,9,9,dtype=torch.float64)
        u,v=a.stem(x),b.stem(x);self.assertTrue(torch.equal(u,v))
        u.square().mean().backward();v.square().mean().backward()
        for (ka,pa),(kb,pb) in zip(a.stem.named_parameters(),b.stem.named_parameters()):
            self.assertEqual(ka,kb);self.assertTrue(torch.equal(pa.grad,pb.grad),ka)

    def test_full_forward_zero_and_nonzero_inheritance_axis_and_empty(self):
        from tests.test_lvsr_conr_v1 import inputs
        x=inputs();h=PairPartialVisualNormSizeHead().double()
        o=h(*x[:-1]);self.assertTrue(torch.equal(o['boxes_original'],x[-2]))
        with torch.no_grad():
            for q in h.terminal.branches:q.bias.fill_(.002)
        for swap in (False,True):
            z=[v.clone() for v in x]
            if swap:
                for k in (2,3,5):z[k][:,2:4]=z[k][:,2:4].flip(-1)
            o=h(*z[:-1]);self.assertTrue(torch.isfinite(o['delta_roi']).all())
            self.assertTrue(torch.equal(o['boxes_original'][:,[0,1,4,5]],z[-2][:,[0,1,4,5]]))
        empty=h(*[v[:0] for v in x[:-1]]);self.assertEqual(len(empty['boxes_original']),0)

    def test_config_and_all_epoch_ranking(self):
        from crane_project.tools import train_lvsr_partial_visual_norm_v1 as t
        cfg=t.check_config();self.assertEqual(cfg['partial_normalization']['channels'],2)
        self.assertEqual(cfg['loss'],'unchanged original SmoothL1 beta0.1 log-size residual; no ConR')
        a=dict(epoch=1,eligibility_vs_L5=dict(pass_all=False),selection_key=[-675,1],checkpoint=dict(path='one'))
        b=dict(epoch=2,eligibility_vs_L5=dict(pass_all=True),selection_key=[-674,2],checkpoint=dict(path='two'))
        self.assertEqual(t.choose([a,b])['representative_epoch'],1);self.assertIsNone(t.choose([a,b])['selected_epoch'])

if __name__=='__main__':torch.set_num_threads(1);unittest.main()
