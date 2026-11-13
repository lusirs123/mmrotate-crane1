"""Production operator is exactly the reviewed one; fixed protocol and ranking."""
from copy import deepcopy
import unittest
import torch
from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead
from crane_project.utils.lvsr_shared_spatial_review_v1 import SharedSpatialReviewHead
from crane_project.utils.port_size_pair_shared_spatial_v1 import PairSharedSpatialSizeHead
from crane_project.tools.diagnose_lvsr_small_fit_v1 import step
from tests.test_lvsr_conr_v1 import inputs


class SharedSpatialChecks(unittest.TestCase):
    def test_formal_matches_reviewed_tensors_rng_forward_gradients_and_two_updates(self):
        torch.manual_seed(1703);a=SharedSpatialReviewHead().double();rng=torch.get_rng_state()
        torch.manual_seed(1703);b=PairSharedSpatialSizeHead().double()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in b.parameters()),2480)
        self.assertEqual(list(a.state_dict()),list(b.state_dict()))
        x=inputs();self.assertTrue(torch.equal(b(*x[:-1])['boxes_original'],x[-2]))
        oa=torch.optim.Adam(a.parameters(),lr=.001);ob=torch.optim.Adam(b.parameters(),lr=.001)
        for _ in range(2):
            self.assertEqual(step(a,oa,x),step(b,ob,x))
            for pa,pb in zip(a.parameters(),b.parameters()):
                self.assertTrue(torch.equal(pa,pb));self.assertTrue(torch.equal(pa.grad,pb.grad))
                for key in oa.state[pa]:self.assertTrue(torch.equal(oa.state[pa][key],ob.state[pb][key]))

    def test_disabled_nonzero_original_output_and_inheritance(self):
        torch.manual_seed(1703);a=PairSpecificSizeHead().double();b=PairSharedSpatialSizeHead().double()
        with torch.no_grad():
            for layer in a.terminal.branches:layer.weight.fill_(.001);layer.bias.fill_(.002)
        b.load_state_dict(a.state_dict(),strict=True);b.set_spatial_mixing(False);x=inputs()
        aa,bb=a(*x[:-1]),b(*x[:-1])
        for key,value in aa.items():
            if torch.is_tensor(value):self.assertTrue(torch.equal(value,bb[key]),key)
        b.set_spatial_mixing(True);out=b(*x[:-1])
        self.assertTrue(torch.equal(out['common_support'],aa['common_support']))
        self.assertTrue(torch.equal(out['boxes_original'][:,[0,1,4,5]],x[-2][:,[0,1,4,5]]))

    def test_axis_order_and_empty_preserve_original_boundaries(self):
        h=PairSharedSpatialSizeHead().double();x=inputs()
        with torch.no_grad():
            for q in h.terminal.branches:q.bias.fill_(.002)
        for swap in (False,True):
            z=[v.clone() for v in x]
            if swap:
                for i in (2,3,5):z[i][:,2:4]=z[i][:,2:4].flip(-1)
            o=h(*z[:-1]);self.assertTrue(torch.isfinite(o['delta_roi']).all())
            self.assertTrue(torch.equal(o['boxes_original'][:,[0,1,4,5]],z[-2][:,[0,1,4,5]]))
        self.assertEqual(len(h(*[v[:0] for v in x[:-1]])['boxes_original']),0)

    def test_fixed_settings_and_all_epoch_ranking(self):
        from crane_project.tools import train_lvsr_shared_spatial_v1 as t
        c=t.check_config();self.assertEqual(c['spatial_mixing'],dict(local_dilation=1,distant_dilation=2,mix_weight=.5))
        self.assertEqual(c['lr'],.001);self.assertEqual(c['parameters'],2480)
        a=dict(epoch=1,eligibility_vs_L5=dict(pass_all=False),selection_key=[-675,1],checkpoint=dict(path='one'))
        b=dict(epoch=2,eligibility_vs_L5=dict(pass_all=True),selection_key=[-674,2],checkpoint=dict(path='two'))
        self.assertEqual(t.choose([a,b])['representative_epoch'],1);self.assertIsNone(t.choose([a,b])['selected_epoch'])


if __name__=='__main__':torch.set_num_threads(1);unittest.main()
