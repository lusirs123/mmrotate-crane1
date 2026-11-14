"""Direct-control readout, copied inputs, geometry and optimizer recovery."""
from copy import deepcopy
import math
from types import SimpleNamespace
import unittest
import torch
from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead
from crane_project.utils.port_size_pair_direct_control_v1 import PairDirectSizeControlHead
from crane_project.utils.port_size_structure_v1 import common_input
from crane_project.utils import port_size_qalign_v1 as model
from tests.test_lvsr_conr_v1 import inputs


class VDChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_copied_stems_capacity_and_zero_terminal(self):
        torch.manual_seed(1703)
        v=PairSpecificSizeHead();d=PairDirectSizeControlHead(v)
        self.assertEqual(sum(p.numel() for p in d.parameters()),2450)
        self.assertEqual(sum(p.numel() for p in v.parameters()),2480)
        for key,value in v.stem.state_dict().items():
            self.assertTrue(torch.equal(value,d.stem.state_dict()[key]),key)
            self.assertNotEqual(value.data_ptr(),d.stem.state_dict()[key].data_ptr())
        self.assertTrue(all(not bool(q.abs().any()) for q in d.terminal.parameters()))

    def test_same267_channel_input_and_support_as_V(self):
        x=inputs();v=PairSpecificSizeHead().double();d=PairDirectSizeControlHead(v).double()
        captured=[]
        hook=v.stem.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach()))
        v(*x[:-1]);hook.remove()
        dense,w=common_input(*x[:-1])
        torch.testing.assert_close(captured[0],dense,atol=0,rtol=0)
        o=d(*x[:-1]);torch.testing.assert_close(w,o['common_support'],atol=0,rtol=0)

    def test_neutral_empty_and_exact_M_inheritance(self):
        x=inputs();d=PairDirectSizeControlHead().double()
        self.assertTrue(torch.equal(d(*x[:-1])['boxes_original'],x[-2]))
        empty=d(*[v[:0] for v in x[:-1]])
        self.assertEqual(tuple(empty['boxes_original'].shape),(0,6))
        with torch.no_grad():
            d.terminal.branches[0].bias.fill_(math.log(1.02))
            d.terminal.branches[1].bias.fill_(math.log(.99))
        o=d(*x[:-1]);self.assertTrue(bool(o['accepted'].all()))
        self.assertTrue(torch.equal(o['boxes_original'][:,[0,1,4,5]],x[-2][:,[0,1,4,5]]))
        torch.testing.assert_close(o['boxes_original'][:,2:4],x[-2][:,2:4]*x[2].new_tensor([1.02,.99]))

    def test_canonical_axis_mapping_under_raw_swaps(self):
        for swap in (False,True):
            x=inputs()
            if swap:
                for i in (2,3,5):x[i][:,2:4]=x[i][:,2:4].flip(-1)
            d=PairDirectSizeControlHead().double()
            with torch.no_grad():
                d.terminal.branches[0].bias.fill_(math.log(1.02))
                d.terminal.branches[1].bias.fill_(math.log(.99))
            o=d(*x[:-1]);actual=o['boxes_original'][:,2:4].sort(descending=True)[0]
            expected=x[-2][:,2:4].sort(descending=True)[0]*x[2].new_tensor([1.02,.99])
            torch.testing.assert_close(actual,expected)

    def test_weighted_dense_log_readout_not_pooled_input(self):
        x=inputs();x[1][:,:,:,0:3]=0;d=PairDirectSizeControlHead().double()
        with torch.no_grad():
            for terminal in d.terminal.branches:terminal.weight.fill_(.01)
        dense,w=common_input(*x[:-1]);local=d.terminal(d.stem(dense)).flatten(2).transpose(1,2)
        expected=(local*w.reshape(4,81,1)).sum(1)/w.reshape(4,81).sum(1)[:,None]
        o=d(*x[:-1]);torch.testing.assert_close(expected,o['delta_roi'])

    def test_anisotropic_rounding_model_axis_exchange(self):
        x=inputs()
        x[2][:,2:4]=x[2].new_tensor([61.,60.]);x[5]=x[2].clone()
        x[4][:]=x[4].new_tensor([.49,.51])
        x[3]=x[2][:,:5].clone();x[3][:,:4]*=x[4][:,[0,1,0,1]]
        self.assertTrue(bool((x[3][:,2]<x[3][:,3]).all()))
        d=PairDirectSizeControlHead().double()
        self.assertTrue(torch.equal(d(*x[:-1])['boxes_original'],x[-2]))
        with torch.no_grad():
            d.terminal.branches[0].bias.fill_(math.log(1.02))
            d.terminal.branches[1].bias.fill_(math.log(.99))
        o=d(*x[:-1])
        torch.testing.assert_close(o['delta_roi'],x[2].new_tensor([[math.log(.99),math.log(1.02)]]).expand(4,-1))
        self.assertTrue(torch.equal(o['boxes_original'][:,[0,1,4,5]],x[-2][:,[0,1,4,5]]))

    def test_original_supervision_and_all_active_parameter_gradients(self):
        x=inputs();d=PairDirectSizeControlHead().double()
        opt=torch.optim.Adam(d.parameters(),lr=.001)
        losses=[]
        for step in range(2):
            opt.zero_grad();o=d(*x[:-1]);loss=model.size_loss(o,x[-1],x[-2],x[3])
            target=model.residual.size_targets(x[-1],x[-2],x[3])['log_residual']
            torch.testing.assert_close(loss,torch.nn.functional.smooth_l1_loss(o['delta_roi'],target,beta=.1))
            loss.backward();self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in d.parameters()))
            if step==0:self.assertTrue(all(not bool(p.grad.abs().any()) for p in d.stem.parameters()))
            else:self.assertGreater(sum(float(p.grad.abs().sum()) for p in d.stem.parameters()),0)
            opt.step();losses.append(float(loss.detach()))

    def test_whole_box_fallback_for_large_or_nonfinite_output(self):
        for bias in (math.log(2.),float('nan')):
            x=inputs();d=PairDirectSizeControlHead().double()
            with torch.no_grad():d.terminal.branches[0].bias.fill_(bias)
            o=d(*x[:-1]);self.assertTrue(torch.equal(o['boxes_original'],x[-2]))
            self.assertFalse(bool(o['accepted'].any()))

    def test_adam_restore_next_step(self):
        x=inputs();a=PairDirectSizeControlHead().double();oa=torch.optim.Adam(a.parameters(),lr=.001)
        def update(h,o):
            o.zero_grad();out=h(*x[:-1]);loss=model.size_loss(out,x[-1],x[-2],x[3]);loss.backward();o.step()
        update(a,oa);state=deepcopy(a.state_dict());optimizer=deepcopy(oa.state_dict())
        b=PairDirectSizeControlHead().double();b.load_state_dict(state);ob=torch.optim.Adam(b.parameters(),lr=.001);ob.load_state_dict(optimizer)
        update(a,oa);update(b,ob)
        for k,v in a.state_dict().items():self.assertTrue(torch.equal(v,b.state_dict()[k]),k)

    def test_fixed_recipe_and_gate_free_representative(self):
        from crane_project.tools import train_lvsr_vd_control_v1 as t
        from crane_project.utils import symeood_r4_lvsr_protocol_v1 as p
        cfg=t.checked_config();self.assertEqual(cfg['updates'],24*639)
        def metric(j):
            return {d:dict(edge_residual=dict(joint10=j,long_MAE=.03,short_MAE=.04)) for d in ('overall','real','sim')}
        a=dict(epoch=1,metrics=metric(680),eligibility=False);b=dict(epoch=2,metrics=metric(679),eligibility=True)
        self.assertIs(t.pick([a,b],p),a)


if __name__=='__main__':unittest.main()
