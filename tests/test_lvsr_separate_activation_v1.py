"""Meaningful single-factor, gradient, initialization and selection checks."""
from copy import deepcopy
import unittest
import torch

from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead
from crane_project.utils.port_size_pair_separate_activation_v1 import PairSeparateActivationSizeHead


class SeparateActivationChecks(unittest.TestCase):
    def test_identical_initial_parameters_rng_and_inherited_forward(self):
        torch.manual_seed(1703); original = PairSpecificSizeHead(); rng = torch.get_rng_state()
        torch.manual_seed(1703); candidate = PairSeparateActivationSizeHead()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(list(original.state_dict()), list(candidate.state_dict()))
        self.assertEqual(sum(p.numel() for p in candidate.parameters()), 2480)
        for k, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, candidate.state_dict()[k]), k)
        self.assertIs(PairSpecificSizeHead.forward, PairSeparateActivationSizeHead.forward)
        for layer in candidate.terminal.branches:
            self.assertTrue(torch.equal(layer.weight, torch.zeros_like(layer.weight)))
            self.assertTrue(torch.equal(layer.bias, torch.zeros_like(layer.bias)))

    def test_joint_control_matches_outputs_and_parameter_gradients(self):
        torch.manual_seed(1703); original = PairSpecificSizeHead().double()
        torch.manual_seed(1703); candidate = PairSeparateActivationSizeHead().double()
        for branch in candidate.stem.branches:branch[0].joint_control = True
        x = torch.randn(2,267,9,9,dtype=torch.float64)
        a, b = original.stem(x), candidate.stem(x)
        self.assertTrue(torch.allclose(a,b,atol=1e-12,rtol=1e-12))
        a.square().sum().backward();b.square().sum().backward()
        for (ka,pa),(kb,pb) in zip(original.stem.named_parameters(),candidate.stem.named_parameters()):
            self.assertEqual(ka,kb)
            self.assertTrue(torch.allclose(pa.grad,pb.grad,atol=1e-10,rtol=1e-10),ka)

    def test_separate_activation_is_not_linear_relabel_and_geometry_can_learn(self):
        h = PairSeparateActivationSizeHead().double(); layer = h.stem.branches[0][0]
        with torch.no_grad():
            layer.weight.zero_();layer.bias.zero_()
            layer.weight[:,0] = -2.;layer.weight[:,256] = 1.
        x = torch.zeros(1,267,1,1,dtype=torch.float64);x[:,0] = 1.;x[:,256] = 1.
        out = layer(x)
        self.assertTrue(torch.equal(out,torch.ones_like(out)))
        out.sum().backward()
        self.assertTrue((layer.weight.grad[:,256] > 0).all())
        layer.joint_control = True
        self.assertTrue(torch.equal(layer(x),torch.zeros_like(out)))

    def test_gate_strict_joint_domain_protection_and_original_chain_separation(self):
        from crane_project.tools import train_lvsr_separate_activation_v1 as t
        from crane_project.utils import symeood_r4_lvsr_protocol_v1 as p
        base=dict(long_MAE=.06,short_MAE=.05,joint10=674,AP75=78.,mean_RIoU=.86,TP50=879)
        ref={d:dict(frozen_identity=True,midpoint=deepcopy(base),b=dict(base,AP75=88.),edge_residual=deepcopy(base)) for d in ('overall','real','sim')}
        candidate=deepcopy(ref)
        self.assertFalse(t.assess(candidate,ref)['pass_all'])
        for d in candidate:candidate[d]['edge_residual'].update(long_MAE=.059,short_MAE=.049,joint10=675)
        self.assertTrue(t.assess(candidate,ref)['pass_all'])
        self.assertTrue(t.increment_gate(candidate)['pass_all'])
        self.assertFalse(p.assess(candidate)['pass_all'])
        for field,bad in (('long_MAE',.061),('short_MAE',.051),('joint10',673),('AP75',77.9),('mean_RIoU',.85),('TP50',878)):
            x=deepcopy(candidate);x['real']['edge_residual'][field]=bad
            self.assertFalse(t.assess(x,ref)['pass_all'],field)
        candidate['sim']['frozen_identity']=False
        self.assertFalse(t.assess(candidate,ref)['pass_all'])

    def test_selection_null_and_fixed_training_settings(self):
        from crane_project.tools import train_lvsr_separate_activation_v1 as t
        cfg=t.check_config();self.assertEqual(cfg['lr'],.001);self.assertEqual(cfg['epochs'],24)
        a=dict(epoch=1,eligibility_vs_L5=dict(pass_all=False),selection_key=[-675,1],checkpoint=dict(path='one'))
        b=dict(epoch=2,eligibility_vs_L5=dict(pass_all=True),selection_key=[-674,2],checkpoint=dict(path='two'))
        self.assertEqual(t.choose([a,b])['selected_epoch'],2)
        v=t.choose([a]);self.assertIsNone(v['selected_epoch']);self.assertEqual(v['representative_epoch'],1)
        self.assertFalse(v['automatic_adoption']);self.assertFalse(v['selection_on_test'])


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
