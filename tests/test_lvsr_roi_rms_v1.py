"""RMS statistic axes, support, geometry, initialization and ranking checks."""
from copy import deepcopy
import unittest
import torch
from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead
from crane_project.utils.port_size_pair_roi_rms_v1 import (
    PairROIRMSSizeHead, RMS_EPSILON, normalize_visual_input,
)


class ROIRMSChecks(unittest.TestCase):
    def test_weighted_scalar_preserves_geometry_and_relative_visual_pattern(self):
        x = torch.zeros(2, 267, 1, 3, dtype=torch.float64)
        x[0, :256, 0, :2] = torch.tensor([2., 4.])
        x[1, :256, 0, :2] = torch.tensor([6., 8.])
        x[:, 256, 0] = torch.tensor([1., .5, 0.])
        x[:, 257:] = torch.arange(10, dtype=x.dtype).reshape(1, 10, 1, 1)
        snapshot = x.clone(); y = normalize_visual_input(x)
        expected = torch.tensor([(4.+.5*16.)/1.5, (36.+.5*64.)/1.5], dtype=x.dtype)
        expected = (expected+RMS_EPSILON).sqrt().reshape(2, 1, 1, 1)
        torch.testing.assert_close(y[:, :256], x[:, :256]/expected)
        self.assertTrue(torch.equal(y[:, 256:], x[:, 256:]))
        self.assertTrue(torch.equal(x, snapshot))
        # Each ROI uses its own scalar, including all channels and valid cells.
        torch.testing.assert_close(y[0, :256, 0, 1], 2*y[0, :256, 0, 0])
        one = normalize_visual_input(x[:1])
        self.assertTrue(torch.equal(one, y[:1]))

    def test_positive_scaling_sensitivity_removed_up_to_fixed_epsilon(self):
        torch.manual_seed(3)
        x = torch.randn(2, 267, 9, 9, dtype=torch.float64)
        x[:, 256] = 1.
        a = normalize_visual_input(x); scaled = x.clone(); scaled[:, :256] *= 3.
        b = normalize_visual_input(scaled)
        torch.testing.assert_close(a[:, :256], b[:, :256], atol=5e-8, rtol=1e-8)
        self.assertTrue(torch.equal(a[:, 256:], b[:, 256:]))

    def test_zero_features_and_empty_support_stay_finite(self):
        x = torch.zeros(2, 267, 9, 9, dtype=torch.float64, requires_grad=True)
        y = normalize_visual_input(x)
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(torch.equal(y, x))
        y.square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_original_parameters_rng_forward_and_zero_terminal_preserved(self):
        torch.manual_seed(1703); a = PairSpecificSizeHead(); rng = torch.get_rng_state()
        torch.manual_seed(1703); b = PairROIRMSSizeHead()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(list(a.state_dict()), list(b.state_dict()))
        self.assertEqual(sum(p.numel() for p in b.parameters()), 2480)
        for k, v in a.state_dict().items():
            self.assertTrue(torch.equal(v, b.state_dict()[k]), k)
        self.assertIs(PairSpecificSizeHead.forward, PairROIRMSSizeHead.forward)
        for layer in b.terminal.branches:
            self.assertEqual(torch.count_nonzero(layer.weight).item(), 0)
            self.assertEqual(torch.count_nonzero(layer.bias).item(), 0)

    def test_disabled_normalization_exact_control_including_parameter_gradients(self):
        torch.manual_seed(7); a = PairSpecificSizeHead().double()
        torch.manual_seed(7); b = PairROIRMSSizeHead().double()
        b.stem.normalization_enabled = False
        x = torch.randn(2, 267, 9, 9, dtype=torch.float64)
        u, v = a.stem(x), b.stem(x)
        self.assertTrue(torch.equal(u, v))
        u.square().sum().backward(); v.square().sum().backward()
        for (ka, pa), (kb, pb) in zip(a.stem.named_parameters(), b.stem.named_parameters()):
            self.assertEqual(ka, kb)
            self.assertTrue(torch.equal(pa.grad, pb.grad), ka)

    def test_all_epoch_ranking_is_not_filtered_by_diagnostic_gates(self):
        from crane_project.tools import train_lvsr_roi_rms_v1 as t
        cfg = t.check_config()
        self.assertEqual(cfg['lr'], .001)
        self.assertEqual(cfg['normalization_epsilon'], RMS_EPSILON)
        a = dict(epoch=1, eligibility_vs_L5=dict(pass_all=False), selection_key=[-675, 1], checkpoint=dict(path='one'))
        b = dict(epoch=2, eligibility_vs_L5=dict(pass_all=True), selection_key=[-674, 2], checkpoint=dict(path='two'))
        chosen = t.choose([a, b])
        self.assertEqual(chosen['representative_epoch'], 1)
        self.assertEqual(chosen['eligible_epochs'], [2])
        self.assertIsNone(chosen['selected_epoch'])
        self.assertFalse(chosen['selection_on_test'])
        self.assertFalse(chosen['automatic_adoption'])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
