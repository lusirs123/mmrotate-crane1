"""Loss mathematics, masks, axis identity, neutral delivery and actual updates."""
import math
import unittest
import torch
from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead
from crane_project.utils import port_size_pair_conr_v1 as c


def inputs():
    torch.manual_seed(41)
    n = 4
    b = torch.tensor([[150., 180., 100., 60., .2, .9]] * n, dtype=torch.float64)
    bm = b[:, :5].clone()
    gt = bm.clone(); gt[:, 2:4] *= torch.tensor([1., 1.005, 1.09, .92], dtype=b.dtype)[:, None]
    return [torch.randn(n, 256, 9, 9, dtype=b.dtype), torch.ones(n, 1, 9, 9, dtype=b.dtype),
            b, bm, torch.ones(n, 2, dtype=b.dtype), b.clone(), gt]


class ConRChecks(unittest.TestCase):
    def test_loss_matches_explicit_pair_formula_and_excludes_self(self):
        f = torch.tensor([[[[1.]], [[0.]]], [[[.8]], [[.6]]], [[[0.]], [[1.]]]],
                         dtype=torch.float64, requires_grad=True)
        target = torch.tensor([[0., 0.], [.5*c.WINDOW, 0.], [3*c.WINDOW, 0.]], dtype=f.dtype, requires_grad=True)
        pred = torch.zeros_like(target, requires_grad=True)
        loss, stats = c.contrastive_loss(f, torch.ones(3, 1, 1, 1), target, pred)
        z = torch.nn.functional.normalize(f.flatten(1), dim=1)
        s = z @ z.T / c.TEMPERATURE
        expected = (torch.logsumexp(torch.stack([s[0, 1], s[0, 2]+.03]), 0)-s[0, 1]
            + torch.logsumexp(torch.stack([s[1, 0], s[1, 2]+.025]), 0)-s[1, 0])/3
        torch.testing.assert_close(loss, expected)
        self.assertEqual(stats['active_anchors'], 2)
        self.assertEqual(stats['positive_pairs'], 2)
        self.assertEqual(stats['negative_pairs'], 4)
        loss.backward()
        self.assertTrue(torch.isfinite(f.grad).all()); self.assertGreater(f.grad.abs().sum().item(), 0.)
        self.assertIsNone(pred.grad); self.assertIsNone(target.grad)

    def test_no_pairs_no_negatives_zero_support_are_finite(self):
        for n in (1, 3):
            for support in (0., 1.):
                f = torch.zeros(n, 8, 9, 9, requires_grad=True)
                t = torch.zeros(n, 2)
                loss, stats = c.contrastive_loss(f, torch.full((n, 1, 9, 9), support), t, t)
                self.assertEqual(loss.item(), 0.); self.assertEqual(stats['active_anchors'], 0)
                loss.backward(); self.assertTrue(torch.isfinite(f.grad).all())
        # All labels distinct => no positive; distant predictions => no negative.
        f = torch.randn(3, 2, 1, 1, requires_grad=True)
        t = torch.tensor([[0., 0.], [.2, .2], [.4, .4]])
        for p in (t, torch.zeros_like(t)):
            loss, stats = c.contrastive_loss(f, torch.ones(3, 1, 1, 1), t, p)
            self.assertEqual(loss.item(), 0.)

    def test_support_and_prediction_selection(self):
        torch.manual_seed(5)
        f = torch.randn(3, 2, 1, 2, dtype=torch.float64, requires_grad=True)
        w = torch.tensor([[[[1., 0.]]]] * 3, dtype=f.dtype)
        t = torch.tensor([[0., 0.], [.001, 0.], [.1, 0.]], dtype=f.dtype)
        a, s = c.contrastive_loss(f, w, t, torch.zeros_like(t))
        changed = f.detach().clone(); changed[..., 1] = 1e8
        b, _ = c.contrastive_loss(changed, w, t, torch.zeros_like(t))
        torch.testing.assert_close(a, b)
        a.backward(); self.assertEqual(f.grad[..., 1].abs().sum().item(), 0.)
        no, info = c.contrastive_loss(f, w, t, t)
        self.assertEqual(no.item(), 0.); self.assertEqual(info['negative_pairs'], 0)

    def test_initialization_and_full_forward_identity_training_eval_and_empty(self):
        torch.manual_seed(1703); a = PairSpecificSizeHead().double(); state = torch.get_rng_state()
        torch.manual_seed(1703); b = c.PairConRSizeHead().double()
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in b.parameters()), 2480)
        self.assertEqual(list(a.state_dict()), list(b.state_dict()))
        for k in a.state_dict():self.assertTrue(torch.equal(a.state_dict()[k], b.state_dict()[k]))
        x = inputs()
        for train in (False, True):
            a.train(train); b.train(train)
            aa, bb = a(*x[:-1]), b(*x[:-1])
            self.assertEqual('training_features' in bb, train)
            for k in aa:
                if isinstance(aa[k], torch.Tensor):self.assertTrue(torch.equal(aa[k], bb[k]), k)
            self.assertTrue(torch.equal(bb['boxes_original'], x[-2]))
            self.assertEqual(len(b.stem._forward_hooks), 0)
        empty = b(*[v[:0] for v in x[:-1]])
        self.assertEqual(len(empty['boxes_original']), 0)
        self.assertEqual(len(b.stem._forward_hooks), 0)

    def test_zero_weight_exact_adam_control_and_nonzero_auxiliary(self):
        from crane_project.tools.run_symeood_110_lvsr_v1 import update as original_update
        torch.manual_seed(1703); a = PairSpecificSizeHead().double()
        torch.manual_seed(1703); b = c.PairConRSizeHead().double()
        oa = torch.optim.Adam(a.parameters(), lr=.001); ob = torch.optim.Adam(b.parameters(), lr=.001)
        x = inputs()
        for _ in range(2):
            old = original_update(a, oa, x, 'fixed_b', torch)
            new = c.update(b, ob, x, 'fixed_b', torch, weight=0.)
            self.assertEqual(old['loss'], new['loss'])
            for (ka, pa), (kb, pb) in zip(a.named_parameters(), b.named_parameters()):
                self.assertEqual(ka, kb); self.assertTrue(torch.equal(pa, pb), ka)
                self.assertTrue(torch.equal(pa.grad, pb.grad), ka)
                for k in oa.state[pa]:self.assertTrue(torch.equal(oa.state[pa][k], ob.state[pb][k]), k)
        new = c.update(b, ob, x, 'fixed_b', torch)
        self.assertGreater(new['active_anchors'], 0); self.assertGreater(new['weighted_conr_loss'], 0.)
        self.assertTrue(math.isfinite(new['grad_norm']))

    def test_axis_swaps_and_frozen_config(self):
        v = torch.tensor([[1., 2.]]*4)
        bm = torch.tensor([[0,0,10,5,0],[0,0,5,10,0],[0,0,10,5,0],[0,0,5,10,0.]])
        m = torch.tensor([[0,0,10,5,0],[0,0,10,5,0],[0,0,5,10,0],[0,0,5,10,0.]])
        self.assertTrue(torch.equal(c.canonical_pair(v,bm,m), torch.tensor([[1.,2.],[2.,1.],[2.,1.],[1.,2.]])))
        from crane_project.tools import train_lvsr_conr_v1 as trainer
        cfg = trainer.check_config()
        self.assertEqual(cfg['lr'], .001); self.assertEqual(cfg['conr']['weight'], .001)
        self.assertEqual(cfg['physical_gpu'], 2)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
