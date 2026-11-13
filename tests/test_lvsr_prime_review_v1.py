"""Independent formula/gradient/provenance checks; no parameter updates."""
import math
import unittest

import torch

from crane_project.utils import lvsr_prime_review_v1 as p
from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead
from tests.test_lvsr_conr_v1 import inputs


class PrimeReviewChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_proxy_global_pair_formula_and_squared_spread(self):
        z = torch.tensor([[1., .2], [-.3, 1.], [-1., -.4]], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([[0., 0.], [.03, .01], [-.02, .04]], dtype=z.dtype, requires_grad=True)
        normalized = z / z.norm(dim=1, keepdim=True)
        pairs = [(i, j) for i in range(3) for j in range(3) if i != j]
        dt = [float((y[i]-y[j]).norm().detach())/p.WINDOW for i, j in pairs]
        df = [(normalized[i]-normalized[j]).norm() for i, j in pairs]
        target_exp = [math.exp(-d) for d in dt]
        q = torch.stack([torch.exp(-5*d) for d in df]); q = q/q.sum()
        expected = sum((target_exp[k]/sum(target_exp)) *
                       (math.log(target_exp[k]/sum(target_exp))-q[k].log())
                       - .005*dt[k]*(1-normalized[i].dot(normalized[j])).square()
                       for k, (i, j) in enumerate(pairs))
        actual, _ = p.proxy_loss(z, y)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        actual.backward(); self.assertIsNone(y.grad)
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_alignment_cross_entropy_batch_independent_and_target_detached(self):
        z = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64, requires_grad=True)
        proxies = p.new_proxy_features(3, 2, dtype=z.dtype)
        labels = torch.tensor([[0., 0.], [.04, -.02]], dtype=z.dtype, requires_grad=True)
        centers = torch.tensor([[0., 0.], [.03, -.02], [-.01, .04]], dtype=z.dtype, requires_grad=True)
        losses, target, log_a = p.alignment_per_sample(z, labels, proxies, centers)
        expected = []
        for i in range(2):
            logits_t = torch.stack([-((labels[i]-q).detach().norm()/p.WINDOW) for q in centers])
            logits_f = torch.stack([-5*(z[i]-q/q.norm()).norm() for q in proxies])
            expected.append(-(logits_t.softmax(0)*logits_f.log_softmax(0)).sum())
        torch.testing.assert_close(losses, torch.stack(expected))
        one, _, _ = p.alignment_per_sample(z[:1], labels[:1], proxies, centers)
        torch.testing.assert_close(one[0], losses[0])
        losses.sum().backward()
        self.assertIsNone(labels.grad); self.assertIsNone(centers.grad)
        self.assertGreater(float(z.grad.norm()), 0.)

    def test_support_is_masked_and_detached_even_for_zero_embeddings(self):
        f = torch.randn(2, 8, 2, 2, dtype=torch.float64, requires_grad=True)
        w = torch.tensor([[[[1., 0.], [1., 0.]]]]*2, dtype=f.dtype, requires_grad=True)
        z, norm = p.embedding(f, w)
        z.sum().backward()
        self.assertEqual(float(f.grad[..., 1].abs().sum()), 0.)
        self.assertIsNone(w.grad)
        zero, _ = p.embedding(torch.zeros_like(f), torch.zeros_like(w))
        losses, _, _ = p.alignment_per_sample(zero, torch.zeros(2, 2, dtype=f.dtype),
            p.new_proxy_features(3, 32, dtype=f.dtype), torch.randn(3, 2, dtype=f.dtype))
        self.assertTrue(torch.isfinite(losses).all())

    def test_kmeans_deterministic_nonmutating_and_known_means(self):
        y = torch.tensor([[0., 0.], [.002, 0.], [1., 1.], [1.002, 1.]], dtype=torch.float64, requires_grad=True)
        before = y.detach().clone()
        a, receipt = p.kmeans_targets(y, count=2)
        b, other = p.kmeans_targets(y, count=2)
        self.assertTrue(torch.equal(a, b)); self.assertEqual(receipt, other)
        self.assertTrue(torch.equal(y, before)); self.assertFalse(a.requires_grad)
        self.assertEqual(sorted(receipt['counts']), [2, 2])
        actual = a[a[:, 0].argsort()]
        torch.testing.assert_close(actual, torch.tensor([[.001, 0.], [1.001, 1.]], dtype=y.dtype))
        with self.assertRaises(ValueError):p.kmeans_targets(torch.zeros(4, 2), count=2)

    def test_axes_and_proxy_initialization_do_not_change_head_rng_or_state(self):
        torch.manual_seed(1703); head = PairSpecificSizeHead().double()
        before = {k: v.clone() for k, v in head.state_dict().items()}; rng = torch.get_rng_state()
        q = p.new_proxy_features(20, 648, dtype=torch.float64)
        self.assertEqual(q.numel(), 12960); self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(sum(v.numel() for v in head.parameters()), 2480)
        self.assertTrue(all(torch.equal(before[k], v) for k, v in head.state_dict().items()))
        bm = torch.tensor([[0., 0., 20., 10., 0.], [0., 0., 10., 20., 0.]])
        m = torch.tensor([[0., 0., 10., 20., 0., .9]]*2)
        self.assertTrue(torch.equal(p.canonical_pair(torch.tensor([[1., 2.]]*2), bm, m), torch.tensor([[2., 1.], [1., 2.]])))

    def test_fixed_iteration_limit_reports_approximation_without_restart(self):
        y = torch.tensor([[0., 0.], [.1, .01], [.2, .02], [.3, .03], [1., 1.]])
        centers, receipt = p.kmeans_targets(y, count=2, max_iterations=1)
        self.assertEqual(receipt['iterations'], 1)
        self.assertEqual(sum(receipt['counts']), len(y))
        self.assertTrue(torch.isfinite(centers).all())
        self.assertEqual(receipt['exact_assignment_converged'], receipt['final_assignment_changes'] == 0)

    def test_neutral_original_forward_and_aux_reaches_only_stem(self):
        from crane_project.utils import port_size_qalign_v1 as model
        torch.manual_seed(1703); head = PairSpecificSizeHead().double().train(); x = inputs()
        captured = []
        hook = head.stem.register_forward_hook(lambda module, args, out: captured.append(out))
        try:out = head(*x[:-1])
        finally:hook.remove()
        self.assertTrue(torch.equal(out['boxes_original'], x[-2]))
        self.assertEqual(len(head.stem._forward_hooks), 0)
        target = model.residual.size_targets(x[-1], x[-2], x[3])['log_residual']
        target = p.canonical_pair(target, x[3], x[-2])
        z, _ = p.embedding(captured[0], out['common_support'])
        proxies = p.new_proxy_features(3, z.shape[1], dtype=z.dtype)
        labels = torch.tensor([[-.05, -.03], [0., 0.], [.04, .06]], dtype=z.dtype)
        aux, _, _ = p.alignment_per_sample(z, target, proxies, labels)
        size = model.size_loss(out, x[-1], x[-2], x[3])
        self.assertIs(p.combined_loss(size, aux.mean(), aux.mean(), 0., 0.), size)
        gs = torch.autograd.grad(size, list(head.stem.parameters()), retain_graph=True)
        self.assertEqual(sum(float(g.abs().sum()) for g in gs), 0.)
        ga = torch.autograd.grad(aux.mean(), list(head.parameters()), allow_unused=True)
        for (name, parameter), gradient in zip(head.named_parameters(), ga):
            if name.startswith('terminal'):self.assertIsNone(gradient)
        self.assertGreater(sum(float(g.abs().sum()) for g in ga if g is not None), 0.)
        self.assertFalse(any(v.grad is not None for v in head.parameters()))


if __name__ == '__main__':
    unittest.main()
