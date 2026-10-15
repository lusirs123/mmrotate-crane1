import copy
import unittest
import numpy as np
import torch
from crane_project.utils import port_reliability_response_stats_v1 as core
from crane_project.utils import port_reliability_response_stats_v1_torch as native


class ResponseTests(unittest.TestCase):
    def test_statistics_known_cases_and_reference(self):
        p = np.full((2, 4, 9, 9), 1/81.)
        s = core.statistics(p)
        np.testing.assert_allclose(s[..., 0], 1.)
        np.testing.assert_allclose(s[..., 1], 1/81.)
        np.testing.assert_allclose(s[..., 3], 32/81.)
        p[:] = 0.; p[:, :, 4, 4] = 1.
        np.testing.assert_allclose(core.statistics(p), np.broadcast_to([0, 1, 0, 0], (2, 4, 4)))
        p[:] = 0.; p[:, :, 0, 0] = 1.
        np.testing.assert_allclose(core.statistics(p)[..., 3], 1.)
        np.testing.assert_allclose(native.statistics(torch.tensor(p)).numpy(), core.statistics(p), atol=1e-14)

    def test_prior_difference_and_zero_mass(self):
        rng = np.random.RandomState(1)
        p = rng.uniform(size=(5, 4, 9, 9)); p /= p.sum((2, 3), keepdims=True)
        np.testing.assert_array_equal(core.response_features(p, p), np.zeros((5, 16)))
        q = np.full_like(p, 1/81.)
        np.testing.assert_allclose(core.response_features(p, q),
            native.response_features(torch.tensor(p), torch.tensor(q)).numpy(), atol=1e-14)
        for value in (np.zeros_like(p), -p, p*2, p[:, :, :, :8]):
            with self.assertRaises(ValueError):
                core.statistics(value)

    def test_statistics_rotation_invariance_and_map_pairing(self):
        rng = np.random.RandomState(2)
        p = rng.uniform(size=(3, 4, 9, 9)); p /= p.sum((2, 3), keepdims=True)
        np.testing.assert_allclose(core.statistics(p), core.statistics(np.rot90(p, axes=(2, 3))), atol=1e-14)
        perm = [3, 1, 0, 2]
        np.testing.assert_allclose(core.statistics(p[:, perm]), core.statistics(p)[:, perm])

    def test_shared_initialization_and_export(self):
        for seed in core.SEEDS:
            a, b = native.Risk(seed, 3), native.Risk(seed, 19)
            self.assertEqual(sum(p.numel() for p in a.parameters()), 41)
            self.assertEqual(sum(p.numel() for p in b.parameters()), 169)
            x = torch.randn(7, 3, dtype=torch.float64)
            extra = torch.randn(7, 16, dtype=torch.float64)
            self.assertTrue(torch.equal(a(x), b(torch.cat((x, extra), 1))))
            self.assertTrue(torch.equal(a.network[0].weight, b.network[0].weight[:, :3]))

    def test_gradient_finite_difference_and_extra_learning(self):
        torch.set_num_threads(1)
        rng = np.random.RandomState(6)
        x = torch.tensor(rng.normal(size=(28, 19)), dtype=torch.float64)
        y = (x[:, 3] > 0).double()
        w = torch.where(y == 1, .5/y.sum(), .5/(len(y)-y.sum()))
        model = native.Risk(1701, 19)
        loss = native.objective(model, x, y, w); loss.backward()
        analytic = model.network[0].weight.grad[0, 3].item()
        self.assertGreater(abs(analytic), 1e-10)
        p = model.network[0].weight
        with torch.no_grad():
            original = p[0, 3].item(); eps = 1e-6
            p[0, 3] = original+eps; plus = native.objective(model, x, y, w).item()
            p[0, 3] = original-eps; minus = native.objective(model, x, y, w).item()
            p[0, 3] = original
        self.assertAlmostEqual(analytic, (plus-minus)/(2*eps), places=8)
        obj, exported, records = native.fit(x.numpy(), y.numpy(), 1701)
        self.assertTrue(exported['converged'])
        self.assertGreater(np.linalg.norm(np.asarray(exported['weights']['network.0.weight'])[:, 3:]), 0)
        np.testing.assert_allclose(torch.sigmoid(obj(torch.tensor(core.normalize(x.numpy(), exported['normalizer'])))).detach().numpy(),
                                   core.numpy_risk(exported, x.numpy()), atol=1e-12)

    def test_train_normalization_and_flag_protection(self):
        with self.assertRaises(ValueError):
            core.normalize_fit(np.ones((4, 19)), 'VAL')
        original = dict(final_box_original=[1, 2, 3, 4, .1, .9], center_accepted=True,
                        angle_accepted=False, size_accepted=True, risks=dict(size=.1, angle=.8))
        snapshot = copy.deepcopy(original)
        output = core.size_decision(original, .8, .4)
        self.assertFalse(output['size_accepted']); self.assertTrue(output['center_accepted'])
        self.assertEqual({k: v for k, v in output.items() if k != 'size_accepted'},
                         {k: v for k, v in original.items() if k != 'size_accepted'})
        self.assertEqual(original, snapshot)
        missing = dict(original, final_box_original=None, center_accepted=False, size_accepted=False)
        self.assertFalse(core.size_decision(missing, None, .4)['size_accepted'])
        with self.assertRaises(ValueError):
            core.size_decision(missing, .1, .4)


if __name__ == '__main__':
    unittest.main()
