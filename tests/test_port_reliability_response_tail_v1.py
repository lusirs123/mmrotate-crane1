import copy
import math
import unittest
import numpy as np
import torch
from crane_project.utils import port_reliability_response_tail_v1 as core
from crane_project.utils import port_reliability_response_stats_v1 as response
from crane_project.utils.port_reliability_response_stats_v1_torch import Risk
from crane_project.tools.review_port_reliability_response_tail_v1 import independent_model


class TailTests(unittest.TestCase):
    def test_pair_direction_top_goods_and_shift(self):
        z = torch.tensor([-2., .2, 1.2, -.4, 2.], dtype=torch.float64)
        y = torch.tensor([0., 0., 0., 1., 1.], dtype=torch.float64)
        reference = max(sum(math.log1p(math.exp(g-b)) for b in [-.4, 2.])/2 for g in [-2., .2, 1.2])
        self.assertAlmostEqual(core.data_loss(z, y, 'tail').item(), reference, places=12)
        self.assertAlmostEqual(core.data_loss(z+4, y, 'tail').item(), reference, places=12)
        better = z.clone(); better[:3] -= 1; better[3:] += 1
        self.assertLess(core.data_loss(better, y, 'tail'), core.data_loss(z, y, 'tail'))
        with self.assertRaises(ValueError):
            core.data_loss(z, torch.zeros_like(y), 'tail')

    def test_tail_two_percent_rounding_and_gradient(self):
        z = torch.linspace(-1, 1, 103, dtype=torch.float64, requires_grad=True)
        y = torch.cat((torch.zeros(101), torch.ones(2))).double()
        h = torch.nn.functional.softplus(z[:101, None]-z[101:][None]).mean(1)
        self.assertAlmostEqual(core.data_loss(z, y, 'tail').item(), h[-3:].mean().item(), places=12)
        value = core.data_loss(z, y, 'tail'); value.backward()
        self.assertTrue(torch.equal(z.grad[:98], torch.zeros_like(z.grad[:98])))
        self.assertTrue(bool((z.grad[98:101] > 0).all()))
        self.assertTrue(bool((z.grad[101:] < 0).all()))
        self.assertAlmostEqual(z.grad.sum().item(), 0., places=12)

    def test_both_objectives_manual_gradient_and_finite_difference(self):
        rng = np.random.RandomState(9); raw = rng.normal(size=(25, 19))
        normalizer = response.normalize_fit(raw, 'TRAIN')
        x = torch.tensor(response.normalize(raw, normalizer)); y = torch.tensor([0.]*19+[1.]*6)
        for kind in ('bce', 'tail'):
            model = Risk(1701, 19)
            with torch.no_grad():
                model.network[0].weight[:, 3:] = torch.tensor(rng.normal(scale=.1, size=(8, 16)))
            loss = core.objective(model, x, y, kind); loss.backward()
            maximum = max(p.grad.abs().max().item() for p in model.parameters())
            exported = dict(normalizer=normalizer, objective_kind=kind,
                weights={k: v.detach().tolist() for k, v in model.state_dict().items()},
                objective_final=loss.item(), max_abs_gradient=maximum)
            values, proof = independent_model(exported, raw, y.numpy())
            np.testing.assert_allclose(values, response.numpy_risk(exported, raw), atol=1e-12)
            parameter = model.network[0].weight; analytic = parameter.grad[0, 4].item()
            with torch.no_grad():
                original = parameter[0, 4].item(); eps = 1e-6
                parameter[0, 4] = original+eps; plus = core.objective(model, x, y, kind).item()
                parameter[0, 4] = original-eps; minus = core.objective(model, x, y, kind).item()
                parameter[0, 4] = original
            self.assertAlmostEqual(analytic, (plus-minus)/(2*eps), places=7)

    def test_fixed_update_export_and_initialization(self):
        rng = np.random.RandomState(1); raw = rng.normal(size=(24, 19)); y = np.asarray([0.]*18+[1.]*6)
        torch.set_num_threads(1)
        saved = copy.deepcopy(core.SETTINGS)
        try:
            core.SETTINGS['steps'] = 4
            a, ma, records = core.fit(raw, y, 1701, 'bce')
            b, mb, _ = core.fit(raw, y, 1701, 'tail')
            self.assertEqual(ma['initial_state_sha256'], mb['initial_state_sha256'])
            self.assertNotEqual(ma['final_state_sha256'], mb['final_state_sha256'])
            self.assertEqual(records[-1]['step'], 4)
            self.assertEqual(ma['parameters'], 169)
            self.assertTrue(ma['gradient_is_diagnostic_only'])
            self.assertGreater(np.linalg.norm(np.asarray(mb['weights']['network.0.weight'])[:, 3:]), 0)
            independent_model(mb, raw, y)
        finally:
            core.SETTINGS.clear(); core.SETTINGS.update(saved)

    def test_zero_added_path_and_flag_preservation(self):
        a, b = Risk(1701, 3), Risk(1701, 19)
        self.assertTrue(torch.equal(a.network[0].weight, b.network[0].weight[:, :3]))
        self.assertEqual(torch.count_nonzero(b.network[0].weight[:, 3:]).item(), 0)
        original = dict(final_box_original=[1, 2, 3, 4, .1, .8], center_accepted=True,
                        angle_accepted=True, size_accepted=True, risks=dict(size=.1))
        decision = response.size_decision(original, .8, .2)
        self.assertEqual({k: v for k, v in original.items() if k != 'size_accepted'},
                         {k: v for k, v in decision.items() if k != 'size_accepted'})
        self.assertFalse(decision['size_accepted'])

    def test_one_global_cut_protects_both_and_keeps_ties(self):
        saved = copy.deepcopy(core.SETTINGS)
        try:
            core.SETTINGS.update(primary_CR=3, minimum_Real_CR=2)
            rows = []
            for i, (domain, score, wrong) in enumerate([
                    ('sim', .1, False), ('sim', .2, False), ('real', .3, False),
                    ('sim', .4, True), ('real', .5, False), ('sim', .5, False)]):
                rows.append(dict(image=str(i), domain=domain, pred=[0, 0, 20 if wrong else 10, 10, 0, .9],
                    gt=[0, 0, 10, 10, 0], audit_risks={'test': score}))
            self.assertEqual(core.global_cut(rows, 'test'), .5)
            # Whole tie at .5 retains one more good than the real requirement.
            self.assertEqual(sum(r['audit_risks']['test'] <= .5 for r in rows), 6)
        finally:
            core.SETTINGS.clear(); core.SETTINGS.update(saved)


if __name__ == '__main__':
    unittest.main()
