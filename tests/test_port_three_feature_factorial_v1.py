import copy
import unittest
import numpy as np
import torch
from crane_project.utils import port_three_feature_factorial_v1 as c


def rows():
    result = []
    for i in range(12):
        result.append(dict(image=str(i), frame_id=i, sequence='real_seq01', domain='real',
            reliability_role='train', pred=[0, 0, 10+i, 5, 0, .2+.05*i],
            gt=[0, 0, 10 if i%2 else 10+i, 5, 0], image_size=[100, 100], audit_risks={}))
    return result


def normalizer(material):
    x = np.asarray([c.linear.descriptor(r['pred'], r['image_size']) for r in material])
    return dict(mean=x.mean(0).tolist(), scale=x.std(0).tolist())


class FactorialTests(unittest.TestCase):
    def test_counts_and_paired_initialization(self):
        a, b = c.SmallRisk(1701), c.SmallRisk(1701)
        self.assertEqual(sum(p.numel() for p in a.parameters()), 41)
        self.assertTrue(all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items()))
        self.assertTrue(all(p.dtype == torch.float64 for p in a.parameters()))

    def test_objective_gradient_and_class_weights(self):
        r = rows(); x, y, w, _ = c.prepare(r, normalizer(r))
        self.assertAlmostEqual(w[y == 0].sum(), .5)
        self.assertAlmostEqual(w[y == 1].sum(), .5)
        model = c.SmallRisk(1701)
        tx, ty, tw = (torch.tensor(v, dtype=torch.float64) for v in (x, y, w))
        loss = c.objective(model, tx, ty, tw, .1); loss.backward()
        for p in model.parameters():
            expected = p.grad.reshape(-1)[0].item(); old = p.reshape(-1)[0].item()
            with torch.no_grad(): p.reshape(-1)[0] = old+1e-6
            hi = c.objective(model, tx, ty, tw, .1).item()
            with torch.no_grad(): p.reshape(-1)[0] = old-1e-6
            lo = c.objective(model, tx, ty, tw, .1).item()
            with torch.no_grad(): p.reshape(-1)[0] = old
            self.assertAlmostEqual(expected, (hi-lo)/2e-6, places=7)

    def test_numpy_replay_and_online_input_scope(self):
        model = c.SmallRisk(1701); r = rows(); norm = normalizer(r)
        exported = dict(kind='three_feature_relu8', weights={k: v.detach().tolist() for k, v in model.state_dict().items()}, **norm)
        matrix = np.asarray([c.linear.descriptor(v['pred'], v['image_size']) for v in r])
        expected = torch.sigmoid(model(torch.tensor((matrix-norm['mean'])/norm['scale'], dtype=torch.float64))).detach().numpy()
        np.testing.assert_allclose(c.numpy_risk(exported, matrix), expected, atol=1e-15, rtol=0)
        a, b = copy.deepcopy(r[0]), copy.deepcopy(r[0]); b['gt'] = [0, 0, 99, 1, 0]; b['sequence'] = 'another'
        c.add_scores([a], {'test': exported}); c.add_scores([b], {'test': exported})
        self.assertEqual(a['audit_risks'], b['audit_risks'])

    def test_training_scope_and_missing(self):
        r = rows(); norm = normalizer(r)
        r.append(dict(r[0], image='missing', pred=None))
        self.assertEqual(len(c.prepare(r, norm)[1]), 12)
        with self.assertRaises(ValueError): c.prepare(r+r, norm)
        r[0]['reliability_role'] = 'val'
        with self.assertRaises(ValueError): c.prepare(r, norm)

    def test_real_training_updates(self):
        torch.set_num_threads(1)
        r = rows(); settings = dict(l2=.1, max_iterations=30, max_evaluations=50,
            gradient_tolerance=1e-8, change_tolerance=1e-12, stationarity_limit=1e-5)
        _, exported, log = c.fit(r, normalizer(r), 1701, settings)
        self.assertGreater(exported['optimizer_iterations'], 0)
        self.assertLess(exported['objective_final'], exported['objective_initial'])
        self.assertEqual(exported['train_rows'], 12)
        self.assertEqual(exported['function_evaluations'], len(log))

    def test_gate_requires_each_seed_and_convergence(self):
        def item(fa, real=189):
            return {'at_frozen_VAL_cut': {'all': {'states': dict(FA=fa, CR=653)},
                'domain:real': {'states': dict(CR=real)}}}
        results = dict(formal=item(190)); models = {}
        for s in c.SEEDS:
            results['mlp_old_'+str(s)] = item(190)
            results['mlp_augmented_'+str(s)] = item(180)
            for a in c.ARMS: models['mlp_'+a+'_'+str(s)] = dict(converged=True)
        self.assertTrue(c.gate(results, models)['passed'])
        models['mlp_old_1702']['converged'] = False
        self.assertFalse(c.gate(results, models)['passed'])
        models['mlp_old_1702']['converged'] = True
        results['mlp_augmented_1703'] = item(190)
        self.assertFalse(c.gate(results, models)['passed'])


if __name__ == '__main__': unittest.main()
