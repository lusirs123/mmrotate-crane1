import inspect
import json
from pathlib import Path
import unittest
from copy import deepcopy
import numpy as np
from crane_project.utils import port_reliability_hard_error_pauc_v1 as c
from test_port_reliability_temporal_fusion_v1 import toy


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = toy(); cls.base = c.parent.base.fit(cls.raw)
        cls.rows = c.parent.base.score(cls.raw, cls.base)
        cls.old = c.parent.fit(cls.rows, cls.base)
        cls.rows = c.parent.score(cls.rows, cls.base, cls.old)
        cls.hard = c.hard_set(cls.rows)

    def test_weighted_gradient_hessian(self):
        rng = np.random.default_rng(13); good = rng.normal(size=(7, 4)); wrong = rng.normal(size=(3, 4))
        q = np.array([2., 1., 1.]) / 4; v = np.array([.2, -.1, .1, .3, 1.]); eps = 1e-5
        _, g, H = c.objective(v, good, wrong, q)
        for j in range(5):
            d = np.eye(5)[j] * eps
            l, r = c.objective(v - d, good, wrong, q), c.objective(v + d, good, wrong, q)
            self.assertAlmostEqual(g[j], (r[0] - l[0]) / (2 * eps), places=7)
            np.testing.assert_allclose(H[:, j], (r[1] - l[1]) / (2 * eps), atol=2e-7, rtol=0)
        self.assertGreater(np.linalg.eigvalsh(H).min(), 0)

    def test_uniform_objective_restores_parent(self):
        rng = np.random.default_rng(17); a = rng.normal(size=(9, 4)); b = rng.normal(size=(4, 4)); v = rng.normal(size=5)
        old, new = c.parent.objective(v, a, b), c.objective(v, a, b, np.ones(4) / 4)
        for x, y in zip(old, new): np.testing.assert_allclose(x, y, atol=2e-13, rtol=0)

    def test_fixed_ties_and_no_VAL_threshold(self):
        rows = deepcopy(self.rows)
        for r in rows:
            if c.level.metrics.bad(r) is True: r['risks']['partial_auc4'] = .3
        h = c.hard_set(rows)
        self.assertEqual(h['actual_count'], h['wrong'])
        self.assertTrue(h['no_VAL_threshold_used'])
        for r in rows: r['unused_VAL_cutoff'] = .999
        self.assertEqual(c.hard_set(rows), h)

    def test_reject_roles_and_changed_set(self):
        rows = deepcopy(self.rows); rows[0]['reliability_role'] = 'val'
        with self.assertRaises(ValueError): c.hard_set(rows)
        with self.assertRaises(ValueError): c.fit(rows, self.base, self.hard, True)
        h = dict(self.hard, images=[])
        with self.assertRaises(ValueError): c.fit(self.rows, self.base, h, True)

    def test_fits_preserve_inputs_and_control(self):
        saved = deepcopy((self.rows, self.base, self.hard))
        uniform = c.fit(self.rows, self.base, self.hard, False)
        weighted = c.fit(self.rows, self.base, self.hard, True)
        np.testing.assert_allclose(uniform['weights'], self.old['weights'], atol=1e-11, rtol=0)
        self.assertEqual(saved, (self.rows, self.base, self.hard))
        q = weighted['error_weights']; hard = set(self.hard['images'])
        ratios = [v for i, v in zip(weighted['wrong_images'], q) if i in hard]
        rest = [v for i, v in zip(weighted['wrong_images'], q) if i not in hard]
        if rest: self.assertAlmostEqual(ratios[0] / rest[0], 2.)
        self.assertLessEqual(weighted['max_abs_gradient'], 1e-8)

    def test_bad_weights_and_nonfinite(self):
        for q in ([0., 1.], [.5, .4], [float('nan'), .5]):
            with self.assertRaises(ValueError): c.objective(np.zeros(5), np.ones((3, 4)), np.ones((2, 4)), q)

    def test_GT_free_score_and_missing(self):
        model = c.fit(self.rows, self.base, self.hard, True)
        changed = deepcopy(self.rows)
        for r in changed: r['gt'] = [0., 0., 999., 111., 0.]
        a, b = c.score(self.rows, {c.METHOD: model}), c.score(changed, {c.METHOD: model})
        self.assertEqual([r['risks'] for r in a], [r['risks'] for r in b])
        self.assertNotIn('gt', inspect.signature(c.CausalHardErrorPAUC.decide).parameters)
        r = deepcopy(self.rows[0]); r['pred'] = None; r['risks']['partial_auc4'] = None
        self.assertIsNone(c.score([r], {c.METHOD: model})[0]['risks'][c.METHOD])

    def test_locked_fallback_flags(self):
        r = deepcopy(self.rows[0]); r['temporal']['available'] = False
        r['risks']['partial_auc4'] = .45; r['risks'][c.METHOD] = .45
        self.assertEqual(c.accepted([r], c.METHOD, .99), set())
        r['risks']['partial_auc4'] = r['risks'][c.METHOD] = c.FALLBACK_CUTOFF
        self.assertEqual(c.accepted([r], c.METHOD, .01), {r['image']})

    def test_locked_matching_counts(self):
        scored = c.score(self.rows, {c.CONTROL: self.old})
        held = c.fixed(scored); target = len(held) + 3
        result = c.locked_point(scored, c.CONTROL, count=target)
        self.assertEqual(result['accepted_outputs'], target)
        self.assertTrue(result['fallback_decisions_locked'])

    def test_no_budget_or_architecture_change(self):
        for key, value in c.parent.SETTINGS.items(): self.assertEqual(c.SETTINGS[key], value)
        self.assertEqual(c.SETTINGS['hard_multiplier'], 2.)
        self.assertEqual(c.SETTINGS['hard_fraction'], .25)

    def test_online_flag_and_geometry_protection(self):
        policy = json.loads((Path(__file__).resolve().parents[1] / 'work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1/fit/policy.json').read_text())['simple_policy']
        api = c.CausalHardErrorPAUC(self.base, self.old, .99, policy)
        original = c.parent.base.simple.SimpleComponentReliability(policy)
        for i in range(6):
            p = None if i == 4 else [0., 0., 100., 50., 0., .8]
            before = deepcopy(p); a = api.decide(p, [1000, 1000], i); b = original.decide(p, [1000, 1000])
            for key in ('raw_b_output', 'center_accepted', 'angle_accepted', 'risks'): self.assertEqual(a[key], b[key])
            if not a['size_partial_auc_history_available']:
                risk = a['size_partial_auc_risk']
                self.assertEqual(a['size_accepted'], risk is not None and risk <= c.FALLBACK_CUTOFF)
            self.assertEqual(before, p)

    def test_complete_calibration_and_metrics_branch_contract(self):
        rows = []
        for i in range(887):
            r = deepcopy(self.rows[i % len(self.rows)])
            r.update(image='real_seq01_%05d' % i, sequence='real_seq01', frame_id=i, split='val', reliability_role='val')
            r['temporal']['acquisition_boundary'] = i == 0
            r['risks']['crossfit_fusion'] = r['risks']['partial_auc4']
            rows.append(r)
        models = {c.CONTROL: self.old, c.METHOD: self.old}; rows = c.score(rows, models)
        cuts = {m: c.calibrate(rows, m) for m in models}
        evaluation = c.evaluate(rows, cuts)
        for method in models:
            counts = evaluation['groups']['all'][method]['candidate']['states']
            self.assertEqual(sum(counts.values()), 887)
            self.assertGreaterEqual(counts['CR'], np.ceil(.95 * (counts['CR'] + counts['FR'])))
        self.assertFalse(c.gate(evaluation)['passed'])  # Identical arms cannot claim strict gain.


if __name__ == '__main__': unittest.main()
