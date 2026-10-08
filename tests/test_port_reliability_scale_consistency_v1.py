from copy import deepcopy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from crane_project.utils import port_reliability_scale_consistency_v1 as m
from crane_project.utils import port_reliability_feature_ablation_v1 as ab
from crane_project.tools import run_port_reliability_scale_consistency_v1 as runner


def row(i=0, bad=False, pred=True):
    p = [20., 30., 100., 20., .2, .8] if pred else None
    gt = [20., 30., 100., 16. if bad else 20., .2]
    d = dict(center_accepted=pred, size_accepted=pred, angle_accepted=pred,
             final_box_original=p, risks=dict(size=.2 if pred else None, angle=.3 if pred else None))
    return dict(image='real_seq01_'+str(i), sequence='real_seq01', domain='real',
                split='train', reliability_role='train', frame_id=i, pred=p, gt=gt,
                image_size=[1000, 500], original_simple_decision=d, size_risks=dict(full_simple=d['risks']['size']))


class ScaleEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.p = row()['pred']

    def test_canonical_swap_pi(self):
        q = deepcopy(self.p); q[2:4] = [20., 100.]; q[4] += math.pi/2
        self.assertEqual(m.descriptor(self.p, q, [1000, 500])[-2], 0.)
        q[4] += math.pi
        self.assertEqual(m.descriptor(self.p, q, [1000, 500])[-2], 0.)

    def test_disagreement_and_isotropic_unit_invariance(self):
        q = deepcopy(self.p); q[3] *= .8
        x = m.descriptor(self.p, q, [1000, 500])
        self.assertAlmostEqual(x[-2], -math.log(.8))
        a, b = deepcopy(self.p), deepcopy(q)
        a[:4] = [v*3 for v in a[:4]]; b[:4] = [v*3 for v in b[:4]]
        np.testing.assert_allclose(x, m.descriptor(a, b, [3000, 1500]), atol=1e-14)

    def test_aux_missing_is_feature_main_missing_not_quality(self):
        np.testing.assert_equal(m.descriptor(self.p, None, [1000, 500])[-2:], [0., 1.])
        self.assertIsNone(m.descriptor(None, self.p, [1000, 500]))

    def test_invalid_aux_is_not_silently_missing(self):
        q = deepcopy(self.p); q[3] = float('nan')
        with self.assertRaises(ValueError): m.descriptor(self.p, q, [1000, 500])

    def test_unbounded_ratios_avoid_overflow(self):
        a, b = deepcopy(self.p), deepcopy(self.p)
        a[2:4] = [1e300, 1e290]; b[2:4] = [1e-290, 1e-300]
        self.assertTrue(np.isfinite(m.descriptor(a, b, [1000, 500])).all())

    def test_same_solver_as_three_feature_control(self):
        x = np.array([[i/10, (i%3)/7, (i%5)/11] for i in range(30)])
        y = [i%4 == 0 for i in range(30)]
        a = m.fit(x, y); b = ab.fit(x, y, (0, 1, 2))
        np.testing.assert_allclose(a['weights'], b['weights'], atol=1e-14)
        np.testing.assert_allclose(m.risk(a, x), ab.risk(b, x), atol=1e-14)

    def test_constant_missing_feature_and_determinism(self):
        x = [[i/10, 0., 0., .1*(i%2), 0.] for i in range(30)]
        a = m.fit(x, [i%3 == 0 for i in range(30)])
        self.assertEqual(a, m.fit(x, [i%3 == 0 for i in range(30)]))
        self.assertEqual(a['weights'][-1], 0.)

    def test_single_class_and_invalid_model_rejected(self):
        with self.assertRaises(ValueError): m.fit(np.zeros((5, 5)), [0]*5)
        a = m.fit(np.zeros((6, 5)), [0, 0, 0, 1, 1, 1]); a['scale'][0] = 0
        with self.assertRaises(ValueError): m.risk(a, [0.]*5)

    def test_online_only_changes_size_and_never_mutates(self):
        r = row(); d = deepcopy(r['original_simple_decision'])
        model = m.fit(np.zeros((6, 5)), [0, 0, 0, 1, 1, 1])
        result = m.decide(model, 0., d, r['pred'], None, r['image_size'])
        self.assertFalse(result['size_accepted'])
        for k in ('center_accepted', 'angle_accepted', 'final_box_original'):
            self.assertEqual(result[k], d[k])
        self.assertEqual(result['risks']['angle'], d['risks']['angle'])
        self.assertEqual(d, r['original_simple_decision'])

    def test_missing_center_not_recovered_from_auxiliary(self):
        r = row(pred=False)
        self.assertEqual(m.decide({}, .5, r['original_simple_decision'], None, self.p, r['image_size']), r['original_simple_decision'])

    def test_whole_tie_global_cutoff_and_VAL_rejected(self):
        rows = [row(i) for i in range(20)]
        risks = {r['image']: .5 for r in rows}
        p = m.cutoff(rows, risks)
        self.assertEqual(p['risk_le'], .5)
        self.assertEqual(p['requirements']['all']['required'], 19)
        rows[0]['reliability_role'] = 'val'
        with self.assertRaises(ValueError): m.cutoff(rows, risks)

    def test_group_failure_cannot_be_hidden_by_pooled_gain(self):
        def entry(fa, control):
            a = dict(states=dict(CR=95, FR=5, FA=fa), runs=dict(correct_rejection=dict(longest=1)))
            return dict(actual=a, same_count_controls={n: dict(runs=dict(correct_rejection=dict(longest=1))) for n in m.CONTROLS},
                        control_tie_bounds={n: dict(bad_min_over_tie=control) for n in m.CONTROLS})
        self.assertTrue(m.gate({'all': entry(1, 2), 'domain:real': entry(1, 2)})['passed'])
        stats = {'all': entry(1, 2), 'domain:real': entry(1, 2), 'sequence:x': entry(3, 2)}
        self.assertFalse(m.gate(stats)['passed'])

    def test_tied_no_improvement_is_not_success(self):
        rows = [row(i, bad=i%4 == 0) for i in range(20)]
        scores = {n: {r['image']: .5 for r in rows} for n in ('scale_consistency',)+m.CONTROLS}
        stats = m.describe(rows, scores, .5)
        self.assertFalse(m.gate(stats)['passed'])
        self.assertEqual(stats['all']['actual']['states']['FA'], 5)

    def test_missing_excluded_and_FR_distinct_from_ED(self):
        rows = [row(0, bad=True), row(1), row(2, pred=False)]
        scores = {n: {r['image']: None if r['pred'] is None else .8 for r in rows} for n in ('scale_consistency',)+m.CONTROLS}
        a = m.describe(rows, scores, .5)['all']['actual']
        self.assertEqual(a['states'], dict(FA=0, FR=1, ED=1, CR=0, MISSING=1))


class PairTests(unittest.TestCase):
    def test_sealed_source_and_protocol_contract(self):
        p, identity = runner.checked_sources()
        self.assertEqual(p['features'], list(m.FEATURES))
        self.assertEqual(set(identity['sources']), set(runner.SOURCE_FILES))

    def fixture(self):
        p = row()
        paired = [dict(image=p['image'], sequence=p['sequence'], domain='real', frame_id=0, split='train',
                       sample_role='fit', scale=s, midpoint=deepcopy(p['pred']), gt=deepcopy(p['gt']),
                       candidate='POISON_FAILED_CANDIDATE_DO_NOT_USE') for s in (1., .5)]
        plan = dict(records=[dict(image=p['image'], sample_role='fit')])
        return [p], paired, plan

    def test_failed_geometry_candidate_is_ignored(self):
        a, b, c = self.fixture()
        self.assertEqual(runner.pair_train(a, b, c)[0]['auxiliary_pred_original'], a[0]['pred'])

    def test_standard_replay_is_exact_not_tolerant(self):
        a, b, c = self.fixture(); b[0]['midpoint'][3] += 1e-8
        with self.assertRaises(ValueError): runner.pair_train(a, b, c)

    def test_duplicate_or_missing_view_rejected(self):
        a, b, c = self.fixture()
        with self.assertRaises(ValueError): runner.pair_train(a, b+[b[0]], c)
        with self.assertRaises(ValueError): runner.pair_train(a, b[:1], c)

    def test_frame_and_GT_mismatch_rejected(self):
        a, b, c = self.fixture(); b[1]['frame_id'] = 1
        with self.assertRaises(ValueError): runner.pair_train(a, b, c)
        a, b, c = self.fixture(); b[1]['gt'][3] += .1
        with self.assertRaises(ValueError): runner.pair_train(a, b, c)

    def test_role_leakage_and_TEST_rejected(self):
        a, b, c = self.fixture(); b[1]['sample_role'] = 'probe'
        with self.assertRaises(ValueError): runner.pair_train(a, b, c)
        a, b, c = self.fixture(); a[0]['split'] = 'test'
        with self.assertRaises(ValueError): runner.pair_train(a, b, c)

    def test_output_scope_and_overwrite_guards(self):
        with self.assertRaises(ValueError): runner.checked_out(Path('/tmp/train_check'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            out = root/'work_dirs'/m.VERSION/'tests'/'train_check'
            out.mkdir(parents=True)
            with patch.object(runner, 'ROOT', root):
                with self.assertRaises(FileExistsError): runner.checked_out(out)


if __name__ == '__main__':
    unittest.main()
