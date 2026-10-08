from copy import deepcopy
import inspect
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from crane_project.utils import port_reliability_scale_asymmetry_v1 as m
from crane_project.utils import port_reliability_scale_consistency_v1 as old
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.tools import run_port_reliability_scale_asymmetry_v1 as runner


def row(i=0, bad=False, aux_bad=False, missing=False, aux_missing=False):
    gt = [20., 30., 100., 20., .2]
    p = None if missing else [20., 30., 100., 25. if bad else 20., .2, .8]
    aux = None if aux_missing else [20., 30., 100., 25. if aux_bad else 20., .2, .7]
    d = dict(center_accepted=not missing, size_accepted=not missing, angle_accepted=not missing,
             final_box_original=p, risks=dict(size=None if missing else .2, angle=None if missing else .3))
    return dict(image='real_seq01_'+str(i).zfill(3), sequence='real_seq01', domain='real',
        split='train', reliability_role='train', sample_role='fit', frame_id=i, pred=p, gt=gt,
        image_size=[1000, 500], original_simple_decision=d,
        auxiliary_pred_original=aux, size_risks=dict(full_simple=d['risks']['size']))


def neutral():
    return m.fit(np.zeros((6, 8)), [0, 0, 0, 1, 1, 1])


class DescriptorTests(unittest.TestCase):
    def test_exact_old_prefix_and_signed_axis_identity(self):
        r = row(); a = deepcopy(r['pred']); a[2] *= 1.2; a[3] *= .8
        x = m.descriptor(r['pred'], a, r['image_size'])
        np.testing.assert_equal(x[:5], old.descriptor(r['pred'], a, r['image_size']))
        self.assertAlmostEqual(x[5], math.log(1.2)); self.assertAlmostEqual(x[6], math.log(.8))
        self.assertAlmostEqual(x[3], -math.log(.8)); self.assertEqual(x[7], 0.)

    def test_swapped_width_height_and_pi_do_not_change_size(self):
        r = row(); a = deepcopy(r['auxiliary_pred_original'])
        a[2:4] = a[2:4][::-1]; a[4] += 1.5*math.pi
        np.testing.assert_equal(m.descriptor(r['pred'], a, r['image_size']),
            m.descriptor(r['pred'], r['auxiliary_pred_original'], r['image_size']))

    def test_isotropic_unit_invariance(self):
        r = row(); x = m.descriptor(r['pred'], r['auxiliary_pred_original'], r['image_size'])
        for p in (r['pred'], r['auxiliary_pred_original']): p[:4] = [v*3 for v in p[:4]]
        np.testing.assert_allclose(x, m.descriptor(r['pred'], r['auxiliary_pred_original'], [3000, 1500]), atol=1e-14)

    def test_score_delta_sign_and_clamp(self):
        r = row(); r['pred'][5] = .5; r['auxiliary_pred_original'][5] = 1.
        self.assertAlmostEqual(m.descriptor(r['pred'], r['auxiliary_pred_original'], r['image_size'])[7],
                               math.log((1-1e-6)/1e-6), places=8)
        r['auxiliary_pred_original'][5] = .1
        self.assertLess(m.descriptor(r['pred'], r['auxiliary_pred_original'], r['image_size'])[7], 0.)

    def test_missing_aux_keeps_main_with_zero_deltas(self):
        r = row(aux_missing=True)
        np.testing.assert_equal(m.descriptor(r['pred'], None, r['image_size'])[3:], [0, 1, 0, 0, 0])

    def test_no_main_no_quality_and_invalid_aux_rejected(self):
        r = row(missing=True)
        self.assertIsNone(m.descriptor(None, r['auxiliary_pred_original'], r['image_size']))
        r['auxiliary_pred_original'][3] = float('nan')
        with self.assertRaises(ValueError): m.descriptor(None, r['auxiliary_pred_original'], r['image_size'])

    def test_invalid_box_score_image_rejected(self):
        for field, value in ((3, -1), (5, .01), (4, float('inf'))):
            r = row(); r['auxiliary_pred_original'][field] = value
            with self.assertRaises(ValueError): m.descriptor(r['pred'], r['auxiliary_pred_original'], r['image_size'])
        r = row()
        with self.assertRaises(ValueError): m.descriptor(r['pred'], None, [0, 500])

    def test_extreme_edges_remain_finite(self):
        r = row(); r['pred'][2:4] = [1e300, 1e290]; r['auxiliary_pred_original'][2:4] = [1e-290, 1e-300]
        self.assertTrue(np.isfinite(m.descriptor(r['pred'], r['auxiliary_pred_original'], r['image_size'])).all())


class SolverAndOnlineTests(unittest.TestCase):
    def test_exact_eight_dimensions_and_two_classes(self):
        for x, y in ((np.zeros((6, 5)), [0, 1]*3), (np.zeros((6, 8)), [0]*6),
                     (np.ones((6, 8))*float('nan'), [0, 1]*3)):
            with self.assertRaises(ValueError): m.fit(x, y)

    def test_zero_init_constant_features_and_determinism(self):
        a = neutral(); self.assertEqual(a, neutral())
        self.assertEqual(a['weights'], [0.]*9); self.assertEqual(a['scale'], [1.]*8)
        self.assertEqual(a['initialization'], 'all_zero'); self.assertEqual(a['settings'], old.FIT)

    def test_objective_gradient_and_hessian_finite_difference(self):
        rng = np.random.RandomState(8); design = rng.normal(size=(24, 9)); design[:, 0] = 1.
        y = np.array([0., 1.]*12); sw = np.ones(24)/24; w = rng.normal(size=9)*.1
        loss, grad, h = simple.logistic_objective(w, design, y, sw, .1)
        eps = 1e-5
        for j in range(9):
            step = np.zeros(9); step[j] = eps
            a = simple.logistic_objective(w+step, design, y, sw, .1)
            b = simple.logistic_objective(w-step, design, y, sw, .1)
            self.assertAlmostEqual(grad[j], (a[0]-b[0])/(2*eps), places=8)
            np.testing.assert_allclose(h[:, j], (a[1]-b[1])/(2*eps), atol=1e-8)
        self.assertGreater(loss, simple.logistic_objective(w-np.linalg.solve(h, grad), design, y, sw, .1)[0])

    def test_models_and_features_wrong_schema_rejected(self):
        for change in ('names', 'scale', 'settings', 'weights'):
            a = neutral()
            if change == 'names': a['feature_names'] = list(old.FEATURES)
            elif change == 'scale': a['scale'][0] = 0.
            elif change == 'settings': a['settings']['l2'] = .2
            else: a['weights'][0] = float('nan')
            with self.assertRaises(ValueError): m.risk(a, [0.]*8)
        with self.assertRaises(ValueError): m.risk(neutral(), [0.]*5)

    def test_online_only_changes_size_without_mutation(self):
        r = row(); before = deepcopy(r)
        d = m.decide(neutral(), 0., r['original_simple_decision'], r['pred'], r['auxiliary_pred_original'], r['image_size'])
        expected = deepcopy(r['original_simple_decision']); expected['size_accepted'] = False; expected['risks']['size'] = .5
        self.assertEqual(d, expected); self.assertEqual(r, before)
        self.assertEqual(tuple(inspect.signature(m.descriptor).parameters), ('main', 'auxiliary', 'image_size'))
        self.assertEqual(tuple(inspect.signature(m.decide).parameters),
            ('model', 'cutoff', 'original_decision', 'main', 'auxiliary', 'image_size'))

    def test_missing_main_cannot_be_recovered(self):
        r = row(missing=True)
        self.assertEqual(m.decide({}, .5, r['original_simple_decision'], None, r['auxiliary_pred_original'], r['image_size']), r['original_simple_decision'])
        r['original_simple_decision']['size_accepted'] = True
        with self.assertRaises(ValueError): m.decide({}, .5, r['original_simple_decision'], None, None, r['image_size'])

    def test_invalid_cutoff_and_original_identity_rejected(self):
        r = row()
        for cutoff in (-1., 2., float('nan')):
            with self.assertRaises(ValueError): m.decide(neutral(), cutoff, r['original_simple_decision'], r['pred'], None, r['image_size'])
        r['original_simple_decision']['center_accepted'] = False
        with self.assertRaises(ValueError): m.decide(neutral(), .5, r['original_simple_decision'], r['pred'], None, r['image_size'])


class OfflineContractTests(unittest.TestCase):
    def test_combination_denominators_missing_and_four_states(self):
        rows = [row(0), row(1, aux_bad=True), row(2, bad=True), row(3, bad=True, aux_bad=True), row(4, aux_missing=True), row(5, missing=True)]
        s = m.combination_summary(rows, {rows[0]['image'], rows[2]['image']})
        self.assertEqual(s['main_good_aux_bad']['states']['FR'], 1)
        self.assertEqual(s['main_bad_aux_good']['states']['FA'], 1)
        self.assertEqual(s['main_bad_aux_bad']['states']['ED'], 1)
        self.assertEqual(s['main_good_aux_good']['states']['CR'], 1)
        self.assertIsNone(s['main_bad_aux_missing']['false_acceptance_rate'])
        self.assertEqual(sum(v['support'] for v in s.values()), 5)
        with self.assertRaises(ValueError): m.combination_summary(rows, {rows[-1]['image']})

    def test_same_count_selection_is_whole_group_before_GT_combination(self):
        rows = [row(i, bad=i==2, aux_bad=i==1) for i in range(5)]
        scores = {name: {r['image']: i/10 for i, r in enumerate(rows)} for name in (m.ARM,)+m.CONTROLS}
        scores['scale_consistency'] = {r['image']: (4-i)/10 for i, r in enumerate(rows)}
        a = m.describe(rows, scores, .1)['all']
        self.assertEqual(a['actual']['accepted_outputs'], 2)
        s = a['combinations']['same_count_controls']['scale_consistency']
        self.assertEqual(s['main_good_aux_bad']['states']['FR'], 1)
        self.assertEqual(s['main_bad_aux_good']['states']['ED'], 1)
        self.assertEqual(s['main_good_aux_good']['states']['CR'], 2)

    def test_tie_order_is_image_only_not_label(self):
        rows = [row(0, bad=True), row(1), row(2)]
        scores = {name: {r['image']: .5 for r in rows} for name in (m.ARM,)+m.CONTROLS}
        scores[m.ARM][rows[0]['image']] = .1
        a = m.describe(rows, scores, .1)['all']
        self.assertEqual(a['same_count_controls']['scale_consistency']['states']['FA'], 1)
        self.assertEqual(a['control_tie_bounds']['scale_consistency']['bad_min_over_tie'], 0)
        self.assertEqual(a['combinations']['same_count_controls']['scale_consistency']['main_bad_aux_good']['states']['FA'], 1)

    def test_cutoff_whole_tie_and_no_VAL(self):
        rows = [row(i) for i in range(20)]; risks = {r['image']: .5 for r in rows}
        self.assertEqual(m.cutoff(rows, risks)['risk_le'], .5)
        rows[0]['reliability_role'] = 'val'
        with self.assertRaises(ValueError): m.cutoff(rows, risks)

    def entry(self):
        a = dict(states=dict(CR=95, FR=5, FA=1), runs=dict(correct_rejection=dict(longest=1)))
        combo = {key: dict(states=dict(FA=0, FR=0)) for key in m.COMBINATIONS}
        return dict(actual=a, same_count_controls={n: dict(runs=dict(correct_rejection=dict(longest=1))) for n in m.CONTROLS},
            control_tie_bounds={n: dict(bad_min_over_tie=2) for n in m.CONTROLS},
            combinations=dict(actual=deepcopy(combo), same_count_controls={n: deepcopy(combo) for n in m.CONTROLS}))

    def test_joint_protections_cannot_be_hidden_by_overall_improvement(self):
        for key, state in (('main_good_aux_bad', 'FR'), ('main_bad_aux_good', 'FA'), ('main_bad_aux_bad', 'FA')):
            a = self.entry(); self.assertTrue(m.gate({'all': a})['passed'])
            a['combinations']['actual'][key]['states'][state] = 1
            failures = m.gate({'all': a})['failures']
            self.assertEqual(len(failures), 1); self.assertEqual(failures[0]['combination'], key)

    def test_FR_retention_and_per_video_gates_remain_hard(self):
        a = self.entry(); a['actual']['states']['CR'] = 90
        self.assertIn('correct_retention', [f['check'] for f in m.gate({'all': a})['failures']])
        a = self.entry(); a['actual']['runs']['correct_rejection']['longest'] = 2
        self.assertEqual(len(m.gate({'all': a})['failures']), 4)
        a = self.entry(); a['actual']['states']['FA'] = 3
        self.assertFalse(m.gate({'sequence:x': a})['passed'])

    def test_missing_output_excluded_from_confusion(self):
        rows = [row(0, bad=True), row(1), row(2, missing=True)]
        scores = {name: {r['image']: None if r['pred'] is None else .8 for r in rows} for name in (m.ARM,)+m.CONTROLS}
        a = m.describe(rows, scores, .5)['all']['actual']
        self.assertEqual(a['states'], dict(FA=0, FR=1, ED=1, CR=0, MISSING=1))

    def test_sealed_source_protocol_and_output_guards(self):
        p, s = runner.checked_sources(); self.assertEqual(p['features'], list(m.FEATURES))
        self.assertEqual(set(s['sources']), set(runner.SOURCE_FILES))
        with self.assertRaises(ValueError): runner.checked_out(Path('/tmp/train_check'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); out = root/'work_dirs'/m.VERSION/'tests'/'train_check'; out.mkdir(parents=True)
            with patch.object(runner, 'ROOT', root):
                with self.assertRaises(FileExistsError): runner.checked_out(out)

    def test_source_and_control_tampering_fail_before_fit(self):
        with patch.object(runner, 'sha', return_value='tampered'):
            with self.assertRaises(ValueError): runner.checked_sources()
            with self.assertRaises(ValueError): runner.checked_control_artifacts()

    def test_entry_has_no_VAL_TEST_inference_route(self):
        source = Path(runner.__file__).read_text()
        self.assertNotIn('init_detector', source); self.assertNotIn('inference_detector', source)
        self.assertNotIn('method.fit(features[:', source)
        with patch.object(runner, 'preflight', side_effect=ValueError('identity')), patch.object(m, 'fit') as fit:
            with patch.object(runner, 'checked_out', return_value=Path('/tmp/train_check')):
                with self.assertRaises(ValueError): runner.run(Path('/tmp/train_check'))
            fit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
