import copy
import unittest
import numpy as np
from crane_project.utils import port_simple_pilot_increment_v1 as c


def row(name, wrong=False, missing=False):
    return dict(image=name, sequence='real_seq11', domain='real', frame_id=int(name),
        split='extra_source_real_seq11_pilot_k1p9_v2_train', reliability_role='train',
        image_size=[1920, 1080], gt=[0, 0, 10, 5, 0],
        pred=None if missing else [0, 0, 12 if wrong else 10, 5, 0, .5])


class IncrementTests(unittest.TestCase):
    def test_missing_not_fitted_and_both_classes_retained(self):
        rows = [row('0'), row('1', True), row('2', missing=True)]
        fitted = c.fit(rows, dict(l2=.1, max_iterations=100, gradient_tolerance=1e-8))
        self.assertEqual((fitted['train_rows'], fitted['good_rows'], fitted['bad_rows']), (2, 1, 1))
        with self.assertRaises(ValueError): c.fit(rows+rows, {})
        rows[0]['reliability_role'] = 'val'
        with self.assertRaises(ValueError): c.fit(rows, {})

    def test_scores_do_not_read_gt_or_video(self):
        model = dict(weights=[.1, .2, .3, .4], mean=[0]*3, scale=[1]*3)
        a = row('0'); a['audit_risks'] = {}; b = copy.deepcopy(a)
        b['gt'] = [3, 4, 80, 2, 1]; b['sequence'] = 'another_video'
        c.add_scores([a, b], {'test': model})
        self.assertEqual(a['audit_risks'], b['audit_risks'])
        self.assertEqual(a['pred'], b['pred'])

    def test_primary_whole_tie(self):
        rows = [row('0'), row('1', True), row('2')]
        for r in rows: r['audit_risks'] = {'test': .5}
        p = c.Ranking(rows, 'test').same_cr(1)
        self.assertEqual((p['CR'], p['FA'], p['accepted']), (2, 1, 3))
        self.assertFalse(p['exact_CR'])

    def test_continuity_missing_and_original_gaps(self):
        rows = [row('0'), row('2'), row('3', missing=True), row('4', True)]
        for r in rows: r['audit_risks'] = {'test': None if r['pred'] is None else .9}
        d = c.at_cut(rows, 'test', .1)['all']
        self.assertEqual(d['states'], dict(FA=0, FR=2, ED=1, CR=0, MISSING=1))
        self.assertEqual(d['longest_correct_FR'], 1)
        self.assertEqual(d['longest_FR_observations'], 2)
        self.assertEqual(d['longest_without_CR_observations'], 4)

    def test_gate_does_not_reward_lost_corrects(self):
        def metric(fa, cr=653, real=189):
            return {'at_frozen_VAL_cut': {'all': {'states': {'FA': fa, 'CR': cr}},
                    'domain:real': {'states': {'CR': real}}}}
        v = {'formal': metric(190), 'augmented': metric(189)}
        self.assertTrue(c.decision(v)['passed'])
        for change in ((190, 653, 189), (150, 652, 189), (150, 653, 188)):
            v['augmented'] = metric(*change)
            self.assertFalse(c.decision(v)['passed'])

    def test_role_and_prediction_preservation(self):
        r = row('0'); r['decision'] = dict(final_box_original=r['pred'], center_accepted=True)
        c.checked_rows([r], 'PILOT', 1)
        r['split'] = 'val'
        with self.assertRaises(ValueError): c.checked_rows([r], 'PILOT', 1)


if __name__ == '__main__': unittest.main()
