"""Numeric and provenance tests; no detector, GPU, TEST data or fitting."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from crane_project.tools import analyze_port_reliability_tradeoff_v1 as entry
from crane_project.utils import port_reliability_tradeoff_v1 as trade
from crane_project.utils import port_reliability_separability_v1 as sep
from crane_project.utils import port_simple_component_reliability_v1 as simple


def policy():
    model = dict(feature_names=list(simple.FEATURES), converged=True,
        weights=[0., -1., .2, .1], mean=[0., 0., 0.], scale=[1., 1., 1.])
    return dict(protocol=simple.VERSION, feature_names=list(simple.FEATURES),
        center_policy='retain_valid_B_output_no_extra_rejection',
        models=dict(size=deepcopy(model), angle=deepcopy(model)),
        cutoffs={m: {c: dict(risk_le=.5) for c in ('size', 'angle')} for m in sep.METHODS})


def rows():
    out = []
    for seq, domain, scores in (('real_a', 'real', [.9, .8, .3, None]),
                                ('sim_b', 'sim', [.8, .3, .3])):
        for i, score in enumerate(scores):
            gt = [100., 80., 40., 20., .1]
            pred = gt+[score] if score is not None else None
            if i == 1 and pred is not None:
                pred[3] = 15.
            out.append(dict(image=seq+'_%05d' % i, sequence=seq, domain=domain,
                split='val', frame_id=i, gt=gt, pred=pred, image_size=[200, 160],
                angle_axis_well_defined=True, train_angle_eligible=False))
    return out


class TradeoffTests(unittest.TestCase):
    def test_decimal_ceiling_has_no_float_off_by_one(self):
        self.assertEqual(trade.required_good(140, .95), 133)
        self.assertEqual(trade.required_good(663, .95), 630)
        self.assertEqual(trade.required_good(63, .95), 60)

    def test_retention_point_keeps_whole_ties_not_best_risk_interval(self):
        pts = sep.acceptance_curve([.1,.2,.2,.9], [0,1,0,0], list('abcd'), [.1,.2,.2,.9], 5, 5)
        self.assertEqual(trade.retention_point(pts, .5)['accepted_assessed_outputs'], 3)
        self.assertEqual(trade.retention_point(pts, .95)['accepted_assessed_outputs'], 4)
        self.assertEqual(trade.point_at(pts, .19)['accepted_assessed_outputs'], 1)

    def test_empty_acceptance_risk_and_no_good_target_are_unavailable(self):
        pts = sep.acceptance_curve([.5], [1], ['a'], [.5], 2, 2)
        self.assertIsNone(trade.retention_point(pts, .95))
        self.assertIsNone(trade.point_at(pts, None)['error_fraction_on_accepted'])
        self.assertEqual(trade.point_at(pts, 0.)['accepted_assessed_outputs'], 0)

    def test_global_points_use_one_threshold_and_all_group_constraints(self):
        report, _, prepared = trade.analyze(rows(), policy(), 'val', [.5,.95])
        for a in report['retention_assessments']:
            t = a['global_diagnostic_risk_le']
            for g, p in a['groups'].items():
                values = prepared if g == 'all' else [r for r in prepared if r[g.split(':')[0]] == g.split(':')[1]]
                self.assertEqual(p['global_diagnostic_risk_le'], t)
                accepted = [r for r in values if r['pred'] is not None and
                            r['methods'][a['method']]['risks'][a['component']] <= t]
                self.assertEqual(len(accepted), p['accepted_assessed_outputs'])
                if a['constraint_scope'] == 'all_domains_and_videos':
                    self.assertTrue(p['retention_constraint_met'])
            self.assertFalse(a['deployable'])

    def test_missing_outputs_count_as_unavailable_and_gaps_reset(self):
        prepared = sep.prepare_rows(rows(), policy())[:4]
        prepared[1]['frame_id'] = 7
        prepared[2]['frame_id'] = 8
        prepared[3]['frame_id'] = 9
        p = trade.continuity(prepared, 'simple', 'size', None)
        self.assertEqual(p['longest_flag_unavailable_observed_consecutive_frames'], 3)
        self.assertEqual(p['adjacent_observed_pairs'], 2)
        p = trade.continuity(prepared, 'simple', 'size', 1.)
        self.assertEqual(p['online_flag_accepted_all_frames'], 3)
        self.assertEqual(p['longest_flag_unavailable_observed_consecutive_frames'], 1)
        self.assertEqual(p['flag_switches_on_adjacent_pairs'], 1)

    def test_inputs_gt_boxes_scores_policy_not_mutated(self):
        r, p = rows(), policy(); before = simple.fingerprint([r,p])
        report, curves, prepared = trade.analyze(r, p, 'val', [.95])
        self.assertEqual(before, simple.fingerprint([r,p]))
        for a,b in zip(r,prepared):
            self.assertEqual(a['pred'], b['methods']['simple']['raw_b_output'])
        self.assertTrue(report['inputs_policy_unchanged'])

    def test_TEST_and_role_mix_rejected(self):
        r = rows(); r[0]['split'] = 'test'
        with self.assertRaises(ValueError): trade.analyze(r, policy(), 'val', [.95])
        r = rows(); r[0]['split'] = 'train'
        with self.assertRaises(ValueError): trade.analyze(r, policy(), 'val', [.95])

    def test_duplicate_sequence_frame_rejected_before_analysis(self):
        r=rows(); r[1]['frame_id']=r[0]['frame_id']
        with self.assertRaisesRegex(ValueError, 'sequence/frame'): entry.validate_rows(r)

    def test_online_angle_flags_and_GT_qualification_separate(self):
        r=rows(); r[0]['gt'][2]=22.; r[0]['angle_axis_well_defined']=False
        report, _, _ = trade.analyze(r, policy(), 'val', [.95])
        a=next(a for a in report['retention_assessments'] if a['component']=='angle' and a['method']=='score_only')
        p=a['groups']['all']
        self.assertEqual(report['strata']['all']['components']['angle']['eligible_frames'],6)
        self.assertGreater(p['continuity']['online_flag_accepted_all_frames'],p['accepted_assessed_outputs'])

    def test_VAL_replay_detects_changed_box_flag_and_risk(self):
        prepared=sep.prepare_rows(rows(),policy())
        saved=[dict(image=r['image'],methods={m:dict((('final_box_original' if k=='raw_b_output' else k),deepcopy(v)) for k,v in d.items()) for m,d in r['methods'].items()}) for r in prepared]
        self.assertTrue(trade.replay_val(prepared,saved)['flags_boxes_exact'])
        for kind in ('box','flag','risk'):
            altered=deepcopy(saved)
            if kind=='box': altered[0]['methods']['simple']['final_box_original'][2] += .01
            if kind=='flag': altered[0]['methods']['simple']['size_accepted'] = False
            if kind=='risk': altered[0]['methods']['simple']['risks']['size'] += 1e-6
            with self.assertRaises(ValueError): trade.replay_val(prepared,altered)

    def test_frozen_SHA_source_contract_checked_without_model_data(self):
        p,m=entry.checked_sources()
        self.assertEqual(p['primary_good_retention'],.95)
        self.assertEqual(p['targets'],[.90,.95,.99])
        self.assertFalse(p['test_read'])
        self.assertTrue(m['sources'])
        with tempfile.TemporaryDirectory() as d:
            fake=Path(d)/'fit'; fake.mkdir();(fake/'policy.json').write_text('{}')
            with self.assertRaises((ValueError,FileNotFoundError)): entry.checked_inputs(Path(d),p)


if __name__ == '__main__':
    unittest.main()
