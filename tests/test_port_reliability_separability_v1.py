"""Independent CPU metric/contract tests; no fitting, GPU or TEST records."""
from argparse import Namespace
from copy import deepcopy
import ast
import inspect
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from crane_project.tools import diagnose_port_reliability_separability_v1 as entry
from crane_project.utils import port_reliability_separability_v1 as diagnostic
from crane_project.utils import port_simple_component_reliability_v1 as simple


def policy():
    model = dict(feature_names=list(simple.FEATURES), converged=True,
        weights=[0., -1., .2, .1], mean=[0., 0., 0.], scale=[1., 1., 1.])
    return dict(protocol=simple.VERSION, feature_names=list(simple.FEATURES),
        center_policy='retain_valid_B_output_no_extra_rejection',
        models=dict(size=deepcopy(model), angle=deepcopy(model)),
        cutoffs={m: {c: dict(risk_le=.5) for c in ('size', 'angle')} for m in diagnostic.METHODS})


def rows(role='val'):
    result = []
    for i in range(5):
        gt = [100., 80., 40., 20., .1]
        pred = gt+[.8 if i < 3 else .2]
        if i == 1:
            pred[0] += 15.; pred[2] = 25.; pred[4] += math.radians(8)
        if i == 2:
            gt[2] = 22.; pred[2] = 22.; pred[4] += math.radians(8)
        if i == 4:
            pred = None
        result.append(dict(image='real_seq01_%05d' % i, sequence='real_seq01', domain='real',
            split=role, frame_id=i, gt=gt, pred=pred, image_size=[200, 160],
            angle_axis_well_defined=i != 2, train_angle_eligible=role == 'train'))
    return result


class SeparabilityTests(unittest.TestCase):
    def test_auroc_matches_independent_pairwise_tie_credit(self):
        risks = [.1, .3, .3, .6, .9, .9]
        labels = [0, 1, 0, 0, 1, 1]
        positive = [r for r, y in zip(risks, labels) if y]
        negative = [r for r, y in zip(risks, labels) if not y]
        expected = sum(float(p > n)+.5*float(p == n) for p in positive for n in negative)/(len(positive)*len(negative))
        self.assertAlmostEqual(diagnostic.rank_metrics(risks, labels)['error_auroc'], expected)

    def test_average_precision_groups_ties_and_is_not_trapezoid_PR(self):
        metrics = diagnostic.rank_metrics([.9, .9, .2], [1, 0, 1])
        self.assertAlmostEqual(metrics['error_average_precision'], .5*.5+.5*2/3)
        same = diagnostic.rank_metrics([.5]*4, [0, 0, 0, 1])
        self.assertEqual(same['error_auroc'], .5)
        self.assertEqual(same['error_average_precision'], .25)

    def test_rank_invariant_to_tie_order_and_monotone_calibration(self):
        r = np.array([.1, .7, .7, .9]); y = np.array([0, 1, 0, 1])
        base = diagnostic.rank_metrics(r, y)
        other = diagnostic.rank_metrics(r[[0, 2, 1, 3]], y[[0, 2, 1, 3]])
        calibrated = diagnostic.rank_metrics(r*r, y)
        for key in ('error_auroc', 'error_average_precision'):
            self.assertEqual(base[key], other[key])
            self.assertEqual(base[key], calibrated[key])

    def test_empty_and_single_class_groups_are_explicit(self):
        for risks, bad in (([], []), ([.2, .7], [0, 0]), ([.2, .7], [1, 1])):
            value = diagnostic.rank_metrics(risks, bad)
            self.assertIsNone(value['error_auroc'])
            self.assertFalse(value['both_classes'])
            self.assertEqual(value['error_average_precision'], 1. if bad and sum(bad) else None)

    def test_invalid_risk_label_and_denominator_rejected(self):
        for risks, labels in (([float('nan')], [1]), ([1.1], [1]), ([.5], [2]), ([[.5]], [1])):
            with self.subTest(risks=risks), self.assertRaises(ValueError):
                diagnostic.rank_metrics(risks, labels)
        with self.assertRaises(ValueError):
            diagnostic.confusion([1, 0], [True, False], 1, 1)

    def test_state_confusion_and_coverage_denominators(self):
        # Good accepted=2, bad accepted=1, good rejected=1, bad rejected=2.
        s = diagnostic.confusion([0, 0, 1, 0, 1, 1], [1, 1, 1, 0, 0, 0], 10, 8)
        self.assertEqual((s['correct_accepted'], s['incorrect_accepted'], s['correct_rejected'], s['incorrect_rejected']), (2, 1, 1, 2))
        self.assertEqual(s['state_errors'], 2)
        self.assertAlmostEqual(s['state_accuracy_on_assessed_outputs'], 4/6)
        self.assertEqual(s['assessed_acceptance_coverage_on_all_frames'], .3)
        self.assertEqual(s['assessed_acceptance_coverage_on_eligible_frames'], 3/8)
        self.assertEqual(s['acceptance_coverage_on_assessed_outputs'], .5)
        self.assertEqual(s['correct_accepted_coverage_on_all_frames'], .2)

    def test_whole_tie_curve_matches_independent_boolean_counts(self):
        risks = np.array([.1, .2, .2, .9]); bad = np.array([0, 1, 0, 1], dtype=bool)
        points = diagnostic.acceptance_curve(risks, bad, list('abcd'), [.1, .3, .2, .8], 6, 5)
        self.assertEqual([p['accepted_assessed_outputs'] for p in points], [0, 1, 3, 4])
        for p in points:
            kept = np.zeros(4, dtype=bool) if p['risk_le'] is None else risks <= p['risk_le']
            self.assertEqual(p['incorrect_accepted'], int(np.sum(kept & bad)))
            self.assertEqual(p['correct_rejected'], int(np.sum(~kept & ~bad)))
            self.assertEqual(p['incorrect_rejected'], int(np.sum(~kept & bad)))
        self.assertIsNone(points[0]['correctness_on_accepted'])
        self.assertEqual(points[-1]['error_detection_rate'], 0.)

    def test_same_count_score_uses_image_not_GT_and_reports_cut_tie_bounds(self):
        s = diagnostic.same_count_reference([.2, .2, .2, .9], [1, 0, 1, 1], ['c', 'a', 'b', 'd'], [0, 1, 2, 4])
        self.assertEqual(s[1]['incorrect_accepted'], 0)  # image a wins the tie.
        self.assertEqual(s[1]['bad_min_over_tie'], 0)
        self.assertEqual(s[1]['bad_max_over_tie'], 1)
        self.assertAlmostEqual(s[1]['bad_expected_random_tie'], 2/3)
        self.assertEqual(s[2]['boundary_tie_size'], 3)
        self.assertEqual(s[2]['boundary_tie_selected'], 2)
        self.assertEqual((s[2]['bad_min_over_tie'], s[2]['bad_max_over_tie']), (1, 2))
        with self.assertRaises(ValueError):
            diagnostic.same_count_reference([.2, .2], [0, 1], ['a', 'a'], [1])

    def test_missing_and_direction_unassessed_are_not_correct_rejections(self):
        s, _, _ = diagnostic.analyze(rows(), policy(), 'val')
        all_rows = s['all']; angle = all_rows['components']['angle']
        self.assertEqual(all_rows['output_frames'], 4)
        self.assertEqual(all_rows['center_hits'], 3)
        self.assertEqual(all_rows['center_hit_rate_on_outputs'], .75)
        self.assertEqual(all_rows['all_frame_center_correct_coverage'], .6)
        self.assertEqual(angle['eligible_frames'], 4)
        self.assertEqual(angle['assessed_outputs'], 3)
        self.assertEqual(angle['missing_eligible_outputs'], 1)
        fixed = angle['methods']['simple']['fixed_workpoint']
        self.assertEqual(fixed['unassessed_accepted'], 1)
        self.assertEqual(fixed['flag_accepted_all_frames'], 3)
        self.assertEqual(fixed['accepted_assessed_outputs'], 2)

    def test_TRAIN_direction_fit_labels_and_primary_evaluation_are_separate(self):
        s, _, _ = diagnostic.analyze(rows('train'), policy(), 'train')
        components = s['all']['components']
        self.assertEqual(components['angle']['assessed_outputs'], 3)
        self.assertEqual(components['angle']['bad_outputs'], 1)
        self.assertEqual(components['angle_training_qualification']['assessed_outputs'], 4)
        self.assertEqual(components['angle_training_qualification']['bad_outputs'], 2)

    def test_only_original_global_workpoint_used_in_each_group(self):
        q = policy(); s, _, _ = diagnostic.analyze(rows(), q, 'val')
        for group in s.values():
            for component in ('size', 'angle'):
                for method in diagnostic.METHODS:
                    self.assertEqual(group['components'][component]['methods'][method]['fixed_workpoint']['frozen_global_risk_le'], q['cutoffs'][method][component]['risk_le'])

    def test_angle_changes_risk_neither_inputs_nor_policy_are_mutated(self):
        original = rows(); q = policy(); before = deepcopy([original, q])
        prepared = diagnostic.prepare_rows(original, q)
        changed = deepcopy(original); changed[0]['pred'][4] += .2
        other = diagnostic.prepare_rows(changed, q)
        self.assertEqual(prepared[0]['methods']['simple']['risks'], other[0]['methods']['simple']['risks'])
        diagnostic.analyze(original, q, 'val')
        self.assertEqual([original, q], before)

    def test_TEST_and_mixed_roles_or_invalid_qualification_rejected(self):
        for kind in ('test', 'mixed', 'qualification', 'bool', 'image_size'):
            v = rows()
            if kind == 'test':
                for r in v: r['split'] = 'test'
            if kind == 'mixed': v[0]['split'] = 'train'
            if kind == 'qualification': v[2]['angle_axis_well_defined'] = True
            if kind == 'bool': v[0]['angle_axis_well_defined'] = 1
            if kind == 'image_size': v[-1]['image_size'] = [0, 160]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                diagnostic.analyze(v, policy(), 'val')

    def test_entry_and_metrics_do_not_import_detector_frameworks_or_test_entry(self):
        for module in (entry, diagnostic):
            tree = ast.parse(inspect.getsource(module))
            names = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
            names += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
            self.assertFalse(any(n and n.split('.')[0] in ('torch', 'mmcv', 'cv2', 'mmrotate') for n in names))
            self.assertNotIn('eval_port_simple_reliability_v1_test', inspect.getsource(module))

    def test_policy_wrong_bytes_and_failure_bundle_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory); p = d/'policy.json'; p.write_text('{}')
            contract = dict(reviewed_policy_files={'policy.json': entry.parent.sha(p)})
            with patch.object(entry.parent, 'load_policy', return_value=policy()):
                entry.checked_policy(p, contract, {}, {'parent': {}})
                p.write_text('{"changed":true}')
                with self.assertRaisesRegex(ValueError, 'exact reviewed'):
                    entry.checked_policy(p, contract, {}, {'parent': {}})
                (d/'failure.json').write_text('{}')
                with self.assertRaisesRegex(ValueError, 'completed original'):
                    entry.checked_policy(p, contract, {}, {'parent': {}})

    def test_saved_VAL_replay_requires_exact_flags_and_boxes(self):
        prepared = diagnostic.prepare_rows(rows(), policy())
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'policy.json'
            saved = [dict(image=r['image'], methods=deepcopy(r['methods'])) for r in prepared]
            f = p.parent/'val_decisions.jsonl'
            f.write_text('\n'.join(json.dumps(r) for r in saved))
            result = entry.replay_saved_val(prepared, p, 1e-12)
            self.assertTrue(result['original_boxes_and_all_flags_exact'])
            saved[0]['methods']['simple']['size_accepted'] = False
            f.write_text('\n'.join(json.dumps(r) for r in saved))
            with self.assertRaisesRegex(ValueError, 'box/flag'):
                entry.replay_saved_val(prepared, p, 1e-12)

    def test_frozen_source_contract_and_Python38_syntax(self):
        contract, _, _ = entry.checked_sources()
        self.assertFalse(contract['test_read'])
        for module in (entry, diagnostic):
            ast.parse(inspect.getsource(module), feature_version=(3, 8))

    def test_full_synthetic_run_outputs_sealed_report_without_fit_or_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory); out = d/'out'; out.mkdir()
            p = d/'policy.json'; q = policy(); p.write_text(json.dumps(q))
            prepared = diagnostic.prepare_rows(rows(), q)
            (d/'val_decisions.jsonl').write_text('\n'.join(json.dumps(dict(image=r['image'], methods=r['methods'])) for r in prepared))
            entry.parent.write_new(out/'input_check.json', {'synthetic': True})
            args = Namespace(out_dir=out, policy=p)
            sources, proof = {'parent': {}}, {'synthetic': True}
            contract = dict(risk_replay_abs_tolerance=1e-12)
            with patch.object(entry, 'inputs', return_value=([], [], {}, proof)), patch.object(entry, 'contract_original', return_value={}), patch.object(entry, 'checked_sources', return_value=(contract, {}, sources)), patch.object(simple, 'fit_linear_risk', side_effect=AssertionError('No fitting')), patch.object(entry.parent, 'build_online', side_effect=AssertionError('No inference')):
                # Include sim records to exercise the display, without fitting.
                train, val = rows('train'), rows()
                for dataset in (train, val):
                    sim = deepcopy(dataset[0]); sim.update(image='sim_seq08_00000', sequence='sim_seq08', domain='sim')
                    if dataset is train: sim['split'] = 'train_sim'
                    dataset.append(sim)
                prepared = diagnostic.prepare_rows(val, q)
                (d/'val_decisions.jsonl').write_text('\n'.join(json.dumps(dict(image=r['image'], methods=r['methods'])) for r in prepared))
                entry.run(args, train, val, q, proof, contract, sources)
            completion = json.loads((out/'completion.json').read_text())
            self.assertFalse(completion['new_policy_created'])
            self.assertFalse(completion['test_read'])
            self.assertFalse((out/'policy.json').exists())
            for name, digest in completion['files_sha256'].items():
                self.assertEqual(entry.parent.sha(out/name), digest)
            self.assertIn('curve_points.csv', completion['files_sha256'])

    def test_existing_output_directory_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            arguments = ['entry', '--mode', 'run', '--out-dir', directory]
            with patch('sys.argv', arguments), self.assertRaises(FileExistsError):
                entry.main()


if __name__ == '__main__':
    unittest.main()
