"""CPU unittest checks for frozen-cache closure, with no new fitting."""
from copy import deepcopy
import inspect
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from crane_project.tools import eval_port_simple_reliability_v1_test as entry
from crane_project.utils import port_simple_component_reliability_v1 as simple


def policy():
    # Explicit frozen fixture, not a fit of TEST or a reproduction of the fitter.
    model = dict(feature_names=list(simple.FEATURES), converged=True,
                 weights=[0., -1., 0., 0.], mean=[0., 0., 0.], scale=[1., 1., 1.])
    return dict(protocol=simple.VERSION, feature_names=list(simple.FEATURES),
        center_policy='retain_valid_B_output_no_extra_rejection',
        models=dict(size=deepcopy(model), angle=deepcopy(model)),
        cutoffs={m: {c: {'risk_le': .5} for c in ('size', 'angle')}
                 for m in ('simple', 'score_only')})


def rows():
    values = []
    for i in range(4):
        gt = [100., 80., 40., 20., .1]
        pred = gt+[.8 if i < 2 else .2]
        if i == 1:
            pred[2] = 20.; pred[4] += math.radians(8)
        if i == 2:
            pred[0] += 15.
        if i == 3:
            pred = None
        values.append(dict(image='real_seq03_%05d' % i, sequence='real_seq03',
            frame_id=i, domain='real', split='test', gt=gt, pred=pred,
            image_size=[200, 160], angle_axis_well_defined=True,
            image_sha256='a'*64, annotation_sha256='b'*64))
    return values


class ClosureTests(unittest.TestCase):
    def test_entry_does_not_import_detector_frameworks(self):
        import ast
        tree = ast.parse(inspect.getsource(entry))
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        self.assertFalse(any(n and n.split('.')[0] in ('torch', 'mmcv', 'mmrotate', 'cv2') for n in imports))

    def test_wrong_hash_and_failed_bundle_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); (p/'data.json').write_text('{}')
            hashes = {'data.json': entry.parent.sha(p/'data.json')}
            self.assertEqual(entry.exact_bundle(p, hashes), hashes)
            (p/'data.json').write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, 'exact reviewed'):
                entry.exact_bundle(p, hashes)
            (p/'failure.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Failed bundle'):
                entry.exact_bundle(p, hashes)

    def test_missing_frame_role_order_duplicate_and_bad_size(self):
        for change in ('missing', 'role', 'duplicate', 'order', 'size', 'multioutput'):
            v = rows()
            if change == 'missing': v.pop()
            if change == 'role': v[0]['split'] = 'val'
            if change == 'duplicate': v[1]['image'] = v[0]['image']
            if change == 'order': v.reverse()
            if change == 'size': v[-1]['image_size'] = [0, 160]
            if change == 'multioutput': v[0]['pred'] = [v[0]['pred']]*2
            with self.subTest(change=change), self.assertRaises(ValueError):
                entry.validate_rows(v, {'real_seq03': 4})

    def test_saved_angle_qualification_is_offline_and_validated(self):
        v = rows(); v[0]['gt'][2] = v[0]['gt'][3]
        v[0]['angle_axis_well_defined'] = False
        entry.validate_rows(v, {'real_seq03': 4})
        decisions, stats = entry.score(v, policy())
        self.assertTrue(decisions[0]['methods']['simple']['angle_accepted'])
        self.assertEqual(stats['all']['components']['angle']['simple']['unassessed_accepted'], 1)
        self.assertEqual(stats['all']['components']['angle']['simple']['flag_accepted_all_frames'], 2)
        v[0]['angle_axis_well_defined'] = True
        with self.assertRaisesRegex(ValueError, 'qualification'):
            entry.validate_rows(v, {'real_seq03': 4})

    def test_center_denominators_missing_states_and_state_confusion(self):
        records, s = entry.score(rows(), policy()); a = s['all']
        self.assertEqual(a['output_frames'], 3)
        self.assertEqual(a['center_hits'], 2)
        self.assertAlmostEqual(a['center_hit_rate_on_outputs'], 2/3)
        self.assertEqual(a['all_frame_center_correct_coverage'], .5)
        self.assertFalse(any(records[-1]['methods']['simple'][c+'_accepted'] for c in simple.COMPONENTS))
        size = a['components']['size']['simple']
        self.assertEqual(size['incorrect_accepted'], 1)
        self.assertEqual(size['correct_rejected'], 1)
        self.assertEqual(size['state_errors_on_outputs'], 2)
        self.assertAlmostEqual(size['state_accuracy_on_outputs'], 1/3)
        self.assertEqual(size['balanced_state_accuracy'], .25)

    def test_GT_and_old_quality_fields_cannot_change_online_flags(self):
        v = rows(); before = deepcopy(v); q = policy()
        with patch.object(simple, 'create_policy', side_effect=AssertionError('No refitting')), patch.object(simple, 'coverage_cutoff', side_effect=AssertionError('No TEST calibration')):
            a, _ = entry.score(v, q)
            altered = deepcopy(v)
            for r in altered:
                r['gt'] = [1., 2., 60., 10., 1.4]
                r['qualities'] = {'roi': [0., 0., 0.], 'structure': [1., 1., 1.]}
            b, _ = entry.score(altered, q)
        self.assertEqual([r['methods'] for r in a], [r['methods'] for r in b])
        self.assertEqual(v, before)
        self.assertEqual(q, policy())

    def test_equivalent_width_swap_and_raw_preservation(self):
        v = rows(); swapped = deepcopy(v)
        for r in swapped:
            if r['pred']:
                p = r['pred']; p[2], p[3] = p[3], p[2]; p[4] += math.pi/2
        a, sa = entry.score(v, policy()); b, sb = entry.score(swapped, policy())
        self.assertEqual(sa['all']['components']['size']['simple']['incorrect_accepted'],
                         sb['all']['components']['size']['simple']['incorrect_accepted'])
        for r, source in zip(b, swapped):
            self.assertEqual(r['methods']['simple']['raw_b_output'], source['pred'])

    def test_rejection_independent_and_complete_OBB_separate(self):
        q = policy(); q['cutoffs']['simple']['size']['risk_le'] = 0.
        q['cutoffs']['simple']['angle']['risk_le'] = 1.
        records, stats = entry.score(rows(), q)
        for r in records[:3]:
            self.assertTrue(r['methods']['simple']['center_accepted'])
            self.assertFalse(r['methods']['simple']['size_accepted'])
            self.assertTrue(r['methods']['simple']['angle_accepted'])
        self.assertEqual(stats['all']['complete_obb']['simple']['accepted_frames'], 0)

    def test_missing_all_outputs_gives_null_quality_without_dividing_by_zero(self):
        v = rows()
        for r in v:r['pred'] = None
        _, s = entry.score(v, policy())
        self.assertIsNone(s['all']['center_hit_rate_on_outputs'])
        self.assertIsNone(s['all']['components']['size']['simple']['state_accuracy_on_outputs'])
        self.assertEqual(s['all']['all_frame_center_correct_coverage'], 0.)

    def test_current_source_contract_preserves_original_method(self):
        contract, sources = entry.checked_sources()
        self.assertFalse(contract['test_used_for_selection'])
        self.assertEqual(contract['methods'], ['raw', 'score_only', 'simple'])
        self.assertEqual(sources['parent']['manifest_sha256'], contract['parent_manifest_sha256'])

    def test_output_no_overwrite_and_invalid_json_does_not_publish(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'result.json'; entry.parent.write_new(p, {'ok': True})
            with self.assertRaises(FileExistsError):entry.parent.write_new(p, {'ok': False})
            with self.assertRaises(ValueError):entry.parent.write_new(Path(d)/'bad.json', {'bad': float('nan')})
            self.assertEqual(json.loads(p.read_text()), {'ok': True})
            self.assertEqual(sorted(x.name for x in Path(d).iterdir()), ['result.json'])


if __name__ == '__main__':
    unittest.main()
