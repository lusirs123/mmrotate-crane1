"""Diagnostic isolation, exact update scheduling and comparison checks; no GPU/data."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np
from crane_project.tools import diagnose_symeood_r4_midpoint_leavevideo_v1 as d
from crane_project.tools import train_symeood_110_midpoint_v1 as original


def records():
    out = []
    for seq, domain, count in [('real_seq01', 'real', 3), ('real_seq13', 'real', 5), ('sim_seq08', 'sim', 2)]:
        for scale in (1., .5):
            for i in range(count):
                out.append(dict(image=seq+'_'+str(i), sequence=seq, domain=domain, scale=scale,
                                role='train', eligible=i != 1))
    out.append(dict(image='val_1', sequence='real_seq07', domain='real', scale=1., role='val', eligible=True))
    return out


def row(error=(.02, .04), image='a', present=True):
    return dict(image=image, scale=1., b=[0, 0, 20, 10, 0, .8] if present else None,
        gt=[0, 0, 20, 10, 0], midpoint=[0, 0, 20, 10, 0, .8] if present else None,
        metrics=dict(midpoint=dict(output=present, long_edge_relative_error=error[0], short_edge_relative_error=error[1])))


class DiagnosticChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original.np = np

    def test_original_sealed_sources(self):
        from crane_project.utils import symeood_r4_midpoint_protocol_v1 as p
        try:
            identity = p.checked_sources()
        except FileNotFoundError as exc:
            self.skipTest('Original source closure incomplete locally; required on server: '+str(exc.filename))
        self.assertEqual(identity['settings'], original.SETTINGS)

    def test_fixed_config(self):
        spec = d.checked_config()
        self.assertEqual(spec['updates_per_arm'], 20*899)
        self.assertIn('no TEST', spec['selection'])

    def test_relative_source_path_pins_match_absolute_path(self):
        from pathlib import Path
        relative = Path('crane_project/configs/crane_midpoint_leavevideo_diag_v1.json')
        self.assertEqual(d.pin_files([relative]), d.pin_files([d.ROOT/relative]))

    def test_exact_budget_and_both_scale_exclusion_using_original_batcher(self):
        rr = records()
        subset, batches, receipt = d.schedule(original, rr, 'real_seq01', 11)
        self.assertEqual(len(batches), 11)
        self.assertEqual(receipt['slots'], 88)
        self.assertEqual(receipt['excluded_views'], 6)
        self.assertEqual(receipt['excluded_eligible'], 4)
        expected = []
        for epoch in range(1, 7):
            expected.extend(original.epoch_batches(subset, epoch))
        self.assertEqual(batches, expected[:11])
        self.assertTrue(all(subset[i]['role'] == 'train' and subset[i]['eligible'] for bb in batches for i in bb))
        self.assertFalse(any('real_seq01' in k or 'seq07' in k for k in receipt['video_scale_exposure']))
        self.assertEqual(d.schedule(original, rr, 'real_seq01', 11)[2], receipt)
        self.assertNotEqual(d.schedule(original, rr, 'real_seq01', 12)[2]['ordered_view_sha256'], receipt['ordered_view_sha256'])

    def test_no_missing_or_single_scale_exclusion(self):
        rr = records()
        with self.assertRaises(ValueError):
            d.schedule(original, rr, 'unknown', 8)
        with self.assertRaises(ValueError):
            d.schedule(original, [r for r in rr if r['sequence'] != 'real_seq01' or r['scale'] == 1.], 'real_seq01', 8)

    def test_schedule_is_not_equal_epoch_count(self):
        rr = records()
        a = d.schedule(original, rr, 'real_seq01', 11)
        b = d.schedule(original, rr, 'real_seq13', 11)
        self.assertEqual(a[2]['slots'], b[2]['slots'])
        self.assertNotEqual(a[2]['virtual_epochs'], b[2]['virtual_epochs'])
        self.assertLess(a[2]['virtual_epochs'][-1]['used_steps'], a[2]['virtual_epochs'][-1]['full_steps'])

    def test_replay_requires_score_presence_and_coordinate_identity(self):
        r = row()
        saved = {('a', 1.): deepcopy(r)}
        self.assertEqual(d.compare_replay([r], saved, np), 0.)
        for key, index, delta in [('midpoint', 5, .01), ('midpoint', 0, .1), ('gt', 2, 1), ('b', 4, .1)]:
            wrong = deepcopy(saved)
            wrong['a', 1.][key][index] += delta
            with self.assertRaises(ValueError):
                d.compare_replay([r], wrong, np)
        with self.assertRaises(ValueError):
            d.compare_replay([row(present=False)], saved, np)

    def test_paired_accurate_edges_and_joint_migration(self):
        a, b = [row((.02, .04)), row((.15, .02), 'b')], [row((.025, .12)), row((.05, .01), 'b')]
        result = d.paired(a, b)
        self.assertEqual(result['repaired_joint10'], 1)
        self.assertEqual(result['damaged_joint10'], 1)
        self.assertEqual(result['long_accurate_le3pct']['damaged'], 1)
        self.assertEqual(result['short_accurate_le3pct']['damaged'], 0)
        wrong = deepcopy(b)
        wrong[0]['image'] = 'another'
        with self.assertRaises(ValueError):
            d.paired(a, wrong)

    def test_residual_sign_canonical_and_marginal_distance(self):
        r = row()
        r['midpoint'][2:4] = [9, 22]
        values = d.signed(r)
        self.assertLess(values[0], 0.)
        self.assertGreater(values[1], 0.)
        np.testing.assert_allclose(d.w1([[0., 0.]], [[.1, -.2]], np), [.1, .2], atol=1e-15, rtol=0.)
        self.assertEqual(d.w1([[0., 0.]], [[0., 0.]], np), [0., 0.])


if __name__ == '__main__':
    unittest.main()
