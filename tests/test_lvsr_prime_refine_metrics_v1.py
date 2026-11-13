"""Threshold/missing-output and parent-identity checks for the paired audit."""
from copy import deepcopy
import unittest

from crane_project.utils.lvsr_prime_refine_metrics_v1 import boundary_pairs


def rows(widths):
    return [dict(image='real_a_%05d'%i, scale=1., sequence='real_a', domain='real',
        gt=[0., 0., 100., 50., 0.], b=[0., 0., 100., 50., 0., .9], midpoint=[0., 0., 100., 50., 0., .9],
        edge_residual=[0., 0., w, 50., 0., .9] if w else None) for i, w in enumerate(widths)]


class BoundaryChecks(unittest.TestCase):
    def test_repairs_damage_and_small_crossing(self):
        a, b = rows([109., 112., 100.]), rows([110.5, 109., 100.])
        v = boundary_pairs(a, b)['overall']
        self.assertEqual((v['repairs'], v['broken'], v['broken_new_error_le11pct']), (1, 1, 1))
        self.assertEqual(v['broken_long_crossings'], 1); self.assertEqual(v['broken_short_crossings'], 0)

    def test_missing_output_is_still_full_frame_failure(self):
        v = boundary_pairs(rows([100., None]), rows([None, 100.]))['overall']
        self.assertEqual((v['frames'], v['repairs'], v['broken']), (2, 1, 1))

    def test_rejects_identity_or_duplicate_change(self):
        a = rows([100., 100.]); b = deepcopy(a); b[0]['midpoint'][2] = 99.
        with self.assertRaises(ValueError): boundary_pairs(a, b)
        with self.assertRaises(ValueError): boundary_pairs(a, [a[0], a[0]])


if __name__ == '__main__': unittest.main()
