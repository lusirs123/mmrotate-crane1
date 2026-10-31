"""Independent analytic geometry and probe-coordinate contracts (stdlib only)."""
import math
import unittest

from crane_project.utils.eood_B_localization_probe_v1 import (
    cases, decode, derivative_intervals, diou, losses, matches_intervals, riou, sym_distance)


class GeometryChecks(unittest.TestCase):
    def test_analytic_translation_containment_and_disjoint(self):
        a = [0., 0., 10., 4., 0.]
        self.assertAlmostEqual(riou(a, a), 1.)
        self.assertAlmostEqual(riou(a, [2., 0., 10., 4., 0.]), 2./3.)
        self.assertAlmostEqual(riou(a, [0., 0., 5., 2., .0]), .25)
        self.assertEqual(riou(a, [100., 0., 10., 4., 0.]), 0.)

    def test_rotated_axis_equivalence_scale_and_symmetry(self):
        a = [500., -400., 80., 20., .37]
        b = decode(a, [.03, -.02, .04, -.01, .07])
        value = riou(a, b)
        self.assertAlmostEqual(value, riou(b, a), places=12)
        swap = [a[0], a[1], a[3], a[2], a[4]+math.pi/2]
        period = a[:4]+[a[4]+math.pi]
        for equivalent in (swap, period):
            self.assertAlmostEqual(value, riou(equivalent, b), places=12)
            self.assertAlmostEqual(sym_distance(equivalent, b), sym_distance(a, b), places=11)
        for scale in (.5, 2.):
            scaled = lambda x: [v*scale for v in x[:4]]+x[4:]
            self.assertAlmostEqual(value, riou(scaled(a), scaled(b)), places=12)
            self.assertAlmostEqual(sym_distance(a, b), sym_distance(scaled(a), scaled(b)), places=11)

    def test_jeffreys_closed_forms_and_square_angle_blindness(self):
        a = [0., 0., 80., 20., 0.]
        self.assertAlmostEqual(sym_distance(decode(a, [.03, 0, 0, 0, 0]), a), 4*.03**2)
        self.assertAlmostEqual(sym_distance(decode(a, [0, 0, .03, 0, 0]), a), math.cosh(.06)-1)
        angle = .07
        # Production sym_kld is the SUM of both KL directions, not their mean.
        expected = (4.**2+4.**-2-2)*math.sin(angle)**2
        self.assertAlmostEqual(sym_distance(decode(a, [0, 0, 0, 0, angle]), a), expected)
        square = [0., 0., 20., 20., 0.]
        changed = square[:4]+[.1]
        self.assertAlmostEqual(sym_distance(square, changed), 0.)
        self.assertLess(riou(square, changed), 1.)

    def test_probe_inventory_and_independent_diou_slope(self):
        rows = cases()
        self.assertEqual(len(rows), 3780)
        self.assertEqual(len({(r['scale'], r['aspect'], r['angle'], r['family'], r['sign'], r['error']) for r in rows}), len(rows))
        a = [0., 0., 80., 20., 0.]
        error, delta = .03, 1e-6
        plus = diou(decode(a, [error+delta, 0, 0, 0, 0]), a)
        minus = diou(decode(a, [error-delta, 0, 0, 0, 0]), a)
        center = lambda e: (e*80)**2/((80*(1+e))**2+20**2+1e-6)
        expected = 4/(1+error)**2+2*(center(error+delta)-center(error-delta))/(2*delta)
        self.assertAlmostEqual((plus-minus)/(2*delta), expected, places=7)
        self.assertAlmostEqual(losses(a, [error, 0, 0, 0, 0])['hybrid025'],
                               losses(a, [error, 0, 0, 0, 0])['DIoU']+.25*losses(a, [error, 0, 0, 0, 0])['SymKLD'])

    def test_nonsmooth_identity_retains_subgradient_interval(self):
        target = [0., 0., 40., 20., .37]
        intervals = derivative_intervals(target, [0.]*5, 1e-5)
        self.assertTrue(matches_intervals([0.]*5, intervals['DIoU']))
        self.assertTrue(matches_intervals([0.]*5, intervals['SymKLD']))
        self.assertFalse(matches_intervals([0., 1e6, 0., 0., 0.], intervals['DIoU']))


if __name__ == '__main__':
    unittest.main()
