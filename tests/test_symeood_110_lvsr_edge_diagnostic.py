"""Fixed-M membership, paired tradeoff and missing-output denominators."""
import unittest
from crane_project.tools.diagnose_symeood_110_lvsr_edges import edge_diagnostic,reference_map,protected


def row(name,m,c,scale=1.):
    gt=[0.,0.,100.,50.,0.]
    return dict(image=name,scale=scale,domain='sim',split='train_sim',sequence='sim_seq',
        gt=gt,midpoint=None if m is None else [0.,0.,m[0],m[1],0.,.9],
        edge_residual=None if c is None else [0.,0.,c[0],c[1],0.,.9])


class EdgeDiagnosis(unittest.TestCase):
    def test_near_set_is_fixed_by_midpoint_and_includes_material_harm(self):
        rr=[row('a',(102.,45.),(104.,48.)),row('b',(110.,50.),(100.,50.)),row('missing',None,None)]
        d,_=edge_diagnostic(rr,'edge_residual')
        self.assertEqual(d['views'],3);self.assertEqual(d['outputs'],2);self.assertEqual(d['missing'],1)
        self.assertEqual(d['long_near_count'],1);self.assertEqual(d['long_near_crossed_3pct'],1)
        self.assertEqual(d['long_near_material_harmed_0p1pp'],1)
        self.assertEqual(d['short_improved_long_worsened'],1)
        self.assertEqual(d['short_gain_with_long_harm_fraction'],1.)

    def test_protection_uses_tighter_reference_and_all_edge_denominator(self):
        base=[row('a',(102.,45.),(101.,48.)),row('b',(110.,50.),(105.,50.))]
        refs=reference_map(base)
        self.assertAlmostEqual(refs['a',1.]['reference'][0],.01)
        candidate=[row('a',(102.,45.),(103.,48.)),row('b',(110.,50.),(105.,50.))]
        d=protected(candidate,'edge_residual',refs)
        self.assertAlmostEqual(d['long_protected_excess_sum'],.02)
        self.assertAlmostEqual(d['long_protected_excess_per_all_edges'],.005)


if __name__=='__main__':unittest.main()
