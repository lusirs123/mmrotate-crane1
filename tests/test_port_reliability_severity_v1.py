import unittest
from copy import deepcopy
from crane_project.tools import analyze_port_reliability_severity_v1 as a


def row(i, long=100., short=40., accepted=True, risk=.2, missing=False):
    return dict(image='r_%05d'%i,sequence='real_s',domain='real',frame_id=i,
        gt=[0.,0.,100.,40.,0.],pred=None if missing else [0.,0.,long,short,0.,.8],
        original_simple_decision=dict(size_accepted=accepted,center_accepted=not missing,angle_accepted=not missing),
        risks=dict(simple=None if missing else risk,score=None if missing else risk),
        temporal=dict(acquisition_boundary=False))


class TestSeverity(unittest.TestCase):
    def test_exact_upper_boundaries(self):
        for value,expected in [(110.,'correct'),(115.,'mild'),(125.,'moderate'),(125.01,'severe')]:
            self.assertEqual(a.describe(row(0,long=value))['severity'],expected)
    def test_axis_exchange_and_pi(self):
        r=row(0,long=80.,short=30.)
        other=deepcopy(r);other['pred'][2:4]=reversed(other['pred'][2:4]);other['gt'][2:4]=reversed(other['gt'][2:4]);other['pred'][4]+=a.math.pi
        self.assertEqual(a.describe(r),a.describe(other))
    def test_type_sign(self):
        for long,short,kind,sign in [(80.,40.,'long_only','small'),(100.,50.,'short_only','large'),(80.,30.,'both','small_small'),(120.,30.,'both','large_small')]:
            d=a.describe(row(0,long,short));self.assertEqual((d['error_type'],d['sign']),(kind,sign))
    def test_missing_is_neither_rejected_error_nor_correct(self):
        rows=[row(0),row(1,missing=True,accepted=False)]
        result,dd=a.analyze(rows);s=result['groups']['all']['formal']['states']
        self.assertEqual(s,dict(FA=0,FR=0,ED=0,CR=1,MISSING=1));self.assertEqual(a.review(rows,result,dd)['rows'],2)
    def test_correct_reference_excludes_other_errors(self):
        rows=[row(0,risk=.2),row(1,long=112.,risk=.1),row(2,long=130.,risk=.9)]
        z,_=a.analyze(rows);r=z['groups']['all']['ranking']
        self.assertEqual(r['mild']['AUROC']['simple'],0.);self.assertEqual(r['severe']['AUROC']['simple'],1.)
        self.assertEqual((r['mild']['correct'],r['mild']['wrong']),(1,1))
    def test_auroc_ties_and_empty_strata(self):
        z,_=a.analyze([row(0),row(1,long=120.)]);r=z['groups']['all']['ranking']
        self.assertEqual(r['moderate']['AUROC']['simple'],.5);self.assertIsNone(r['severe']['AUROC']['simple'])
    def test_no_mutation(self):
        rows=[row(0),row(1,long=120.,accepted=False)];before=deepcopy(rows)
        z,d=a.analyze(rows);a.review(rows,z,d);self.assertEqual(rows,before)
    def test_partition_reconciles(self):
        rows=[row(0),row(1,long=112.),row(2,short=48.,accepted=False),row(3,long=140.,short=60.)]
        z,d=a.analyze(rows);a.review(rows,z,d)
        for section in ('severity','error_type'):
            self.assertEqual(sum(v['FA'] for v in z['groups']['all']['errors'][section].values()),2)
    def test_whole_tie_CR_and_count_bounds(self):
        rows=[row(0),row(1),row(2,long=120.)]
        rows[1]['original_simple_decision']['size_accepted']=False
        z,d=a.analyze(rows);a.review(rows,z,d)
        p=z['groups']['all']['offline_points']
        self.assertFalse(p['same_CR']['simple']['point']['exact_CR'])
        self.assertTrue(p['same_count']['simple']['point']['tie_bounds']['partial_tie'])
    def test_corrupt_summary_rejected(self):
        rows=[row(0),row(1,long=120.)];z,d=a.analyze(rows)
        z['groups']['all']['ranking']['moderate']['AUROC']['simple']=.8
        with self.assertRaises(ValueError):a.review(rows,z,d)
    def test_corrupt_derived_rejected(self):
        rows=[row(0),row(1,long=120.)];z,d=a.analyze(rows);d[1]['severity']='severe'
        with self.assertRaises(ValueError):a.review(rows,z,d)
    def test_zero_error_and_zero_correct(self):
        for rows in ([row(0)],[row(0,long=130.)]):
            z,d=a.analyze(rows);a.review(rows,z,d)
            self.assertIsNone(z['groups']['all']['ranking']['all']['AUROC']['simple'])
    def test_invalid_size_fails(self):
        with self.assertRaises(ValueError):a.describe(row(0,long=0))


if __name__=='__main__':unittest.main()
