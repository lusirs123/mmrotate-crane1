import copy
import math
import random
import unittest

from crane_project.utils import port_reliability_paired_ranking_audit_v1 as m


def row(i, wrong=False, risk=.5, missing=False):
    p = None if missing else [0., 0., 12. if wrong else 10., 5., 0., .6]
    d = dict(final_box_original=p, center_accepted=not missing,
             size_accepted=not missing, angle_accepted=not missing,
             risks=dict(size=None if missing else .5, angle=None if missing else .5))
    return dict(image='real_seq01_%05d'%i, sequence='real_seq01', domain='real',
                split='train', frame_id=i, image_size=[100,100], pred=p,
                gt=[0.,0.,10.,5.,0.], reliability_role='train',
                original_simple_decision=d, audit_risks=dict(a=None if missing else risk,
                                                            b=None if missing else 1-risk))


class SavedRankingTests(unittest.TestCase):
    def test_python38_relative_script_source_pin(self):
        from crane_project.tools import analyze_port_reliability_paired_ranking_audit_v1 as cli
        pins=cli.source_pins('crane_project/tools/analyze_port_reliability_paired_ranking_audit_v1.py')
        self.assertEqual(len(pins),5)
        self.assertIn('crane_project/tools/analyze_port_reliability_paired_ranking_audit_v1.py',pins)

    def test_tie_auc_and_count_bounds(self):
        rows=[row(0,False,.1),row(1,True,.2),row(2,False,.2),row(3,True,.8)]
        rank=m.Ranking(rows,'a')
        self.assertEqual(rank.auc(),.875)
        p=rank.same_count(2)
        self.assertEqual(p['tie_bounds'],dict(FA_min=0,FA_max=1,partial_tie=True))
        self.assertEqual(rank.same_cr(2)['accepted'],3)
        self.assertEqual(rank.same_cr(0)['accepted'],0)

    def test_whole_cr_tie_inconclusive(self):
        rows=[row(0,False,.1),row(1,False,.1),row(2,True,.1)]
        rank=m.Ranking(rows,'a')
        self.assertFalse(rank.same_cr(1)['exact_CR'])
        self.assertEqual(rank.same_cr(1)['CR'],2)
        self.assertEqual(rank.same_cr(1)['FA'],1)

    def test_missing_is_not_rejection_and_empty_risk_null(self):
        rows=[row(0,False,.1),row(1,True,.9),row(2,missing=True)]
        d=m.summarize(rows,set())
        self.assertEqual(d['states'],dict(FA=0,FR=1,ED=1,CR=0,MISSING=1))
        self.assertIsNone(d['accepted_error_rate'])
        with self.assertRaises(ValueError):m.summarize(rows,{rows[2]['image']})

    def test_single_class_and_no_support(self):
        for rows in ([row(0)], [row(0,True)], [row(0,missing=True)]):
            rank=m.Ranking(rows,'a')
            self.assertIsNone(rank.auc())
            self.assertEqual(rank.curve()[-1]['accepted'],int(rows[0]['pred'] is not None))

    def test_label_is_canonical_both_edges(self):
        a=row(0);a['pred'][2:4]=[5.,10.]
        self.assertFalse(m.bad(a))
        a['pred'][2:4]=[5.,11.]
        self.assertFalse(m.bad(a))
        a['pred'][3]=11.00001
        self.assertTrue(m.bad(a))
        a['pred'][2]=float('nan')
        with self.assertRaises(ValueError):m.bad(a)

    def test_identity_and_roles(self):
        rows=[row(0)]
        base=m.validate_rows(rows,'TRAIN',1)
        for key in ('pred','gt','original_simple_decision'):
            changed=copy.deepcopy(base)
            changed[rows[0]['image']][key]=None
            with self.assertRaises(ValueError):m.pair_identity(base,changed)
        with self.assertRaises(ValueError):m.validate_rows(rows,'TEST',1)
        with self.assertRaises(ValueError):m.validate_rows(rows+rows,'TRAIN',2)
        changed=copy.deepcopy(rows);changed[0]['split']='val'
        with self.assertRaises(ValueError):m.validate_rows(changed,'TRAIN',1)

    def test_correct_fr_runs_reset_at_gap_and_missing(self):
        rows=[row(0),row(1),row(3),row(4,missing=True),row(5)]
        self.assertEqual(m.summarize(rows,set())['longest_correct_FR'],2)

    def test_full_curve_and_scalar_reconstruction_random_ties(self):
        rng=random.Random(1701)
        rows=[row(i,rng.random()<.25,rng.randrange(8)/8.) for i in range(60)]
        original=copy.deepcopy(rows)
        rank=m.Ranking(rows,'a')
        cut=.5;kept={r['image'] for r in rows if r['audit_risks']['a']<=cut}
        check=m.independent_check(rank,m.summarize(rows,kept),cut)
        self.assertTrue(check['frozen_states_verified'])
        for n in range(61):
            p=rank.same_count(n)
            self.assertLessEqual(p['tie_bounds']['FA_min'],p['FA'])
            self.assertGreaterEqual(p['tie_bounds']['FA_max'],p['FA'])
        for g in range(rank.good+1):
            p=rank.same_cr(g)
            self.assertGreaterEqual(p['CR'],g)
        self.assertEqual(rows,original)

    def test_high_region_uses_all_integer_correct_targets(self):
        rows=[row(i,False,i/100) for i in range(20)]+[row(20,True,.99)]
        a=m.Ranking(rows,'a');b=m.Ranking(rows,'b')
        r=m.compare_high_region(a,b)
        self.assertEqual(r['better'],2) # CR19 and CR20, not 20 independent samples
        self.assertEqual(r['worse'],0)

    def test_nonfinite_risk_rejected(self):
        with self.assertRaises(ValueError):m.Ranking([row(0,risk=math.inf)],'a')


if __name__ == '__main__':unittest.main()
