import copy
import unittest
from crane_project.tools import analyze_port_midpoint_simple_disagreement_v1 as m
from test_port_reliability_paired_ranking_audit_v1 import row


def sample(i,wrong=False,s=.2,q=.3,missing=False):
    r=row(i,wrong,missing=missing)
    r['audit_risks']=dict(simple=None if missing else s,mlp=None if missing else q)
    return r


class DisagreementTests(unittest.TestCase):
    def test_four_groups_and_missing(self):
        a=[sample(0),sample(1,True),sample(2),sample(3,True),sample(4),sample(5,True),sample(6),sample(7,True),sample(8,missing=True)]
        t=m.table(a,{a[i]['image'] for i in (0,1,2,3)},{a[i]['image'] for i in (0,1,4,5)})
        self.assertEqual(t['changes'],dict(rescued_FA=1,new_FR=1,restored_CR=1,new_FA=1))
        self.assertEqual(t['MISSING'],1)
        m.independent_table_check(a,t)
        for c in m.CELLS:self.assertEqual((t['cells'][c]['correct'],t['cells'][c]['wrong']),(1,1))

    def test_error_axes_sign_severity_and_canonical_swap(self):
        a=[sample(i,True) for i in range(3)]
        a[0]['pred'][2:4]=[8.,5.] # L small20%
        a[1]['pred'][2:4]=[10.,6.] # S large20%
        a[2]['pred'][2:4]=[13.,4.] # both mixed30%
        x=m.error_description(a)
        self.assertEqual(x['axis'],dict(L_only=1,S_only=1,both=1))
        self.assertEqual(x['direction'],dict(small=1,large=1,mixed=1))
        self.assertEqual(x['severity'],dict(**{'15_25':2,'over25':1}))
        b=copy.deepcopy(a)
        for r in b:r['pred'][2:4]=r['pred'][2:4][::-1]
        self.assertEqual(m.error_description(b),x)

    def test_gt_does_not_enter_score_features(self):
        a=[sample(0,False,.1,.2),sample(1,True,.8,.9)]
        x=m.score_evidence(a)
        self.assertEqual(x['AUROC'],dict(simple=1.,mlp=1.))
        self.assertEqual(x['pair_dominance']['error_higher_both'],1)
        self.assertNotIn('gt',x)

    def test_conflicting_scores_and_single_class(self):
        a=[sample(0,False,.1,.9),sample(1,True,.9,.1)]
        x=m.score_evidence(a)
        self.assertEqual(x['pair_dominance']['conflicting'],1)
        self.assertEqual(x['AUROC'],dict(simple=1.,mlp=0.))
        x=m.score_evidence(a[:1]);self.assertIsNone(x['AUROC']['simple'])
        self.assertEqual(x['pairs'],0)

    def test_tie_auc_and_quantile(self):
        a=[sample(0),sample(1,True)]
        self.assertEqual(m.score_evidence(a)['AUROC']['simple'],.5)
        self.assertEqual(m.quantile([0,10],.95),9.5)
        self.assertIsNone(m.quantile([],.5))

    def test_matched_sets_tie_policy(self):
        a=[sample(0,False,.1),sample(1,True,.1),sample(2,False,.1)]
        r=m.rank.Ranking(a,'simple');p=r.same_cr(1)
        self.assertFalse(p['exact_CR'])
        self.assertEqual(len(m.accepted(r,p,'same_CR')),3)
        self.assertEqual(len(m.accepted(r,r.same_count(1),'same_count')),1)

    def test_missing_cannot_enter_cell(self):
        a=[sample(0,missing=True)]
        with self.assertRaises(ValueError):m.table(a,{a[0]['image']},set())

    def test_does_not_mutate_input(self):
        a=[sample(0),sample(1,True)];old=copy.deepcopy(a)
        m.table(a,{a[0]['image']},{a[1]['image']});m.score_evidence(a)
        self.assertEqual(a,old)


if __name__=='__main__':unittest.main()
