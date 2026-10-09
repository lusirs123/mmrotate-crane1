import math
import unittest
from crane_project.utils import port_reliability_complementarity_v1 as core
from crane_project.tools import review_port_reliability_complementarity_v1 as independent


def row(i, bad=False, sequence='real_seq01', frame=None, missing=False, risk=.2):
    return dict(image=sequence+'_%05d'%i,sequence=sequence,frame_id=i if frame is None else frame,
        domain='real',split='train',reliability_role='train',gt=[0,0,100,50,0],
        pred=None if missing else [0,0,120 if bad else 100,50,0,.8],
        risks=dict(candidate=risk,full_simple=risk,score_only=risk))


class ComplementarityTests(unittest.TestCase):
    def test_bad_boundary_and_axis_swap(self):
        r=row(1);r['pred'][2]=110
        self.assertFalse(core.bad(r))
        r['pred'][2]=110.00001;self.assertTrue(core.bad(r))
        r['pred'][2:4]=[50,100];self.assertFalse(core.bad(r))

    def test_invalid_edges(self):
        for v in (0,-1,float('nan'),float('inf')):
            r=row(1);r['pred'][2]=v
            with self.assertRaises(ValueError):core.bad(r)

    def test_missing_is_not_confusion(self):
        r=row(1,missing=True)
        self.assertIsNone(core.bad(r));self.assertEqual(core.state(r,False),'MISSING')
        with self.assertRaises(ValueError):core.state(r,True)
        self.assertEqual(core.accepted([r],'candidate',.5),set())

    def test_fixed_threshold_uses_no_truth(self):
        a=row(1,risk=.5);b=row(2,bad=True,risk=.5)
        selected=core.accepted([a,b],'candidate',.5)
        self.assertEqual(selected,{a['image'],b['image']})
        a['gt'][2]=5;b['gt'][2]=5
        self.assertEqual(core.accepted([a,b],'candidate',.5),selected)

    def test_auc_ties(self):
        rows=[row(1,risk=.2),row(2,bad=True,risk=.2),row(3,bad=True,risk=.8)]
        self.assertEqual(core.auc(rows,'candidate'),.75)
        self.assertEqual(core.auc(rows,'candidate'),independent.pair_auc(rows,'candidate'))

    def test_auc_single_class(self):
        self.assertIsNone(core.auc([row(1)],'candidate'))
        self.assertIsNone(core.auc([row(1,missing=True)],'candidate'))

    def test_state_change_decomposition(self):
        rows=[row(1,bad=True),row(2),row(3),row(4,bad=True),row(5,missing=True)]
        baseline={rows[0]['image'],rows[1]['image']};candidate={rows[2]['image'],rows[3]['image']}
        point=core.complement(rows,baseline,candidate,'candidate')
        self.assertEqual(point['changes'],dict(rescued_FA=1,new_FR=1,restored_CR=1,new_FA=1))
        self.assertEqual(point['reject_only_AND']['states'],dict(FA=0,FR=2,ED=2,CR=0,MISSING=1))
        self.assertFalse(point['GT_oracle_upper_bound']['executable_policy'])

    def test_group_intersects_global_acceptance(self):
        a,b=row(1),row(2)
        point=core.complement([a],{a['image'],b['image']},{b['image']},'candidate')
        self.assertEqual(point['changes']['new_FR'],1)

    def test_same_count_tie_bounds_are_not_claimed_gain(self):
        rows=[row(1,bad=True,risk=.2),row(2,risk=.2),row(3,risk=.2)]
        summary=core.continuity.summarize(rows,{rows[0]['image'],rows[1]['image']})
        c=core.matched(rows,'candidate',summary)
        self.assertEqual(c['same_count_tie_bounds']['bad_min'],0)
        self.assertEqual(c['same_count_tie_bounds']['bad_max'],1)
        self.assertFalse(c['same_CR_exact'])
        self.assertEqual(c['same_CR']['states']['CR'],2)

    def test_same_cr_zero(self):
        rows=[row(1,bad=True),row(2)]
        c=core.matched(rows,'candidate',core.continuity.summarize(rows,set()))
        self.assertTrue(c['same_CR_exact']);self.assertEqual(c['same_CR']['accepted_outputs'],0)

    def test_false_rejection_runs_reset_on_gaps_and_sequence(self):
        rows=[row(1),row(2),row(4),row(5,sequence='real_seq02'),row(6,sequence='real_seq02')]
        value=core.continuity.summarize(rows,set())
        self.assertEqual(value['runs']['correct_rejection']['longest'],2)
        self.assertEqual(independent.count(rows,set())[1],2)

    def test_rejecting_errors_does_not_make_fr_run(self):
        rows=[row(1),row(2,bad=True),row(3)]
        self.assertEqual(core.continuity.summarize(rows,set())['runs']['correct_rejection']['longest'],1)

    def test_gate_does_not_accept_oracle(self):
        rows=[row(1,bad=True),row(2)]
        p=core.complement(rows,{r['image'] for r in rows},set(),'candidate')
        p['GT_oracle_upper_bound']['rescued_FA']=100
        self.assertFalse(core.gate({'all':p})['passed'])

    def test_center_denominators(self):
        rows=[row(1),row(2),row(3,missing=True)];rows[1]['pred'][0]=15
        c=core.center(rows)
        self.assertEqual(c['hit_rate_on_outputs'],.5)
        self.assertEqual(c['output_coverage'],2/3)
        self.assertEqual(c['correct_coverage_all_frames'],1/3)

    def test_test_role_rejected(self):
        r=row(1);r['split']='test'
        with self.assertRaises(ValueError):core.continuity.summarize([r],set())


if __name__=='__main__':unittest.main()
