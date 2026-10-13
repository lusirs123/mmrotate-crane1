import inspect
import math
import unittest
from copy import deepcopy
from crane_project.utils import port_reliability_temporal_trend_v1 as core
from test_port_reliability_temporal_evidence_v1 import row


class TrendTests(unittest.TestCase):
    def feed(self, logs):
        t = core.CausalTrendEvidence()
        return [t.observe([0,0,math.exp(x),math.exp(x)/2,0,.8],i,[1000,1000])
                for i,x in enumerate(logs)]

    def test_constant_and_log_linear(self):
        for logs in ([4.]*7, [4.+i*.06 for i in range(7)]):
            points = self.feed(logs)
            for p in points[3:]: self.assertAlmostEqual(p['primary'],0,places=12)
        self.assertGreater(self.feed([4.+i*.06 for i in range(4)])[-1]['level_primary'],.1)

    def test_independent_long_short_trends(self):
        t = core.CausalTrendEvidence()
        for i in range(4):
            p = t.observe([0,0,100*math.exp(.02*i),50*math.exp(-.03*i),0,.8],i,[1000,1000])
        self.assertAlmostEqual(p['primary'],0,places=12)
        for got, expected in zip(p['slope'],[.02,-.03]):self.assertAlmostEqual(got,expected)

    def test_jump_and_signed_residual(self):
        p = self.feed([4.,4.,4.,4.2])[-1]
        self.assertAlmostEqual(p['primary'],.2)
        self.assertAlmostEqual(p['signed_log_change'][0],.2)
        q = self.feed([4.,4.,4.,3.8])[-1]
        self.assertAlmostEqual(q['signed_log_change'][0],-.2)

    def test_current_and_future_do_not_change_prediction(self):
        a = self.feed([4.,4.05,4.1,4.15,4.2])
        b = self.feed([4.,4.05,4.1,4.8,3.])
        self.assertEqual(a[3]['predicted_log_edges'],b[3]['predicted_log_edges'])
        self.assertEqual(a[3]['slope'],b[3]['slope'])
        self.assertNotEqual(a[3]['primary'],b[3]['primary'])
        self.assertNotIn('gt',inspect.signature(core.CausalTrendEvidence.observe).parameters)

    def test_gt_invariance_original_fields_and_legacy_replay(self):
        rows=[row(i,100+i) for i in range(7)]
        original=deepcopy(rows);a=core.enrich(rows,{})
        self.assertEqual(rows,original)
        for r in rows:r['gt']=[1,2,999,888,.5]
        b=core.enrich(rows,{})
        self.assertEqual([r['temporal'] for r in a],[r['temporal'] for r in b])
        old=core.previous.enrich(original,{})
        view=core.level_view(a)
        self.assertEqual([r['temporal'] for r in old],[r['temporal'] for r in view])
        for x,y in zip(a,original):
            for k in ('pred','original_simple_decision','image_size'):self.assertEqual(x[k],y[k])

    def test_isotropic_width_swap_pi_equivalence(self):
        rows=[row(i,100+i*i) for i in range(8)];a=core.enrich(rows,{})
        for r in rows:
            r['pred'][2:4]=[r['pred'][3]*3,r['pred'][2]*3]
            r['pred'][4]+=math.pi/2;r['image_size']=[3000,3000]
        b=core.enrich(rows,{})
        for x,y in zip(a,b):
            if x['temporal']['available']:
                self.assertAlmostEqual(x['temporal']['primary'],y['temporal']['primary'],places=12)

    def test_missing_gap_resolution_boundary_and_video_reset(self):
        rows=[row(i) for i in range(10)];rows[4]=row(4,None)
        a=core.enrich(rows,{'real_seq01':[9]})
        self.assertTrue(a[3]['temporal']['available'])
        self.assertFalse(a[7]['temporal']['available'])
        self.assertTrue(a[8]['temporal']['available'])
        self.assertFalse(a[9]['temporal']['available'])
        for index in (4,7,9):self.assertIsNone(a[index]['temporal']['predicted_log_edges'])
        t=core.CausalTrendEvidence()
        for i in range(4):t.observe([0,0,100,50,0,.8],i,[1000,1000])
        self.assertFalse(t.observe([0,0,100,50,0,.8],6,[1000,1000])['available'])
        self.assertFalse(t.observe([0,0,100,50,0,.8],7,[900,900])['available'])
        other=row(0);other['sequence']='real_seq02';other['image']='real_seq02_00000'
        self.assertFalse(core.enrich([row(i) for i in range(4)]+[other],{})[-1]['temporal']['available'])

    def test_three_point_formula_is_not_outlier_robust(self):
        a=self.feed([4.,4.05,4.1,4.15])[-1]
        b=self.feed([4.,5.,4.1,4.15])[-1]
        self.assertEqual(a['predicted_log_edges'],b['predicted_log_edges'])
        c=self.feed([4.,4.05,5.,4.15])[-1]
        self.assertGreater(c['primary'],1.)

    def test_stable_wrong_cannot_be_detected(self):
        a=core.enrich([row(i,size=70) for i in range(8)],{})
        self.assertTrue(all(r['temporal']['primary']==0 for r in a[3:]))
        self.assertEqual(len(core.previous.veto_accept(a,0)),8)

    def test_only_size_veto_and_unsupported_fallback(self):
        a=core.enrich([row(i,size=120 if i==3 else 100) for i in range(4)],{})
        self.assertEqual(core.previous.veto_accept(a,.1),{r['image'] for r in a[:3]})
        for r in a:r['original_simple_decision']['size_accepted']=False
        self.assertEqual(core.previous.veto_accept(a,100),set())
        self.assertEqual(core.previous.veto_accept(core.enrich([row(0,None)],{}),100),set())

    def test_fit_TRAIN_only_and_VAL_does_not_change_threshold(self):
        a=core.enrich([row(i,100+i*i/20) for i in range(30)],{})
        for r in a:r['gt']=r['pred'][:5]
        fit=core.fit_cutoff(a);snapshot=deepcopy(fit)
        core.evaluate(core.enrich([row(i,90,100,'val') for i in range(8)],{}),fit['risk_le'])
        self.assertEqual(fit,snapshot)
        with self.assertRaises(ValueError):core.fit_cutoff(core.enrich([row(i,split='val') for i in range(4)],{}))
        with self.assertRaises(ValueError):core.enrich([row(i,split='test') for i in range(4)],{})

    def test_old_new_transition_accounting(self):
        rows=core.enrich([row(i,100 if i<4 else 70) for i in range(9)],{})
        c=core.comparison(rows,.1,.1)['all']
        o,n=c['level']['states'],c['trend']['states'];d=c['transitions']['counts']
        self.assertEqual(n['FA']-o['FA'],d['new_FA']-d['rescued_FA'])
        self.assertEqual(n['FR']-o['FR'],d['new_FR']-d['restored_CR'])

    def test_matched_veto_preserves_fallback_and_original_rejects(self):
        rows=core.enrich([row(i,70 if i==4 else 100) for i in range(8)],{})
        rows[3]['original_simple_decision']['size_accepted']=False
        p=core.matched_veto(rows,count=5)
        self.assertTrue(p['feasible']);self.assertEqual(p['accepted_outputs'],5)
        self.assertFalse(core.matched_veto(rows,count=1)['feasible'])
        self.assertFalse(core.matched_veto(rows,correct=1)['feasible'])
        q=core.same_FA_veto(rows,0)
        self.assertTrue(q['feasible']);self.assertEqual(q['states']['FA'],0)
        self.assertLessEqual(q['accepted_outputs'],7)

    def test_whole_tie_FA_budget(self):
        rows=core.enrich([row(i,70 if i==4 else 100) for i in range(7)],{})
        for r in rows[3:]:r['risks']['temporal']=r['temporal']['primary']=.2
        q=core.same_FA_veto(rows,0)
        self.assertEqual(q['states']['CR'],3)
        p=core.matched_veto(rows,correct=4)
        self.assertFalse(p['exact_CR']);self.assertEqual(p['states']['FA'],1)
        p=core.matched_veto(rows,count=4)
        self.assertEqual(p['tie_bounds']['FA_min'],0);self.assertEqual(p['tie_bounds']['FA_max'],1)


if __name__=='__main__':unittest.main()
