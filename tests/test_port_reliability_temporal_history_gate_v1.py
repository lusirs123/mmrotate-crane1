import inspect
import math
import unittest
from copy import deepcopy
from crane_project.utils import port_reliability_temporal_history_gate_v1 as core
from test_port_reliability_temporal_evidence_v1 import row


class HistoryGateTests(unittest.TestCase):
    def feed(self, logs, q=.02):
        tracker=core.CausalHistoryGateEvidence(q)
        return [tracker.observe([0,0,math.exp(x),math.exp(x)/2,0,.8],i,[1000,1000]) for i,x in enumerate(logs)]

    def test_linear_history_selects_frozen_trend(self):
        p=self.feed([4.,4.05,4.1,4.15])[-1]
        self.assertEqual(p['branch'],'trend');self.assertAlmostEqual(p['primary'],0)
        self.assertEqual(p['selected_cutoff'],core.TREND_CUTOFF)

    def test_middle_outlier_selects_frozen_level(self):
        p=self.feed([4.,4.5,4.1,4.15])[-1]
        self.assertEqual(p['branch'],'level')
        self.assertAlmostEqual(p['history_linearity'],.45)
        self.assertAlmostEqual(p['primary'],p['level_primary']/core.LEVEL_CUTOFF)
        self.assertEqual(p['selected_cutoff'],core.LEVEL_CUTOFF)

    def test_endpoint_outlier_invalidates_linear_history(self):
        p=self.feed([4.,4.05,4.8,4.15])[-1]
        self.assertEqual(p['branch'],'level')
        self.assertGreater(p['trend_primary'],p['level_primary'])

    def test_q_ignores_current_future_and_gt(self):
        a=self.feed([4.,4.05,4.1,4.15,4.2]);b=self.feed([4.,4.05,4.1,5.,3.])
        self.assertEqual(a[3]['history_linearity'],b[3]['history_linearity'])
        self.assertEqual(a[3]['branch'],b[3]['branch'])
        rows=[row(i,100+i*i) for i in range(7)];a=core.enrich(rows,{})
        for r in rows:r['gt']=[999,888,700,600,.5]
        b=core.enrich(rows,{})
        self.assertEqual([r['temporal'] for r in a],[r['temporal'] for r in b])
        self.assertNotIn('gt',inspect.signature(core.CausalHistoryGateEvidence.observe).parameters)

    def test_independent_axes_and_isotropic_width_equivalence(self):
        rows=[row(i,100+i*i) for i in range(7)];a=core.apply_gate(core.enrich(rows,{}),.01)
        for r in rows:
            r['pred'][2:4]=[r['pred'][3]*3,r['pred'][2]*3]
            r['pred'][4]+=math.pi/2;r['image_size']=[3000,3000]
        b=core.apply_gate(core.enrich(rows,{}),.01)
        for x,y in zip(a,b):
            if x['temporal']['available']:
                self.assertAlmostEqual(x['temporal']['history_linearity'],y['temporal']['history_linearity'])
                self.assertAlmostEqual(x['temporal']['primary'],y['temporal']['primary'])
                self.assertEqual(x['temporal']['branch'],y['temporal']['branch'])

    def test_missing_gaps_resolution_and_boundary_reset(self):
        rows=[row(i) for i in range(10)];rows[4]=row(4,None)
        a=core.apply_gate(core.enrich(rows,{'real_seq01':[9]}),.01)
        self.assertTrue(a[3]['temporal']['available']);self.assertFalse(a[7]['temporal']['available'])
        self.assertTrue(a[8]['temporal']['available']);self.assertFalse(a[9]['temporal']['available'])
        for i in (4,7,9):self.assertEqual(a[i]['temporal']['branch'],'unsupported')
        t=core.CausalHistoryGateEvidence(.01)
        for i in range(4):t.observe([0,0,100,50,0,.8],i,[1000,1000])
        self.assertFalse(t.observe([0,0,100,50,0,.8],6,[1000,1000])['available'])
        self.assertFalse(t.observe([0,0,100,50,0,.8],7,[900,900])['available'])

    def test_frozen_arm_replay_and_input_preservation(self):
        rows=[row(i,100+i*i) for i in range(8)];snapshot=deepcopy(rows)
        a=core.enrich(rows,{});b=core.apply_gate(a,.01)
        old=core.trend.enrich(rows,{})
        self.assertEqual([r['temporal'] for r in core.trend_view(b)],[r['temporal'] for r in old])
        self.assertEqual(rows,snapshot)
        for r,s in zip(b,snapshot):
            for k in ('pred','gt','original_simple_decision','image_size'):self.assertEqual(r[k],s[k])

    def test_gate_boundary_and_frozen_raw_acceptance(self):
        for branch,q in [('trend',.01),('level',.03)]:
            cutoff=core.TREND_CUTOFF if branch=='trend' else core.LEVEL_CUTOFF
            for offset in (-1e-15,0,1e-15):
                p=dict(available=True,history_linearity=q,trend_primary=cutoff+offset,level_primary=cutoff+offset)
                s=core.select(p,.02)
                self.assertEqual(s['primary']<=1,offset<=0);self.assertEqual(s['branch'],branch)
        p=dict(available=True,history_linearity=.02,trend_primary=0.,level_primary=.1)
        self.assertEqual(core.select(p,.02)['branch'],'trend')
        with self.assertRaises(ValueError):core.CausalHistoryGateEvidence(float('nan'))
        with self.assertRaises(ValueError):core.select(p,-1)

    def test_TRAIN_fit_whole_ties_group_max_and_empty_support(self):
        rows=[row(i,100+i*i/20) for i in range(30)]
        for r in rows:r['gt']=r['pred'][:5]
        other=row(0);other.update(sequence='real_seq05',image='real_seq05_00000',original_simple_decision={'size_accepted':False})
        a=core.enrich(rows+[other],{})
        fit=core.fit_history_gate(a)
        self.assertEqual(fit['q_le'],max(v['cutoff'] for v in fit['group_support'].values() if v['cutoff'] is not None))
        self.assertEqual(fit['group_support']['sequence:real_seq05']['eligible_histories'],0)
        self.assertIsNone(fit['group_support']['sequence:real_seq05']['cutoff'])
        before=deepcopy(fit)
        core.apply_gate(core.enrich([row(i,split='val') for i in range(5)],{}),fit['q_le'])
        self.assertEqual(fit,before)

    def test_all_three_history_truth_required_only_for_fitting(self):
        rows=[row(i) for i in range(8)];rows[1]['gt'][2]=200
        a=core.enrich(rows,{})
        fit=core.fit_history_gate(a)
        # Current3 and4 have an erroneous frame1 in history; current5,6,7 eligible.
        self.assertEqual(fit['group_support']['all']['eligible_histories'],3)
        self.assertTrue(a[3]['temporal']['available'])

    def test_roles_rejected_and_warmup_preserves_simple(self):
        with self.assertRaises(ValueError):core.fit_history_gate(core.enrich([row(i,split='val') for i in range(4)],{}))
        with self.assertRaises(ValueError):core.enrich([row(0,split='test')],{})
        a=core.apply_gate(core.enrich([row(i) for i in range(3)],{}),0)
        self.assertEqual(core.previous.veto_accept(a,1),{r['image'] for r in a})
        self.assertEqual(core.previous.veto_accept(core.apply_gate(core.enrich([row(0,None)],{}),0),1),set())
        a[0]['original_simple_decision']['size_accepted']=False
        self.assertNotIn(a[0]['image'],core.previous.veto_accept(a,1))

    def test_stable_wrong_history_still_looks_linear(self):
        a=core.apply_gate(core.enrich([row(i,70) for i in range(8)],{}),0)
        self.assertEqual(len(core.previous.veto_accept(a,1)),8)
        stats=core.branch_diagnostics(a,0)['groups']['all']
        self.assertEqual(stats['trend']['history_has_error'],5)
        self.assertEqual(stats['trend']['history_all_good'],0)


if __name__=='__main__':unittest.main()
