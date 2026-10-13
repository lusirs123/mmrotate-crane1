import inspect
import math
import unittest
from copy import deepcopy
from crane_project.utils import port_reliability_temporal_evidence_v1 as core


def row(frame, size=100., gt=100., split='train', keep=True):
    missing=size is None
    return dict(image='real_seq01_%05d'%frame,sequence='real_seq01',domain='real',
        split=split,reliability_role='val' if split=='val' else 'train',frame_id=frame,
        image_size=[1000,1000],pred=None if missing else [0,0,size,size/2,0,.8],
        gt=[0,0,gt,gt/2,0],size_risks={'full_simple':None if missing else .2},
        original_simple_decision=dict(size_accepted=keep and not missing))


class TemporalTests(unittest.TestCase):
    def test_causal_and_future_invariance(self):
        source=[row(i) for i in range(5)];source[3]['pred'][2]=120
        a=core.enrich(source,{})
        source[4]['pred'][2]=999;source[4]['gt'][2]=888
        b=core.enrich(source,{})
        self.assertEqual(a[3]['temporal'],b[3]['temporal'])
        self.assertAlmostEqual(a[3]['temporal']['primary'],math.log(1.2))
        self.assertEqual(a[3]['temporal']['history_frames'],[0,1,2])
        self.assertNotIn('gt',inspect.signature(core.CausalSizeEvidence.observe).parameters)

    def test_gt_does_not_affect_online_evidence(self):
        source=[row(i) for i in range(6)]
        a=core.enrich(source,{})
        for r in source:r['gt']=[900,800,777,666,1.4]
        b=core.enrich(source,{})
        self.assertEqual([r['temporal'] for r in a],[r['temporal'] for r in b])

    def test_isotropic_and_width_angle_equivalence(self):
        source=[row(i,100+i) for i in range(6)]
        a=core.enrich(source,{})
        for r in source:
            r['pred'][2:4]=[r['pred'][3]*2,r['pred'][2]*2]
            r['pred'][4]+=math.pi/2; r['image_size']=[2000,2000]
        b=core.enrich(source,{})
        for x,y in zip(a,b):
            if x['temporal']['available']:self.assertAlmostEqual(x['temporal']['primary'],y['temporal']['primary'])

    def test_missing_gap_boundary_and_resolution_reset(self):
        source=[row(i) for i in range(10)];source[4]=row(4,None)
        result=core.enrich(source,{'real_seq01':[9]})
        self.assertTrue(result[3]['temporal']['available'])
        self.assertFalse(result[4]['temporal']['available'])
        self.assertFalse(result[7]['temporal']['available'])
        self.assertTrue(result[8]['temporal']['available'])
        self.assertFalse(result[9]['temporal']['available'])
        t=core.CausalSizeEvidence()
        for i in range(4):t.observe([0,0,100,50,0,.8],i,[1000,1000])
        self.assertFalse(t.observe([0,0,100,50,0,.8],6,[1000,1000])['available'])
        self.assertFalse(t.observe([0,0,100,50,0,.8],7,[900,900])['available'])

    def test_duplicate_and_TEST_rejected(self):
        with self.assertRaises(ValueError):core.enrich([row(1),row(1)],{})
        with self.assertRaises(ValueError):core.enrich([row(1,split='test')],{})
        with self.assertRaises(ValueError):core.fit_cutoff(core.enrich([row(i,split='val') for i in range(4)],{}))

    def test_acquisition_join_breaks_reported_runs_without_renumbering(self):
        a=core.enrich([row(i,size=70) for i in range(8)],{'real_seq01':[4]})
        s=core.summarize(a,{r['image'] for r in a})
        runs=s['runs']['incorrect_acceptance']['segments']
        self.assertEqual([(r['start_frame'],r['end_frame'],r['length']) for r in runs],[(0,3,4),(4,7,4)])
        self.assertEqual(s['states']['FA'],8)

    def test_constant_wrong_sizes_are_not_detected(self):
        a=core.enrich([row(i,size=70) for i in range(10)],{})
        self.assertTrue(all(r['temporal']['primary']==0 for r in a[3:]))
        p=core.raw_error_profile(a,0)
        self.assertEqual(p['longest_error_run'],10)
        self.assertEqual(p['categories']['long5plus']['temporal_below_cutoff'],7)

    def test_warmup_preserves_existing_flag(self):
        a=core.enrich([row(i) for i in range(4)],{})
        a[-1]['temporal']['primary']=.2
        kept=core.veto_accept(a,.1)
        self.assertEqual(kept,{r['image'] for r in a[:3]})
        self.assertEqual(core.veto_accept([core.enrich([row(0,None)],{})[0]],.1),set())

    def test_whole_ties_and_count_bounds(self):
        rows=core.enrich([row(0,gt=100),row(1,gt=200),row(2,gt=100)],{})
        for r in rows:r['risks']['temporal']=0.
        p=core.point(rows,'temporal',correct=1)
        self.assertFalse(p['exact_CR']);self.assertEqual(p['states']['CR'],2)
        c=core.point(rows,'temporal',count=1)
        self.assertEqual(c['tie_bounds']['FA_min'],0);self.assertEqual(c['tie_bounds']['FA_max'],1)

    def test_cutoff_is_single_group_protective_and_unchanged_by_VAL(self):
        a=core.enrich([row(i,100+i) for i in range(30)],{})
        for r in a:r['gt']=r['pred'][:5]
        fit=core.fit_cutoff(a)
        self.assertEqual(fit['risk_le'],max(v['cutoff'] for v in fit['group_protection'].values()))
        before=deepcopy(fit);core.evaluate(core.enrich([row(i,90,100,'val') for i in range(8)],{}),fit['risk_le'])
        self.assertEqual(before,fit)


if __name__=='__main__':unittest.main()
