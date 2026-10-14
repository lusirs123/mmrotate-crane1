import inspect
import json
import math
from copy import deepcopy
from pathlib import Path
import unittest
import numpy as np
from crane_project.utils import port_reliability_candidate_support_v1 as c
from test_port_reliability_temporal_fusion_v1 import toy


def pool():
    return [[0,0,100,50,0,.9],[0,0,110,55,0,.8],
            [1,1,120,60,0,.7],[-1,-1,130,65,0,.6]]


class Tests(unittest.TestCase):
    def test_actual_source_excluded_not_nearest(self):
        p=pool();p[1]=p[0].copy();p[1][5]=.8
        e=c.evidence(p[0],p,0)
        self.assertEqual([n['index'] for n in e['neighbors']],[1,2,3])
        self.assertTrue(e['available'])

    def test_signed_median_and_count(self):
        p=pool();e=c.evidence(p[0],p,0)
        np.testing.assert_allclose(e['features'],[math.log(1.2)]*2,atol=1e-15)
        self.assertFalse(c.evidence(p[0],p[:3],0)['available'])
        self.assertIsNone(c.evidence(p[0],p[:3],0)['features'])

    def test_rotated_center_and_angle_filters(self):
        p=pool();p[1][0]=51;p[2][1]=26;p[3][4]=math.radians(16)
        self.assertEqual(c.evidence(p[0],p,0)['eligible'],0)
        p=pool();p[0][4]=math.pi/2
        for b in p[1:]:b[4]=math.pi/2;b[0]=20;b[1]=40
        self.assertEqual(c.evidence(p[0],p,0)['eligible'],3)

    def test_swap_pi_scale_invariance(self):
        p=pool();old=c.evidence(p[0],p,0)['features']
        changed=deepcopy(p)
        for b in changed:b[:4]=[2*b[0],2*b[1],2*b[3],2*b[2]];b[4]+=1.5*math.pi
        np.testing.assert_allclose(c.evidence(changed[0],changed,0)['features'],old,atol=1e-14)

    def test_cap_ties_and_input_immutable(self):
        p=[pool()[0]]+[[0,0,100+i,50,0,.6] for i in range(12)]
        before=deepcopy(p);ev=c.evidence(p[0],p,0)
        self.assertEqual(len(ev['neighbors']),8);self.assertEqual(ev['neighbors'][0]['index'],1)
        self.assertEqual(p,before)

    def test_missing_and_invalid_inputs(self):
        self.assertEqual(c.evidence(None,np.empty((0,6)),None)['reason'],'missing')
        with self.assertRaises(ValueError):c.evidence(None,pool(),None)
        for chosen in (None,-1,9):
            with self.assertRaises(ValueError):c.evidence(pool()[0],pool(),chosen)
        p=pool();p[1][2]=float('nan')
        with self.assertRaises(ValueError):c.evidence(p[0],p,0)

    def test_generalized_objective_matches_original4(self):
        rng=np.random.default_rng(2);g=rng.normal(size=(10,4));w=rng.normal(size=(3,4));v=np.array([.1,.2,-.3,.2,1.])
        for a,b in zip(c.objective(v,g,w),c.parent.objective(v,g,w)):np.testing.assert_allclose(a,b,atol=1e-13)

    def test_six_gradient_hessian(self):
        rng=np.random.default_rng(3);g=rng.normal(size=(10,6));w=rng.normal(size=(3,6));v=rng.normal(size=7)*.1
        loss,grad,H=c.objective(v,g,w)
        for j in range(7):
            delta=np.eye(7)[j]*1e-5;a=c.objective(v-delta,g,w);b=c.objective(v+delta,g,w)
            self.assertAlmostEqual(grad[j],(b[0]-a[0])/2e-5,places=7)
            np.testing.assert_allclose(H[:,j],(b[1]-a[1])/2e-5,atol=3e-7,rtol=0)

    def enriched(self):
        rows=toy();base=c.parent.base.fit(rows);rows=c.parent.base.score(rows,base)
        for r in rows:
            r['candidate_evidence']=c.evidence(r['pred'],pool(),0)
            r['risks'][c.parent.METHOD]=r['risks']['fusion4']
        return rows,base

    def test_paired_fit_and_gt_free_score(self):
        rows,base=self.enriched();before=deepcopy(rows)
        models={name:c.fit(rows,dim,base['fusion4']['weights'][0]) for name,dim in ((c.CONTROL,4),(c.METHOD,6))}
        self.assertEqual(models[c.CONTROL]['train_images'],models[c.METHOD]['train_images'])
        self.assertLessEqual(models[c.METHOD]['max_abs_gradient'],1e-8)
        a=c.score(rows,models);changed=deepcopy(rows)
        for r in changed:r['gt']=[0,0,9999,1111,0]
        b=c.score(changed,models)
        self.assertEqual([r['risks'] for r in a],[r['risks'] for r in b]);self.assertEqual(rows,before)
        self.assertNotIn('gt',inspect.signature(c.CausalCandidateSupport.decide).parameters)
        rows[0]['reliability_role']='val'
        with self.assertRaises(ValueError):c.fit(rows,6,0.)

    def test_fallback_decisions_preserved_even_different_threshold(self):
        rows,base=self.enriched();m=c.fit(rows,6,0.)
        rows[3]['candidate_evidence']['available']=False
        for r in rows:r['risks'][c.METHOD]=.99
        ids=c.accepted(rows,c.METHOD,0.,1.)
        self.assertIn(rows[0]['image'],ids);self.assertIn(rows[3]['image'],ids)
        self.assertNotIn(rows[4]['image'],ids)

    def test_online_geometry_score_center_angle_preserved(self):
        rows,models=self.enriched();pm=c.parent.fit(rows,models);m=c.fit(rows,6,0.)
        policy=json.loads((Path(__file__).resolve().parents[1]/'work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1/fit/policy.json').read_text())['simple_policy']
        api=c.CausalCandidateSupport(models,pm,.5,m,.5,policy);old=c.parent.CausalPartialAUC(models,pm,.5,policy)
        for i in range(8):
            pred=pool()[0] if i!=5 else None;p=pool() if pred else np.empty((0,6));chosen=0 if pred else None
            saved=deepcopy(pred);a=api.decide(pred,[1000,1000],i,p,chosen);b=old.decide(pred,[1000,1000],i)
            for key in ('center_accepted','angle_accepted','raw_b_output','risks'):self.assertEqual(a[key],b[key])
            self.assertEqual(pred,saved)
            if not pred:self.assertFalse(a['size_accepted'])

    def test_no_candidate_screen_stops_without_fit(self):
        rows,_=self.enriched();rows=(rows*math.ceil(2558/len(rows)))[:2558]
        rows=[deepcopy(r) for r in rows]
        for i,r in enumerate(rows):
            r['candidate_evidence'].update(available=False,features=None)
            if i>1200:r['domain']='sim';r['sequence']='sim_seq08'
        s=c.screen(rows)
        self.assertFalse(s['passed']);self.assertIn('coverage_real',s['failures'])
        self.assertIn('no_consistent_two_video_signal',s['failures'])

    def test_independent_geometry_audit_and_tamper(self):
        from crane_project.tools.review_port_reliability_candidate_support_v1 import independent_evidence
        p=pool();r=dict(pred=p[0],b_original=p[0],candidate_pool=p,candidate_chosen=0,
            candidate_trace={'score_thr':.05},candidate_evidence=c.evidence(p[0],p,0))
        independent_evidence(r)
        changed=deepcopy(r);changed['candidate_evidence']['features'][0]+=.1
        with self.assertRaises(ValueError):independent_evidence(changed)
        changed=deepcopy(r);changed['candidate_chosen']=1
        with self.assertRaises(ValueError):independent_evidence(changed)

    def test_independent_fit_stationarity(self):
        from crane_project.tools.review_port_reliability_candidate_support_v1 import fit_check
        rows,base=self.enriched();m=c.fit(rows,6,base['fusion4']['weights'][0])
        self.assertLessEqual(fit_check(m,rows,6),1.1e-8)
        changed=deepcopy(m);changed['weights'][1]+=.1
        with self.assertRaises(ValueError):fit_check(changed,rows,6)

    def test_screen_requires_real_video_increment_not_pooled_only(self):
        rows,_=self.enriched();rows=[deepcopy(r) for r in (rows*86)[:2558]]
        for i,r in enumerate(rows):
            r['domain']='real' if i<1900 else 'sim'
            r['sequence']='real_seq01' if i<1900 else 'sim_seq08'
            r['temporal']['available']=True
            value=float(c.parent.base.level.metrics.bad(r));r['candidate_evidence']['features']=[value,value]
        s=c.screen(rows);self.assertFalse(s['passed']);self.assertEqual(s['pooled_AUC'],[1.,1.])
        self.assertIn('no_consistent_two_video_signal',s['failures'])


if __name__=='__main__':unittest.main()
