import inspect
import json
import math
import unittest
from copy import deepcopy
from pathlib import Path
import numpy as np
from crane_project.utils import port_reliability_size_temporal_interaction_v1 as c
from test_port_reliability_temporal_fusion_v1 import toy
from test_port_reliability_temporal_evidence_v1 import row


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train=toy();cls.models=c.base.fit(cls.train)
        cls.scored=c.base.score(cls.train,cls.models)

    def test_scalar_derivatives_finite_difference(self):
        a=np.array([-.7,.9,2.]);phi=np.array([-.3,1.2,.5]);y=np.array([0.,1.,0.]);w=np.array([.25,.5,.25])
        g=.2;eps=1e-5;loss,grad,h=c.objective(g,a,phi,y,w)
        left=c.objective(g-eps,a,phi,y,w);right=c.objective(g+eps,a,phi,y,w)
        self.assertAlmostEqual(grad,(right[0]-left[0])/(2*eps),places=8)
        self.assertAlmostEqual(h,(right[1]-left[1])/(2*eps),places=8)
        self.assertGreater(h,0)

    def test_only_gamma_fit_and_base_not_mutated(self):
        before=deepcopy(self.models);fit=c.fit(self.scored,self.models)
        self.assertEqual(before,self.models);self.assertEqual(fit['settings'],c.SETTINGS)
        self.assertLessEqual(abs(fit['final_gradient']),1e-8)
        self.assertLessEqual(fit['final_objective'],fit['initial_objective'])
        self.assertNotIn('weights',fit);self.assertNotIn('mean',fit)

    def test_neutral_exact_identity(self):
        for r in self.scored:
            self.assertEqual(c.online_risk(r['pred'],r['image_size'],r['temporal'],self.models,0.),r['risks']['fusion4'])

    def test_fallback_unchanged_with_nonzero_gamma(self):
        r=self.scored[0]
        self.assertFalse(r['temporal']['available'])
        self.assertEqual(c.online_risk(r['pred'],r['image_size'],r['temporal'],self.models,.8),r['risks']['fusion4'])

    def test_no_interaction_without_history(self):
        r=self.scored[0]
        with self.assertRaises(ValueError):c.terms(r['pred'],r['image_size'],r['temporal'],self.models)

    def test_train_roles_and_identity_enforced(self):
        for split in ['val','test']:
            rows=deepcopy(self.scored);rows[0]['split']=split;rows[0]['reliability_role']=split
            with self.assertRaises(ValueError):c.fit(rows,self.models)
        with self.assertRaises(ValueError):c.fit(self.scored[:-1],self.models)

    def test_GT_does_not_change_score(self):
        rows=deepcopy(self.scored)
        for r in rows:r['gt']=[0,0,999,333,0]
        self.assertEqual([r['risks'] for r in c.score(rows,self.models,.2)],
                         [r['risks'] for r in c.score(self.scored,self.models,.2)])
        self.assertNotIn('gt',inspect.signature(c.CausalInteraction.decide).parameters)

    def test_nonzero_matches_explicit_formula(self):
        r=self.scored[4];logit,phi=c.terms(r['pred'],r['image_size'],r['temporal'],self.models)
        self.assertAlmostEqual(c.online_risk(r['pred'],r['image_size'],r['temporal'],self.models,.2),1/(1+math.exp(-(logit+.2*phi))))

    def test_finite_gamma_and_stable_extreme_logits(self):
        r=self.scored[4]
        with self.assertRaises(ValueError):c.online_risk(r['pred'],r['image_size'],r['temporal'],self.models,float('nan'))
        value=c.online_risk(r['pred'],r['image_size'],r['temporal'],self.models,100000.)
        self.assertTrue(math.isfinite(value));self.assertTrue(0<=value<=1)

    def test_missing_and_gap_reset(self):
        rows=[row(i) for i in range(8)];rows[4]=row(4,None)
        enriched=c.base.level.enrich(rows,{})
        self.assertIsNone(c.online_risk(enriched[4]['pred'],enriched[4]['image_size'],enriched[4]['temporal'],self.models,.2))
        self.assertFalse(enriched[7]['temporal']['available'])
        tracker=c.base.level.CausalSizeEvidence()
        for i in range(4):tracker.observe(row(i)['pred'],i,[1000,1000])
        self.assertFalse(tracker.observe(row(9)['pred'],9,[1000,1000])['available'])

    def test_size_swap_pi_and_isotropic_invariance(self):
        original=c.score(self.scored,self.models,.2);changed=deepcopy(self.scored)
        for r in changed:
            r['pred'][2:4]=[2*r['pred'][3],2*r['pred'][2]];r['pred'][4]+=math.pi/2+math.pi;r['image_size']=[2000,2000]
            r['risks']['fusion4']=c.base.online_risks(r['pred'],r['image_size'],r['temporal'],self.models)['fusion4']
        altered=c.score(changed,self.models,.2)
        np.testing.assert_allclose([r['risks'][c.METHOD] for r in original],[r['risks'][c.METHOD] for r in altered],atol=1e-12,rtol=0)

    def test_preserve_components_output_and_future_causality(self):
        policy=json.loads((Path(__file__).resolve().parents[1]/'work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1/fit/policy.json').read_text())['simple_policy']
        api=c.CausalInteraction(self.models,.2,.5,policy);old=c.base.simple.SimpleComponentReliability(policy)
        for i in range(6):
            pred=None if i==4 else [0,0,100,50,0,.8];before=deepcopy(pred)
            a=api.decide(pred,[1000,1000],i);b=old.decide(pred,[1000,1000])
            for k in ('raw_b_output','center_accepted','angle_accepted','risks'):self.assertEqual(a[k],b[k])
            self.assertEqual(pred,before)
        raw=[row(i) for i in range(8)];original=c.base.level.enrich(raw,{})
        raw[-1]['pred'][2]=888;changed=c.base.level.enrich(raw,{})
        for a,b in zip(original[:-1],changed[:-1]):
            self.assertEqual(c.online_risk(a['pred'],a['image_size'],a['temporal'],self.models,.2),c.online_risk(b['pred'],b['image_size'],b['temporal'],self.models,.2))

    def test_score_no_mutation_or_baseline_risk_replacement(self):
        saved=deepcopy(self.scored);c.score(self.scored,self.models,.2);self.assertEqual(saved,self.scored)
        changed=deepcopy(self.scored);changed[4]['risks']['fusion4']+=.001
        with self.assertRaises(ValueError):c.score(changed,self.models,.2)

    def test_gate_JSON_roundtrip(self):
        scored=c.score(self.scored,self.models,.2)
        for r in scored:r['split']='val';r['reliability_role']='val'
        result=c.evaluate(scored,.8,.8)
        a=c.gate(result);b=c.gate(json.loads(json.dumps(result,sort_keys=True)))
        norm=lambda g:sorted(json.dumps(x,sort_keys=True) for x in g['failures'])
        self.assertEqual(norm(a),norm(b));self.assertFalse(a['candidate_adopted'])

    def test_independent_stationarity_rejects_gamma_tamper(self):
        from crane_project.tools.review_port_reliability_size_temporal_interaction_v1 import scalar_check
        fit=c.fit(self.scored,self.models)
        supported=[r for r in self.scored if r['temporal']['available']]
        self.assertLessEqual(abs(scalar_check(fit,supported,self.models)),1.1e-8)
        fit['gamma']+=.1
        with self.assertRaises(ValueError):scalar_check(fit,supported,self.models)

    def test_resolution_and_acquisition_boundary_reset(self):
        tracker=c.base.level.CausalSizeEvidence()
        for i in range(4):tracker.observe(row(i)['pred'],i,[1000,1000])
        self.assertFalse(tracker.observe(row(4)['pred'],4,[2000,2000])['available'])
        raw=[row(i) for i in range(120,127)]
        for r in raw:r['sequence']='real_seq12'
        enriched=c.base.level.enrich(raw,{'real_seq12':[122]})
        self.assertFalse(enriched[4]['temporal']['available'])
        self.assertTrue(enriched[5]['temporal']['available'])

    def test_metric_histogram_roundtrip_keeps_counts_exact(self):
        from crane_project.tools.review_port_reliability_size_temporal_interaction_v1 import equivalent_metrics
        raw={'histogram':{1:3,2:4},'states':{'FA':3},'image':'a'}
        saved=json.loads(json.dumps(raw,sort_keys=True))
        self.assertTrue(equivalent_metrics(saved,raw))
        saved['states']['FA']=4;self.assertFalse(equivalent_metrics(saved,raw))

    def test_history_cross_platform_tolerance_keeps_frames_exact(self):
        from crane_project.tools.review_port_reliability_conditional_evidence_v1 import equivalent_payload
        a={'primary':.1,'available':True,'history_frames':[1,2,3]};b=deepcopy(a)
        b['primary']=np.nextafter(.1,1.)
        self.assertTrue(equivalent_payload(a,b))
        b['history_frames'][-1]=4;self.assertFalse(equivalent_payload(a,b))


if __name__=='__main__':unittest.main()
