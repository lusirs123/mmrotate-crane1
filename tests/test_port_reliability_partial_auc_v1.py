import inspect
import json
import math
import unittest
from copy import deepcopy
from pathlib import Path
import numpy as np
from crane_project.utils import port_reliability_partial_auc_v1 as c
from test_port_reliability_temporal_fusion_v1 import toy
from test_port_reliability_temporal_evidence_v1 import row


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train=toy();cls.models=c.base.fit(cls.train)
        cls.scored=c.base.score(cls.train,cls.models)
        cls.model=c.fit(cls.scored,cls.models)

    def test_gradient_and_hessian_finite_difference(self):
        rng=np.random.default_rng(23);good=rng.normal(size=(9,4));wrong=rng.normal(size=(3,4));v=np.array([.2,-.1,.3,.1,1.])
        loss,g,H=c.objective(v,good,wrong);eps=1e-5
        for j in range(5):
            d=np.eye(5)[j]*eps;l=c.objective(v-d,good,wrong);r=c.objective(v+d,good,wrong)
            self.assertAlmostEqual(g[j],(r[0]-l[0])/(2*eps),places=7)
            np.testing.assert_allclose(H[:,j],(r[1]-l[1])/(2*eps),atol=2e-7,rtol=0)
        self.assertGreater(np.linalg.eigvalsh(H).min(),0)

    def test_fit_stationarity_base_immutable(self):
        saved=deepcopy(self.models);model=c.fit(self.scored,self.models)
        self.assertEqual(saved,self.models);self.assertLessEqual(model['max_abs_gradient'],1e-8)
        self.assertEqual(model['settings'],c.SETTINGS);self.assertLessEqual(model['final_objective'],model['initial_objective'])
        self.assertEqual(model['weights'][0],self.models['fusion4']['weights'][0])

    def test_tail_focus_order_and_uniform_ties(self):
        good=np.zeros((20,4));good[0,0]=4;wrong=np.zeros((2,4));v=np.array([1,0,0,0,1])
        h=np.logaddexp(0,(good[:,None,:]-wrong[None,:,:])@v[:4]).mean(axis=1)
        a=c.base.simple.sigmoid((h-v[4])/.1)
        self.assertGreater(a[0],a[1]);self.assertEqual(a[1],a[2])

    def test_reject_roles_identity_and_nonfinite(self):
        for role in ('val','test'):
            rows=deepcopy(self.scored);rows[0]['split']=role;rows[0]['reliability_role']=role
            with self.assertRaises(ValueError):c.fit(rows,self.models)
        with self.assertRaises(ValueError):c.fit(self.scored[:-1],self.models)
        with self.assertRaises(ValueError):c.objective([0,0,0,0,float('nan')],np.ones((3,4)),np.ones((2,4)))

    def test_fallback_missing_and_tau_not_online(self):
        r=self.scored[0];self.assertFalse(r['temporal']['available'])
        self.assertEqual(c.online_risk(r['pred'],r['image_size'],r['temporal'],self.models,self.model),r['risks']['fusion4'])
        self.assertIsNone(c.online_risk(None,r['image_size'],r['temporal'],self.models,self.model))
        m=deepcopy(self.model);m['tail_threshold']=100000
        self.assertEqual(c.score(self.scored,self.models,m),c.score(self.scored,self.models,self.model))

    def test_gt_ignored_and_input_unmodified(self):
        before=deepcopy(self.scored);changed=deepcopy(self.scored)
        for r in changed:r['gt']=[0,0,999,111,0]
        a=c.score(self.scored,self.models,self.model);b=c.score(changed,self.models,self.model)
        self.assertEqual(before,self.scored);self.assertEqual([r['risks'] for r in a],[r['risks'] for r in b])
        self.assertNotIn('gt',inspect.signature(c.CausalPartialAUC.decide).parameters)
        bad=deepcopy(self.scored);bad[4]['risks']['fusion4']+=.001
        with self.assertRaises(ValueError):c.score(bad,self.models,self.model)

    def test_exact_neutral_restore_using_base_weights(self):
        for r in self.scored:self.assertEqual(c.online_risk(r['pred'],r['image_size'],r['temporal'],self.models,self.models['fusion4']),r['risks']['fusion4'])

    def test_swap_pi_isotropic_and_future_causality(self):
        original=c.score(self.scored,self.models,self.model);changed=deepcopy(self.scored)
        for r in changed:
            r['pred'][2:4]=[2*r['pred'][3],2*r['pred'][2]];r['pred'][4]+=math.pi*1.5;r['image_size']=[2000,2000]
            r['risks']['fusion4']=c.base.online_risks(r['pred'],r['image_size'],r['temporal'],self.models)['fusion4']
        altered=c.score(changed,self.models,self.model)
        np.testing.assert_allclose([r['risks'][c.METHOD] for r in original],[r['risks'][c.METHOD] for r in altered],atol=1e-12,rtol=0)
        raw=[row(i) for i in range(8)];a=c.base.level.enrich(raw,{})
        raw[-1]['pred'][2]=888;b=c.base.level.enrich(raw,{})
        self.assertEqual(a[:-1],b[:-1])

    def test_online_preserves_center_angle_score(self):
        policy=json.loads((Path(__file__).resolve().parents[1]/'work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1/fit/policy.json').read_text())['simple_policy']
        api=c.CausalPartialAUC(self.models,self.model,.5,policy);old=c.base.simple.SimpleComponentReliability(policy)
        for i in range(6):
            pred=None if i==4 else [0,0,100,50,0,.8];saved=deepcopy(pred)
            a=api.decide(pred,[1000,1000],i);b=old.decide(pred,[1000,1000])
            for k in ('raw_b_output','center_accepted','angle_accepted','risks'):self.assertEqual(a[k],b[k])
            self.assertEqual(saved,pred)

    def test_history_resets_missing_gap_resolution_join(self):
        raw=[row(i) for i in range(8)];raw[4]=row(4,None);a=c.base.level.enrich(raw,{})
        self.assertFalse(a[7]['temporal']['available'])
        tracker=c.base.level.CausalSizeEvidence()
        for i in range(4):tracker.observe(row(i)['pred'],i,[1000,1000])
        self.assertFalse(tracker.observe(row(9)['pred'],9,[1000,1000])['available'])
        self.assertFalse(tracker.observe(row(10)['pred'],10,[2000,2000])['available'])
        raw=[row(i) for i in range(120,127)]
        for r in raw:r['sequence']='real_seq12'
        a=c.base.level.enrich(raw,{'real_seq12':[122]});self.assertFalse(a[4]['temporal']['available']);self.assertTrue(a[5]['temporal']['available'])

    def test_pauc_exact_ties_and_orientation(self):
        from crane_project.tools.review_port_reliability_partial_auc_v1 import exact_pauc
        rows=deepcopy(self.scored)
        for r in rows:r['risks'][c.METHOD]=.5
        self.assertAlmostEqual(c.partial_auc(rows,c.METHOD),.025)
        self.assertAlmostEqual(exact_pauc(rows,c.METHOD),.025)
        for r in rows:r['risks'][c.METHOD]=float(c.base.level.metrics.bad(r))
        self.assertAlmostEqual(c.partial_auc(rows,c.METHOD),1.)
        self.assertAlmostEqual(exact_pauc(rows,c.METHOD),1.)

    def test_independent_stationarity_and_tamper(self):
        from crane_project.tools.review_port_reliability_partial_auc_v1 import fit_check
        rows=[r for r in self.scored if r['temporal']['available']]
        self.assertLessEqual(fit_check(self.model,rows,self.models),1.1e-8)
        m=deepcopy(self.model);m['weights'][1]+=.1
        with self.assertRaises(ValueError):fit_check(m,rows,self.models)

    def test_gate_and_metrics_json_exact_counts(self):
        from crane_project.tools.review_port_reliability_partial_auc_v1 import equivalent_metrics
        rows=c.score(self.scored,self.models,self.model)
        for r in rows:r['split']='val';r['reliability_role']='val'
        a=c.evaluate(rows,.8,.8);b=json.loads(json.dumps(a,sort_keys=True))
        self.assertTrue(equivalent_metrics(a,b));self.assertEqual(c.gate(a)['passed'],c.gate(b)['passed'])
        b['groups']['all']['candidate']['states']['FA']+=1;self.assertFalse(equivalent_metrics(a,b))


if __name__=='__main__':unittest.main()
