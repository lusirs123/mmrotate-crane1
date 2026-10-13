import inspect
import io
import json
import math
import unittest
from copy import deepcopy
from pathlib import Path
import numpy as np
from crane_project.utils import port_reliability_image_temporal_crossfit_v1 as c
from test_port_reliability_temporal_fusion_v1 import toy
from test_port_reliability_temporal_evidence_v1 import row


def fixture():
    rows=toy()
    for i,r in enumerate(rows):r['sequence']='seq'+str(i//10)
    return rows


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=fixture();cls.supported=[r for r in cls.rows if r['temporal']['available']]
        cls.rp=np.linspace(.2,.8,len(cls.supported));cls.ri=np.linspace(.8,.2,len(cls.supported))
        cls.fit=c.fit_fusion(cls.supported,cls.rp,cls.ri)

    def test_folds_exclude_whole_video_once(self):
        folds=c.folds(self.rows);seen=[]
        for f in folds:
            self.assertFalse(set(f['fit'])&set(f['test']))
            self.assertTrue(all(self.rows[i]['sequence']!=f['heldout'] for i in f['fit']))
            self.assertTrue(all(self.rows[i]['sequence']==f['heldout'] for i in f['test']));seen+=f['test']
        self.assertEqual(sorted(seen),list(range(len(self.rows))))

    def test_roles_duplicates_and_single_class_folds_rejected(self):
        for role in ('val','test'):
            rows=deepcopy(self.rows);rows[0]['reliability_role']=role
            with self.assertRaises(ValueError):c.folds(rows)
        with self.assertRaises(ValueError):c.folds(self.rows+[self.rows[0]])
        with self.assertRaises(ValueError):c.fit_fusion(self.rows,self.rp,self.ri)

    def test_fusion_gradient_hessian(self):
        X,off,_=c.fusion_design([.2,.3,.8],[.7,.5,.2]);w=np.array([.1,.2,-.1]);y=np.array([0,1,1]);sw=np.array([.5,.25,.25]);loss,g,H=c.objective(w,X,off,y,sw);eps=1e-5
        for j in range(3):
            d=np.eye(3)[j]*eps;a=c.objective(w-d,X,off,y,sw);b=c.objective(w+d,X,off,y,sw)
            self.assertAlmostEqual(g[j],(b[0]-a[0])/(2*eps),places=8);np.testing.assert_allclose(H[:,j],(b[1]-a[1])/(2*eps),atol=1e-8,rtol=0)

    def test_zero_residual_exact_and_static_fallback(self):
        model=deepcopy(self.fit);model['weights']=[0,0,0]
        self.assertEqual(c.fused_risk(.314159,.9,model,True),.314159)
        self.assertEqual(c.fused_risk(.314159,None,self.fit,False),.314159)
        self.assertIsNone(c.fused_risk(None,None,self.fit,False))

    def test_risk_extremes_and_clipping_fixed(self):
        np.testing.assert_allclose(c.bounded_logit([0,.5,1]),[-12,0,12],atol=1e-10)
        with self.assertRaises(ValueError):c.bounded_logit([float('nan')])
        with self.assertRaises(ValueError):c.bounded_logit([1.1])

    def test_fit_train_only_stationarity(self):
        self.assertLessEqual(self.fit['max_abs_gradient'],1e-8)
        self.assertLessEqual(self.fit['final_objective'],self.fit['initial_objective'])
        rows=deepcopy(self.supported);rows[0]['split']='val'
        with self.assertRaises(ValueError):c.fit_fusion(rows,self.rp,self.ri)
        from crane_project.tools.review_port_reliability_image_temporal_crossfit_v1 import fusion_check
        self.assertLessEqual(fusion_check(self.fit,self.supported,self.rp,self.ri),1.1e-8)
        m=deepcopy(self.fit);m['weights'][1]+=.2
        with self.assertRaises(ValueError):fusion_check(m,self.supported,self.rp,self.ri)

    def test_feature_membership_and_descriptor_validation(self):
        from crane_project.tools.run_port_reliability_image_temporal_crossfit_v1 import feature_matrix
        import tempfile
        matrix=np.array([np.r_[c.temporal.base.simple.descriptor(r['pred'],r['image_size']),np.zeros(256)] for r in self.rows])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'x.npz';np.savez(p,features=matrix,images=[r['image'] for r in self.rows]);m,rows=feature_matrix(p,self.rows)
            np.testing.assert_array_equal(m,matrix)
            matrix[0,0]+=.1;np.savez(p,features=matrix,images=[r['image'] for r in self.rows])
            with self.assertRaises(ValueError):feature_matrix(p,self.rows)

    def test_score_no_GT_no_mutation_no_baseline_replacement(self):
        rows=deepcopy(self.rows)
        for i,r in enumerate(rows):r['risks']={'partial_auc4':.4,'fusion4':.5,'full_simple':.6,'score_only':.7,'static3':.6}
        image={r['image']:.3 for r in rows};models={a:self.fit for a in c.ARMS};old=deepcopy(rows)
        a=c.score(rows,image,models);self.assertEqual(old,rows)
        for r in rows:r['gt']=[0,0,999,333,0]
        b=c.score(rows,image,models);self.assertEqual([r['risks'] for r in a],[r['risks'] for r in b])
        for r,o in zip(a,old):self.assertTrue(all(r['risks'][k]==v for k,v in o['risks'].items()))

    def test_online_preserves_output_and_GT_free_signature(self):
        base=c.temporal.base.fit(toy());partial=base['fusion4'];pca=dict(mean=np.zeros(2304).tolist(),components=np.eye(256,2304).tolist())
        head={'network.0.weight':np.zeros((16,258)).tolist(),'network.0.bias':np.zeros(16).tolist(),'network.2.weight':np.zeros((8,16)).tolist(),'network.2.bias':np.zeros(8).tolist(),'network.4.weight':np.zeros((1,8)).tolist(),'network.4.bias':[0.],'beta':[0.]}
        norm=dict(mean=np.zeros(259).tolist(),scale=np.ones(259).tolist())
        policy=json.loads((Path(__file__).resolve().parents[1]/'work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1/fit/policy.json').read_text())['simple_policy']
        api=c.CausalImageTemporal(base,partial,pca,head,norm,self.fit,.5,policy);old=c.temporal.base.simple.SimpleComponentReliability(policy)
        self.assertNotIn('gt',inspect.signature(api.decide).parameters)
        for i in range(6):
            pred=None if i==4 else [0,0,100,50,0,.8];saved=deepcopy(pred)
            a=api.decide(pred,[1000,1000],i,np.zeros(2304));b=old.decide(pred,[1000,1000])
            for key in ('raw_b_output','center_accepted','angle_accepted','risks'):self.assertEqual(a[key],b[key])
            self.assertEqual(pred,saved)

    def test_causal_future_and_resets(self):
        raw=[row(i) for i in range(8)];a=c.temporal.base.level.enrich(raw,{})
        raw[-1]['pred'][2]=777;b=c.temporal.base.level.enrich(raw,{});self.assertEqual(a[:-1],b[:-1])
        tracker=c.temporal.base.level.CausalSizeEvidence()
        for i in range(4):tracker.observe(row(i)['pred'],i,[1000,1000])
        self.assertFalse(tracker.observe(None,4,[1000,1000])['available'])
        self.assertFalse(tracker.observe(row(9)['pred'],9,[2000,2000])['available'])
        raw=[row(i) for i in range(120,127)]
        for r in raw:r['sequence']='real_seq12'
        a=c.temporal.base.level.enrich(raw,{'real_seq12':[122]});self.assertFalse(a[4]['temporal']['available']);self.assertTrue(a[5]['temporal']['available'])

    def test_invariant_swap_pi_isotropic_descriptors(self):
        a=c.temporal.base.simple.descriptor([0,0,100,50,.2,.8],[1000,1000])
        b=c.temporal.base.simple.descriptor([0,0,100,200,.2+math.pi*1.5,.8],[2000,2000])
        np.testing.assert_allclose(a,b,atol=1e-12,rtol=0)

    def test_independent_head_replay(self):
        from crane_project.tools.review_port_reliability_image_temporal_crossfit_v1 import head_predict
        rng=np.random.RandomState(1);matrix=rng.randn(4,259)
        head={'model':{'network.0.weight':rng.randn(16,258).tolist(),'network.0.bias':rng.randn(16).tolist(),'network.2.weight':rng.randn(8,16).tolist(),'network.2.bias':rng.randn(8).tolist(),'network.4.weight':rng.randn(1,8).tolist(),'network.4.bias':[.2],'beta':[.1]},'normalizer':dict(mean=np.zeros(259).tolist(),scale=np.ones(259).tolist())}
        expected=c.temporal.base.simple.sigmoid(c.image.numpy_logits(head['model'],matrix,head['normalizer']))
        np.testing.assert_allclose(head_predict(head,matrix),expected,atol=1e-12,rtol=0)

    def test_gate_json_counts_exact(self):
        from crane_project.tools.review_port_reliability_image_temporal_crossfit_v1 import equivalent
        a={'states':{'FA':1},'histogram':{2:1}};b=json.loads(json.dumps(a));self.assertTrue(equivalent(a,b));b['states']['FA']=2;self.assertFalse(equivalent(a,b))

    def test_torch_heldout_labels_and_features_do_not_affect_fit(self):
        try:import torch
        except ImportError:self.skipTest('Local Torch unavailable; server required')
        from crane_project.utils.port_reliability_image_temporal_crossfit_v1_torch import train_head
        rows=fixture();matrix=np.random.RandomState(1).normal(size=(len(rows),259));split=c.folds(rows)[0]
        torch.set_num_threads(1);a=train_head(matrix,rows,split['fit'],'a','cpu',io.StringIO())
        changed=deepcopy(rows);new=matrix.copy()
        for i in split['test']:changed[i]['gt']=[0,0,999,999,0];new[i]=float('nan')
        b=train_head(new,changed,split['fit'],'b','cpu',io.StringIO())
        self.assertEqual(a,b)


if __name__=='__main__':unittest.main()
