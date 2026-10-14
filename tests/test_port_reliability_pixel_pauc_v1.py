import ast
from copy import deepcopy
import inspect
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from crane_project.utils import port_reliability_pixel_pauc_v1 as c
from test_port_reliability_temporal_fusion_v1 import toy


def evidence(a=.1, b=.2, supported=True):
    return {k:dict(supported=supported,mismatch=v if supported else None) for k,v in [('L',a),('S',b)]}


def model(dim):
    return dict(dim=dim,mean=[0.]*dim,scale=[1.]*dim,weights=[-.5]+[.1]*dim)


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=c.parent.base.fit(toy())
        cls.rows=c.parent.base.score(toy(),cls.base)
        cls.old=c.parent.fit(cls.rows,cls.base)
        cls.rows=c.parent.score(cls.rows,cls.base,cls.old)
        for r in cls.rows:
            r['risks']['crossfit_fusion']=r['risks']['partial_auc4']
            r['fallback_decision']=r['pred'] is not None and r['risks']['partial_auc4']<=c.FALLBACK_CUTOFF
            r['pixel_evidence']=evidence()
        root=Path(__file__).resolve().parents[1]
        cls.policy=json.loads((root/'work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1/fit/policy.json').read_text())['simple_policy']

    def test_four_objective_exact_parent_restore(self):
        rng=np.random.default_rng(12);g=rng.normal(size=(7,4));b=rng.normal(size=(3,4));v=rng.normal(size=5)
        for x,y in zip(c.objective(v,g,b),c.parent.objective(v,g,b)):
            np.testing.assert_allclose(x,y,rtol=0,atol=1e-13)

    def test_six_gradient_hessian(self):
        rng=np.random.default_rng(3);g=rng.normal(size=(6,6));b=rng.normal(size=(3,6));v=rng.normal(size=7)*.2;eps=1e-5
        _,grad,H=c.objective(v,g,b)
        for i in range(7):
            shift=np.eye(7)[i]*eps
            minus=c.objective(v-shift,g,b);plus=c.objective(v+shift,g,b)
            self.assertAlmostEqual(grad[i],(plus[0]-minus[0])/(2*eps),places=7)
            np.testing.assert_allclose(H[:,i],(plus[1]-minus[1])/(2*eps),rtol=0,atol=2e-7)
        self.assertGreater(np.linalg.eigvalsh(H).min(),0.)

    def test_constant_extra_inputs_do_not_change_numerical_control(self):
        rng=np.random.default_rng(19);g=rng.normal(size=(9,4));b=rng.normal(size=(3,4))
        saved=deepcopy((g,b));a=c.fit_arrays(g,b)
        six=c.fit_arrays(np.column_stack((g,np.zeros((9,2)))),np.column_stack((b,np.zeros((3,2)))))
        np.testing.assert_allclose(a['slopes'],six['slopes'][:4],atol=1e-12,rtol=0)
        np.testing.assert_array_equal(six['slopes'][4:],[0.,0.])
        np.testing.assert_array_equal(g,saved[0]);np.testing.assert_array_equal(b,saved[1])
        self.assertLessEqual(len(a['trace']),100)

    def test_nonfinite_empty_wrong_dimension_rejected(self):
        for g,b,v in [(np.ones((3,5)),np.ones((2,5)),np.zeros(6)),(np.ones((3,6)),np.empty((0,6)),np.zeros(7)),(np.full((3,6),np.nan),np.ones((2,6)),np.zeros(7))]:
            with self.assertRaises(ValueError):c.objective(v,g,b)

    def test_online_features_GT_and_group_free(self):
        r=deepcopy(self.rows[4]);a=c.features(r['pred'],r['image_size'],r['temporal'],r['pixel_evidence'],6)
        r['gt']=[0,0,9,8,0];r['sequence']='unused';r['domain']='unused'
        np.testing.assert_array_equal(a,c.features(r['pred'],r['image_size'],r['temporal'],r['pixel_evidence'],6))
        params=inspect.signature(c.CausalPixelPAUC.decide).parameters
        self.assertNotIn('gt',params);self.assertNotIn('sequence',params)

    def test_pixel_missing_and_history_missing_hold_exact_old_flag(self):
        for mode in ('pixel','history'):
            r=deepcopy(self.rows[4]);r['risks']['partial_auc4']=.45
            if mode=='pixel':r['pixel_evidence']=evidence(supported=False)
            else:r['temporal']['available']=False
            scored=c.score([r],{c.CONTROL:model(4),c.METHOD:model(6)})
            for name in (c.CONTROL,c.METHOD):
                self.assertEqual(scored[0]['risks'][name],.45)
                self.assertEqual(c.accepted(scored,name,.99),set())
            r['risks']['partial_auc4']=c.FALLBACK_CUTOFF
            self.assertEqual(c.accepted([r],c.METHOD,.001),{r['image']})

    def test_missing_output_separate(self):
        r=deepcopy(self.rows[0]);r['pred']=None;r['pixel_evidence']=None;r['risks']['partial_auc4']=None
        r['temporal']['available']=False
        a=c.score([r],{c.METHOD:model(6)})
        self.assertIsNone(a[0]['risks'][c.METHOD]);self.assertEqual(c.fixed(a),set())
        self.assertEqual(c.locked_point(a,c.METHOD,count=0)['states']['MISSING'],1)

    def test_negative_or_fake_unsupported_values_rejected(self):
        with self.assertRaises(ValueError):c.pixel_supported(evidence(-.1))
        e=evidence();e['S']['supported']=False
        with self.assertRaises(ValueError):c.pixel_supported(e)
        self.assertFalse(c.pixel_supported(None))

    def test_score_preserves_frames_and_original_risks(self):
        old=deepcopy(self.rows);scored=c.score(self.rows,{c.CONTROL:model(4),c.METHOD:model(6)})
        self.assertEqual(old,self.rows)
        for a,b in zip(scored,old):
            self.assertEqual({k:v for k,v in a.items() if k!='risks'},{k:v for k,v in b.items() if k!='risks'})
            self.assertTrue(all(a['risks'][k]==v for k,v in b['risks'].items()))

    def test_locked_count_CR_and_infeasible_CR(self):
        rows=c.score(self.rows,{c.CONTROL:model(4),c.METHOD:model(6)})
        count=len(c.fixed(rows))+3;matched=c.locked_point(rows,c.METHOD,count=count)
        self.assertEqual(matched['accepted_outputs'],count)
        self.assertTrue(matched['fallback_decisions_locked'])
        r=deepcopy(rows[0]);r['temporal']['available']=False;r['risks']['partial_auc4']=.9;r['gt']=r['pred'][:5]
        self.assertFalse(c.locked_point([r],c.METHOD,correct=1)['feasible'])

    def synthetic_val(self):
        rows=[]
        for i in range(887):
            r=deepcopy(self.rows[3+i%27]);r.update(image='real_seq01_%05d'%i,sequence='real_seq01',frame_id=i,split='val',reliability_role='val')
            r['temporal']['acquisition_boundary']=i==0
            # Include fixed accepted and fixed rejected correct fallback observations.
            if i%50==0:
                r['pixel_evidence']=None;r['gt']=r['pred'][:5]
                r['risks']['partial_auc4']=.2 if i%100==0 else .9
            rows.append(r)
        return c.score(rows,{c.CONTROL:model(4),c.METHOD:dict(model(6),weights=[-.5]+[.1]*4+[0.,0.])})

    def test_complete_synthetic_VAL_calibration_evaluation_identical_arms(self):
        rows=self.synthetic_val();cal={name:c.calibrate(rows,name) for name in (c.CONTROL,c.METHOD)}
        e=c.evaluate(rows,cal)
        for name in cal:
            summary=e['groups']['all'][name]['candidate']
            self.assertGreaterEqual(summary['states']['CR'],math.ceil(.95*(summary['states']['CR']+summary['states']['FR'])))
            self.assertEqual(sum(summary['states'].values()),887)
        self.assertFalse(c.gate(e)['passed'])
        self.assertEqual(c.accepted(rows,c.CONTROL,cal[c.CONTROL]['risk_le']),c.accepted(rows,c.METHOD,cal[c.METHOD]['risk_le']))

    def test_online_preserves_geometry_center_angle_and_native_dimensions(self):
        obj=c.CausalPixelPAUC(self.base,self.old,model(6),.5,self.policy)
        gray=np.zeros((1000,1000));gray[0:25,0:80]=200
        pred=[60,50,80,40,0,.8];saved=deepcopy(pred)
        original=c.parent.base.simple.SimpleComponentReliability(self.policy)
        for i in range(5):
            x=obj.decide(pred,[1000,1000],i,gray=gray,reset=i==0);expected=original.decide(pred,[1000,1000])
            for key in ('raw_b_output','center_accepted','angle_accepted','risks'):self.assertEqual(x[key],expected[key])
        self.assertEqual(pred,saved)
        with self.assertRaises(ValueError):obj.decide(pred,[1000,1000],5,gray=np.zeros((10,10)))

    def test_online_fallback_exact_old_partial_not_new_threshold(self):
        obj=c.CausalPixelPAUC(self.base,self.old,model(6),.99,self.policy)
        tracker=c.level.CausalSizeEvidence()
        for i,r in enumerate(self.rows[:6]):
            x=obj.decide(r['pred'],r['image_size'],i,gray=None,reset=i==0)
            point=tracker.observe(r['pred'],i,r['image_size'],reset=i==0)
            risk=c.parent.online_risk(r['pred'],r['image_size'],point,self.base,self.old)
            self.assertEqual(x['size_accepted'],risk<=c.FALLBACK_CUTOFF)
            self.assertFalse(x['size_pixel_pauc_active'])
        self.assertFalse(obj.decide(None,[1000,1000],6)['center_accepted'])

    def test_online_active_matches_offline_and_swap_pi(self):
        from crane_project.utils import port_reliability_pixel_border_v1 as pixels
        gray=np.zeros((1000,1000));gray[75:126,50:151]=200
        pred=[100,100,100,50,0,.8]
        expected=[]
        for p in (pred,[100,100,50,100,math.pi/2,.8],[100,100,100,50,math.pi,.8]):
            obj=c.CausalPixelPAUC(self.base,self.old,model(6),.7,self.policy)
            track=c.level.CausalSizeEvidence()
            for i in range(4):
                result=obj.decide(p,[1000,1000],i,gray=gray,reset=i==0)
                point=track.observe(p,i,[1000,1000],reset=i==0)
            self.assertTrue(result['size_pixel_pauc_active'])
            e=pixels.evidence(gray,p)
            x=c.features(p,[1000,1000],point,e,6)
            risk=float(c.parent.base.simple.linear_risk(model(6),x[None])[0])
            self.assertAlmostEqual(result['size_pixel_pauc_risk'],risk,places=12)
            expected.append(risk)
        np.testing.assert_allclose(expected,[expected[0]]*3,atol=1e-12,rtol=0)

    def test_fit_and_attach_reject_nonTRAIN_and_duplicates(self):
        with self.assertRaises(ValueError):c.fit(self.rows,{},6)
        with self.assertRaises(ValueError):c.prepare(self.rows,self.base)
        with self.assertRaises(ValueError):c.attach(self.rows,[],'TEST')
        rows=self.synthetic_val();pix=[dict(image=r['image'],pred=r['pred'],domain=r['domain'],sequence=r['sequence'],wrong=c.level.metrics.bad(r),accepted=r['fallback_decision'],low_temporal=r['temporal']['available'] and r['temporal']['primary']<=c.parent.base.LEVEL_CUTOFF,evidence=r['pixel_evidence']) for r in rows]
        joined=c.attach(rows,pix,'VAL');self.assertEqual(len(joined),887)
        pix[0]['image']=pix[1]['image']
        with self.assertRaises(ValueError):c.attach(rows,pix,'VAL')

    def test_synthetic_paired_prepare_fit_and_normalizers(self):
        rows=deepcopy(self.rows)
        for i in range(30,2558):
            r=deepcopy(rows[0]);r.update(image='real_seq01_%05d'%i,frame_id=i)
            rows.append(r)
        before=deepcopy(rows)
        prepared=c.prepare(rows,self.base)
        control=c.fit(rows,prepared,4);candidate=c.fit(rows,prepared,6)
        from crane_project.tools.review_port_reliability_pixel_pauc_v1 import fit_check
        self.assertLessEqual(fit_check(control,rows,prepared),1.1e-8)
        self.assertLessEqual(fit_check(candidate,rows,prepared),1.1e-8)
        self.assertEqual(control['train_images'],candidate['train_images'])
        self.assertEqual(control['wrong_images'],candidate['wrong_images'])
        self.assertEqual(control['mean'],candidate['mean'][:4])
        self.assertEqual(control['scale'],candidate['scale'][:4])
        self.assertEqual(candidate['scale'][4:],[1.,1.])
        self.assertEqual(rows,before)
        np.testing.assert_allclose(control['weights'],self.old['weights'],atol=1e-11,rtol=0)
        bad=deepcopy(rows);bad[1]['reliability_role']='val'
        with self.assertRaises(ValueError):c.fit(bad,prepared,6)
        bad_prepare=deepcopy(prepared);bad_prepare['train_images'].pop()
        with self.assertRaises(ValueError):c.fit(rows,bad_prepare,4)

    def test_synthetic_executor_saves_models_before_VAL_and_keeps_old_inputs(self):
        from crane_project.tools import run_port_reliability_pixel_pauc_v1 as runner
        train=deepcopy(self.rows)
        for i in range(30,2558):
            r=deepcopy(train[0]);r.update(image='real_seq01_%05d'%i,frame_id=i);train.append(r)
        val=self.synthetic_val()
        for i,r in enumerate(val):r.update(image='real_val_%05d'%i)
        before=deepcopy((train,val))
        prepared=c.prepare(train,self.base)
        previous=dict(base_models=self.base,model=self.old)
        protocol=dict(reference_result='unused_reference')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'work_dirs'/c.VERSION/'synthetic'/'result'
            pix=[dict(image=r['image'],pred=r['pred'],domain=r['domain'],sequence=r['sequence'],
                evidence=r['pixel_evidence'],wrong=c.level.metrics.bad(r),accepted=r['fallback_decision'],
                low_temporal=r['temporal']['available'] and r['temporal']['primary']<=c.parent.base.LEVEL_CUTOFF) for r in val]
            def extract(*args):
                self.assertTrue((out/'models.json').is_file())
                self.assertTrue((out/'TRAIN_prepare.json').is_file())
                return pix
            with patch.object(runner,'ROOT',root),patch.object(runner,'load',return_value=(protocol,{},[],previous)),patch.object(runner,'prepare',return_value=(train,prepared)),patch.object(runner.io,'read',return_value=val),patch.object(runner,'extract_val',side_effect=extract),patch.object(runner.subprocess,'check_output',return_value='synthetic-local-test'):
                runner.run(type('Args',(),dict(allow_absent_frontend=False,check_only=False,out=out,data_root=root))())
            report=json.loads((out/'report.json').read_text())
            self.assertEqual(report['fit_calls'],2)
            self.assertTrue(report['model_saved_before_VAL'])
            self.assertFalse(report['TEST_opened'])
            self.assertFalse(report['candidate_adopted'])
            self.assertEqual(len(runner.io.read(out/'scored_TRAIN.jsonl.gz')),2558)
            self.assertEqual(len(runner.io.read(out/'scored_VAL.jsonl.gz')),887)
        self.assertEqual(before,(train,val))

    def test_runner_save_order_check_only_and_no_server(self):
        from crane_project.tools import run_port_reliability_pixel_pauc_v1 as runner
        src=inspect.getsource(runner.run)
        self.assertLess(src.index("io.write(out/'models.json'"),src.index("scored_VAL.jsonl.gz"))
        self.assertLess(src.index("if args.check_only:"),src.index('core.fit('))
        self.assertNotIn('ssh',src)
        self.assertNotIn('hard_set',src)
        self.assertNotIn('extract_val',inspect.getsource(runner.prepare))
        ast.parse(inspect.getsource(runner))
        with patch.object(runner,'load',return_value=({}, {}, ['weight'], {})),patch.object(runner,'prepare',return_value=([],dict(active_rows=1,wrong_images=['x']))),patch.object(c,'fit',side_effect=AssertionError('must not fit')),patch.object(runner,'extract_val',side_effect=AssertionError('must not read VAL')):
            runner.run(type('Args',(),dict(allow_absent_frontend=True,check_only=True))())


if __name__=='__main__':unittest.main()
