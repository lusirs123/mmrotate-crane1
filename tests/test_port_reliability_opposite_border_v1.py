from copy import deepcopy
import inspect
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from crane_project.utils import port_reliability_opposite_border_v1 as m
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils import port_reliability_state_continuity_v1 as states
from crane_project.tools import run_port_reliability_opposite_border_v1 as runner


def row(i=0,bad=False,missing=False):
    p=None if missing else [100.,100.,100.,25. if bad else 20.,0.,.8]
    d=dict(center_accepted=not missing,size_accepted=not missing,angle_accepted=not missing,
           final_box_original=p,risks=dict(size=None if missing else .2,angle=None if missing else .3))
    return dict(image='real_seq01_'+str(i).zfill(3),sequence='real_seq01',domain='real',split='train',
        frame_id=i,pred=p,gt=[100.,100.,100.,20.,0.],image_size=[1000,500],
        reliability_role='train',sample_role='fit',original_simple_decision=d)


def evidence(main=None,base=None,xy=None,fields=None,support=None):
    main=[100.,100.,80.,40.,0.,.8] if main is None else main
    base=[100.,100.,100.,60.,0.,.8] if base is None else base
    xy=np.array([1.,1.]) if xy is None else np.array(xy)
    bm=np.array(base[:5]);bm[:4]*=xy[[0,1,0,1]]
    fields=np.ones((2,9,9)) if fields is None else fields
    support=np.ones((9,9)) if support is None else support
    return m.sample_evidence(main,base,bm,xy,fields,support)


def neutral(arm=m.ARM):
    return m.fit(np.zeros((6,len(m.SCHEMAS[arm]))),[0,0,0,1,1,1],arm)


class SamplingTests(unittest.TestCase):
    def test_channel_summary_known_values(self):
        a=np.empty((256,9,9));a[:128]=3.;a[128:]=-1.
        np.testing.assert_equal(m.summarize_channels(a)[0],np.ones((9,9)))
        np.testing.assert_equal(m.summarize_channels(a)[1],np.full((9,9),math.sqrt(5)))

    def test_channel_summary_rejects_wrong_shape_nonfinite(self):
        for a in (np.zeros((255,9,9)),np.full((256,9,9),np.nan)):
            with self.assertRaises(ValueError):m.summarize_channels(a)

    def test_bilinear_linear_plane_and_endpoints(self):
        yy,xx=np.mgrid[0:9,0:9];a=np.stack((2*xx+3*yy,xx-yy))
        p=np.array([[-.5,-.5],[.5,.5],[.17,-.23]])
        x,y=((p+.5)*8).T
        np.testing.assert_allclose(m.bilinear(a,p),np.stack((2*x+3*y,x-y),1),atol=1e-13)

    def test_no_extrapolation_and_explicit_support(self):
        p=np.array([[.5001,0],[0,0]])
        np.testing.assert_equal(m.bilinear(np.ones((1,9,9)),p),[[0],[1]])
        a=m.pooled(np.ones((2,9,9)),np.ones((9,9)),p[None])
        np.testing.assert_equal(a,[[1,1,.5]])
        np.testing.assert_equal(m.pooled(np.ones((2,9,9)),np.zeros((9,9)),p[None]),[[0,0,0]])

    def test_actual_anisotropic_raw_restoration(self):
        b=[100.,200.,100.,30.,.4,.8];xy=[.6,.61]
        bm=np.array(b[:5]);bm[:4]*=np.array(xy)[[0,1,0,1]]
        _,got,_,_=m.transform(b,bm,xy);np.testing.assert_equal(got,xy)
        wrong=bm.copy();wrong[3]=b[3]*xy[0]
        with self.assertRaises(ValueError):m.transform(b,wrong,xy)

    def test_canonical_width_height_association_of_cache(self):
        b=[10.,20.,20.,80.,.3,.7];bm=np.array(b[:5])
        cb,_,_,rot=m.transform(b,bm,[1,1])
        self.assertEqual(cb[2],80);self.assertEqual(cb[3],20)
        np.testing.assert_allclose(rot.T@rot,np.eye(2),atol=1e-15)
        wrong=bm.copy();wrong[2:4]=wrong[2:4][::-1]
        with self.assertRaises(ValueError):m.transform(b,wrong,[1,1])

    def test_sample_world_to_cache_rotation_and_sx_sy(self):
        b=[100.,100.,100.,60.,.3,.8];p=[103.,104.,80.,40.,-.1,.8];xy=np.array([.7,.71])
        e=evidence(p,b,xy);bm=np.array(b[:5]);bm[:4]*=xy[[0,1,0,1]]
        cb=simple.canonical(bm);c,s=np.cos(cb[4]),np.sin(cb[4]);rot=np.array([[c,-s],[s,c]])
        cm=simple.canonical(p[:5]);normal=np.array([np.cos(cm[4]),np.sin(cm[4])])
        tangent=np.array([-np.sin(cm[4]),np.cos(cm[4])])
        world=cm[:2]+cm[2]/2*normal-.35*cm[3]*tangent
        expected=(world*xy-cb[:2])@rot/np.maximum(cb[2:4]*1.5,16)
        np.testing.assert_allclose(e['border_local'][0,0,1,0],expected,atol=1e-15)

    def test_bands_have_distinct_source_spacing(self):
        e=evidence();self.assertTrue(e['axis_resolution_qualified'].all())
        self.assertTrue((e['cache_band_separation_Linf']>=1-1e-12).all())
        self.assertTrue((e['native_P3_band_separation']>=1-1e-12).all())
        self.assertGreater(np.max(abs(e['border_local'][0,0,0]-e['border_local'][0,0,1])),0)

    def test_thin_axis_zeros_evidence_without_dropping_frame(self):
        e=evidence(main=[100,100,80,5,0,.8]);self.assertFalse(e['axis_resolution_qualified'][1])
        np.testing.assert_equal(e['opposite_border'].reshape(2,2,3,3)[1],0)
        self.assertEqual(len(m.descriptors([100,100,80,5,0,.8],[1000,500],e)[m.ARM]),39)

    def test_regular_roi_keeps_3x3_spatial_regions(self):
        yy,xx=np.mgrid[0:9,0:9];f=np.stack((xx+yy,xx+yy+1))
        p=[100,100,100,60,0,.8];e=evidence(main=p,fields=f)
        q=e['ordinary_roi'].reshape(3,3,3)
        np.testing.assert_allclose(q[:,:,0],[[2,5,8],[5,8,11],[8,11,14]],atol=1e-14)
        np.testing.assert_equal(q[:,:,2],1)
        self.assertGreater(q[-1,-1,0],q[0,0,0])

    def test_native_isotropic_unit_invariance(self):
        p=[100,100,80,40,.2,.8];b=[100,100,100,60,.1,.8]
        a=evidence(p,b,[.8,.81]);p2=p.copy();b2=b.copy()
        p2[:4]=[v*3 for v in p2[:4]];b2[:4]=[v*3 for v in b2[:4]]
        z=evidence(p2,b2,[.8/3,.81/3])
        for k in ('ordinary_roi','opposite_border','border_local'):
            np.testing.assert_allclose(a[k],z[k],atol=1e-13)
        np.testing.assert_allclose(z['offsets_original'],a['offsets_original']*3,atol=1e-13)

    def test_main_swapped_edges_and_pi_identity(self):
        p=[100,100,80,40,.2,.8];a=evidence(p);q=p.copy();q[2:4]=q[2:4][::-1];q[4]-=math.pi/2
        b=evidence(q);q[4]+=math.pi;c=evidence(q)
        for k in ('ordinary_roi','opposite_border'):
            np.testing.assert_allclose(a[k],b[k],atol=1e-13);np.testing.assert_allclose(a[k],c[k],atol=1e-13)

    def test_invalid_fields_and_support_rejected(self):
        for f,s in ((np.full((2,9,9),np.nan),np.ones((9,9))),
                    (np.full((2,9,9),-1),np.ones((9,9))),
                    (np.ones((2,9,9)),np.ones((9,9))*2)):
            with self.assertRaises(ValueError):evidence(fields=f,support=s)

    def test_missing_main_has_no_evidence_descriptor(self):
        self.assertIsNone(m.sample_evidence(None,None,None,None,None,None))
        self.assertEqual(m.descriptors(None,[1000,500],None),dict.fromkeys(m.SCHEMAS))
        with self.assertRaises(ValueError):m.descriptors(None,[1000,500],{})

    def test_descriptor_prefix_and_exact_fixed_schema(self):
        p=[100,100,80,40,0,.8];d=m.descriptors(p,[1000,500],evidence())
        for arm,n in [('refit_simple',3),('ordinary_roi',30),(m.ARM,39)]:
            self.assertEqual(len(d[arm]),n);np.testing.assert_equal(d[arm][:3],simple.descriptor(p,[1000,500]))
        self.assertEqual(len(set(m.BORDER_NAMES)),36)

    def test_sampling_api_has_no_GT_role_aux_or_domain(self):
        self.assertEqual(tuple(inspect.signature(m.sample_evidence).parameters),
            ('main','base_original','base_model','scale_xy','fields','support'))
        self.assertNotIn('gt',inspect.signature(m.decide).parameters)


class FittingAndDecisionTests(unittest.TestCase):
    def test_zero_initialization_constant_features_deterministic(self):
        a=neutral();self.assertEqual(a,neutral());self.assertEqual(a['weights'],[0.]*40)
        self.assertEqual(a['scale'],[1.]*39);self.assertEqual(a['initialization'],'all_zero')

    def test_invalid_fit_schema_or_single_class_rejected(self):
        for x,y in ((np.zeros((6,38)),[0,1]*3),(np.zeros((6,39)),[0]*6),
                    (np.full((6,39),np.inf),[0,1]*3)):
            with self.assertRaises(ValueError):m.fit(x,y,m.ARM)

    def test_actual_objective_derivatives_and_update(self):
        rng=np.random.RandomState(2);x=rng.normal(size=(24,39));y=[0,1]*12
        check=runner.engineering_check(x,y);self.assertTrue(check['passed'])
        self.assertLess(check['one_Newton_update_loss'],check['zero_init_loss'])
        design=np.c_[np.ones(24),x];w=rng.normal(size=40)*.1
        a=simple.logistic_objective(w,design,np.array(y),np.ones(24)/24,.1)
        b=simple.logistic_objective(w,design,np.array(y),np.ones(24)/24,0)
        self.assertEqual(a[1][0],b[1][0]);self.assertEqual(a[2][0,0],b[2][0,0])

    def test_invalid_model_and_cutoff_rejected(self):
        for key in ('settings','scale','weights','feature_names'):
            a=neutral()
            if key=='settings':a[key]['l2']=.2
            elif key=='scale':a[key][0]=0
            elif key=='weights':a[key][0]=np.nan
            else:a[key]=a[key][:-1]
            with self.assertRaises(ValueError):m.risk(a,np.zeros(39),m.ARM)
        for c in (-1,2,np.nan):
            with self.assertRaises(ValueError):m.decide(neutral(),c,row()['original_simple_decision'],np.zeros(39))

    def test_only_size_changes_with_all_other_output_identity(self):
        r=row();old=deepcopy(r);expected=deepcopy(r['original_simple_decision'])
        expected['size_accepted']=False;expected['risks']['size']=.5
        self.assertEqual(m.decide(neutral(),0.,r['original_simple_decision'],np.zeros(39)),expected)
        self.assertEqual(r,old)

    def test_no_missing_frame_quality_or_fake_center(self):
        r=row(missing=True);d=r['original_simple_decision']
        self.assertEqual(m.decide({},.5,d,None),d)
        with self.assertRaises(ValueError):m.decide({},.5,d,np.zeros(39))
        r=row();r['original_simple_decision']['center_accepted']=False
        with self.assertRaises(ValueError):m.decide(neutral(),.5,r['original_simple_decision'],np.zeros(39))


class OfflineContractTests(unittest.TestCase):
    def test_GT_pair_size_exact_angle_rounding_only_original_preserved(self):
        gt=[10.,20.,100.,20.,.2];a=gt.copy();a[4]+=1.8e-7;before=a.copy()
        self.assertLess(runner.checked_gt_pair(a,gt),2e-6);self.assertEqual(a,before)
        for j,d in ((0,1e-6),(2,1e-6),(4,1e-3)):
            a=gt.copy();a[j]+=d
            with self.assertRaises(ValueError):runner.checked_gt_pair(a,gt)

    def test_fit_only_cutoff_and_whole_ties(self):
        rows=[row(i) for i in range(20)]+[row(20,bad=True)]
        risks={r['image']:(.4 if i<19 else .8) for i,r in enumerate(rows)}
        p=m.fit_cutoff(rows,risks);self.assertEqual(p['risk_le'],.4);self.assertFalse(p['probe_calibration'])
        self.assertEqual(sum(v<=p['risk_le'] for v in risks.values()),19)
        for role in ('probe','legacy','val'):
            rs=deepcopy(rows);rs[0]['sample_role']=role
            with self.assertRaises(ValueError):m.fit_cutoff(rs,risks)
        rs=deepcopy(rows);rs[0]['reliability_role']='val'
        with self.assertRaises(ValueError):m.fit_cutoff(rs,risks)

    def test_per_video_retention_controls_global_cutoff(self):
        a=row();b=row(1);b.update(image='real_seq02_001',sequence='real_seq02')
        self.assertEqual(m.fit_cutoff([a,b],{a['image']:.2,b['image']:.9})['risk_le'],.9)

    def test_missing_outputs_excluded_and_states_separated(self):
        rows=[row(0,bad=True),row(1),row(2,missing=True)]
        scores={arm:{r['image']:None if r['pred'] is None else i*.1 for i,r in enumerate(rows)} for arm in (m.ARM,)+m.CONTROLS}
        d=m.describe(rows,scores,.05)['all']['actual']
        self.assertEqual(d['states'],dict(FA=1,FR=1,ED=0,CR=0,MISSING=1))
        self.assertEqual(d['runs']['incorrect_acceptance']['longest'],1)
        self.assertEqual(d['runs']['correct_rejection']['longest'],1)

    def test_gate_strict_FA_and_longest_FR_not_total_unavailable(self):
        def summary(fa,fr,longest):
            return dict(states=dict(FA=fa,FR=fr,CR=100-fr,ED=10-fa),
                runs=dict(correct_rejection=dict(longest=longest),unavailable=dict(longest=100)))
        v=dict(actual=summary(2,0,0),control_tie_bounds={c:dict(bad_min_over_tie=3) for c in m.CONTROLS},
            matched_CR_controls={c:summary(3,0,0) for c in m.CONTROLS},
            same_count_controls={c:summary(3,1,1) for c in m.CONTROLS})
        self.assertTrue(m.gate({'all':v})['passed'])
        v['actual']['runs']['correct_rejection']['longest']=2
        self.assertEqual({f['check'] for f in m.gate({'all':v})['failures']},{'longest_FR'})
        v['actual']=summary(3,6,0)
        self.assertIn('correct_retention',{f['check'] for f in m.gate({'all':v})['failures']})

    def test_no_test_stage_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory(dir=str(runner.ROOT/'work_dirs')) as temp:
            with self.assertRaises(ValueError):runner.checked_out(Path(temp)/'train_check','train')
        path=runner.ROOT/'work_dirs'/m.VERSION/'unit_existing'/'train_check'
        with patch.object(Path,'exists',return_value=True):
            with self.assertRaises(FileExistsError):runner.checked_out(path,'train')
        with self.assertRaises(ValueError):states.checked_order([dict(row(),split='test')])

    def test_VAL_cannot_open_after_failed_probe(self):
        path=runner.ROOT/'work_dirs'/m.VERSION/'unit_guard'/'train_check'
        receipt=dict(protocol=m.VERSION,artifacts={})
        report=dict(protocol=m.VERSION,probe_gate=dict(passed=False))
        def read(p,*a,**k):return json.dumps(receipt if p.name=='completion.json' else report)
        with patch.object(Path,'read_text',read):
            with self.assertRaises(ValueError):runner.frozen_train(path)

    def test_sealed_source_and_existing_frozen_control_replay(self):
        protocol,proof=runner.checked_sources();self.assertEqual(protocol['protocol'],m.VERSION)
        self.assertIn('tests/test_port_reliability_opposite_border_v1.py',proof['sources'])
        # Full real data replay happens before server fitting, without a control refit.


if __name__=='__main__':unittest.main()
