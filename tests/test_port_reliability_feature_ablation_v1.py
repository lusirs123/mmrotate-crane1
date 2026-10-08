"""Meaningful solver, calibration, selection, immutability and TEST-gate tests."""
from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from crane_project.utils import port_reliability_feature_ablation_v1 as ab
from crane_project.tools import run_port_reliability_feature_ablation_v1 as entry


def original():
    m=dict(feature_names=list(ab.simple.FEATURES),converged=True,weights=[0.,-1.,.2,.1],
           mean=[0.,0.,0.],scale=[1.,1.,1.])
    sp=dict(protocol=ab.simple.VERSION,feature_names=list(ab.simple.FEATURES),
        center_policy='retain_valid_B_output_no_extra_rejection',models={'size':deepcopy(m),'angle':deepcopy(m)},
        cutoffs={method:{c:dict(risk_le=.5) for c in ('size','angle')} for method in ('simple','score_only')})
    front=dict(frozen_b=dict(checkpoint_sha256=ab.binding.B_SHA),midpoint_checkpoint=dict(
        path='head_epoch_03.pth',sha256=ab.binding.HEAD_SHA,epoch=3,sigma_cells=1.5,updates=2706))
    return dict(protocol=ab.binding.VERSION,front_end=front,simple_policy=sp,test_read=False)


def data():
    rng=np.random.RandomState(8);x=rng.randn(60,3)
    return x,(x[:,0]+.7*x[:,1]-.2*x[:,2]>.1).astype(float)


def rows():
    values=[]
    for sequence,domain in [('real_a','real'),('sim_b','sim')]:
        for i in range(5):
            gt=[100.,80.,40.,20.,.1];pred=gt+[.8]
            if i==1:pred[3]=15.
            if i==4:pred=None
            values.append(dict(image=sequence+'_%05d'%i,sequence=sequence,frame_id=i,domain=domain,
                split='val',gt=gt,pred=pred,image_size=[200,160],angle_axis_well_defined=True))
    return values


def policy(arm='drop_relative_size'):
    x,y=data()
    return dict(protocol=ab.VERSION,arm=arm,size_model=ab.fit(x,y,ab.ARMS[arm]),
        size_cutoff=dict(single_global_cutoff=True,risk_le=.5),frozen_original_policy=original())


class AblationTests(unittest.TestCase):
    def test_full_width_generalization_matches_original_fixed_solver(self):
        x,y=data();a=ab.fit(x,y,(0,1,2));b=ab.simple.fit_linear_risk(x,y,ab.FIT)
        for key in ('weights','mean','scale','initial_objective','final_objective'):
            np.testing.assert_allclose(a[key],b[key],rtol=0,atol=1e-12)
        self.assertEqual(a['iterations'],b['iterations'])

    def test_removed_feature_cannot_influence_fit_or_inference(self):
        x,y=data()
        for name,indices in ab.ARMS.items():
            omitted=next(i for i in range(3) if i not in indices);z=x.copy();z[:,omitted]+=50.
            a,b=ab.fit(x,y,indices),ab.fit(z,y,indices)
            self.assertEqual(a,b);np.testing.assert_array_equal(ab.risk(a,x),ab.risk(a,z))

    def test_ablation_refits_retained_coefficients_not_zeroing_old_weights(self):
        x,y=data();full=ab.fit(x,y,(0,1,2));reduced=ab.fit(x,y,(0,2))
        projected=np.array(full['weights'])[[0,1,3]]
        self.assertGreater(float(np.max(np.abs(projected-reduced['weights']))),1e-4)
        self.assertEqual(len(reduced['weights']),3)
        self.assertEqual(reduced['bad_rows']+reduced['good_rows'],60)

    def test_single_class_and_invalid_features_rejected(self):
        x,y=data()
        with self.assertRaises(ValueError):ab.fit(x,np.zeros(60),(0,2))
        with self.assertRaises(ValueError):ab.fit(x,y,(0,))
        with self.assertRaises(ValueError):ab.fit(x[:,:2],y,(0,2))
        x[0,1]=float('nan')
        with self.assertRaises(ValueError):ab.fit(x,y,(0,2))

    def test_constant_feature_scale_stays_finite(self):
        x,y=data();x[:,2]=1.;m=ab.fit(x,y,(0,2))
        self.assertEqual(m['scale'][1],1.)
        self.assertTrue(np.isfinite(ab.risk(m,x)).all())

    def test_single_global_cutoff_preserves_each_video_and_whole_ties(self):
        r=rows();risk={v['image']:.2 if v['domain']=='real' else .8 for v in r}
        c=ab.calibrate(r,risk)
        self.assertEqual(c['risk_le'],.8)
        s=ab.describe(r,risk,c['risk_le'],{'full_simple':risk,'score_only':risk})
        self.assertEqual(s['all']['accepted_assessed_outputs'],8)
        self.assertEqual(s['all']['missing_outputs'],2)
        for v in s.values():self.assertEqual(v['good_retention_rate'],1.)

    def test_cutoff_cannot_be_fitted_on_TRAIN_or_TEST(self):
        for split in ('train','test'):
            r=rows();r[0]['split']=split
            with self.assertRaises(ValueError):ab.calibrate(r,{v['image']:.2 for v in r})

    def test_counts_and_continuity_include_missing_and_reset_gaps(self):
        r=rows()[:5];r[2]['frame_id']=8;r[3]['frame_id']=9;r[4]['frame_id']=10
        self.assertEqual(ab.unavailable(r,set()),3)
        self.assertEqual(ab.unavailable(r,{v['image'] for v in r if v['pred'] is not None}),1)

    def test_gate_does_not_accept_equal_rank_or_local_deterioration(self):
        r=rows();risk={v['image']: .2 for v in r};s=ab.describe(r,risk,.2,{'full_simple':risk,'score_only':risk})
        self.assertFalse(ab.gate(s)['passed'])
        checks={v['check'] for v in ab.gate(s)['failures']}
        self.assertIn('strict_overall_matched_FA_gain',checks)
        s['sequence:real_a']['longest_flag_unavailable']+=1
        self.assertIn('matched_continuity_nonincrease',{v['check'] for v in ab.gate(s)['failures']})

    def test_gate_can_accept_actual_error_ordering_gain_with_protection(self):
        r=rows()
        candidate={v['image']:(.9 if v['pred'] is not None and ab.size_bad(v) else .2) for v in r}
        control={v['image']:(.05 if v['pred'] is not None and ab.size_bad(v) else .5+.01*v['frame_id']) for v in r}
        stats=ab.describe(r,candidate,.2,{'full_simple':control,'score_only':control})
        self.assertTrue(ab.gate(stats)['passed'])
        self.assertEqual(stats['all']['incorrect_accepted'],0)

    def test_selection_is_VAL_only_and_exact_tie_retains_original(self):
        def v(fa,cr,ok=True):return dict(gate=dict(passed=ok),stats={'all':dict(incorrect_accepted=fa,correct_accepted=cr)})
        self.assertIsNone(ab.choose({'a':v(2,10,False),'b':v(1,11,False)}))
        self.assertIsNone(ab.choose({'a':v(2,10),'b':v(2,10)}))
        self.assertEqual(ab.choose({'a':v(2,10),'b':v(1,10)}),'b')

    def test_online_copies_final_box_center_angle_score_and_missing(self):
        p=policy();api=ab.binding.Sigma15Reliability(p['frozen_original_policy'],p['frozen_original_policy']['front_end'])
        before=ab.simple.fingerprint(p)
        for row in rows():
            a=api.decide(row['pred'],row['image_size']);b=ab.decide(p,row['pred'],row['image_size'])
            for key in ('center_accepted','angle_accepted','final_box_original'):self.assertEqual(a[key],b[key])
            self.assertEqual(a['risks']['angle'],b['risks']['angle'])
            if row['pred'] is None:self.assertFalse(b['size_accepted'])
        self.assertEqual(before,ab.simple.fingerprint(p))

    def test_full_components_protect_center_angle_and_qualified_joint_denominator(self):
        r=rows();r[0]['gt'][2]=22.;r[0]['angle_axis_well_defined']=False;p=policy()
        stats=ab.component_report(r,original(),{'candidate':p})['all']
        self.assertEqual(stats['output_frames'],8)
        for c in ('center','angle'):
            self.assertEqual(stats['methods']['original_simple']['components'][c],stats['methods']['candidate']['components'][c])
        self.assertEqual(stats['methods']['candidate']['components']['angle']['eligible_frames'],9)

    def test_TEST_paths_never_opened_without_frozen_passing_selection(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'completion.json').write_text(json.dumps(dict(protocol=ab.VERSION,
                status='FEATURE_ABLATION_VAL_COMPLETE_REVIEW_REQUIRED',test_read=False,selected_arm=None)))
            args=Namespace(fit_dir=root,out=root/'out',geometry_test_dir=root/'forbidden',metadata_dir=root/'forbidden')
            with patch.object(entry,'checked_sources',side_effect=AssertionError('Must not reach TEST source stage')):
                with self.assertRaisesRegex(ValueError,'TEST must not be opened'):entry.run_test(args)
            self.assertFalse(args.out.exists())

    def test_frozen_fit_tampering_rejected_before_TEST(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);f=root/'fit_report.json';f.write_text('{}')
            receipt=dict(protocol=ab.VERSION,status='FEATURE_ABLATION_VAL_COMPLETE_REVIEW_REQUIRED',test_read=False,
                         selected_arm='drop_relative_size',artifacts={'fit_report.json':'0'*64})
            (root/'completion.json').write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError,'artifact changed'):entry.checked_frozen_selection(root)

    def test_original_policy_and_scoring_inputs_not_modified(self):
        r=rows();p=original();x,y=data();models={k:ab.fit(x,y,v) for k,v in ab.ARMS.items()}
        before=ab.simple.fingerprint([r,p]);a=ab.scores(r,p,models)
        ab.describe(r,a['drop_aspect'],.5,{m:a[m] for m in ab.CONTROLS})
        self.assertEqual(before,ab.simple.fingerprint([r,p]))

    def test_sealed_protocol_sources_match_without_model_or_TEST_data(self):
        p,s=entry.checked_sources()
        self.assertEqual(p['feature_arms'],{k:list(v) for k,v in ab.ARMS.items()})
        self.assertEqual(p['fitting'],ab.FIT);self.assertEqual(p['target_good_retention'],.95)
        self.assertFalse(p['test_used_for_selection']);self.assertTrue(s['sources'])


if __name__=='__main__':unittest.main()
