"""Frozen evaluation contracts: full denominators, raw coordinates, no fitting."""
from copy import deepcopy
import inspect
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import eval_port_frozen_downstream_v1 as e


def fixture(n=3):
    manifest=dict(sequence_id='fixture',split='unknown_test',run_instance_id='run',
        camera=dict(intrinsics=dict(fx=1000.,fy=1000.)),
        obb_reference_geometry=dict(long_edge_mean_m=2.,short_edge_mean_m=1.))
    pin=dict(count=n,evidence_role='previously_exposed_TEST')
    calibration=dict(parameters=dict(c=1.,beta=2.,b_m=0.,q_signed_min=-.01,q_signed_max=.01))
    truths=[];preds={a:[] for a in e.ARMS}
    for i in range(n):
        truths.append(dict(sequence_id='fixture',run_instance_id='run',frame_index=i,
            image_file='images/frame_%05d.jpg'%i,obb_valid=True,truth_valid=True,
            frame_truth_audit=dict(passed=True),camera_geometry=dict(z_cg_opt_m=20.),
            pivot_relative=dict(theta_total_deg=1.),
            obb_geometry=dict(cx_px=500.,cy_px=500.,w_px=100.,h_px=50.,gamma_deg=0.,
                plumb_w_px=1.,plumb_h_px=1.,plumb_gamma_deg=80.)))
        for a in preds:preds[a].append(dict(frame_id='frame_%05d'%i,box=[500.,500.,100.,50.,0.,.8]))
    return preds,truths,manifest,pin,calibration


class DepthTests(unittest.TestCase):
    def test_raw_coordinates_and_optical_truth_not_plumb(self):
        p,t,m,pin,c=fixture();rows,s=e.evaluate_depth(p,t,m,pin,c)
        for a in (*e.ARMS,'gt_obb'):
            self.assertEqual(s['groups'][a]['depth_metrics']['mae_m'],0.)
            self.assertEqual(s['direct_depth_errors'][a]['abs_error_coverage']['1.0']['count'],3)
        self.assertFalse(s['homography_or_truth_attitude_used'])

    def test_q_outside_fit_support_remains_in_primary_metrics(self):
        p,t,m,pin,c=fixture()
        for r in p['eood']:r['box'][2]=70.
        _,s=e.evaluate_depth(p,t,m,pin,c)
        self.assertEqual(s['groups']['eood']['q_out_of_fit_support_count'],3)
        self.assertEqual(s['groups']['eood']['depth_metrics']['count'],3)
        self.assertGreater(s['groups']['eood']['depth_metrics']['mae_m'],0.)

    def test_missing_outputs_keep_full_frame_denominators(self):
        p,t,m,pin,c=fixture();p['eood'][0]['box']=None
        _,s=e.evaluate_depth(p,t,m,pin,c);g=s['groups']['eood']
        self.assertEqual(g['numeric_depth_count'],2)
        self.assertEqual(g['center_correct_rate_output_frames'],1.)
        self.assertEqual(g['center_correct_coverage_all_frames'],2/3)
        self.assertEqual(s['direct_depth_errors']['eood']['abs_error_coverage']['1.0']['denominator'],3)

    def test_numeric_failure_not_filtered_or_encoded_as_correct(self):
        p,t,m,pin,c=fixture();p['eood'][0]['box'][3]=1e-200
        _,s=e.evaluate_depth(p,t,m,pin,c)
        self.assertEqual(s['groups']['eood']['numeric_failure_count'],1)
        self.assertEqual(s['direct_depth_errors']['eood']['abs_error_coverage']['1.0']['fraction'],2/3)

    def test_b_midpoint_presence_and_score_immutable(self):
        for change in ('score','missing'):
            p,t,m,pin,c=fixture()
            if change=='score':p['symeood_b_midpoint'][0]['box'][5]=.7
            else:p['symeood_b_midpoint'][0]['box']=None
            with self.assertRaises(ValueError):e.evaluate_depth(p,t,m,pin,c)

    def test_truncated_shuffled_truth_and_wrong_run_rejected(self):
        for change in ('truncate','shuffle','run','invalid'):
            p,t,m,pin,c=fixture()
            if change=='truncate':t.pop()
            elif change=='shuffle':t.reverse()
            elif change=='run':t[0]['run_instance_id']='other'
            else:t[0]['truth_valid']=False
            with self.assertRaises(ValueError):e.evaluate_depth(p,t,m,pin,c)

    def test_historical_coverage_gate_enforced(self):
        p,t,m,pin,c=fixture()
        m['collection_design']=dict(coverage_gate=dict(min_valid_frames=1,min_z_cg_opt_span_m=8.,min_theta_bin_counts={}))
        with self.assertRaises(ValueError):e.evaluate_depth(p,t,m,pin,c)

    def test_frame_pairing_and_four_arms_required(self):
        p,t,m,pin,c=fixture();p['eood'][0]['frame_id']='wrong'
        with self.assertRaises(ValueError):e.evaluate_depth(p,t,m,pin,c)
        p,t,m,pin,c=fixture();del p['eood']
        with self.assertRaises(ValueError):e.evaluate_depth(p,t,m,pin,c)

    def test_formula_and_input_objects_unchanged(self):
        args=fixture();before=deepcopy(args);e.evaluate_depth(*args)
        self.assertEqual(args,before)


class ContractTests(unittest.TestCase):
    def test_new_source_closure_and_original_parents(self):
        protocol,calibration,identity=e.checked_sources()
        self.assertFalse(protocol['test_used_for_selection'])
        self.assertTrue(protocol['test_repeatedly_exposed'])
        self.assertEqual(calibration['coordinate_contract']['id'],'raw_opt_v1')
        self.assertEqual(protocol['depth_order'][0],'unknown_test_02_coverage_retry')
        self.assertFalse(protocol['scope']['snow_analysis'])

    def test_wrong_artifact_hash_and_failure_generation_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'file').write_text('a')
            with self.assertRaises(ValueError):e.indexed_bundle(p,{'file':'wrong'})
            (p/'failure.json').write_text('{}')
            with self.assertRaises(ValueError):e.indexed_bundle(p,{'file':e.o.sha(p/'file')})

    def test_inference_has_no_metric_truth_read_and_no_hardcoded_981(self):
        for f in (e.native_b_midpoint,e.native_baseline):
            source=inspect.getsource(f)
            self.assertNotIn('981',source)
            self.assertNotIn('read_rows(',source)
            self.assertNotIn('truth_box(',source)
        self.assertLess(inspect.getsource(e.run_depth).index('native_baseline('),inspect.getsource(e.run_depth).index('read_rows(tp)'))

    def test_no_fit_or_cutoff_update_in_evaluation(self):
        for f in (e.run_reliability,e.score_reliability,e.evaluate_depth,e.run_depth):
            source=inspect.getsource(f)
            self.assertNotIn('create_policy(',source)
            self.assertNotIn('fit_linear_risk(',source)
            self.assertNotIn('optimizer',source)


class ReliabilityTests(unittest.TestCase):
    def test_frozen_flags_keep_boxes_missing_center_and_all_four_states(self):
        simple=e.flags.simple
        model=dict(feature_names=list(simple.FEATURES),converged=True,
            weights=[0.,0.,0.,0.],mean=[0.,0.,0.],scale=[1.,1.,1.])
        base=dict(protocol=simple.VERSION,feature_names=list(simple.FEATURES),
            center_policy='retain_valid_B_output_no_extra_rejection',
            models={c:deepcopy(model) for c in ('size','angle')},
            cutoffs={m:{c:dict(risk_le=.4) for c in ('size','angle')} for m in ('simple','score_only')})
        front=dict(frozen_b=dict(checkpoint_sha256=e.o.B_SHA),
            midpoint_checkpoint=dict(path='head_epoch_03.pth',sha256=e.o.HEAD_SHA,epoch=3,sigma_cells=1.5,updates=2706))
        policy=dict(protocol=e.reliability.VERSION,front_end=front,simple_policy=base)
        rows=[]
        for i,box in enumerate(([100.,100.,80.,30.,0.,.8],[100.,100.,120.,30.,.2,.8],None)):
            rows.append(dict(image='real_seq03_%05d'%i,sequence='real_seq03',frame_id=i,
                domain='real',split='test',gt=[100.,100.,80.,30.,0.],pred=box,
                image_size=[640,480],angle_axis_well_defined=True,image_sha256='0'*64,annotation_sha256='1'*64))
        before=deepcopy([rows,policy]);records,stats=e.score_reliability(rows,policy)
        self.assertEqual([rows,policy],before)
        self.assertEqual(stats['all']['output_frames'],2)
        self.assertEqual(stats['all']['center_hit_rate_on_outputs'],1.)
        self.assertEqual(stats['all']['all_frame_center_correct_coverage'],2/3)
        for r,source in zip(records,rows):
            self.assertEqual(r['methods']['simple']['final_box_original'],source['pred'])
            self.assertEqual(r['methods']['simple']['center_accepted'],source['pred'] is not None)
            self.assertFalse(r['methods']['simple']['size_accepted'])
        size=stats['all']['components']['size']['simple']
        self.assertEqual(size['incorrect_rejected'],1)
        self.assertEqual(size['correct_rejected'],1)
        self.assertEqual(size['missing_outputs'],1)

    def test_paired_cached_output_presence_or_score_change_rejected(self):
        saved=dict(image='real_seq03_00001',sequence='real_seq03',frame_id=1,domain='real',split='test',
            gt=[100.,100.,80.,30.,0.],pred=[100.,100.,80.,30.,0.,.8],image_size=[640,480],
            angle_axis_well_defined=True,image_sha256='0'*64,annotation_sha256='1'*64)
        r=dict(image=saved['image'],sequence=saved['sequence'],domain='real',frame_id=1,
            scale=1.,gt=saved['gt'],b=saved['pred'],midpoint=saved['pred'],original_b_raw_exact_before_after=True)
        with patch.object(e.flags,'validate_rows',side_effect=lambda rows,counts:rows):
            for changed in (None,saved['pred'][:5]+[.7]):
                bad=deepcopy(r);bad['midpoint']=changed
                with self.assertRaises(ValueError):e.bind_reliability([saved],[bad])


if __name__=='__main__':unittest.main()
