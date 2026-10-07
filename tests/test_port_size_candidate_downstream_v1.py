"""Actual candidate identity, fixed transfer policy, no fitting or TEST selection."""
from copy import deepcopy
import inspect
import json
import unittest
from unittest.mock import patch
from tools import eval_port_size_candidate_downstream_v1 as e
from tests.test_port_frozen_downstream_v1 import fixture

class Tests(unittest.TestCase):
    def data(self):
        p,t,m,pin,c=fixture()
        rows=[dict(frame_id=r['frame_id'],midpoint=deepcopy(r['box']),size_candidate=deepcopy(r['box'])) for r in p['eood']]
        return rows,t,m,pin,c

    def test_exact_boxes_depth_oracle(self):
        rows,t,m,pin,c=self.data();r,s=e.depth_evaluation(rows,t,m,pin,c)
        for k in (*e.METHODS,'gt_obb'):self.assertEqual(s['groups'][k]['depth_metrics']['mae_m'],0.)
        self.assertEqual(s['paired_absolute_error']['tied'],3)
        self.assertFalse(s['automatic_promotion'])

    def test_actual_candidate_sizes_change_depth(self):
        rows,t,m,pin,c=self.data()
        for r in rows:r['size_candidate'][2]*=.95;r['size_candidate'][3]*=.95
        _,s=e.depth_evaluation(rows,t,m,pin,c)
        self.assertAlmostEqual(s['groups']['size_candidate']['depth_metrics']['bias_m'],20/.95-20)
        self.assertEqual(s['paired_absolute_error']['worsened'],3)

    def test_missing_and_numeric_full_denominators(self):
        rows,t,m,pin,c=self.data();rows[0].update(midpoint=None,size_candidate=None)
        rows[1]['size_candidate'][3]=1e-200
        _,s=e.depth_evaluation(rows,t,m,pin,c)
        self.assertEqual(s['groups']['size_candidate']['numeric_failure_count'],1)
        self.assertEqual(s['groups']['size_candidate']['center_correct_rate_output_frames'],1.)
        self.assertEqual(s['direct_depth_errors']['size_candidate']['abs_error_coverage']['1.0']['denominator'],3)
        self.assertEqual(s['direct_depth_errors']['size_candidate']['abs_error_coverage']['1.0']['count'],1)

    def test_q_support_does_not_filter_candidate(self):
        rows,t,m,pin,c=self.data()
        for r in rows:r['size_candidate'][2]=75.
        _,s=e.depth_evaluation(rows,t,m,pin,c)
        self.assertEqual(s['groups']['size_candidate']['q_out_of_fit_support_count'],3)
        self.assertEqual(s['groups']['size_candidate']['numeric_depth_count'],3)

    def test_frozen_identity_including_long_axis(self):
        m=[1.,2.,10.,5.,0.,.8]
        e.preserve(m,[1.,2.,11.,6.,0.,.8]);e.preserve(None,None)
        for i in (0,1,4,5):
            n=m[:];n[i]+=.1
            with self.assertRaises(ValueError):e.preserve(m,n)
        with self.assertRaises(ValueError):e.preserve(m,[1.,2.,4.,5.,0.,.8])
        with self.assertRaises(ValueError):e.preserve(m,None)

    def test_finite200_header_only(self):
        p=json.loads(e.PROTOCOL.read_text())
        good=dict(protocol='port_geometry_size_boundary_continuous_v1',stage='finite',arm='boundary_continuous',updates=200,experimental_only=True,automatic_promotion=False,head_digest=p['candidate_digest'],head_state={})
        e.checkpoint_header(good,p)
        for k,v in [('stage','smoke'),('updates',2),('arm','boundary_bucket'),('automatic_promotion',True),('head_digest',{})]:
            changed=dict(good);changed[k]=v
            with self.assertRaises(ValueError):e.checkpoint_header(changed,p)

    def test_sources_do_not_change_old_protocol(self):
        p,old,c,identity=e.checked_sources()
        self.assertFalse(p['original_TRAIN_gate_passed']);self.assertFalse(p['automatic_promotion'])
        self.assertEqual(p['candidate_sha256'],'de94b8d8a4d17d7770ea332be7589912fba762bde6e773fac44560bbb686858b')
        self.assertGreater(len(identity['sources']),134)

    def test_inference_inputs_gt_free_and_no_optimizer(self):
        self.assertEqual(e.INPUT_NAMES,('roi','support','boxes_original','boxes_model','scale_xy','midpoint_original'))
        source=inspect.getsource(e.candidate)
        self.assertNotIn('.backward(',source);self.assertNotIn('optim.',source)
        self.assertLess(source.index("o.write_new(out/'inference_proof.json'"),source.index('reliability_evaluation('))
        self.assertLess(source.index("o.write_new(out/'inference_proof.json'"),source.index('depth_evaluation('))

    def test_fixed_policy_transfer_counts_geometry_without_refitting(self):
        simple=e.base.flags.simple
        model=dict(feature_names=list(simple.FEATURES),converged=True,weights=[0.,0.,0.,0.],mean=[0.,0.,0.],scale=[1.,1.,1.])
        policy=dict(simple_policy=dict(protocol=simple.VERSION,feature_names=list(simple.FEATURES),center_policy='retain_valid_B_output_no_extra_rejection',models={k:deepcopy(model) for k in ('size','angle')},cutoffs={m:{k:dict(risk_le=.6) for k in ('size','angle')} for m in ('simple','score_only')}))
        metadata=[];predictions=[]
        for i,box in enumerate(([100.,100.,100.,30.,0.,.8],None)):
            metadata.append(dict(image='real_seq03_%05d'%i,sequence='real_seq03',frame_id=i,domain='real',split='test',gt=[100.,100.,80.,30.,0.],pred=box,image_size=[640,480],angle_axis_well_defined=True,image_sha256='0'*64,annotation_sha256='1'*64))
            predictions.append(dict(image=metadata[-1]['image'],image_sha256='0'*64,midpoint=box,size_candidate=[100.,100.,80.,30.,0.,.8] if box else None))
        before=deepcopy([metadata,predictions,policy])
        with patch.object(e.base.flags,'validate_rows',side_effect=lambda rows,counts:rows):
            decisions,report=e.reliability_evaluation(predictions,metadata,policy)
        self.assertEqual([metadata,predictions,policy],before)
        self.assertFalse(report['policy_refit'])
        self.assertTrue(report['geometry_changes_are_not_discriminator_improvement'])
        old=report['groups']['midpoint']['all']['components']['size']['simple']
        new=report['groups']['size_candidate']['all']['components']['size']['simple']
        self.assertEqual(old['incorrect_accepted'],1);self.assertEqual(new['correct_accepted'],1)
        self.assertEqual(new['missing_outputs'],1)
        self.assertEqual(report['groups']['size_candidate']['all']['all_frame_center_correct_coverage'],.5)
        for k in e.METHODS:self.assertEqual(decisions[k][0]['methods']['simple']['raw_b_output'],predictions[0][k])

    def test_replay_rejects_different_score_or_count(self):
        r=dict(frame_id=1,b=[1.,2.,10.,5.,0.,.8],midpoint=[1.,2.,10.,5.,0.,.8])
        self.assertEqual(e.replay([r],[r])['max_abs_scalar_delta'],0.)
        changed=deepcopy(r);changed['midpoint'][5]=.81
        with self.assertRaises(ValueError):e.replay([r],[changed])
        with self.assertRaises(ValueError):e.replay([r],[])

if __name__=='__main__':unittest.main()
