"""Fixed cached TEST; CPU, no model loading/fitting/GT-dependent online flags."""
from argparse import Namespace
import ast
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from crane_project.tools import eval_port_midpoint_simple_reliability_v1_test as entry

simple=entry.simple


def policy():
    model=dict(feature_names=list(simple.FEATURES),converged=True,
        weights=[0.,-1.,0.,0.],mean=[0.,0.,0.],scale=[1.,1.,1.])
    return dict(protocol=simple.VERSION,feature_names=list(simple.FEATURES),
        center_policy='retain_valid_B_output_no_extra_rejection',
        models=dict(size=deepcopy(model),angle=deepcopy(model)),
        cutoffs={m:{c:{'risk_le':.5} for c in ('size','angle')} for m in ('simple','score_only')})


def metadata():
    result=[]
    for i in range(4):
        gt=[100.,80.,40.,20.,.1];pred=gt+[.8 if i<2 else .2]
        if i==1: pred[2]*=.6;pred[4]+=math.radians(8)
        if i==2: pred[0]+=15.
        if i==3: pred=None
        result.append(dict(image='real_seq03_%05d'%i,sequence='real_seq03',frame_id=i,
            domain='real',split='test',gt=gt,pred=pred,image_size=[200,160],
            angle_axis_well_defined=True,image_sha256='a'*64,annotation_sha256='b'*64))
    return result


def native_rows(meta):
    result=[]
    for r in meta:
        final=deepcopy(r['pred'])
        if final is not None: final[0]-=.5
        result.append(dict(image=r['image'],sequence=r['sequence'],frame_id=r['frame_id'],
            domain=r['domain'],scale=1.,gt=deepcopy(r['gt']),b=deepcopy(r['pred']),midpoint=final,
            candidate=deepcopy(final),accepted=final is not None if final is not None else None,
            original_b_raw_exact_before_after=True))
    return result


def bundle_fixture():
    front=dict(frozen_b=dict(checkpoint_sha256='b24',config_sha256='config'),
        midpoint_checkpoint=dict(epoch=23,path='head_epoch_23.pth',sha256='h23',head_digest={'frozen':23}),
        selection_sha256='selection',formal_sources_sha256='formaltrain')
    wrapper=dict(protocol=entry.migration.new.VERSION,front_end=front,simple_policy=policy(),test_read=False)
    selection=dict(selected_checkpoint=front['midpoint_checkpoint'],selection_sha256='selection',
        completion_sha256='completion',cache_manifest_sha256='cache')
    data=dict(frames=4,sequence_counts={'real_seq03':4})
    protocol=dict(front_end=front,reviewed_formal_selection=selection,
        formal_evaluation_protocol='formal_eval',formal_evaluation_sources_sha256='formaleval',
        test_counts={'real_seq03':4},test_data_identity=data)
    report=dict(protocol='formal_eval',split='test',status='FROZEN_FORMAL_MIDPOINT_TEST_COMPLETE_REVIEW_REQUIRED',
        selection=selection,identity=dict(training_identity=dict(sources_sha256='formaltrain',frozen_b=front['frozen_b']),sources_sha256='formaleval'),
        data_identity=data,frames=4,state_before=dict(head={'frozen':23}),state_after=dict(head={'frozen':23}),
        detector_updates=0,head_updates=0,selection_on_test=False,automatic_promotion=False,
        test_access=True,test_repeatedly_exposed=True,feature_extractions=4,native_head_calls=12)
    hashes={'completion.json':'completion','test_rows.jsonl':'rows','artifacts.json':'artifacts'}
    return wrapper,protocol,report,{'files':{'completion.json':'completion','test_rows.jsonl':'rows'}},hashes


class InputTests(unittest.TestCase):
    def test_checked_sources_preserve_fixed_frontend_scope_and_Python38(self):
        protocol,_,_=entry.checked_sources()
        self.assertEqual(protocol['front_end']['midpoint_checkpoint']['epoch'],23)
        self.assertFalse(protocol['test_used_for_selection'])
        self.assertFalse(protocol['reference_or_ROI_or_structure_scores_used'])
        self.assertEqual(protocol['methods'],['raw','score_only','simple'])
        for path in (Path(entry.__file__),Path(__file__)):
            tree=ast.parse(path.read_text(),feature_version=(3,8))
            imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
            imports += [alias.name for n in ast.walk(tree) if isinstance(n,ast.Import) for alias in n.names]
            self.assertFalse(any(n and n.split('.')[0] in ('torch','mmcv','mmrotate','cv2') for n in imports))
            if path==Path(entry.__file__):
                attrs={n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)}
                self.assertFalse(attrs & {'create_policy','coverage_cutoff','fit_linear_risk','build_online','load_checkpoint'})

    def test_fresh_process_import_does_not_import_Torch_or_GPU_framework(self):
        code='from crane_project.tools import eval_port_midpoint_simple_reliability_v1_test; import sys; assert not ({"torch","mmcv","mmrotate","cv2"} & set(sys.modules))'
        value=subprocess.run([sys.executable,'-c',code],cwd=str(entry.ROOT),capture_output=True,text=True)
        self.assertEqual(value.returncode,0,value.stderr)

    def test_bind_uses_formal_boxes_and_ignores_old_cache_predictions_qualities(self):
        meta=metadata();native=native_rows(meta);before=deepcopy([meta,native])
        rows=entry.bind_rows(meta,native,{'real_seq03':4})
        self.assertEqual(rows[0]['pred'],native[0]['midpoint'])
        for r in meta:
            if r['pred'] is not None:r['pred'][0]+=100.
            r['qualities']={'roi':[0,0,0],'structure':[1,1,1]}
        altered=entry.bind_rows(meta,native,{'real_seq03':4})
        self.assertEqual(rows,altered)
        self.assertEqual(native,before[1])

    def test_bind_small_GT_precision_difference_retains_formal_GT(self):
        meta=metadata();native=native_rows(meta)
        native[0]['gt'][4]+=5e-8
        rows=entry.bind_rows(meta,native,{'real_seq03':4})
        self.assertEqual(rows[0]['gt'],native[0]['gt'])
        self.assertNotEqual(rows[0]['gt'],meta[0]['gt'])
        native[0]['gt'][0]+=1.
        with self.assertRaises(ValueError):entry.bind_rows(meta,native,{'real_seq03':4})

    def test_bind_rejects_wrong_order_missing_duplicate_split_and_audit(self):
        for kind in ('order','missing','duplicate','split','audit','scale','identity'):
            meta=metadata();native=native_rows(meta)
            if kind=='order':native.reverse()
            if kind=='missing':native.pop()
            if kind=='duplicate':native[1]=deepcopy(native[0])
            if kind=='split':meta[0]['split']='val'
            if kind=='audit':native[0]['original_b_raw_exact_before_after']=False
            if kind=='scale':native[0]['scale']=.5
            if kind=='identity':native[0]['domain']='sim'
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                entry.bind_rows(meta,native,{'real_seq03':4})

    def test_bind_rejects_score_output_and_fallback_changes(self):
        for kind in ('score','missing','candidate','fallback','accept'):
            meta=metadata();native=native_rows(meta)
            if kind=='score':native[0]['midpoint'][5]=.7
            if kind=='missing':native[3]['midpoint']=deepcopy(native[0]['midpoint'])
            if kind=='candidate':native[0]['candidate'][0]+=2
            if kind=='fallback':native[0]['accepted']=False
            if kind=='accept':native[0]['accepted']=None
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                entry.bind_rows(meta,native,{'real_seq03':4})

    def test_completed_formal_bundle_not_short_fit_wrong_epoch_or_stale_state(self):
        wrapper,protocol,report,artifacts,hashes=bundle_fixture()
        entry.validate_formal_bundle(report,artifacts,hashes,wrapper,protocol)
        for kind in ('shortfit','epoch','state','updates','split','selection','counts','sources','digest','calls','artifacts'):
            changed=deepcopy(report);a=deepcopy(artifacts)
            if kind=='shortfit':changed['protocol']='shortfit'
            if kind=='epoch':changed['selection']['selected_checkpoint']['epoch']=22
            if kind=='state':changed['state_after']['head']={'changed':True}
            if kind=='updates':changed['head_updates']=1
            if kind=='split':changed['split']='val'
            if kind=='selection':changed['selection_on_test']=True
            if kind=='counts':changed['frames']=3
            if kind=='sources':changed['identity']['training_identity']['sources_sha256']='other'
            if kind=='digest':changed['state_before']=changed['state_after']=dict(head={'other':True})
            if kind=='calls':changed['feature_extractions']=3
            if kind=='artifacts':a['files']['test_rows.jsonl']='other'
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                entry.validate_formal_bundle(changed,a,hashes,wrapper,protocol)

    def test_midpoint_policy_requires_exact_reviewed_completion_and_bytes(self):
        wrapper,_,_,_,_=bundle_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            wrapper['contract']=dict(front_end=wrapper['front_end'],test_read=False)
            (root/'policy.json').write_text(json.dumps(wrapper))
            artifacts={'policy.json':entry.base.sha(root/'policy.json')}
            completed=dict(status='MIDPOINT_SIMPLE_POLICY_FIT_COMPLETE',contract=wrapper['contract'],
                artifacts=artifacts,test_read=False,detector_updates=0,midpoint_updates=0,reference_updates=0)
            (root/'completion.json').write_text(json.dumps(completed))
            protocol=dict(front_end=wrapper['front_end'],reviewed_midpoint_policy=dict(artifacts=artifacts,
                completion_sha256=entry.base.sha(root/'completion.json'),contract_sha256=simple.fingerprint(wrapper['contract'])))
            loaded,_=entry.load_midpoint_policy(root/'policy.json',protocol)
            self.assertEqual(wrapper,loaded)
            (root/'policy.json').write_text('{}')
            with self.assertRaises(ValueError):entry.load_midpoint_policy(root/'policy.json',protocol)
            with self.assertRaises(ValueError):entry.load_midpoint_policy(root/'other.json',protocol)


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.rows=entry.bind_rows(metadata(),native_rows(metadata()),{'real_seq03':4})
        self.wrapper=bundle_fixture()[0]

    def test_three_flags_and_center_denominators_are_final_box_metrics(self):
        records,s,b=entry.score(self.rows,self.wrapper,policy())
        self.assertEqual(s['all']['output_frames'],3)
        self.assertEqual(s['all']['center_hits'],3)
        self.assertEqual(s['all']['center_hit_rate_on_outputs'],1.)
        self.assertEqual(s['all']['all_frame_center_correct_coverage'],.75)
        self.assertEqual(b['all']['center_hits'],2)
        self.assertFalse(any(records[-1]['methods']['simple'][c+'_accepted'] for c in simple.COMPONENTS))
        self.assertEqual(s['all']['components']['size']['simple']['bad_accepted'],1)
        self.assertEqual(s['all']['components']['size']['simple']['good_false_rejected'],1)
        self.assertAlmostEqual(s['all']['components']['size']['simple']['state_accuracy_on_outputs'],1/3)
        for r,source in zip(records,self.rows):
            self.assertEqual(r['methods']['simple']['final_box_original'],source['pred'])
            self.assertEqual(r['b_original'],source['b_original'])

    def test_GT_does_not_enter_flags_or_trigger_fit_threshold_selection(self):
        before=deepcopy([self.rows,self.wrapper])
        with patch.object(simple,'create_policy',side_effect=AssertionError('No fitting')), \
                patch.object(simple,'coverage_cutoff',side_effect=AssertionError('No calibration')):
            a,_,_=entry.score(self.rows,self.wrapper,policy())
            changed=deepcopy(self.rows)
            for r in changed:r['gt']=[1.,2.,60.,10.,1.4]
            b,_,_=entry.score(changed,self.wrapper,policy())
        self.assertEqual([r['methods'] for r in a],[r['methods'] for r in b])
        self.assertEqual([self.rows,self.wrapper],before)

    def test_rejection_keeps_center_and_components_independent(self):
        self.wrapper['simple_policy']['cutoffs']['simple']['size']['risk_le']=0.
        self.wrapper['simple_policy']['cutoffs']['simple']['angle']['risk_le']=1.
        records,s,_=entry.score(self.rows,self.wrapper,policy())
        for r in records[:3]:
            self.assertTrue(r['methods']['simple']['center_accepted'])
            self.assertFalse(r['methods']['simple']['size_accepted'])
            self.assertTrue(r['methods']['simple']['angle_accepted'])
        self.assertEqual(s['all']['complete_obb']['simple']['accepted_frames'],0)

    def test_ambiguous_angle_is_assessed_separately_from_online_acceptance(self):
        self.rows[0]['gt'][2]=self.rows[0]['gt'][3]
        self.rows[0]['angle_axis_well_defined']=False
        records,s,_=entry.score(self.rows,self.wrapper,policy())
        self.assertTrue(records[0]['methods']['simple']['angle_accepted'])
        self.assertEqual(s['all']['components']['angle']['simple']['unassessed_accepted'],1)
        self.assertEqual(s['all']['components']['angle']['simple']['eligible_frames'],3)

    def test_all_missing_has_null_conditional_correctness(self):
        for r in self.rows:r['pred']=r['b_original']=None
        _,s,_=entry.score(self.rows,self.wrapper,policy())
        self.assertIsNone(s['all']['center_hit_rate_on_outputs'])
        self.assertEqual(s['all']['all_frame_center_correct_coverage'],0.)
        self.assertIsNone(s['all']['components']['size']['simple']['state_accuracy_on_outputs'])

    def test_equivalent_width_height_swap_preserves_size_direction_decisions(self):
        altered=deepcopy(self.rows)
        for r in altered:
            if r['pred'] is not None:
                p=r['pred'];p[2],p[3]=p[3],p[2];p[4]+=math.pi/2
        a,sa,_=entry.score(self.rows,self.wrapper,policy())
        b,sb,_=entry.score(altered,self.wrapper,policy())
        for c in ('size','angle'):
            first=sa['all']['components'][c]['simple']
            second=sb['all']['components'][c]['simple']
            self.assertEqual(set(first),set(second))
            for key,value in first.items():
                if isinstance(value,float):
                    self.assertAlmostEqual(value,second[key],places=10,msg=key)
                else:
                    self.assertEqual(value,second[key],key)
        self.assertEqual([r['methods']['simple']['size_accepted'] for r in a],
                         [r['methods']['simple']['size_accepted'] for r in b])
        self.assertEqual([r['methods']['simple']['angle_accepted'] for r in a],
                         [r['methods']['simple']['angle_accepted'] for r in b])

    def test_same_count_score_boundary_ties_report_bounds_without_GT_selection(self):
        records,s,_=entry.score(self.rows,self.wrapper,policy())
        z=s['all']['components']['size']['matched_score_diagnostic']
        self.assertEqual(z['tie_bounds']['accepted'],2)
        self.assertEqual(z['tie_bounds']['incorrect_accepted'],z['bad_accepted'])
        self.assertIn('bad_min_over_tie',z['tie_bounds'])


class EntryReplayTests(unittest.TestCase):
    def test_complete_saved_rows_csv_report_checksums_and_no_overwrite(self):
        meta=metadata()
        for i,r in enumerate(deepcopy(meta)):
            r.update(image='sim_seq09_%05d'%i,sequence='sim_seq09',domain='sim')
            meta.append(r)
        rows=entry.bind_rows(meta,native_rows(meta),{'real_seq03':4,'sim_seq09':4})
        wrapper,_,_,_,_=bundle_fixture()
        protocol=dict(front_end=wrapper['front_end'],evidence_role='fixed_exposed_TEST')
        prepared=(protocol,{'source':'fixed'},None,wrapper,policy(),rows,{'data':'fixed'})
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);policy_path=root/'policy.json';old_path=root/'old_policy.json'
            policy_path.write_text(json.dumps(wrapper));old_path.write_text(json.dumps(policy()))
            out=root/'result';out.mkdir()
            args=Namespace(out_dir=out,policy=policy_path,old_policy=old_path,mode='run')
            entry.base.write_new(out/'input_check.json',{'passed':True})
            status=entry.write_result(args,prepared)
            with patch.object(entry,'prepare',return_value=prepared):entry.finish(args,prepared,status)
            report=json.loads((out/'test_compare.json').read_text())
            decisions=[json.loads(l) for l in (out/'test_decisions.jsonl').read_text().splitlines()]
            self.assertEqual(len(decisions),8)
            self.assertEqual(report['strata'],entry.score(rows,wrapper,policy())[1])
            self.assertEqual(len((out/'component_metrics.csv').read_text().splitlines()),61)
            self.assertIn('同接受数score为离线诊断',(out/'test_summary.md').read_text())
            complete=json.loads((out/'completion.json').read_text())
            for name,sha in complete['artifacts'].items():self.assertEqual(entry.base.sha(out/name),sha)
            with self.assertRaises(FileExistsError):entry.write_result(args,prepared)
            with patch.object(entry,'prepare',return_value=prepared):
                with self.assertRaises(FileExistsError):entry.finish(args,prepared,status)


if __name__=='__main__':unittest.main()
