"""CPU checks of fixed provenance, denominators, reader use and full entry replay."""
from argparse import Namespace
import ast
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from crane_project.tools import eval_port_midpoint_size_template_v1 as entry
from crane_project.utils import port_midpoint_size_template_v1 as result

simple = entry.simple
SETTINGS = json.loads(entry.migration.PROTOCOL.read_text())


def box():
    return [128., 128., 80., 40., .37]


def meta():
    return dict(img_shape=[256,256,3], ori_shape=[256,256,3], pad_shape=[256,256,3],
                scale_factor=[1.,1.,1.,1.], flip=False)


def source(name, split='val', bad=False, missing=False, domain='real'):
    pred = box()+[.8]
    if bad:
        pred[2] *= 1.3
    if missing:
        pred = None
    return dict(image=name, split=split, domain=domain, sequence=domain+'_seq',
        frame_id=int(name.split('_')[-1]), image_size=[256,256], gt=box(),
        pred=pred, b_original=deepcopy(pred), angle_axis_well_defined=True,
        train_angle_eligible=True)


def runtime():
    train = [source('fit_%d'%i, 'train', bad=bool(i%2)) for i in range(16)]
    for i,r in enumerate(train):
        r['pred'][4] += .1 if i%3 else 0.
        r['pred'][5] = .3+i*.03
    val = [source('val_%d'%i, bad=bool(i%2)) for i in range(12)]
    policy = simple.create_policy(train, val, json.loads(entry.base.PROTOCOL.read_text()))
    wrapper = dict(protocol=entry.migration.new.VERSION, front_end={'frozen':True}, simple_policy=policy)
    return wrapper, entry.migration.new.MidpointReliability(wrapper, wrapper['front_end'])


def numeric_records():
    policy, api = runtime()
    target,_ = entry.migration.reference.size.target_map(box(), meta())
    rows = []
    for role,split in (('reference_holdout_train','train'), ('val','val')):
        for i in range(5):
            src = source(role+'_%d'%i, split, bad=i in (1,2), missing=i==4)
            evidence = result.online_readings(target,src['pred'],src['image_size'],meta(),api,SETTINGS)
            rows.append(result.record(src,evidence,role))
    return policy, rows


class FixedReaderStatisticsTests(unittest.TestCase):
    def test_actual_reader_sizes_and_GT_free_scoring_preserve_detection(self):
        _, api = runtime()
        target,_ = entry.migration.reference.size.target_map(box(),meta())
        pred = box()+[.8]; original=deepcopy(pred)
        evidence = result.online_readings(target,pred,[256,256],meta(),api,SETTINGS)
        self.assertEqual(pred,original)
        self.assertFalse(evidence['new_reader_flags_created'])
        self.assertLess(evidence['risks']['template'],1e-5)
        good = result.record(source('x_0'),evidence,'val')
        changed = source('x_0'); changed['gt']=[64,64,20,10,-.3]
        bad = result.record(changed,evidence,'val')
        self.assertEqual(good['risks'],bad['risks'])
        self.assertEqual(good['frozen_decisions'],bad['frozen_decisions'])
        self.assertNotEqual(good['size_bad'],bad['size_bad'])

    def test_missing_outputs_have_no_risk_or_flags_and_separate_denominators(self):
        _,rows = numeric_records()
        stats=result.summarize(rows)['val']['all']
        self.assertEqual((stats['frames'],stats['outputs'],stats['missing_outputs']),(5,4,1))
        self.assertEqual(stats['center_hit_rate_on_outputs'],1.)
        self.assertEqual(stats['all_frame_center_correct_coverage'],.8)
        missing=rows[-1]
        self.assertIsNone(missing['size_bad'])
        self.assertTrue(all(v is None for v in missing['risks'].values()))
        self.assertFalse(any(missing['frozen_decisions']['simple'][k+'_accepted'] for k in ('center','size','angle')))

    def test_unavailable_reference_is_not_trusted_or_removed_from_raw_stats(self):
        _,rows=numeric_records()
        for r in rows:
            if r['pred'] is not None and r['frame_id'] in (0,1):
                r['references']['template']=dict(defined=False,reason='weak_evidence')
                r['risks']['template']=None
                r['reference_errors']['template']=None
        stats=result.summarize(rows)['val']['all']
        self.assertEqual(stats['raw_size_bad'],2)
        self.assertEqual(stats['readers']['template']['unavailable_good'],1)
        self.assertEqual(stats['readers']['template']['unavailable_bad'],1)
        self.assertEqual(stats['readers']['moments']['defined'],4)
        self.assertEqual(stats['common_defined']['support_outputs'],2)
        point=stats['common_defined']['same_count_points'][1]
        self.assertTrue(point['availability_limited'])
        self.assertTrue(all(m['accepted_assessed_outputs']==2 for m in point['methods'].values()))
        full=point['all_output_same_count_availability_as_reject']['template']
        self.assertEqual(full['assessed_outputs'],4)
        self.assertEqual(full['good_false_rejected'],1)
        self.assertEqual(full['bad_correctly_rejected'],1)
        self.assertEqual(full['assessed_acceptance_coverage_on_all_frames'],.4)

    def test_empty_common_support_and_single_class_have_null_AUC(self):
        _,rows=numeric_records()
        for r in rows:
            r['references']['template']=dict(defined=False,reason='not_a_gaussian_peak')
            r['risks']['template']=None; r['reference_errors']['template']=None
        stats=result.summarize(rows)['val']['all']
        self.assertEqual(stats['common_defined']['support_outputs'],0)
        self.assertIsNone(stats['common_defined']['rank']['template']['error_auroc'])
        self.assertEqual(stats['common_defined']['same_count_points'][0]['actual_count'],0)
        _,rows=numeric_records()
        for r in rows:
            if r['size_bad'] is not None: r['size_bad']=False
        self.assertIsNone(result.summarize(rows)['val']['all']['full_output_baselines']['rank']['score']['error_auroc'])

    def test_ties_have_label_free_selection_and_explicit_bad_bounds(self):
        _,rows=numeric_records()
        output=[r for r in rows if r['assessment_role']=='val' and r['pred'] is not None]
        stats=result.subset_report(output,list(range(4)),('score',),5,[.4])
        point=stats['same_count_points'][0]
        tie=point['methods']['score']['tie_bounds']
        self.assertEqual((tie['boundary_tie_size'],tie['boundary_tie_selected']),(4,2))
        self.assertEqual((tie['bad_min_over_tie'],tie['bad_max_over_tie']),(0,2))
        changed=deepcopy(output)
        for r in changed: r['size_bad']=not r['size_bad']
        other=result.subset_report(changed,list(range(4)),('score',),5,[.4])
        self.assertEqual(point['methods']['score']['accepted_images_sha256'],
            other['same_count_points'][0]['methods']['score']['accepted_images_sha256'])

    def test_role_guards_and_input_immutability(self):
        _,rows=numeric_records(); before=deepcopy(rows)
        result.summarize(rows); self.assertEqual(rows,before)
        with self.assertRaises(ValueError): result.summarize(rows+[rows[0]])
        with self.assertRaises(ValueError): result.record(dict(rows[0],split='test'),rows[0],'val')
        with self.assertRaises(ValueError): result.record(rows[0],rows[0],'test')

    def test_online_parity_preserves_missing_and_exact_score(self):
        b=box()+[.8]; changed=deepcopy(b); changed[0]+=1e-5
        self.assertGreater(result.paired_prediction(b,changed,'x'),0)
        self.assertEqual(result.paired_prediction(None,None,'x'),0)
        for bad in (None,box()+[.80000001],box()+[.8+.1]):
            with self.assertRaises(ValueError): result.paired_prediction(b,bad,'x')
        changed[0]+=1.
        with self.assertRaises(ValueError): result.paired_prediction(b,changed,'x')


class FixedInputsTests(unittest.TestCase):
    def test_manifest_protocol_and_Python38(self):
        protocol,_=entry.checked_sources()
        self.assertEqual(protocol['frames'],{'reference_holdout_train':432,'val':887})
        self.assertFalse(protocol['deployment_cutoff_created'])
        for path in (Path(entry.__file__),Path(result.__file__),Path(__file__)):
            tree=ast.parse(path.read_text(),feature_version=(3,8))
            if path==Path(entry.__file__):
                names={n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)}
                self.assertFalse(names & {'backward','step','create_policy','fit_linear_risk'})

    def test_fixed_plan_counts_and_no_reference_fit_or_guard(self):
        train=[source('train_%d'%i,'train') for i in range(2558)]
        val=[source('val_%d'%i) for i in range(887)]
        partition=dict(holdout=train[:432],fit=train[432:816],guard=train[816:848])
        counts={'reference_holdout_train':432,'val':887}
        plan=entry.evaluation_rows(train,val,partition,counts)
        self.assertEqual(len(plan),1319)
        self.assertEqual([r['image'] for _,r in plan[:432]],[r['image'] for r in train[:432]])
        self.assertEqual(train[0]['pred'],box()+[.8])
        for bad in (dict(partition,fit=train[:1]),dict(partition,holdout=train[:431])):
            with self.assertRaises(ValueError): entry.evaluation_rows(train,val,bad,counts)
        val[-1]['split']='test'
        with self.assertRaises(ValueError): entry.evaluation_rows(train,val,partition,counts)

    def stage(self,path):
        old=dict(front_end={'midpoint':'epoch23'},sources={'reviewed':'old'},test_read=False,
                 formal_proof={'selected':23})
        current=dict(old,sources={'reviewed':'fixed_probe_interface'})
        (path/'predictions.jsonl').write_text('unchanged\n')
        completion=dict(status='MIDPOINT_TRAIN_VAL_COLLECTION_COMPLETE',contract=old,test_read=False,
                        artifacts={'predictions.jsonl':entry.base.sha(path/'predictions.jsonl')})
        (path/'completion.json').write_text(json.dumps(completion))
        pins=dict(contract_sha256=simple.fingerprint(old),collect=dict(
            completion_sha256=entry.base.sha(path/'completion.json'),artifacts=completion['artifacts']))
        return old,current,pins

    def test_only_exact_reviewed_historical_sources_can_be_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory); old,current,pins=self.stage(path)
            self.assertEqual(entry.reviewed_stage(path,'collect',current,pins),old)
            for field,value in (('front_end',{'midpoint':'epoch24'}),('formal_proof',{'selected':24})):
                with self.assertRaises(ValueError): entry.reviewed_stage(path,'collect',dict(current,**{field:value}),pins)
            (path/'predictions.jsonl').write_text('changed\n')
            with self.assertRaises(ValueError): entry.reviewed_stage(path,'collect',current,pins)

    def test_completion_rewrite_or_failure_is_not_a_provenance_fix(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory); old,current,pins=self.stage(path)
            (path/'failure.json').write_text('{}')
            with self.assertRaises(ValueError): entry.reviewed_stage(path,'collect',current,pins)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory); old,current,pins=self.stage(path)
            (path/'completion.json').write_text('{}')
            with self.assertRaises(ValueError): entry.reviewed_stage(path,'collect',current,pins)


class FullEntryTests(unittest.TestCase):
    def test_stream_saved_rows_summary_flags_completion_and_state_guards(self):
        policy,api=runtime()
        plan=[('reference_holdout_train',source('train_%d'%i,'train',domain=d))
              for i,d in enumerate(('real','sim'))]
        plan += [('val',source('val_%d'%i,missing=i==2,domain=d))
                 for i,d in enumerate(('real','sim','real'))]
        indexed={r['image']:r for _,r in plan}
        target,_=entry.migration.reference.size.target_map(box(),meta())
        class Tensor:
            def sigmoid(self): return self
            def __getitem__(self,k): return self
            def cpu(self): return self
            def numpy(self): return target
        torch=SimpleNamespace(no_grad=nullcontext,cuda=SimpleNamespace(
            max_memory_allocated=lambda gpu:1024,max_memory_reserved=lambda gpu:2048))
        protocol=dict(frames={'reference_holdout_train':2,'val':3},coverage_fractions=[.9,.95])
        old=[None]*10; old[7]={}
        previous=(SETTINGS,old,None,None,None,{'front_end':policy['front_end']})
        prepared=(protocol,previous,policy,plan,{'locked':True})
        def online(*args): return None,torch,'detector','head','pipeline',lambda features:Tensor()
        def view(row,*args): return [row['image']],meta(),[]
        def final(formal,torch,detector,head,features,meta):
            row=indexed[features[0]]
            return dict(b=deepcopy(row['b_original']),midpoint=deepcopy(row['pred']))
        with tempfile.TemporaryDirectory() as directory:
            args=Namespace(out_dir=Path(directory),mode='run',gpu=0)
            with patch.object(entry.migration,'online_modules',side_effect=online), \
                    patch.object(entry.migration.reference,'view',side_effect=view), \
                    patch.object(entry.migration,'midpoint_from_features',side_effect=final), \
                    patch.object(entry.base,'state_digest',return_value='same'):
                status=entry.run(args,prepared)
            with patch.object(entry,'prepare',return_value=prepared): entry.finish(args,prepared,status)
            report=json.loads((args.out_dir/'assessment.json').read_text())
            completed=json.loads((args.out_dir/'completion.json').read_text())
            rows=[json.loads(line) for line in (args.out_dir/'assessment_rows.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),5)
            self.assertEqual(report['summary'],result.summarize(rows))
            self.assertEqual(report['state_before'],report['state_after'])
            self.assertFalse(report['new_reader_flags_created'])
            self.assertTrue(report['online_formal_parity_passed'])
            self.assertEqual(report['summary']['val']['all']['missing_outputs'],1)
            for name,sha in completed['artifacts'].items(): self.assertEqual(entry.base.sha(args.out_dir/name),sha)
            with patch.object(entry,'prepare',return_value=prepared):
                with self.assertRaises(FileExistsError): entry.finish(args,prepared,status)


if __name__=='__main__':
    unittest.main()
