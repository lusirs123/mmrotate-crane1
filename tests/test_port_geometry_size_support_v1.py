import ast
from copy import deepcopy
import io
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from crane_project.utils import port_geometry_size_support_v1 as d
from crane_project.tools import diagnose_port_geometry_size_support_v1 as entry


def row(sequence='real_seq01', index=0, role='train', pred=None, missing=False):
    gt = [100., 200., 100., 40., .2]
    b = None if missing else (pred or gt+[.8])
    return dict(image=sequence+'_%05d'%index, sequence=sequence,
        domain=sequence.split('_')[0], split='train_sim' if role=='train' and sequence.startswith('sim') else role,
        reliability_role=role, frame_id=index, gt=gt, pred=deepcopy(b),
        b_original=deepcopy(b), midpoint_candidate=deepcopy(b),
        image_size=[640,480], midpoint_accepted=not missing,
        train_angle_eligible=True, angle_axis_well_defined=True)


def full_fixture():
    values = [row(seq,i,role) for role,counts in d.COUNTS.items()
              for seq,n in counts.items() for i in range(n)]
    labels = dict(records=[dict(image=r['image'], sequence=r['sequence'], split=r['split'],
        annotation_family='existing', annotation_sha256='sealed') for r in values])
    samples = dict(samples=[dict(image=r['image'],sequence=r['sequence'],split=r['split'],
        role='probe' if i%4==0 else 'fit',annotation_sha256='sealed') for i,r in enumerate(values[:64])])
    return values, labels, samples


class GeometryTests(unittest.TestCase):
    def test_canonical_swap_pi_and_isotropic_relative_errors(self):
        gt = [10.,20.,100.,40.,.2]
        b = [11.,21.,90.,38.,.3,.8]
        expected = d.size_errors(gt,b)
        swapped = gt[:2]+[gt[3],gt[2],gt[4]+math.pi/2]
        transformed = b[:2]+[b[3],b[2],b[4]+math.pi/2,b[5]]
        self.assertEqual(d.size_errors(swapped,transformed),expected)
        resized = [v/2 for v in b[:4]]+b[4:]
        resized_gt = [v/2 for v in gt[:4]]+gt[4:]
        result = d.size_errors(resized_gt,resized)
        for key in ('long_relative','short_relative','ratio_log','scale_log'):
            self.assertAlmostEqual(result[key],expected[key],places=14)

    def test_exact_10pct_correct_and_just_over_wrong(self):
        gt = [0.,0.,100.,40.,0.]
        self.assertFalse(d.size_errors(gt,[0.,0.,110.,36.,0.,.8])['size_wrong'])
        self.assertTrue(d.size_errors(gt,[0.,0.,110.00001,36.,0.,.8])['size_wrong'])

    def test_common_scale_vs_aspect_are_separate(self):
        gt = [0.,0.,100.,40.,0.]
        same = d.size_errors(gt,[0.,0.,80.,32.,0.,.8])
        self.assertTrue(same['both_small']); self.assertAlmostEqual(same['ratio_log'],0.)
        mixed = d.size_errors(gt,[0.,0.,110.,32.,0.,.8])
        self.assertTrue(mixed['short_only_wrong']); self.assertTrue(mixed['ratio_relative_over_10pct'])
        self.assertAlmostEqual(mixed['ratio_log'],math.log(1.1/.8))

    def test_missing_center_denominators_and_no_mutation(self):
        values,labels,samples = full_fixture()
        values[0] = row(missing=True)
        values[1] = row(index=1,pred=[120.,200.,100.,40.,.2,.8])
        before = deepcopy(values)
        result = entry.numeric_diagnosis(values,labels,samples,{})
        stats = result['groups']['train/sequence:real_seq01']['midpoint']
        self.assertEqual((stats['frames'],stats['outputs'],stats['center_hits_on_outputs']),(339,338,337))
        self.assertEqual(stats['center_hit_rate_on_outputs'],337/338)
        self.assertEqual(stats['all_frame_center_correct_coverage'],337/339)
        self.assertEqual(stats['output_coverage'],338/339)
        self.assertEqual(values,before)

    def test_center_exact_15px_not_correct(self):
        values,labels,samples = full_fixture()
        values[0] = row(pred=[115.,200.,100.,40.,.2,.8])
        report = entry.numeric_diagnosis(values,labels,samples,{})
        self.assertEqual(report['groups']['train/all']['midpoint']['center_hits_on_outputs'],2557)

    def test_runs_split_source_segment_gap_missing_and_role(self):
        def record(i,stamp=None,segment='1',missing=False,role='train'):
            e = None if missing else d.size_errors([0.,0.,100.,40.,0.],[0.,0.,80.,32.,0.,.8])
            return dict(image='real_seq12_%05d'%i,sequence='real_seq12',frame_id=i,role=role,
                sampling_role='remaining_train',b=e,midpoint=e,
                sampling=None if stamp is None else dict(source_video='v',segment=segment,
                    source_frame_index=i,source_time_seconds=stamp))
        rows = [record(0,0.),record(1,.333),record(2,.666,segment='2'),
                record(3,4.,segment='2'),record(4,4.333,segment='2',missing=True),
                record(5,4.666,segment='2')]
        runs = [r for r in d.error_runs(rows) if r['arm']=='midpoint' and r['flag']=='size_wrong']
        self.assertEqual([r['frames'] for r in runs],[2,1,1,1])
        self.assertTrue(all(not r['independent_scene_confirmed'] for r in runs))
        self.assertFalse(d.contiguous(record(0),record(2)))

    def test_quantiles_and_empty_outputs_are_not_zero_errors(self):
        self.assertEqual(d.quantile([1.,3.],.95),2.9)
        self.assertIsNone(d.distribution([])['mean'])
        v = d.summary([dict(midpoint=None)],'midpoint')
        self.assertIsNone(v['center_hit_rate_on_outputs'])
        self.assertEqual(v['all_frame_both_size_correct_coverage'],0.)

    def test_polygon_and_axis_short_are_not_depth_truth(self):
        geometry = d.polygon_geometry('0 0 100 0 100 40 0 40 grab 0\n')
        self.assertEqual(geometry['aspect'],2.5)
        self.assertTrue(d.annotation_comparison([50.,20.,100.,40.,0.],geometry)['within_rounding_tolerance'])
        self.assertFalse(d.annotation_comparison([50.,20.,100.,39.,0.],geometry)['within_rounding_tolerance'])
        with self.assertRaises(ValueError):d.polygon_geometry('0 0 100 0 100 40 0 40 other 0')


class EvidenceTests(unittest.TestCase):
    def test_full_roles_and_wrong_sigma_data_metadata_refused(self):
        values,_,_ = full_fixture()
        d.checked_rows(values)
        for key,new_value in [('reliability_role','test'),('split','test'),('sequence','real_seq03'),
                              ('domain','sim'),('frame_id',True)]:
            changed = deepcopy(values); changed[0][key] = new_value
            with self.assertRaises(ValueError):d.checked_rows(changed)
        with self.assertRaises(ValueError):d.checked_rows(values+[values[0]])

    def test_scores_presence_fallback_and_nonfinite_refused(self):
        values,_,_ = full_fixture()
        for change in ('score','presence','fallback','nan','bool_edge'):
            changed = deepcopy(values)
            if change=='score':changed[0]['pred'][5]=.7
            elif change=='presence':changed[0]['pred']=None
            elif change=='fallback':changed[0]['midpoint_accepted']=False; changed[0]['pred'][2]=99.
            elif change=='nan':changed[0]['gt'][2]=float('nan')
            else:changed[0]['gt'][2]=True
            with self.assertRaises(ValueError):d.checked_rows(changed)

    def test_native_missing_output_uses_null_midpoint_decision(self):
        values,_,_=full_fixture()
        values[0]=row(missing=True)
        values[0]['midpoint_accepted']=None
        d.checked_rows(values)
        values[0]['midpoint_accepted']=True
        with self.assertRaises(ValueError):d.checked_rows(values)
        values[0]=row();values[0]['midpoint_accepted']=None
        with self.assertRaises(ValueError):d.checked_rows(values)

    def test_legacy64_roles_and_train_error_membership_preserved(self):
        values,labels,samples = full_fixture()
        values[0] = row(pred=[100.,200.,80.,32.,.2,.8])
        values[100] = row(index=100,pred=[100.,200.,80.,32.,.2,.8])
        report = entry.numeric_diagnosis(values,labels,samples,{})
        support = report['train_error_support']['size_wrong']
        self.assertEqual(support['frames'],2)
        self.assertEqual(support['legacy_sampling_roles'],{'legacy64_probe':1,'remaining_train':1})
        self.assertEqual(report['groups']['train/sampling_role:legacy64_fit']['midpoint']['frames'],48)
        self.assertEqual(report['groups']['train/sampling_role:remaining_train']['midpoint']['frames'],2494)
        samples['samples'][0]['image']='real_seq07_00000'
        with self.assertRaises(ValueError):entry.numeric_diagnosis(values,labels,samples,{})

    def test_wrong_bytes_and_archive_paths_duplicates_links_refused(self):
        for member_name,linked,duplicate in [('collect/completion.json',False,False),
                ('../escape',False,False),('/absolute',False,False),
                ('collect/completion.json',True,False),('same',False,True)]:
            with tempfile.TemporaryDirectory() as temp:
                archive = Path(temp)/'input.tar.gz'
                with tarfile.open(archive,'w:gz') as tar:
                    for _ in range(2 if duplicate else 1):
                        item = tarfile.TarInfo(member_name)
                        if linked:item.type=tarfile.SYMTYPE;item.linkname='target';tar.addfile(item)
                        else:item.size=2;tar.addfile(item,io.BytesIO(b'{}'))
                with self.assertRaises((ValueError,KeyError)):
                    entry.load_collection(archive=archive)

    def test_existing_output_refused(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'existing';out.mkdir();(out/'keep').write_text('evidence')
            with patch.object(entry,'checked_sources',return_value=({},{})):
                with self.assertRaises(FileExistsError):entry.run(SimpleNamespace(stage='check',out_dir=out,
                    collection_dir=None,collection_archive=None))
            self.assertEqual((out/'keep').read_text(),'evidence')

    def test_completed_collection_directory_replays_and_wrong_frontend_refused(self):
        values,_,_=full_fixture()
        head_digest={'parameters':'frozen','buffers':'frozen'}
        front=dict(frozen_b=dict(checkpoint_sha256=entry.B_SHA),selection_sha256=entry.SELECTION_SHA,
            midpoint_checkpoint=dict(epoch=3,sigma_cells=1.5,updates=2706,sha256=entry.HEAD_SHA,
                                     head_digest=head_digest))
        val=[dict(r) for r in values if r['reliability_role']=='val']
        for r in val:r.pop('reliability_role')
        contract=dict(front_end=front,test_read=False,final_VAL_fingerprint=entry.fingerprint(val))
        report=dict(head_state_before=head_digest,head_state_after=head_digest,detector_loaded=False,
                    test_read=False,feature_extractions=0,head_updates=0,detector_updates=0)
        def contents():
            blobs={'collect/collect_report.json':json.dumps(report).encode(),
                   'collect/input_check.json':json.dumps(contract).encode(),
                   'collect/final_rows.jsonl':b'\n'.join(json.dumps(r).encode() for r in values),
                   'collect/independent_B_cache_comparisons.json':b'{}'}
            complete=dict(protocol='port_midpoint_sigma15_reliability_v1',
                status='SIGMA15_RELIABILITY_COLLECTION_COMPLETE',contract=contract,
                artifacts={Path(n).name:entry.digest(b) for n,b in blobs.items()},
                test_read=False,detector_updates=0,midpoint_updates=0,reference_updates=0)
            blobs['collect/completion.json']=json.dumps(complete).encode()
            return blobs
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);(folder/'collect').mkdir()
            def publish():
                blobs=contents()
                for n,b in blobs.items():(folder/n).write_bytes(b)
                return {n:entry.digest(b) for n,b in blobs.items()}
            with patch.object(entry,'INPUT_FILES',publish()):
                actual,proof=entry.load_collection(directory=folder)
                self.assertEqual(actual,values);self.assertEqual(proof['front_end'],front)
                with tarfile.open(folder/'data.tar.gz','w:gz') as tar:
                    for name in entry.INPUT_FILES:tar.add(folder/name,arcname=name)
                archived,_=entry.load_collection(archive=folder/'data.tar.gz')
                self.assertEqual(archived,actual)
            for key,value in [('sigma_cells',1.),('epoch',23),('sha256','wrong')]:
                original=front['midpoint_checkpoint'][key];front['midpoint_checkpoint'][key]=value
                with patch.object(entry,'INPUT_FILES',publish()):
                    with self.assertRaises(ValueError):entry.load_collection(directory=folder)
                front['midpoint_checkpoint'][key]=original
            with patch.object(entry,'INPUT_FILES',publish()):
                (folder/'collect/failure.json').write_text('{}')
                with self.assertRaises(ValueError):entry.load_collection(directory=folder)

    def test_changed_source_receipt_refused(self):
        identity,_=entry.checked_sources()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name in [*identity['sources'],entry.SOURCES]:
                target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes((entry.ROOT/name).read_bytes())
            (root/entry.CALIBRATION).write_text('{}')
            with self.assertRaises(ValueError):entry.checked_sources(root)

    def test_axis_contract_checks_bytes_generated_short_and_unconfirmed_physics(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            r=row(sequence='real_seq12')
            r['gt']=[100.,200.,100.,100./2.1,0.]
            annotation=b'50 176.2 150 176.2 150 223.8 50 223.8 grab 0\n'
            name='train/annfiles/'+r['image']+'.txt'
            target=root/entry.DATA/name;target.parent.mkdir(parents=True);target.write_bytes(annotation)
            axis=json.dumps(dict(shapes=[dict(label='axis',shape_type='line',points=[[50.,200.],[150.,200.]])])).encode()
            axis_path=root/entry.DATA/'provenance/axis_k2p1/real_seq12/axis_json'/Path(r['image']+'.json')
            axis_path.parent.mkdir(parents=True);axis_path.write_bytes(axis)
            conversion=dict(k=2.1,sequences=dict(real_seq12=dict(k0=2.1,model_output_used=False,
                target_geometry='central_grab_structure_defined_by_axis',records=[dict(filename=r['image']+'.txt',
                source_json=r['image']+'.json',source_json_sha256=entry.digest(axis),axis_length_px=100.)])))
            (root/entry.CONVERSION).write_text(json.dumps(conversion))
            cal=root/entry.CALIBRATION;cal.parent.mkdir(parents=True)
            cal.write_text(json.dumps(dict(calibration_id='test',coordinate_contract={'target':'z'},
                                          formula='not executed',deployable_inputs=[])))
            labels=dict(annotation_policy='test-only axis',records=[dict(image=r['image'],sequence=r['sequence'],
                split='train',annfile=name,annotation_sha256=entry.digest(annotation),annotation_family='axis_k2p1')])
            actual=entry.annotation_audit([r],labels,root)
            self.assertEqual(actual['groups']['real_seq12']['axis_GT_mismatches'],0)
            self.assertFalse(actual['records'][0]['axis_check']['short_edge_independently_annotated'])
            self.assertTrue(actual['physical_reference_match'].startswith('UNCONFIRMED'))
            r['gt'][3]=40.
            self.assertEqual(entry.annotation_audit([r],labels,root)['groups']['real_seq12']['axis_GT_mismatches'],1)
            axis_path.write_bytes(b'{}')
            with self.assertRaises(ValueError):entry.annotation_audit([r],labels,root)

    def test_source_contract_and_real_sampling_boundary(self):
        identity,labels = entry.checked_sources()
        self.assertEqual(len(labels['records']),3445)
        self.assertTrue(identity['sources'])
        source = entry.sampling_map()
        a=dict(sequence='real_seq12',frame_id=121,sampling=source['real_seq12_00121'])
        b=dict(sequence='real_seq12',frame_id=122,sampling=source['real_seq12_00122'])
        self.assertFalse(d.contiguous(a,b))

    def test_stdlib_only_and_no_training_or_policy_import(self):
        for module in (d,entry):
            tree=ast.parse(Path(module.__file__).read_text())
            imports=[]
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):imports.extend(a.name for a in node.names)
                elif isinstance(node,ast.ImportFrom):imports.append(node.module or '')
            self.assertFalse(any(any(word in name for word in ('torch','numpy','mmcv','cv2','reliability'))
                                 for name in imports))
        self.assertTrue(all(v==0 for v in entry.protocol_document()['scope'].values()))

    def test_shell_stops_on_test_failure_and_archives_at_work_dirs_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);tool=root/'crane_project/tools/run_port_geometry_size_support_v1.sh'
            tool.parent.mkdir(parents=True)
            tool.write_bytes((entry.ROOT/'crane_project/tools'/tool.name).read_bytes())
            binaries=root/'bin';binaries.mkdir();calls=root/'calls'
            python=binaries/'python'
            python.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> '+shlex.quote(str(calls))+'\nexit 7\n')
            python.chmod(0o755)
            hashing=binaries/'sha256sum';hashing.write_text('#!/usr/bin/env bash\nprintf "test-only-hash %s\\n" "$1"\n')
            hashing.chmod(0o755)
            env=dict(os.environ,PATH=str(binaries)+os.pathsep+os.environ['PATH'])
            result=subprocess.run(['bash',str(tool)],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,7,result.stdout+result.stderr)
            self.assertEqual(len(calls.read_text().splitlines()),1)
            packages=list((root/'work_dirs').glob('*.tar.gz'))
            self.assertEqual(len(packages),1)
            with tarfile.open(packages[0]) as tar:
                paths=[n for n in tar.getnames() if n.endswith('/run_exit_code.txt')]
                self.assertEqual(tar.extractfile(paths[0]).read(),b'7\n')
            self.assertFalse(list((root/'work_dirs').glob('*/diagnose')))


if __name__ == '__main__':
    unittest.main()
