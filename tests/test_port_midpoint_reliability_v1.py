"""Bounded NumPy geometry, final-box binding and migration stage contracts."""
from argparse import Namespace
from contextlib import nullcontext
from copy import deepcopy
import ast
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from crane_project.tools import run_port_midpoint_reliability_v1 as entry
from crane_project.utils import port_midpoint_reliability_v1 as new
from crane_project.utils import port_size_reference_v1 as size
from crane_project.utils import port_simple_component_reliability_v1 as simple

SETTINGS = json.loads(entry.PROTOCOL.read_text())
OLD = json.loads(entry.base.PROTOCOL.read_text())


def meta(scale=1., flip=False):
    return dict(img_shape=[int(256*scale)]*2+[3], ori_shape=[256,256,3],
        pad_shape=[256,256,3], scale_factor=[scale]*4,
        flip=flip, flip_direction='horizontal' if flip else None)


def box(): return [128.,128.,80.,40.,.37]


def fixtures():
    train=[]; val=[]
    for dest, split, count in ((train,'train',32),(val,'val',20)):
        for domain in ('real','sim'):
            for i in range(count):
                b=box()+[.4+i/(count*2)]
                if i%2: b[2]*=1.3
                if i%3: b[4]+=.1
                dest.append(dict(image=split+'_'+domain+'_%03d'%i, domain=domain,
                    sequence=domain+'_seq',frame_id=i,split=split,
                    image_size=[256,256],gt=box(),pred=b,b_original=deepcopy(b),
                    train_angle_eligible=True,angle_axis_well_defined=True))
    val[0]['pred']=val[0]['b_original']=None
    return train,val


class ReaderTests(unittest.TestCase):
    def test_gaussian_sizes_center_direction_original_coordinates(self):
        for theta in (0.,.37,1.1,-1.4):
            b=box(); b[4]=theta
            for scale,flip in ((1.,False),(.5,False),(.5,True)):
                m=meta(scale,flip); target,_=size.target_map(b,m)
                r=new.template_reference(target,b,m,SETTINGS)
                self.assertTrue(r['defined'],r)
                np.testing.assert_allclose([r['long_original_px'],r['short_original_px']],b[2:4],rtol=1e-5)
                np.testing.assert_allclose(r['center_original'],b[:2],atol=1e-5)
                self.assertLess(abs((r['angle_rad']-theta+math.pi/2)%math.pi-math.pi/2),1e-5)

    def test_controlled_low_response_tail_distorts_moments_but_not_core_fit(self):
        m=meta(); target,_=size.target_map(box(),m); xy,_=size.grid(m)
        distance=np.linalg.norm(xy-box()[:2],axis=-1)
        probability=np.clip(target+.03*((distance>60)&(distance<105)),0.,1.)
        old=size.reference_from_map(probability,box(),m,SETTINGS)
        fresh=new.template_reference(probability,box(),m,SETTINGS)
        self.assertTrue(old['defined']); self.assertTrue(fresh['defined'])
        self.assertGreater(old['short_original_px']/40.,3.)
        self.assertAlmostEqual(fresh['short_original_px'],40.,places=4)
        diagnostic=new.offline_map_evidence(probability,box(),box(),m,SETTINGS)
        self.assertGreater(diagnostic['gt_outside_short_moment_fraction'],.9)
        self.assertEqual(diagnostic['core_outside_gt_cells'],0)

    def test_offline_map_diagnostic_accepts_actual_six_value_detection_without_mutation(self):
        target,_=size.target_map(box(),meta())
        expected=new.offline_map_evidence(target,box(),box(),meta(),SETTINGS)
        for score in (.06,.8,1.):
            detection=box()+[score]; before=deepcopy(detection)
            actual=new.offline_map_evidence(target,box(),detection,meta(),SETTINGS)
            self.assertEqual(actual,expected); self.assertEqual(detection,before)
        for bad in (box()+[float('nan')],box()+[.01],box()+[1.1],box()+[.8,.1]):
            with self.assertRaises(ValueError): new.offline_map_evidence(target,box(),bad,meta(),SETTINGS)

    def test_fixed_reference_detects_both_size_directions_not_center_or_angle(self):
        target,_=size.target_map(box(),meta()); r=new.template_reference(target,box(),meta(),SETTINGS)
        row=dict(image='fixed',sequence='real_seq',domain='real',gt=box())
        probe=entry.reference.probe_record(row,r,SETTINGS)
        self.assertTrue(all(v>.3 for v in probe['size_probe_risk_gaps'].values()))
        self.assertLess(probe['nuisance_control_max_risk_difference'],1e-12)
        swapped=box(); swapped[2:4]=swapped[3:1:-1]; swapped[4]+=math.pi/2
        r2=new.template_reference(target,swapped,meta(),SETTINGS)
        self.assertEqual(r,r2)

    def test_uniform_low_mass_maps_explicitly_unavailable(self):
        for value in (0.,.01,.9):
            r=new.template_reference(np.full((128,128),value),box(),meta(),SETTINGS)
            self.assertFalse(r['defined'])
            self.assertIsNone(size.size_reading(box(),r)['risk'])

    def test_invalid_map_or_anisotropic_restore_rejected(self):
        for a in (np.zeros((5,5)),np.full((128,128),np.nan),np.full((128,128),1.1)):
            with self.assertRaises(ValueError): new.template_reference(a,box(),meta(),SETTINGS)
        m=meta(); m['scale_factor']=[.5,.8,.5,.8]
        with self.assertRaises(ValueError): new.template_reference(np.zeros((128,128)),box(),m,SETTINGS)

    def test_unresolved_or_image_clipped_core_is_unavailable(self):
        for b,reason in (([128,128,80,8,.37],'unresolved_short_edge'),([4,128,80,40,0.],'core_clipped')):
            target,_=size.target_map(b,meta()); r=new.template_reference(target,b,meta(),SETTINGS)
            self.assertFalse(r['defined']); self.assertEqual(r['reason'],reason)

    def test_non_gaussian_plateau_and_disconnected_peak_guards(self):
        a=np.zeros((128,128)); a[45:65,45:65]=.9
        r=new.template_reference(a,box(),meta(),SETTINGS)
        self.assertFalse(r['defined']); self.assertEqual(r['reason'],'not_a_gaussian_peak')
        mask=np.zeros((7,7),bool); mask[1:3,1:3]=True; mask[5:7,5:7]=True
        self.assertEqual(new.component_at_peak(mask,(1,1)).sum(),4)

    def test_preview_is_fixed_three_panel_png(self):
        from PIL import Image
        t,v=size.target_map(box(),meta())
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'view.png'; new.save_map_preview(path,t,t,v)
            with Image.open(path) as image: self.assertEqual(image.size,(384,152))


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.train,self.val=fixtures()
        self.internal=simple.create_policy(self.train,self.val,OLD)
        self.front={'B':'fixed','midpoint':'ep23'}

    def test_new_contract_three_flags_and_exact_final_box_preservation(self):
        p=dict(protocol=new.VERSION,front_end=self.front,simple_policy=self.internal)
        runtime=new.MidpointReliability(p,self.front)
        final=box()+[.8]; before=deepcopy(final)
        for method in ('raw','score_only','simple'):
            d=runtime.decide(final,[256,256],method)
            self.assertEqual(d['final_box_original'],before)
            self.assertTrue(d['center_accepted'])
            self.assertTrue(all(type(d[c+'_accepted']) is bool for c in ('center','size','angle')))
        self.assertEqual(final,before)
        self.assertEqual(runtime.decide(None,[256,256])['final_box_original'],None)
        with self.assertRaises(ValueError): new.MidpointReliability(p,{'midpoint':'ep24'})

    def test_geometry_changes_features_and_correctness_even_at_same_score(self):
        b=box()+[.8]; improved=deepcopy(b); improved[2]*=1.15
        self.assertNotEqual(simple.geometry_errors(box(),b),simple.geometry_errors(box(),improved))
        self.assertFalse(np.array_equal(simple.descriptor(b,[256,256]),simple.descriptor(improved,[256,256])))
        self.assertEqual(b[5],improved[5])

    def test_pair_final_boxes_to_labels_not_the_legacy_prediction_values(self):
        source=self.train[0]; b=box()+[.8]; final=deepcopy(b); final[2]=84.
        p=dict(image=source['image'],sequence=source['sequence'],domain=source['domain'],
            frame_id=source['frame_id'],gt=box(),b=b,midpoint=final,candidate=final,accepted=True)
        paired=new.paired_rows([source],[p])[0]
        self.assertEqual(paired['pred'],final); self.assertEqual(paired['b_original'],b)
        self.assertEqual(source,self.train[0])
        for change in ({'accepted':False},{'midpoint':None},{'gt':[1,2,3,4,0]},
                       {'midpoint':final[:5]+[.7]},{'accepted':None}):
            with self.assertRaises(ValueError): new.paired_rows([source],[dict(p,**change)])
        p.update(b=None,midpoint=None,candidate=None,accepted=None)
        self.assertIsNone(new.paired_rows([source],[p])[0]['pred'])

    def test_probe_selection_is_14_fixed_reference_roles_independent_of_labels(self):
        settings=json.loads(entry.reference.PROTOCOL.read_text())
        counts=dict(real_seq01=339,real_seq05=560,real_seq06=466,real_seq12=141,real_seq13=304,sim_seq08=748)
        rows=[dict(image=s+'_%05d'%i,sequence=s,frame_id=i) for s,n in counts.items() for i in range(n)]
        part=size.partition(rows,settings); chosen=new.fixed_probe_rows(part)
        self.assertEqual(sum(r['reference_role']=='fit' for r in chosen),10)
        self.assertEqual(sum(r['reference_role']=='holdout' for r in chosen),4)
        for r in rows: r['gt']='changed'; r['pred']='changed'
        self.assertEqual([r['image'] for r in chosen],[r['image'] for r in new.fixed_probe_rows(size.partition(rows,settings))])

    def test_fit_calibrates_on_VAL_coverage_without_using_VAL_GT_for_parameters(self):
        changed=deepcopy(self.val)
        for r in changed: r['gt']=[30,60,12,6,-.8]
        self.assertEqual(self.internal,simple.create_policy(self.train,changed,OLD))

    def test_rank_separates_domains_sequences_and_empty_angle_support(self):
        result=entry.ranked(self.val,self.internal)
        self.assertEqual(result['real']['size']['simple']['outputs'],19)
        self.assertIn('real_seq',result)
        subset=deepcopy(self.val)
        for r in subset: r['angle_axis_well_defined']=False
        self.assertIsNone(entry.ranked(subset,self.internal)['real']['angle']['simple']['error_auroc'])

    def test_stage_completion_rejects_failed_changed_or_wrong_frontend_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); artifact=p/'report.json'; artifact.write_text('{}')
            contract={'front_end':self.front}
            proof=dict(status='pass',contract=contract,test_read=False,artifacts={'report.json':entry.base.sha(artifact)})
            (p/'completion.json').write_text(json.dumps(proof))
            entry.completed(p,'pass',contract)
            with self.assertRaises(ValueError): entry.completed(p,'pass',{'front_end':'other'})
            artifact.write_text('changed')
            with self.assertRaises(ValueError): entry.completed(p,'pass',contract)
            (p/'failure.json').write_text('{}')
            with self.assertRaises(ValueError): entry.completed(p,'pass',contract)

    def test_fit_outputs_three_paired_comparisons_no_input_mutation(self):
        final=deepcopy(self.val)
        for r in final:
            if r['pred'] is not None: r['pred'][2]*=.98
        original=(None,None,self.train,self.val,self.internal,None,None,None,None)
        contract={'front_end':self.front}
        before=deepcopy(final)
        with tempfile.TemporaryDirectory() as d:
            args=Namespace(out_dir=Path(d),collection_dir=Path('unused'))
            with patch.object(entry,'read_collection',return_value=(self.train,final)),patch.object(entry.base,'checked_sources',return_value=(OLD,{})):
                status=entry.fit(args,(SETTINGS,original,None,None,None,contract))
            report=json.loads((Path(d)/'fit_report.json').read_text())
            self.assertEqual(status,'MIDPOINT_SIMPLE_POLICY_FIT_COMPLETE')
            self.assertTrue(report['save_reload_policy_exact'])
            self.assertTrue(all(k in report for k in ('paired_formal_B_old_policy','midpoint_old_policy','midpoint_refit_policy')))
            self.assertEqual(report['genuine_TRAIN_error_support']['real']['center_bad'],0)
            self.assertEqual(report['error_limits'],OLD['error_limits'])
            self.assertFalse(report['test_read'])
        self.assertEqual(final,before)

    def test_source_manifest_and_Python38_syntax(self):
        p,identity=entry.checked_sources()
        self.assertEqual(p['midpoint_epoch'],23); self.assertFalse(p['test_read'])
        for name in identity['sources']:
            if name.endswith('.py'): ast.parse((entry.ROOT/name).read_text(),feature_version=(3,8))

    def test_prepare_uses_old_policy_for_provenance_without_mutating_new_policy_arg(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'head_epoch_23.pth').write_bytes(b'head')
            b_path=root/'B.pth'; b_path.write_bytes(b'B')
            frozen={'checkpoint_sha256':entry.base.sha(b_path)}
            identity={'frozen_b':frozen,'sources_sha256':'formal-source'}
            cache=dict(identity=identity,test_access=False,detector_updates=0,
                status='COMPLETE_FROZEN_B_TRAIN_VAL_CACHE',record_counts={'train_s1':2558,'train_s05':2558,'val_s1':887})
            cache_path=root/'cache_manifest.json'; cache_path.write_text(json.dumps(cache))
            chosen=dict(epoch=23,path='head_epoch_23.pth',sha256=entry.base.sha(root/'head_epoch_23.pth'))
            selection=dict(identity=identity,cache_manifest_sha256=entry.base.sha(cache_path))
            selection_path=root/'selection.json'; selection_path.write_text(json.dumps(selection))
            source=self.train[0]; b=box()+[.8]
            predictions=[dict(image=source['image'],sequence=source['sequence'],domain=source['domain'],
                frame_id=source['frame_id'],gt=source['gt'],b=b,midpoint=b,candidate=None,accepted=False)]
            original=(None,None,[],[source],{},None,None,None,{'frozen_b':frozen})
            args=Namespace(selection=selection_path,formal_cache=root,b_checkpoint=b_path,
                           policy=Path('NEW-policy'),old_policy=Path('OLD-policy'))
            proto=dict(SETTINGS,midpoint_sha256=chosen['sha256'])
            def parent(received):
                self.assertEqual(received.policy,args.old_policy); return original
            with patch.object(entry,'checked_sources',return_value=(proto,{})),patch.object(entry.reference,'prepare',side_effect=parent),patch.object(entry.formal_check,'validate_inputs',return_value=(predictions,{},dict(selected_checkpoint=chosen))):
                prepared=entry.prepare(args)
            self.assertEqual(args.policy,Path('NEW-policy'))
            self.assertEqual(prepared[4][0]['pred'],b)

    def test_collection_count_labels_fallback_and_locked_VAL_cannot_be_rebound(self):
        source=self.train[0]
        train=[dict(source,image='train_%04d'%i,frame_id=i) for i in range(2558)]
        val=[dict(source,image='val_%04d'%i,frame_id=i,split='val') for i in range(887)]
        def pair(rows):
            p=[dict(image=r['image'],sequence=r['sequence'],domain=r['domain'],frame_id=r['frame_id'],
                gt=r['gt'],b=r['pred'],midpoint=r['pred'],candidate=None,accepted=False) for r in rows]
            return new.paired_rows(rows,p)
        a,b=pair(train),pair(val)
        contract={'native_VAL_final_rows_sha256':simple.fingerprint(b)}
        old=(None,None,train,val)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); path=root/'predictions.jsonl'
            rows=[dict(r,reliability_role=role) for role,part in (('train',a),('val',b)) for r in part]
            def save():
                path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
                proof=dict(status='MIDPOINT_TRAIN_VAL_COLLECTION_COMPLETE',contract=contract,test_read=False,
                    artifacts={'predictions.jsonl':entry.base.sha(path)})
                (root/'completion.json').write_text(json.dumps(proof))
            save(); actual=entry.read_collection(root,contract,old)
            self.assertEqual(tuple(map(len,actual)),(2558,887))
            rows[-1]['pred']=deepcopy(rows[-1]['pred']); rows[-1]['b_original']=deepcopy(rows[-1]['b_original'])
            rows[-1]['pred'][5]=rows[-1]['b_original'][5]=.123
            save()
            with self.assertRaisesRegex(ValueError,'selected formal VAL'): entry.read_collection(root,contract,old)

    def test_collect_replay_tolerance_preserves_authoritative_selected_VAL(self):
        def predictions(rows,shift=0.):
            values=[]
            for r in rows:
                final=deepcopy(r['pred'])
                if final is not None: final[0]+=shift
                values.append(dict(image=r['image'],sequence=r['sequence'],domain=r['domain'],frame_id=r['frame_id'],
                    gt=r['gt'],b=r['pred'],midpoint=final,candidate=final,accepted=True if final else None))
            return values
        locked=new.paired_rows(self.val,predictions(self.val))
        original=(None,None,self.train,self.val)
        formal=SimpleNamespace(g=SimpleNamespace(state_digest=lambda head:'same'),
            evaluate=lambda head,records,device:predictions(self.val,1e-6) if records[0]['role']=='val' else predictions(self.train))
        torch=SimpleNamespace(load=lambda path,map_location:payloads[Path(path).stem],
                              cuda=SimpleNamespace(max_memory_allocated=lambda gpu:0))
        payloads={name:dict(identity='fixed',detector_state='same',records=[dict(role=role,scale=1.) for r in rows])
            for name,role,rows in (('train_s1','train',self.train),('val_s1','val',self.val))}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); out=root/'out'; out.mkdir()
            for name in payloads: (root/(name+'.pt')).write_bytes(b'fixture')
            cache=dict(detector_state='same',files={name+'.pt':entry.base.sha(root/(name+'.pt')) for name in payloads})
            args=Namespace(out_dir=out,formal_cache=root,gpu=0)
            with patch.object(entry,'load_midpoint',return_value=(formal,torch,object(),'cpu')):
                status=entry.collect(args,(SETTINGS,original,{'identity':'fixed'},cache,locked,{}))
            rows=[json.loads(s) for s in (out/'predictions.jsonl').read_text().splitlines()]
            actual=[{k:v for k,v in r.items() if k!='reliability_role'} for r in rows if r['reliability_role']=='val']
            self.assertEqual(actual,locked)
            report=json.loads((out/'collection_report.json').read_text())
            self.assertGreater(report['VAL_replay_max_absolute_box_difference'],0.)
            self.assertEqual(status,'MIDPOINT_TRAIN_VAL_COLLECTION_COMPLETE')

    def test_reader_summary_reports_common_support_and_undefined(self):
        target,_=size.target_map(box(),meta()); ref=new.template_reference(target,box(),meta(),SETTINGS)
        p=entry.reference.probe_record(dict(image='x',sequence='r',domain='real',gt=box()),ref,SETTINGS)
        records=[dict(image='x',reference_role='fit',references={'moments':ref,'template':ref},
                      reference_errors={'moments':.01,'template':.02},probes={'moments':p,'template':p})]
        result=entry.probe_summary(records)
        self.assertEqual(result['all']['common_defined']['views'],1)
        self.assertEqual(result['holdout']['template']['defined'],0)

    def test_full_probe_writes_14_diagnostics_with_six_value_final_boxes(self):
        target,_=size.target_map(box(),meta())
        class MapTensor:
            def sigmoid(self): return self
            def __getitem__(self,index): return self
            def cpu(self): return self
            def numpy(self): return target.copy()
        parts={}
        for role,sequences,start in (('fit',('real1','real2','real3','real4','sim1'),0),
                                     ('holdout',('real5','sim1'),100)):
            parts[role]=[dict(image=s+'_%03d'%i,sequence=s,frame_id=i,
                domain='sim' if s.startswith('sim') else 'real',gt=box(),image_size=[256,256])
                for s in sequences for i in range(start,start+3)]
        old=(None,None,None,None,None,None,parts,{},None)
        b=box()+[.8]; before=deepcopy(b)
        modules=(None,SimpleNamespace(no_grad=nullcontext,cuda=SimpleNamespace(max_memory_allocated=lambda gpu:0)),
                 object(),object(),object(),lambda feature:MapTensor())
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); checkpoint=root/'epoch_04.pth'; checkpoint.write_bytes(b'fixture')
            args=Namespace(out_dir=root,reference_checkpoint=checkpoint,gpu=0)
            with patch.object(entry,'online_modules',return_value=modules),patch.object(entry.base,'state_digest',return_value='unchanged'),patch.object(entry.reference,'numeric_meta',return_value=meta()),patch.object(entry.reference,'view',return_value=([object()],meta(),[meta()])),patch.object(entry,'midpoint_from_features',return_value=dict(b=b,midpoint=b)):
                status=entry.probe(args,(SETTINGS,old,None,None,None,{}))
            self.assertEqual(status,'FIXED_MIDPOINT_TRAIN_READER_COMPARISON_COMPLETE_REVIEW_REQUIRED')
            rows=[json.loads(line) for line in (root/'probe_rows.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),14)
            self.assertEqual(sum(r['reference_role']=='fit' for r in rows),10)
            self.assertTrue(all(r['offline_map_evidence']['defined'] for r in rows))
            self.assertTrue(all(r['final_box_original']==before for r in rows))
            self.assertEqual(len(list(root.glob('*.npz'))),14)
            self.assertEqual(len(list(root.glob('*.png'))),14)
            report=json.loads((root/'probe_report.json').read_text())
            self.assertEqual(report['state_before'],report['state_after'])
        self.assertEqual(b,before)


@unittest.skipIf(importlib.util.find_spec('torch') is None,'Actual Torch adapter test requires server Torch')
class TorchAdapterTests(unittest.TestCase):
    def test_shared_detached_P3_and_native_boxes_are_immutable_including_missing(self):
        import torch
        from crane_project.utils import port_geometry_refine_g_v1 as geometry
        from crane_project.utils.port_geometry_midpoint_v1 import SpatialMidpointHead
        m=dict(img_shape=[1024,1024,3],ori_shape=[256,256,3],pad_shape=[1024,1024,3],
               scale_factor=[4.]*4,flip=False,flip_direction=None)
        class Detector(torch.nn.Module):
            def __init__(self,b):
                super().__init__(); self.register_buffer('boxes',b)
            def simple_test_from_features(self,features,metas,rescale=False):
                b=self.boxes.clone()
                if not rescale: b[:,:5]=geometry.map_boxes(b[:,:5],metas[0])
                return [[b.numpy()]]
        g=SimpleNamespace(map_boxes=geometry.map_boxes,sample_local=geometry.sample_local,
                          flatten_prediction=lambda result:result[0][0].copy())
        formal=SimpleNamespace(g=g)
        features=[torch.zeros(1,256,128,128)]
        head=SpatialMidpointHead().eval().requires_grad_(False)
        for b in (torch.tensor([box()+[.8]]),torch.empty(0,6)):
            detector=Detector(b).eval().requires_grad_(False)
            before=entry.base.state_digest(head),entry.base.state_digest(detector)
            first=entry.midpoint_from_features(formal,torch,detector,head,features,m)
            again=entry.midpoint_from_features(formal,torch,detector,head,features,m)
            self.assertEqual(first,again)
            self.assertEqual(before,(entry.base.state_digest(head),entry.base.state_digest(detector)))
            self.assertEqual(first['b'],b[0].tolist() if len(b) else None)
            if len(b): self.assertEqual(first['midpoint'][5],b[0,5].item())
        with self.assertRaises(ValueError):
            entry.midpoint_from_features(formal,torch,detector,head,[features[0].requires_grad_()],m)


if __name__=='__main__': unittest.main()
