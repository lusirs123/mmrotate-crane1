#!/usr/bin/env python3
"""Frozen Train-03: add only EOOD24/SymEOOD20; reuse sealed B24/midpoint outputs."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.utils import port_depth_four_arm_v1 as c
from tools.depth import audit_port_midpoint_depth_v1 as prior

o=c.original
PROTOCOL=ROOT/'tools/depth/port_depth_four_arm_v1_protocol.json'
SOURCES=ROOT/'tools/depth/port_depth_four_arm_v1_sources.json'
EXTRA={
    'crane_project/utils/port_depth_four_arm_v1.py','tools/depth/audit_port_depth_four_arm_v1.py',
    'tools/depth/port_depth_four_arm_v1_protocol.json','tests/test_port_depth_four_arm_v1.py',
    'tools/depth/run_port_depth_four_arm_v1.sh',str(prior.SOURCES.relative_to(ROOT)),
    'crane_project/configs/crane_eood_k1.py',c.BASELINES['eood']['config'],c.BASELINES['symeood']['config'],
    'mmrotate/models/detectors/base.py','mmrotate/models/detectors/single_stage.py',
    'mmrotate/models/detectors/eood.py','mmrotate/models/dense_heads/eood_head.py',
    'mmrotate/models/dense_heads/rotated_eood_head.py',
    'mmrotate/core/post_processing/bbox_nms_rotated.py',
    c.ARCHIVE+'/manifest.json',
    *{c.ARCHIVE+'/'+spec['stem']+suffix for spec in c.BASELINES.values()
      for suffix in ('_port_day2night_seq06_v1.py.txt','_port_day2night_v1.py.txt')},
}


def checked_sources():
    previous,cal=prior.checked_sources()
    manifest=json.loads(SOURCES.read_text());paths=set(previous['sources'])|EXTRA
    actual={p:o.sha(ROOT/p) for p in sorted(paths)}
    if (manifest.get('protocol')!=c.VERSION or manifest.get('sources')!=actual
            or manifest.get('parent_manifest_sha256')!=o.sha(prior.SOURCES)
            or json.loads(PROTOCOL.read_text())!=c.protocol_document()):
        raise ValueError('Four-arm depth source/protocol differs')
    return dict(protocol=c.VERSION,sources=actual,parent_identity=previous,
                source_manifest_sha256=o.sha(SOURCES)),cal


def load_reference(path):
    """Hash all artifact bytes, but parse only predictions/receipts before inference."""
    path=Path(path).resolve()
    if o.sha(path)!=c.REFERENCE_SHA:
        raise ValueError('Requires the already reviewed 20261006 B/M archive; do not rerun B/M')
    with tarfile.open(path) as tar:
        members=tar.getmembers();names={m.name for m in members}
        if (len(names)!=len(members) or any(m.name.startswith('/') or '..' in Path(m.name).parts
                or not (m.isfile() or m.isdir()) for m in members)):
            raise ValueError('Unsafe reference archive')
        audit='port_midpoint_depth_v1_audit_20261006_095629/'
        def raw(name):return tar.extractfile(audit+name).read()
        def read(name):return json.loads(raw(name))
        index=read('artifacts.json')
        for name,digest in index['sha256'].items():
            if hashlib.sha256(raw(name)).hexdigest()!=digest:
                raise ValueError('Reference artifact bytes differ')
        source=read('source_identity.json');proof=read('inference_proof.json')
        if (source!=prior.checked_sources()[0] or read('protocol.json')!=o.protocol_document()
                or read('completion.json')['status']!='FROZEN_TRAIN03_DEPTH_DIAGNOSTIC_COMPLETE_REVIEW_REQUIRED'
                or proof['detector_state_before']!=proof['detector_state_after']
                or proof['midpoint_state_before']!=proof['midpoint_state_after']
                or proof['midpoint_state_after']!=o.HEAD_DIGEST
                or proof['counters']!=dict(feature_extractions=981,native_head_calls=2943)
                or proof['detector_updates']!=0 or proof['midpoint_updates']!=0):
            raise ValueError('Sealed B/M proof differs')
        weight=read('weight_identity.json');dataset=read('dataset_identity.json')
        if (weight['b_sha256']!=o.B_SHA or weight['midpoint_sha256']!=o.HEAD_SHA
                or dataset['manifest_sha256']!=o.MANIFEST_SHA
                or dataset['frames_jsonl_sha256']!=o.TRUTH_SHA
                or dataset['image_identity_sha256']!=o.IMAGE_SHA):
            raise ValueError('Sealed reference model/data identity differs')
        predictions=[json.loads(line) for line in raw('predictions.jsonl').splitlines()]
        if len(predictions)!=981 or any(p['frame_id']!='frame_%05d'%i for i,p in enumerate(predictions)):
            raise ValueError('Incomplete reference stream')
        # Summary bytes remain uninterpreted until both new streams have closed.
        saved_summary=raw('summary.json')
    return predictions,proof,dict(path=str(path),sha256=c.REFERENCE_SHA),saved_summary


def count_main_forward(head,counts):
    # SymEOOD.simple_test calls .forward directly, which bypasses nn.Module hooks.
    # Wrap only for counting; original bound body/arguments/results are unchanged.
    native=head.forward
    def counted(*args,**kwargs):
        counts[1]+=1
        return native(*args,**kwargs)
    head.forward=counted
    return native


def native_baseline(arm,images,weight,selection,out,gpu,expected_runtime):
    import numpy as np
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmcv.utils import import_modules_from_strings
    from mmdet.datasets.pipelines import Compose
    from mmrotate.models import build_detector
    from crane_project.tools import eval_port_geometry_midpoint_sigma15_v1 as sealed
    torch,g=sealed.torch,sealed.g
    if int(os.environ.get('WORLD_SIZE','1'))!=1 or not torch.cuda.is_available() or not 0<=gpu<torch.cuda.device_count():
        raise ValueError('Single logical CUDA device required')
    torch.cuda.set_device(gpu);torch.set_num_threads(1)
    for key,value in c.protocol_document()['numerical'].items():
        if key=='cudnn_benchmark':torch.backends.cudnn.benchmark=value
        elif key=='cudnn_deterministic':torch.backends.cudnn.deterministic=value
        elif key=='matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32=value
        elif key=='cudnn_allow_tf32':torch.backends.cudnn.allow_tf32=value
    spec_identity=c.BASELINES[arm]
    cfg=Config.fromfile(str(ROOT/spec_identity['config']))
    bcfg=g.check_cfg()
    if cfg.data.test.pipeline!=bcfg.data.test.pipeline:
        raise ValueError('Standard native transform differs from sealed B pipeline')
    import_modules_from_strings(**cfg.custom_imports)
    model=deepcopy(cfg.model);model.pretrained=None
    # EOOD shared predictor construction retains its original train_cfg.
    if arm!='eood':model.train_cfg=None
    detector=build_detector(model)
    payload=load_checkpoint(detector,str(weight),map_location='cpu',strict=True)
    meta=c.checkpoint_meta(payload['meta'],cfg.to_dict(),spec_identity['epoch']);del payload
    g.freeze_detector(detector.cuda(gpu))
    if any(p.dtype!=torch.float32 for p in detector.parameters()):
        raise ValueError('Fixed baseline must use original FP32 parameters')
    runtime=sealed.f.runtime(gpu)
    for key in ('torch','cuda','cudnn','mmcv','mmdet','mmrotate','opencv'):
        if runtime.get(key)!=expected_runtime.get(key):raise ValueError('Runtime differs from sealed B/M: '+key)
    pipeline=Compose(deepcopy(cfg.data.test.pipeline));initial=g.state_digest(detector)
    counts=[0,0];records=[];start=time.monotonic()
    def features_hook(*args):counts[0]+=1
    hook=detector.neck.register_forward_hook(features_hook)
    main_forward=count_main_forward(detector.bbox_head,counts)
    try:
        with (out/(arm+'.predictions.jsonl')).open('x',encoding='utf-8') as stream:
            for i,source in enumerate(images):
                if o.sha(source['image_path'])!=source['sha256']:raise ValueError('Input image changed')
                image,metas=prior.native_view(pipeline,source,gpu,sealed.old)
                if image.dtype!=torch.float32:raise ValueError('FP32 native input required')
                with torch.no_grad():value=detector.simple_test(image,metas,rescale=True)
                if len(value)!=1 or len(value[0])!=1:raise ValueError('One image/one class required')
                candidates=np.asarray(value[0][0])
                if candidates.ndim!=2 or candidates.shape[1]!=6:raise ValueError('Expected native Nx6')
                boxes=candidates.tolist();box,index=c.top1(boxes,arm)
                record=dict(frame_id=source['frame_id'],image_sha256=source['sha256'],
                    original_size_hw=list(metas[0]['ori_shape'][:2]),
                    scale_xy=[float(v) for v in metas[0]['scale_factor'][:2]],
                    box=box,selected_native_index=index,native_candidate_count=len(boxes),native_candidates=boxes)
                stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();records.append(record)
                if i==0 or (i+1)%25==0 or i+1==981:
                    print('%s inference %d/981 %.1fs'%(arm,i+1,time.monotonic()-start),flush=True)
    finally:
        hook.remove();detector.bbox_head.forward=main_forward
    final=g.state_digest(detector);g.assert_detector_frozen(detector)
    if initial!=final or counts!=[981,981]:raise ValueError('Incomplete native work or changed frozen state')
    if o.sha(weight)!=spec_identity['checkpoint_sha256'] or o.sha(selection)!=spec_identity['selection_sha256']:
        raise ValueError('Fixed input changed during inference')
    proof=dict(runtime=runtime,checkpoint_meta=meta,state_before=initial,state_after=final,
        feature_extractions=counts[0],native_main_head_calls=counts[1],updates=0,
        metric_truth_read=False,native_policy_modified=False,wall_seconds=time.monotonic()-start,
        wall_time_role='diagnostic duration, not deployment FPS')
    o.write_new(out/(arm+'.proof.json'),proof)
    del detector,pipeline;torch.cuda.empty_cache()
    return records,proof


def evaluate(reference,baseline_predictions,truths,manifest,calibration):
    refrows,refs=o.evaluate(reference,truths,manifest,calibration)
    rows={'symeood_b':refrows,'symeood_b_midpoint':refrows}
    keys={'symeood_b':'b','symeood_b_midpoint':'midpoint'}
    groups={'symeood_b':refs['groups']['b'],'symeood_b_midpoint':refs['groups']['midpoint']}
    for arm,preds in baseline_predictions.items():
        rows[arm],groups[arm]=c.evaluate_one(preds,truths,manifest,calibration);keys[arm]='b'
    groups['gt_obb']=refs['groups']['gt_obb']
    result=dict(protocol=c.VERSION,sequence_id=o.SEQUENCE,frame_count=981,groups=groups,
        direct_depth_errors={arm:c.direct_errors(rows[arm],keys[arm]) for arm in rows},
        offline_error_propagation={arm:c.propagation(rows[arm],calibration,keys[arm]) for arm in rows},
        by_truth_depth={},by_theta={},paired_deltas={},parameter_refit=False,
        reliability_used=False,dino_included=False,fixed_dev_read=False,unknown_read=False,
        evidence_role='calibration-train diagnostic; reused historical B/M vs new baseline outputs')
    for label,low,high in [('around12m',0,14),('around16m',14,18),('around20m',18,22),('around24m',22,float('inf'))]:
        result['by_truth_depth'][label]={arm:o.summarize([r for r in rs if low<=r['truth_z_m']<high],keys[arm]) for arm,rs in rows.items()}
    for label,low,high in [('theta_0_3',0,3),('theta_3_5',3,5),('theta_5_8',5,8),('theta_8_inf',8,float('inf'))]:
        result['by_theta'][label]={arm:o.summarize([r for r in rs if low<=r['theta_total_deg']<high],keys[arm]) for arm,rs in rows.items()}
    available=[arm for arm in c.ARMS if arm in rows]
    for left,right in zip(available,available[1:]):
        changes=[]
        for a,b in zip(rows[left],rows[right]):
            za,zb=a[keys[left]]['depth']['z_m'],b[keys[right]]['depth']['z_m']
            if za is not None and zb is not None:
                changes.append(abs(zb-b['truth_z_m'])-abs(za-a['truth_z_m']))
        result['paired_deltas'][left+'__to__'+right]=dict(paired_numeric_count=len(changes),
            absolute_error_delta_m=o.describe(changes),improved=sum(v<-1e-12 for v in changes),
            worsened=sum(v>1e-12 for v in changes),tied=sum(abs(v)<=1e-12 for v in changes))
    return rows,keys,result,refs


def same_summary(actual,expected):
    """Preserve all fields/counts; allow only reviewed scalar reproduction error."""
    import math
    if isinstance(expected,dict):
        if not isinstance(actual,dict) or actual.keys()!=expected.keys():raise ValueError('Summary fields differ')
        for key in expected:same_summary(actual[key],expected[key])
    elif isinstance(expected,list):
        if not isinstance(actual,list) or len(actual)!=len(expected):raise ValueError('Summary length differs')
        for a,b in zip(actual,expected):same_summary(a,b)
    elif isinstance(expected,float):
        if not isinstance(actual,(int,float)) or not math.isclose(actual,expected,rel_tol=2e-12,abs_tol=1e-10):
            raise ValueError('Summary numeric reproduction differs')
    elif actual!=expected:raise ValueError('Summary identity/count differs')


def artifacts(out):
    o.write_new(out/'artifacts.json',dict(protocol=c.VERSION,sha256={
        p.name:o.sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='artifacts.json'}))


def run(args):
    identity,cal=checked_sources();out=Path(args.out_dir).resolve();out.mkdir(parents=True,exist_ok=False)
    o.write_new(out/'sources.json',identity);o.write_new(out/'protocol.json',c.protocol_document())
    if args.stage=='check':
        o.write_new(out/'completion.json',dict(status='STATIC_FOUR_ARM_DEPTH_CONTRACT_PASS',protocol=c.VERSION))
        artifacts(out);print('STATIC_FOUR_ARM_DEPTH_CONTRACT_PASS');return
    reference,refproof,refidentity,summary_bytes=load_reference(args.reference_archive)
    manifest,truth_path,images,data_identity=prior.checked_dataset(args.data_root)
    o.write_new(out/'dataset_identity.json',data_identity);o.write_new(out/'image_sources.json',images)
    o.write_new(out/'reused_reference.json',refidentity)
    predictions={};proofs={};weight_receipts={}
    if args.stage=='audit':
        for arm,spec in c.BASELINES.items():
            weight=o.checked_weight(ROOT,getattr(args,arm+'_checkpoint'),'epoch_%d.pth'%spec['epoch'],spec['checkpoint_sha256'])
            selection=o.checked_weight(ROOT,getattr(args,arm+'_selection'),'sweep_results.json',spec['selection_sha256'])
            receipt=c.selection_contract(ROOT,arm,json.loads(selection.read_text()))
            weight_receipts[arm]=dict(checkpoint=str(weight),checkpoint_sha256=spec['checkpoint_sha256'],
                selection=str(selection),selection_sha256=spec['selection_sha256'],original_selection=receipt)
        o.write_new(out/'weights.json',weight_receipts)
        for arm in c.BASELINES:
            receipt=weight_receipts[arm]
            predictions[arm],proofs[arm]=native_baseline(arm,images,Path(receipt['checkpoint']),Path(receipt['selection']),out,args.gpu,refproof['runtime'])
    # No new GT parsing until both new prediction streams are closed.
    truth=prior.load_truth(truth_path,manifest)
    rows,keys,result,refsummary=evaluate(reference,predictions,truth,manifest,cal)
    old_summary=json.loads(summary_bytes)
    for arm,key in [('symeood_b','b'),('symeood_b_midpoint','midpoint')]:
        same_summary(refsummary['groups'][key],old_summary['groups'][key])
    if args.stage=='audit':
        with (out/'depth_rows.jsonl').open('x',encoding='utf-8') as stream:
            for i,t in enumerate(truth):
                row=dict(frame_id='frame_%05d'%i,truth_z_m=t['camera_geometry']['z_cg_opt_m'],methods={})
                for arm in c.ARMS:
                    value=rows[arm][i][keys[arm]];z=value['depth']['z_m']
                    row['methods'][arm]=dict(value,depth_signed_error_m=z-row['truth_z_m'] if z is not None else None,
                        depth_abs_error_m=abs(z-row['truth_z_m']) if z is not None else None)
                stream.write(json.dumps(row,allow_nan=False)+'\n')
    o.write_new(out/'summary.json',result)
    if checked_sources()[0]!=identity or o.sha(truth_path)!=o.TRUTH_SHA:raise ValueError('Sources/truth changed')
    status=('FOUR_ARM_DEPTH_BASELINE_EXTENSION_COMPLETE_REVIEW_REQUIRED' if args.stage=='audit'
            else 'SEALED_B_M_DEPTH_PROPAGATION_DIAGNOSIS_COMPLETE')
    o.write_new(out/'completion.json',dict(protocol=c.VERSION,status=status,new_inference_arms=list(predictions),
        reused_arms=['symeood_b','symeood_b_midpoint'],new_frame_forwards=981*len(predictions),
        updates=0,parameter_refit=False,proofs=proofs))
    artifacts(out);print(status,flush=True)
    for arm in c.ARMS:
        if arm in result['groups']:
            g=result['groups'][arm];m=g['depth_metrics'];d=result['direct_depth_errors'][arm]
            print('%s outputs=%d/981 numeric=%d/981 MAE=%s median=%s P95=%s bias=%s within1m=%d/981'%(arm,
                g['output_frame_count'],g['numeric_depth_count'],m['mae_m'],d['median_abs_error_m'],m['p95_abs_error_m'],m['bias_m'],d['abs_error_coverage']['1.0']['count']),flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('check','diagnose','audit'),required=True);p.add_argument('--out-dir',required=True)
    p.add_argument('--data-root',type=Path,default=ROOT/'crane_project/data/webots_depth')
    p.add_argument('--reference-archive',type=Path,default=ROOT/'work_dirs/port_midpoint_depth_v1_20261006_095629.tar.gz')
    p.add_argument('--gpu',type=int,default=0)
    for arm in c.BASELINES:
        p.add_argument('--'+arm+'-checkpoint',type=Path);p.add_argument('--'+arm+'-selection',type=Path)
    return p


if __name__=='__main__':run(parser().parse_args())
