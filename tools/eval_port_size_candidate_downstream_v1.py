#!/usr/bin/env python3
"""User-authorized fixed failed-TRAIN candidate diagnosis; never fits/promotes.

Collect B/M and GT-free ROI with sealed cuDNN8801, then run the actual final
size head with its training cuDNN8500. Metric truth enters only after all three
prediction streams have closed. Original protocols/gates remain unchanged.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import eval_port_frozen_downstream_v1 as base
from crane_project.utils import port_midpoint_depth_v1 as o
from crane_project.utils import port_depth_four_arm_v1 as c
VERSION='port_size_candidate_downstream_v1_fixed_diagnostic'
PROTOCOL=ROOT/'tools/port_size_candidate_downstream_v1_protocol.json'
SOURCES=ROOT/'tools/port_size_candidate_downstream_v1_sources.json'
METHODS=('midpoint','size_candidate')
INPUT_NAMES=('roi','support','boxes_original','boxes_model','scale_xy','midpoint_original')


def checked_sources():
    p=json.loads(PROTOCOL.read_text());s=json.loads(SOURCES.read_text())
    if p['protocol']!=VERSION or s['protocol']!=VERSION or s['protocol_sha256']!=o.sha(PROTOCOL):
        raise ValueError('Wrong diagnostic protocol')
    actual={n:o.sha(ROOT/n) for n in s['sources']}
    if actual!=s['sources']:raise ValueError('Reviewed source bytes differ')
    for n,digest in s['parents'].items():
        parent=json.loads((ROOT/n).read_text())
        if o.sha(ROOT/n)!=digest or any(actual.get(k)!=v for k,v in parent['sources'].items()):
            raise ValueError('Parent source closure changed')
    old,calibration,_=base.checked_sources()
    if p['base_protocol_sha256']!=o.sha(base.PROTOCOL):raise ValueError('Base contract changed')
    return p,old,calibration,dict(manifest_sha256=o.sha(SOURCES),sources=actual)


def write_rows(path,rows):
    with Path(path).open('x') as stream:
        for r in rows:stream.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')


def preserve(m,n):
    if (m is None)!=(n is None):raise ValueError('Size correction changed missing identity')
    if m is None:return
    o.canonical(n)
    if len(n)!=6 or any(m[i]!=n[i] for i in (0,1,4,5)):
        raise ValueError('Center/raw angle/score changed')
    if (m[2]<m[3])!=(n[2]<n[3]) or o.canonical(m)[4]!=o.canonical(n)[4]:
        raise ValueError('Canonical long-axis identity changed')


def replay(rows,saved,fields=('b','midpoint')):
    if len(rows)!=len(saved):raise ValueError('Baseline replay count differs')
    maximum=0.
    for a,b in zip(rows,saved):
        if a['frame_id']!=b['frame_id']:raise ValueError('Baseline frame order differs')
        for k in fields:
            x,y=a[k],b[k]
            if (x is None)!=(y is None):raise ValueError('Baseline output replay differs')
            if x is not None:
                error=max(abs(v-w) for v,w in zip(x,y));maximum=max(maximum,error)
                if len(x)!=6 or len(y)!=6 or x[5]!=y[5] or not np.allclose(x[:4],y[:4],atol=1e-4,rtol=0.) or abs(x[4]-y[4])>1e-6:
                    raise ValueError('Baseline native replay differs')
    return dict(frames=len(rows),max_abs_scalar_delta=maximum,score_and_missing_exact=True)


def checkpoint_header(payload,p):
    expected=dict(protocol='port_geometry_size_boundary_continuous_v1',stage='finite',
        arm='boundary_continuous',updates=200,experimental_only=True,automatic_promotion=False)
    if any(payload.get(k)!=v for k,v in expected.items()) or payload.get('head_digest')!=p['candidate_digest']:
        raise ValueError('Wrong final candidate header/digest')
    if 'head_state' not in payload:raise ValueError('Missing trained candidate state')


def port_sources(args,old):
    base.indexed_bundle(args.metadata_dir,old['reliability']['metadata_files'])
    # Strip all label/quality fields from the inference inventory.
    metadata=base.read_rows(args.metadata_dir/'test_qualities.jsonl')
    if len(metadata)!=1440:raise ValueError('Full port TEST required')
    result=[]
    for r in metadata:
        path=args.port_root/(r['image']+'.jpg')
        if o.sha(path)!=r['image_sha256']:raise ValueError('Port image identity changed')
        result.append(dict(frame_id=r['frame_id'],image=r['image'],image_path=str(path.resolve()),
            sha256=r['image_sha256'],expected_hw=[r['image_size'][1],r['image_size'][0]]))
    if len({r['image'] for r in result})!=1440:raise ValueError('Duplicate port identity')
    return result


def view(pipeline,source,gpu,helpers):
    from mmcv.parallel import collate,scatter
    value=scatter(collate([pipeline(dict(img_info=dict(filename=source['image_path']),img_prefix=None))],samples_per_gpu=1),[gpu])[0]
    if len(value['img'])!=1 or len(value['img_metas'])!=1:raise ValueError('Single view required')
    image,metas=value['img'][0],value['img_metas'][0]
    if (tuple(image.shape)!=(1,3,1024,1024) or len(metas)!=1 or metas[0].get('flip',False)
        or Path(metas[0]['filename']).resolve()!=Path(source['image_path']).resolve()
        or list(metas[0]['ori_shape'][:2])!=source['expected_hw']):raise ValueError('Native original-coordinate view differs')
    helpers.g.checked_meta(metas[0])
    return image,metas


def collect(args,p,old,identity,out):
    # Inventory labels are discarded; no GT enters the head or ROI payload.
    inventories={'port_test':port_sources(args,old)}
    for name in old['depth_order']:
        _,_,sources=base.checked_dataset(args.data_root,name,old['sequences'][name])
        inventories[name]=[dict(r,expected_hw=[1024,1024]) for r in sources]
    detector,head,pipeline,helpers,runtime=base.prior.load_native(ROOT/old['weight_paths']['symeood_b'],ROOT/old['weight_paths']['symeood_b_midpoint'],args.gpu)
    torch,g=helpers.torch,helpers.g
    before=[g.state_digest(detector),g.state_digest(head)];counts=dict(feature_extractions=0,native_head_calls=0)
    def audit(e):counts['feature_extractions' if e['stage']=='test_feature_extracted' else 'native_head_calls']+=1
    captured=[]
    def hook(module,inputs,output):
        if len(inputs)!=5:raise ValueError('Unexpected midpoint forward contract')
        captured.append({k:v.detach().cpu().clone() for k,v in zip(INPUT_NAMES,(*inputs,output['boxes_original']))})
    handle=head.register_forward_hook(hook);artifacts={};start=time.monotonic()
    try:
        for name,sources in inventories.items():
            folder=out/name;folder.mkdir();cache=[];rows=[]
            for i,s in enumerate(sources):
                if o.sha(s['image_path'])!=s['sha256']:raise ValueError('Image changed during inference')
                image,metas=view(pipeline,s,args.gpu,helpers)
                result=helpers.capture(detector,head,image,metas,audit)
                if len(captured)!=1:raise ValueError('Expected exactly one midpoint input capture')
                tensors=captured.pop()
                if set(tensors)!=set(INPUT_NAMES) or any(t.requires_grad or t.grad is not None for t in tensors.values()):raise ValueError('ROI cache contains gradients')
                cache.append(tensors)
                result.update(frame_id=s['frame_id'],image=s.get('image',s['frame_id']),image_sha256=s['sha256'],
                    original_size_hw=list(metas[0]['ori_shape'][:2]),scale_xy=[float(v) for v in metas[0]['scale_factor'][:2]])
                rows.append(result)
                if i==0 or (i+1)%100==0 or i+1==len(sources):print('collect',name,i+1,'/',len(sources),round(time.monotonic()-start,1),flush=True)
            write_rows(folder/'predictions.jsonl',rows)
            torch.save(dict(protocol=VERSION,GT_online=False,input_names=list(INPUT_NAMES),rows=cache),str(folder/'roi.pt'))
            saved=(base.read_rows(args.geometry_test_dir/'test_rows.jsonl') if name=='port_test' else base.read_rows(args.baseline_dir/name/'predictions.jsonl'))
            if name=='port_test':base.indexed_bundle(args.geometry_test_dir,old['reliability']['geometry_files'])
            elif o.sha(args.baseline_dir/name/'predictions.jsonl')!=p['baseline_prediction_sha256'][name]:raise ValueError('Reviewed baseline stream bytes differ')
            proof=replay(rows,saved)
            artifacts[name]=dict(count=len(rows),roi_sha256=o.sha(folder/'roi.pt'),predictions_sha256=o.sha(folder/'predictions.jsonl'),baseline_replay=proof)
            del cache
    finally:handle.remove()
    after=[g.state_digest(detector),g.state_digest(head)]
    g.assert_detector_frozen(detector)
    total=sum(v['count'] for v in artifacts.values())
    if before!=after or counts!=dict(feature_extractions=total,native_head_calls=total*3) or head.training or any(x.grad is not None or x.requires_grad for x in head.parameters()):raise ValueError('Frozen native work/state failed')
    result=dict(protocol=VERSION,status='GT_FREE_ROI_COLLECTION_COMPLETE',runtime=runtime,source_identity=identity,
        state_before=before,state_after=after,counters=counts,artifacts=artifacts,GT_online=False,updates=0)
    o.write_new(out/'completion.json',result);return result


def depth_evaluation(rows,truths,manifest,pin,calibration):
    coverage=base.validate_truth(truths,manifest,pin)
    if len(rows)!=len(truths):raise ValueError('Full paired frame denominator required')
    result=[];params=calibration['parameters'];intr=manifest['camera']['intrinsics'];geom=manifest['obb_reference_geometry']
    for i,(p,t) in enumerate(zip(rows,truths)):
        if p['frame_id']!='frame_%05d'%i:raise ValueError('Depth pairing changed')
        preserve(p['midpoint'],p['size_candidate'])
        gt=o.truth_box(t);oracle=o.depth(gt,intr,geom,params)
        r=dict(frame_id=p['frame_id'],frame_index=i,truth_z_m=t['camera_geometry']['z_cg_opt_m'],theta_total_deg=t['pivot_relative']['theta_total_deg'],gt_box=gt)
        for name,box in [(k,p[k]) for k in METHODS]+[('gt_obb',gt)]:
            dep=o.depth(box,intr,geom,params) if box is not None else dict(z_m=None,q_signed=None,status='missing_output')
            r[name]=dict(box=box,depth=dep,q_in_fit_support=dep['q_signed'] is not None and params['q_signed_min']<=dep['q_signed']<=params['q_signed_max'],geometry=o.geometry_row(box,gt,dep,oracle) if box is not None else None)
        result.append(r)
    changes=[abs(r['size_candidate']['depth']['z_m']-r['truth_z_m'])-abs(r['midpoint']['depth']['z_m']-r['truth_z_m']) for r in result if all(r[k]['depth']['z_m'] is not None for k in METHODS)]
    report=dict(sequence_id=manifest['sequence_id'],evidence_role=pin['evidence_role'],coverage_audit=coverage,
        groups={k:o.summarize(result,k) for k in (*METHODS,'gt_obb')},direct_depth_errors={k:c.direct_errors(result,k) for k in (*METHODS,'gt_obb')},
        offline_error_propagation={k:c.propagation(result,calibration,k) for k in METHODS},
        paired_absolute_error=dict(improved=sum(x<-1e-12 for x in changes),worsened=sum(x>1e-12 for x in changes),tied=sum(abs(x)<=1e-12 for x in changes),delta_m=o.describe(changes)),
        formula_refit=False,reliability_filter=False,GT_online=False,automatic_promotion=False)
    return result,report


def reliability_evaluation(rows,metadata,policy):
    if len(rows)!=len(metadata):raise ValueError('Full reliability denominator required')
    reports={};decisions={}
    for k in METHODS:
        bound=[]
        for p,t in zip(rows,metadata):
            if p['image']!=t['image'] or p['image_sha256']!=t['image_sha256']:raise ValueError('Reliability pair identity differs')
            preserve(p['midpoint'],p['size_candidate'])
            r={f:deepcopy(t[f]) for f in base.flags.FIELDS};r['pred']=deepcopy(p[k]);bound.append(r)
        base.flags.validate_rows(bound,dict(real_seq03=200,real_seq04=668,sim_seq09=572))
        decisions[k],reports[k]=base.flags.score(bound,policy['simple_policy'])
        if any(d['methods']['simple']['center_accepted']!=(r['pred'] is not None) for d,r in zip(decisions[k],bound)):raise ValueError('Reliability erased usable center')
    return decisions,dict(role='original_M_policy_transfer_diagnostic_on_actual_candidate_boxes',
        policy_refit=False,threshold_updates=0,GT_online=False,groups=reports,
        geometry_changes_are_not_discriminator_improvement=True,automatic_promotion=False)


def port_geometry(rows,metadata):
    from crane_project.tools import eval_port_geometry_midpoint_v1_test as evaluator
    summaries={}
    for group in ['all','real','sim',*sorted({r['sequence'] for r in metadata})]:
        summaries[group]={}
        for k in METHODS:
            values=[]
            for p,t in zip(rows,metadata):
                if group not in ('all',t['domain'],t['sequence']):continue
                values.append(dict(t,pred=p[k],metrics=evaluator.g.decompose(np.asarray(t['gt']),np.asarray(p[k][:5]) if p[k] is not None else None)))
            summaries[group][k]=evaluator.summary(values)
    return summaries


def candidate(args,p,old,calibration,identity,out):
    import torch
    from crane_project.utils import port_geometry_size_boundary_continuous_v1 as model
    from crane_project.tools import run_port_geometry_size_boundary_continuous_v1 as trained
    if not torch.cuda.is_available() or int(os.environ.get('WORLD_SIZE','1'))!=1:raise ValueError('Single CUDA inference required')
    torch.cuda.set_device(args.gpu);torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
        gpu=torch.cuda.get_device_name(args.gpu),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),dtype='float32')
    if any(runtime[k]!=v for k,v in p['candidate_runtime'].items()):raise ValueError('Candidate runtime differs from original training')
    if o.sha(args.candidate_head)!=p['candidate_sha256']:raise ValueError('Must load actual fixed final200 size head')
    payload=trained.sealed.torch_load(torch,args.candidate_head);checkpoint_header(payload,p)
    head=model.ContinuousBoundarySizeHead().cuda(args.gpu);head.load_state_dict(payload['head_state'],strict=True);head.eval().requires_grad_(False)
    before=trained.sealed.state_digest(head)
    if before!=p['candidate_digest']:raise ValueError('Actual loaded tensor identity differs')
    neutral=model.ContinuousBoundarySizeHead().cuda(args.gpu).eval().requires_grad_(False)
    collected=json.loads((args.collect_dir/'completion.json').read_text())
    if collected['source_identity']!=identity or collected['status']!='GT_FREE_ROI_COLLECTION_COMPLETE' or collected['state_before']!=collected['state_after']:raise ValueError('Wrong ROI generation')
    result={};proof={};start=time.monotonic()
    # All streams close BEFORE policy scoring / geometry / metric truth reading.
    for name in ['port_test',*old['depth_order']]:
        folder=args.collect_dir/name;pin=collected['artifacts'][name]
        if o.sha(folder/'roi.pt')!=pin['roi_sha256'] or o.sha(folder/'predictions.jsonl')!=pin['predictions_sha256']:raise ValueError('ROI/source stream changed')
        cache=trained.sealed.torch_load(torch,folder/'roi.pt');saved=base.read_rows(folder/'predictions.jsonl')
        if cache['protocol']!=VERSION or cache['GT_online'] is not False or cache['input_names']!=list(INPUT_NAMES) or len(cache['rows'])!=len(saved) or len(saved)!=pin['count']:raise ValueError('Unexpected cache fields/length')
        output=[];fallbacks=0
        for i,(t,r) in enumerate(zip(cache['rows'],saved)):
            if set(t)!=set(INPUT_NAMES) or any(v.requires_grad or v.grad is not None or v.dtype!=torch.float32 or not bool(torch.isfinite(v).all()) for v in t.values()):raise ValueError('Invalid detached GT-free input')
            inputs=[t[k].cuda(args.gpu) for k in INPUT_NAMES]
            with torch.no_grad():
                n=neutral(*inputs);v=head(*inputs)
            if not torch.equal(n['boxes_original'],inputs[-1]):raise ValueError('Neutral does not exactly restore M')
            boxes=v['boxes_original'].cpu().tolist();expected=t['midpoint_original'].tolist()
            if len(boxes)>1 or len(boxes)!=len(expected):raise ValueError('Output count changed')
            m=expected[0] if expected else None;box=boxes[0] if boxes else None
            if m!=r['midpoint']:raise ValueError('Cached M source differs')
            preserve(m,box)
            row=dict(r,size_candidate=box,size_delta_roi=v['delta_roi'].cpu().tolist(),
                size_candidate_unprotected=v['candidate_original'].cpu().tolist(),size_accepted=v['accepted'].cpu().tolist(),
                size_failed_checks=[k for k,x in v['checks'].items() if len(x) and not bool(x[0])])
            fallbacks+=int(bool(row['size_accepted']) and not row['size_accepted'][0]);output.append(row)
        destination=out/name;destination.mkdir();write_rows(destination/'predictions.jsonl',output)
        result[name]=output;proof[name]=dict(frames=len(output),fallbacks=fallbacks,predictions_sha256=o.sha(destination/'predictions.jsonl'),neutral_bitwise_exact=True,center_canonical_angle_score_output_exact=True)
        print('actual trained size head inferred',name,len(output),round(time.monotonic()-start,1),flush=True)
        del cache
    after=trained.sealed.state_digest(head)
    if before!=after or head.training or any(x.grad is not None or x.requires_grad for x in head.parameters()) or o.sha(args.candidate_head)!=p['candidate_sha256']:raise ValueError('Frozen candidate changed')
    native_proof=dict(runtime=runtime,checkpoint_sha256=p['candidate_sha256'],head_state_before=before,head_state_after=after,
        source_identity=identity,collection_completion_sha256=o.sha(args.collect_dir/'completion.json'),GT_online=False,updates=0,streams=proof)
    o.write_new(out/'inference_proof.json',native_proof)
    base.indexed_bundle(args.policy.parent,old['reliability']['policy_files'])
    base.indexed_bundle(args.metadata_dir,old['reliability']['metadata_files'])
    policy=json.loads(args.policy.read_text());metadata=base.read_rows(args.metadata_dir/'test_qualities.jsonl')
    base.indexed_bundle(args.geometry_test_dir,old['reliability']['geometry_files'])
    metadata=base.bind_reliability(metadata,base.read_rows(args.geometry_test_dir/'test_rows.jsonl'))
    decisions,rel=reliability_evaluation(result['port_test'],metadata,policy)
    rel['policy_sha256']=o.sha(args.policy);rel['port_geometry']=port_geometry(result['port_test'],metadata)
    for k in METHODS:write_rows(out/'port_test'/(k+'.decisions.jsonl'),decisions[k])
    o.write_new(out/'port_test/summary.json',rel)
    reports={}
    for name in old['depth_order']:
        manifest,tp,_=base.checked_dataset(args.data_root,name,old['sequences'][name])
        rows,summary=depth_evaluation(result[name],base.read_rows(tp),manifest,old['sequences'][name],calibration)
        write_rows(out/name/'evaluation_rows.jsonl',rows);o.write_new(out/name/'summary.json',summary);reports[name]=summary
    final=dict(protocol=VERSION,status='ACTUAL_SIZE_CANDIDATE_DOWNSTREAM_DIAGNOSTIC_COMPLETE',
        checkpoint_sha256=p['candidate_sha256'],inference_proof=native_proof,source_identity=identity,
        original_TRAIN_gate_passed=False,original_TRAIN_failed_checks=16,diagnostic_user_authorized=True,
        automatic_promotion=False,training_updates=0,policy_refit=False,formula_refit=False,test_used_for_selection=False,
        reliability_summary_sha256=o.sha(out/'port_test/summary.json'),depth_summary_sha256={n:o.sha(out/n/'summary.json') for n in reports})
    o.write_new(out/'completion.json',final);print(final['status'],flush=True)
    return final


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['check','collect','candidate'],required=True)
    parser.add_argument('--out-dir',type=Path,required=True);parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--data-root',type=Path,default=ROOT/'crane_project/data/webots_depth')
    parser.add_argument('--port-root',type=Path,default=ROOT/'crane_project/data/crane_grab_port_day2night_v1/test/images')
    parser.add_argument('--metadata-dir',type=Path,default=ROOT/'work_dirs/port_reliability_branches_v1_test_cached_v1')
    parser.add_argument('--geometry-test-dir',type=Path,default=ROOT/'work_dirs/port_results/geometry/port_geometry_midpoint_sigma15_v1_test_eval')
    parser.add_argument('--baseline-dir',type=Path,default=ROOT/'work_dirs/port_depth_four_arm_v1/frozen_downstream_20261007_163625_1260566')
    parser.add_argument('--policy',type=Path,default=ROOT/'work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1/fit/policy.json')
    parser.add_argument('--candidate-head',type=Path,default=ROOT/'work_dirs/port_geometry_size_boundary_v1/port_geometry_size_boundary_continuous_v1_20261007_154944_1078779/finite/boundary_continuous/experimental_head.pth')
    parser.add_argument('--collect-dir',type=Path)
    args=parser.parse_args();p,old,calibration,identity=checked_sources()
    args.out_dir.mkdir(parents=True,exist_ok=False)
    o.write_new(args.out_dir/'protocol.json',p);o.write_new(args.out_dir/'sources.json',json.loads(SOURCES.read_text()))
    try:
        if args.stage=='check':o.write_new(args.out_dir/'completion.json',dict(protocol=VERSION,status='ACTUAL_CANDIDATE_STATIC_CONTRACT_PASS',source_identity=identity))
        elif args.stage=='collect':collect(args,p,old,identity,args.out_dir)
        else:
            if args.collect_dir is None:raise ValueError('Candidate stage requires sealed --collect-dir')
            candidate(args,p,old,calibration,identity,args.out_dir)
    except Exception as error:
        o.write_new(args.out_dir/'failure.json',dict(error_type=type(error).__name__,error=str(error)));raise

if __name__=='__main__':main()
