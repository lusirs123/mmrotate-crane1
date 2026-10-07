#!/usr/bin/env python3
"""Fixed downstream reporting only: cached sigma15 TEST flags and four-arm Raw-opt.

No training, fitting, threshold/formula/checkpoint selection. Previously exposed
TEST is descriptive. Historical Train03 and sigma1 entry points stay unchanged.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crane_project.utils import port_midpoint_depth_v1 as o
from crane_project.utils import port_depth_four_arm_v1 as c
from crane_project.utils import port_midpoint_sigma15_reliability_v1 as reliability
from crane_project.tools import eval_port_simple_reliability_v1_test as flags
from tools.depth import audit_port_midpoint_depth_v1 as prior
from tools.depth import audit_port_depth_four_arm_v1 as original_four
VERSION = 'port_frozen_downstream_v1_exposed_test_report'
PROTOCOL = ROOT/'tools/port_frozen_downstream_v1_protocol.json'
SOURCES = ROOT/'tools/port_frozen_downstream_v1_sources.json'
ARMS = c.ARMS


def checked_sources():
    contract=json.loads(PROTOCOL.read_text()); indexed=json.loads(SOURCES.read_text())
    if contract['protocol']!=VERSION or indexed['protocol']!=VERSION:
        raise ValueError('Wrong frozen downstream protocol')
    actual={name:o.sha(ROOT/name) for name in indexed['sources']}
    if actual!=indexed['sources'] or indexed['protocol_sha256']!=o.sha(PROTOCOL):
        raise ValueError('Frozen source or protocol bytes differ')
    for name,digest in indexed['parents'].items():
        parent=json.loads((ROOT/name).read_text())
        if o.sha(ROOT/name)!=digest or any(actual.get(p)!=s for p,s in parent['sources'].items()):
            raise ValueError('Historical parent source closure differs')
    if o.sha(ROOT/o.CALIBRATION)!=o.CAL_SHA:
        raise ValueError('Frozen Raw-opt calibration differs')
    calibration=json.loads((ROOT/o.CALIBRATION).read_text())
    if calibration['coordinate_contract']['id']!='raw_opt_v1' or calibration['coordinate_contract']['target']!='z_cg_opt_m':
        raise ValueError('Raw original-camera coordinate contract required')
    return contract,calibration,dict(manifest_sha256=o.sha(SOURCES),sources=actual)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def indexed_bundle(directory, expected):
    directory=Path(directory)
    if (directory/'failure.json').exists():raise ValueError('Failed artifact generation')
    values={name:o.sha(directory/name) for name in expected}
    if values!=expected:raise ValueError('Frozen input bytes differ: '+str(directory))
    return values


def bind_reliability(metadata, native):
    counts={'real_seq03':200,'real_seq04':668,'sim_seq09':572}
    flags.validate_rows(metadata,counts)
    if [r['image'] for r in metadata]!=[r['image'] for r in native]:
        raise ValueError('Cached geometry/metadata identities differ')
    rows=[]
    for saved,r in zip(metadata,native):
        if (any(saved[k]!=r[k] for k in ('sequence','frame_id','domain')) or r['scale']!=1.
                or not np.allclose(saved['gt'],r['gt'],atol=1e-4,rtol=1e-6)
                or r.get('original_b_raw_exact_before_after') is not True):
            raise ValueError('Cached geometry identity/GT/native proof differs')
        b,m=r['b'],r['midpoint']
        if (b is None)!=(m is None) or (b is not None and b[5]!=m[5]):
            raise ValueError('Midpoint changed output presence or score')
        row={k:deepcopy(saved[k]) for k in flags.FIELDS}
        row.update(gt=deepcopy(r['gt']),pred=deepcopy(m))
        eligible=bool(flags.simple.canonical(r['gt'])[2]/flags.simple.canonical(r['gt'])[3]>=1.2)
        if eligible!=saved['angle_axis_well_defined']:raise ValueError('Angle qualification differs')
        rows.append(row)
    return flags.validate_rows(rows,counts)


def score_reliability(rows,policy):
    before=flags.simple.fingerprint([rows,policy])
    api=reliability.Sigma15Reliability(policy,policy['front_end'])
    records,strata=flags.score(rows,policy['simple_policy'])
    for source,record in zip(rows,records):
        for method in ('raw','score_only','simple'):
            delivered=record['methods'][method]
            delivered['final_box_original']=delivered.pop('raw_b_output')
            if delivered!=api.decide(source['pred'],source['image_size'],method):
                raise ValueError('Frozen online API differs from offline evaluation')
            if delivered['center_accepted']!=(source['pred'] is not None):
                raise ValueError('Size/angle rejection erased center output')
    if before!=flags.simple.fingerprint([rows,policy]):raise ValueError('Frozen inputs changed')
    return records,strata


def run_reliability(args,contract,source_identity,out):
    pins=contract['reliability']; policy_dir=args.policy.parent
    proof=dict(policy=indexed_bundle(policy_dir,pins['policy_files']),
        geometry=indexed_bundle(args.geometry_test_dir,pins['geometry_files']),
        metadata=indexed_bundle(args.metadata_dir,pins['metadata_files']))
    policy=json.loads(args.policy.read_text()); complete=json.loads((policy_dir/'completion.json').read_text())
    if (args.policy.name!='policy.json' or complete['status']!='SIGMA15_SIMPLE_FIT_COMPLETE_REVIEW_REQUIRED'
            or policy['contract']!=complete['contract'] or policy['test_read'] is not False
            or complete['test_read'] is not False or any(complete[k] for k in ('detector_updates','midpoint_updates','reference_updates'))):
        raise ValueError('Incomplete or wrong matched frozen policy')
    reliability.checked_frontend(policy['front_end'])
    geo=json.loads((args.geometry_test_dir/'completion.json').read_text())
    if (geo['status']!='FROZEN_SIGMA15_TEST_COMPLETE_REVIEW_REQUIRED' or geo['split']!='test'
            or geo['frames']!=1440 or geo['state_before']!=geo['state_after']
            or geo['fixed_checkpoint']!=policy['front_end']['midpoint_checkpoint']
            or geo['selection']['arm_artifact_sha256']['selection.json']!=policy['front_end']['selection_sha256']
            or geo['head_updates'] or geo['detector_updates'] or geo['selection_on_test']
            or not geo['test_repeatedly_exposed']):
        raise ValueError('Sealed sigma15 TEST generation differs')
    metadata=[{k:r[k] for k in flags.FIELDS} for r in read_rows(args.metadata_dir/'test_qualities.jsonl')]
    native=read_rows(args.geometry_test_dir/'test_rows.jsonl')
    rows=bind_reliability(metadata,native); records,strata=score_reliability(rows,policy)
    report=dict(protocol=VERSION,role='fixed_sigma15_policy_on_exposed_TEST',strata=strata,
        input_proof=proof,source_identity=source_identity,policy_sha256=o.sha(args.policy),
        updates=0,detector_inferences=0,policy_refit=False,threshold_updates=0,
        test_used_for_selection=False,test_repeatedly_exposed=True,GT_online=False)
    with (out/'test_decisions.jsonl').open('x') as stream:
        for r in records:stream.write(json.dumps(r,allow_nan=False)+'\n')
    o.write_new(out/'summary.json',report)
    (out/'summary.md').write_text(flags.table_text('固定σ1.5/ep03＋匹配policy：已暴露TEST',strata)+'\n')
    print('FROZEN_SIGMA15_RELIABILITY_TEST_COMPLETE',flush=True)
    return report


def image_sources(root,name,pin):
    paths=sorted((Path(root)/name/'images').glob('*.jpg'))
    if [p.name for p in paths]!=['frame_%05d.jpg'%i for i in range(pin['count'])]:
        raise ValueError('Full ordered sequence JPEGs required')
    digest=hashlib.sha256();records=[]
    for p in paths:
        s=o.sha(p);digest.update(p.name.encode());digest.update(b'\0');digest.update(bytes.fromhex(s))
        records.append(dict(frame_id=p.stem,image_path=str(p.resolve()),sha256=s))
    if digest.hexdigest()!=pin['image_identity_sha256']:raise ValueError('Frozen image identity differs')
    return records


def checked_dataset(root,name,pin):
    folder=Path(root)/name/'metadata'; mp=folder/'manifest.json'; tp=folder/'frames.jsonl'
    if o.sha(mp)!=pin['manifest_sha256'] or o.sha(tp)!=pin['truth_sha256']:
        raise ValueError('Frozen sequence metadata bytes differ')
    m=json.loads(mp.read_text())
    if m['sequence_id']!=name or m['split']!=pin['split'] or m['run_instance_id']!=pin['run_instance_id']:
        raise ValueError('Sequence role/run identity differs')
    original=json.loads((ROOT/'docs/data/webots_depth_formal_frame_index_v1.manifest.json').read_text())['constant_contract']
    if m['camera']!=original['camera'] or any(m['obb_reference_geometry'][k]!=original['obb_reference_geometry'][k]
            for k in ('long_edge_mean_m','short_edge_mean_m')):
        raise ValueError('Frozen physical top-beam/camera contract differs')
    return m,tp,image_sources(root,name,pin)


def validate_truth(rows,manifest,pin):
    if len(rows)!=pin['count']:raise ValueError('Full truth denominator required')
    bins={k:0 for k in ('theta_0_3_deg','theta_3_5_deg','theta_5_8_deg','theta_8_inf_deg')};depths=[]
    for i,r in enumerate(rows):
        if (r['sequence_id']!=manifest['sequence_id'] or r['run_instance_id']!=manifest['run_instance_id']
                or r['image_file']!='images/frame_%05d.jpg'%i or r['frame_index']!=i
                or not r['obb_valid'] or not r['truth_valid'] or not r['frame_truth_audit']['passed']):
            raise ValueError('Truth validity/order/run audit differs')
        o.truth_box(r);z=r['camera_geometry']['z_cg_opt_m'];theta=r['pivot_relative']['theta_total_deg']
        if not math.isfinite(z) or z<=0 or not math.isfinite(theta) or theta<0:raise ValueError('Invalid metric truth')
        depths.append(z)
        bins['theta_0_3_deg' if theta<3 else 'theta_3_5_deg' if theta<5 else 'theta_5_8_deg' if theta<8 else 'theta_8_inf_deg']+=1
    coverage=dict(frames=len(rows),audit_pass_rate=1.,z_span_m=max(depths)-min(depths),theta_bins=bins)
    gate=manifest.get('collection_design',{}).get('coverage_gate')
    if gate and (len(rows)<gate['min_valid_frames'] or coverage['z_span_m']<gate['min_z_cg_opt_span_m']
            or any(bins[k]<v for k,v in gate['min_theta_bin_counts'].items())):
        raise ValueError('Historical preregistered coverage gate failed')
    return coverage


def evaluate_depth(predictions,truths,manifest,pin,calibration):
    coverage=validate_truth(truths,manifest,pin)
    if set(predictions)!=set(ARMS) or any(len(v)!=len(truths) for v in predictions.values()):
        raise ValueError('Full four-arm paired streams required')
    params=calibration['parameters'];intr=manifest['camera']['intrinsics'];geom=manifest['obb_reference_geometry'];rows=[]
    for i,t in enumerate(truths):
        frame='frame_%05d'%i;gt=o.truth_box(t);oracle=o.depth(gt,intr,geom,params)
        if oracle['status']!='finite_formula':raise ValueError('GT formula diagnostic cannot be computed')
        row=dict(frame_id=frame,frame_index=i,truth_z_m=t['camera_geometry']['z_cg_opt_m'],
            theta_total_deg=t['pivot_relative']['theta_total_deg'],gt_box=gt)
        for arm in ARMS:
            p=predictions[arm][i]
            if p['frame_id']!=frame:raise ValueError('Paired frame order differs')
            box=p['box'];depth=o.depth(box,intr,geom,params) if box is not None else dict(z_m=None,q_signed=None,status='missing_output')
            row[arm]=dict(box=box,depth=depth,q_in_fit_support=depth['q_signed'] is not None and params['q_signed_min']<=depth['q_signed']<=params['q_signed_max'],
                geometry=o.geometry_row(box,gt,depth,oracle) if box is not None else None)
        b,m=row['symeood_b']['box'],row['symeood_b_midpoint']['box']
        if (b is None)!=(m is None) or (b is not None and b[5]!=m[5]):raise ValueError('B/M output presence or score changed')
        row['gt_obb']=dict(box=gt,depth=oracle,q_in_fit_support=params['q_signed_min']<=oracle['q_signed']<=params['q_signed_max'],geometry=o.geometry_row(gt,gt,oracle,oracle))
        rows.append(row)
    groups={a:o.summarize(rows,a) for a in (*ARMS,'gt_obb')}
    summary=dict(protocol=VERSION,sequence_id=manifest['sequence_id'],split=manifest['split'],evidence_role=pin['evidence_role'],
        coordinate_contract='raw_opt_v1',coverage_audit=coverage,groups=groups,
        direct_depth_errors={a:c.direct_errors(rows,a) for a in (*ARMS,'gt_obb')},
        offline_error_propagation={a:c.propagation(rows,calibration,a) for a in ARMS},
        parameter_refit=False,threshold_tuning=False,model_selection=False,reliability_used=False,
        homography_or_truth_attitude_used=False,by_theta={},by_truth_depth={},paired_B_to_midpoint={})
    for name,low,high in [('theta_0_3',0,3),('theta_3_5',3,5),('theta_5_8',5,8),('theta_8_inf',8,math.inf)]:
        selected=[r for r in rows if low<=r['theta_total_deg']<high]
        summary['by_theta'][name]={a:o.summarize(selected,a) if selected else None for a in groups}
    for name,low,high in [('around12m',0,14),('around16m',14,18),('around20m',18,22),('around24m',22,math.inf)]:
        selected=[r for r in rows if low<=r['truth_z_m']<high]
        summary['by_truth_depth'][name]={a:o.summarize(selected,a) if selected else None for a in groups}
    changes=[abs(r['symeood_b_midpoint']['depth']['z_m']-r['truth_z_m'])-abs(r['symeood_b']['depth']['z_m']-r['truth_z_m'])
        for r in rows if r['symeood_b']['depth']['z_m'] is not None and r['symeood_b_midpoint']['depth']['z_m'] is not None]
    summary['paired_B_to_midpoint']=dict(count=len(changes),absolute_error_delta_m=o.describe(changes),
        improved=sum(v<-1e-12 for v in changes),worsened=sum(v>1e-12 for v in changes),tied=sum(abs(v)<=1e-12 for v in changes))
    return rows,summary

def native_b_midpoint(out, images, b_path, head_path, gpu):
    detector, head, pipeline, helpers, runtime = prior.load_native(b_path, head_path, gpu)
    g, torch = helpers.g, helpers.torch
    initial_detector, initial_head = g.state_digest(detector), g.state_digest(head)
    counters = dict(feature_extractions=0, native_head_calls=0)
    def audit(event):
        if event['stage'] == 'test_feature_extracted':
            counters['feature_extractions'] += 1
        elif event['stage'] == 'test_native_head_called':
            counters['native_head_calls'] += 1
        else:
            raise ValueError('Undeclared inference event')
    predictions = []
    start = time.monotonic()
    # A sealed stream is finished before metric truth is parsed/evaluated.
    with (out/'predictions.jsonl').open('x', encoding='utf-8') as stream:
        for index, source in enumerate(images):
            if o.sha(source['image_path']) != source['sha256']:
                raise ValueError('Image bytes changed after preflight')
            image, metas = prior.native_view(pipeline, source, gpu, helpers)
            result = helpers.capture(detector, head, image, metas, audit)
            result.update(frame_id=source['frame_id'], image_sha256=source['sha256'],
                original_size_hw=list(metas[0]['ori_shape'][:2]),
                scale_xy=[float(v) for v in metas[0]['scale_factor'][:2]])
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+'\n')
            stream.flush()
            predictions.append(result)
            if index == 0 or (index+1) % 25 == 0 or index+1 == len(images):
                print('inference %d/%d elapsed %.1fs'%(index+1, len(images), time.monotonic()-start), flush=True)
    final_detector, final_head = g.state_digest(detector), g.state_digest(head)
    if initial_detector != final_detector or initial_head != final_head:
        raise ValueError('Frozen detector/midpoint parameters or buffers changed')
    g.assert_detector_frozen(detector)
    if head.training or any(p.requires_grad or p.grad is not None for p in head.parameters()):
        raise ValueError('Midpoint received gradients or left eval mode')
    if counters != dict(feature_extractions=len(images), native_head_calls=3*len(images)):
        raise ValueError('Incomplete/extra inference work')
    if o.sha(b_path) != o.B_SHA or o.sha(head_path) != o.HEAD_SHA:
        raise ValueError('Weights changed during inference')
    proof = dict(runtime=runtime, detector_state_before=initial_detector,
        detector_state_after=final_detector, midpoint_state_before=initial_head,
        midpoint_state_after=final_head, counters=counters, detector_updates=0,
        midpoint_updates=0, model_metric_truth_read=False, output_count_and_scores_preserved=True,
        wall_seconds=time.monotonic()-start, wall_time_role='diagnostic duration, not deployment FPS')
    del detector, head, pipeline
    torch.cuda.empty_cache()
    return predictions, proof

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
    main_forward=original_four.count_main_forward(detector.bbox_head,counts)
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
                if i==0 or (i+1)%25==0 or i+1==len(images):
                    print('%s inference %d/%d %.1fs'%(arm,i+1,len(images),time.monotonic()-start),flush=True)
    finally:
        hook.remove();detector.bbox_head.forward=main_forward
    final=g.state_digest(detector);g.assert_detector_frozen(detector)
    if initial!=final or counts!=[len(images),len(images)]:raise ValueError('Incomplete native work or changed frozen state')
    if o.sha(weight)!=spec_identity['checkpoint_sha256'] or o.sha(selection)!=spec_identity['selection_sha256']:
        raise ValueError('Fixed input changed during inference')
    proof=dict(runtime=runtime,checkpoint_meta=meta,state_before=initial,state_after=final,
        feature_extractions=counts[0],native_main_head_calls=counts[1],updates=0,
        metric_truth_read=False,native_policy_modified=False,wall_seconds=time.monotonic()-start,
        wall_time_role='diagnostic duration, not deployment FPS')
    o.write_new(out/(arm+'.proof.json'),proof)
    del detector,pipeline;torch.cuda.empty_cache()
    return records,proof


def run_depth(args,contract,calibration,identity,out):
    # Resolve frozen weights/VAL selections before opening the new TEST images.
    def resolve(preferred,filename,digest):
        path=ROOT/preferred
        return o.checked_weight(ROOT,path if path.is_file() else None,filename,digest)
    weights={a:resolve(contract['weight_paths'][a],'epoch_%02d.pth'%c.BASELINES[a]['epoch'],c.BASELINES[a]['checkpoint_sha256']) for a in ('eood','symeood')}
    selections={a:resolve(contract['selection_paths'][a],'sweep_results.json',c.BASELINES[a]['selection_sha256']) for a in weights}
    for a in weights:c.selection_contract(ROOT,a,json.loads(selections[a].read_text()))
    bp=resolve(contract['weight_paths']['symeood_b'],'epoch_24.pth',o.B_SHA)
    hp=resolve(contract['weight_paths']['symeood_b_midpoint'],'head_epoch_03.pth',o.HEAD_SHA)
    for name in contract['depth_order']:
        pin=contract['sequences'][name];directory=out/name;directory.mkdir()
        manifest,tp,images=checked_dataset(args.data_root,name,pin)
        o.write_new(directory/'image_sources.json',images)
        paired,proof=native_b_midpoint(directory,images,bp,hp,args.gpu)
        o.write_new(directory/'b_midpoint.proof.json',proof)
        predictions={a:[dict(frame_id=p['frame_id'],box=p[k]) for p in paired] for a,k in [('symeood_b','b'),('symeood_b_midpoint','midpoint')]}
        for a in ('eood','symeood'):
            predictions[a],_=native_baseline(a,images,weights[a],selections[a],directory,args.gpu,proof['runtime'])
        # Every arm stream is closed. Parse metric truth only after this point.
        rows,summary=evaluate_depth(predictions,read_rows(tp),manifest,pin,calibration)
        if o.sha(tp)!=pin['truth_sha256']:raise ValueError('Truth bytes changed during evaluation')
        with (directory/'evaluation_rows.jsonl').open('x') as stream:
            for r in rows:stream.write(json.dumps(r,allow_nan=False)+'\n')
        o.write_new(directory/'summary.json',summary)
        o.write_new(directory/'completion.json',dict(status='FROZEN_FOUR_ARM_DEPTH_SEQUENCE_COMPLETE',sequence=name,
            source_identity=identity,calibration_sha256=o.CAL_SHA,dataset=pin,weights={a:o.sha(p) for a,p in weights.items()},
            b_sha256=o.B_SHA,midpoint_sha256=o.HEAD_SHA,metric_truth_read_after_all_prediction_streams_closed=True,updates=0))
        print('FROZEN_FOUR_ARM_DEPTH_SEQUENCE_COMPLETE',name,flush=True)
        for a,s in summary['groups'].items():
            m=s['depth_metrics'];d=summary['direct_depth_errors'][a]
            print(a,'numeric',s['numeric_depth_count'],'/',s['frame_count'],'MAE',m['mae_m'],'RMSE',m['rmse_m'],'P95',m['p95_abs_error_m'],'within1m',d['abs_error_coverage']['1.0']['count'],flush=True)


def publish(out):
    o.write_new(out/'artifacts.json',dict(protocol=VERSION,files={str(p.relative_to(out)):o.sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('check','reliability','depth'),required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--policy',type=Path,default=ROOT/'work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1/fit/policy.json')
    p.add_argument('--geometry-test-dir',type=Path,default=ROOT/'work_dirs/port_geometry_midpoint_sigma15_v1_test_eval')
    p.add_argument('--metadata-dir',type=Path,default=ROOT/'work_dirs/port_reliability_branches_v1_test_cache_score')
    p.add_argument('--data-root',type=Path,default=ROOT/'crane_project/data/webots_depth')
    p.add_argument('--gpu',type=int,default=0)
    args=p.parse_args();contract,calibration,identity=checked_sources();out=args.out_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    o.write_new(out/'protocol.json',contract);o.write_new(out/'sources.json',identity)
    if args.stage=='reliability':run_reliability(args,contract,identity,out)
    elif args.stage=='depth':run_depth(args,contract,calibration,identity,out)
    if checked_sources()[2]!=identity:raise ValueError('Sources changed during evaluation')
    o.write_new(out/'completion.json',dict(status='FROZEN_DOWNSTREAM_'+args.stage.upper()+'_COMPLETE',protocol=VERSION,
        updates=0,policy_refit=False,threshold_updates=0,formula_refit=False,test_used_for_selection=False))
    publish(out)


if __name__=='__main__':main()
