#!/usr/bin/env python3
"""Frozen M -> continuous opposite-border mean -> TRAIN-gated finite200 candidate.

No detector construction, feature extraction, TEST, policy or depth fitting.
Reuse fixed TRAIN/VAL ROIs, roles/schedule and old delivery protections.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import math
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.tools import preflight_port_geometry_midpoint_edge_residual_v1 as old
from crane_project.tools import diagnose_port_geometry_size_support_v1 as support
from crane_project.utils import port_geometry_size_contrast_v1 as design
from crane_project.tools import run_port_geometry_size_contrast_v1 as parent

sealed=old.sealed
VERSION='port_geometry_size_boundary_continuous_v1'
ARMS=('boundary_continuous',)
MODEL_SETTINGS=dict(seed=1703, in_channels=256, hidden_channels=8, roi_size=9,
    bins=9, log_range=.5, tangent_samples=9, prior_sigma_bins=1., decoded_beta=.1)
BASE='crane_project/tools/'+VERSION
PROTOCOL=ROOT/(BASE+'_protocol.json')
SOURCES=ROOT/(BASE+'_sources.json')
NEW={BASE+'_protocol.json', 'crane_project/utils/'+VERSION+'.py',
     'crane_project/tools/run_'+VERSION+'.py', 'crane_project/tools/run_'+VERSION+'.sh',
     'tests/test_'+VERSION+'.py',
     'crane_project/tools/port_geometry_size_contrast_v1_sources.json',
     'crane_project/utils/port_geometry_size_boundary_v1.py'}


def protocol_document():
    return dict(protocol=VERSION, stages=['check','prepare','smoke','finite'], settings=design.SETTINGS,
        model_settings=MODEL_SETTINGS, arms=list(ARMS), parameter_count=2760, fixed_checkpoint=sealed.FIXED,
        cache_manifest_sha256=sealed.CACHE_SHA, collection_artifacts=support.INPUT_FILES,
        sampling='TRAIN only. Per-video last floor(n/5) diagnostic probe, preceding10 indices purged; '
            'all legacy64 excluded from gradients and reported separately. Roles use identity order, never GT errors. '
            'Equal real/sim and standard/half slots; round-robin videos; alternate-step first slot '
            'prioritizes error strata within same video, global per-view error repeat cap4; others prefer correct frames.',
        objective='Mean SmoothL1 of actual continuous decoded two-axis log-size residuals, beta0.1. '
            'No auxiliary CE, GT-bucket fine loss, ratio, DFL, depth or quality loss. No clipping of GT.',
        readout='Identical hard-boundary 2760 parameters, feature extraction and initialization. '
            'Nine log-extent buckets, tangent support-weighted max pooling, Gaussian prior sigma1. '
            'delta=sum(softmax(logits)*(bucket+fine))-sum(softmax(initial_prior)*bucket), '
            'same expanded-shape arithmetic to center exact neutral output. Same formula train/inference. '
            'Mode only diagnostic. Direct gradients to probability and fine. Existing delivery guards.',
        comparison='Same roles/schedule/Adam/seed and independent initialization as hard boundary. '
            'Smoke2 discarded; finite200 final only. Readout and corresponding supervision change together. '
            'No VAL tensors before TRAIN gate on delivered output. Per-scale/domain fit and probe '
            'long/short MAE/P95, joint10, abs short log-bias and abs log-ratio MAE/P95 protected '
            'with continuous tolerance1e-6; standard fit pair-average relative MAE strictly improves '
            'by more than1e-6 in both domains. Fail TRAIN gate -> retain M, no VAL. '
            'Pass -> full frozen VAL neutral replay and existing geometry/probe/bias/ratio gates. '
            'No checkpoint selection, retries, extra steps, threshold/weight sweeps or promotion.',
        scope=dict(detector_updates=0,frozen_midpoint_updates=0,feature_extractions=0,
            test_access=False,policy_modified=False,depth_formula_modified=False,
            dataset_splits_modified=False,labels_modified=False,automatic_promotion=False,
            formal_training=False),
        limitations=['TRAIN probe is exposed to original detector/M training, not independent generalization.',
            'Existing common-underscale support is scarce; no new images or labels are created.',
            'Physical reference remains unconfirmed; no real metric-depth guarantee.',
            'This changes continuous readout and its corresponding supervision together, not isolated causality.',
            'GT OBB borders are not necessarily visible physical contours; fixed-k width remains a limitation.',
            'Probability means may average wrong candidates; accurate mean does not calibrate probabilities.',
            'CUDA grid_sample backward may be nondeterministic despite fixed seeds and cudnn settings.',
            'Reliability/depth benefits require separate matched downstream verification.'])


def checked_sources():
    parent_identity=parent.checked_sources()
    manifest=sealed.read(SOURCES)
    required=set(parent_identity['sources'])|NEW
    if (manifest['protocol']!=VERSION or set(manifest['sources'])!=required or
            sealed.read(PROTOCOL)!=protocol_document()):
        raise ValueError('New contrast source/protocol scope differs')
    for name,digest in manifest['sources'].items():
        if sealed.sha(ROOT/name)!=digest: raise ValueError('Contrast source SHA differs: '+name)
    return dict(protocol_sha256=sealed.sha(PROTOCOL),sources_sha256=sealed.sha(SOURCES),
        sources=manifest['sources'],parent_identity=parent_identity)


def resolve_collection(explicit=None,root=ROOT):
    if explicit: return Path(explicit).resolve()
    default=root/support.DEFAULT_COLLECTION
    if (default/'collect/completion.json').is_file(): return default.resolve()
    expected=support.INPUT_FILES['collect/completion.json']
    found={p.parent.parent.resolve() for p in (root/'work_dirs').rglob('completion.json')
           if p.parent.name=='collect' and sealed.sha(p)==expected}
    if len(found)!=1:
        raise ValueError('Locate reviewed collect via work_dirs/port_results/INDEX.md and pass '
                         '--collection-dir or --collection-archive; matching directories: '+str(sorted(map(str,found))))
    return found.pop()


def resolve_head(explicit=None,root=ROOT):
    if explicit: return Path(explicit).resolve()
    default=root/sealed.DEFAULT_HEAD
    if (default/sealed.FIXED['path']).is_file(): return default.resolve()
    found={p.parent.resolve() for p in (root/'work_dirs').rglob(sealed.FIXED['path'])
           if sealed.sha(p)==sealed.FIXED['sha256']}
    if len(found)!=1:
        raise ValueError('Locate fixed sigma1.5/ep03 via work_dirs/port_results/INDEX.md and pass '
                         '--head-dir; matching directories: '+str(sorted(map(str,found))))
    return found.pop()


def current_plan(args):
    inputs=argparse.Namespace(**vars(args))
    if not inputs.collection_archive: inputs.collection_dir=resolve_collection(inputs.collection_dir)
    return parent.current_plan(inputs)


def checked_plan(path,plan,proof,identity):
    if not path: raise ValueError('smoke/finite require prepared --plan-file')
    path=Path(path).resolve()
    index=sealed.read(path.parent/'artifacts.json')
    completion=sealed.read(path.parent/'completion.json')
    if (index['protocol']!=VERSION or completion['status']!='SIZE_CONTINUOUS_PREPARED' or
        sealed.read(path)!=plan or sealed.read(path.parent/'input_identity.json')!=proof or
        sealed.read(path.parent/'source_identity.json')!=identity or
        sealed.read(path.parent/'protocol.json')!=protocol_document()):
        raise ValueError('Prepared TRAIN support/source differs')
    for name in ('plan.json','completion.json','input_identity.json','source_identity.json','protocol.json'):
        if index['files'].get(name)!=sealed.sha(path.parent/name):
            raise ValueError('Prepared artifact SHA differs: '+name)


def publish(directory,completion):
    sealed.write(directory/'completion.json',completion)
    sealed.write(directory/'artifacts.json',dict(protocol=VERSION,files={
        p.name:sealed.sha(p) for p in directory.iterdir() if p.is_file() and p.name!='artifacts.json'}))


def checked_output_directory(path,root=ROOT):
    directory=Path(path).resolve()
    parent_dir=(root/'work_dirs'/'port_geometry_size_boundary_v1').resolve()
    if (directory.parent.parent!=parent_dir or directory.name not in ('check','prepare','smoke','finite') or
            not directory.parent.name.startswith(VERSION+'_')):
        raise ValueError('Use shared work_dirs/'+VERSION+'/RUN_NAME/{check,prepare,smoke,finite}')
    if directory.exists(): raise FileExistsError(str(directory))
    return directory


def load_records(cache_dir,cache,plan,torch,original,include_val,include_train=True):
    indexed={r['image']:r for r in plan['records']}
    train,val=[],[]
    bytes_read=0
    labels=('train_s1','train_s05') if include_train else ()
    for label in labels+(('val_s1',) if include_val else ()):
        path=cache_dir/(label+'.pt')
        if sealed.sha(path)!=cache['files'][path.name]: raise ValueError('Sealed ROI shard changed: '+label)
        payload=sealed.torch_load(torch,path)
        expected,role,_=sealed.SHARDS[label]
        rows=payload['records']
        if (payload['identity']!=cache['identity'] or payload['detector_state']!=cache['detector_state'] or
            len(rows)!=expected or len({r['image'] for r in rows})!=expected or
            Counter(r['sequence'] for r in rows)!=Counter(sealed.TRAIN_COUNTS if role=='train' else sealed.VAL_COUNTS)):
            raise ValueError('Cached role/count/state differs: '+label)
        for r in rows:
            bytes_read+=sealed.checked_record(r,label,torch,original)
            if bool((r['gt_original'][:,2:4]<=0).any()): raise ValueError('Nonpositive cached GT')
            if role=='train':
                entry=indexed[r['image']]
                if any(r[k]!=entry[k] for k in ('domain','sequence','split','frame_id')):
                    raise ValueError('New plan/cache TRAIN identity differs')
                train.append(dict(r,shard=label,sample_role=entry['sample_role'],stratum=entry['stratum']))
            else: val.append(dict(r,shard=label,sample_role='val'))
        if bytes_read>old.SETTINGS['max_cpu_tensor_bytes']: raise ValueError('CPU tensor budget exceeded')
    train.sort(key=lambda r:(r['scale'],r['image']))
    return train,sorted(val,key=lambda r:(r['sequence'],r['frame_id'])),bytes_read


def verify_train_replay(records,plan):
    indexed={r['image']:r for r in plan['records']}
    for r in records:
        if r['scale']!=1.: continue
        entry=indexed[r['image']]
        for value,key in ((r['gt_original'][0].tolist(),'standard_gt'),
            (r['boxes_original'][0].tolist() if len(r['boxes_original']) else None,'standard_b'),
            (r['midpoint_original'][0].tolist() if len(r['midpoint_original']) else None,'standard_m')):
            expected=entry[key]
            if (value is None)!=(expected is None) or (value is not None and
                (len(value)!=len(expected) or any(not math.isclose(a,b,rel_tol=1e-6,abs_tol=1e-4)
                                                for a,b in zip(value,expected)))):
                raise ValueError('TRAIN support numeric replay differs: '+r['image']+' '+key)
            if key!='standard_gt' and value is not None and value[5]!=expected[5]:
                raise ValueError('TRAIN score identity differs')


def update(head,optimizer,frozen,records,indices,device,torch,arm):
    if arm not in ARMS: raise ValueError('Only the fixed boundary candidate is allowed')
    from crane_project.utils import port_geometry_size_boundary_continuous_v1 as model
    owned=[id(p) for group in optimizer.param_groups for p in group['params']]
    if len(owned)!=len(set(owned)) or set(owned)!={id(p) for p in head.parameters()}:
        raise ValueError('Optimizer may own only new independent size head')
    roi,mask,b,bm,xy,m,gt=old.batch(records,indices,device,torch)
    optimizer.zero_grad()
    output=head(roi,mask,b,bm,xy,m)
    terms=model.continuous_loss(output,gt,m,bm)
    if not bool(torch.isfinite(terms['total'])): raise ValueError('Nonfinite decoded size loss')
    params=list(head.parameters())
    logit_grad,fine_grad=torch.autograd.grad(terms['total'],
        (output['bucket_logits'],output['fine_log']),retain_graph=True)
    gradient_norm=lambda x:math.sqrt(float(x.detach().double().square().sum()))
    if not bool(torch.isfinite(logit_grad).all() and torch.isfinite(fine_grad).all()):
        raise ValueError('Nonfinite decoded branch gradients')
    term_norms=dict(decoded_logit_grad_norm=gradient_norm(logit_grad),
                   decoded_fine_grad_norm=gradient_norm(fine_grad))
    terms['total'].backward()
    if (any(p.requires_grad or p.grad is not None for p in frozen.parameters()) or
        any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params)):
        raise ValueError('Frozen gradient or missing/nonfinite new gradient')
    norm=lambda values:math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in values))
    stem=norm(head.stem.parameters()); terminal=norm(list(head.u.parameters())+list(head.v.parameters()))
    before=float(torch.nn.utils.clip_grad_norm_(params,design.SETTINGS['clip_norm']))
    after=norm(params)
    optimizer.step()
    event=dict(loss=float(terms['total'].detach()),decoded_loss=float(terms['decoded'].detach()),
        target_outside_grid=int(terms['target']['outside_grid'].sum()),
        decoded_log_mae=float((output['delta_roi']-terms['target']['log_residual']).abs().mean().detach()),
        stem_grad_norm=stem,terminal_grad_norm=terminal,total_grad_norm_before_clip=before,
        total_grad_norm_after_clip=after,clipped=before>design.SETTINGS['clip_norm'],**term_norms)
    if any(not math.isfinite(v) for v in event.values()) or any(not bool(torch.isfinite(p).all()) for p in params):
        raise ValueError('Nonfinite gradient/update')
    return event


def save_reload(directory,stage,arm,head,optimizer,replay,proof,device,torch,model):
    path=directory/'experimental_head.pth'
    before=evaluate_rows(head,replay,device,torch)
    payload=dict(protocol=VERSION,stage=stage,arm=arm,
        updates=design.SETTINGS['smoke_steps' if stage=='smoke' else 'steps'],proof=proof,
        head_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()},
        head_digest=sealed.state_digest(head),optimizer_state=optimizer.state_dict(),
        rng=old.rng_state(torch),experimental_only=True,automatic_promotion=False)
    with path.open('xb') as stream: torch.save(payload,stream)
    restored=sealed.torch_load(torch,path)
    if not old.tree_equal(payload,restored,torch): raise ValueError('Checkpoint serialization differs')
    reloaded=model.ContinuousBoundarySizeHead().to(device)
    reloaded.load_state_dict(restored['head_state'],strict=True)
    opt=torch.optim.Adam(reloaded.parameters(),lr=design.SETTINGS['lr'],weight_decay=0.)
    opt.load_state_dict(restored['optimizer_state']); old.restore_rng(restored['rng'],torch)
    if (sealed.state_digest(reloaded)!=payload['head_digest'] or
        not old.tree_equal(opt.state_dict(),optimizer.state_dict(),torch) or
        not old.tree_equal(restored['rng'],old.rng_state(torch),torch) or
        before!=evaluate_rows(reloaded,replay,device,torch)):
        raise ValueError('GPU prediction/optimizer/RNG replay differs')
    return reloaded,dict(path=path.name,sha256=sealed.sha(path),save_reload_exact=True,
                        head_digest=payload['head_digest'],updates=payload['updates'])


def checked_smoke(directory,identity,proof):
    if not directory: raise ValueError('finite requires successful --smoke-dir')
    directory=Path(directory).resolve()
    completion=sealed.read(directory/'completion.json')
    if (completion['protocol']!=VERSION or completion['status']!='SIZE_CONTINUOUS_SMOKE_COMPLETE' or
        completion['source_identity']!=identity or completion['proof']!=proof or
        completion['val_tensor_access'] is not False or completion['arms']!=list(ARMS)):
        raise ValueError('Matching no-VAL smoke required')
    index=sealed.read(directory/'artifacts.json')
    if index['protocol']!=VERSION or index['files'].get('completion.json')!=sealed.sha(directory/'completion.json'):
        raise ValueError('Smoke completion SHA differs')
    results={}
    for arm in ARMS:
        path=directory/arm
        report=sealed.read(path/'completion.json')
        artifacts=sealed.read(path/'artifacts.json')
        if artifacts['protocol']!=VERSION: raise ValueError('Smoke arm protocol differs')
        for name in ('completion.json','experimental_head.pth'):
            if artifacts['files'].get(name)!=sealed.sha(path/name): raise ValueError('Smoke arm artifact differs')
        if (report['protocol']!=VERSION or report['stage']!='smoke' or report['arm']!=arm or
            report['proof']!=proof or report['updates']!=2 or
            not old.metrics.ENGINEERING_REQUIRED <= set(report['engineering']) or
            not all(v is True for v in report['engineering'].values())):
            raise ValueError('Smoke arm engineering failed')
        results[arm]=report
    return results


def evaluate_rows(head,records,device,torch):
    # Preserve the proven historical evaluation schema; edge_residual is a
    # compatibility field name, explicitly mapped to this new candidate.
    rows=old.evaluate_rows(head,records,device,torch)
    from crane_project.utils import port_geometry_size_boundary_continuous_v1 as model
    with torch.no_grad():
        for r,row in zip(records,rows):
            inputs=[r[k].to(device) for k in sealed.TENSOR_KEYS[:-1]+('midpoint_original',)]
            output=head(*inputs)
            if not len(inputs[-1]):
                row['boundary']=None
                continue
            target=model.boundary_targets(r['gt_original'].to(device),inputs[-1],inputs[3])
            selected_support=output['border_support'].gather(2,
                target['bucket_index'][:,:,None,None].expand(-1,-1,1,2)).squeeze(2)
            row['boundary']=dict(predicted_bucket=output['bucket_index'][0].cpu().tolist(),
                target_bucket=target['bucket_index'][0].cpu().tolist(),
                target_outside_grid=target['outside_grid'][0].cpu().tolist(),
                target_fine_log=target['fine_log'][0].cpu().tolist(),
                predicted_fine_log=output['fine_log'][0].gather(-1,
                    output['bucket_index'][0,:,None]).squeeze(-1).cpu().tolist(),
                fine_log_all=output['fine_log'][0].cpu().tolist(),
                expected_bucket_log=output['expected_bucket_log'][0].cpu().tolist(),
                expected_fine_log=output['expected_fine_log'][0].cpu().tolist(),
                initial_expected_log=output['initial_expected_log'][0].cpu().tolist(),
                readout='continuous_probability_mean_mode_diagnostic_only',
                target_border_support=selected_support[0].cpu().tolist(),
                bucket_probability=output['bucket_probability'][0].cpu().tolist(),
                border_support=output['border_support'][0].cpu().tolist())
    return rows


def evaluate(rows):
    summary=old.metrics.evaluate(rows)
    for shard,subsets in summary.items():
        base=[r for r in rows if r['shard']==shard]
        for role,groups in subsets.items():
            subset=base if role=='all' else [r for r in base if
                r['sample_role']==role.split('_')[0] and (not role.endswith('_eligible') or r['eligible_train'])]
            for name,group in groups.items():
                selected=[r for r in subset if name=='overall' or r['domain']==name or r['sequence']==name]
                for method in old.metrics.METHODS:
                    values=[r['metrics'][method] for r in selected if r['metrics'][method]['output']]
                    ratio=[abs(v['long_signed_log_ratio']-v['short_signed_log_ratio']) for v in values]
                    group[method]['continuous_sizes']=dict(log_ratio_mae=sum(ratio)/len(ratio) if ratio else None,
                        log_ratio_p95=old.metrics.percentile(ratio),
                        short_relative_bias=sum(old.metrics.canonical(r[method])[3]/
                            old.metrics.canonical(r['gt'])[3]-1. for r in selected if r[method] is not None)/len(values)
                            if values else None)
    return summary


def boundary_gates(summary,engineering):
    gate=parent.extra_probe_gates(summary,old.metrics.finite_gates(summary,engineering))
    def down(a,b):
        return (a is None and b is None) or (a is not None and b is not None and
            math.isfinite(a) and math.isfinite(b) and b<=a+1e-6)
    for name,group in summary['val_s1']['all'].items():
        if name=='overall': continue
        a,b=(group[method] for method in old.metrics.METHODS)
        prefix='val/'+name+'/'
        x,y=a['edges']['short']['signed_log_mean'],b['edges']['short']['signed_log_mean']
        gate['checks'][prefix+'abs_short_log_bias']=down(None if x is None else abs(x),None if y is None else abs(y))
        for key in ('log_ratio_mae','log_ratio_p95'):
            gate['checks'][prefix+key]=down(a['continuous_sizes'][key],b['continuous_sizes'][key])
        if name in ('real','sim'):
            x,y=a['edges']['short']['mae'],b['edges']['short']['mae']
            gate['checks'][prefix+'short_mae_strict_gain']=x is not None and y is not None and y<x-1e-6
    gate['failed_checks']=[k for k,v in gate['checks'].items() if not v]
    gate['finite_joint_gate_pass']=not gate['failed_checks']
    gate['status']='CONTINUOUS_FINITE_GATE_PASS_REVIEW_REQUIRED' if not gate['failed_checks'] else 'CONTINUOUS_FINITE_GATE_FAIL_KEEP_M'
    return gate


def train_gates(summary):
    checks={}
    def down(a,b):
        return (a is None and b is None) or (a is not None and b is not None and
            math.isfinite(a) and math.isfinite(b) and b<=a+1e-6)
    for shard in ('train_s1','train_s05'):
        for role in ('fit','probe'):
            for domain in ('real','sim'):
                group=summary[shard][role][domain]
                a,b=(group[m] for m in old.metrics.METHODS)
                prefix=shard+'/'+role+'/'+domain+'/'
                checks[prefix+'frozen_identity']=group['paired']['frozen_identity_preserved'] is True
                checks[prefix+'frame_output_identity']=(a['frames']==b['frames'] and
                    a['output_coverage']==b['output_coverage'])
                checks[prefix+'joint_size10']=b['joint_size10_full_frame']['numerator']>=a['joint_size10_full_frame']['numerator']
                for edge in ('long','short'):
                    for metric in ('mae','p95'):
                        checks[prefix+edge+'_'+metric]=down(a['edges'][edge][metric],b['edges'][edge][metric])
                x,y=a['edges']['short']['signed_log_mean'],b['edges']['short']['signed_log_mean']
                checks[prefix+'abs_short_log_bias']=down(None if x is None else abs(x),None if y is None else abs(y))
                for key in ('log_ratio_mae','log_ratio_p95'):
                    checks[prefix+key]=down(a['continuous_sizes'][key],b['continuous_sizes'][key])
                if shard=='train_s1' and role=='fit':
                    x,y=([m['edges'][e]['mae'] for e in ('long','short')] for m in (a,b))
                    checks[prefix+'pair_mae_strict_gain']=(all(v is not None and math.isfinite(v) for v in x+y)
                        and sum(y)/2<sum(x)/2-1e-6)
    failed=[k for k,v in checks.items() if not v]
    return dict(checks=checks,failed_checks=failed,train_gate_pass=not failed,
        status='CONTINUOUS_TRAIN_GATE_PASS' if not failed else 'CONTINUOUS_TRAIN_GATE_FAIL_KEEP_M',
        val_tensor_access=False,checkpoint_selection=False)


def candidate_summary(rows):
    # Same continuous summary schema, but raw candidates bypass delivery guards.
    # No performance gate uses this diagnostic in place of delivered boxes.
    candidates=[dict(r,edge_residual=r['candidate']) for r in rows]
    return evaluate(candidates)


def load_val_after_gate(gate,cache_dir,cache,plan,torch,original):
    if gate.get('train_gate_pass') is not True:
        raise ValueError('TRAIN delivered-output gate must pass before VAL tensor load')
    _,val,bytes_read=load_records(cache_dir,cache,plan,torch,original,True,include_train=False)
    return val,bytes_read


def run(args):
    identity=checked_sources()
    directory=checked_output_directory(args.out_dir)
    # Avoid nesting new output in frozen/cache/checkpoint/source inputs.
    protected=[ROOT/'crane_project',ROOT/'tests',ROOT/'docs']
    if args.head_dir: protected.append(Path(args.head_dir).resolve())
    if args.cache_dir: protected.append(Path(args.cache_dir).resolve())
    if args.collection_dir: protected.append(Path(args.collection_dir).resolve())
    if args.plan_file: protected.append(Path(args.plan_file).resolve().parent)
    if args.smoke_dir: protected.append(Path(args.smoke_dir).resolve())
    if any(p==directory or p in directory.parents for p in protected):
        raise ValueError('Output must be separate from all fixed/prepared inputs')
    directory.mkdir(parents=True,exist_ok=False)
    completion=dict(protocol=VERSION,stage=args.stage,status='SIZE_CONTINUOUS_FAILED',
        started_utc=datetime.now(timezone.utc).isoformat(),source_identity=identity,
        val_tensor_access=False,arms=list(ARMS),**protocol_document()['scope'])
    started=time.monotonic()
    try:
        sealed.write(directory/'protocol.json',protocol_document())
        sealed.write(directory/'source_identity.json',identity)
        if args.stage=='check':
            completion.update(status='SIZE_CONTINUOUS_STATIC_CONTRACT_PASS',model_loaded=False,parameter_updates=0)
            publish(directory,completion)
            print(completion['status'],flush=True)
            return
        plan,input_proof=current_plan(args)
        sealed.write(directory/'input_identity.json',input_proof)
        if args.stage=='prepare':
            sealed.write(directory/'plan.json',plan)
            completion.update(status='SIZE_CONTINUOUS_PREPARED',role_counts=plan['role_counts'],
                              model_loaded=False,parameter_updates=0)
            publish(directory,completion)
            print(completion['status'],plan['role_counts'],flush=True)
            return
        checked_plan(args.plan_file,plan,input_proof,identity)
        cache_dir=sealed.resolve_cache(args.cache_dir)
        cache=sealed.checked_cache(cache_dir,identity['parent_identity']['parent_identity']['parent_identity'])
        head_dir=resolve_head(args.head_dir)
        selection,references,indexed=sealed.checked_selection(head_dir,identity['parent_identity']['parent_identity']['parent_identity'])
        import torch
        from crane_project.utils import port_geometry_midpoint_sigma_v1 as frozen_model
        from crane_project.utils import port_geometry_size_boundary_continuous_v1 as model
        if model.SETTINGS!=MODEL_SETTINGS or model.PARAMETER_COUNT!=2760:
            raise ValueError('Boundary size model contract changed')
        device=torch.device(args.device)
        if os.environ.get('WORLD_SIZE','1')!='1' or device.type!='cuda' or not torch.cuda.is_available():
            raise ValueError('Existing single CUDA device required')
        torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
        old.seed_all(torch)
        runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
            device=str(device),gpu=torch.cuda.get_device_name(device),dtype='torch.float32',
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            tf32_matmul=getattr(torch.backends.cuda.matmul,'allow_tf32',None),
            tf32_cudnn=getattr(torch.backends.cudnn,'allow_tf32',None))
        proof=dict(source_identity=identity,plan_sha256=sealed.sha(args.plan_file),
            collection_artifacts=support.INPUT_FILES,cache_manifest_sha256=sealed.CACHE_SHA,
            detector_state=cache['detector_state'],runtime=runtime,fixed_checkpoint=sealed.FIXED,
            selected_artifacts=indexed)
        smoke=checked_smoke(args.smoke_dir,identity,proof) if args.stage=='finite' else None
        frozen=sealed.load_head(torch,frozen_model,head_dir,selection,cache,device)
        frozen_before=sealed.state_digest(frozen)
        train,val,bytes_read=load_records(cache_dir,cache,plan,torch,frozen_model.original,False)
        batches=design.schedule(train)
        schedule=design.schedule_report(train,batches)
        schedule['batches']=[[dict(image=train[i]['image'],scale=train[i]['scale']) for i in b] for b in batches]
        sealed.write(directory/'schedule.json',schedule)
        completion.update(proof=proof,plan_sha256=proof['plan_sha256'],input_tensor_bytes=bytes_read,
                          val_tensor_access=bool(val),schedule_sha256=sealed.sha(directory/'schedule.json'))
        if args.stage=='finite' and completion['schedule_sha256']!=sealed.sha(Path(args.smoke_dir)/'schedule.json'):
            raise ValueError('Smoke/finite schedule must be identical before updates')
        if args.stage=='smoke':
            selected=sorted(set(i for b in batches[:2] for i in b))
            remap={i:n for n,i in enumerate(selected)}
            train=[train[i] for i in selected]; batches=[[remap[i] for i in b] for b in batches[:2]]
        train=old.prepare_frozen(train,frozen,device,torch,frozen_model.original,model,references)
        verify_train_replay(train,plan)
        val=old.prepare_frozen(val,frozen,device,torch,frozen_model.original,model,references)
        records=train+val
        results={}; baseline_rows=None; initial_digest=None; arm_summaries={}; train_gate=None
        for arm in ARMS:
            arm_dir=directory/arm; arm_dir.mkdir()
            old.seed_all(torch)
            head=model.ContinuousBoundarySizeHead().to(device)
            optimizer=torch.optim.Adam(head.parameters(),lr=design.SETTINGS['lr'],weight_decay=0.)
            initial=sealed.state_digest(head)
            if initial_digest is None: initial_digest=initial
            if initial!=initial_digest or (smoke and (initial!=smoke[arm]['initial_digest'] or
                initial==smoke[arm]['final_digest'])):
                raise ValueError('Arms must share neutral init and discard smoke state')
            if baseline_rows is None:
                baseline_rows=evaluate_rows(head,records,device,torch)
                if any(r['midpoint']!=r['edge_residual'] for r in baseline_rows):
                    raise ValueError('Neutral head changed M')
                old.write_rows(directory/'baseline_rows.jsonl',baseline_rows)
                if args.stage=='finite':
                    baseline_train_summary=evaluate(baseline_rows)
                    sealed.write(directory/'baseline_train_summary.json',baseline_train_summary)
            gradients=[]; count=design.SETTINGS['smoke_steps' if args.stage=='smoke' else 'steps']
            head.train()
            with (arm_dir/'progress.jsonl').open('x',encoding='utf-8') as stream:
                for step,indices in enumerate(batches[:count],1):
                    event=update(head,optimizer,frozen,train,indices,device,torch,arm)
                    gradients.append(event)
                    stream.write(json.dumps(dict(step=step,arm=arm,**event),sort_keys=True,allow_nan=False)+'\n')
                    stream.flush()
                    if step in (1,2) or step%25==0: print(arm,step,'/',count,'loss',event['loss'],flush=True)
            if gradients[0]['stem_grad_norm']!=0. or gradients[0]['terminal_grad_norm']<=0. or gradients[1]['stem_grad_norm']<=0.:
                raise ValueError('Independent gradient path check failed')
            replay=[train[i] for i in batches[0]]
            head,checkpoint=save_reload(arm_dir,args.stage,arm,head,optimizer,replay,proof,device,torch,model)
            final_rows=evaluate_rows(head,records,device,torch)
            engineering=dict(neutral_exact=True,gpu_save_reload=True,
                frozen_midpoint_state=sealed.state_digest(frozen)==frozen_before==sealed.FIXED['head_digest'],
                frozen_midpoint_no_grad=all(not p.requires_grad and p.grad is None for p in frozen.parameters()),
                whole_run_frozen_identity=all(old.metrics.frozen_identity(r) for r in final_rows),
                independent_parameter_count=sealed.state_digest(head)['parameter_count']==2760,
                first_stem_zero=gradients[0]['stem_grad_norm']==0.,first_terminal_learns=gradients[0]['terminal_grad_norm']>0.,
                subsequent_stem_learns=gradients[1]['stem_grad_norm']>0.,
                fixed_head_weight_sha=sealed.sha(head_dir/sealed.FIXED['path'])==sealed.FIXED['sha256'],
                cache_manifest_sha=sealed.sha(cache_dir/'cache_manifest.json')==sealed.CACHE_SHA,
                source_closure=checked_sources()==identity,
                fixed_val_replay=len(val)==887 if val else True,
                smoke_not_reused=initial==smoke[arm]['initial_digest'] if smoke else True,
                final_budget=count==200 if args.stage=='finite' else count==2,
                decoded_grad_to_logits=gradients[0]['decoded_logit_grad_norm']>0.,
                decoded_grad_to_fine=gradients[0]['decoded_fine_grad_norm']>0.,
                only_new_fit_gradients=all(train[i]['sample_role']=='fit' for b in batches[:count] for i in b))
            if not all(engineering.values()): raise ValueError('Engineering protection failed')
            old.write_rows(arm_dir/'final_rows.jsonl',final_rows)
            gate=None
            if args.stage=='finite':
                train_summary=evaluate(final_rows)
                train_gate=train_gates(train_summary)
                sealed.write(arm_dir/'train_summary.json',train_summary)
                sealed.write(arm_dir/'train_gate.json',train_gate)
                sealed.write(arm_dir/'train_candidate_summary.json',candidate_summary(final_rows))
                gate=dict(finite_joint_gate_pass=False,failed_checks=train_gate['failed_checks'],
                          status=train_gate['status'],checks=train_gate['checks'])
                if train_gate['train_gate_pass']:
                    val,val_bytes=load_val_after_gate(train_gate,cache_dir,cache,plan,torch,frozen_model.original)
                    if bytes_read+val_bytes>old.SETTINGS['max_cpu_tensor_bytes']:
                        raise ValueError('Combined TRAIN/VAL CPU tensor budget exceeded')
                    val=old.prepare_frozen(val,frozen,device,torch,frozen_model.original,model,references)
                    if len(val)!=887: raise ValueError('Full fixed VAL required after TRAIN gate')
                    neutral=model.ContinuousBoundarySizeHead().to(device)
                    neutral_rows=evaluate_rows(neutral,val,device,torch)
                    if any(r['midpoint']!=r['edge_residual'] for r in neutral_rows):
                        raise ValueError('Neutral full VAL replay changed M')
                    baseline_rows=baseline_rows+neutral_rows
                    baseline_summary=evaluate(baseline_rows)
                    for domain,counts in {'real':(374,362,199),'sim':(512,512,464)}.items():
                        value=baseline_summary['val_s1']['all'][domain]['midpoint']
                        if tuple(value[k]['numerator'] for k in ('output_coverage','center_correct_full_frame',
                            'joint_size10_full_frame'))!=counts: raise ValueError('Fixed M VAL counts differ')
                    old.write_rows(directory/'baseline_val_rows.jsonl',neutral_rows)
                    sealed.write(directory/'baseline_summary.json',baseline_summary)
                    final_rows=final_rows+evaluate_rows(head,val,device,torch)
                    engineering['fixed_val_replay']=True
                    engineering['whole_run_frozen_identity']=all(old.metrics.frozen_identity(r) for r in final_rows)
                    completion.update(val_tensor_access=True,input_tensor_bytes=bytes_read+val_bytes)
                    if not all(engineering.values()): raise ValueError('Post-gate VAL engineering failed')
                    old.write_rows(arm_dir/'final_val_rows.jsonl',final_rows[len(train):])
            if val:
                summary=evaluate(final_rows)
                arm_summaries[arm]=summary
                gate=boundary_gates(summary,engineering)
                sealed.write(arm_dir/'summary.json',summary)
                sealed.write(arm_dir/'val_candidate_summary.json',candidate_summary(final_rows))
            # These continuous dimensions are diagnostic geometry, not depth estimates.
            rows=[dict(image=r['image'],role=r['sample_role'],sequence=r['sequence'],scale=r['scale'],
                midpoint=support.d.size_errors(r['gt'],r['midpoint']),
                corrected=support.d.size_errors(r['gt'],r['edge_residual'])) for r in final_rows]
            sealed.write(arm_dir/'continuous_sizes.json',dict(records=rows))
            bucket_events=[]
            for sequence in sorted({r['sequence'] for r in final_rows}):
                for shard in sorted({r['shard'] for r in final_rows}):
                    chain=sorted((r for r in final_rows if r['sequence']==sequence and r['shard']==shard),
                                 key=lambda r:r['frame_id'])
                    adjacent=[(a,b) for a,b in zip(chain,chain[1:]) if b['frame_id']==a['frame_id']+1
                              and a['boundary'] is not None and b['boundary'] is not None]
                    bucket_events.append(dict(sequence=sequence,shard=shard,pairs=len(adjacent),
                        switches=[sum(a['boundary']['predicted_bucket'][i]!=b['boundary']['predicted_bucket'][i]
                                      for a,b in adjacent) for i in range(2)]))
            sealed.write(arm_dir/'bucket_switches.json',dict(records=bucket_events,
                note='ROI-axis buckets; missing frames/gaps excluded, no automatic confidence or threshold.'))
            if gate is not None: sealed.write(arm_dir/'finite_gate.json',gate)
            result=dict(protocol=VERSION,stage=args.stage,arm=arm,proof=proof,updates=count,
                engineering=engineering,initial_digest=initial,final_digest=sealed.state_digest(head),
                checkpoint=checkpoint,gate_pass=None if gate is None else gate['finite_joint_gate_pass'],
                failed_checks=[] if gate is None else gate['failed_checks'])
            publish(arm_dir,result); results[arm]=result
        if val:
            comparison={}
            for group in ('real','sim','real_seq07','real_seq14','sim_seq10'):
                comparison[group]=dict(midpoint=baseline_summary['val_s1']['all'][group]['midpoint'],
                    **{arm:summary['val_s1']['all'][group]['edge_residual']
                       for arm,summary in arm_summaries.items()})
            sealed.write(directory/'arm_comparison.json',dict(protocol=VERSION,val=comparison,
                method_field_mapping={'midpoint':'fixed_sigma1p5_epoch03','edge_residual':'boundary_continuous_candidate'},
                identical_schedule_to_prior_protocol=True,automatic_promotion=False,
                downstream_benefits_verified=False))
        if current_plan(args)!=(plan,input_proof): raise ValueError('Collection/plan changed during run')
        status=('SIZE_CONTINUOUS_SMOKE_COMPLETE' if args.stage=='smoke' else
            'SIZE_CONTINUOUS_FINITE_COMPLETE_REVIEW_REQUIRED' if val else 'SIZE_CONTINUOUS_TRAIN_GATE_FAIL_KEEP_M')
        completion.update(status=status,train_gate_pass=None if train_gate is None else train_gate['train_gate_pass'],
            results={arm:dict(gate_pass=r['gate_pass'],failed_checks=r['failed_checks'],updates=r['updates'])
                     for arm,r in results.items()},
            frozen_state_after=sealed.state_digest(frozen),elapsed_seconds=time.monotonic()-started,
            peak_torch_memory_allocated_bytes=torch.cuda.max_memory_allocated(device),
            downstream_benefits_verified=False)
        publish(directory,completion)
        print(completion['status'],flush=True)
        for arm,result in results.items(): print(arm,'updates',result['updates'],'gate',result['gate_pass'],
                                                'failed_checks',len(result['failed_checks']),flush=True)
    except Exception as error:
        completion.update(error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-started)
        publish(directory,completion)
        raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('check','prepare','smoke','finite'),required=True)
    p.add_argument('--out-dir',required=True)
    inputs=p.add_mutually_exclusive_group()
    inputs.add_argument('--collection-dir'); inputs.add_argument('--collection-archive')
    p.add_argument('--cache-dir'); p.add_argument('--head-dir')
    p.add_argument('--device',default='cuda:0'); p.add_argument('--plan-file'); p.add_argument('--smoke-dir')
    return p


if __name__=='__main__': run(parser().parse_args())
