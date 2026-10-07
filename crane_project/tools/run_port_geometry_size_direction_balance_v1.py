#!/usr/bin/env python3
"""Fixed200 fresh-init uniform vs direction-balanced TRAIN loss contribution.

Same cached inputs, head, initialization, schedule, optimizer and loss form.
No old failed checkpoint loads; no TEST, formula/policy fit or promotion.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.tools import run_port_geometry_size_boundary_continuous_v1 as control
from crane_project.utils import port_geometry_size_direction_balance_v1 as balance

old=control.old;sealed=control.sealed;design=control.design;support=control.support
VERSION=balance.VERSION;ARMS=balance.ARMS;MODEL_SETTINGS=control.MODEL_SETTINGS
BASE='crane_project/tools/'+VERSION
PROTOCOL=ROOT/(BASE+'_protocol.json');SOURCES=ROOT/(BASE+'_sources.json')
EXPOSURE=ROOT/(BASE+'_exposure.json')
NEW={BASE+'_protocol.json',BASE+'_exposure.json',
     'crane_project/utils/'+VERSION+'.py','crane_project/tools/run_'+VERSION+'.py',
     'crane_project/tools/run_'+VERSION+'.sh','tests/test_'+VERSION+'.py',
     'crane_project/tools/port_geometry_size_boundary_continuous_v1_sources.json'}
EXPECTED_INITIAL=dict(buffer_count=0,buffers='44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a',
    parameter_count=2760,parameters='baffcbce4031ed78f054efd6271149771b5a9513d49cb390f083930aa0f8c5ce')
RUNTIME_CORE=dict(torch='1.13.1+cu117',cuda='11.7',cudnn=8500,dtype='torch.float32',
                  tf32_matmul=False,tf32_cudnn=True)
resolve_collection=control.resolve_collection;resolve_head=control.resolve_head
current_plan=control.current_plan;load_records=control.load_records
verify_train_replay=control.verify_train_replay;evaluate_rows=control.evaluate_rows
evaluate=control.evaluate;candidate_summary=control.candidate_summary


def protocol_document():
    return dict(protocol=VERSION,stages=['check','prepare','smoke','finite'],arms=list(ARMS),
        parameter_count=2760,model_settings=MODEL_SETTINGS,settings=design.SETTINGS,
        direction_settings=balance.SETTINGS,expected_initial=EXPECTED_INITIAL,runtime_core=RUNTIME_CORE,
        fixed_checkpoint=sealed.FIXED,cache_manifest_sha256=sealed.CACHE_SHA,
        collection_artifacts=support.INPUT_FILES,original_baseline_sha256=balance.BASELINE_SHA,
        original_schedule_sha256=balance.SCHEDULE_SHA,exposure_sha256=sealed.sha(EXPOSURE),
        single_factor='Only per-view loss weight changes. Original fixed batches, inputs, head, initialization, Adam, lr, clipping and200 updates unchanged. Fresh independent2-step smoke discarded, fresh200 final only per arm.',
        groups='Original M canonical original-image relative errors. Near_correct if both abs<=.03; otherwise need_enlarge if both negative, need_shrink if both positive, mixed otherwise. Missing never enters updates. Domain/scale cells are offline training organization only.',
        weights='Actual original200 schedule exposure, not full-fit counts. Per-cell capped inverse exposure: w_g=min(4,lambda/n_g), sum(n_g*w_g)=400. All four groups must have scheduled exposure. Fixed entire-run weights; no per-batch normalization, sampler change or GT in forward. Uniform control calls exact original mean SmoothL1 beta.1; balanced uses mean(weight*same pointwise loss).',
        gates='Final delivered TRAIN outputs only; inherited fit/probe and geometry guards plus fixed M direction-group continuous scale/short/ratio/tail protection, strict fit common-scale gain for enlarge/shrink, near-correct no new joint10 errors. Balanced must pass before any VAL tensor load. Then evaluate BOTH final heads on same full887 VAL as a predeclared controlled diagnostic; failed control is not thereby promoted. Original VAL/probe/bias/ratio gates apply, balanced also must meet direct two-arm protection. No best checkpoint/arm selection.',
        scope=dict(detector_updates=0,frozen_midpoint_updates=0,feature_extractions=0,test_access=False,
            policy_modified=False,depth_formula_modified=False,dataset_splits_modified=False,labels_modified=False,
            automatic_promotion=False,formal_training=False),
        limitations=['Exposure imbalance is not a proven root cause; group weights test one training factor.',
            'TRAIN/probe were exposed to detector/M training; no new events or independent labels.',
            'Direction groups use projected OBB sizes, not verified physical short-edge measurements.',
            'Gradient clipping may alter effective contributions; raw/weighted group losses and pre/post clip norms are logged.',
            'CUDA grid_sample backward may be nondeterministic; fresh control is required, old200 results are contextual only.',
            'Geometry pass does not establish reliability or metric-depth benefit; downstream requires separate authorization/evaluation.'])


def checked_sources():
    prior=control.checked_sources();manifest=sealed.read(SOURCES)
    if manifest['protocol']!=VERSION or set(manifest['sources'])!=set(prior['sources'])|NEW or sealed.read(PROTOCOL)!=protocol_document():
        raise ValueError('Single-factor source/protocol closure differs')
    for name,digest in manifest['sources'].items():
        if sealed.sha(ROOT/name)!=digest:raise ValueError('Source SHA differs: '+name)
    contract=sealed.read(EXPOSURE)
    if contract['protocol']!=VERSION or contract['settings']!=balance.SETTINGS or contract['original_baseline_sha256']!=balance.BASELINE_SHA or contract['original_schedule_sha256']!=balance.SCHEDULE_SHA:
        raise ValueError('Original TRAIN exposure contract differs')
    return dict(protocol_sha256=sealed.sha(PROTOCOL),sources_sha256=sealed.sha(SOURCES),
        sources=manifest['sources'],control_identity=prior)


def cache_identity(identity):
    return identity['control_identity']['parent_identity']['parent_identity']['parent_identity']


def combined_train_gate(rows):
    inherited=control.train_gates(evaluate(rows));direction=balance.direction_gates(balance.direction_summary(rows))
    checks=dict(inherited['checks'],**{'direction/'+k:v for k,v in direction['checks'].items()})
    failed=[k for k,v in checks.items() if not v]
    return dict(checks=checks,failed_checks=failed,train_gate_pass=not failed,
        status='SIZE_DIRECTION_TRAIN_PASS' if not failed else 'SIZE_DIRECTION_TRAIN_FAIL_KEEP_M',
        val_tensor_access=False,checkpoint_selection=False)


def checked_smoke(directory,identity,proof):
    if not directory:raise ValueError('Matching independent smoke is required')
    directory=Path(directory);report=sealed.read(directory/'completion.json');index=sealed.read(directory/'artifacts.json')
    if report['protocol']!=VERSION or report['status']!='SIZE_DIRECTION_SMOKE_COMPLETE' or report['source_identity']!=identity or report['proof']!=proof or report['val_tensor_access'] or report['arms']!=list(ARMS) or index['files'].get('completion.json')!=sealed.sha(directory/'completion.json'):
        raise ValueError('Matching two-arm no-VAL smoke required')
    result={}
    for arm in ARMS:
        p=directory/arm;r=sealed.read(p/'completion.json');ix=sealed.read(p/'artifacts.json')
        required=old.metrics.ENGINEERING_REQUIRED|{'decoded_grad_to_logits','decoded_grad_to_fine','only_new_fit_gradients','fixed_schedule_and_weights'}
        if ix['protocol']!=VERSION or any(ix['files'].get(n)!=sealed.sha(p/n) for n in ('completion.json','experimental_head.pth')) or r['protocol']!=VERSION or r['arm']!=arm or r['stage']!='smoke' or r['proof']!=proof or r['updates']!=2 or not required<=set(r['engineering']) or not all(r['engineering'].values()):
            raise ValueError('Smoke engineering/checkpoint proof differs')
        result[arm]=r
    return result


def checked_output_directory(path,root=ROOT):
    directory=Path(path).resolve()
    parent_dir=(root/'work_dirs'/'port_geometry_size_boundary_v1').resolve()
    if (directory.parent.parent!=parent_dir or directory.name not in ('check','prepare','smoke','finite') or
            not directory.parent.name.startswith(VERSION+'_')):
        raise ValueError('Use work_dirs/port_geometry_size_boundary_v1/'+VERSION+'_RUN/{check,prepare,smoke,finite}')
    if directory.exists(): raise FileExistsError(str(directory))
    return directory

def publish(directory,completion):
    sealed.write(directory/'completion.json',completion)
    sealed.write(directory/'artifacts.json',dict(protocol=VERSION,files={
        p.name:sealed.sha(p) for p in directory.iterdir() if p.is_file() and p.name!='artifacts.json'}))

def checked_plan(path,plan,proof,identity):
    if not path: raise ValueError('smoke/finite require prepared --plan-file')
    path=Path(path).resolve()
    index=sealed.read(path.parent/'artifacts.json')
    completion=sealed.read(path.parent/'completion.json')
    if (index['protocol']!=VERSION or completion['status']!='SIZE_DIRECTION_PREPARED' or
        sealed.read(path)!=plan or sealed.read(path.parent/'input_identity.json')!=proof or
        sealed.read(path.parent/'source_identity.json')!=identity or
        sealed.read(path.parent/'protocol.json')!=protocol_document()):
        raise ValueError('Prepared TRAIN support/source differs')
    for name in ('plan.json','completion.json','input_identity.json','source_identity.json','protocol.json','exposure_contract.json'):
        if index['files'].get(name)!=sealed.sha(path.parent/name):
            raise ValueError('Prepared artifact SHA differs: '+name)

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

def update(head,optimizer,frozen,records,indices,device,torch,arm):
    if arm not in ARMS: raise ValueError('Only the fixed boundary candidate is allowed')
    from crane_project.utils import port_geometry_size_boundary_continuous_v1 as model
    owned=[id(p) for group in optimizer.param_groups for p in group['params']]
    if len(owned)!=len(set(owned)) or set(owned)!={id(p) for p in head.parameters()}:
        raise ValueError('Optimizer may own only new independent size head')
    roi,mask,b,bm,xy,m,gt=old.batch(records,indices,device,torch)
    if any(records[i].get('direction_group') not in balance.GROUPS or
           records[i].get('direction_weight') is None for i in indices):
        raise ValueError('Only fixed scheduled TRAIN groups/weights may enter updates')
    optimizer.zero_grad()
    output=head(roi,mask,b,bm,xy,m)
    weights=[1. if arm=='uniform_control' else records[i]['direction_weight'] for i in indices]
    terms=balance.loss(output,gt,m,bm,weights,arm)
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
    event['direction_contribution']={}
    for group in balance.GROUPS:
        selected=[j for j,i in enumerate(indices) if records[i]['direction_group']==group]
        event['direction_contribution'][group]=dict(slots=len(selected),weight_mass=sum(weights[j] for j in selected),
            raw_loss_contribution=sum(float(terms['per_view_unweighted'][j].detach()) for j in selected)/len(indices),
            weighted_loss_contribution=sum(float(terms['per_view_unweighted'][j].detach())*weights[j] for j in selected)/len(indices))
    event['direction_contribution_by_cell']={}
    for domain in ('real','sim'):
        for scale in (1.,.5):
            for group in balance.GROUPS:
                selected=[j for j,i in enumerate(indices) if (records[i]['domain'],records[i]['scale'],records[i]['direction_group'])==(domain,scale,group)]
                event['direction_contribution_by_cell'][domain+'/'+str(scale)+'/'+group]=dict(
                    slots=len(selected),weight_mass=sum(weights[j] for j in selected),
                    raw_loss_contribution=sum(float(terms['per_view_unweighted'][j].detach()) for j in selected)/len(indices),
                    weighted_loss_contribution=sum(float(terms['per_view_unweighted'][j].detach())*weights[j] for j in selected)/len(indices))
    return event


def contribution_summary(events):
    groups={}
    for key in events[0]['direction_contribution_by_cell']:
        groups[key]={name:sum(e['direction_contribution_by_cell'][key][name] for e in events)
            for name in ('slots','weight_mass','raw_loss_contribution','weighted_loss_contribution')}
    return dict(updates=len(events),by_cell_group=groups,clipped_updates=sum(e['clipped'] for e in events),
        max_grad_norm_before_clip=max(e['total_grad_norm_before_clip'] for e in events),
        max_grad_norm_after_clip=max(e['total_grad_norm_after_clip'] for e in events),
        note='Actual objective contributions before clipping; these are not parameter-gradient directions or causal attribution.')

def prepare_weights(records,batches,baseline_rows):
    schedule=design.schedule_report(records,batches)
    schedule['batches']=[[dict(image=records[i]['image'],scale=records[i]['scale']) for i in b] for b in batches]
    contract=balance.exposure_contract(baseline_rows,schedule)
    if contract!=sealed.read(EXPOSURE):raise ValueError('Actual cached TRAIN groups/exposure differ from fixed contract')
    lookup={(r['image'],r['scale']):r for r in contract['scheduled_views']}
    result=[]
    for r in records:
        entry=lookup.get((r['image'],r['scale']))
        result.append(dict(r,direction_group=None if entry is None else entry['group'],
            direction_weight=None if entry is None else entry['weight']))
    return result,schedule,contract


def controlled_gates(uniform,balanced):
    checks={};tol=balance.SETTINGS['metric_tolerance']
    a=balance.direction_summary(uniform);b=balance.direction_summary(balanced)
    if set(a)!=set(b):raise ValueError('Controlled direction group identities differ')
    for key,x in a.items():
        y=b[key]
        if x['delivered_group_identity']!=y['delivered_group_identity']:raise ValueError('Controlled frame pairing differs')
        label=key.split('/')[-1]
        for metric in ('common_log_mae','short_mae','ratio_log_mae'):
            checks[key+'/'+metric]=y['corrected'][metric]<=x['corrected'][metric]+tol
        if label=='near_correct':
            checks[key+'/no_new_size10']=y['new_size10_errors_from_correct']<=x['new_size10_errors_from_correct']
        if key.startswith('train_s1/fit/') and label=='need_enlarge':
            checks[key+'/common_scale_strict_gain']=y['corrected']['common_log_mae']<x['corrected']['common_log_mae']-tol
    failed=[k for k,v in checks.items() if not v]
    return dict(checks=checks,failed_checks=failed,controlled_gate_pass=not failed)


def train_gate_for_val(gates,contrast):
    weighted=gates['direction_balanced'];checks=dict(weighted['checks'],**{'vs_uniform/'+k:v for k,v in contrast['checks'].items()})
    failed=[k for k,v in checks.items() if not v]
    return dict(checks=checks,failed_checks=failed,train_gate_pass=not failed,
        status='SIZE_DIRECTION_TRAIN_PASS' if not failed else 'SIZE_DIRECTION_TRAIN_FAIL_KEEP_M',
        val_tensor_access=False,checkpoint_selection=False)


def controlled_val_gates(uniform,balanced):
    checks={};tol=balance.SETTINGS['metric_tolerance']
    for name,x in uniform['val_s1']['all'].items():
        if name=='overall':continue
        a=x['edge_residual'];b=balanced['val_s1']['all'][name]['edge_residual']
        checks[name+'/joint10']=b['joint_size10_full_frame']['numerator']>=a['joint_size10_full_frame']['numerator']
        for edge in ('long','short'):
            for metric in ('mae','p95'):
                v,w=a['edges'][edge][metric],b['edges'][edge][metric]
                checks[name+'/'+edge+'_'+metric]=v is not None and w is not None and math.isfinite(w) and w<=v+tol
        for metric in ('log_ratio_mae','log_ratio_p95'):
            v,w=a['continuous_sizes'][metric],b['continuous_sizes'][metric]
            checks[name+'/'+metric]=v is not None and w is not None and math.isfinite(w) and w<=v+tol
    failed=[k for k,v in checks.items() if not v]
    return dict(checks=checks,failed_checks=failed,controlled_val_gate_pass=not failed)


def load_val_after_gate(gate,*args):
    if gate.get('train_gate_pass') is not True:raise ValueError('Balanced direction/geometry/control TRAIN gate must pass before VAL tensors')
    return control.load_val_after_gate(gate,*args)


def run(args):
    identity=checked_sources();directory=checked_output_directory(args.out_dir)
    protected=[ROOT/'crane_project',ROOT/'tests',ROOT/'docs']
    for key in ('head_dir','cache_dir','collection_dir','smoke_dir'):
        if getattr(args,key):protected.append(Path(getattr(args,key)).resolve())
    if args.plan_file:protected.append(Path(args.plan_file).resolve().parent)
    if any(p==directory or p in directory.parents for p in protected):raise ValueError('Output must be separate from fixed inputs')
    directory.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    completion=dict(protocol=VERSION,stage=args.stage,status='SIZE_DIRECTION_FAILED',
        started_utc=datetime.now(timezone.utc).isoformat(),source_identity=identity,
        val_tensor_access=False,arms=list(ARMS),**protocol_document()['scope'])
    try:
        sealed.write(directory/'protocol.json',protocol_document());sealed.write(directory/'source_identity.json',identity)
        if args.stage=='check':
            completion.update(status='SIZE_DIRECTION_STATIC_CONTRACT_PASS',model_loaded=False,parameter_updates=0)
            publish(directory,completion);print(completion['status'],flush=True);return
        plan,input_proof=current_plan(args)
        sealed.write(directory/'input_identity.json',input_proof)
        contract=sealed.read(EXPOSURE)
        planned={r['image']:r for r in plan['records']}
        if any(planned[r['image']]['sample_role']!='fit' or planned[r['image']]['domain']!=r['domain'] for r in contract['scheduled_views']):
            raise ValueError('Pinned exposure crossed original TRAIN roles')
        sealed.write(directory/'exposure_contract.json',contract)
        if args.stage=='prepare':
            sealed.write(directory/'plan.json',plan)
            completion.update(status='SIZE_DIRECTION_PREPARED',role_counts=plan['role_counts'],slots=contract['slots'],
                exposure_sha256=sealed.sha(EXPOSURE),model_loaded=False,parameter_updates=0)
            publish(directory,completion);print(completion['status'],plan['role_counts'],flush=True);return
        checked_plan(args.plan_file,plan,input_proof,identity)
        if sealed.read(Path(args.plan_file).parent/'exposure_contract.json')!=contract:raise ValueError('Prepared fixed weights differ')
        cache_dir=sealed.resolve_cache(args.cache_dir);cache=sealed.checked_cache(cache_dir,cache_identity(identity))
        head_dir=resolve_head(args.head_dir)
        selection,references,indexed=sealed.checked_selection(head_dir,cache_identity(identity))
        import torch
        from crane_project.utils import port_geometry_midpoint_sigma_v1 as frozen_model
        from crane_project.utils import port_geometry_size_boundary_continuous_v1 as model
        if model.SETTINGS!=MODEL_SETTINGS or model.PARAMETER_COUNT!=2760:raise ValueError('Original continuous implementation differs')
        device=torch.device(args.device)
        if os.environ.get('WORLD_SIZE','1')!='1' or device.type!='cuda' or not torch.cuda.is_available():raise ValueError('Single CUDA device required')
        torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device);old.seed_all(torch)
        runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
            device=str(device),gpu=torch.cuda.get_device_name(device),dtype='torch.float32',cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            tf32_matmul=getattr(torch.backends.cuda.matmul,'allow_tf32',None),tf32_cudnn=getattr(torch.backends.cudnn,'allow_tf32',None))
        if any(runtime[k]!=v for k,v in RUNTIME_CORE.items()):raise ValueError('Requires original Torch1.13/cu117/cuDNN8500/TF32 flags: '+str(runtime))
        proof=dict(source_identity=identity,plan_sha256=sealed.sha(args.plan_file),exposure_sha256=sealed.sha(EXPOSURE),
            collection_artifacts=support.INPUT_FILES,cache_manifest_sha256=sealed.CACHE_SHA,detector_state=cache['detector_state'],
            runtime=runtime,fixed_checkpoint=sealed.FIXED,selected_artifacts=indexed)
        smoke=checked_smoke(args.smoke_dir,identity,proof) if args.stage=='finite' else None
        frozen=sealed.load_head(torch,frozen_model,head_dir,selection,cache,device);frozen_before=sealed.state_digest(frozen)
        train,val,bytes_read=load_records(cache_dir,cache,plan,torch,frozen_model.original,False)
        if val:raise ValueError('VAL was loaded before TRAIN gate')
        batches=design.schedule(train)
        schedule=design.schedule_report(train,batches)
        schedule['batches']=[[dict(image=train[i]['image'],scale=train[i]['scale']) for i in b] for b in batches]
        sealed.write(directory/'schedule.json',schedule)
        if sealed.sha(directory/'schedule.json')!=balance.SCHEDULE_SHA:raise ValueError('Original1600 slots/order changed')
        train=old.prepare_frozen(train,frozen,device,torch,frozen_model.original,model,references)
        verify_train_replay(train,plan)
        old.seed_all(torch);neutral=model.ContinuousBoundarySizeHead().to(device)
        if sealed.state_digest(neutral)!=EXPECTED_INITIAL:raise ValueError('Original neutral initialized parameters differ')
        baseline=evaluate_rows(neutral,train,device,torch)
        if any(r['midpoint']!=r['edge_residual'] for r in baseline):raise ValueError('Neutral M replay differs')
        train,_,actual=prepare_weights(train,batches,baseline)
        sealed.write(directory/'actual_exposure.json',actual)
        completion.update(proof=proof,input_tensor_bytes=bytes_read,schedule_sha256=sealed.sha(directory/'schedule.json'))
        if args.stage=='finite' and completion['schedule_sha256']!=sealed.sha(Path(args.smoke_dir)/'schedule.json'):
            raise ValueError('Smoke/finite schedules differ')
        old.write_rows(directory/'baseline_rows.jsonl',baseline);sealed.write(directory/'baseline_train_summary.json',evaluate(baseline))
        if args.stage=='smoke':
            # Full baseline/exposure verified before limiting the two-step update set.
            selected=sorted({i for batch in batches[:2] for i in batch});remap={i:j for j,i in enumerate(selected)}
            train=[train[i] for i in selected];batches=[[remap[i] for i in batch] for batch in batches[:2]]
        count=2 if args.stage=='smoke' else 200
        results={};heads={};final_train={};gates={}
        for arm in ARMS:
            arm_dir=directory/arm;arm_dir.mkdir();old.seed_all(torch)
            head=model.ContinuousBoundarySizeHead().to(device);initial=sealed.state_digest(head)
            if initial!=EXPECTED_INITIAL or (smoke and (initial!=smoke[arm]['initial_digest'] or initial==smoke[arm]['final_digest'])):
                raise ValueError('Fresh matching neutral initialization required; no failed/smoke weights may continue')
            optimizer=torch.optim.Adam(head.parameters(),lr=design.SETTINGS['lr'],weight_decay=0.)
            gradients=[];head.train()
            with (arm_dir/'progress.jsonl').open('x',encoding='utf-8') as stream:
                for step,indices in enumerate(batches[:count],1):
                    event=update(head,optimizer,frozen,train,indices,device,torch,arm);gradients.append(event)
                    stream.write(json.dumps(dict(step=step,arm=arm,**event),sort_keys=True,allow_nan=False)+'\n');stream.flush()
                    if step in (1,2) or step%25==0:print(arm,step,'/',count,'loss',event['loss'],flush=True)
            if gradients[0]['stem_grad_norm']!=0 or gradients[0]['terminal_grad_norm']<=0 or gradients[1]['stem_grad_norm']<=0:
                raise ValueError('Original independent gradient path failed')
            sealed.write(arm_dir/'training_contributions.json',contribution_summary(gradients))
            head,checkpoint=save_reload(arm_dir,args.stage,arm,head,optimizer,[train[i] for i in batches[0]],proof,device,torch,model)
            rows=evaluate_rows(head,train,device,torch);old.write_rows(arm_dir/'final_rows.jsonl',rows)
            engineering=dict(neutral_exact=True,gpu_save_reload=True,
                frozen_midpoint_state=sealed.state_digest(frozen)==frozen_before==sealed.FIXED['head_digest'],
                frozen_midpoint_no_grad=all(not p.requires_grad and p.grad is None for p in frozen.parameters()),
                whole_run_frozen_identity=all(old.metrics.frozen_identity(r) for r in rows),independent_parameter_count=sealed.state_digest(head)['parameter_count']==2760,
                first_stem_zero=gradients[0]['stem_grad_norm']==0,first_terminal_learns=gradients[0]['terminal_grad_norm']>0,
                subsequent_stem_learns=gradients[1]['stem_grad_norm']>0,fixed_head_weight_sha=sealed.sha(head_dir/sealed.FIXED['path'])==sealed.FIXED['sha256'],
                cache_manifest_sha=sealed.sha(cache_dir/'cache_manifest.json')==sealed.CACHE_SHA,source_closure=checked_sources()==identity,
                fixed_val_replay=True,smoke_not_reused=initial==smoke[arm]['initial_digest'] if smoke else True,final_budget=count==(200 if args.stage=='finite' else 2),
                decoded_grad_to_logits=gradients[0]['decoded_logit_grad_norm']>0,decoded_grad_to_fine=gradients[0]['decoded_fine_grad_norm']>0,
                only_new_fit_gradients=all(train[i]['sample_role']=='fit' and train[i]['eligible'] for batch in batches[:count] for i in batch),fixed_schedule_and_weights=True)
            if not all(engineering.values()):raise ValueError('Engineering protection failed')
            if args.stage=='finite':
                gates[arm]=combined_train_gate(rows);sealed.write(arm_dir/'train_gate.json',gates[arm])
                sealed.write(arm_dir/'train_summary.json',evaluate(rows));sealed.write(arm_dir/'train_candidate_summary.json',candidate_summary(rows))
                sealed.write(arm_dir/'direction_summary.json',balance.direction_summary(rows))
            heads[arm]=head;final_train[arm]=rows
            results[arm]=dict(protocol=VERSION,stage=args.stage,arm=arm,proof=proof,updates=count,
                engineering=engineering,initial_digest=initial,final_digest=sealed.state_digest(head),checkpoint=checkpoint,
                gate_pass=None,failed_checks=[] if args.stage=='smoke' else gates[arm]['failed_checks'])
            publish(arm_dir,results[arm])
        gate=None
        if args.stage=='finite':
            comparison=controlled_gates(final_train['uniform_control'],final_train['direction_balanced'])
            sealed.write(directory/'controlled_direction_gate.json',comparison)
            gate=train_gate_for_val(gates,comparison);sealed.write(directory/'train_gate.json',gate)
            if gate['train_gate_pass']:
                val,val_bytes=load_val_after_gate(gate,cache_dir,cache,plan,torch,frozen_model.original)
                if bytes_read+val_bytes>old.SETTINGS['max_cpu_tensor_bytes']:raise ValueError('Combined CPU tensor budget exceeded')
                val=old.prepare_frozen(val,frozen,device,torch,frozen_model.original,model,references)
                old.seed_all(torch);neutral=model.ContinuousBoundarySizeHead().to(device)
                base_val=evaluate_rows(neutral,val,device,torch)
                if len(val)!=887 or any(r['midpoint']!=r['edge_residual'] for r in base_val):raise ValueError('Full887 neutral VAL replay differs')
                old.write_rows(directory/'baseline_val_rows.jsonl',base_val)
                base_summary=evaluate(baseline+base_val)
                for domain,counts in {'real':(374,362,199),'sim':(512,512,464)}.items():
                    v=base_summary['val_s1']['all'][domain]['midpoint']
                    if tuple(v[k]['numerator'] for k in ('output_coverage','center_correct_full_frame','joint_size10_full_frame'))!=counts:raise ValueError('Fixed M VAL counts differ')
                sealed.write(directory/'baseline_summary.json',base_summary)
                val_summaries={}
                for arm in ARMS:
                    rows=evaluate_rows(heads[arm],val,device,torch);old.write_rows(directory/arm/'final_val_rows.jsonl',rows)
                    if not all(old.metrics.frozen_identity(r) for r in rows):raise ValueError('Post-gate VAL frozen identity failed')
                    summary=evaluate(final_train[arm]+rows);sealed.write(directory/arm/'summary.json',summary)
                    val_summaries[arm]=summary
                    vg=control.boundary_gates(summary,results[arm]['engineering']);sealed.write(directory/arm/'val_gate.json',vg)
                    results[arm].update(gate_pass=vg['finite_joint_gate_pass'] if arm=='direction_balanced' else None,
                        val_geometry_gate_pass=vg['finite_joint_gate_pass'],control_diagnostic_only=arm=='uniform_control')
                    if arm=='direction_balanced':results[arm]['failed_checks']=vg['failed_checks']
                    publish(directory/arm,results[arm])
                vg_control=controlled_val_gates(val_summaries['uniform_control'],val_summaries['direction_balanced'])
                sealed.write(directory/'controlled_val_gate.json',vg_control)
                weighted=results['direction_balanced']
                weighted['gate_pass']=weighted['gate_pass'] and vg_control['controlled_val_gate_pass']
                weighted['failed_checks']+=['vs_uniform_val/'+name for name in vg_control['failed_checks']]
                publish(directory/'direction_balanced',weighted)
                completion.update(val_tensor_access=True,input_tensor_bytes=bytes_read+val_bytes)
            else:
                results['direction_balanced'].update(gate_pass=False,failed_checks=gate['failed_checks'])
                publish(directory/'direction_balanced',results['direction_balanced'])
        if current_plan(args)!=(plan,input_proof) or checked_sources()!=identity or sealed.state_digest(frozen)!=frozen_before:
            raise ValueError('Inputs/source/frozen state changed')
        status='SIZE_DIRECTION_SMOKE_COMPLETE' if args.stage=='smoke' else 'SIZE_DIRECTION_FINITE_COMPLETE_REVIEW_REQUIRED' if completion['val_tensor_access'] else 'SIZE_DIRECTION_TRAIN_GATE_FAIL_KEEP_M'
        completion.update(status=status,train_gate_pass=None if gate is None else gate['train_gate_pass'],
            results={arm:dict(updates=r['updates'],gate_pass=r['gate_pass'],failed_checks=r['failed_checks']) for arm,r in results.items()},
            frozen_state_after=sealed.state_digest(frozen),elapsed_seconds=time.monotonic()-started,
            peak_torch_memory_allocated_bytes=torch.cuda.max_memory_allocated(device),downstream_benefits_verified=False)
        publish(directory,completion);print(status,flush=True)
        for arm,r in results.items():print(arm,'updates',r['updates'],'gate',r['gate_pass'],'failed_checks',len(r['failed_checks']),flush=True)
    except Exception as error:
        completion.update(error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-started)
        publish(directory,completion);raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('check','prepare','smoke','finite'),required=True);p.add_argument('--out-dir',required=True)
    inputs=p.add_mutually_exclusive_group();inputs.add_argument('--collection-dir');inputs.add_argument('--collection-archive')
    p.add_argument('--cache-dir');p.add_argument('--head-dir');p.add_argument('--plan-file');p.add_argument('--smoke-dir');p.add_argument('--device',default='cuda:0')
    return p


if __name__=='__main__':run(parser().parse_args())
