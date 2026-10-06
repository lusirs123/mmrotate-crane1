#!/usr/bin/env python3
"""Prepare new TRAIN support -> discarded smoke -> two fixed200-step arms.

No detector construction, feature extraction, TEST, policy or depth fitting.
Reuse fixed TRAIN/VAL ROIs, online size head and delivery protections.
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

sealed=old.sealed
VERSION=design.VERSION
BASE='crane_project/tools/'+VERSION
PROTOCOL=ROOT/(BASE+'_protocol.json')
SOURCES=ROOT/(BASE+'_sources.json')
NEW={BASE+'_protocol.json', BASE.replace('tools/','utils/')+'.py',
     'crane_project/tools/run_port_geometry_size_contrast_v1.py',
     'crane_project/tools/run_port_geometry_size_contrast_v1.sh',
     'tests/test_port_geometry_size_contrast_v1.py',
     'crane_project/tools/port_geometry_midpoint_edge_residual_v1_sources.json',support.SOURCES}


def protocol_document():
    return dict(protocol=VERSION, stages=['prepare','smoke','finite'], settings=design.SETTINGS,
        arms=list(design.ARMS), parameter_count=2818, fixed_checkpoint=sealed.FIXED,
        cache_manifest_sha256=sealed.CACHE_SHA, collection_artifacts=support.INPUT_FILES,
        sampling='TRAIN only. Per-video last floor(n/5) diagnostic probe, preceding10 indices purged; '
            'all legacy64 excluded from gradients and reported separately. Roles use identity order, never GT errors. '
            'Equal real/sim and standard/half slots; round-robin videos; alternate-step first slot '
            'prioritizes error strata within same video, global per-view error repeat cap4; others prefer correct frames.',
        objective='dual_log: existing unclipped mean two-edge SmoothL1 beta.1. dual_log_ratio: '
            'same plus .25 SmoothL1 of predicted vs GT-projected log-ratio residual, beta.1. '
            'Coefficient fixed before new VAL; no physical-ratio target or q=0.',
        comparison='Identical source/roles/schedule/initial weights/Adam/budget across two arms. '
            'Discard separate2-step smoke weights. Finite200 final only, no epoch sweep or retries. '
            'Full fixed M VAL replay precedes every update. Inherited geometry gates, plus probe '
            'edge MAE/P95 protection; results need human review; no winner automatically promoted.',
        scope=dict(detector_updates=0,frozen_midpoint_updates=0,feature_extractions=0,
            test_access=False,policy_modified=False,depth_formula_modified=False,
            dataset_splits_modified=False,labels_modified=False,automatic_promotion=False,
            formal_training=False),
        limitations=['TRAIN probe is exposed to original detector/M training, not independent generalization.',
            'Existing common-underscale support is scarce; no new images or labels are created.',
            'Physical reference remains unconfirmed; no real metric-depth guarantee.',
            'This tests supervision on the new sampler; does not isolate sampling-only causality.',
            'Reliability/depth benefits require separate matched downstream verification.'])


def checked_sources():
    parent=old.checked_sources()
    support_identity,_=support.checked_sources()
    manifest=sealed.read(SOURCES)
    required=set(parent['sources'])|set(support_identity['sources'])|NEW
    if (manifest['protocol']!=VERSION or set(manifest['sources'])!=required or
            sealed.read(PROTOCOL)!=protocol_document()):
        raise ValueError('New contrast source/protocol scope differs')
    for name,digest in manifest['sources'].items():
        if sealed.sha(ROOT/name)!=digest: raise ValueError('Contrast source SHA differs: '+name)
    return dict(protocol_sha256=sealed.sha(PROTOCOL),sources_sha256=sealed.sha(SOURCES),
        sources=manifest['sources'],parent_identity=parent,support_identity=support_identity)


def current_plan(args):
    rows,proof=support.load_collection(args.collection_dir,args.collection_archive)
    # No VAL row is supplied to role construction or error-aware sampling.
    train=[r for r in rows if r['reliability_role']=='train']
    legacy={r['image']:r['role'] for r in sealed.read(ROOT/support.SAMPLES)['samples']}
    return design.build_plan(train,legacy,support.sampling_map()),proof


def checked_plan(path,plan,proof,identity):
    if not path: raise ValueError('smoke/finite require prepared --plan-file')
    path=Path(path).resolve()
    index=sealed.read(path.parent/'artifacts.json')
    completion=sealed.read(path.parent/'completion.json')
    if (index['protocol']!=VERSION or completion['status']!='SIZE_CONTRAST_PREPARED' or
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


def load_records(cache_dir,cache,plan,torch,original,include_val):
    indexed={r['image']:r for r in plan['records']}
    train,val=[],[]
    bytes_read=0
    for label in ('train_s1','train_s05')+(('val_s1',) if include_val else ()):
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
    owned=[id(p) for group in optimizer.param_groups for p in group['params']]
    if len(owned)!=len(set(owned)) or set(owned)!={id(p) for p in head.parameters()}:
        raise ValueError('Optimizer may own only new independent size head')
    roi,mask,b,bm,xy,m,gt=old.batch(records,indices,device,torch)
    optimizer.zero_grad()
    output=head(roi,mask,b,bm,xy,m)
    terms=design.losses(output['delta_roi'],gt,m,bm,arm)
    if any(not bool(torch.isfinite(terms[k])) for k in ('total','dual','ratio')):
        raise ValueError('Nonfinite contrast loss')
    params=list(head.parameters())
    term_norms={}
    for name,weight in (('dual',1.),('ratio',terms['ratio_weight'])):
        grads=torch.autograd.grad(terms[name]*weight,params,retain_graph=True,allow_unused=True)
        term_norms[name+'_weighted_grad_norm']=math.sqrt(sum(float(g.detach().double().square().sum())
                                                          for g in grads if g is not None))
    terms['total'].backward()
    if (any(p.requires_grad or p.grad is not None for p in frozen.parameters()) or
        any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params)):
        raise ValueError('Frozen gradient or missing/nonfinite new gradient')
    norm=lambda values:math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in values))
    stem=norm(head.stem.parameters()); terminal=norm(list(head.u.parameters())+list(head.v.parameters()))
    before=float(torch.nn.utils.clip_grad_norm_(params,design.SETTINGS['clip_norm']))
    after=norm(params)
    optimizer.step()
    event=dict(loss=float(terms['total'].detach()),dual_loss=float(terms['dual'].detach()),
        ratio_loss=float(terms['ratio'].detach()),ratio_weight=terms['ratio_weight'],
        stem_grad_norm=stem,terminal_grad_norm=terminal,total_grad_norm_before_clip=before,
        total_grad_norm_after_clip=after,clipped=before>design.SETTINGS['clip_norm'],**term_norms)
    if any(not math.isfinite(v) for v in event.values()) or any(not bool(torch.isfinite(p).all()) for p in params):
        raise ValueError('Nonfinite gradient/update')
    return event


def save_reload(directory,stage,arm,head,optimizer,replay,proof,device,torch,model):
    path=directory/'experimental_head.pth'
    before=old.evaluate_rows(head,replay,device,torch)
    payload=dict(protocol=VERSION,stage=stage,arm=arm,
        updates=design.SETTINGS['smoke_steps' if stage=='smoke' else 'steps'],proof=proof,
        head_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()},
        head_digest=sealed.state_digest(head),optimizer_state=optimizer.state_dict(),
        rng=old.rng_state(torch),experimental_only=True,automatic_promotion=False)
    with path.open('xb') as stream: torch.save(payload,stream)
    restored=sealed.torch_load(torch,path)
    if not old.tree_equal(payload,restored,torch): raise ValueError('Checkpoint serialization differs')
    reloaded=model.IndependentEdgeResidualHead().to(device)
    reloaded.load_state_dict(restored['head_state'],strict=True)
    opt=torch.optim.Adam(reloaded.parameters(),lr=design.SETTINGS['lr'],weight_decay=0.)
    opt.load_state_dict(restored['optimizer_state']); old.restore_rng(restored['rng'],torch)
    if (sealed.state_digest(reloaded)!=payload['head_digest'] or
        not old.tree_equal(opt.state_dict(),optimizer.state_dict(),torch) or
        not old.tree_equal(restored['rng'],old.rng_state(torch),torch) or
        before!=old.evaluate_rows(reloaded,replay,device,torch)):
        raise ValueError('GPU prediction/optimizer/RNG replay differs')
    return reloaded,dict(path=path.name,sha256=sealed.sha(path),save_reload_exact=True,
                        head_digest=payload['head_digest'],updates=payload['updates'])


def checked_smoke(directory,identity,proof):
    if not directory: raise ValueError('finite requires successful --smoke-dir')
    directory=Path(directory).resolve()
    completion=sealed.read(directory/'completion.json')
    if (completion['protocol']!=VERSION or completion['status']!='SIZE_CONTRAST_SMOKE_COMPLETE' or
        completion['source_identity']!=identity or completion['proof']!=proof or
        completion['val_tensor_access'] is not False or completion['arms']!=list(design.ARMS)):
        raise ValueError('Matching no-VAL smoke required')
    index=sealed.read(directory/'artifacts.json')
    if index['protocol']!=VERSION or index['files'].get('completion.json')!=sealed.sha(directory/'completion.json'):
        raise ValueError('Smoke completion SHA differs')
    results={}
    for arm in design.ARMS:
        path=directory/arm
        report=sealed.read(path/'completion.json')
        artifacts=sealed.read(path/'artifacts.json')
        if artifacts['protocol']!=VERSION: raise ValueError('Smoke arm protocol differs')
        for name in ('completion.json','experimental_head.pth'):
            if artifacts['files'].get(name)!=sealed.sha(path/name): raise ValueError('Smoke arm artifact differs')
        if (report['protocol']!=VERSION or report['stage']!='smoke' or report['arm']!=arm or
            report['proof']!=proof or report['updates']!=2 or not all(report['engineering'].values())):
            raise ValueError('Smoke arm engineering failed')
        results[arm]=report
    return results


def extra_probe_gates(summary,gate):
    # Same frozen acceptance gate as v1; add continuous probe protection.
    for shard in ('train_s1','train_s05'):
        for domain in ('real','sim'):
            group=summary[shard]['probe'][domain]
            for edge in ('long','short'):
                for metric in ('mae','p95'):
                    a=group['midpoint']['edges'][edge][metric]; b=group['edge_residual']['edges'][edge][metric]
                    gate['checks'][shard+'/'+domain+'/probe_'+edge+'_'+metric]=(
                        a is None and b is None) or (a is not None and b is not None and b<=a+1e-6)
    gate['failed_checks']=[k for k,v in gate['checks'].items() if not v]
    gate['finite_joint_gate_pass']=not gate['failed_checks']
    gate['status']='FINITE_JOINT_GATE_PASS_REVIEW_REQUIRED' if not gate['failed_checks'] else 'FINITE_JOINT_GATE_FAIL_KEEP_M'
    return gate


def run(args):
    identity=checked_sources()
    plan,input_proof=current_plan(args)
    directory=Path(args.out_dir).resolve()
    # Avoid nesting new output in frozen/cache/checkpoint/source inputs.
    protected=[ROOT/'crane_project',ROOT/'tests',ROOT/'docs',Path(args.head_dir).resolve()]
    if args.cache_dir: protected.append(Path(args.cache_dir).resolve())
    if args.collection_dir: protected.append(Path(args.collection_dir).resolve())
    if args.plan_file: protected.append(Path(args.plan_file).resolve().parent)
    if args.smoke_dir: protected.append(Path(args.smoke_dir).resolve())
    if any(p==directory or p in directory.parents for p in protected):
        raise ValueError('Output must be separate from all fixed/prepared inputs')
    directory.mkdir(parents=True,exist_ok=False)
    completion=dict(protocol=VERSION,stage=args.stage,status='SIZE_CONTRAST_FAILED',
        started_utc=datetime.now(timezone.utc).isoformat(),source_identity=identity,
        val_tensor_access=False,arms=list(design.ARMS),**protocol_document()['scope'])
    started=time.monotonic()
    try:
        sealed.write(directory/'protocol.json',protocol_document())
        sealed.write(directory/'source_identity.json',identity)
        sealed.write(directory/'input_identity.json',input_proof)
        if args.stage=='prepare':
            sealed.write(directory/'plan.json',plan)
            completion.update(status='SIZE_CONTRAST_PREPARED',role_counts=plan['role_counts'],
                              model_loaded=False,parameter_updates=0)
            publish(directory,completion)
            print(completion['status'],plan['role_counts'],flush=True)
            return
        checked_plan(args.plan_file,plan,input_proof,identity)
        cache_dir=sealed.resolve_cache(args.cache_dir)
        cache=sealed.checked_cache(cache_dir,identity['parent_identity']['parent_identity'])
        head_dir=Path(args.head_dir).resolve()
        selection,references,indexed=sealed.checked_selection(head_dir,identity['parent_identity']['parent_identity'])
        import torch
        from crane_project.utils import port_geometry_midpoint_sigma_v1 as frozen_model
        from crane_project.utils import port_geometry_midpoint_edge_residual_v1 as model
        if model.SETTINGS!=old.MODEL_SETTINGS or model.PARAMETER_COUNT!=2818:
            raise ValueError('Existing online size model contract changed')
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
        train,val,bytes_read=load_records(cache_dir,cache,plan,torch,frozen_model.original,args.stage=='finite')
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
        if args.stage=='finite' and len(val)!=887: raise ValueError('Full fixed VAL replay required before updates')
        records=train+val
        results={}; baseline_rows=None; initial_digest=None; arm_summaries={}
        for arm in design.ARMS:
            arm_dir=directory/arm; arm_dir.mkdir()
            old.seed_all(torch)
            head=model.IndependentEdgeResidualHead().to(device)
            optimizer=torch.optim.Adam(head.parameters(),lr=design.SETTINGS['lr'],weight_decay=0.)
            initial=sealed.state_digest(head)
            if initial_digest is None: initial_digest=initial
            if initial!=initial_digest or (smoke and (initial!=smoke[arm]['initial_digest'] or
                initial==smoke[arm]['final_digest'])):
                raise ValueError('Arms must share neutral init and discard smoke state')
            if baseline_rows is None:
                baseline_rows=old.evaluate_rows(head,records,device,torch)
                if any(r['midpoint']!=r['edge_residual'] for r in baseline_rows):
                    raise ValueError('Neutral head changed M')
                old.write_rows(directory/'baseline_rows.jsonl',baseline_rows)
                if val:
                    baseline_summary=old.metrics.evaluate(baseline_rows)
                    for domain,counts in {'real':(374,362,199),'sim':(512,512,464)}.items():
                        value=baseline_summary['val_s1']['all'][domain]['midpoint']
                        if tuple(value[k]['numerator'] for k in ('output_coverage','center_correct_full_frame',
                            'joint_size10_full_frame'))!=counts: raise ValueError('Fixed M VAL counts differ')
                    sealed.write(directory/'baseline_summary.json',baseline_summary)
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
            final_rows=old.evaluate_rows(head,records,device,torch)
            engineering=dict(neutral_exact=True,gpu_save_reload=True,
                frozen_midpoint_state=sealed.state_digest(frozen)==frozen_before==sealed.FIXED['head_digest'],
                frozen_midpoint_no_grad=all(not p.requires_grad and p.grad is None for p in frozen.parameters()),
                whole_run_frozen_identity=all(old.metrics.frozen_identity(r) for r in final_rows),
                independent_parameter_count=sealed.state_digest(head)['parameter_count']==2818,
                first_stem_zero=gradients[0]['stem_grad_norm']==0.,first_terminal_learns=gradients[0]['terminal_grad_norm']>0.,
                subsequent_stem_learns=gradients[1]['stem_grad_norm']>0.,
                fixed_head_weight_sha=sealed.sha(head_dir/sealed.FIXED['path'])==sealed.FIXED['sha256'],
                cache_manifest_sha=sealed.sha(cache_dir/'cache_manifest.json')==sealed.CACHE_SHA,
                source_closure=checked_sources()==identity,
                fixed_val_replay=len(val)==887 if val else True,
                smoke_not_reused=initial==smoke[arm]['initial_digest'] if smoke else True,
                final_budget=count==200 if val else count==2,
                only_new_fit_gradients=all(train[i]['sample_role']=='fit' for b in batches[:count] for i in b))
            if not all(engineering.values()): raise ValueError('Engineering protection failed')
            old.write_rows(arm_dir/'final_rows.jsonl',final_rows)
            gate=None
            if val:
                summary=old.metrics.evaluate(final_rows)
                arm_summaries[arm]=summary
                gate=extra_probe_gates(summary,old.metrics.finite_gates(summary,engineering))
                sealed.write(arm_dir/'summary.json',summary); sealed.write(arm_dir/'finite_gate.json',gate)
                # Continuous short-scale and ratio summaries are reported, not depth estimates.
                rows=[dict(image=r['image'],role=r['sample_role'],sequence=r['sequence'],scale=r['scale'],
                    midpoint=support.d.size_errors(r['gt'],r['midpoint']),
                    corrected=support.d.size_errors(r['gt'],r['edge_residual'])) for r in final_rows]
                sealed.write(arm_dir/'continuous_sizes.json',dict(records=rows))
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
                identical_initialization=True,identical_schedule=True,automatic_promotion=False,
                downstream_benefits_verified=False))
        if current_plan(args)!=(plan,input_proof): raise ValueError('Collection/plan changed during run')
        completion.update(status='SIZE_CONTRAST_SMOKE_COMPLETE' if not val else 'SIZE_CONTRAST_FINITE_COMPLETE_REVIEW_REQUIRED',
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
    p.add_argument('--stage',choices=('prepare','smoke','finite'),required=True)
    p.add_argument('--out-dir',required=True)
    inputs=p.add_mutually_exclusive_group()
    inputs.add_argument('--collection-dir'); inputs.add_argument('--collection-archive')
    p.add_argument('--cache-dir'); p.add_argument('--head-dir',default=str(ROOT/sealed.DEFAULT_HEAD))
    p.add_argument('--device',default='cuda:0'); p.add_argument('--plan-file'); p.add_argument('--smoke-dir')
    return p


if __name__=='__main__': run(parser().parse_args())
