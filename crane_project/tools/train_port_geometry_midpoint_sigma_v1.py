#!/usr/bin/env python3
"""Fixed original-midpoint sigma sensitivity: .5/1/1.5, full TRAIN/full VAL.

Reuse reviewed sigma=1 text/checkpoint, fit only .5/1.5 for24 epochs each.
Reuse the original CPU ROI cache. No detector construction, extraction or TEST.
"""
import argparse
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import train_port_geometry_midpoint_formal_v1 as f
from crane_project.utils import port_geometry_midpoint_sigma_v1 as model

torch, g = f.torch, f.g
VERSION = 'port_geometry_midpoint_sigma_v1'
PROTOCOL = ROOT/'crane_project/tools/port_geometry_midpoint_sigma_v1_protocol.json'
SOURCES = ROOT/'crane_project/tools/port_geometry_midpoint_sigma_v1_sources.json'
NEW_SIGMAS = (.5, 1.5)
SMOKE_STATUS = 'SIGMA_SMOKE_SAVE_RELOAD_PASS_DISCARDED'


def sha(path):
    return g.ready.sha(Path(path))


def read(path):
    return json.loads(Path(path).read_text(),
        parse_constant=lambda v: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+v)))


def write(path, value):
    g.write_new(Path(path), value)


def label(sigma):
    return 'sigma_'+str(float(sigma)).replace('.', 'p')


def protocol_document():
    return dict(protocol=VERSION, sigmas=list(model.SIGMAS), new_sigmas=list(NEW_SIGMAS),
        formal_settings=deepcopy(f.SETTINGS), original_midpoint_settings=deepcopy(f.m.SETTINGS),
        parameter_count=f.m.PARAMETER_COUNT, selection_config=deepcopy(f.SELECTION_CONFIG),
        variable='Only B-only Gaussian prior sigma_cells .5/1/1.5, with matching neutral expectation. Sigma=1 forward delegates exactly to original. Never mutate old SETTINGS.',
        fairness='Same seed1703 zero-output initialization, full5116 TRAIN views, eligible domain-balanced epoch schedule, Adam.001,24 epochs per condition. Reuse reviewed original sigma=1; train only .5/1.5 from scratch, never from selected ep23.',
        cache='Original TRAIN1/.5 and VAL1 CPU ROI cache,3 shards. No B loading/inference/extraction or cache rebuilding. Training two scales; main full887-frame VAL one standard scale.',
        smoke='Each new sigma:2 updates, save/reload exact head/optimizer/RNG/inference, one continuation on each copy;4 updates discarded. No detector online call; old coordinate/sampler/decoder unchanged.',
        selection='All24 complete full-VAL checkpoints under unchanged original rule per sigma. Report all three selected results and all epoch tables. No automatic cross-sigma winner, no averaging/TEST/reselection.',
        reporting='Primary real/sim coverage, output-frame R_center, full-frame center correctness, mean RIoU, sim penalty A-RMSE, DFR,ACI,TDR,MCML,defined MRF. Keep historical R_center key all-frame meaning. Standard-scale VAL only; residuals supplementary.',
        checkpoint='Separate new sigma protocol, identity and explicit parameter metadata in every checkpoint. Refuse wrong sigma/proof/source/cache. Stream atomic no-overwrite IO and reload verification inherited.',
        reuse='Require completed original24-epoch sources/cache/runtime/support, all24 VAL summaries, budget/log/source/selection consistency, indexed selected rows and checkpoint. Reproduce selected sigma=1 cached VAL before new optimization. Do not silently retrain on mismatch.',
        scope=dict(new_formal_conditions=2,reused_conditions=1,new_epochs=48,reused_epochs=24,
            smoke_updates_total=8,detector_updates=0,detector_forward_calls=0,feature_extractions=0,
            cache_rebuilt=False,test_access=False,selection_on_test=False,automatic_promotion=False),
        limitations=['Single fixed seed sensitivity, not significance or global optimality.',
            'Main VAL is standard-scale, training also uses half-scale. Not a TEST sweep.',
            'Original failed finite gates and measured ep23 results retain their identity.',
            'TEST repeatedly exposed; never select sigma/epoch/threshold on TEST.',
            'Isotropic transforms, sx/sy restoration, raw w/h-angle/depth interface retained; depth accuracy unverified.'])


def checked_sources():
    parent = f.checked_sources()
    fixed = read(SOURCES)
    required = set(parent['sources']) | {
        str(f.SOURCES.relative_to(ROOT)), str(PROTOCOL.relative_to(ROOT)),
        'crane_project/utils/port_geometry_midpoint_sigma_v1.py',
        'crane_project/tools/train_port_geometry_midpoint_sigma_v1.py',
        'tests/test_port_geometry_midpoint_sigma_v1.py'}
    if (fixed['protocol'] != VERSION or set(fixed['sources']) != required or
            fixed['parent_sources_sha256'] != sha(f.SOURCES) or
            read(PROTOCOL) != protocol_document()):
        raise ValueError('Fixed sigma protocol/source scope differs')
    for name,digest in fixed['sources'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Sigma source SHA differs: '+name)
    return dict(protocol_sha256=sha(PROTOCOL),sources_sha256=sha(SOURCES),
        sources=fixed['sources'],parent_identity=parent)


def checked_cache_metadata(directory, parent):
    path = Path(directory)/'cache_manifest.json'; cache = read(path)
    if (cache['protocol'] != f.VERSION or cache['identity'] != parent or
            cache['status'] != 'COMPLETE_FROZEN_B_TRAIN_VAL_CACHE' or
            cache['record_counts'] != {'train_s1':2558,'train_s05':2558,'val_s1':887} or
            set(cache['files']) != {'train_s1.pt','train_s05.pt','val_s1.pt'} or
            cache['feature_extractions'] != 6003 or cache['native_head_calls'] != 18009 or
            cache['detector_updates'] != 0 or cache['test_access'] is not False or
            cache['tensor_bytes'] > f.SETTINGS['max_cpu_tensor_bytes']):
        raise ValueError('Original completed TRAIN/VAL cache metadata differs')
    cache['manifest_sha256'] = sha(path)
    return cache


def audit_baseline(directory, parent, cache):
    """Audit reuse from indexed text; weights are hashed, never deserialized here."""
    directory = Path(directory); index = read(directory/'artifacts.json')
    if index['protocol'] != f.VERSION:
        raise ValueError('Requires original formal baseline')
    selection = read(directory/'selection.json'); complete = read(directory/'completion.json')
    names = {'head_epoch_%02d.pth'%e for e in range(1,25)}
    if (selection['protocol'] != f.VERSION or selection['identity'] != parent or
            selection['split'] != 'val' or selection['test_access'] is not False or
            selection['selection_on_test'] is not False or
            selection['selection_config'] != f.SELECTION_CONFIG or
            set(selection['all_checkpoints']) != names or
            complete['protocol'] != f.VERSION or
            complete['status'] != 'FORMAL_MIDPOINT_TRAIN_VAL_COMPLETE_REVIEW_REQUIRED' or
            complete['epochs_completed'] != 24 or complete['identity'] != parent or
            complete['detector_updates'] != 0 or complete['test_access'] is not False or
            complete['runtime'] != cache['runtime'] or complete['selection'] != selection or
            complete['cache_manifest_sha256'] != cache['manifest_sha256'] or
            selection['cache_manifest_sha256'] != cache['manifest_sha256']):
        raise ValueError('Baseline complete training/cache/runtime/selection differs')
    selected = selection['selected_checkpoint']; chosen,_,info = f.select_best_checkpoint(
        selection['all_checkpoints'], f.SELECTION_CONFIG)
    if chosen != selected['path'] or info != selection['selection_info'] or chosen not in names:
        raise ValueError('Baseline original VAL checkpoint choice differs')
    paths = {'completion.json','selection.json','selected_val_compare.json','val_baseline.json',
        'progress.jsonl','val_epoch_%02d.rows.jsonl'%selected['epoch'],chosen}
    paths.update('val_epoch_%02d.json'%e for e in range(1,25))
    files = {}
    for name in sorted(paths):
        digest = sha(directory/name)
        if index['files'].get(name) != digest:
            raise ValueError('Baseline artifact SHA differs: '+name)
        files[name] = digest
    if files[chosen] != selected['sha256']:
        raise ValueError('Baseline selected weight SHA differs')
    epochs = {}
    for epoch in range(1,25):
        name = 'head_epoch_%02d.pth'%epoch
        entry = selection['all_checkpoints'][name]; checkpoint = entry['checkpoint']
        value = read(directory/('val_epoch_%02d.json'%epoch))
        if (value['split'] != 'val' or checkpoint['epoch'] != epoch or
                checkpoint['path'] != name or checkpoint['save_reload_exact'] is not True or
                index['files'].get(name) != checkpoint['sha256'] or
                entry['metrics'] != value['groups']['overall']['midpoint']['metric_protocol_v2']):
            raise ValueError('Baseline epoch checkpoint/VAL differs')
        epochs[name] = value
    if (selection['all_checkpoints'][chosen]['checkpoint'] != selected or
            epochs[chosen] != read(directory/'selected_val_compare.json')):
        raise ValueError('Baseline selected VAL summary differs')
    logs = [json.loads(z) for z in (directory/'progress.jsonl').read_text().splitlines() if z.strip()]
    if Counter(z['stage'] for z in logs) != Counter(train=complete['updates'],epoch_complete=24):
        raise ValueError('Baseline update event budget differs')
    actual = 0; epoch_steps = []
    for epoch in range(1,25):
        steps = [z for z in logs if z['stage']=='train' and z['epoch']==epoch]
        endings = [z for z in logs if z['stage']=='epoch_complete' and z['epoch']==epoch]
        if not steps or len(endings)!=1 or [z['step'] for z in steps] != list(range(1,len(steps)+1)):
            raise ValueError('Baseline epoch log sequence differs')
        for z in steps:
            actual += 1
            if (z['updates'] != actual or z['epoch_steps'] != len(steps) or
                    not all(math.isfinite(z[k]) for k in ('loss','grad_norm_before_clip','grad_norm_after_clip','stem_grad_norm','output_grad_norm'))):
                raise ValueError('Baseline training log accounting differs')
        checkpoint = selection['all_checkpoints']['head_epoch_%02d.pth'%epoch]['checkpoint']
        if (checkpoint['updates'] != actual or endings[0]['updates'] != actual or
                endings[0]['checkpoint'] != checkpoint or
                endings[0]['selection_metrics'] != selection['all_checkpoints'][checkpoint['path']]['metrics']):
            raise ValueError('Baseline epoch budget/selection log differs')
        epoch_steps.append(len(steps))
    if actual != complete['updates'] or complete['updates'] != selection['all_checkpoints']['head_epoch_24.pth']['checkpoint']['updates']:
        raise ValueError('Baseline final update count differs')
    rows = [json.loads(z) for z in (directory/('val_epoch_%02d.rows.jsonl'%selected['epoch'])).read_text().splitlines() if z.strip()]
    if (len(rows)!=887 or Counter(r['sequence'] for r in rows)!=Counter(g.ready.VAL_COUNTS) or
            len({r['image'] for r in rows})!=887 or any(r['scale']!=1. for r in rows)):
        raise ValueError('Baseline selected full VAL rows differ')
    value = read(directory/'selected_val_compare.json')
    if f.summaries(rows) != value:
        raise ValueError('Baseline saved full VAL rows/summary differ')
    return dict(completion=complete,selection=selection,value=value,rows=rows,
        epoch_steps=epoch_steps,files=files,epochs=epochs)


def checked_contract(baseline_dir, cache_dir):
    identity = checked_sources()
    cache = checked_cache_metadata(cache_dir, identity['parent_identity'])
    baseline = audit_baseline(baseline_dir, identity['parent_identity'], cache)
    proof = dict(identity=identity,cache_manifest_sha256=cache['manifest_sha256'],
        baseline_files=baseline['files'],baseline_selected=baseline['selection']['selected_checkpoint'])
    return identity,cache,baseline,proof


def save_checkpoint(path, head, optimizer, epoch, updates, proof, cache):
    payload = dict(protocol=VERSION,proof=proof,sigma_cells=head.prior_sigma_cells,
        epoch=epoch,updates=updates,head_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()},
        optimizer_state=deepcopy(optimizer.state_dict()),rng=f.rng_state(),
        head_digest=g.state_digest(head),context=f.checkpoint_context(cache),frozen_b=g.ready.FROZEN_B)
    f.atomic_save(Path(path),payload)
    if not f.tree_equal(payload,torch.load(str(path),map_location='cpu')):
        raise ValueError('Sigma checkpoint exact reload differs')
    return dict(path=Path(path).name,sha256=sha(path),epoch=epoch,updates=updates,
        sigma_cells=head.prior_sigma_cells,head_digest=payload['head_digest'],save_reload_exact=True)


def load_head(path, sigma, proof, cache, device):
    payload = torch.load(str(path),map_location='cpu')
    if (payload['protocol'] != VERSION or payload['proof'] != proof or
            payload['sigma_cells'] != model.checked_sigma(sigma) or
            payload['context'] != f.checkpoint_context(cache) or payload['frozen_b'] != g.ready.FROZEN_B):
        raise ValueError('Sigma checkpoint parameter/source/cache differs')
    head = model.SigmaMidpointHead(sigma).to(device)
    head.load_state_dict(payload['head_state'],strict=True)
    if g.state_digest(head) != payload['head_digest']:
        raise ValueError('Sigma checkpoint head digest differs')
    return head,payload


def verify_loaded_inputs(train, val, baseline):
    if f.support_report(train) != baseline['completion']['train_support']:
        raise ValueError('Original TRAIN support differs')
    schedules = [f.epoch_batches(train,e) for e in range(1,25)]
    if [len(z) for z in schedules] != baseline['epoch_steps']:
        raise ValueError('Original24-epoch budget differs')
    f.seed_all(); reference = f.m.SpatialMidpointHead()
    if g.state_digest(reference) != baseline['completion']['initial_state']:
        raise ValueError('Original initialization differs')
    initial = deepcopy(reference.state_dict()); del reference
    digest = hashlib.sha256(json.dumps(schedules,separators=(',',':')).encode()).hexdigest()
    return initial,dict(epoch_steps=baseline['epoch_steps'],schedule_sha256=digest,
        initial_state=baseline['completion']['initial_state'])


def replay_default(baseline_dir, val, baseline, parent, cache, device):
    selected = baseline['selection']['selected_checkpoint']
    path = Path(baseline_dir)/selected['path']
    payload = torch.load(str(path),map_location='cpu')
    if (payload['protocol']!=f.VERSION or payload['identity']!=parent or
            payload['epoch']!=selected['epoch'] or payload['updates']!=selected['updates'] or
            payload['frozen_b']!=g.ready.FROZEN_B or
            payload['context']!=f.checkpoint_context(cache) or payload['head_digest']!=selected['head_digest']):
        raise ValueError('Reused default checkpoint metadata differs')
    head = model.SigmaMidpointHead(1.).to(device); head.load_state_dict(payload['head_state'],strict=True)
    if g.state_digest(head)!=selected['head_digest']:
        raise ValueError('Reused default head digest differs')
    rows = f.evaluate(head,val,device)
    if rows != baseline['rows'] or f.summaries(rows)!=baseline['value']:
        raise ValueError('Default sigma1 full-VAL replay differs')
    del head,payload
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return dict(sigma_cells=1.,selected_checkpoint=selected,frames=len(rows),exact_cached_val_replay=True)


def tracked_update(head, optimizer, records, indices, device, progress, context):
    """Reuse original math; count a completed step even if post-step checks fail."""
    original_step = optimizer.step
    def counted_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        progress(dict(context, stage='optimizer_step'))
        return result
    optimizer.step = counted_step
    try:
        return f.update(head, optimizer, records, indices, device)
    finally:
        optimizer.step = original_step


def smoke_condition(sigma, train, val, initial, proof, cache, device, out, progress):
    f.seed_all(); head = model.SigmaMidpointHead(sigma).to(device)
    head.load_state_dict(initial,strict=True); optimizer=f.optimizer_for(head)
    if any(r['b']!=r['midpoint'] for r in f.evaluate(head,f.smoke_views(val),device)):
        raise ValueError('Sigma zero initialization not exact B')
    batches=f.epoch_batches(train,1); logs=[]
    for i in range(2):
        logs.append(tracked_update(head,optimizer,train,batches[i%len(batches)],device,
            progress,dict(sigma_cells=sigma,step=i+1,phase='smoke')))
    if logs[0]['output_grad_norm']<=0 or logs[1]['stem_grad_norm']<=0:
        raise ValueError('Sigma initial gradients ineffective')
    checkpoint=save_checkpoint(out/'discarded_smoke_head.pth',head,optimizer,0,2,proof,cache)
    replay,payload=load_head(out/checkpoint['path'],sigma,proof,cache,device)
    opt=f.optimizer_for(replay); opt.load_state_dict(payload['optimizer_state'])
    first=f.evaluate(head,f.smoke_views(val),device); second=f.evaluate(replay,f.smoke_views(val),device)
    if first!=second:raise ValueError('Sigma smoke inference reload differs')
    f.restore_rng(payload['rng']); next_a=tracked_update(head,optimizer,train,batches[0],device,
        progress,dict(sigma_cells=sigma,step=3,phase='smoke'))
    f.restore_rng(payload['rng']); next_b=tracked_update(replay,opt,train,batches[0],device,
        progress,dict(sigma_cells=sigma,step=4,phase='smoke'))
    if (next_a!=next_b or g.state_digest(head)!=g.state_digest(replay) or
            not f.tree_equal(optimizer.state_dict(),opt.state_dict())):
        raise ValueError('Sigma smoke optimizer/RNG continuation differs')
    return dict(sigma_cells=sigma,checkpoint=checkpoint,updates_executed=4,discarded=True,
        inference_reload_exact=True,optimizer_continuation_exact=True,logs=logs)


def train_condition(sigma, train, val, initial, proof, cache, baseline, schedule, device, out, progress):
    f.seed_all(); head=model.SigmaMidpointHead(sigma).to(device); head.load_state_dict(initial,strict=True)
    if g.state_digest(head)!=schedule['initial_state']:
        raise ValueError('Sigma full-training initialization differs')
    optimizer=f.optimizer_for(head); initial_rows=f.evaluate(head,val,device)
    if any(r['b']!=r['midpoint'] for r in initial_rows):
        raise ValueError('Sigma initial full VAL not exact B')
    baseline_value=f.summaries(initial_rows)
    if any(baseline_value['groups'][key]['b']!=baseline['value']['groups'][key]['b'] for key in baseline_value['groups']):
        raise ValueError('Sigma paired frozen B VAL differs')
    write(out/'val_baseline.json',baseline_value)
    entries={}; summaries={}; updates=0
    for epoch in range(1,25):
        batches=f.epoch_batches(train,epoch); losses=[]
        for step,indices in enumerate(batches,1):
            value=tracked_update(head,optimizer,train,indices,device,progress,
                dict(sigma_cells=sigma,epoch=epoch,step=step,updates=updates+1,phase='train'))
            updates+=1;losses.append(value['loss'])
            progress(dict(stage='train',sigma_cells=sigma,epoch=epoch,step=step,
                epoch_steps=len(batches),updates=updates,**value))
        checkpoint=save_checkpoint(out/('head_epoch_%02d.pth'%epoch),head,optimizer,epoch,updates,proof,cache)
        rows=f.evaluate(head,val,device);value=f.summaries(rows)
        write(out/('val_epoch_%02d.json'%epoch),value)
        with (out/('val_epoch_%02d.rows.jsonl'%epoch)).open('x') as stream:
            for row in rows:stream.write(json.dumps(g.json_native(row),ensure_ascii=False,allow_nan=False)+'\n')
        name=checkpoint['path'];entries[name]=dict(checkpoint=checkpoint,
            metrics=value['groups']['overall']['midpoint']['metric_protocol_v2']);summaries[name]=value
        progress(dict(stage='epoch_complete',sigma_cells=sigma,epoch=epoch,updates=updates,
            loss_mean=sum(losses)/len(losses),checkpoint=checkpoint,selection_metrics=entries[name]['metrics']))
        print('SIGMA',sigma,'epoch',epoch,'updates',updates,'point_loss',sum(losses)/len(losses),flush=True)
    if updates!=sum(schedule['epoch_steps']):raise ValueError('Sigma full update budget differs')
    chosen,_,info=f.select_best_checkpoint(entries,f.SELECTION_CONFIG)
    if chosen is None:raise ValueError('No full-VAL checkpoint selected')
    del head,optimizer
    head,payload=load_head(out/chosen,sigma,proof,cache,device)
    rows=f.evaluate(head,val,device);value=f.summaries(rows)
    if value!=summaries[chosen]:raise ValueError('Selected sigma head reload full VAL differs')
    selection=dict(protocol=VERSION,sigma_cells=sigma,proof=proof,split='val',
        cache_manifest_sha256=cache['manifest_sha256'],selected_checkpoint=entries[chosen]['checkpoint'],
        selection_config=deepcopy(f.SELECTION_CONFIG),selection_info=info,all_checkpoints=entries,
        comparison=f.comparison_checks(value),test_access=False,selection_on_test=False,automatic_promotion=False)
    write(out/'selection.json',selection);write(out/'selected_val_compare.json',value)
    complete=dict(protocol=VERSION,status='SIGMA_FULL_TRAIN_VAL_COMPLETE_REVIEW_REQUIRED',sigma_cells=sigma,
        proof=proof,epochs_completed=24,updates=updates,schedule=schedule,selection=selection,
        train_support=f.support_report(train),detector_updates=0,test_access=False,automatic_promotion=False)
    write(out/'completion.json',complete);publish(out)
    del head,payload
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return dict(selection=selection,value=value,updates=updates)


def report_rows(values, selected_epochs):
    """Main table one standard VAL scale; three center denominators stay distinct."""
    result=[]
    conditions=[(None,'b',values[1.])] + [(s,'midpoint',values[s]) for s in model.SIGMAS]
    for sigma,method,value in conditions:
        for domain in ('real','sim'):
            summary=value['groups'][domain][method];metrics=summary['metric_protocol_v2']
            row=dict(method=method,sigma_cells=sigma,selected_epoch=24 if method=='b' else selected_epochs[sigma],
                origin='frozen_b_reference' if method=='b' else ('reused' if sigma==1. else 'new'),
                split='val',scale=1.,domain=domain,
                frames=summary['frames'],output_frames=summary['output_frames'],
                output_coverage_pct=summary['output_coverage_pct'],
                output_frame_r_center_pct=summary['output_center_hit_pct'],
                full_frame_center_correct_pct=summary['all_frame_center_hit_pct'],
                center_correct_count=summary['conditional_center_correct_fraction']['numerator'],
                mean_riou=metrics[domain+'/mean_RIoU'],a_rmse_deg=metrics.get(domain+'/A-RMSE(deg)'),
                dfr_pct_per_frame=metrics[domain+'/DFR(%/frame)'],aci=metrics[domain+'/ACI'],
                tdr_w10_pct=metrics[domain+'/TDR_w10(%)'],mcml_max_frames=metrics[domain+'/MCML_max(frames)'],
                mcml_mean_frames=metrics[domain+'/MCML_mean(frames)'],mcml_pass=metrics[domain+'/MCML_pass(limit=5)'],
                mrf_frames=metrics.get(domain+'/MRF(frames)'))
            result.append(row)
    return result


def publish(out):
    out=Path(out)
    write(out/'artifacts.json',dict(protocol=VERSION,
        files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='artifacts.json'}))


def checked_smoke(path, proof, schedule):
    path=Path(path);value=read(path)
    index=read(path.parent/'artifacts.json')
    if index['files'].get(path.name)!=sha(path):raise ValueError('Sigma smoke artifact SHA differs')
    if (value['protocol']!=VERSION or value['status']!=SMOKE_STATUS or value['proof']!=proof or
            value['schedule']!=schedule or value['head_updates_total']!=8 or value['test_access'] is not False or
            value['detector_updates']!=0 or value['default_replay']['exact_cached_val_replay'] is not True or
            set(value['conditions'])!={label(s) for s in NEW_SIGMAS}):
        raise ValueError('Requires fresh successful discarded two-sigma smoke')
    for sigma in NEW_SIGMAS:
        arm=value['conditions'][label(sigma)]
        if (arm['sigma_cells']!=sigma or arm['discarded'] is not True or arm['updates_executed']!=4 or
                arm['inference_reload_exact'] is not True or arm['optimizer_continuation_exact'] is not True or
                arm['checkpoint']['sigma_cells']!=sigma or
                sha(path.parent/label(sigma)/arm['checkpoint']['path'])!=arm['checkpoint']['sha256']):
            raise ValueError('Discarded sigma smoke checkpoint differs')
    return value


def run(args):
    out=Path(args.out_dir)
    if any(out.resolve()==Path(p).resolve() or Path(p).resolve() in out.resolve().parents
           for p in (args.baseline_dir,args.cache_dir)):
        raise ValueError('Output must be separate from immutable baseline/cache inputs')
    try:out.mkdir(parents=True,exist_ok=False)
    except FileExistsError:
        raise FileExistsError('Output exists; preserve results and use a new --out-dir suffix.') from None
    report=dict(protocol=VERSION,stage=args.stage,status='STARTED',head_updates_total=0,
        detector_updates=0,detector_forward_calls=0,feature_extractions=0,cache_tensor_loads=0,
        test_access=False,selection_on_test=False,automatic_promotion=False,selected_sigma=None)
    started=time.perf_counter()
    with (out/'progress.jsonl').open('x') as stream:
        def progress(value):
            if value['stage']=='optimizer_step':report['head_updates_total']+=1
            stream.write(json.dumps(g.json_native(value),ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
        try:
            identity,cache,baseline,proof=checked_contract(args.baseline_dir,args.cache_dir)
            report['proof']=proof
            write(out/'protocol.json',protocol_document());write(out/'sources.json',read(SOURCES))
            write(out/'reuse_sources.json',dict(baseline_dir=str(Path(args.baseline_dir)),files=baseline['files']))
            if args.stage=='check':
                report['status']='STATIC_SIGMA_CONTRACT_PASS_NO_TENSOR_LOAD_NO_GPU_NO_UPDATES'
            else:
                runtime=f.runtime(args.gpu)
                if runtime!=cache['runtime']:raise ValueError('Original cache/runtime/GPU differs')
                report['runtime']=runtime
                train,val,loaded=f.load_cache(args.cache_dir,identity['parent_identity'])
                loaded['manifest_sha256']=sha(Path(args.cache_dir)/'cache_manifest.json')
                if loaded!=cache:raise ValueError('Loaded cache metadata differs')
                report['cache_tensor_loads']=3
                initial,schedule=verify_loaded_inputs(train,val,baseline);report['schedule']=schedule
                if args.stage=='train':
                    if not args.smoke_report:raise ValueError('--smoke-report required')
                    checked_smoke(args.smoke_report,proof,schedule)
                    report['smoke_report_sha256']=sha(args.smoke_report)
                report['default_replay']=replay_default(args.baseline_dir,val,baseline,
                    identity['parent_identity'],cache,'cuda:'+str(args.gpu))
                device='cuda:'+str(args.gpu);report['conditions']={}
                if args.stage=='smoke':
                    for sigma in NEW_SIGMAS:
                        arm=out/label(sigma);arm.mkdir()
                        value=smoke_condition(sigma,train,val,initial,proof,cache,device,arm,progress)
                        report['conditions'][label(sigma)]=value
                        write(arm/'completion.json',value);publish(arm)
                    if report['head_updates_total']!=8:raise ValueError('Sigma smoke budget differs')
                    report['status']=SMOKE_STATUS
                else:
                    values={1.:baseline['value']};epoch_values={'sigma_1p0':baseline['epochs']}
                    selected_epochs={1.:baseline['selection']['selected_checkpoint']['epoch']}
                    for sigma in NEW_SIGMAS:
                        arm=out/label(sigma);arm.mkdir()
                        torch.cuda.reset_peak_memory_stats(args.gpu)
                        value=train_condition(sigma,train,val,initial,proof,cache,baseline,schedule,device,arm,progress)
                        values[sigma]=value['value']
                        selected_epochs[sigma]=value['selection']['selected_checkpoint']['epoch']
                        epoch_values[label(sigma)]={('head_epoch_%02d.pth'%e):read(arm/('val_epoch_%02d.json'%e)) for e in range(1,25)}
                        report['conditions'][label(sigma)]=dict(updates=value['updates'],epochs=24,
                            selected_checkpoint=value['selection']['selected_checkpoint'],
                            cuda_peak_bytes=dict(allocated=torch.cuda.max_memory_allocated(args.gpu),reserved=torch.cuda.max_memory_reserved(args.gpu)))
                    expected=2*baseline['completion']['updates']
                    if report['head_updates_total']!=expected:raise ValueError('Sigma grid full-training budget differs')
                    rows=report_rows(values,selected_epochs)
                    write(out/'val_sigma_compare.json',dict(protocol=VERSION,split='val',scale=1.,proof=proof,rows=rows,
                        groups={label(s):values[s]['groups'] for s in model.SIGMAS},all_epoch_values=epoch_values,
                        selected_epochs={label(s):selected_epochs[s] for s in model.SIGMAS},
                        selected_sigma=None,automatic_promotion=False,test_access=False))
                    with (out/'val_sigma_compare.csv').open('x',newline='') as handle:
                        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                    for row in rows:
                        print('VAL',row['method'],'sigma',row['sigma_cells'],'epoch',row['selected_epoch'],
                            row['domain'],'R_center(output)',row['output_frame_r_center_pct'],
                            'coverage',row['output_coverage_pct'],'R_center(all)',row['full_frame_center_correct_pct'],
                            'RIoU',row['mean_riou'],
                            'A-RMSE',row['a_rmse_deg'],'DFR',row['dfr_pct_per_frame'],'ACI',row['aci'],flush=True)
                    report.update(status='SIGMA_FULL_GRID_TRAIN_VAL_COMPLETE_REVIEW_REQUIRED',
                        reused_updates=baseline['completion']['updates'],represented_updates=3*baseline['completion']['updates'])
            report['elapsed_seconds']=time.perf_counter()-started
            progress(dict(stage='complete',status=report['status']))
            write(out/'completion.json',report)
        except Exception as error:
            report.update(status='FAILED_SIGMA_REVIEW_REQUIRED',error=type(error).__name__+': '+str(error),
                elapsed_seconds=time.perf_counter()-started)
            progress(dict(stage='failed',error=report['error']));write(out/'failure.json',report)
            publish(out);raise
    publish(out)
    print('Saved',out/'completion.json','status',report['status'],flush=True)
    return report


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('check','smoke','train'),required=True)
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--baseline-dir',default='work_dirs/crane_symeood_k1_port_day2night_midpoint_formal_v1')
    p.add_argument('--cache-dir',default='work_dirs/port_geometry_midpoint_formal_v1_roi_cache')
    p.add_argument('--smoke-report');p.add_argument('--out-dir',required=True)
    return p


if __name__=='__main__':run(parser().parse_args())
