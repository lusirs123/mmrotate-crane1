#!/usr/bin/env python3
"""Sealed R reuse versus one geometry-condition candidate; complete VAL/epoch."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import run_port_geometry_size_distribution_v1 as old
from crane_project.tools import eval_port_geometry_size_distribution_val_v2 as val_replay

sealed, metrics = old.sealed, old.metrics
VERSION = 'port_geometry_size_condition_v1'
CONTRACT = ROOT/('crane_project/tools/'+VERSION+'_protocol.json')
MANIFEST = ROOT/('crane_project/tools/'+VERSION+'_sources.json')


def protocol():
    return dict(protocol=VERSION, settings=old.SETTINGS, parameter_count=2712,
        input='Same frozen B P3 ROI/support, B original/model OBB, actual sx/sy and frozen M as R; full2558 TRAIN identities standard/half5116 views,5112 present.',
        change='Only replace four repeated additive geometry entries with geometry*log_bucket_coordinate in both terminal linear readouts. Retain all four entries, including context constants. Effective freedom increases despite same nominal count.',
        supervision='Original R decoded SmoothL1 beta0.1 only; same targets, gradients, delivery guards, grid and initialization. B/M frozen.',
        control='Reuse all24 R epochs only after immutable source/cache/weights/schedule/initial/runtime and every checkpoint plus saved VAL rows/summary SHA verification. Replay R01,R05,R24 predictions exactly. A mismatch stops, never silently retrains control.',
        evaluation='Candidate every24 epochs complete887 VAL, independent of TRAIN diagnosis. Overall/Real/Sim/video custom metrics, three geometry components, coverage, AP11 cached-GT supplement; full TRAIN groups per epoch are explanatory only.',
        selection='All24 per arm descriptive representative: Real long MAE, short MAE, earliest epoch. Independently select best adoption-passing epoch by same key. If none, report representative and keep M. Both arms receive same24-epoch window and new adoption checks.',
        adoption=dict(real='Both long/short MAE strictly lower than both M and B by1e-6 relative units; full mean_RIoU>M+1e-5.',
                      sim='Both long/short MAE<=M+1e-6 relative units; full mean_RIoU>=M-1e-5. Tolerances are numerical, no substantive degradation allowance.',
                      protection='Exact M center/raw and canonical direction/score/output. Per-domain MCML max/mean nonincrease and TDR hits nondecrease with identical denominators; ACI equality. DFR disclosed, no independent rejection threshold.',
                      diagnostic_only='joint10 and transitions, P95/bias/a/r/projected ratios, near/mixed damage, AP; not separate all-must-pass gates.'),
        mechanism='Separately compare candidate best adoption-passing or representative with control best adoption-passing or representative: Real long MAE lower>1e-6, short no worse; Sim both MAE no worse and both domains mean_RIoU no worse within1e-5. Indicates usefulness of the readout change, not proof of semantic error-group recognition.',
        engineering='Byte SHA, full real TRAIN/VAL zero identity and original M reference replay, model-axis/resize/delivery,2 discarded updates and exact checkpoint+optimizer+RNG replay. Restore seed1703 after smoke; verify original R schedule and initial digest. Full per-view geometry/context clamp counts saved.',
        stop='Engineering invalidity stops; no TRAIN admission. One24-epoch candidate, no sweeps/additional modules/budget. No adopted candidate keeps M. No TEST/depth/reliability access or automatic promotion.',
        limits=dict(test_access=False,depth_access=False,detector_updates=0,frozen_midpoint_updates=0,
                    feature_extractions=0,cache_rebuilt=False,automatic_promotion=False))


def checked_sources():
    parent = old.checked_sources()
    if sealed.read(CONTRACT) != protocol():
        raise ValueError('Frozen condition protocol changed')
    source = sealed.read(MANIFEST)
    required = {'crane_project/utils/'+VERSION+'.py', 'crane_project/tools/run_'+VERSION+'.py',
                'tests/test_'+VERSION+'.py', 'crane_project/tools/'+VERSION+'_protocol.json',
                'crane_project/tools/eval_port_geometry_size_distribution_val_v2.py',
                'crane_project/utils/port_simple_component_reliability_v1.py'}
    if source['protocol'] != VERSION or set(source['sources']) != required:
        raise ValueError('Condition source scope differs')
    for name, digest in source['sources'].items():
        if sealed.sha(ROOT/name) != digest:
            raise ValueError('Condition source drift: '+name)
    return dict(parent=parent, condition=source, manifest_sha256=sealed.sha(MANIFEST))


def adoption(value):
    checks = {}
    for domain in ('real', 'sim'):
        cell=value[domain]; m,b,c=[cell[k] for k in ('midpoint','b','edge_residual')]
        p=domain+'/'
        for edge in ('long','short'):
            x=c['sizes']['edges'][edge]['mae']; reference=m['sizes']['edges'][edge]['mae']
            checks[p+edge] = (x < min(reference,b['sizes']['edges'][edge]['mae'])-1e-6
                              if domain=='real' else x<=reference+1e-6)
        checks[p+'riou']=(c['mean_riou_full_frame']>m['mean_riou_full_frame']+1e-5 if domain=='real'
                          else c['mean_riou_full_frame']>=m['mean_riou_full_frame']-1e-5)
        checks[p+'identity']=cell['paired']['frozen_identity'] and cell['paired']['new_output']==cell['paired']['lost_output']==0
        mt,ct=m['temporal'][domain],c['temporal'][domain]
        checks[p+'continuity']=ct['mcml_max']<=mt['mcml_max'] and ct['mcml_mean']<=mt['mcml_mean'] and ct['tdr_w10']['denominator']==mt['tdr_w10']['denominator'] and ct['tdr_w10']['numerator']>=mt['tdr_w10']['numerator']
        checks[p+'angle_center']=ct['aci']==mt['aci'] and ct['aci_adjacent_pairs']==mt['aci_adjacent_pairs'] and c['center_correct_full_frame']==m['center_correct_full_frame'] and c['pure_angle_rmse_deg']==m['pure_angle_rmse_deg']
    return dict(pass_all=all(checks.values()), checks=checks, failed=[k for k,v in checks.items() if not v])


def choose(results):
    if sorted(r['epoch'] for r in results)!=list(range(1,25)):
        raise ValueError('All24 complete epochs required')
    eligible=[r for r in results if r['adoption']['pass_all']]
    return dict(representative=min(results,key=lambda r:r['selection_key']),
                selected=min(eligible,key=lambda r:r['selection_key']) if eligible else None,
                passing_epochs=[r['epoch'] for r in eligible], epochs=results)


def components(rows):
    from crane_project.utils.port_simple_component_reliability_v1 import canonical, geometry_errors
    groups={'overall':rows}
    groups.update({d:[r for r in rows if r['domain']==d] for d in ('real','sim')})
    groups.update({s:[r for r in rows if r['sequence']==s] for s in sorted({r['sequence'] for r in rows})})
    result={}
    for label, subset in groups.items():
        result[label]={}
        for method in metrics.METHODS:
            counts=dict(frames=len(subset), outputs=0, missing=0, center_correct=0, size_correct=0,
                        angle_eligible_frames=0, angle_eligible_outputs=0, angle_correct=0)
            for r in subset:
                g=canonical(r['gt']); eligible=g[2]/g[3]>=1.2
                counts['angle_eligible_frames']+=int(eligible)
                if r[method] is None:
                    counts['missing']+=1; continue
                counts['outputs']+=1; e=geometry_errors(r['gt'],r[method])
                counts['center_correct']+=int(e['center_px']<15)
                counts['size_correct']+=int(e['size_max_relative']<=.1)
                counts['angle_eligible_outputs']+=int(eligible)
                counts['angle_correct']+=int(eligible and e['angle_deg']<=3)
            result[label][method]=counts
    return result


def save(path,head,opt,epoch,proof,torch):
    rng=dict(python=random.getstate(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
    payload=dict(protocol=VERSION,epoch=epoch,updates=epoch*639,proof=proof,
        head_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()},
        optimizer_state=deepcopy(opt.state_dict()),rng=rng,head_digest=sealed.state_digest(head))
    with path.open('xb') as f:torch.save(payload,f)
    from crane_project.utils.port_geometry_size_condition_v1 import GeometryConditionSizeHead
    restored=sealed.torch_load(torch,path); reload=GeometryConditionSizeHead().to(next(head.parameters()).device)
    reload.load_state_dict(restored['head_state'],strict=True)
    replay_opt=torch.optim.Adam(reload.parameters(),lr=.001,weight_decay=0.)
    replay_opt.load_state_dict(restored['optimizer_state'])
    if sealed.state_digest(reload)!=payload['head_digest']:raise ValueError('Checkpoint head replay differs')
    for a,b in zip(opt.state_dict()['state'].values(),replay_opt.state_dict()['state'].values()):
        for k in a:
            if torch.is_tensor(a[k]):assert torch.equal(a[k].cpu(),b[k].cpu())
            else:assert a[k]==b[k]
    assert restored['rng']['python']==rng['python'] and torch.equal(restored['rng']['cpu'],rng['cpu'])
    assert all(torch.equal(a,b) for a,b in zip(restored['rng']['cuda'],rng['cuda']))
    random.setstate(rng['python']);torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state_all(rng['cuda'])
    return reload,dict(sha256=sealed.sha(path),head_digest=payload['head_digest'],save_reload_exact=True)


def geometry_inventory(packed,torch):
    roi,support,b,bm,xy,m,gt=packed['tensors']
    result=[]
    for i in range(0,len(b),64):
        sl=slice(i,i+64)
        _,cb,_,sides,_,_=old.head_module().residual.checked_inputs(roi[sl],support[sl],b[sl],bm[sl],xy[sl],m[sl])
        g=old.head_module().residual.raw_roi_pair(m[sl,2:4]/b[sl,2:4],bm[sl]).log()
        extra=(cb[:,2:4]/sides).log()
        for j in range(len(cb)):
            result.append(dict(geometry=torch.cat((g[j],extra[j])).cpu().tolist(),
                context_floor=(cb[j,2:4]*1.5<16).cpu().tolist(),
                model_original_swap_diff=bool((bm[i+j,2]<bm[i+j,3])!=(b[i+j,2]<b[i+j,3]))))
    return result


def run(args):
    sources=checked_sources();out=Path(args.out_dir).resolve()
    if out.parent!=(ROOT/'work_dirs'/VERSION).resolve() or out.exists():raise ValueError('Fresh output path required')
    out.mkdir(parents=True)
    proof=dict(source_identity=sources,git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT),text=True).strip())
    report=dict(protocol=VERSION,started_utc=datetime.now(timezone.utc).isoformat(),status='FAILED',
                **protocol()['limits']);started=time.monotonic()
    try:
        sealed.write(out/'protocol.json',protocol());sealed.write(out/'source_identity.json',sources)
        if args.stage=='check':
            report.update(status='STATIC_CONTRACT_PASS',head_updates=0);return
        import torch
        from crane_project.utils import port_geometry_midpoint_sigma_v1 as frozen_model
        from crane_project.utils.port_geometry_size_condition_v1 import GeometryConditionSizeHead
        if not torch.cuda.is_available() or not os.environ.get('CUDA_VISIBLE_DEVICES') or len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))!=1 or os.environ.get('WORLD_SIZE','1')!='1':raise ValueError('Explicit single GPU required')
        torch.set_num_threads(1);device=torch.device('cuda:0');torch.cuda.set_device(device);old.seed_all(torch)
        run_dir,val_dir,cache_dir,head_dir,b_weight=[Path(getattr(args,n)).resolve() for n in ('control_run','control_val','cache_dir','head_dir','b_weight')]
        for p in (run_dir,val_dir,cache_dir,head_dir,b_weight):
            if out==p or out in p.parents or p in out.parents:raise ValueError('Output/input overlap')
        old_index=sealed.read(run_dir/'artifacts.json');vindex=sealed.read(val_dir/'artifacts.json')
        control_file=lambda name:val_replay.checked_file(run_dir,old_index['files'],name)
        val_file=lambda name:val_replay.checked_file(val_dir,vindex['files'],name)
        oldproof=sealed.read(control_file('input_runtime_proof.json'))
        if oldproof['source_identity']!=sources['parent']:raise ValueError('Original R source identity differs')
        completion=sealed.read(control_file('completion.json'))
        if completion['head_updates']!=30672:raise ValueError('Incomplete original training')
        historical,differences=old.historical_identity();cache=sealed.checked_cache(cache_dir,historical)
        selection,references,indexed=sealed.checked_selection(head_dir,historical)
        if sealed.sha(b_weight)!=old.B_SHA:raise ValueError('B weight SHA differs')
        runtime=dict(python=sys.version,executable=sys.executable,torch=str(torch.__version__),cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0),cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
            tf32_matmul=getattr(torch.backends.cuda.matmul,'allow_tf32',None),tf32_cudnn=getattr(torch.backends.cudnn,'allow_tf32',None))
        for key in runtime:
            if key not in ('cuda_visible_devices',) and runtime[key]!=oldproof['runtime'][key]:raise ValueError('Control runtime differs: '+key)
        proof.update(runtime=runtime,cache_manifest_sha256=sealed.CACHE_SHA,shards=cache['files'],fixed_b_sha256=old.B_SHA,
            fixed_m=sealed.FIXED,control_run=str(run_dir),control_val=str(val_dir),control_artifacts_sha256=sealed.sha(run_dir/'artifacts.json'),
            control_val_artifacts_sha256=sealed.sha(val_dir/'artifacts.json'),reference_artifacts=indexed,unused_historical_differences=differences)
        frozen=sealed.load_head(torch,frozen_model,head_dir,selection,cache,device);frozen_digest=sealed.state_digest(frozen)
        train,val,_=old.load_records(cache_dir,cache,frozen,device,torch,old.head_module(),references)
        tp,vp=old.pack(train,device,torch),old.pack(val,device,torch)
        if (len(train),len(tp['indices']),len(val),len(vp['indices']))!=(5116,5112,887,886):raise ValueError('Full counts differ')
        schedule=sealed.read(control_file('schedule.json'));batches=[old.epoch_batches(5112,e) for e in range(1,25)]
        identities=[dict(image=train[i]['image'],scale=train[i]['scale']) for i in tp['indices']]
        if schedule['batches']!=batches or schedule['train_view_identity']!=identities or schedule['updates_per_arm']!=15336:raise ValueError('Control schedule differs')
        sealed.write(out/'schedule.json',schedule)
        old.seed_all(torch);h=GeometryConditionSizeHead().to(device);initial=sealed.state_digest(h)
        if initial!=sealed.read(control_file('preflight.json'))['initial_state']:raise ValueError('R initial digest differs')
        baseline_train=old.evaluate_rows(h,train,tp,torch);baseline_val=old.evaluate_rows(h,val,vp,torch)
        if any(r['midpoint']!=r['edge_residual'] for r in baseline_train+baseline_val):raise ValueError('Real full neutral replay differs')
        for name,rows in [('baseline_train.rows.jsonl.gz',baseline_train),('baseline_val.rows.jsonl.gz',baseline_val)]:
            original=old.read_rows(control_file(name))
            val_replay.compare_numeric(rows,original);old.write_rows(out/name,rows)
        riou_cache={};base_summary=metrics.summary(baseline_val,riou_cache)
        val_replay.compare_numeric(base_summary,sealed.read(control_file('baseline_val_summary.json')))
        sealed.write(out/'baseline_val_summary.json',base_summary)
        sealed.write(out/'baseline_components.json',components(baseline_val))
        inventory=geometry_inventory(tp,torch)
        sealed.write(out/'real_train_geometry_inventory.json',dict(view_identity=identities,values=inventory,
            context_floor_count=[sum(x['context_floor'][k] for x in inventory) for k in (0,1)],
            model_original_swap_diff=sum(x['model_original_swap_diff'] for x in inventory)))
        control=[]
        for epoch in range(1,25):
            stem='decoded_control/';ckname=stem+'head_epoch_%02d.pth'%epoch
            meta=sealed.read(control_file(stem+'epoch_%02d.json'%epoch));path=control_file(ckname)
            if sealed.sha(path)!=meta['checkpoint']['sha256']:raise ValueError('R checkpoint metadata differs')
            payload=sealed.torch_load(torch,path);val_replay.check_checkpoint_metadata(payload,meta,'decoded_control',epoch,oldproof)
            control_rows=old.read_rows(val_file(stem+'val_epoch_%02d.rows.jsonl.gz'%epoch))
            value=sealed.read(val_file(stem+'val_epoch_%02d.summary.json'%epoch))
            recomputed=metrics.summary(control_rows,riou_cache);val_replay.compare_numeric(value,recomputed)
            if epoch in (1,5,24):
                rh=old.head_module().DistributionSizeHead().to(device);rh.load_state_dict(payload['head_state'],strict=True)
                if sealed.state_digest(rh)!=payload['head_digest']:raise ValueError('R digest differs')
                replay=old.evaluate_rows(rh,val,vp,torch)
                # summary() adds derived fields to saved rows; compare raw keys.
                val_replay.compare_numeric([{k:v for k,v in r.items() if k!='metrics'} for r in control_rows],replay)
            result=dict(epoch=epoch,selection_key=list(metrics.selection_key(value,epoch)),adoption=adoption(value),checkpoint_sha256=sealed.sha(path),reused=True)
            control.append(result)
            dest=out/'control';dest.mkdir(exist_ok=True)
            sealed.write(dest/('val_epoch_%02d.summary.json'%epoch),value)
            sealed.write(dest/('val_epoch_%02d.components.json'%epoch),components(control_rows))
            sealed.write(dest/('epoch_%02d.json'%epoch),result)
        controls=choose(control);sealed.write(out/'control/selection.json',controls)
        # Discard exactly two engineering updates; fresh seed/state afterwards.
        old.seed_all(torch);smoke=GeometryConditionSizeHead().to(device);opt=torch.optim.Adam(smoke.parameters(),lr=.001,weight_decay=0.)
        events=[old.update(smoke,opt,tp,b,'decoded_control',torch) for b in batches[0][:2]]
        if events[0]['stem_grad_norm']!=0 or events[0]['terminal_grad_norm']<=0 or events[1]['stem_grad_norm']<=0:raise ValueError('Candidate gradient smoke failed')
        reload,ck=save(out/'smoke.pth',smoke,opt,0,proof,torch)
        with torch.no_grad():
            x=[t[:8] for t in tp['tensors'][:-1]]
            for key in ('bucket_logits','boxes_original'):
                if not torch.equal(smoke(*x)[key],reload(*x)[key]):raise ValueError('Smoke prediction replay differs')
        sealed.write(out/'preflight.json',dict(status='ENGINEERING_PASS',real_zero_train_val_exact=True,initial_state=initial,
            control_schedule_init_runtime_exact=True,control_replay_epochs=[1,5,24],smoke_updates_discarded=2,events=events,checkpoint=ck))
        sealed.write(out/'input_runtime_proof.json',proof)
        print('ENGINEERING_PASS; control reused24; candidate fresh24',flush=True)
        old.seed_all(torch);h=GeometryConditionSizeHead().to(device)
        if sealed.state_digest(h)!=initial:raise ValueError('Smoke state inherited')
        opt=torch.optim.Adam(h.parameters(),lr=.001,weight_decay=0.)
        dest=out/'candidate';dest.mkdir();results=[]
        with (dest/'progress.jsonl').open('x') as log:
            for epoch,eb in enumerate(batches,1):
                tick=time.monotonic();losses=[]
                for step,batch in enumerate(eb,1):
                    event=old.update(h,opt,tp,batch,'decoded_control',torch);losses.append(event['loss'])
                    log.write(json.dumps(dict(epoch=epoch,step=step,**event),allow_nan=False)+'\n')
                    if step%100==0 or step==639:log.flush();print('candidate',epoch,step,event['loss'],flush=True)
                tr=old.evaluate_rows(h,train,tp,torch);train_report=metrics.train_report(tr)
                old.write_rows(dest/('train_epoch_%02d.rows.jsonl.gz'%epoch),tr);sealed.write(dest/('train_epoch_%02d.summary.json'%epoch),train_report)
                _,ck=save(dest/('head_epoch_%02d.pth'%epoch),h,opt,epoch,proof,torch)
                rows=old.evaluate_rows(h,val,vp,torch);value=metrics.summary(rows,riou_cache)
                old.write_rows(dest/('val_epoch_%02d.rows.jsonl.gz'%epoch),rows)
                sealed.write(dest/('val_epoch_%02d.summary.json'%epoch),value);sealed.write(dest/('val_epoch_%02d.components.json'%epoch),components(rows))
                result=dict(epoch=epoch,updates=epoch*639,checkpoint=ck,selection_key=list(metrics.selection_key(value,epoch)),
                    adoption=adoption(value),train_pass_diagnostic_only=train_report['train_pass'],val_evaluated=True,
                    mean_loss=sum(losses)/len(losses),elapsed_seconds=time.monotonic()-tick)
                results.append(result);sealed.write(dest/('epoch_%02d.json'%epoch),result)
                if sealed.state_digest(frozen)!=frozen_digest or any(p.grad is not None or p.requires_grad for p in frozen.parameters()):raise ValueError('M updated')
                print('EPOCH_COMPLETE',epoch,'key',result['selection_key'],'adoption',result['adoption']['pass_all'],flush=True)
        candidates=choose(results);sealed.write(dest/'selection.json',candidates)
        chosen=lambda x:x['selected'] or x['representative']
        ce,re=chosen(candidates)['epoch'],chosen(controls)['epoch']
        c=sealed.read(dest/('val_epoch_%02d.summary.json'%ce));r=sealed.read(out/'control'/('val_epoch_%02d.summary.json'%re))
        mechanism_checks={}
        for domain in ('real','sim'):
            for edge in ('long','short'):
                a,b=c[domain]['edge_residual']['sizes']['edges'][edge]['mae'],r[domain]['edge_residual']['sizes']['edges'][edge]['mae']
                mechanism_checks[domain+'/'+edge]=(a<b-1e-6 if (domain,edge)==('real','long') else a<=b+1e-6)
            mechanism_checks[domain+'/riou']=c[domain]['edge_residual']['mean_riou_full_frame']>=r[domain]['edge_residual']['mean_riou_full_frame']-1e-5
        if checked_sources()!=sources or sealed.sha(b_weight)!=old.B_SHA or sealed.sha(head_dir/sealed.FIXED['path'])!=sealed.FIXED['sha256']:raise ValueError('End-run immutable drift')
        for label,digest in cache['files'].items():
            if sealed.sha(cache_dir/label)!=digest:raise ValueError('End-run shard drift')
        for name,digest in old_index['files'].items():
            if sealed.sha(run_dir/name)!=digest:raise ValueError('Original run mutated')
        report.update(status='COMPLETE_REVIEW_REQUIRED' if candidates['selected'] else 'COMPLETE_NO_ADOPTED_VERSION_KEEP_M',
            control=controls,candidate=candidates,mechanism_checks=mechanism_checks,readout_supported=all(mechanism_checks.values()),
            formal_head_updates=15336,engineering_head_updates_discarded=2,control_training_updates=0,
            complete_val_epochs_per_arm=24,control_reused=True,frozen_m_unchanged=True,peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(0))
    except BaseException as exc:
        report['error']=repr(exc);raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started;report['finished_utc']=datetime.now(timezone.utc).isoformat()
        sealed.write(out/'completion.json',report)
        sealed.write(out/'artifacts.json',dict(protocol=VERSION,files={str(p.relative_to(out)):sealed.sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifacts.json'}))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('check','run'),required=True);p.add_argument('--out-dir',required=True)
    for name in ('cache-dir','head-dir','b-weight','control-run','control-val'):p.add_argument('--'+name)
    return p


if __name__=='__main__':run(parser().parse_args())
