#!/usr/bin/env python3
"""Separately authorized ESH6 from ep7. Reuse accepted math, no smoke/retries."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.tools import run_symeood_110_lvsr_c_ep7 as control
from crane_project.tools import run_symeood_110_lvsr_esh_short_v1 as accepted
from crane_project.tools.run_port_size_native_train_v1 import sha,read,write,rows_write,seed,rng_get,rng_set,equal,ranking,near_joint

COMMON=control.COMMON
OUT=COMMON/'continuation_ESH_ep7'
CONTRACT=ROOT/'crane_project/data_contracts/symeood_110_lvsr_esh_ep7.json'


def status(state,gpu,**details):
    value=dict(state=state,time=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),physical_gpu=gpu,
        TEST=False,depth=False,detector_updates=0,midpoint_updates=0,automatic_retry=False,**details)
    temp=OUT/'status.tmp';temp.write_text(json.dumps(value,allow_nan=False));temp.replace(OUT/'status.json')


def checked_sources():
    proof=read(CONTRACT);accepted.check_sources()
    if proof['epochs']!=6 or proof['expected_updates']!=3834 or proof['parent_sha256']!=control.PARENT_SHA:
        raise ValueError('Fixed ESH6 budget/parent differs')
    for name,h in proof['sources'].items():
        if sha(ROOT/name)!=h:raise ValueError('Accepted ESH/source differs: '+name)
    for name,h in proof['control_files'].items():
        if sha(ROOT/name)!=h:raise ValueError('Original C evidence differs: '+name)
    for name,h in proof['short_evidence'].items():
        if sha(ROOT/name)!=h:raise ValueError('Accepted short evidence differs: '+name)
    if read(accepted.REVIEW/'run32/completion.json')['actual_updates']!=64:
        raise ValueError('Original 32-step acceptance missing')
    return proof


def schedules(train,proof):
    pp=[r for r in train if len(r['midpoint_original'])]
    if len(pp)!=5112 or math.ceil(len(pp)/8)!=639:raise ValueError('Original TRAIN/updates count differs')
    result={}
    for ep in range(1,7):
        ids=control.base.schedule_indices(len(pp),ep,continuation=True)
        h=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
        if h!=proof['schedule_sha256'][str(ep)]:raise ValueError('Original C schedule differs')
        result[str(ep)]=dict(index_sha256=h,ordered_view_sha256=hashlib.sha256(json.dumps([(pp[i]['image'],pp[i]['scale']) for i in ids],separators=(',',':')).encode()).hexdigest())
    return pp,result


def update(h,opt,ema,rr,settings,updates,torch):
    """Same accepted edge/preview/solve/Adam path; only compact statistics saved."""
    from crane_project.utils.port_size_qalign_v1 import size_loss
    h.train();opt.zero_grad();params=tuple(h.parameters());x=accepted.pack(rr,torch,'cuda:0')
    out=h(*x[:-1]);points=accepted.edge_points(out,x[-1],x[-2],x[3],torch)
    loss=size_loss(out,x[-1],x[-2],x[3])
    if not bool(torch.isfinite(loss)) or not bool(out['vector_valid'].all()) or not bool(out['candidate_valid'].all()):
        raise ValueError('Nonfinite/invalid ESH TRAIN candidate')
    native=accepted.flat(torch.autograd.grad(loss,params,retain_graph=True),torch).detach()
    G=accepted.edge_gradients(points,params,torch)
    error=float((G.sum(0)-native).abs().max())
    if not torch.allclose(G.sum(0),native,rtol=2e-5,atol=2e-7) or not torch.allclose(points.mean(),loss,rtol=1e-6,atol=1e-7):
        raise ValueError('Accepted canonical16 normalization changed')
    base=accepted.preview(params,opt,native,torch)
    if base['step']!=updates+1:raise ValueError('Pre-step Adam counter differs')
    weights,solver,raw,effective=accepted.solve(G,base['delta'],base['kappa'],ema,settings,torch)
    gradient=G.T@weights
    native_norm=float(native.double().norm());grad_norm=float(gradient.double().norm())
    if not bool(torch.isfinite(gradient).all()) or max(grad_norm,native_norm)>10:
        raise ValueError('Nonfinite gradient or clip10 activated: stop, no parameter retry')
    opt.zero_grad();accepted.parameter_grad_assign(params,gradient)
    clipped_norm=float(torch.nn.utils.clip_grad_norm_(params,10.));opt.step()
    if any(int(opt.state[p]['step'])!=updates+1 or not bool(torch.isfinite(p).all()) for p in params):
        raise ValueError('Actual Adam counter/parameter differs')
    return dict(size=float(loss.detach()),weighted_size=float((points.detach().reshape(-1)*weights).sum()/16),
        long_size=float(points[:,0].mean().detach()),short_size=float(points[:,1].mean().detach()),
        grad_norm=grad_norm,native_grad_norm=native_norm,clip_returned_norm=clipped_norm,
        weight_mean=float(weights.mean()),weight_min=float(weights.min()),weight_max=float(weights.max()),
        long_weight_mean=float(weights[0::2].mean()),short_weight_mean=float(weights[1::2].mean()),
        weights_ge_2=int((weights>=2).sum()),weights_near_1=int((weights<=1+1e-6).sum()),
        weight_bins=[int(((weights>=a)&(weights<b)).sum()) for a,b in [(1.,1.05),(1.05,1.25),(1.25,1.5),(1.5,2.),(2.,3.000001)]],
        solver_iterations=int(solver.get('iterations',0)),solver_guard=int(solver.get('guard',0)),
        identity_fallback=bool(solver.get('fallback_identity',False)),
        raw_objective_before=solver.get('raw_objective_before'),raw_objective_after=solver.get('raw_objective_after'),
        effective_energy_ratio=solver.get('effective_energy_ratio'),normalization_max_abs=error)


def fit(proof,gpu,resume,torch):
    from crane_project.tools import train_symeood_110_midpoint_v1 as f
    from crane_project.utils.port_size_qalign_v1 import AlignedSizeHead
    from crane_project.utils import port_geometry_size_distribution_metrics_v1 as metrics
    from crane_project.tools.run_port_geometry_size_condition_v1 import components
    from crane_project.tools.diagnose_symeood_110_lvsr_edges import groups,edge_diagnostic
    train,val,parent,parent_payload=control.load_inputs(torch);present,orders=schedules(train,proof)
    runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0))
    prior=read(control.OUT/'identity.json')['runtime']
    if any(runtime[k]!=v for k,v in prior.items()):raise ValueError('Original C numerical environment differs')
    short_runtime=read(accepted.DEST/'identity.json')['runtime']
    if any(runtime[k]!=v for k,v in short_runtime.items()):raise ValueError('Accepted short numerical environment differs')
    tables={s:control.objective.reference_table(parent[s+'_rows']) for s in ('train','val')}
    for split in ('train','val'):
        reference=read(control.OUT/('C/'+split+'_fixed_references.json'))
        if list(tables[split].values())!=reference:raise ValueError('Original M21/ep7 references differ')
    dest=OUT/'ESH';seed(torch);h=AlignedSizeHead('fixed_b').cuda();h.load_state_dict(parent_payload['head_state'],strict=True)
    opt=torch.optim.Adam(h.parameters(),lr=.0001,betas=(.9,.999),eps=1e-8,weight_decay=0.,amsgrad=False)
    ema=[torch.tensor(-1.,device='cuda:0'),torch.tensor(-1.,device='cuda:0')]
    if opt.state or f.g.state_digest(h)!=read(control.OUT/'C/selection.json')['initial_digest']:
        raise ValueError('Fresh ep7/Adam differs from original C')
    settings={k:proof['solver'][k] for k in ('sum_grads','constraint','box_bound','radius_bound','allow_neg_weights','reg','warm_start_iter','max_iter','tol','mom_stats_Q','max_raise_condition_Q','max_raise_dgdg')}
    settings['use_wandb']=False
    events=[];updates=0;start_epoch=1;contract_sha=sha(CONTRACT);riou_cache={}
    if resume:
        saved=torch.load(str(resume),map_location='cpu')
        if (saved['protocol']!=proof['protocol'] or saved['contract_sha256']!=contract_sha or
            saved['parent_sha256']!=control.PARENT_SHA or saved['schedule_sha256']!=proof['schedule_sha256']):
            raise ValueError('Resume identity differs')
        if saved['epoch']>=6:raise ValueError('Completed six epochs cannot resume')
        updates=saved['updates'];events=saved['events'];start_epoch=saved['epoch']+1
        events[-1]['checkpoint_sha256']=sha(resume)
        lines=[json.loads(x) for x in (dest/'progress.jsonl').read_text().splitlines()]
        if len(lines)!=updates or lines[-1]['updates']!=updates:
            raise ValueError('Resume only at saved epoch boundary; preserve any later partial updates')
        h.load_state_dict(saved['head_state'],strict=True);opt.load_state_dict(saved['optimizer_state'])
        for target,value in zip(ema,saved['ESH_state']['ema']):target.copy_(value.to('cuda:0'))
        if any(int(opt.state[p]['step'])!=updates for p in h.parameters()):raise ValueError('Resume Adam counter differs')
        rng_set(saved['rng'],torch)
        write(OUT/('resume_epoch%02d.json'%start_epoch),dict(checkpoint=str(resume),sha256=sha(resume),updates=updates,
            restored_head_optimizer_ESH_RNG=True,time=datetime.now(timezone.utc).isoformat()))
    else:
        dest.mkdir()
        write(OUT/'identity.json',dict(contract_sha256=contract_sha,runtime=runtime,parent_epoch=7,parent_sha256=control.PARENT_SHA,
            detector_sha256=control.base.DETECTOR_SHA,midpoint_sha256=control.base.MIDPOINT_SHA,physical_gpu=gpu,
            initial_digest=f.g.state_digest(h),optimizer='fresh Adam',schedules=orders,steps_per_epoch=639,
            train_present=5112,VAL_frames=len(val),feature_extractions=0,midpoint_forwards=0,preflight_optimizer_updates=0))
        write(OUT/'preparation_check.json',dict(passed=True,scope='Fixed sources,parent,cache,schedules,and accepted short only',
            reused_unit_tests=True,reused_short_64updates=True,new_smoke_updates=0,identity_differences=[],TEST=False,depth=False))
    status('training_ESH',gpu,epoch=start_epoch-1,updates=updates)
    for epoch in range(start_epoch,7):
        tick=time.monotonic();ids=control.base.schedule_indices(len(present),epoch,continuation=True)
        totals={};bins=[0]*5;guards=Counter();fallbacks=0;maxgrad=0.;minweight=3.;maxweight=1.
        with (dest/'progress.jsonl').open('a') as log:
            for start in range(0,len(ids),8):
                rr=[present[i] for i in ids[start:start+8]];batch_tick=time.monotonic()
                stats=update(h,opt,ema,rr,settings,updates,torch);updates+=1
                maxgrad=max(maxgrad,stats['grad_norm']);minweight=min(minweight,stats['weight_min']);maxweight=max(maxweight,stats['weight_max'])
                for j,n in enumerate(stats['weight_bins']):bins[j]+=n
                guards[stats['solver_guard']]+=1;fallbacks+=stats['identity_fallback']
                for k,v in stats.items():
                    if isinstance(v,(int,float)) and not isinstance(v,bool):totals[k]=totals.get(k,0.)+v
                log.write(json.dumps(dict(epoch=epoch,step=start//8+1,updates=updates,batch_size=len(rr),elapsed_seconds=time.monotonic()-batch_tick,**stats),allow_nan=False)+'\n');log.flush()
                if (start//8+1)%100==0:
                    status('training_ESH',gpu,epoch=epoch,step=start//8+1,updates=updates)
                    print(json.dumps(dict(epoch=epoch,step=start//8+1,updates=updates)),flush=True)
        if updates!=epoch*639 or sum(bins)!=5112*2:raise ValueError('Six-epoch update/exposure counter differs')
        status('VAL',gpu,epoch=epoch,updates=updates)
        vr=control.old.evaluate(h,val,tables['val'],torch)
        if len(vr)!=887:raise ValueError('Full original VAL required')
        summary=metrics.summary(vr,riou_cache);diag=near_joint(vr);comp=components(vr)
        gate=control.objective.eligibility(summary,diag,parent['summary'])
        damage={g:edge_diagnostic(rr,'edge_residual')[0] for g,rr in groups(vr).items()}
        ep=dict(epoch=epoch,updates=updates,updates_per_epoch=639,metrics=summary,components=comp,diagnostics=diag,damage=damage,
            protection=control.objective.protection_report(vr,tables['val']),eligibility=gate,selection_key=ranking(summary,epoch),
            schedule_sha256=orders[str(epoch)]['index_sha256'],training_losses={k:v/639 for k,v in totals.items()},max_grad_norm=maxgrad,
            weight_distribution=dict(min=minweight,max=maxweight,bins=bins,bin_edges=[1.,1.05,1.25,1.5,2.,3.],guard_counts=dict(guards),identity_fallbacks=fallbacks),
            epoch_seconds=time.monotonic()-tick)
        rows_write(dest/('epoch_%02d_val.rows.jsonl.gz'%epoch),vr)
        events.append(ep)
        rng=rng_get(torch)
        payload=dict(protocol=proof['protocol'],arm='ESH',epoch=epoch,updates=updates,updates_per_epoch=639,
            parent_epoch=7,parent_sha256=control.PARENT_SHA,input_identity=parent_payload['input_identity'],
            contract_sha256=contract_sha,schedule_sha256=proof['schedule_sha256'],events=deepcopy(events),
            head_state={k:v.detach().cpu().clone() for k,v in h.state_dict().items()},optimizer_state=deepcopy(opt.state_dict()),
            ESH_state=dict(ema=[v.detach().cpu().clone() for v in ema],settings=settings),rng=rng)
        path=dest/('head_epoch_%02d.pth'%epoch)
        with path.open('xb') as stream:torch.save(payload,stream)
        if not equal(payload,torch.load(str(path),map_location='cpu'),torch):raise ValueError('Complete checkpoint serialization differs')
        rng_set(rng,torch)
        ep['checkpoint_sha256']=sha(path)
        write(dest/('epoch_%02d_val.json'%epoch),ep)
        status('epoch_complete',gpu,epoch=epoch,updates=updates,eligible=gate['pass_all'])
        print(json.dumps(dict(epoch=epoch,updates=updates,eligible=gate['pass_all'],failed=gate['failed'],
            VAL_joint=summary['overall']['edge_residual']['sizes']['joint10'],epoch_seconds=ep['epoch_seconds'])),flush=True)
    selected=control.versions(events)
    write(dest/'selection.json',dict(initial_digest=read(OUT/'identity.json')['initial_digest'],formal_updates=updates,epochs=events,**selected))
    backups={};by_epoch={}
    for role,n in [('selected',selected['selected']),('representative',selected['representative']),('fixed_ESH6',6)]:
        if n is None:continue
        if n not in by_epoch:
            by_epoch[n]=control.base.backup(dest/('head_epoch_%02d.pth'%n),'ESH_ep7_epoch%02d'%n,OUT,torch,
                'eligible_selected' if n==selected['selected'] else 'descriptive_only')
        backups[role]=by_epoch[n]
    if checked_sources()!=proof:raise ValueError('Fixed source/evidence changed while running')
    write(OUT/'completion.json',dict(state='complete',finished=datetime.now(timezone.utc).isoformat(),contract_sha256=contract_sha,
        authorization='Separate ESH6 budget, preserves failed short gate; no retrospective short pass',parent_epoch=7,
        parent_sha256=control.PARENT_SHA,formal_updates=updates,versions=selected,backups=backups,
        full_VAL_epochs=6,TEST=False,depth=False,automatic_extension=False,preflight_updates=0,
        files={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.suffix!='.pth'}))
    status('complete',gpu,updates=updates,**selected)


def main(args):
    proof=checked_sources()
    if not args.run:
        print(json.dumps(dict(static_check=True,protocol=proof['protocol'],new_updates=0,short_gate_still_failed=True)));return
    if args.gpu is None or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.gpu):raise ValueError('Explicit single GPU required')
    if args.resume:
        checkpoint=Path(args.resume).resolve()
        if checkpoint.parent!=OUT/'ESH' or not checkpoint.is_file() or (OUT/'completion.json').exists():
            raise ValueError('Resume must use this unfinished ESH experiment epoch checkpoint')
    else:
        checkpoint=None
        if OUT.exists():raise ValueError('Existing results preserved; no automatic retraining')
        OUT.mkdir()
    try:
        import torch
        torch.set_num_threads(1);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.6,0)
        fit(proof,args.gpu,checkpoint,torch)
    except BaseException as e:
        status('stopped_on_error',args.gpu,error=repr(e));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true')
    p.add_argument('--gpu',type=int,choices=range(4));p.add_argument('--resume')
    main(p.parse_args())
