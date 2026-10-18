#!/usr/bin/env python3
"""S035 one conditional local-size auxiliary run, exact reused V control.

Only --run with a complete SHA-bound real-update diagnosis starts fitting.
Control is the original24-epoch size_only run, not its best single-side epoch.
"""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.tools.run_port_size_native_train_v1 import sha,read,write,rows_write,seed,rng_get,rng_set,equal,eligibility,ranking,near_joint
VERSION='port_size_local_aux_v1'


def protocol():
    return dict(protocol=VERSION,hypothesis='Keep original V forward; local size consistency supplies position-resolved size targets without complete midpoint pose supervision. Not guaranteed useful for background locations.',
        control='Original S034 size_only24epochs reused only after exact CPU and real-cache loss/gradient/Adam update equivalence, initial state and24schedule checks. No old G control, no V20 warmstart.',
        candidate='Fresh original2800parameter LocalVectorSizeHead, identical input/support/geometry/output/decoder/delivery; Lfinal+0.1Llocal. No complete GT midpoint geometry loss.',
        local_loss='Each existing support>0 location predicts annotated canonical lengths. Original-pixel local norms / original neutral-reference mean -> log residual, GT/M canonical targets; mean2 SmoothL1 beta0.1, geometric-support-normalized per view, meanN. Not foreground selection. No GT filtering, quality mask, lambda search or new evidence.',
        budget='FullTRAIN5116views/5112outputs,24epochs,batch8,639updates/epoch,Adam0.001/wd0/clip10/seed1703,Random(1703+epoch). Everyepoch fullVAL887, noTRAIN admission.',
        selection='Exact inherited S034 eligibility and unique lexicographic Simlong,Simshort,Reallong,Realshort,epoch; no eligible => descriptive representative only. Same control full24epoch rule.',
        adoption='S034 Sim both<M-1e-6,Realboth<=M+0.001,Sim a/r<=M+1e-6,4near<=M+0.001;overallRIoU>=M-1e-5,domains>=M-0.001;identityexact,DFR/MCMLnonincrease,TDRnondecrease,ACIequal. Near permits0.1pp, not zero damage.',
        mechanism='At selected-or-descriptive representatives candidateSim bothMAE<control-1e-6,Realboth<=control+0.001,overallRIoU>=control-1e-5; additionally describe direction harm/usefulshort retention and Realshort/near. Main adoption separate, no gate on all diagnostics.',
        attribution='Only auxiliary objective changes against V; same nominal parameters. Extra loss also changes gradient magnitude; no claim all weighting/regularization causes isolated. Reused control is verified existing result, not a newly completed two-arm run.',
        stop='Single candidate24epochs only. No TEST/depth,newlambda/mask/features/epochs or automatic M promotion. InvalidSHA/nonfinite/cache/mapping/resource stops.',
        resource='One physicalGPU2or3 only;process<7000MiB,stop6800,allocator6000.')


def proof():
    from crane_project.tools.audit_port_size_auxiliary_conflict_v1 import sources
    own=['crane_project/tools/run_port_size_local_aux_v1.py']
    contract=ROOT/'crane_project/tools/port_size_local_aux_v1_protocol.json'
    if read(contract)!=protocol():raise ValueError('Frozen local-size protocol drift')
    return dict(parent=sources(),own={n:sha(ROOT/n) for n in own},protocol_sha256=sha(contract))


def update(h,opt,x,arm,torch):
    from crane_project.utils import port_size_auxiliary_conflict_v1 as aux
    h.train();opt.zero_grad();o=h(*x[:-1]);loss,size,local=aux.losses(o,x[-1],x[1],*x[2:6],arm)
    if not bool(o['vector_valid'].all()) or not bool(torch.isfinite(loss)):raise ValueError('Invalid local loss/vector')
    loss.backward()
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in h.parameters()):raise ValueError('Invalid local gradient')
    norm=float(torch.nn.utils.clip_grad_norm_(h.parameters(),10.));opt.step()
    if any(not bool(torch.isfinite(p).all()) for p in h.parameters()):raise ValueError('Nonfinite updated parameter')
    return dict(loss=float(loss.detach()),size_loss=float(size.detach()),local_loss=float(local.detach()),grad_norm=norm)


def control_equivalence(train,out,torch):
    from crane_project.utils import port_size_local_vectors_v1 as model
    from crane_project.tools.run_port_size_local_vectors_v1 import batch,update as original_update
    seed(torch);a=model.LocalVectorSizeHead().cuda();b=model.LocalVectorSizeHead().cuda();b.load_state_dict(a.state_dict())
    oa=torch.optim.Adam(a.parameters(),lr=.001);ob=torch.optim.Adam(b.parameters(),lr=.001)
    present=[r for r in train if len(r['midpoint_original'])];checks=[]
    for start in (0,8):
        x=batch(present[start:start+8],torch)
        v=original_update(a,oa,x,'size_only',torch);z=update(b,ob,x,'size_only',torch)
        if v['loss']!=z['loss'] or not equal(a.state_dict(),b.state_dict(),torch) or not equal(oa.state_dict(),ob.state_dict(),torch):raise ValueError('Reused control not equivalent')
        checks.append(dict(step=start//8+1,loss=v['loss'],parameters_optimizer_exact=True))
    seed(torch);candidate=model.LocalVectorSizeHead().cuda();oc=torch.optim.Adam(candidate.parameters(),lr=.001)
    first=update(candidate,oc,batch(present[:8],torch),'size_plus_local',torch)
    snap=dict(head=deepcopy(candidate.state_dict()),opt=deepcopy(oc.state_dict()),rng=rng_get(torch))
    stream=io.BytesIO();torch.save(snap,stream);stream.seek(0);got=torch.load(stream)
    clone=model.LocalVectorSizeHead().cuda();clone.load_state_dict(got['head']);od=torch.optim.Adam(clone.parameters(),lr=.001);od.load_state_dict(got['opt'])
    y=batch(present[8:16],torch);rng_set(snap['rng'],torch);second=update(candidate,oc,y,'size_plus_local',torch);state=rng_get(torch)
    rng_set(got['rng'],torch);other=update(clone,od,y,'size_plus_local',torch)
    if not equal(second,other,torch) or not equal(candidate.state_dict(),clone.state_dict(),torch) or not equal(oc.state_dict(),od.state_dict(),torch) or not equal(state,rng_get(torch),torch):raise ValueError('Local candidate Adam/RNG continuation')
    write(out/'control_equivalence.json',dict(checks=checks,candidate_first=first,candidate_second=second,candidate_optimizer_rng_continuation_exact=True,discarded=True,scope='Same original forward, bit-exact control loss/Adam two-step real-cache update; full24control verified source/init/schedule/results reused.'))
    del a,b,oa,ob,x,candidate,oc,clone,od,y;torch.cuda.empty_cache();seed(torch)


def train(args,out,torch):
    from crane_project.utils import port_size_local_vectors_v1 as model
    from crane_project.tools.run_port_size_local_vectors_v1 import batch,evaluate
    from crane_project.tools import run_port_geometry_size_distribution_v1 as prior
    from crane_project.utils import port_geometry_midpoint_sigma_v1 as frozen_model
    from crane_project.utils import port_geometry_size_distribution_metrics_v1 as metrics
    from crane_project.tools.run_port_geometry_size_condition_v1 import components
    from crane_project.tools.diagnose_port_geometry_midpoint_edge_feasibility_v1 import state_digest
    diag=read(args.diagnosis/'diagnosis.json');done=read(args.diagnosis/'finished.json')
    if diag['status']!=done['status'] or done['status']!='COMPLETE':raise ValueError('Real mechanism diagnosis incomplete')
    for n,s in read(args.diagnosis/'artifacts.json')['files'].items():
        if sha(args.diagnosis/n)!=s:raise ValueError('Diagnosis SHA')
    old=read(args.parent/'training.json');control=old['arms']['size_only'];original_index=read(args.parent/'artifacts.json')['files']
    for n,s in original_index.items():
        if sha(args.parent/n)!=s:raise ValueError('Reused original result drift')
    historical,_=prior.historical_identity();sealed=prior.sealed
    cache=sealed.checked_cache(args.cache,historical);selection,refs,_=sealed.checked_selection(args.head,historical)
    frozen=sealed.load_head(torch,frozen_model,args.head,selection,cache,'cuda:0')
    tr,va,bytes_read=prior.load_records(args.cache,cache,frozen,'cuda:0',torch,model,refs)
    control_equivalence(tr,out,torch)
    seed(torch);h=model.LocalVectorSizeHead().cuda();initial=state_digest(h)
    if initial!=control['initial_state_digest']:raise ValueError('Control initial state differs')
    opt=torch.optim.Adam(h.parameters(),lr=.001,weight_decay=0.);present=[r for r in tr if len(r['midpoint_original'])]
    for data in (tr,va):
        if any(r['midpoint']!=r['edge_residual'] for r in evaluate(h,data,torch)):raise ValueError('Full neutral identity')
    report=dict(protocol=VERSION,status='RUNNING',control_reference=str(args.parent.relative_to(ROOT)),control_artifacts_sha256=sha(args.parent/'artifacts.json'),control=control,
                diagnosis_artifacts_sha256=sha(args.diagnosis/'artifacts.json'),initial_state_digest=initial,formal_updates=0,epochs=[],test_reads=0,depth_reads=0,feature_extractions=0,source_commit=args.source_commit)
    for epoch in range(1,25):
        began=time.monotonic();ids=list(range(len(present)));random.Random(1703+epoch).shuffle(ids)
        order=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
        if order!=control['epochs'][epoch-1]['schedule_sha256']:raise ValueError('Control schedule differs')
        sums={k:0. for k in ('loss','size_loss','local_loss')};count=0
        for start in range(0,len(ids),8):
            rr=[present[i] for i in ids[start:start+8]];x=batch(rr,torch);step=update(h,opt,x,'size_plus_local',torch)
            for k in sums:sums[k]+=step[k]*len(rr)
            count+=len(rr);report['formal_updates']+=1
            if start%1600==0:print(json.dumps(dict(stage='train_local_size',epoch=epoch,view=start,**step)),flush=True)
        v=evaluate(h,va,torch);s=metrics.summary(v);d=near_joint(v);c=components(v);gate=eligibility(s,d)
        rr=evaluate(h,tr,torch);td=dict(size_groups=metrics.train_report(rr),joint=near_joint(rr))
        payload=dict(protocol=VERSION,arm='size_plus_local',epoch=epoch,updates=epoch*639,source_commit=args.source_commit,head_state={k:v.detach().cpu().clone() for k,v in h.state_dict().items()},optimizer_state=deepcopy(opt.state_dict()),rng=rng_get(torch))
        path=out/('head_epoch_%02d.pth'%epoch)
        with path.open('xb') as f:torch.save(payload,f)
        if not equal(payload,torch.load(str(path),map_location='cpu'),torch):raise ValueError('Checkpoint reload')
        ep=dict(epoch=epoch,selection_key=ranking(s,epoch),eligibility=gate,metrics=s,components=c,diagnostics=d,training_losses={k:v/count for k,v in sums.items()},schedule_sha256=order,checkpoint_sha256=sha(path),epoch_seconds=time.monotonic()-began)
        rows_write(out/('epoch_%02d_val.rows.jsonl.gz'%epoch),v);write(out/('epoch_%02d_val.json'%epoch),ep)
        rows_write(out/('epoch_%02d_train.rows.jsonl.gz'%epoch),rr);write(out/('epoch_%02d_train.json'%epoch),td)
        report['epochs'].append(ep);print(json.dumps(dict(stage='epoch_finished',epoch=epoch,eligible=gate['pass_all'],failed=gate['failed'],selection_key=ep['selection_key'],elapsed_s=ep['epoch_seconds'])),flush=True)
    good=[e for e in report['epochs'] if e['eligibility']['pass_all']]
    if report['formal_updates']!=15336:raise ValueError('Budget mismatch')
    report.update(status='COMPLETE',selected=min(good,key=lambda e:e['selection_key'])['epoch'] if good else None,representative=min(report['epochs'],key=lambda e:e['selection_key'])['epoch'],eligible_epochs=[e['epoch'] for e in good]);write(out/'training.json',report)


def main(args):
    out=args.out.resolve()
    if out.parent!=(ROOT/'work_dirs'/VERSION).resolve() or out.exists():raise ValueError('Fresh result-only output')
    out.mkdir(parents=True);stop=None;finished=dict(protocol=VERSION,status='FAILED',started_utc=datetime.now(timezone.utc).isoformat())
    try:
        sources=proof();write(out/'source_identity.json',sources);write(out/'protocol.json',protocol())
        from crane_project.tools.audit_port_size_native_p3_v1 import setup_gpu
        resource,stop,thread=setup_gpu(out,7000,6800,6000);finished['resource']=resource
        import torch
        torch.set_num_threads(1);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(6000*2**20/torch.cuda.get_device_properties(0).total_memory,0);seed(torch)
        train(args,out,torch)
        if proof()!=sources:raise ValueError('End source drift')
        finished.update(status='COMPLETE',allocated_peak_mib=torch.cuda.max_memory_allocated()/2**20)
    except Exception as e:finished['error']=type(e).__name__+': '+str(e);raise
    finally:
        if stop is not None:stop.set();thread.join(timeout=11)
        finished['ended_utc']=datetime.now(timezone.utc).isoformat();write(out/'finished.json',finished)
        write(out/'artifacts.json',dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--source-commit',required=True)
    p.add_argument('--parent',type=Path,default=ROOT/'work_dirs/port_size_local_vectors_v1/train_20261018_v1');p.add_argument('--diagnosis',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=ROOT/'work_dirs/port_results/geometry/port_geometry_midpoint_formal_v1_roi_cache');p.add_argument('--head',type=Path,default=ROOT/'work_dirs/crane_symeood_k1_port_day2night_midpoint_sigma_v1/sigma_1p5')
    main(p.parse_args())
