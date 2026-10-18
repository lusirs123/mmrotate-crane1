#!/usr/bin/env python3
"""S035 fixed V20/G20 full-TRAIN independent single-step interventions.

No VAL tensors, epoch search, historical checkpoint overwrite or accumulated
training. Every batch/objective/optimizer starts from the same frozen parent.
"""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.tools.run_port_size_native_train_v1 import sha,read,write,seed,equal
VERSION='port_size_auxiliary_conflict_v1'
PARENTS=dict(size_only='0a44b4254e5e1b1c929b672421dce40b0acd5ed75671f2767a08134c8aa8ca24',
             size_plus_geometry='0621d3ceb81cbf3fbe407aa22e773db061d6cbd0a5d5e26958138b58317b94bd')


def protocol():
    return dict(protocol=VERSION,parents=PARENTS,data='Full TRAIN5116views only,5112outputs; no VAL tensor/test/depth reads; scales not independent identities.',
        interventions='Independent per-batch temporary copy reset to parent state before every objective and optimizer. Evaluate SAME batch before/after, no accumulated fit, not new trained checkpoints or generalization.',
        schedule='Sorted(scale,image)present views then Random(1703+21)shuffle, batch8,639batches,parentV20/G20fixed.',
        main_optimizer='SGD lr0.001,momentum0,clip10. size gradient,0.1geometry gradient,size+0.1geometry; same fixed lr, no scan.',
        sensitivity='Fresh zero-state Adam lr0.001,clip10 for same3objectives; no historical momentum contamination. Adam normalizes gradient scales:0.1geometry is not0.1step amplitude.',
        decomposition='Actual full geometry output-gradient converted as covector to original coordinates, split commontranslation/oppositepairradial/tangential/paircenter remainder, pulled back and VJP to parameters. SGD only, each weighted0.1. Sum checked; nonlinear finite parameter effects not assumed additive.',
        outcomes='Actual delivered canonical edge absolute relative error before/after; preguard candidates separately, both parent-near<=3% and fixedMnear,domain/scale/video; size beneficial retained/suppressed by combination. EPS1e-7relative for counts; raw means/sums retained.',
        candidate_rule='Do not train merely on loss/cosine. If real updates show geometry damages accurate sizes and/or combination sacrifices size-beneficial updates, assess one local-size-only auxiliary objective; otherwise stop without a new candidate.',
        resource='One physicalGPU2or3,process<7000MiB,stop6800,allocator6000.',
        limits=dict(checkpoint_writes=0,accumulated_training=0,epoch_search=0,val_tensor_reads=0,test_reads=0,depth_reads=0))


def sources():
    from crane_project.tools.run_port_size_local_vectors_v1 import checked_sources
    parent=checked_sources()
    own=['crane_project/utils/port_size_auxiliary_conflict_v1.py','crane_project/tools/audit_port_size_auxiliary_conflict_v1.py','tests/test_port_size_auxiliary_conflict_v1.py']
    return dict(parent=parent,own={n:sha(ROOT/n) for n in own})


def load_train(args,torch):
    from crane_project.tools import run_port_geometry_size_distribution_v1 as prior
    from crane_project.utils import port_size_local_vectors_v1 as model
    sealed=prior.sealed;cache=read(args.cache/'cache_manifest.json')
    if sha(args.cache/'cache_manifest.json')!=sealed.CACHE_SHA:raise ValueError('Original cache SHA')
    artifacts=read(args.parent/'artifacts.json')['files']
    name='baseline_train.rows.jsonl.gz'
    if sha(args.parent/name)!=artifacts[name]:raise ValueError('Parent TRAIN reference SHA')
    with gzip.open(args.parent/name,'rt') as f:reference={(r['scale'],r['image']):r for r in map(json.loads,f)}
    records=[];identity={}
    for label in ('train_s1','train_s05'):
        path=args.cache/(label+'.pt')
        if sha(path)!=cache['files'][path.name]:raise ValueError('TRAIN shard SHA')
        payload=torch.load(str(path),map_location='cpu')
        if payload['identity']!=cache['identity'] or payload['detector_state']!=cache['detector_state']:raise ValueError('TRAIN cache identity')
        raw=payload['records']
        if len(raw)!=2558:raise ValueError('TRAIN shard count')
        for r in raw:
            sealed.checked_record(r,label,torch,model.residual.original)
            ref=reference[r['scale'],r['image']]
            if r['gt_original'][0].tolist()!=ref['gt'] or (r['boxes_original'][0].tolist() if len(r['boxes_original']) else None)!=ref['b']:raise ValueError('TRAIN row/B/GT identity')
            rr=dict(r,shard=label,midpoint_original=torch.tensor([ref['midpoint']],dtype=torch.float32) if ref['midpoint'] is not None else torch.empty(0,6))
            records.append(rr)
        identity[label]=cache['files'][path.name]
    if len(records)!=5116 or sum(bool(len(r['midpoint_original'])) for r in records)!=5112:raise ValueError('TRAIN counts')
    return sorted(records,key=lambda r:(r['scale'],r['image'])),identity


def components(h,output,loss_size,loss_geo,x,torch):
    from crane_project.utils.port_size_auxiliary_conflict_v1 import project_geometry_gradient
    params=list(h.parameters())
    gs=torch.autograd.grad(loss_size,params,retain_graph=True)
    gg=torch.autograd.grad(loss_geo,params,retain_graph=True)
    gv=torch.autograd.grad(loss_geo,output['vectors'],retain_graph=True)[0]
    parts=project_geometry_gradient(output['vectors'],gv,*x[2:5])
    error=float((sum(parts.values())-gv).abs().max())
    if error>1e-6+float(gv.abs().max())*2e-5:raise ValueError('Gradient decomposition mismatch')
    gradients=dict(size=gs,geometry=tuple(.1*g for g in gg),combined=tuple(s+.1*g for s,g in zip(gs,gg)))
    for key,value in parts.items():gradients['geo_'+key]=tuple(.1*g for g in torch.autograd.grad(output['vectors'],params,grad_outputs=value,retain_graph=True))
    pe=max(float((sum(gradients['geo_'+k][j] for k in parts)-gradients['geometry'][j]).abs().max()) for j in range(len(params)))
    if pe>1e-6+max(float(g.abs().max()) for g in gradients['geometry'])*2e-5:raise ValueError('Parameter VJP mismatch')
    dot=sum(float((s.double()*g.double()).sum()) for s,g in zip(gs,gg))
    sn=sum(float(g.double().square().sum()) for g in gs)**.5;gn=sum(float(g.double().square().sum()) for g in gg)**.5
    return gradients,dict(vector_sum_max_abs=error,parameter_sum_max_abs=pe,size_grad_norm=sn,geometry_grad_norm=gn,gradient_cosine=dot/(sn*gn) if sn*gn else None)


def run(args,out,torch):
    from crane_project.utils import port_size_local_vectors_v1 as model
    from crane_project.tools.run_port_size_local_vectors_v1 import batch
    records,identity=load_train(args,torch);present=[r for r in records if len(r['midpoint_original'])]
    ids=list(range(len(present)));random.Random(1724).shuffle(ids)
    write(out/'schedule.json',dict(ids=ids,views=[dict(image=r['image'],scale=r['scale']) for r in present],sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest()))
    report=dict(protocol=VERSION,status='RUNNING',train_frames=2558,train_views=5116,present=5112,missing=[dict(image=r['image'],scale=r['scale']) for r in records if not len(r['midpoint_original'])],cache=identity,parents={},interventions_per_batch=10)
    for parent,expected in PARENTS.items():
        ck=args.parent/parent/'head_epoch_20.pth'
        if sha(ck)!=expected:raise ValueError('Frozen parent SHA')
        payload=torch.load(str(ck),map_location='cpu');initial=deepcopy(payload['head_state'])
        if payload['epoch']!=20 or payload['arm']!=parent:raise ValueError('Frozen parent identity')
        ref_path=args.parent/parent/'epoch_20_train.rows.jsonl.gz';index=read(args.parent/'artifacts.json')['files']
        if sha(ref_path)!=index[str(ref_path.relative_to(args.parent))]:raise ValueError('Parent saved TRAIN output SHA')
        with gzip.open(ref_path,'rt') as f:reference={(r['scale'],r['image']):r for r in map(json.loads,f)}
        h=model.LocalVectorSizeHead().cuda();start_time=time.monotonic();max_replay=0.;steps=[]
        with gzip.open(out/(parent+'_responses.jsonl.gz'),'xt') as f:
            for start in range(0,len(ids),8):
                rr=[present[j] for j in ids[start:start+8]];x=batch(rr,torch);h.load_state_dict(initial);h.eval()
                output=h(*x[:-1]);_,ls,lg=model.losses(output,x[-1],x[1],*x[2:6],'size_plus_geometry')
                if not bool(output['vector_valid'].all()):raise ValueError('Invalid parent vectors')
                before=output['boxes_original'].detach().cpu().tolist();pre=output['candidate_original'].detach().cpu().tolist()
                for j,r in enumerate(rr):
                    ref=x[0].new_tensor(reference[r['scale'],r['image']]['edge_residual']);err=float((output['boxes_original'][j]-ref).abs().max());max_replay=max(max_replay,err)
                    if not torch.allclose(output['boxes_original'][j],ref,atol=1e-4,rtol=1e-6):raise ValueError('Parent saved output replay')
                gradients,diag=components(h,output,ls,lg,x,torch)
                if any(not bool(torch.isfinite(g).all()) for gs in gradients.values() for g in gs):raise ValueError('Nonfinite intervention gradient')
                responses={};batch_record=dict(batch=start//8,size_loss=float(ls.detach()),geometry_loss=float(lg.detach()),**diag)
                del output,ls,lg
                for optimizer in ('sgd','fresh_adam'):
                    names=list(gradients) if optimizer=='sgd' else ['size','geometry','combined']
                    for name in names:
                        h.load_state_dict(initial)
                        opt=torch.optim.SGD(h.parameters(),lr=.001) if optimizer=='sgd' else torch.optim.Adam(h.parameters(),lr=.001)
                        opt.zero_grad()
                        for p,g in zip(h.parameters(),gradients[name]):p.grad=g.detach().clone()
                        norm=float(torch.nn.utils.clip_grad_norm_(h.parameters(),10.));opt.step()
                        with torch.no_grad():
                            after=h(*x[:-1]);_,s,g=model.losses(after,x[-1],x[1],*x[2:6],'size_plus_geometry')
                            responses[optimizer+'/'+name]=[dict(box=after['boxes_original'][j].cpu().tolist(),pre_guard=after['candidate_original'][j].cpu().tolist(),accepted=bool(after['accepted'][j]),valid=bool(after['vector_valid'][j])) for j in range(len(rr))]
                            batch_record[optimizer+'/'+name]=dict(size_loss=float(s),geometry_loss=float(g),grad_norm=norm)
                        del after,opt
                for j,r in enumerate(rr):
                    v=dict(image=r['image'],sequence=r['sequence'],domain=r['domain'],scale=r['scale'],batch=start//8,gt=r['gt_original'][0].tolist(),midpoint=r['midpoint_original'][0].tolist(),before=before[j],pre_guard_before=pre[j],after={k:v[j] for k,v in responses.items()})
                    f.write(json.dumps(v,allow_nan=False)+'\n')
                steps.append(batch_record)
                if start%800==0 or start+8==len(ids):print(json.dumps(dict(stage='independent_steps',parent=parent,views=start+8,total=len(ids),elapsed_s=time.monotonic()-start_time)),flush=True)
                del x,gradients
        h.load_state_dict(initial)
        if not equal(h.state_dict(),initial,torch) or sha(ck)!=expected:raise ValueError('Frozen parent mutation')
        write(out/(parent+'_steps.json'),steps);report['parents'][parent]=dict(checkpoint_sha256=expected,batches=len(steps),views=len(present),max_saved_output_replay_abs=max_replay,elapsed_s=time.monotonic()-start_time)
        del h,payload;torch.cuda.empty_cache()
    report['status']='COMPLETE';write(out/'diagnosis.json',report)


def main(args):
    out=args.out.resolve()
    if out.parent!=(ROOT/'work_dirs'/VERSION).resolve() or out.exists():raise ValueError('Fresh result-only directory')
    out.mkdir(parents=True);stop=None;finished=dict(protocol=VERSION,status='FAILED',started_utc=datetime.now(timezone.utc).isoformat())
    try:
        proof=sources();write(out/'sources.json',proof);write(out/'protocol.json',protocol())
        from crane_project.tools.audit_port_size_native_p3_v1 import setup_gpu
        resource,stop,thread=setup_gpu(out,7000,6800,6000);finished['resource']=resource
        import torch
        torch.set_num_threads(1);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(6000*2**20/torch.cuda.get_device_properties(0).total_memory,0);seed(torch)
        run(args,out,torch)
        if sources()!=proof:raise ValueError('End source drift')
        finished.update(status='COMPLETE',allocated_peak_mib=torch.cuda.max_memory_allocated()/2**20)
    except Exception as e:finished['error']=type(e).__name__+': '+str(e);raise
    finally:
        if stop is not None:stop.set();thread.join(timeout=11)
        finished['ended_utc']=datetime.now(timezone.utc).isoformat();write(out/'finished.json',finished)
        write(out/'artifacts.json',dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--parent',type=Path,default=ROOT/'work_dirs/port_size_local_vectors_v1/train_20261018_v1')
    p.add_argument('--cache',type=Path,default=ROOT/'work_dirs/port_results/geometry/port_geometry_midpoint_formal_v1_roi_cache')
    main(p.parse_args())
