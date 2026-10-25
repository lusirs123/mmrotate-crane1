"""Low-VRAM serial version of the bounded cross-entry TRAIN audit.

One live model/rank. Exact saved tensors round-trip through CPU, without
recomputation, precision conversion, loss/assignment or batch changes.
Tensor SHA equality is exact; bounded samples are NOT a global Linf bound.
"""
import argparse
import collections
import gc
import hashlib
import json
import logging
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'crane_project/tools'))
sys.path.insert(0,str(ROOT))
import audit_symeood_cross_entry_short_v1 as core


def compact(snapshot):
    import torch
    result={}
    for key,value in snapshot.items():
        if not torch.is_tensor(value):result[key]=value;continue
        x=value.detach().cpu().contiguous().flatten()
        stride=max(1,(len(x)+2047)//2048);idx=torch.arange(0,len(x),stride)
        result[key]=dict(tensor=True,shape=list(value.shape),dtype=str(value.dtype),
            sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest(),
            norm=float(x.float().norm()) if len(x) else 0.,
            maximum=float(x.abs().max()) if len(x) else 0.,
            indices=idx.tolist(),samples=x[idx].tolist())
    return result


ORDER=['input','rng_before','predictions','assignment_int','assignment_float','labels','losses','logits_gradient','parameter_gradient','clip','momentum','parameters','buffers','counters','rng_after']


def compare(a,b,exact_reference=None,raw_peer=None):
    import torch
    fields={};first=None
    for key in ORDER:
        x,y=a[key],b[key]
        if isinstance(x,dict) and x.get('tensor'):
            equal=all(x[k]==y[k] for k in ('shape','dtype','sha256'))
            samples=(torch.tensor(x['samples'],dtype=torch.float64)-torch.tensor(y['samples'],dtype=torch.float64)).abs()
            row=dict(equal=equal,reference_sha256=x['sha256'],peer_sha256=y['sha256'],
                reference_norm=x['norm'],peer_norm=y['norm'],sample_max_abs=float(samples.max()) if len(samples) else 0.,
                sample_count=len(samples),global_max_abs=None)
            if exact_reference is not None and raw_peer is not None:
                row.update(core.vector_difference(exact_reference[key],raw_peer[key].cpu()))
                row['global_max_abs']=row.get('max_abs')
        else:row=dict(equal=x==y,reference=x,peer=y)
        fields[key]=row
        if not row['equal'] and first is None:first=key
    return dict(equal=first is None,first_field=first,fields=fields)


def snapshot(s,ih,before_rng):
    import torch
    from crane_project.utils.symeood_usg_v1 import tensor_digest
    t=s['tape'];runner=s['runner'];model=s['model'];opt=s['opt'];device=next(model.parameters()).device
    # Move each vector immediately, rather than retaining three GPU snapshots.
    def cpu(xs):return core.flat(xs,device).cpu()
    ps=list(model.parameters())
    return dict(input=ih,rng_before=before_rng,predictions=cpu(t.predictions),
        assignment_int=cpu(t.assign_int),assignment_float=cpu(t.assign_float),labels=cpu(t.labels),
        losses=cpu([x for k in sorted(t.losses) for x in t.losses[k]]),
        logits_gradient=cpu([z.grad for z in t.scores]),parameter_gradient=s['capture'].raw.cpu(),
        clip=dict(norm=float(runner.log_buffer.val_history['grad_norm'][-1]),lr=opt.param_groups[0]['lr'],none=s['capture'].none),
        parameters=cpu(ps),momentum=cpu([opt.state.get(p,{}).get('momentum_buffer',torch.zeros_like(p)) for p in ps]),
        buffers=tensor_digest(model.named_buffers()),
        counters=[model.bbox_head.assigner._local_call_count,int(model.bbox_head.loss_cls._local_iter)],rng_after=core.rng_digest())


def cpu_capture(s):
    capture=s['capture'];original=capture.clip_grads
    def call(params):
        result=original(params);capture.raw=capture.raw.cpu();return result
    capture.clip_grads=call


def case(out,group,phase,mode,rank):
    import torch
    from mmdet.datasets import build_dataset
    from mmdet.apis import set_random_seed
    from crane_project.utils.symeood_usg_v1 import rng_state,restore_rng
    det=mode=='det'
    previous=out/'short'/mode/group/phase
    if det and all((previous/('summary_rank%d.json'%r)).exists() and json.loads((previous/('summary_rank%d.json'%r)).read_text())['all_steps_equal'] for r in (0,1)):
        print('PRESERVE_COMPLETED',mode,group,phase,rank,flush=True);return
    dest=out/'short_serial'/mode/group/phase
    if rank==0:dest.mkdir(parents=True,exist_ok=False)
    torch.distributed.barrier()
    common=torch.load(out/'identity'/('initial_'+group+'.pth'),map_location='cpu') if phase=='early' else torch.load(ROOT/'work_dirs/symeood_ablation_v1'/group/'epoch_24.pth',map_location='cpu')
    ds=build_dataset(core.config(group,'old').data.train)
    expected={r['step']:r for r in json.loads((out/('inputs_rank%d.json'%rank)).read_text()) if r['phase']==phase}
    reference=[];full_reference={};first={};equal_steps={};kinds=['old','bare','full'] if det else ['old','old_repeat','bare','full']
    for kind in kinds:
        torch.backends.cudnn.deterministic=det;torch.backends.cudnn.benchmark=False
        s=core.setup_path(kind,group,phase,dest,common,rank,det);cpu_capture(s)
        set_random_seed(0,deterministic=det);torch.backends.cudnn.deterministic=det
        torch.set_rng_state(torch.load(out/'identity/initial_100.pth',map_location='cpu')['cpu_rng'])
        dl=core.loader(ds,s['cfg']);runner=s['runner'];runner.data_loader=dl;equal_steps[kind]=[]
        for ei in range(2):
            epoch=(0 if phase=='early' else 23)+ei
            s['wrapped'].train();runner._epoch=epoch;dl.sampler.set_epoch(epoch)
            s['lr'].before_train_epoch(runner)
            if s['hook']:s['hook'].before_train_epoch(runner)
            iterator=iter(dl)
            try:
                for inner in range(10):
                    step=ei*10+inner;data=next(iterator);ih=core.input_digest(data)
                    assert ih==expected[step]['input_sha256'],(kind,group,phase,step,'input mismatch')
                    runner.data_batch=data;runner._iter=s['start']+step;runner._inner_iter=inner;s['tape'].clear()
                    s['lr'].before_train_iter(runner)
                    if s['hook']:s['hook'].before_train_iter(runner)
                    before_rng=core.rng_digest()
                    with torch.autograd.graph.save_on_cpu(pin_memory=False):
                        runner.outputs=s['wrapped'].train_step(data,s['opt'])
                        s['capture'].after_train_iter(runner)
                    if s['hook']:s['hook'].after_train_iter(runner)
                    raw=snapshot(s,ih,before_rng);value=compact(raw)
                    assert bool(torch.isfinite(raw['parameters']).all()) and bool(torch.isfinite(raw['losses']).all())
                    if kind=='old':
                        reference.append(value)
                        if step in (0,10):full_reference[step]=raw
                    else:
                        comparison=compare(reference[step],value,full_reference.get(step),raw)
                        core.dump(dest/('comparisons_rank%d.jsonl'%rank),dict(path=kind,step=step,**comparison))
                        equal_steps[kind].append(comparison['equal'])
                        if not comparison['equal'] and kind not in first:first[kind]=dict(step=step,**comparison)
                    core.dump(dest/('digests_rank%d.jsonl'%rank),dict(path=kind,step=step,**value))
                    core.dump(dest/('steps_rank%d.jsonl'%rank),dict(path=kind,step=step,total_loss=float(runner.outputs['loss']),
                        counters=raw['counters'],tau=s['tape'].taus,positive=s['tape'].positive,grad_norm=raw['clip']['norm'],
                        lr=raw['clip']['lr'],nonzero_gradient_tensors=s['capture'].nonzero,
                        allocated_mib=torch.cuda.memory_allocated()/2**20,reserved_mib=torch.cuda.memory_reserved()/2**20,
                        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
                    runner._iter+=1;s['capture'].raw=None;s['tape'].clear()
                    del raw,value,data
                    torch.cuda.empty_cache()
                    if step in (0,9,10,19):print('SERIAL',mode,group,phase,rank,kind,step,[(k,v['step'],v['first_field']) for k,v in first.items()],flush=True)
            finally:iterator._shutdown_workers()
            # Simulate the evaluation hook's mode change, without VAL inference.
            # ep1 has no scheduled VAL; ep24 does. This does not read VAL data.
            if (epoch+1)%2==0:s['wrapped'].eval()
            with torch.autograd.graph.save_on_cpu(pin_memory=False):core.full_hook_epoch(s,rank,dest)
            torch.cuda.empty_cache()
        s['tape'].close()
        del iterator,dl,runner,s;gc.collect();torch.cuda.empty_cache()
        torch.distributed.barrier()
    (dest/('summary_rank%d.json'%rank)).write_text(json.dumps(dict(group=group,phase=phase,mode=mode,steps=20,
        first_divergences=first,all_steps_equal=all(all(x) for x in equal_steps.values()),
        equal_steps={k:sum(v) for k,v in equal_steps.items()},
        resource_mode='serial model; saved tensors CPU, exact byte SHA; full Linf only steps0/10',
        numerical_limit='Nondeterministic flags are historical; allocator/timing and tensor residency differ. No inference about entire historical24epoch gap.'),indent=2))
    del reference,full_reference,common,ds;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    import torch,torch.distributed as dist
    from mmcv.runner import init_dist
    from mmrotate.utils import setup_multi_processes
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['det','nondet'],required=True);p.add_argument('--local_rank',type=int,default=0)
    args=p.parse_args();setup_multi_processes(core.config('100','old'));init_dist('pytorch')
    torch.cuda.set_per_process_memory_fraction(.28)
    logging.getLogger('mmcv').setLevel(logging.WARNING)
    for group in ('100','101','110','111'):
        for phase in ('early','mature'):case(args.out.resolve(),group,phase,args.mode,dist.get_rank())
    dist.barrier();dist.destroy_process_group()
