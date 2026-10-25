"""Bounded cross-entry audit with actual DDP, LR/optimizer and USG hooks.

Two diagnostic mini-epochs of ten updates, never a formal run or VAL selector.
TRAIN sampler prefixes are frozen before executing any model path.
"""
import argparse
import collections
import copy
import gc
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def digest_bytes(b):return hashlib.sha256(b).hexdigest()
def dump(path,row):
    with Path(path).open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')


def rng_digest():
    import numpy as np,torch
    n=np.random.get_state()
    h=hashlib.sha256(json.dumps([random.getstate(),[n[0],n[1].tolist(),n[2],n[3],n[4]]]).encode())
    h.update(torch.get_rng_state().numpy().tobytes());h.update(torch.cuda.get_rng_state().cpu().numpy().tobytes())
    return h.hexdigest()


def input_digest(data):
    from crane_project.utils.symeood_usg_v1 import tensor_digest
    image=data['img'].data[0];items=[('img',image)]
    for key in ('gt_bboxes','gt_labels'):items.extend((key+str(i),t) for i,t in enumerate(data[key].data[0]))
    meta=[{k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in r.items() if k in ('filename','img_shape','pad_shape','scale_factor','flip','flip_direction','port_shrink_scale')} for r in data['img_metas'].data[0]]
    return digest_bytes((tensor_digest(items)+json.dumps(meta,sort_keys=True)).encode())


def config(group,kind):
    from mmcv import Config
    reg='iou' if group[1]=='0' else 'kld';obj='u' if group[-1]=='0' else 'g'
    if kind.startswith('old'):p=ROOT/'work_dirs/symeood_ablation_v1'/group/('crane_symeood_k1_ablation_'+group+'_v1.py')
    else:p=ROOT/'work_dirs/symeood_usg_v1'/('s0_'+reg+'_'+obj)/('crane_symeood_usg_'+reg+'_'+obj+'_v1.py')
    return Config.fromfile(str(p))


def loader(dataset,cfg):
    from mmdet.datasets import build_dataloader
    return build_dataloader(dataset,samples_per_gpu=2,workers_per_gpu=2,dist=True,shuffle=True,seed=0,
                            runner_type='EpochBasedRunner',persistent_workers=False)


def prepare(out,rank):
    import torch
    from mmdet.datasets import build_dataset
    from mmdet.apis import set_random_seed
    from crane_project.utils.symeood_usg_v1 import rng_state,restore_rng
    cfg=config('100','old');ds=build_dataset(cfg.data.train);assert len(ds)==2558
    state=torch.load(out/'identity/initial_100.pth',map_location='cpu')
    rows=[];dest=out/'inputs';dest.mkdir(exist_ok=True)
    for phase,epochs in [('early',[0,1]),('mature',[23,24])]:
        set_random_seed(0,deterministic=True);torch.set_rng_state(state['cpu_rng'])
        dl=loader(ds,cfg)
        for ei,epoch in enumerate(epochs):
            dl.sampler.set_epoch(epoch);iterator=iter(dl)
            try:
                for i in range(10):
                    data=next(iterator);file=dest/('%s_rank%d_step%02d.pt'%(phase,rank,ei*10+i))
                    assert not file.exists()
                    plain={k:v.data[0] for k,v in data.items()}
                    torch.save(plain,file)
                    rows.append(dict(phase=phase,rank=rank,step=ei*10+i,sampler_epoch=epoch,
                        input_sha256=input_digest(data),file=str(file),file_sha256=digest_bytes(file.read_bytes()),
                        names=[m['filename'] for m in plain['img_metas']]))
            finally:iterator._shutdown_workers()
    with (out/('inputs_rank%d.json'%rank)).open('x') as f:json.dump(rows,f,indent=2)
    print('INPUTS_FROZEN',rank,len(rows),flush=True)


class Tape:
    def __init__(self,model):
        self.model=model;self.enabled=True;self.clear()
        head=model.bbox_head
        self.h=head.register_forward_hook(self.forward)
        self.oldloss=head.loss_cls.forward;self.oldassign=head.assigner.assign;self.oldtrain=model.forward_train
        def loss(*args,**kwargs):
            value=self.oldloss(*args,**kwargs)
            if self.enabled:
                labels=args[1];weight=kwargs['weight'];gt=kwargs['gt_bboxes']
                self.labels.extend([labels.detach().flatten(),weight.detach().flatten(),gt.detach().flatten()])
                self.taus.append(float(head.loss_cls._get_current_tau()))
            return value
        def assign(*args,**kwargs):
            result=self.oldassign(*args,**kwargs)
            if self.enabled:
                self.assign_int.extend([result.gt_inds.detach().flatten(),result.labels.detach().flatten()])
                if result.max_overlaps is not None:self.assign_float.append(result.max_overlaps.detach().flatten())
                self.positive.append(int((result.gt_inds>0).sum()))
            return result
        def train(*args,**kwargs):
            result=self.oldtrain(*args,**kwargs)
            if self.enabled:self.losses={k:[x.detach() for x in v] if isinstance(v,list) else [v.detach()] for k,v in result.items() if 'loss' in k}
            return result
        head.loss_cls.forward=loss;head.assigner.assign=assign;model.forward_train=train

    def clear(self):
        self.scores=[];self.predictions=[];self.labels=[];self.assign_int=[];self.assign_float=[];self.positive=[];self.taus=[];self.losses={}

    def forward(self,m,i,o):
        if not self.enabled:return
        for t in o[0]:t.retain_grad()
        self.scores.extend(o[0]);self.predictions.extend(t.detach().flatten().clone() for xs in o for t in xs)

    def close(self):
        h=self.model.bbox_head;self.h.remove();h.loss_cls.forward=self.oldloss;h.assigner.assign=self.oldassign;self.model.forward_train=self.oldtrain;self.clear()


def flat(values,device):
    import torch
    values=list(values)
    return torch.cat([x.detach().flatten() for x in values]) if values else torch.empty(0,device=device)


def state_vectors(model,optimizer):
    import torch
    ps=list(model.parameters());device=ps[0].device
    return dict(parameters=flat(ps,device),momentum=flat([optimizer.state.get(p,{}).get('momentum_buffer',torch.zeros_like(p)) for p in ps],device))


def vector_difference(a,b):
    import torch
    if a.shape!=b.shape:return dict(equal=False,shape_mismatch=True,reference=list(a.shape),peer=list(b.shape))
    if not len(a):return dict(equal=True,max_abs=0.,different=0)
    diff=(a-b).abs()
    return dict(equal=bool(torch.equal(a,b)),max_abs=float(diff.max()),different=int(torch.count_nonzero(diff)))


def setup_path(kind,group,phase,out,common,rank,deterministic):
    import torch
    from mmcv.parallel import MMDistributedDataParallel
    from mmcv.runner import EpochBasedRunner,build_optimizer,OptimizerHook
    from mmcv.runner.hooks.lr_updater import StepLrUpdaterHook
    from mmdet.models import build_detector
    from mmdet.apis import set_random_seed
    from crane_project.utils.symeood_usg_v1 import USGContractHookV1,rng_state,tensor_digest
    cfg=config(group,kind);set_random_seed(0,deterministic=deterministic)
    if not deterministic:torch.backends.cudnn.deterministic=False
    model=build_detector(copy.deepcopy(cfg.model));model.load_state_dict(common['state_dict'],strict=True)
    start=0 if phase=='early' else int(common['meta']['iter'])
    model.bbox_head.assigner._local_call_count=2*start
    model.bbox_head.loss_cls._local_iter.fill_(5*start)
    model.cuda().train();wrapped=MMDistributedDataParallel(model,device_ids=[torch.cuda.current_device()],broadcast_buffers=False,find_unused_parameters=cfg.get('find_unused_parameters',False))
    opt=build_optimizer(wrapped,cfg.optimizer)
    if phase=='mature':opt.load_state_dict(common['optimizer'])
    work=out/kind;work.mkdir(parents=True,exist_ok=True)
    runner=EpochBasedRunner(wrapped,optimizer=opt,work_dir=str(work),logger=logging.getLogger('cross_short'),meta={'seed':0},max_epochs=26)
    lr=StepLrUpdaterHook(**{k:v for k,v in cfg.lr_config.items() if k!='policy'});lr.before_run(runner)
    class Capture(OptimizerHook):
        def clip_grads(self,params):
            ps=list(params)
            self.none=[p.grad is None for p in ps]
            self.raw=flat([p.grad if p.grad is not None else torch.zeros_like(p) for p in ps],ps[0].device)
            self.nonzero=sum(p.grad is not None and bool(p.grad.count_nonzero()) for p in ps)
            assert self.nonzero>0 and bool(torch.isfinite(self.raw).all())
            return super().clip_grads(ps)
    capture=Capture(grad_clip=copy.deepcopy(cfg.optimizer_config.grad_clip))
    tape=Tape(model);hook=None
    if kind=='full':
        hc=copy.deepcopy(cfg.custom_hooks[0]);hc.pop('type');hc.pop('priority')
        hook=USGContractHookV1(**hc)
        # Mature counters/weights are an injected diagnostic state, not a
        # claim that old unsaved assigner/RNG state has been recovered.
        hook.before_run(runner)
    runner._iter=start
    return dict(kind=kind,cfg=cfg,model=model,wrapped=wrapped,opt=opt,runner=runner,lr=lr,capture=capture,tape=tape,hook=hook,start=start,work=work)


def compare_snapshot(reference,peer):
    result={};first=None
    order=['input','rng_before','predictions','assignment_int','assignment_float','labels','losses','logits_gradient','parameter_gradient','clip','momentum','parameters','buffers','counters','rng_after']
    for key in order:
        a,b=reference[key],peer[key]
        if hasattr(a,'shape'):diff=vector_difference(a,b)
        else:diff=dict(equal=a==b,reference=a,peer=b)
        result[key]=diff
        if not diff['equal'] and first is None:first=key
    return dict(equal=first is None,first_field=first,fields=result)


def full_hook_epoch(s,rank,out):
    import torch
    from crane_project.utils.symeood_usg_v1 import rng_state,restore_rng,tensor_digest
    hook=s['hook']
    if hook is None:return
    model=s['model'];ps=list(model.parameters());tape=s['tape']
    tape.clear();tape.enabled=False
    before=dict(parameters=tensor_digest(model.named_parameters()),buffers=tensor_digest(model.named_buffers()),
                counters=[model.bbox_head.assigner._local_call_count,int(model.bbox_head.loss_cls._local_iter)],rng=rng_digest(),
                modes=[m.training for m in model.modules()],grads=flat([p.grad for p in ps if p.grad is not None],ps[0].device).clone())
    try:hook.after_train_epoch(s['runner'])
    finally:tape.enabled=True
    after=dict(parameters=tensor_digest(model.named_parameters()),buffers=tensor_digest(model.named_buffers()),
               counters=[model.bbox_head.assigner._local_call_count,int(model.bbox_head.loss_cls._local_iter)],rng=rng_digest(),modes=[m.training for m in model.modules()])
    checks={k:before[k]==after[k] for k in after}
    checks['parameter_gradients']=torch.equal(before['grads'],flat([p.grad for p in ps if p.grad is not None],ps[0].device))
    dump(out/('hook_conservation_rank%d.jsonl'%rank),dict(epoch=s['runner'].epoch+1,passed=all(checks.values()),checks=checks,
        probe_empty=model.bbox_head.probe is None,records_empty=len(model.bbox_head.loss_cls.records)==0))
    assert all(checks.values()),checks


def run_case(group,phase,mode,out,rank,steps=20):
    import torch
    from mmdet.datasets import build_dataset
    from mmdet.apis import set_random_seed
    from crane_project.utils.symeood_usg_v1 import rng_state,restore_rng,tensor_digest
    deterministic=mode=='det';set_random_seed(0,deterministic=deterministic)
    torch.backends.cudnn.deterministic=deterministic;torch.backends.cudnn.benchmark=False
    common=torch.load(out/'identity'/('initial_'+group+'.pth'),map_location='cpu') if phase=='early' else torch.load(ROOT/'work_dirs/symeood_ablation_v1'/group/'epoch_24.pth',map_location='cpu')
    case=out/'short'/mode/group/phase
    if rank==0:case.mkdir(parents=True,exist_ok=False)
    torch.distributed.barrier()
    kinds=['old','bare','full'] if deterministic else ['old','old_repeat','bare','full']
    paths=[setup_path(k,group,phase,case,common,rank,deterministic) for k in kinds]
    assert len({tensor_digest(s['model'].named_parameters()) for s in paths})==1
    # Independent model instances, common post-initialization random state.
    assert len({next(s['model'].parameters()).data_ptr() for s in paths})==len(paths)
    set_random_seed(0,deterministic=deterministic)
    torch.set_rng_state(torch.load(out/'identity/initial_100.pth',map_location='cpu')['cpu_rng'])
    start_rng=rng_state()
    ds=build_dataset(config(group,'old').data.train)
    for s in paths:s['rng']=copy.deepcopy(start_rng);s['loader']=loader(ds,s['cfg'])
    expected={r['step']:r for r in json.loads((out/('inputs_rank%d.json'%rank)).read_text()) if r['phase']==phase}
    first={};trace=[]
    for epoch_index in range(2):
        epoch=(0 if phase=='early' else 23)+epoch_index
        for s in paths:
            restore_rng(s['rng']);s['wrapped'].train();s['runner']._epoch=epoch;s['runner'].data_loader=s['loader']
            s['loader'].sampler.set_epoch(epoch);s['lr'].before_train_epoch(s['runner'])
            if s['hook']:s['hook'].before_train_epoch(s['runner'])
            s['iterator']=iter(s['loader']);s['rng']=rng_state()
        for inner in range(10):
            step=epoch_index*10+inner;snapshots=[]
            for s in paths:
                restore_rng(s['rng']);data=next(s['iterator']);ih=input_digest(data)
                assert ih==expected[step]['input_sha256'],(group,phase,mode,s['kind'],'input differs',step)
                runner=s['runner'];runner.data_batch=data;runner._inner_iter=inner;runner._iter=s['start']+step
                s['tape'].clear();s['lr'].before_train_iter(runner)
                if s['hook']:s['hook'].before_train_iter(runner)
                before_rng=rng_digest();before=flat(s['model'].parameters(),torch.cuda.current_device()).clone()
                runner.outputs=s['wrapped'].train_step(data,s['opt'])
                s['capture'].after_train_iter(runner)
                if s['hook']:s['hook'].after_train_iter(runner)
                tape=s['tape'];vecs=state_vectors(s['model'],s['opt']);device=before.device
                vecs.update(input=ih,rng_before=before_rng,predictions=flat(tape.predictions,device),
                    assignment_int=flat(tape.assign_int,device),assignment_float=flat(tape.assign_float,device),
                    labels=flat(tape.labels,device),losses=flat([x for k in sorted(tape.losses) for x in tape.losses[k]],device),
                    logits_gradient=flat([t.grad for t in tape.scores],device),parameter_gradient=s['capture'].raw,
                    clip=dict(norm=float(runner.log_buffer.val_history['grad_norm'][-1]),lr=s['opt'].param_groups[0]['lr'],
                              none=s['capture'].none),
                    buffers=tensor_digest(s['model'].named_buffers()),
                    counters=[s['model'].bbox_head.assigner._local_call_count,int(s['model'].bbox_head.loss_cls._local_iter)],rng_after=rng_digest())
                assert bool(torch.isfinite(vecs['parameters']).all()) and bool(torch.isfinite(vecs['losses']).all())
                updated=int(torch.count_nonzero(before-vecs['parameters']));assert updated>0
                scalar=dict(path=s['kind'],step=step,iteration=runner.iter,total_loss=float(runner.outputs['loss']),
                    counters=vecs['counters'],tau=tape.taus,positive=tape.positive,updated_values=updated,
                    nonzero_gradient_tensors=s['capture'].nonzero,
                    losses={k:[float(v) for v in values] for k,values in tape.losses.items()},
                    grad_norm=vecs['clip']['norm'],clip_coefficient=min(1.,10/(vecs['clip']['norm']+1e-6)),lr=vecs['clip']['lr'])
                dump(case/('steps_rank%d.jsonl'%rank),scalar)
                snapshots.append(vecs);s['rng']=rng_state();runner._iter+=1
                del before,data
            for s,snapshot in zip(paths[1:],snapshots[1:]):
                difference=compare_snapshot(snapshots[0],snapshot)
                row=dict(path=s['kind'],step=step,**difference);dump(case/('comparisons_rank%d.jsonl'%rank),row)
                if not difference['equal'] and s['kind'] not in first:
                    first[s['kind']]=row
                    key=difference['first_field']
                    if hasattr(snapshot[key],'shape') and snapshot[key].shape==snapshots[0][key].shape:
                        diff=(snapshot[key]-snapshots[0][key]).abs();index=int(diff.argmax())
                        torch.save(dict(field=key,index=index,reference=snapshots[0][key][max(0,index-128):index+129].cpu(),
                                        peer=snapshot[key][max(0,index-128):index+129].cpu()),case/('first_%s_rank%d.pt'%(s['kind'],rank)))
                        del diff
            trace.append(dict(step=step,all_equal=all(compare_snapshot(snapshots[0],z)['equal'] for z in snapshots[1:])))
            del snapshots,snapshot,vecs,difference
            for s in paths:s['capture'].raw=None;s['tape'].clear()
            if step in (0,9,10,19):print('SHORT',mode,group,phase,rank,step,'first',[(k,v['step'],v['first_field']) for k,v in first.items()],flush=True)
        for s in paths:
            restore_rng(s['rng']);full_hook_epoch(s,rank,case)
            s['rng']=rng_state();s['iterator']._shutdown_workers();del s['iterator']
    (case/('summary_rank%d.json'%rank)).write_text(json.dumps(dict(group=group,phase=phase,mode=mode,steps=20,
        first_divergences=first,all_steps_equal=all(r['all_equal'] for r in trace),
        note='current source, fixed shared inputs; synthetic mini-epochs; no claim of historical24epoch equivalence'),indent=2))
    for s in paths:s['tape'].close()
    del s,tape,runner,paths,common,ds;gc.collect();torch.cuda.empty_cache()


def main(args):
    import torch,torch.distributed as dist
    from mmcv.runner import init_dist
    from mmrotate.utils import setup_multi_processes
    cfg=config('100','old');setup_multi_processes(cfg);init_dist('pytorch');rank=dist.get_rank()
    logging.getLogger('mmcv').setLevel(logging.WARNING)
    out=args.out.resolve()
    if args.stage=='prepare':prepare(out,rank)
    else:
        for group in args.groups:
            for phase in ('early','mature'):run_case(group,phase,args.mode,out,rank)
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=['det','nondet'],default='det');p.add_argument('--groups',nargs='+',default=['100','101','110','111'])
    p.add_argument('--local_rank',type=int,default=0);main(p.parse_args())
