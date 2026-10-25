"""Independent schedule-boundary updates and S contract checks, TRAIN only.

This supplements the frozen 20-step audit; it never runs a VAL selector.
"""
import argparse
import copy
import gc
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'crane_project/tools'))
sys.path.insert(0,str(ROOT))
import audit_symeood_cross_entry_short_v1 as core
import audit_symeood_cross_entry_serial_v1 as serial


def move(x):
    import torch
    if torch.is_tensor(x):return x.cuda()
    if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
    if isinstance(x,list):return [move(v) for v in x]
    if isinstance(x,tuple):return tuple(move(v) for v in x)
    return x


def dc(data):
    from mmcv.parallel import DataContainer
    return dict(img=DataContainer([data['img']],stack=True),
                img_metas=DataContainer([data['img_metas']],cpu_only=True),
                gt_bboxes=DataContainer([data['gt_bboxes']]),gt_labels=DataContainer([data['gt_labels']]))


def update(s,data,iteration,common,random_state):
    import torch
    from crane_project.utils.symeood_usg_v1 import restore_rng,tensor_digest
    model=s['model'];runner=s['runner'];model.load_state_dict(common['state_dict'],strict=True)
    s['opt'].state.clear();s['opt'].zero_grad(set_to_none=True)
    model.bbox_head.assigner._local_call_count=2*iteration
    model.bbox_head.loss_cls._local_iter.fill_(5*iteration)
    runner._iter=iteration;runner._epoch=0;runner._inner_iter=0
    runner.data_loader=SimpleNamespace(num_workers=2,persistent_workers=False)
    data=dc(data);runner.data_batch=data
    if s['hook']:s['hook'].before_train_epoch(runner)
    restore_rng(random_state);s['tape'].clear()
    s['lr'].before_train_epoch(runner);s['lr'].before_train_iter(runner)
    if s['hook']:s['hook'].before_train_iter(runner)
    before_rng=core.rng_digest()
    with torch.autograd.graph.save_on_cpu(pin_memory=False):
        runner.outputs=s['wrapped'].train_step(data,s['opt'])
        s['capture'].after_train_iter(runner)
    if s['hook']:s['hook'].after_train_iter(runner)
    t=s['tape'];v=serial.snapshot(s,core.input_digest(data),before_rng)
    return v,dict(iteration=iteration,path=s['kind'],tau=t.taus,positive=t.positive,
                  counters=v['counters'],total_loss=float(runner.outputs['loss']))


def boundaries(out,rank):
    import torch
    from mmdet.apis import set_random_seed
    from crane_project.utils.symeood_usg_v1 import rng_state
    dest=out/'boundaries';dest.mkdir(exist_ok=True)
    data=torch.load(out/'inputs'/('early_rank%d_step00.pt'%rank),map_location='cpu')
    for group in ('100','101','110','111'):
        common=torch.load(out/'identity'/('initial_'+group+'.pth'),map_location='cpu')
        reference={};exact_reference={}
        for kind in ('old','bare','full'):
            s=core.setup_path(kind,group,'early',dest/group,common,rank,True);serial.cpu_capture(s)
            set_random_seed(0,deterministic=True);random_state=rng_state()
            for iteration in (0,998,999,1000,1998,1999,2000,3998,3999,4000):
                v,row=update(s,data,iteration,common,random_state)
                core.dump(dest/('states_rank%d.jsonl'%rank),dict(group=group,**row))
                value=serial.compact(v)
                if kind=='old':
                    reference[iteration]=value
                    if iteration in (0,4000):exact_reference[iteration]=v
                else:core.dump(dest/('comparisons_rank%d.jsonl'%rank),dict(group=group,iteration=iteration,path=kind,**serial.compare(reference[iteration],value,exact_reference.get(iteration),v)))
                del v,value;s['capture'].raw=None;s['tape'].clear();torch.cuda.empty_cache()
            s['tape'].close();del s;gc.collect();torch.cuda.empty_cache();torch.distributed.barrier()
        del reference,exact_reference,common;gc.collect();torch.cuda.empty_cache()
        print('BOUNDARIES_DONE',group,rank,flush=True)


def s_contract(out,rank):
    import torch
    from mmcv import Config
    from mmdet.models import build_detector
    from mmdet.apis import set_random_seed
    from crane_project.utils.symeood_usg_v1 import rng_state,restore_rng,matched_objectives
    data=move(torch.load(out/'inputs'/('early_rank%d_step00.pt'%rank),map_location='cpu'))
    dest=out/'s_contract';dest.mkdir(exist_ok=True)
    for reg,group in [('iou','100'),('kld','110')]:
        path=ROOT/'work_dirs/symeood_usg_v1'/('s0_'+reg+'_s')/('crane_symeood_usg_'+reg+'_s_v1.py')
        cfg=Config.fromfile(str(path));common=torch.load(out/'identity'/('initial_'+group+'.pth'),map_location='cpu')
        model=build_detector(copy.deepcopy(cfg.model)).train()
        native=build_detector(copy.deepcopy(core.config(group,'old').model)).train()
        for iteration in (0,4000):
            for m in (model,native):
                m.load_state_dict(common['state_dict'],strict=True);m.bbox_head.assigner._local_call_count=2*iteration;m.bbox_head.loss_cls._local_iter.fill_(5*iteration)
            set_random_seed(0,deterministic=True);random_state=rng_state()
            model.cuda()
            h=model.bbox_head;h.capture_probe=True
            with torch.autograd.graph.save_on_cpu(pin_memory=False):losses=model.forward_train(**data)
            p=h.probe
            alpha,expected_global,degenerate=matched_objectives(p['U'],p['G'])
            actual=sum(losses['loss_cls']);actual_dtype=actual.dtype;expected=sum(alpha*r['U'] for r in p['records'])
            pars=[t for t in model.parameters() if t.requires_grad]
            actual_grads=torch.autograd.grad(actual,pars,retain_graph=True,allow_unused=True)
            expected_grads=torch.autograd.grad(expected,pars,retain_graph=True,allow_unused=True)
            all_equal=all(a is None and b is None or a is not None and b is not None and torch.equal(a,b) for a,b in zip(actual_grads,expected_grads))
            g_path=torch.autograd.grad(actual,p['G'],retain_graph=True,allow_unused=True)[0]
            other_values={k:[x.detach().clone() for x in v] for k,v in losses.items() if 'loss' in k and k!='loss_cls'}
            row=dict(regression=reg,iteration=iteration,rank=rank,U=float(p['U']),G=float(p['G']),S=float(actual),alpha=float(alpha),
                level_count=len(p['records']),single_batch_alpha=True,alpha_requires_grad=alpha.requires_grad,
                G_to_actual_gradient_none=g_path is None,per_level_formula_equal=torch.equal(actual,expected),
                all_parameter_gradients_equal=all_equal,classification_G_S_abs=float((p['G']-actual).abs()),
                classification_G_S_relative=float((p['G']-actual).abs()/p['G'].detach().abs().clamp(min=1e-12)),
                global_vs_layer_sum_abs=float((expected_global-actual).abs()),degenerate=degenerate)
            h.probe=None;h.capture_probe=False
            del p,losses,actual,expected,expected_global,actual_grads,expected_grads,pars
            model.cpu();native.cuda();torch.cuda.empty_cache()
            with torch.no_grad():
                restore_rng(random_state);native_losses=native.forward_train(**data)
            other={k:all(torch.equal(x,y) for x,y in zip(v,native_losses[k])) for k,v in other_values.items()}
            row['other_losses_equal']=other
            row['counter_equal']=model.bbox_head.assigner._local_call_count==native.bbox_head.assigner._local_call_count and int(h.loss_cls._local_iter)==int(native.bbox_head.loss_cls._local_iter)
            row['passed']=all([row['level_count']==5,not alpha.requires_grad,g_path is None,row['per_level_formula_equal'],all_equal,all(other.values()),row['counter_equal'],row['classification_G_S_relative']<=4*torch.finfo(actual_dtype).eps])
            core.dump(dest/('checks_rank%d.jsonl'%rank),row)
            del native_losses,other_values
            native.cpu();torch.cuda.empty_cache()
        del model,native,h,common;gc.collect();torch.cuda.empty_cache()
        print('S_CONTRACT_DONE',reg,rank,flush=True)


if __name__=='__main__':
    import torch,torch.distributed as dist
    from mmcv.runner import init_dist
    from mmrotate.utils import setup_multi_processes
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--stage',choices=['boundaries','s_contract'],required=True);p.add_argument('--local_rank',type=int,default=0)
    args=p.parse_args();setup_multi_processes(core.config('100','old'));init_dist('pytorch')
    torch.cuda.set_per_process_memory_fraction(.28)
    logging.getLogger('mmcv').setLevel(logging.WARNING)
    (boundaries if args.stage=='boundaries' else s_contract)(args.out.resolve(),dist.get_rank())
    dist.barrier();dist.destroy_process_group()
