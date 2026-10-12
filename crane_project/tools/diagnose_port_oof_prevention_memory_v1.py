#!/usr/bin/env python3
"""Zero-update allocator trace on an explicitly pinned TRAIN stop, fixed40 repeats."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    import torch
    from mmcv import Config
    from mmcv.runner import build_optimizer
    from mmcv.utils import import_modules_from_strings
    from mmrotate.models import build_detector
    from mmrotate.core.hooks.port_oof_trace_hook import cpu_tree,restore_runtime,restore_rng
    from crane_project.tools.run_port_reliability_oof_v1 import sha
    from crane_project.utils.port_reliability_oof_prevention_v1 import measured_budget_backward
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--expected-step',type=int,default=90)
    parser.add_argument('--all-phases',action='store_true')
    args=parser.parse_args();source=args.source.resolve();out=args.out.resolve()
    if any(ROOT/'work_dirs' not in p.parents for p in (source,out)):
        raise ValueError('Project work_dirs only')
    out.mkdir(parents=True,exist_ok=False)
    trace=source/'result/A/prevention'
    path=trace/'before_anomaly.pt';completion=json.loads((trace/'completion.json').read_text())
    pin=sha(path)
    if pin!=completion['snapshot_sha256'][path.name]:raise ValueError('Snapshot changed')
    saved=torch.load(str(path),map_location='cpu')
    if (saved['iteration']!=args.expected_step-1 or
            completion['completed_updates']!=args.expected_step-1 or not saved['diagnostic_only']):
        raise ValueError('Only explicitly pinned diagnostic TRAIN stop allowed')
    cfg=Config.fromfile(str(source/'result/A/detector_config.py'))
    import_modules_from_strings(**cfg.custom_imports)
    spec=deepcopy(cfg.model);spec.backbone.pop('init_cfg',None);spec.backbone.pretrained=None
    torch.cuda.set_device(0)
    model=build_detector(spec,train_cfg=cfg.get('train_cfg'),test_cfg=cfg.get('test_cfg')).cuda(0)
    model.load_state_dict(saved['model'],strict=True)
    optimizer=build_optimizer(model,cfg.optimizer);optimizer.load_state_dict(saved['optimizer'])
    def device(v):
        if isinstance(v,torch.Tensor):return v.cuda(0)
        if isinstance(v,dict):return {k:device(x) for k,x in v.items()}
        if isinstance(v,(tuple,list)):return type(v)(device(x) for x in v)
        return v
    batch={k:(cpu_tree(v) if k=='img_metas' else device(v)) for k,v in saved['batch'].items()}
    terms=[];original=model.forward_train
    def forward(*a,**k):
        value=original(*a,**k);terms.append(value);return value
    model.forward_train=forward
    parameters=[p for p in model.parameters() if p.requires_grad]
    initial={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
    events=[];records=[];phases=[];last_location=None;last_reserved=torch.cuda.memory_reserved()
    def phase(name):
        torch.cuda.synchronize()
        stats=torch.cuda.memory_stats()
        phases.append(dict(repeat=len(records)+1,phase=name,allocated=torch.cuda.memory_allocated(),
            reserved=torch.cuda.memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),
            inactive_split=stats.get('inactive_split_bytes.all.current'),
            allocation_retries=stats.get('num_alloc_retries')))
        torch.cuda.reset_peak_memory_stats()
    if args.all_phases:
        for name,module in [('backbone',model.backbone),('neck',model.neck),('main_head',model.bbox_head)]+[
                ('aux%d'%i,m) for i,m in enumerate(model.aux_heads)]:
            module.register_forward_pre_hook(lambda m,a,n=name:phase(n+'_before'))
            module.register_forward_hook(lambda m,a,o,n=name:phase(n+'_after'))
    watched={'port_reliability_oof_prevention_v1.py','port_strict_optimizer_hook.py','clip_grad.py'}
    def inspect_line(frame,event,arg):
        nonlocal last_location,last_reserved
        filename=Path(frame.f_code.co_filename)
        selected=(filename.name in watched or (args.all_phases and str(ROOT/'mmrotate/models') in str(filename)))
        if event=='line' and selected:
            current=dict(file=frame.f_code.co_filename,line=frame.f_lineno)
            reserved=torch.cuda.memory_reserved()
            if reserved!=last_reserved:
                stats=torch.cuda.memory_stats()
                events.append(dict(repeat=len(records)+1,previous_line=last_location,current_line=current,
                    reserved_before=last_reserved,reserved_after=reserved,
                    allocated=torch.cuda.memory_allocated(),inactive_split=stats.get('inactive_split_bytes.all.current'),
                    allocation_retries=stats.get('num_alloc_retries')))
            last_location=current;last_reserved=reserved
        return inspect_line
    try:
        for i in range(40):
            model.load_state_dict({n:saved['model'][n] for n,_ in model.named_buffers()},strict=False)
            restore_runtime(model,saved['runtime']);restore_rng(saved['rng'])
            optimizer.zero_grad(set_to_none=True);terms.clear()
            if args.all_phases:
                phase('before_forward');sys.settrace(inspect_line)
            value=model.train_step(batch,optimizer)
            if args.all_phases:phase('after_forward')
            regression=terms[0]['loss_bbox']
            regression=sum(v.mean() for v in regression) if isinstance(regression,(list,tuple)) else regression.mean()
            sys.settrace(inspect_line)
            stats=measured_budget_backward(value['loss'],regression,parameters)
            if args.all_phases:phase('after_budget_backward')
            torch.nn.utils.clip_grad_norm_(parameters,10.)
            if args.all_phases:phase('after_clip')
            sys.settrace(None);torch.cuda.synchronize()
            records.append(dict(repeat=i+1,total_loss=float(value['loss'].detach()),**stats,
                allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
                inactive_split=torch.cuda.memory_stats().get('inactive_split_bytes.all.current')))
            if torch.cuda.memory_reserved()>4*2**30:raise RuntimeError('Fixed4GiB ceiling')
            del value,regression;terms.clear()
        assert all(torch.equal(initial[n],p.detach().cpu()) for n,p in model.named_parameters())
        assert sha(path)==pin
        segments=torch.cuda.memory_snapshot()
        (out/'allocator_segments.json').write_text(json.dumps(segments,indent=2)+'\n')
        report=dict(snapshot_sha256=pin,expected_step=args.expected_step,parameter_updates=0,optimizer_steps=0,
            parameters_unchanged=True,records=records,events=events,phases=phases,
            VAL_scored=False,TEST_read=False,original_snapshot_unchanged=True)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(repeats=len(records),events=len(events),last=records[-1],parameter_updates=0)),flush=True)
    finally:sys.settrace(None)


if __name__=='__main__':main()
