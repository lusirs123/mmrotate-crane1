#!/usr/bin/env python3
"""Four identical TRAIN forwards, zero updates; frozen v1 vs memory-only v2."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
REFERENCE_COMMIT='6270e04'
HELPER='crane_project/utils/port_oof_task_budget_v1.py'


def run(source,reference_source,out,arm):
    import torch
    from mmcv import Config
    from mmcv.runner import build_optimizer
    from mmcv.utils import import_modules_from_strings
    from mmrotate.models import build_detector
    from mmrotate.core.hooks.port_oof_trace_hook import cpu_tree,restore_rng,restore_runtime,sum_loss
    from crane_project.tools.run_port_reliability_oof_v1 import sha
    from crane_project.utils.port_reliability_oof_prevention_v1 import check_config
    from crane_project.utils.port_oof_task_budget_v1 import task_budget_backward,release_loss_graphs,VERSION,MEMORY_REVISION
    folder=source/'result/A/prevention'
    completion=json.loads((folder/'completion.json').read_text())
    path=folder/'before_anomaly.pt';pin=sha(path)
    saved=torch.load(str(path),map_location='cpu')
    if (completion['snapshot_sha256'].get(path.name)!=pin or not saved['diagnostic_only'] or
            saved['iteration']!=completion['completed_updates']):raise ValueError('Pinned TRAIN boundary required')
    reference=subprocess.check_output(['git','show',REFERENCE_COMMIT+':'+HELPER],cwd=str(ROOT))
    reference_pin=json.loads((reference_source/'result/prevention.json').read_text())['sources'][HELPER]
    if hashlib.sha256(reference).hexdigest()!=reference_pin:raise ValueError('Frozen old helper source changed')
    namespace={}
    exec(compile(reference,REFERENCE_COMMIT+':'+HELPER,'exec'),namespace)
    backward=namespace['task_budget_backward'] if arm=='reference' else task_budget_backward
    cfg=Config.fromfile(str(source/'result/A/detector_config.py'));check_config(cfg)
    import_modules_from_strings(**cfg.custom_imports)
    spec=deepcopy(cfg.model);spec.backbone.pop('init_cfg',None);spec.backbone.pretrained=None
    torch.cuda.set_device(0)
    model=build_detector(spec,train_cfg=cfg.get('train_cfg'),test_cfg=cfg.get('test_cfg')).cuda()
    model.load_state_dict(saved['model'],strict=True)
    opt=build_optimizer(model,cfg.optimizer);opt.load_state_dict(saved['optimizer'])
    def device(v):
        if isinstance(v,torch.Tensor):return v.cuda()
        if isinstance(v,dict):return {k:device(x) for k,x in v.items()}
        if isinstance(v,(tuple,list)):return type(v)(device(x) for x in v)
        return v
    batch={k:(cpu_tree(v) if k=='img_metas' else device(v)) for k,v in saved['batch'].items()}
    captured=[];original=model.forward_train
    def capture(*a,**k):
        result=original(*a,**k);captured.append(result);return result
    model.forward_train=capture
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    params=[p for n,p in named]
    def gradient_hash():
        h=hashlib.sha256()
        for n,p in named:
            h.update(n.encode())
            if p.grad is None:h.update(b'None')
            else:h.update(p.grad.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    records=[]
    for i in range(4):
        opt.zero_grad(set_to_none=True);captured.clear()
        model.load_state_dict({n:saved['model'][n] for n,b in model.named_buffers()},strict=False)
        restore_runtime(model,saved['runtime']);restore_rng(saved['rng'])
        # Retain previous outputs until this forward returns, as the real runner.
        value=model.train_step(batch,opt)
        terms=dict(main_cls=sum_loss(captured[0]['loss_cls']),main_reg=sum_loss(captured[0]['loss_bbox']))
        terms['auxiliary']=sum(sum_loss(v) for k,v in captured[0].items() if 'loss' in k and k not in ('loss_cls','loss_bbox'))
        losses={k:float(v.detach()) for k,v in terms.items()};losses['total']=float(value['loss'].detach())
        stats=backward(terms,params)
        if arm=='optimized':
            holder=SimpleNamespace(losses=captured[0],_regression_loss=terms['main_reg'])
            release_loss_graphs(SimpleNamespace(outputs=value),holder)
            del holder
        captured.clear();del terms
        before=gradient_hash()
        torch.nn.utils.clip_grad_norm_(params,10.)
        after=gradient_hash()
        torch.cuda.synchronize()
        a=torch.cuda.memory_stats()
        records.append(dict(repeat=i+1,losses=losses,component_gradients=stats['component_gradients'],
            mixed_norm=stats['gradient_before_clip'],gradient_before_clip_sha256=before,
            gradient_after_clip_sha256=after,allocated_MiB=torch.cuda.memory_allocated()/2**20,
            reserved_MiB=torch.cuda.memory_reserved()/2**20,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
            allocation_retries=a.get('num_alloc_retries'),OOM_events=a.get('num_ooms')))
        if torch.cuda.max_memory_reserved()>4*2**30:raise RuntimeError('Fixed4GiB diagnostic limit')
    unchanged=all(torch.equal(p.detach().cpu(),saved['model'][n]) for n,p in model.named_parameters())
    if not unchanged or sha(path)!=pin:raise ValueError('Parameters or snapshot changed')
    report=dict(arm=arm,task_budget_revision=VERSION,task_memory_revision=MEMORY_REVISION if arm=='optimized' else 'v1',
        snapshot_sha256=pin,reference_source_sha256=reference_pin,optimized_source_sha256=sha(ROOT/HELPER),
        records=records,optimizer_updates=0,parameters_unchanged=unchanged,original_snapshot_unchanged=True,
        VAL_scored=False,TEST_read=False,prevention_passed=False)
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(arm=arm,records=records)),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True);p.add_argument('--reference-source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--arm',choices=('reference','optimized'),required=True)
    a=p.parse_args();source,reference,out=a.source.resolve(),a.reference_source.resolve(),a.out.resolve()
    if any(ROOT/'work_dirs' not in x.parents for x in (source,reference,out)):raise ValueError('Project work_dirs only')
    out.mkdir(parents=True,exist_ok=False)
    run(source,reference,out,a.arm)


if __name__=='__main__':main()
