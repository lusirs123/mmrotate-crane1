#!/usr/bin/env python3
"""Pinned TRAIN replay of restart decision, zero optimizer/parameter updates."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def run(source,out):
    import torch
    from mmcv import Config
    from mmcv.runner import build_optimizer
    from mmcv.utils import import_modules_from_strings
    from mmrotate.models import build_detector
    from mmrotate.core.hooks.port_oof_trace_hook import cpu_tree,restore_rng,restore_runtime,sum_loss
    from crane_project.tools.run_port_reliability_oof_v1 import sha
    from crane_project.utils.port_oof_task_budget_v1 import task_budget_backward,release_loss_graphs,check_task_config
    from crane_project.utils.port_oof_momentum_restart_v1 import configure,momentum_restart_decision,VERSION
    from crane_project.utils.port_reliability_oof_prevention_v1 import reasons
    from types import SimpleNamespace
    folder = source/'result/A/prevention'
    completion = json.loads((folder/'completion.json').read_text())
    path = folder/'before_anomaly.pt';pin = sha(path)
    saved = torch.load(str(path),map_location='cpu')
    expected = json.loads((folder/'updates.jsonl').read_text().splitlines()[-1])
    attribution = json.loads((source/'classification_attribution_v4/report.json').read_text())
    if (completion['snapshot_sha256']['before_anomaly.pt'] != pin or not saved['diagnostic_only'] or
            saved['iteration'] != completion['completed_updates'] or expected['step'] != saved['iteration']+1 or
            expected['optimizer_updated'] or attribution['snapshot_sha256'] != pin or
            attribution['parameter_updates'] != 0 or not attribution['optimizer_unchanged']):
        raise ValueError('Verified pre-update TRAIN attribution required')
    cfg = Config.fromfile(str(source/'result/A/detector_config.py'));check_task_config(cfg)
    configure(cfg);import_modules_from_strings(**cfg.custom_imports)
    spec = deepcopy(cfg.model);spec.backbone.pop('init_cfg',None);spec.backbone.pretrained = None
    torch.cuda.set_device(0)
    model = build_detector(spec,train_cfg=cfg.get('train_cfg'),test_cfg=cfg.get('test_cfg')).cuda()
    model.load_state_dict(saved['model'],strict=True)
    opt = build_optimizer(model,cfg.optimizer);opt.load_state_dict(saved['optimizer'])
    def device(v):
        if isinstance(v,torch.Tensor):return v.cuda()
        if isinstance(v,dict):return {k:device(x) for k,x in v.items()}
        if isinstance(v,(tuple,list)):return type(v)(device(x) for x in v)
        return v
    batch = {k:cpu_tree(v) if k=='img_metas' else device(v) for k,v in saved['batch'].items()}
    captured = [];original = model.forward_train
    def forward(*a,**k):
        value = original(*a,**k);captured.append(value);return value
    model.forward_train = forward
    restore_runtime(model,saved['runtime']);restore_rng(saved['rng'])
    value = model.train_step(batch,opt);loss = captured[0]
    terms = dict(main_cls=sum_loss(loss['loss_cls']),main_reg=sum_loss(loss['loss_bbox']))
    terms['auxiliary'] = sum(sum_loss(v) for k,v in loss.items() if 'loss' in k and k not in ('loss_cls','loss_bbox'))
    losses = {k:float(v.detach()) for k,v in terms.items()};losses['total'] = float(value['loss'].detach())
    if any(abs(losses[k]-expected['losses'][k])>1e-5 for k in expected['losses']):
        raise ValueError('Original anomaly loss must reproduce before repair decision')
    params = [p for p in model.parameters() if p.requires_grad]
    opt.zero_grad(set_to_none=True)
    stats = task_budget_backward(terms,params)
    holder = SimpleNamespace(losses=loss,_regression_loss=terms['main_reg'])
    release_loss_graphs(SimpleNamespace(outputs=value),holder)
    captured.clear();del terms,loss,holder
    torch.nn.utils.clip_grad_norm_(params,10.)
    decision = momentum_restart_decision(opt)
    if abs(decision['gradient_momentum_dot']-attribution['read_only_SGD_direction']['fresh_gradient_momentum_dot'])>1e-3:
        raise ValueError('Production restart dot differs from independent CPU attribution')
    # Deliberately do not apply restart or call step: the original loss guard
    # would still stop this batch before backward in production.
    def identical(a,b):
        if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
        if isinstance(a,dict):return a.keys()==b.keys() and all(identical(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(identical(x,y) for x,y in zip(a,b))
        return a==b
    unchanged = all(torch.equal(p.detach().cpu(),saved['model'][n]) for n,p in model.named_parameters())
    if not unchanged or not identical(cpu_tree(opt.state_dict()),saved['optimizer']) or sha(path)!=pin:
        raise ValueError('Zero-update check changed parameters/optimizer/snapshot')
    torch.cuda.synchronize()
    if torch.cuda.max_memory_reserved()>4*2**30:
        raise RuntimeError('Fixed4GiB diagnostic limit')
    direction = attribution['read_only_SGD_direction']['first_order_classification_loss_change']
    report = dict(momentum_restart_revision=VERSION,snapshot_sha256=pin,step=expected['step'],losses=losses,
        budgeted_gradients=stats,restart_decision=decision,
        first_order_cls_change_before_restart=sum(direction.values()),
        first_order_cls_change_if_restart=sum(v for k,v in direction.items() if k!='momentum'),
        production_guard_alarms=reasons(losses,stats['gradient_before_clip']),
        optimizer_updates=0,parameters_unchanged=True,optimizer_unchanged=True,
        original_snapshot_unchanged=True,diagnostic_only=True,prevention_passed=False,
        restart_applied=False,weights_saved=False,VAL_scored=False,TEST_read=False,
        peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
        peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20)
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report),flush=True)


def main():
    p = argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a = p.parse_args();source,out = a.source.resolve(),a.out.resolve()
    if any(ROOT/'work_dirs' not in x.parents for x in (source,out)):raise ValueError('Project work_dirs only')
    out.mkdir(parents=True,exist_ok=False);run(source,out)


if __name__ == '__main__':main()
