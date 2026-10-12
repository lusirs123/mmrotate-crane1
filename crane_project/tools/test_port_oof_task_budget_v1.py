#!/usr/bin/env python3
"""Zero-update check or64-step counterfactual from a pinned damaged TRAIN state.

Finite replay deliberately records original loss alarms rather than claiming
production guard acceptance. No repaired checkpoint is saved or resumed.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def run(source,out,arm,mode):
    import torch
    from mmcv import Config
    from mmcv.runner import build_optimizer
    from mmcv.utils import import_modules_from_strings
    from mmrotate.models import build_detector
    from mmrotate.core.hooks.port_oof_trace_hook import cpu_tree,restore_rng,restore_runtime,sum_loss
    from crane_project.tools.run_port_reliability_oof_v1 import sha
    from crane_project.utils.port_reliability_oof_prevention_v1 import (
        check_config,measured_budget_backward,reasons,CacheAwareMemoryGuard)
    from crane_project.utils.port_oof_task_budget_v1 import task_budget_backward,release_loss_graphs,VERSION,MEMORY_REVISION
    from types import SimpleNamespace
    from crane_project.utils.port_reliability_oof_followup_v1 import json_safe
    folder=source/'result/A/prevention'
    completion=json.loads((folder/'completion.json').read_text())
    path=folder/'before_anomaly.pt'
    pin=sha(path)
    if pin!=completion['snapshot_sha256'][path.name]:raise ValueError('Pinned snapshot changed')
    saved=torch.load(str(path),map_location='cpu')
    expected=json.loads((folder/'updates.jsonl').read_text().splitlines()[-1])
    if (not saved['diagnostic_only'] or saved['iteration']!=completion['completed_updates'] or
            expected['step']!=saved['iteration']+1 or expected['optimizer_updated']):
        raise ValueError('Reviewed pre-optimizer TRAIN stop required')
    attribution=json.loads((source/'classification_stop_diagnosis/report.json').read_text())
    if (attribution['snapshot_sha256']!=pin or attribution['parameter_updates']!=0 or
            not attribution['parameters_unchanged']):raise ValueError('Verified zero-update attribution required')
    cfg=Config.fromfile(str(source/'result/A/detector_config.py'))
    check_config(cfg)
    import_modules_from_strings(**cfg.custom_imports)
    spec=deepcopy(cfg.model);spec.backbone.pop('init_cfg',None);spec.backbone.pretrained=None
    torch.cuda.set_device(0)
    model=build_detector(spec,train_cfg=cfg.get('train_cfg'),test_cfg=cfg.get('test_cfg')).cuda()
    model.load_state_dict(saved['model'],strict=True)
    opt=build_optimizer(model,cfg.optimizer);opt.load_state_dict(saved['optimizer'])
    def device(v):
        if isinstance(v,torch.Tensor):return v.cuda()
        if isinstance(v,dict):return {k:device(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return type(v)(device(x) for x in v)
        return v
    batch={k:(cpu_tree(v) if k=='img_metas' else device(v)) for k,v in saved['batch'].items()}
    captured=[];forward=model.forward_train
    def capture(*a,**k):
        terms=forward(*a,**k);captured.append(terms);return terms
    model.forward_train=capture
    restore_runtime(model,saved['runtime']);restore_rng(saved['rng'])
    params=[p for p in model.parameters() if p.requires_grad]
    guard=CacheAwareMemoryGuard()
    logs=[];updates=0;failure=None
    for i in range(1 if mode=='zero' else 64):
        captured.clear()
        opt.zero_grad(set_to_none=True)
        try:
            value=model.train_step(batch,opt)
            loss=captured[0]
            terms=dict(main_cls=sum_loss(loss['loss_cls']),main_reg=sum_loss(loss['loss_bbox']))
            terms['auxiliary']=sum(sum_loss(v) for k,v in loss.items() if 'loss' in k and k not in ('loss_cls','loss_bbox'))
            values={k:float(v.detach()) for k,v in terms.items()}
            values['total']=float(value['loss'].detach())
            if i==0 and any(abs(values[k]-expected['losses'][k])>1e-5 for k in expected['losses']):
                raise ValueError('Initial forward must exactly reproduce reviewed stop')
            if not torch.allclose(sum(terms.values()).detach(),value['loss'].detach(),rtol=1e-6,atol=1e-6):
                raise ValueError('Three tasks differ from original total')
            stats=(task_budget_backward(terms,params) if arm=='task_budget' else
                   measured_budget_backward(value['loss'],terms['main_reg'],params))
            if arm=='task_budget':
                holder=SimpleNamespace(losses=loss,_regression_loss=terms['main_reg'])
                release_loss_graphs(SimpleNamespace(outputs=value),holder)
                captured.clear()
                del loss,terms,holder
            alarms=reasons(values,stats['gradient_before_clip'])
            before=float(torch.nn.utils.clip_grad_norm_(params,10.))
            after=float(torch.stack([p.grad.detach().norm() for p in params if p.grad is not None]).norm())
            torch.cuda.synchronize()
            memory=guard.observe(torch.cuda.memory_allocated(),torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
            # Finite counterfactual only: raw production alarms remain visible.
            # Nonfinite/resource violations always stop before any update.
            if 'nonfinite' in alarms:raise FloatingPointError('Nonfinite diagnostic update')
            if mode=='finite':opt.step();updates+=1
            logs.append(dict(index=i+1,losses=values,**stats,gradient_after_clip=after,
                original_guard_alarms=alarms,optimizer_updated=mode=='finite',memory=memory))
            print(json.dumps(dict(arm=arm,index=i+1,losses=values,mixed=stats['gradient_before_clip'],
                                  original_guard_alarms=alarms,reserved_MiB=memory['reserved_bytes']/2**20)),flush=True)
            del value
            if arm!='task_budget':del loss,terms
        except Exception as exc:
            failure=repr(exc);break
    unchanged=all(torch.equal(p.detach().cpu(),saved['model'][n]) for n,p in model.named_parameters())
    if mode=='zero' and not unchanged:raise ValueError('Zero-update verification changed parameters')
    if sha(path)!=pin:raise ValueError('Original snapshot changed')
    report=dict(task_budget_revision=VERSION,task_memory_revision=MEMORY_REVISION,
        arm=arm,mode=mode,records=logs,failure=failure,
        optimizer_updates=updates,fixed_update_budget=0 if mode=='zero' else 64,
        initialization_snapshot_sha256=pin,initial_forward_identical=bool(logs),parameters_unchanged=unchanged,
        original_snapshot_unchanged=True,original_source_run_resumed=False,weights_saved=False,
        raw_loss_guard_passed=bool(logs) and not any(r['original_guard_alarms'] for r in logs),
        batch_order='same saved augmented anomaly batch repeated; not original future training data',
        diagnostic_counterfactual_only=True,full_training_stability_proven=False,
        diagnostic_weights_usable_as_OOF=False,VAL_scored=False,TEST_read=False,
        memory_summary=guard.summary())
    (out/'report.json').write_text(json.dumps(json_safe(report),indent=2,allow_nan=False)+'\n')
    if failure:raise RuntimeError(failure)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--arm',choices=('regression_only','task_budget'),required=True)
    p.add_argument('--mode',choices=('zero','finite'),required=True)
    a=p.parse_args();source,out=a.source.resolve(),a.out.resolve()
    if any(ROOT/'work_dirs' not in x.parents for x in (source,out)):raise ValueError('Project work_dirs only')
    out.mkdir(parents=True,exist_ok=False)
    run(source,out,a.arm,a.mode)


if __name__=='__main__':main()
