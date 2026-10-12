#!/usr/bin/env python3
"""Two 64-update TRAIN snapshot continuations; diagnostic weights are discarded.

Same anomalous pre-forward state, momentum/RNG, batches and budget in both arms.
Only main-regression gradient budgeting differs. No dataset/VAL/TEST is loaded.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import types

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.tools import run_port_reliability_oof_followup_v1 as parent
from crane_project.utils import port_reliability_oof_gradient_chain_v1 as contract


def run(source,out,gpu,arm):
    import torch
    from mmcv import Config
    from mmcv.runner import build_optimizer
    from mmcv.utils import import_modules_from_strings
    from mmrotate.models import build_detector
    from mmrotate.core.hooks.port_oof_trace_hook import cpu_tree,restore_rng,restore_runtime,sum_loss
    from mmrotate.core.hooks.port_oof_regression_budget_hook import PortOOFRegressionBudgetHook
    from mmrotate.core.hooks.port_strict_optimizer_hook import PortStrictOptimizerHook
    from types import SimpleNamespace
    parent.checked(source);parent.old.check_fold(source,'A')
    # Require evidence before any candidate update, not a root-cause declaration.
    attribution=out.parent/'result'
    for boundary in ('previous','anomaly'):
        evidence=json.loads((attribution/(boundary+'_report.json')).read_text())
        if not evidence['comparison']['passed'] or evidence['parameter_updates']!=0:
            raise ValueError('Missing verified unchanged-gradient attribution')
        if any(r['physical_gradient_relative_error64']>1e-3 for r in evidence['positive_boxes']):
            raise ValueError('Numerical discrepancy found; do not apply gradient-budget repair')
    completion=json.loads((source/'A/trace/completion.json').read_text())
    paths={k:source/'A/trace'/n for k,n in (('anomaly','before_anomaly.pt'),('previous','before_previous_update.pt'))}
    snapshots={}
    for k,p in paths.items():
        if parent.old.sha(p)!=completion['snapshot_sha256'][p.name]: raise ValueError('Snapshot changed')
        s=torch.load(str(p),map_location='cpu');contract.validate_boundary(s,completion,k);snapshots[k]=s
    saved=snapshots['anomaly']
    cfg=Config.fromfile(str(source/'A/detector_config.py'));parent.old.training_repair.validate_config(cfg)
    import_modules_from_strings(**cfg.custom_imports)
    spec=deepcopy(cfg.model);spec.backbone.pop('init_cfg',None);spec.backbone.pretrained=None
    torch.cuda.set_device(gpu)
    model=build_detector(spec,train_cfg=cfg.get('train_cfg'),test_cfg=cfg.get('test_cfg')).cuda(gpu)
    model.load_state_dict(saved['model'],strict=True)
    optimizer=build_optimizer(model,cfg.optimizer);optimizer.load_state_dict(saved['optimizer'])
    original_step=optimizer.step;actual_updates=[0]
    def counted_step(*args,**kwargs):
        result=original_step(*args,**kwargs);actual_updates[0]+=1;return result
    optimizer.step=counted_step
    def device(v):
        if isinstance(v,torch.Tensor):return v.cuda(gpu)
        if isinstance(v,dict):return {k:device(x) for k,x in v.items()}
        if isinstance(v,list):return [device(x) for x in v]
        if isinstance(v,tuple):return tuple(device(x) for x in v)
        return v
    batches={k:{n:(cpu_tree(v) if n=='img_metas' else device(v)) for n,v in s['batch'].items()} for k,s in snapshots.items()}
    class Log:
        def __init__(self):self.values={}
        def update(self,values,num_samples):self.values.update(values)
    runner=SimpleNamespace(model=model,optimizer=optimizer,iter=saved['iteration'],log_buffer=Log())
    hook=(PortStrictOptimizerHook if arm=='control' else PortOOFRegressionBudgetHook)(
        grad_clip=dict(max_norm=10,norm_type=2),memory_guard=deepcopy(parent.old.training_repair.MEMORY_GUARD))
    if arm!='control':hook.before_run(runner)
    positives=[];orig=model.bbox_head.loss_bbox.forward
    def capture(_loss,p,q,*args,**kwargs):
        positives.append(dict(pred=p.detach().cpu().tolist(),GT=q.detach().cpu().tolist(),**contract.canonical_error(p.detach(),q.detach())))
        return orig(p,q,*args,**kwargs)
    model.bbox_head.loss_bbox.forward=types.MethodType(capture,model.bbox_head.loss_bbox)
    restore_runtime(model,saved['runtime']);restore_rng(saved['rng'])
    logs=[];updates=0;failure=None
    def forward(k):
        positives.clear();runner.log_buffer.values={}
        value=model.train_step(batches[k],optimizer);runner.outputs=value
        if not torch.isfinite(value['loss']):raise FloatingPointError('Nonfinite total loss')
        return value
    # Counterfactual control repeats old finite-only global clipping. It is not
    # allowed to continue the diagnosed production run or become an OOF source.
    for i in range(contract.FINITE_UPDATES):
        k='anomaly' if i%2==0 else 'previous'
        runner.iter=saved['iteration']+i
        if arm!='control':hook.before_train_iter(runner)
        value=forward(k)
        entry=dict(update_index=i+1,batch=k,logs=dict(value['log_vars']),positive_assignment_boxes=deepcopy(positives),optimizer_updated=False)
        try:
            hook.after_train_iter(runner)
            torch.cuda.synchronize(gpu)
            if torch.cuda.max_memory_reserved(gpu)>contract.LIMIT_BYTES:raise RuntimeError('Fixed4GiB resource limit exceeded')
            entry.update(optimizer_updated=True,gradient_stats=deepcopy(runner.log_buffer.values),
                reserved_MiB=torch.cuda.memory_reserved(gpu)/2**20)
        except Exception as exc:
            failure=repr(exc);entry.update(failure=failure,gradient_stats=deepcopy(runner.log_buffer.values))
        # The historical control checks memory after step; do not misreport an
        # already completed update if that post-step check raises.
        entry['optimizer_updated']=actual_updates[0]>updates
        updates=actual_updates[0]
        logs.append(entry)
        if failure:break
    # Fixed final state only; neither best step nor threshold is selected.
    final=[]
    if not failure:
        for k in ('anomaly','previous'):
            with torch.no_grad():value=forward(k)
            final.append(dict(batch=k,logs=dict(value['log_vars']),positive_assignment_boxes=deepcopy(positives)))
    for k,p in paths.items():
        if parent.old.sha(p)!=completion['snapshot_sha256'][p.name]:raise ValueError('Original snapshot changed')
    report=dict(protocol=contract.VERSION,arm=arm,updates=updates,fixed_updates=contract.FINITE_UPDATES,
        regression_budget=contract.REGRESSION_BUDGET if arm!='control' else None,global_clip=10,
        initialization_snapshot_sha256=completion['snapshot_sha256']['before_anomaly.pt'],
        batch_order='anomaly,previous alternating; saved augmentation; not future original training sequence',
        records=logs,final_batches=final,failure=failure,original_snapshots_unchanged=True,
        original_source_run_resumed=False,weights_saved=False,diagnostic_weights_usable_as_OOF=False,
        VAL_read=False,TEST_read=False,peak_reserved_MiB=torch.cuda.max_memory_reserved(gpu)/2**20,
        full_training_stability_proven=False,reliability_gain_proven=False)
    out.mkdir(parents=True,exist_ok=True);p=out/(arm+'_report.json')
    if p.exists():raise ValueError('Refuse overwrite')
    parent.old.write(p,parent.contract.json_safe(report))
    print(json.dumps(dict(arm=arm,updates=updates,failure=failure,peak_reserved_MiB=report['peak_reserved_MiB'],
        final_logs=[x['logs'] for x in final])),flush=True)
    if failure:raise RuntimeError('Finite continuation failed: '+failure)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--arm',choices=('control','regression_budget'),required=True)
    a=parser.parse_args();s=a.source.resolve();o=a.out.resolve()
    if any(ROOT/'work_dirs' not in p.parents for p in (s,o)):raise ValueError('Projectwork_dirs only')
    if (o/(a.arm+'_report.json')).exists():raise ValueError('Refuse overwrite')
    run(s,o,a.gpu,a.arm)
