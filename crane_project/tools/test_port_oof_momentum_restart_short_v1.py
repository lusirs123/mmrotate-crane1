#!/usr/bin/env python3
"""Two separately launched 32-update TRAIN counterfactuals; no OOF weights.

Repeat the saved transformed batch. Original finite loss limits are recorded
as alarms in this diagnostic only; nonfinite/mixed-gradient/resource failures
still stop. Never resumes the original run or alters production guards.
"""
import argparse
from copy import deepcopy
import inspect
import json
from pathlib import Path
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
UPDATES = 32


def metrics(records):
    # The first row is the identical pre-repair state; row33 is the measured
    # post-update32 forward, with no subsequent update.
    after = [r for r in records if r['index']>1]
    return dict(post_first_cls=after[0]['losses']['main_cls'] if after else None,
        mean_post_cls=sum(r['losses']['main_cls'] for r in after)/len(after) if after else None,
        max_post_cls=max(r['losses']['main_cls'] for r in after) if after else None,
        final_cls=records[-1]['losses']['main_cls'] if records else None,
        final_reg=records[-1]['losses']['main_reg'] if records else None,
        loss_alarm_forwards=sum(bool(r['original_guard_alarms']) for r in records))


def compare(control,candidate):
    if (control['snapshot_sha256']!=candidate['snapshot_sha256'] or
            control['source_sha256']!=candidate['source_sha256'] or
            (control['records'] and candidate['records'] and
             control['records'][0]['losses']!=candidate['records'][0]['losses'])):
        raise ValueError('Both arms must share source, snapshot and first forward')
    eligible = all(r['optimizer_updates']==UPDATES and len(r['records'])==UPDATES+1 and
                   r['failure'] is None and r['original_snapshot_unchanged'] for r in (control,candidate))
    a,b = metrics(control['records']),metrics(candidate['records'])
    criteria = {}
    if eligible:
        criteria = dict(post_first_cls_lower=b['post_first_cls']<a['post_first_cls'],
            mean_post_cls_not_worse=b['mean_post_cls']<=a['mean_post_cls'],
            max_post_cls_not_worse=b['max_post_cls']<=a['max_post_cls'],
            final_cls_not_worse=b['final_cls']<=a['final_cls'],
            final_reg_not_worse=b['final_reg']<=a['final_reg'])
    return dict(control=a,candidate=b,complete=eligible,criteria=criteria,
        local_recovery_increment_supported=eligible and all(criteria.values()),
        full_training_stability_proven=False,prevention_passed=False,adopted=False,
        interpretation='same TRAIN batch repeated; measured local recovery only, no unseen-batch generalization')


def run(source,out,arm):
    import torch
    from mmcv import Config
    from mmcv.runner import build_optimizer
    from mmcv.utils import import_modules_from_strings
    from mmrotate.models import build_detector
    from mmrotate.core.hooks.port_oof_trace_hook import PortOOFTraceHook,cpu_tree,restore_rng,restore_runtime,sum_loss
    from crane_project.tools.run_port_reliability_oof_v1 import sha
    from crane_project.utils.port_oof_task_budget_v1 import task_budget_backward,release_loss_graphs,check_task_config
    from crane_project.utils.port_oof_momentum_restart_v1 import configure,momentum_restart_decision
    from crane_project.utils.port_reliability_oof_prevention_v1 import reasons,CacheAwareMemoryGuard
    from types import SimpleNamespace
    start = time.monotonic()
    folder = source/'result/A/prevention'
    completion = json.loads((folder/'completion.json').read_text())
    path = folder/'before_anomaly.pt';pin = sha(path)
    saved = torch.load(str(path),map_location='cpu')
    expected = json.loads((folder/'updates.jsonl').read_text().splitlines()[-1])
    reviewed = json.loads((source/'momentum_restart_final_check/report.json').read_text())
    if (completion['snapshot_sha256']['before_anomaly.pt']!=pin or not saved['diagnostic_only'] or
            saved['iteration']!=completion['completed_updates'] or expected['step']!=saved['iteration']+1 or
            expected['optimizer_updated'] or reviewed['snapshot_sha256']!=pin or
            reviewed['optimizer_updates']!=0 or not reviewed['parameters_unchanged'] or
            not reviewed['optimizer_unchanged']):
        raise ValueError('Verified pre-update TRAIN boundary required')
    cfg = Config.fromfile(str(source/'result/A/detector_config.py'));check_task_config(cfg)
    source_meta = json.loads((source/'result/prevention.json').read_text())
    if sha(source/'result/A/detector_config.py')!=source_meta['configuration_sha256']:
        raise ValueError('Original configuration SHA changed')
    for p,h in source_meta['sources'].items():
        if sha(ROOT/p)!=h:raise ValueError('Original training source changed: '+p)
    if arm=='restart':configure(cfg)
    import_modules_from_strings(**cfg.custom_imports)
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
    trace = PortOOFTraceHook(out/'unused');trace.model = model;trace.install(model,capture_failure=False)
    nfl = model.bbox_head.loss_cls;original = nfl.forward
    signature = inspect.signature(type(nfl).forward)
    background = []
    def classification(_nfl,*args,**kwargs):
        bound = signature.bind(nfl,*args,**kwargs);bound.apply_defaults();v = bound.arguments
        value = original(*args,**kwargs)
        with torch.no_grad():
            labels = v['labels'];probability = v['pred_logits'].detach().sigmoid().flatten()
            valid = (v['weight']>0);negative = valid & (labels>=model.bbox_head.num_classes)
            p = probability[negative]
            background.append(dict(negative_count=int(negative.sum()),
                negative_probability_mean=float(p.mean()) if len(p) else 0.,
                negative_probability_max=float(p.max()) if len(p) else 0.,
                negative_probability_ge_half=int((p>=.5).sum()),negative_probability_ge_point1=int((p>=.1).sum())))
        return value
    nfl.forward = types.MethodType(classification,nfl)
    restore_runtime(model,saved['runtime']);restore_rng(saved['rng'])
    params = [p for p in model.parameters() if p.requires_grad]
    guard = CacheAwareMemoryGuard();records = [];updates = 0;failure = None
    for index in range(1,UPDATES+2):
        background.clear();opt.zero_grad(set_to_none=True)
        try:
            value = model.train_step(batch,opt);loss = trace.losses
            terms = dict(main_cls=sum_loss(loss['loss_cls']),main_reg=sum_loss(loss['loss_bbox']))
            terms['auxiliary'] = sum(sum_loss(v) for k,v in loss.items() if 'loss' in k and k not in ('loss_cls','loss_bbox'))
            losses = {k:float(v.detach()) for k,v in terms.items()};losses['total'] = float(value['loss'].detach())
            if torch.cuda.max_memory_reserved()>4*2**30 or torch.cuda.max_memory_allocated()>4*2**30:
                raise RuntimeError('Fixed4GiB diagnostic resource limit')
            if index==1 and any(abs(losses[k]-expected['losses'][k])>1e-5 for k in expected['losses']):
                raise ValueError('First forward must reproduce original stop')
            if not torch.allclose(sum(terms.values()).detach(),value['loss'].detach(),atol=1e-6,rtol=1e-6):
                raise ValueError('Task decomposition differs from actual total')
            alarms = reasons(losses,0.)
            if 'nonfinite' in alarms:raise FloatingPointError('Nonfinite diagnostic loss')
            record = dict(index=index,losses=losses,classification=deepcopy(trace.classification),
                          background=deepcopy(background),optimizer_updated=False,original_guard_alarms=alarms)
            if index<=UPDATES:
                stats = task_budget_backward(terms,params)
                holder = SimpleNamespace(losses=loss,_regression_loss=terms['main_reg'])
                release_loss_graphs(SimpleNamespace(outputs=value),holder)
                trace.losses = None;del loss,terms,holder
                alarms = reasons(losses,stats['gradient_before_clip'])
                if 'nonfinite' in alarms or 'gradient_before_clip' in alarms:
                    raise FloatingPointError('Nonfinite or mixed-gradient diagnostic failure')
                torch.nn.utils.clip_grad_norm_(params,10.)
                decision = momentum_restart_decision(opt)
                torch.cuda.synchronize()
                memory = guard.observe(torch.cuda.memory_allocated(),torch.cuda.memory_reserved(),
                    peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
                # Authorized counterfactual: finite loss alarms remain in each
                # record. Production hook still stops on those same alarms.
                opt.step();updates += 1
                record.update(**stats,original_guard_alarms=alarms,memory=memory,
                    momentum_alignment=decision,optimizer_updated=True,
                    momentum_restarted=arm=='restart' and opt.last_restart['momentum_restarted'])
            else:
                trace.losses = None;del loss,terms
            records.append(record)
            print(json.dumps(dict(arm=arm,index=index,losses=losses,alarms=record['original_guard_alarms'],
                optimizer_updated=record['optimizer_updated'],momentum_restarted=record.get('momentum_restarted',False))),flush=True)
            del value
        except Exception as exc:
            failure = repr(exc);break
    if sha(path)!=pin:raise ValueError('Original snapshot changed')
    report = dict(arm=arm,records=records,failure=failure,optimizer_updates=updates,fixed_update_budget=UPDATES,
        snapshot_sha256=pin,original_snapshot_unchanged=True,original_source_run_resumed=False,
        weights_saved=False,diagnostic_weights_usable_as_OOF=False,full_training_stability_proven=False,
        prevention_passed=False,VAL_scored=False,TEST_read=False,raw_loss_guard_mode='record_only_in_counterfactual',
        batch_order='one identical saved transformed TRAIN batch repeated',
        source_sha256=sha(Path(__file__)),elapsed_seconds=time.monotonic()-start,memory_summary=guard.summary())
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if failure:raise RuntimeError(failure)


def main():
    p = argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--arm',choices=('control','restart'),required=True)
    a = p.parse_args();source,out = a.source.resolve(),a.out.resolve()
    if any(ROOT/'work_dirs' not in x.parents for x in (source,out)):raise ValueError('Project work_dirs only')
    out.mkdir(parents=True,exist_ok=False);run(source,out,a.arm)


if __name__=='__main__':main()
