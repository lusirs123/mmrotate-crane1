#!/usr/bin/env python3
"""Pinned TRAIN stop: one forward, component attribution, zero optimizer updates."""
import argparse
from copy import deepcopy
import inspect
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def run(source, out):
    import torch
    import torch.nn.functional as F
    from mmcv import Config
    from mmcv.runner import build_optimizer
    from mmcv.utils import import_modules_from_strings
    from mmrotate.models import build_detector
    from mmrotate.core.hooks.port_oof_trace_hook import (
        PortOOFTraceHook, cpu_tree, restore_rng, restore_runtime, gradient_summary, sum_loss)
    from crane_project.tools.run_port_reliability_oof_v1 import sha
    from crane_project.utils.port_reliability_oof_prevention_v1 import check_config
    from crane_project.utils.port_reliability_oof_followup_v1 import json_safe
    from types import SimpleNamespace

    folder = source/'result/A/prevention'
    completion = json.loads((folder/'completion.json').read_text())
    path = folder/'before_anomaly.pt'
    pin = sha(path)
    if completion['snapshot_sha256'].get(path.name) != pin:
        raise ValueError('Original diagnostic snapshot SHA changed')
    saved = torch.load(str(path), map_location='cpu')
    expected = json.loads((folder/'updates.jsonl').read_text().splitlines()[-1])
    if (completion['status'] != 'ANOMALY_QUARANTINED_BEFORE_OPTIMIZER' or
            not saved['diagnostic_only'] or saved['iteration'] != completion['completed_updates'] or
            saved['iteration']+1 != expected['step'] or expected['optimizer_updated']):
        raise ValueError('Only a pinned before-update TRAIN anomaly is supported')
    cfg = Config.fromfile(str(source/'result/A/detector_config.py'))
    if cfg.optimizer_config.type == 'PortOOFTaskBudgetHook':
        from crane_project.utils.port_oof_task_budget_v1 import check_task_config
        check_task_config(cfg)
    else:
        check_config(cfg)
    import_modules_from_strings(**cfg.custom_imports)
    spec = deepcopy(cfg.model)
    spec.backbone.pop('init_cfg', None)
    spec.backbone.pretrained = None
    torch.cuda.set_device(0)
    model = build_detector(spec, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg')).cuda()
    model.load_state_dict(saved['model'], strict=True)
    optimizer = build_optimizer(model, cfg.optimizer)
    optimizer.load_state_dict(saved['optimizer'])
    def device(v):
        if isinstance(v, torch.Tensor): return v.cuda()
        if isinstance(v, dict): return {k:device(x) for k,x in v.items()}
        if isinstance(v, (tuple, list)): return type(v)(device(x) for x in v)
        return v
    batch = {k:(cpu_tree(v) if k == 'img_metas' else device(v)) for k,v in saved['batch'].items()}
    trace = PortOOFTraceHook(out/'unused')
    trace.model = model
    trace.install(model, capture_failure=False)
    records, features = [], []
    nfl = model.bbox_head.loss_cls
    original = nfl.forward
    original_spatial = nfl._image_spatial_weights
    captured_spatial = []
    def spatial(_nfl, *args, **kwargs):
        result = original_spatial(*args, **kwargs)
        captured_spatial.append(result)
        return result
    nfl._image_spatial_weights = types.MethodType(spatial, nfl)
    signature = inspect.signature(type(nfl).forward)
    def classification(_nfl, *args, **kwargs):
        bound = signature.bind(nfl, *args, **kwargs)
        bound.apply_defaults()
        v = bound.arguments
        value = original(*args, **kwargs)
        if len(captured_spatial) != len(records)+1:
            raise ValueError('Exactly one per-image spatial call per level required')
        records.append(dict(value=value, **{k:v[k] for k in (
            'pred_logits', 'labels', 'pred_bboxes', 'weight', 'avg_factor', 'pred_image_ids')},
            spatial=captured_spatial[-1]))
        return value
    nfl.forward = types.MethodType(classification, nfl)
    handle = model.neck.register_forward_hook(lambda m,a,o:features.extend(o))
    restore_runtime(model, saved['runtime'])
    restore_rng(saved['rng'])
    value = model.train_step(batch, optimizer)
    handle.remove()
    terms = dict(main_cls=sum_loss(trace.losses['loss_cls']), main_reg=sum_loss(trace.losses['loss_bbox']))
    auxiliary = [sum_loss(v) for k,v in trace.losses.items() if 'loss' in k and k not in ('loss_cls','loss_bbox')]
    terms['auxiliary'] = sum(auxiliary) if auxiliary else terms['main_cls']*0.
    actual = {k:float(v.detach()) for k,v in terms.items()}
    actual['total'] = float(value['loss'].detach())
    difference = {k:abs(actual[k]-expected['losses'][k]) for k in expected['losses']}
    if any(v > 1e-5 for v in difference.values()):
        raise ValueError('Forward does not reproduce saved losses: '+str(difference))
    named = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    params = [p for n,p in named]
    summaries, shared, task_cpu = {}, {}, {}
    for label, term in terms.items():
        gs = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
        summaries[label] = gradient_summary(gs, named)
        summaries[label]['largest_parameters'] = sorted(
            [dict(name=n, norm=float(g.detach().norm())) for (n,p),g in zip(named,gs) if g is not None],
            key=lambda x:x['norm'], reverse=True)[:10]
        shared[label] = torch.cat([
            g.detach().float().flatten().cpu() if g is not None else torch.zeros(p.numel())
            for (n,p),g in zip(named,gs) if n.startswith(('backbone.', 'neck.'))])
        task_cpu[label] = [None if g is None else g.detach().float().cpu() for g in gs]
        del gs
        torch.cuda.synchronize()
        if torch.cuda.max_memory_reserved() > 4*2**30:
            raise RuntimeError('Diagnostic exceeds fixed4GiB resource ceiling')
    cosines = {}
    for a,b in [('main_cls','main_reg'), ('main_cls','auxiliary'), ('main_reg','auxiliary')]:
        denominator = float(shared[a].norm()*shared[b].norm())
        cosines[a+'__'+b] = float(torch.dot(shared[a], shared[b]))/denominator if denominator else None
    levels = []
    for i,r in enumerate(records):
        g = torch.autograd.grad(r['value'],r['pred_logits'], retain_graph=True)[0].detach()
        logits = r['pred_logits'].detach().flatten()
        labels = r['labels'].detach()
        ids = r['pred_image_ids']
        positive = labels < model.bbox_head.num_classes
        mu, quality = r['spatial']
        target = F.one_hot(labels, num_classes=model.bbox_head.num_classes+1)[:, :-1].float()
        probability = r['pred_logits'].detach().sigmoid()
        if nfl.use_quality_target:
            target = target*quality.unsqueeze(-1)
            modulation = (target-probability).abs().pow(nfl.gamma)
        else:
            modulation = ((1-probability)*target+probability*(1-target)).pow(nfl.gamma)
        per = F.binary_cross_entropy_with_logits(r['pred_logits'].detach(), target, reduction='none')*\
            modulation*(nfl.alpha*target+(1-nfl.alpha)*(1-target))*mu.unsqueeze(-1)
        per = (per*r['weight'].reshape_as(per)*nfl.loss_weight/r['avg_factor']).sum(-1)
        if not torch.allclose(per.sum(), r['value'].detach(), atol=1e-5, rtol=1e-4):
            raise ValueError('Per-anchor losses do not reconstruct actual classification')
        entries = []
        for image in range(len(batch['img_metas'])):
            mask = ids == image
            neg = mask & ~positive
            top = torch.nonzero(neg, as_tuple=False).flatten()
            top = top[logits[top].topk(min(10,len(top))).indices] if len(top) else top
            ranked = torch.nonzero(neg, as_tuple=False).flatten()
            ranked = ranked[per[ranked].argsort(descending=True)]
            pos = mask & positive
            entries.append(dict(image=Path(batch['img_metas'][image]['filename']).stem,
                positives=int((mask & positive).sum()), negatives=int(neg.sum()),
                weighted_negative_count=int((neg & (r['weight']>0)).sum()),
                positive_loss=float(per[pos].sum()), negative_loss=float(per[neg].sum()),
                positive_logits=logits[pos].cpu().tolist(),
                positive_boxes=r['pred_bboxes'][pos].detach().cpu().tolist(),
                positive_quality=quality[pos].cpu().tolist(),
                positive_spatial_multiplier=mu[pos].cpu().tolist(),
                negative_loss_top1=float(per[ranked[:1]].sum()),
                negative_loss_top10=float(per[ranked[:10]].sum()),
                negative_loss_top100=float(per[ranked[:100]].sum()),
                spatial_negative_loss=float(per[neg & (mu>1)].sum()),
                negative_probability_ge_half=int((logits[neg].sigmoid()>=.5).sum()),
                logit_min=float(logits[mask].min()), logit_max=float(logits[mask].max()),
                logit_gradient_norm=float(g[mask].norm()),
                top_negative_logits=logits[top].cpu().tolist(),
                top_negative_boxes=r['pred_bboxes'][top].detach().cpu().tolist()))
        levels.append(dict(level=i,avg_factor=float(r['avg_factor']),loss=float(r['value']),
                           decomposition=trace.classification[i],images=entries))
    feature_stats = []
    for name,term in terms.items():
        gs = torch.autograd.grad(term,features,retain_graph=True,allow_unused=True)
        feature_stats.append(dict(term=name,levels=[dict(shape=list(f.shape),value_norm=float(f.detach().norm()),
            gradient_norm=0. if g is None else float(g.detach().norm())) for f,g in zip(features,gs)]))
        del gs
    unchanged = all(torch.equal(p.detach().cpu(), saved['model'][n]) for n,p in model.named_parameters())
    if not unchanged or sha(path) != pin: raise ValueError('Diagnostic changed parameters or snapshot')
    momentum = [optimizer.state[p].get('momentum_buffer') for _,p in named]
    momentum_summary = gradient_summary(momentum, named)
    # Read-only first-order attribution of the *actual* budget/clip/SGD
    # direction, including stored momentum and weight decay. No step(), no
    # speculative writes to model parameters or momentum buffers.
    mixed_cpu = []
    scales = {k:min(1.,10./(summaries[k]['all']+1e-12)) for k in terms}
    for index in range(len(params)):
        pieces = [task_cpu[k][index].mul(scales[k]) for k in terms if task_cpu[k][index] is not None]
        mixed_cpu.append(sum(pieces) if pieces else None)
    mixed_norm = float(torch.stack([g.norm() for g in mixed_cpu if g is not None]).norm())
    clip_scale = min(1.,10./(mixed_norm+1e-6))
    contributions = {k:0. for k in ('classification','regression','auxiliary','momentum','weight_decay')}
    predicted_delta_norm_square = 0.
    fresh_momentum_dot = 0.
    group_for = {id(p):group for group in optimizer.param_groups for p in group['params']}
    for index,(_,p) in enumerate(named):
        if mixed_cpu[index] is None: continue
        group = group_for[id(p)]
        if group.get('nesterov',False) or group.get('dampening',0.) != 0.:
            raise ValueError('Only reviewed ordinary SGD momentum supported')
        velocity = mixed_cpu[index].mul(clip_scale)
        decay = p.detach().float().cpu().mul(group['weight_decay'])
        past = optimizer.state[p].get('momentum_buffer')
        past = torch.zeros_like(velocity) if past is None else past.detach().float().cpu().mul(group['momentum'])
        velocity = velocity+decay+past
        fresh_momentum_dot += float(((mixed_cpu[index].mul(clip_scale)+decay)*past).sum())
        predicted_delta_norm_square += float(velocity.square().sum())*group['lr']**2
        cls = task_cpu['main_cls'][index]
        if cls is None: continue
        for task,label in (('main_cls','classification'),('main_reg','regression'),('auxiliary','auxiliary')):
            if task_cpu[task][index] is not None:
                contributions[label] += -group['lr']*float((cls*task_cpu[task][index]).sum())*scales[task]*clip_scale
        contributions['momentum'] += -group['lr']*float((cls*past).sum())
        contributions['weight_decay'] += -group['lr']*float((cls*decay).sum())
    update_direction = dict(mixed_norm=mixed_norm,global_clip_scale=clip_scale,
        fresh_gradient_momentum_dot=fresh_momentum_dot,
        predicted_parameter_delta_norm=predicted_delta_norm_square**.5,
        first_order_classification_loss_change=contributions,
        first_order_classification_loss_change_total=sum(contributions.values()),
        interpretation='local derivative only; not an observed next-forward loss or optimizer update')
    # autograd.grad does not populate .grad or touch SGD state. Compare every
    # optimizer tensor/scalar, not just the model weights.
    def identical(a,b):
        if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
        if isinstance(a,dict): return isinstance(b,dict) and a.keys()==b.keys() and all(identical(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)): return type(a)==type(b) and len(a)==len(b) and all(identical(x,y) for x,y in zip(a,b))
        return a==b
    if not identical(cpu_tree(optimizer.state_dict()),saved['optimizer']):
        raise ValueError('Zero-update attribution changed optimizer state')
    report = dict(step=expected['step'], snapshot_sha256=pin, losses=actual,replay_absolute_error=difference,
        gradients=summaries,shared_gradient_cosines=cosines,classification_levels=levels,
        optimizer_unchanged=True, optimizer_momentum=momentum_summary,
        read_only_SGD_direction=update_direction,
        targets=trace.targets,training_statistics={k:float(sum_loss(v)) for k,v in trace.losses.items() if 'loss' not in k},
        FPN=feature_stats,parameter_updates=0,parameters_unchanged=True,original_snapshot_unchanged=True,
        VAL_scored=False,TEST_read=False,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20)
    (out/'report.json').write_text(json.dumps(json_safe(report),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(step=report['step'],losses=actual,gradients=summaries,
                         peak_reserved_MiB=report['peak_reserved_MiB'])),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    source,out=args.source.resolve(),args.out.resolve()
    if any(ROOT/'work_dirs' not in p.parents for p in (source,out)):
        raise ValueError('All inputs and outputs must remain in project work_dirs')
    out.mkdir(parents=True,exist_ok=False)
    run(source,out)


if __name__=='__main__':main()
