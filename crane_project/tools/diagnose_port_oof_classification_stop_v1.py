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
    signature = inspect.signature(type(nfl).forward)
    def classification(_nfl, *args, **kwargs):
        bound = signature.bind(nfl, *args, **kwargs)
        bound.apply_defaults()
        v = bound.arguments
        value = original(*args, **kwargs)
        records.append(dict(value=value, **{k:v[k] for k in (
            'pred_logits', 'labels', 'pred_bboxes', 'weight', 'avg_factor', 'pred_image_ids')}))
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
    summaries, shared = {}, {}
    for label, term in terms.items():
        gs = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
        summaries[label] = gradient_summary(gs, named)
        summaries[label]['largest_parameters'] = sorted(
            [dict(name=n, norm=float(g.detach().norm())) for (n,p),g in zip(named,gs) if g is not None],
            key=lambda x:x['norm'], reverse=True)[:10]
        shared[label] = torch.cat([
            g.detach().float().flatten().cpu() if g is not None else torch.zeros(p.numel())
            for (n,p),g in zip(named,gs) if n.startswith(('backbone.', 'neck.'))])
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
        entries = []
        for image in range(len(batch['img_metas'])):
            mask = ids == image
            neg = mask & ~positive
            top = torch.nonzero(neg, as_tuple=False).flatten()
            top = top[logits[top].topk(min(10,len(top))).indices] if len(top) else top
            entries.append(dict(image=Path(batch['img_metas'][image]['filename']).stem,
                positives=int((mask & positive).sum()), negatives=int(neg.sum()),
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
    report = dict(step=expected['step'], snapshot_sha256=pin, losses=actual,replay_absolute_error=difference,
        gradients=summaries,shared_gradient_cosines=cosines,classification_levels=levels,
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
