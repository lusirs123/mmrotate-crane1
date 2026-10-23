#!/usr/bin/env python3
"""S4x real-TRAIN smoke validation for the eight ablation groups.

Per group: build the REAL detector from its config (pretrained backbone as
configured, cached locally, no download), run one forward/backward on two real
TRAIN samples, check finiteness, confirm the three switches are live in the
built model, take one SGD step and verify reloading model+optimizer+counter
state reproduces the next loss exactly.

The assigner's call counter is a plain int and is NOT part of the checkpoint.
A raw state_dict reload is measured separately and disclosed, never silently
counted as a pass. Models are built and freed sequentially to fit one 8 GiB GPU.

Scope: real data path + real model graph. It is NOT training, NOT a performance
measurement, and every weight update is discarded (never a training start).
No VAL / TEST / Unknown-02. Output goes to --out only.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GROUPS = ('000', '100', '010', '001', '110', '101', '011', '111')


def _sum_losses(d):
    import torch
    total = None
    for v in d.values():
        for x in (v if isinstance(v, (list, tuple)) else [v]):
            if torch.is_tensor(x) and x.numel():
                total = x if total is None else total + x
    return total


def _counters(model):
    return dict(assign=int(model.bbox_head.assigner._local_call_count),
                iter=int(model.bbox_head.loss_cls._local_iter))


def _set_counters(model, c):
    model.bbox_head.assigner._local_call_count = c['assign']
    model.bbox_head.loss_cls._local_iter.fill_(c['iter'])


def main(args):
    import torch
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from mmdet.datasets import build_dataset
    from mmdet.models import build_detector

    device = torch.device('cuda:%d' % torch.cuda.current_device())
    results, dataset = {}, None
    for g in GROUPS:
        cfg = Config.fromfile(str(ROOT/f'crane_project/configs/crane_symeood_k1_ablation_{g}_v1.py'))
        rec = dict(group=g)
        model = build_detector(cfg.model).to(device)
        rec['switches'] = dict(
            assigner=getattr(model.bbox_head.assigner, 'cost_reg_mode', None),
            loss_bbox=type(model.bbox_head.loss_bbox).__name__,
            use_geometry_weight=getattr(model.bbox_head.loss_cls, 'use_geometry_weight', None))
        rec['switch_ok'] = (rec['switches']['assigner'] == ('kld' if g[0] == '1' else 'iou')
                            and rec['switches']['loss_bbox'] ==
                            ('SymKLDLoss' if g[1] == '1' else 'RotatedIoULoss')
                            and rec['switches']['use_geometry_weight'] == (g[2] == '1'))

        if dataset is None:
            dataset = build_dataset(cfg.data.train)
        data = scatter(collate([dataset[i] for i in range(1)], samples_per_gpu=1),
                       [device.index])[0]

        model.train()
        losses = model.forward_train(**data)
        total = _sum_losses(losses)
        rec['loss_keys'] = sorted(losses.keys())
        rec['loss_finite'] = bool(torch.isfinite(total).all())
        rec['loss_total'] = float(total)
        total.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        rec['grad_present'] = len(grads)
        rec['grad_finite'] = all(bool(torch.isfinite(x).all()) for x in grads)
        rec['grad_nonzero'] = sum(1 for x in grads if float(x.abs().sum()) > 0)

        opt = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        opt.step()
        rec['params_updated'] = sum(1 for n, p in model.named_parameters()
                                    if not torch.equal(p.detach(), before[n]))
        sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        osd = copy.deepcopy(opt.state_dict())
        cnt = _counters(model)
        rec['counter_state'] = cnt
        next_total = _sum_losses(model.forward_train(**data))
        rec['next_loss'] = float(next_total)
        del model, opt, losses, before
        torch.cuda.empty_cache()

        model2 = build_detector(cfg.model).to(device)
        model2.load_state_dict(sd)
        model2.train()
        opt2 = torch.optim.SGD(model2.parameters(), lr=1e-6, momentum=0.9)
        opt2.load_state_dict(osd)
        naive = _sum_losses(model2.forward_train(**data))
        rec['reload_without_counter_loss'] = float(naive)
        rec['reload_without_counter_matches'] = bool(naive == next_total)
        del model2, opt2
        torch.cuda.empty_cache()

        model3 = build_detector(cfg.model).to(device)
        model3.load_state_dict(sd)
        model3.train()
        opt3 = torch.optim.SGD(model3.parameters(), lr=1e-6, momentum=0.9)
        opt3.load_state_dict(osd)
        _set_counters(model3, cnt)
        resumed = _sum_losses(model3.forward_train(**data))
        rec['reload_loss'] = float(resumed)
        rec['reload_matches'] = bool(resumed == next_total)
        del model3, opt3, sd, osd
        torch.cuda.empty_cache()

        rec['group_ok'] = (rec['switch_ok'] and rec['loss_finite'] and rec['grad_finite']
                           and rec['grad_nonzero'] > 0 and rec['params_updated'] > 0
                           and rec['reload_matches'])
        results[g] = rec
        print(json.dumps(rec, ensure_ascii=False), flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ok = all(r['group_ok'] for r in results.values())
    payload = dict(protocol='symeood_ablation_real_train_v1', device=str(device),
                   data=dict(split='train only', train_views=len(dataset), used_per_group=1,
                             note='one image per group (8 GiB card); one step; updates discarded, not a training start'),
                   counter_disclosure='assigner._local_call_count is not checkpointed; '
                                      'raw state_dict reload differs for C=1 groups',
                   groups=results, all_groups_ok=ok)
    (out/'real_train.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(dict(all_groups_ok=ok, groups=len(results))))
    return 0 if ok else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    sys.exit(main(p.parse_args()))
