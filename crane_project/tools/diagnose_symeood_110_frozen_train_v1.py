#!/usr/bin/env python3
"""Bounded frozen E/S TRAIN diagnostics. No optimizer, TEST, or model edits."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def state_digest(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from mmcv.runner import load_checkpoint
    from mmcv.utils import import_modules_from_strings
    from mmrotate.datasets import build_dataset
    from mmrotate.models import build_detector
    from mmrotate.core import rbbox_overlaps
    from mmrotate.models.losses.sym_kld_calculator import sym_kld
    from crane_project.tools.preflight_port_center_size_v1 import fixed_train_specs

    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    # Batch size two is part of the frozen assigner's call-count convention.
    pairs = [(0, 1810), (603, 2059), (1206, 2308), (1809, 2557)]
    arms = [
        ('E', 'work_dirs/crane_eood_k1_port_day2night_aug_b_v1/crane_eood_k1_port_day2night_aug_b_v1.py',
         'work_dirs/crane_eood_k1_port_day2night_aug_b_v1/epoch_24.pth',
         '95da2d648cfef95e1fcd00f2cdfc258d9cf75ed038135bbfc00059f8f6f2a4db'),
        ('S', 'work_dirs/symeood_110_retrain_seed0_20261026_v1/configs/Coff_seed0.py',
         'work_dirs/symeood_110_retrain_seed0_20261026_v1/110/epoch_20.pth',
         '7f35dacc308943949a8fd2e6aaad312d14651564903ec87c8c8dcec0552ec054')]
    report = dict(protocol='symeood_110_frozen_train_diagnostic_v1',
                  training_updates=0, TEST=False, VAL=False,
                  pairs=pairs, scales=[1., .5], arms={},
                  inference_note='Diagnostic oracle is offline TRAIN only; never an output policy.',
                  script_sha256=sha(__file__))
    (out / 'scope.json').write_text(json.dumps(report, indent=2))
    start = time.time()
    try:
        for arm, cfgname, checkpoint, expected in arms:
            random.seed(1703)
            np.random.seed(1703)
            torch.manual_seed(1703)
            cfg = Config.fromfile(str(ROOT / cfgname))
            import_modules_from_strings(**cfg.custom_imports)
            cp = ROOT / checkpoint
            assert sha(cp) == expected, 'Frozen checkpoint identity differs'
            # Avoid downloading initialization: strict loading supplies every tensor.
            modelcfg = deepcopy(cfg.model)
            modelcfg.backbone.init_cfg = None
            model = build_detector(modelcfg)
            load_checkpoint(model, str(cp), map_location='cpu', strict=True)
            model.cuda().eval()
            before = state_digest(model)
            head = model.bbox_head if arm == 'S' else model.bbox_head.predictors[0]
            counts = dict(total=sum(p.numel() for p in model.parameters()),
                          head=sum(p.numel() for p in model.bbox_head.parameters()),
                          cls_tower=len(getattr(model.bbox_head, 'cls_convs', [])),
                          reg_tower=len(getattr(model.bbox_head, 'reg_convs', [])))
            rankrows, gradrows, rawrows = [], [], []
            for scale in (1., .5):
                dataset = build_dataset(fixed_train_specs(cfg, scale))
                assert [len(x) for x in dataset.datasets] == [1810, 748]
                for pair in pairs:
                    batch = scatter(collate([dataset[i] for i in pair], samples_per_gpu=2), [0])[0]
                    metas = batch['img_metas']
                    assert not any(m.get('flip', False) for m in metas)
                    model.eval()
                    with torch.no_grad():
                        feats = model.extract_feat(batch['img'])
                        cls, reg = model.bbox_head(feats)
                        native = model.bbox_head.get_bboxes(cls, reg, metas, rescale=False)
                        anchors = head.anchor_generator.grid_priors(
                            [x.shape[-2:] for x in cls], device=cls[0].device)
                        for bi, meta in enumerate(metas):
                            bags, scores_all, prefilter = [], [], []
                            for cs, rg, an in zip(cls, reg, anchors):
                                scores = cs[bi].permute(1, 2, 0).reshape(-1).sigmoid()
                                deltas = rg[bi].permute(1, 2, 0).reshape(-1, 5)
                                boxes = head.bbox_coder.decode(an, deltas, max_shape=meta['img_shape'])
                                finite = torch.isfinite(boxes).all(-1) & torch.isfinite(scores)
                                mask = finite & (scores > head.test_cfg.score_thr)
                                # E uses per-level score top-k before NMS; S uses no nms_pre.
                                if arm == 'E' and head.test_cfg.nms_pre > 0:
                                    keep = scores.topk(min(head.test_cfg.nms_pre, len(scores))).indices
                                    pre = torch.zeros_like(mask)
                                    pre[keep] = True
                                    mask &= pre
                                bags.append(boxes[finite])
                                scores_all.append(scores[finite])
                                prefilter.append(mask[finite])
                            boxes = torch.cat(bags)
                            scores = torch.cat(scores_all)
                            mask = torch.cat(prefilter)
                            gt = batch['gt_bboxes'][bi]
                            assert len(gt) == 1
                            ious = rbbox_overlaps(boxes, gt).reshape(-1)
                            nb = native[bi][0]
                            top = nb[nb[:, 5].argmax()] if len(nb) else None
                            native_iou = float(rbbox_overlaps(top[None, :5], gt)[0, 0]) if top is not None else None
                            ix = scores[mask].argmax() if mask.any() else None
                            bagtop = boxes[mask][ix] if ix is not None else None
                            same = ((top is None and bagtop is None) or
                                    (top is not None and bagtop is not None and
                                     torch.allclose(top[:5], bagtop, atol=1e-4, rtol=1e-6)))
                            assert same, 'Reconstructed native candidate Top1 differs'
                            best = int(ious.argmax())
                            rankrows.append(dict(arm=arm, image=Path(meta['filename']).stem,
                                scale=scale, native_top1_iou=native_iou,
                                oracle_all_iou=float(ious[best]), oracle_all_score=float(scores[best]),
                                oracle_deployed_bag_iou=float(ious[mask].max()) if mask.any() else None,
                                all_candidates=len(boxes), deployed_bag_candidates=int(mask.sum()),
                                bag_top1_replay_match=same))
                    del feats, cls, reg, anchors, native, boxes, scores, mask, ious
                    # Actual training losses on the same deterministic two-image batch.
                    saved_buffers = {k: v.clone() for k, v in model.named_buffers()}
                    counter = getattr(getattr(head, 'assigner', None), '_local_call_count', None)
                    model.train()
                    if arm == 'S':
                        head.assigner._local_call_count = 2 * max(
                            2 * head.assigner.o2m_warmup_iters, head.assigner.warmup_iters)
                    else:
                        head.epoch = 24
                    handles = []
                    if arm == 'S':
                        def observe(module, args, kwargs=None):
                            pred, target = args[:2]
                            with torch.no_grad():
                                d = sym_kld(pred, target).detach().cpu().double().numpy()
                            rawrows.append(dict(scale=scale, pair=pair,
                                positive_count=len(d), raw_kld=d.tolist(), saturated_count=int((d > 120).sum())))
                        handles.append(head.loss_bbox.register_forward_pre_hook(observe))
                    try:
                        losses = model(return_loss=True, **batch)
                        scalar = {k: sum(v) if isinstance(v, (list, tuple)) else v
                                  for k, v in losses.items() if 'loss' in k}
                        assert all(torch.isfinite(v).all() for v in scalar.values())
                        if arm == 'S':
                            groups = dict(main_cls=scalar['loss_cls'], main_reg=scalar['loss_bbox'],
                                          auxiliary=sum(v for k, v in scalar.items() if k.startswith('aux')))
                        else:
                            groups = dict(main_cls=scalar['loss_0_loss_cls'], main_reg=scalar['loss_0_loss_iou'],
                                          auxiliary=sum(v for k, v in scalar.items() if k.startswith('loss_1_')))
                        params = [p for p in model.neck.parameters() if p.requires_grad]
                        grads = {}
                        for name, value in groups.items():
                            gg = torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
                            assert all(g is None or torch.isfinite(g).all() for g in gg)
                            grads[name] = float(sum(g.detach().square().sum() for g in gg if g is not None).sqrt())
                            del gg
                        gradrows.append(dict(scale=scale, pair=pair,
                            loss={k: float(v.detach()) for k, v in scalar.items()}, fpn_gradient_l2=grads))
                    finally:
                        for h in handles:
                            h.remove()
                        with torch.no_grad():
                            for k, v in model.named_buffers():
                                v.copy_(saved_buffers[k])
                        if counter is not None:
                            head.assigner._local_call_count = counter
                    model.zero_grad(set_to_none=True)
                    del batch, losses, scalar, groups, params, saved_buffers
                    print(arm, 'scale', scale, 'pair', pair, 'done', flush=True)
            after = state_digest(model)
            assert before == after, 'Frozen parameters or buffers changed'
            result = dict(config=cfgname, config_sha256=sha(ROOT / cfgname),
                          checkpoint=checkpoint, checkpoint_sha256=expected,
                          state_before=before, state_after=after, state_unchanged=True,
                          counts=counts, ranking=rankrows, gradients=gradrows, assigned_regression=rawrows)
            report['arms'][arm] = result
            (out / (arm + '.json')).write_text(json.dumps(result, indent=2))
            del model
            torch.cuda.empty_cache()
        report.update(status='COMPLETE', elapsed_seconds=time.time() - start,
                      peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
        (out / 'completion.json').write_text(json.dumps(report, indent=2))
    except Exception as e:
        (out / 'failure.json').write_text(json.dumps(dict(error=repr(e), training_updates=0), indent=2))
        raise


if __name__ == '__main__':
    main()
