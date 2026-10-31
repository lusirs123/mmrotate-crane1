"""Bounded E/E+A/E+B TRAIN diagnosis; one model, no network graph or optimizer."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ARMS = (
    ('E', 'work_dirs/crane_eood_k1_port_day2night_aug_b_v1/crane_eood_k1_port_day2night_aug_b_v1.py',
     'work_dirs/crane_eood_k1_port_day2night_aug_b_v1/epoch_24.pth',
     '95da2d648cfef95e1fcd00f2cdfc258d9cf75ed038135bbfc00059f8f6f2a4db'),
    ('EA', 'work_dirs/eood_A_geometry_20261030_v1/candidate_config.py',
     'work_dirs/eood_A_geometry_20261030_v1/train/epoch_22.pth',
     'fcffdd5853ea2b771195251056fe2692f1f8f6884e329292ec05e60c005cead8'),
    ('EB', 'work_dirs/eood_B_regression_20261031_v3/candidate_config.py',
     'work_dirs/eood_B_regression_20261031_v3/train/epoch_24.pth',
     'f3c25c3e15742497fb692ec3edfb8a5e5c53af68bf0e1b8ee9934b5dcecf9d21'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(4*1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def state_sha(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def parse_memory(text, gpu):
    rows = [tuple(int(x.strip()) for x in line.split(',')) for line in text.splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 5 or rows[0][0] != gpu:
        raise ValueError('Invalid selected GPU memory/temperature response')
    index, free, used, temperature, util = rows[0]
    if min(free, used, temperature, util) < 0:
        raise ValueError('Invalid selected GPU telemetry')
    return dict(index=index, free_mib=free, used_mib=used, temperature_C=temperature, utilization_pct=util)


def telemetry(gpu):
    text = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=index,memory.free,memory.used,temperature.gpu,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    return parse_memory(text, gpu)


def enforce_resources(row, own_mib, before=False):
    # Conservative task limits, not assertions about a hardware damage threshold.
    if row['free_mib'] < (4096 if before else 2048):
        raise RuntimeError('Insufficient free GPU memory')
    if row['temperature_C'] >= 75 or own_mib > 2560:
        raise RuntimeError('Diagnostic temperature/own-memory guard')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(args.gpu)
    assert os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID'
    out = args.out.resolve()
    assert out.is_relative_to(ROOT) if hasattr(out, 'is_relative_to') else str(out).startswith(str(ROOT)+'/')
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    report = dict(protocol='eood_AB_frozen_TRAIN_diagnosis_v1', status='STARTING',
        time_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        physical_gpu=args.gpu, training_updates=0, optimizer_constructed=False,
        TRAIN_only=True, VAL=False, TEST=False, batch=1, scales=[1., .5],
        allocator_cap_mib=2048, own_driver_stop_mib=2560, minimum_free_mib=2048,
        stop_temperature_C=75, source_sha256=sha(__file__), arms={},
        scope='Selected late weights only; deterministic 64 TRAIN identities x two scales. Not a causal account of training history or full VAL generalization.',
        gradient_scope='Gradients of the two original losses with respect to detached decoded boxes only. No backbone/FPN graph, parameter gradients or optimizer.',
        oracle_scope='All-candidate and GT-cost selections are diagnostic oracles, never deployable output policies.')
    save(out/'scope.json', report)
    try:
        import numpy as np
        import torch
        from mmcv import Config
        from mmcv.parallel import collate, scatter
        from mmcv.runner import load_checkpoint
        from mmcv.utils import import_modules_from_strings
        from mmrotate.models import build_detector
        from mmrotate.datasets import build_dataset
        from mmrotate.core import rbbox_overlaps, obb2xyxy, rotated_anchor_inside_flags
        from mmrotate.core.bbox.assigners.pola import PolaAssigner
        from mmrotate.core.bbox.assigners.pola_kld_geometry_v1 import PolaKLDGeometryV1
        from mmrotate.models.losses.sym_kld_calculator import sym_kld
        from mmrotate.models.losses.rotated_iou_loss import RotatedIoULoss
        from mmrotate.models.losses.eood_sym_kld_positive_loss_v1 import EOODSymKLDPositiveLossV1
        from crane_project.tools.preflight_port_center_size_v1 import fixed_train_specs
        torch.set_num_threads(1)
        enforce_resources(telemetry(args.gpu), 0, before=True)
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(2048*2**20/torch.cuda.get_device_properties(0).total_memory, 0)
        # Never construct an optimizer, even accidentally through a later edit.
        def forbidden_optimizer(*a, **k):
            raise AssertionError('Optimizer construction forbidden in frozen diagnosis')
        torch.optim.Optimizer.__init__ = forbidden_optimizer
        indices = np.linspace(0, 1809, 32).round().astype(int).tolist()
        indices += (1810+np.linspace(0, 747, 32).round().astype(int)).tolist()
        assert len(set(indices)) == 64
        original = PolaAssigner()
        kld_assigner = PolaKLDGeometryV1()
        losses = dict(DIoU=RotatedIoULoss(loss_weight=2., mode='linear', diou=True),
                      SymKLD=EOODSymKLDPositiveLossV1(loss_weight=2., strict_finite=True))
        input_hashes = {}
        inventory = json.loads((ROOT/'work_dirs/symeood_110_retrain_seed0_20261026_v1/protocol.json').read_text())['data_inventory']
        inventory = {str((ROOT/r['image']).resolve()):r for r in inventory if r['split']=='train'}
        assert len(inventory) == 2558

        def guard():
            row = telemetry(args.gpu)
            apps = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
                '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader,nounits'], text=True)
            own = [r.split(',') for r in apps.splitlines() if r.split(',')[0].strip() == str(os.getpid())]
            assert len(own) <= 1
            used = int(own[0][2]) if own else 0
            if own:
                expected = ('GPU-a0bb1b92-b540-8a22-8aba-e808f235ae2e' if args.gpu == 0 else
                            'GPU-717daf35-132d-1763-3a16-40a98dee4457')
                assert own[0][1].strip() == expected
            enforce_resources(row, used)
            row.update(own_mib=used, time=time.time())
            with (out/'resources.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')

        def stats(value):
            value = value.detach().float()
            assert torch.isfinite(value).all()
            return dict(minimum=float(value.min()), median=float(value.median()),
                        p90=float(torch.quantile(value, .9)), maximum=float(value.max()))

        def view_diagnosis(model, batch, arm, index, scale):
            head = model.bbox_head.predictors[0]
            meta, gt, labels = batch['img_metas'][0], batch['gt_bboxes'][0], batch['gt_labels'][0]
            assert len(gt) == 1 and not meta.get('flip', False)
            with torch.no_grad():
                features = model.extract_feat(batch['img'])
                cls, reg = model.bbox_head(features)
                assert not any(v.requires_grad for v in features+tuple(cls)+tuple(reg))
                shapes = [v.shape[-2:] for v in cls]
                anchors = head.anchor_generator.grid_priors(shapes, device='cuda:0')
                valid = head.anchor_generator.valid_flags(shapes, meta['pad_shape'], device='cuda:0')
                logits, train_boxes, deploy_boxes, prefilter = [], [], [], []
                for cs, rg, an in zip(cls, reg, anchors):
                    lo = cs[0].permute(1, 2, 0).reshape(-1, 1)
                    delta = rg[0].permute(1, 2, 0).reshape(-1, 5)
                    train_boxes.append(head.bbox_coder.decode(an, delta))
                    deploy_boxes.append(head.bbox_coder.decode(an, delta, max_shape=meta['img_shape']))
                    keep = torch.zeros(len(lo), dtype=torch.bool, device=lo.device)
                    npre = head.test_cfg.nms_pre
                    if npre > 0:
                        keep[lo[:, 0].topk(min(npre, len(lo))).indices] = True
                    else:
                        keep[:] = True
                    prefilter.append(keep & (lo[:, 0].sigmoid() > head.test_cfg.score_thr))
                    logits.append(lo)
                logits, boxes, dep = torch.cat(logits), torch.cat(train_boxes), torch.cat(deploy_boxes)
                inside = rotated_anchor_inside_flags(torch.cat(anchors), torch.cat(valid),
                    meta['img_shape'][:2], head.train_cfg.allowed_border)
                mask = torch.cat(prefilter)
                assert torch.isfinite(boxes).all() and torch.isfinite(logits).all() and (boxes[:, 2:4] > 0).all()
                assert inside.any()
                # Training assignment receives un-clipped decoded boxes and its actual valid mask.
                tb, tl = boxes[inside], logits[inside]
                actual0 = original.assign(tl, tb, labels, gt, meta)
                actual1 = kld_assigner.assign(tl, tb, labels, gt, meta)
                ious = rbbox_overlaps(tb, gt).reshape(-1)
                distance = sym_kld(tb, gt.expand_as(tb))
                probability = tl[:, 0].sigmoid()
                cc = .25*(1-probability)**2 * -(probability+1e-8).log()
                penalty = original.diou(obb2xyxy(tb, 'le90'), obb2xyxy(gt, 'le90')).reshape(-1)
                geo0 = 1-ious+penalty
                ix0 = int((cc+geo0).argmin())
                ix1 = int((cc+distance).argmin())
                assert actual0.gt_inds.nonzero().reshape(-1).tolist() == [ix0]
                assert actual1.gt_inds.nonzero().reshape(-1).tolist() == [ix1]
                candidate_ids = dict(POLA=ix0, KLD=ix1, best_IoU=int(ious.argmax()), top_score=int(probability.argmax()))
                selected = {}
                for key, ix in candidate_ids.items():
                    selected[key] = dict(index=ix, box=tb[ix].cpu().tolist(), score=float(probability[ix]),
                        riou=float(ious[ix]), raw_kld=float(distance[ix]), class_cost=float(cc[ix]),
                        original_geometry=float(geo0[ix]), original_total=float((cc+geo0)[ix]),
                        kld_total=float((cc+distance)[ix]), B_saturated=bool(distance[ix] > 120))
                pool = torch.unique(torch.cat([probability.topk(min(256,len(tb))).indices,
                    ious.topk(min(256,len(tb))).indices]))
                native = model.bbox_head.get_bboxes(cls, reg, batch['img_metas'], rescale=False)[0][0]
                native_top = native[native[:, 5].argmax()] if len(native) else None
                bag_top = dep[mask][logits[mask, 0].argmax()] if mask.any() else None
                assert ((native_top is None and bag_top is None) or (native_top is not None and bag_top is not None
                    and torch.allclose(native_top[:5], bag_top, atol=1e-4, rtol=1e-6)))
                dious = rbbox_overlaps(dep, gt).reshape(-1)
                native_iou = float(rbbox_overlaps(native_top[None,:5], gt)[0,0]) if native_top is not None else 0.
                ranking = dict(native_iou=native_iou, native_output=native_top is not None,
                    best_all_iou=float(dious.max()), best_prefilter_iou=float(dious[mask].max()) if mask.any() else 0.,
                    all_candidates=len(dep), prefilter_candidates=int(mask.sum()),
                    all_good75=int((dious>=.75).sum()), prefilter_good75=int(((dious>=.75)&mask).sum()),
                    native_top1_replayed=True)
                row = dict(arm=arm, index=index, image=Path(meta['filename']).stem,
                    domain='real' if index<1810 else 'sim', scale=scale, gt=gt[0].cpu().tolist(),
                    valid_training_candidates=len(tb), training_vs_deploy_maxdiff=float((boxes-dep).abs().max()),
                    ranking=ranking, selected=selected,
                    all_raw_kld=stats(distance), pool_class_cost=stats(cc[pool]),
                    pool_original_geometry=stats(geo0[pool]), pool_raw_kld=stats(distance[pool]),
                    pool_size=len(pool), assignment_agree=ix0==ix1)
                # Retain only a few detached boxes; release the network activations before autograd.
                unique_ids = sorted(set(candidate_ids.values()))
                probes = tb[unique_ids].clone()
                targets = gt.expand_as(probes).clone()
            del features, cls, reg, anchors, logits, boxes, dep, mask, tb, tl, native, ious, distance
            guard()
            gradients = {}
            for kind, loss in losses.items():
                p = probes.detach().clone().requires_grad_(True)
                value = loss(p, targets, weight=torch.ones(len(p),device=p.device),
                             reduction_override='none')
                grad = torch.autograd.grad(value.sum(), p)[0]
                assert torch.isfinite(value).all() and torch.isfinite(grad).all()
                # Dimensionless local coordinates: GT-axis displacement/GT side,
                # log(predicted sides), and angle in radians. This is a stated
                # diagnostic normalization, not a learned gradient reweighting.
                a = targets[:,4];c,s = a.cos(),a.sin()
                normalized = torch.stack([(grad[:,0]*c+grad[:,1]*s)*targets[:,2],
                    (-grad[:,0]*s+grad[:,1]*c)*targets[:,3],
                    grad[:,2]*p[:,2],grad[:,3]*p[:,3],grad[:,4]],dim=1)
                gradients[kind] = dict(values=value.detach().cpu().tolist(),
                    physical=grad.detach().cpu().tolist(), normalized=normalized.detach().cpu().tolist())
                del p, value, grad, normalized
            g0 = torch.tensor(gradients['DIoU']['normalized'],device=probes.device)
            g1 = torch.tensor(gradients['SymKLD']['normalized'],device=probes.device)
            den = g0.norm(dim=1)*g1.norm(dim=1)
            cos = (g0*g1).sum(1)/den.clamp_min(1e-12)
            row['gradient_probe'] = dict(unique_indices=unique_ids, role_indices=candidate_ids,
                losses=gradients, normalized_cosine=[float(x) if float(d)>1e-12 else None for x,d in zip(cos,den)])
            return row

        for arm, cfgpath, cppath, expected in ARMS:
            random.seed(1703);np.random.seed(1703);torch.manual_seed(1703)
            cfg = Config.fromfile(str(ROOT/cfgpath))
            import_modules_from_strings(**cfg.custom_imports)
            assert sha(ROOT/cppath) == expected
            mc = deepcopy(cfg.model);mc.backbone.init_cfg = None
            model = build_detector(mc)
            load_checkpoint(model, str(ROOT/cppath), map_location='cpu', strict=True)
            model.requires_grad_(False)
            model.cuda().eval()
            assert not any(p.requires_grad for p in model.parameters())
            torch.cuda.reset_peak_memory_stats()
            before = state_sha(model)
            identity = dict(config=cfgpath, config_sha256=sha(ROOT/cfgpath), checkpoint=cppath,
                checkpoint_sha256=expected, before_state=before)
            count = 0
            for scale in (1., .5):
                dataset = build_dataset(fixed_train_specs(cfg,scale))
                assert [len(d) for d in dataset.datasets] == [1810,748]
                for index in indices:
                    guard()
                    batch = scatter(collate([dataset[index]],samples_per_gpu=1),[0])[0]
                    source = inventory[str(Path(batch['img_metas'][0]['filename']).resolve())]
                    for kind in ('image','annotation'):
                        assert sha(ROOT/source[kind]) == source[kind+'_sha256']
                    ih = hashlib.sha256(batch['img'].detach().cpu().numpy().tobytes()+
                        batch['gt_bboxes'][0].cpu().numpy().tobytes()).hexdigest()
                    key = '%s:%s' % (index,scale)
                    if arm == 'E':input_hashes[key] = ih
                    else:assert input_hashes[key] == ih
                    row = view_diagnosis(model,batch,arm,index,scale)
                    row['input_sha256'] = ih
                    with (out/'rows.jsonl').open('a') as f:
                        f.write(json.dumps(row,allow_nan=False)+'\n')
                    del batch,row
                    assert all(p.grad is None for p in model.parameters())
                    count += 1
                    if count%16 == 0:print(arm,count,'/128 views',flush=True)
            after = state_sha(model)
            assert before == after and count == 128
            identity.update(after_state=after,state_unchanged=True,views=count,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
            assert identity['peak_allocated_mib'] <= 2048
            report['arms'][arm] = identity
            save(out/(arm+'_receipt.json'),identity)
            del model
            torch.cuda.empty_cache()
        report.update(status='COMPLETE',elapsed_seconds=time.time()-started,views=384,
            shared_input_checks=256,parameter_gradients_created=False)
        save(out/'completion.json',report)
        print('FROZEN_AB_DIAGNOSIS_COMPLETE',flush=True)
    except BaseException as e:
        save(out/'failure.json',dict(error=repr(e),training_updates=0,TEST=False))
        raise


if __name__ == '__main__':
    main()
