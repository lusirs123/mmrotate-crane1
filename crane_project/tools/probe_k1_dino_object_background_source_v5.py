"""Short source probe for DINO object/background relation supervision.

Use eight annotation-selected source images with sequence-disjoint fit and
held-out sets. Compare matched short detection-only and relation-assisted
adapter updates. No checkpoint, VAL, or TEST is used or written.
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from crane_project.tools.preflight_k1_dino_fpn_gradient_v3 import (
    _check_cached_feature_transform, _loss_sum)
from crane_project.tools.preflight_k1_dino_warmstart_v2 import (
    EXPECTED_BASELINE_SHA256, sha256)


PROTOCOL = 'k1_dino_object_background_source_probe_v5'
CONFIG = 'crane_project/configs/crane_symeood_k1_dino_fpn_gradient_c_v3.py'


def object_background_masks(boxes, meta, height, width, device):
    """Rasterize OBB interiors and a one-cell-gap nearby background ring."""
    if boxes is None or boxes.numel() == 0:
        raise ValueError('Source sample has no GT object')
    pad_h, pad_w = meta['pad_shape'][:2]
    img_h, img_w = meta['img_shape'][:2]
    yy = ((torch.arange(height, device=device) + 0.5)
          * pad_h / height)[:, None]
    xx = ((torch.arange(width, device=device) + 0.5)
          * pad_w / width)[None, :]
    valid = (xx < img_w) & (yy < img_h)
    object_mask = torch.zeros((height, width), dtype=torch.bool, device=device)
    for box in boxes.detach():
        cx, cy, bw, bh, theta = box[:5]
        if not torch.isfinite(box[:5]).all().item() or bw <= 0 or bh <= 0:
            raise ValueError('Invalid source GT OBB')
        dx, dy = xx - cx, yy - cy
        local_x = dx * torch.cos(theta) + dy * torch.sin(theta)
        local_y = -dx * torch.sin(theta) + dy * torch.cos(theta)
        object_mask |= ((local_x.abs() <= bw / 2)
                        & (local_y.abs() <= bh / 2))
    object_mask &= valid
    if not object_mask.any().item():
        raise ValueError('GT OBB covers no teacher token')
    binary = object_mask.float()[None, None]
    outer = F.max_pool2d(binary, kernel_size=9, stride=1, padding=4)[0, 0] > 0
    inner = F.max_pool2d(binary, kernel_size=3, stride=1, padding=1)[0, 0] > 0
    background_mask = outer & ~inner & valid
    if not background_mask.any().item():
        raise ValueError('No valid nearby background token')
    return object_mask, background_mask


def relation_map(features, object_mask):
    """Cosine similarity to this image's mean object feature."""
    if features.ndim != 3 or features.shape[1:] != object_mask.shape:
        raise ValueError('Feature/mask spatial dimensions differ')
    unit = F.normalize(features.float(), dim=0, eps=1e-6)
    prototype = F.normalize(unit[:, object_mask].mean(dim=1),
                            dim=0, eps=1e-6)
    return (unit * prototype[:, None, None]).sum(dim=0)


def relation_measure(features, object_mask, background_mask):
    similarity = relation_map(features, object_mask)
    return similarity, similarity[object_mask].mean() - similarity[
        background_mask].mean()


def heldout_relation_stats(features, object_mask, background_mask):
    """Evaluate unseen object tokens against nearby background tokens."""
    positions = object_mask.flatten().nonzero().flatten()
    if len(positions) < 2:
        return None
    unit = F.normalize(features.float(), dim=0, eps=1e-6)
    flat = unit.flatten(1)
    prototype = F.normalize(flat[:, positions[::2]].mean(dim=1),
                            dim=0, eps=1e-6)
    heldout = (flat[:, positions[1::2]] * prototype[:, None]).sum(dim=0)
    background = (unit[:, background_mask] * prototype[:, None]).sum(dim=0)
    pairwise = heldout[:, None] - background[None, :]
    auc = ((pairwise > 0).float() + 0.5 * (pairwise == 0).float()).mean()
    return dict(gap=float((heldout.mean() - background.mean()).item()),
                auc=float(auc.item()), heldout_object_tokens=len(positions[1::2]),
                background_tokens=len(background))


def heldout_teacher_gap(features, object_mask, background_mask):
    stats = heldout_relation_stats(features, object_mask, background_mask)
    return None if stats is None else stats['gap']


def balanced_relation_loss(student, teacher, obj, bg):
    """Give object and background equal weight regardless of token counts."""
    return 0.5 * (F.mse_loss(student[obj], teacher[obj])
                  + F.mse_loss(student[bg], teacher[bg]))


def select_source_records(infos, domain, offset):
    """Select before seeing predictions: disjoint sequences, two sizes/role."""
    candidates = []
    for index, info in enumerate(infos):
        boxes = info['ann']['bboxes']
        if len(boxes) != 1:
            continue
        name = Path(info['filename']).stem
        if not name.startswith(domain + '_'):
            raise ValueError('Unexpected domain in source annotations')
        sequence = name.rsplit('_', 1)[0]
        candidates.append(dict(index=offset + index, sequence=sequence,
                               domain=domain, image=info['filename'],
                               original_short_edge=float(min(boxes[0][2:4]))))
    sequences = sorted({r['sequence'] for r in candidates})
    if len(sequences) < 2:
        raise ValueError('Need two source sequences for ' + domain)
    selected = []
    for role, seqs in [('fit', sequences[::2]), ('heldout', sequences[1::2])]:
        pool = sorted([r for r in candidates if r['sequence'] in seqs],
                      key=lambda r: (r['original_short_edge'], r['image']))
        if len(pool) < 2:
            raise ValueError('Not enough source images for ' + domain + '/' + role)
        for fraction in (0.25, 0.75):
            record = dict(pool[round((len(pool) - 1) * fraction)], role=role)
            selected.append(record)
    if len({r['index'] for r in selected}) != 4:
        raise ValueError('Source selection duplicated an image')
    return selected


def _classification_stats(model, features, obj, bg):
    logits = model.bbox_head.forward_single(features[0])[0].float()
    # Interpolate probabilities separately; sigmoid(interpolated logits)
    # would not equal interpolated detector probabilities.
    maps = dict(logit=logits, score=logits.sigmoid())
    result = {}
    for name, tensor in maps.items():
        value = F.interpolate(tensor, size=obj.shape, mode='bilinear',
                              align_corners=False)[0].mean(dim=0)
        result[name + '_object'] = float(value[obj].mean().item())
        result[name + '_background'] = float(value[bg].mean().item())
        result[name + '_gap'] = (result[name + '_object']
                                - result[name + '_background'])
    return result


def _tensor_sha256(value):
    """Stable byte hash for the regression branch used by the no-regression check."""
    values = value if isinstance(value, (tuple, list)) else (value,)
    digest = hashlib.sha256()
    for item in values:
        array = item.detach().float().cpu().contiguous().numpy()
        digest.update(str(array.shape).encode('ascii'))
        digest.update(array.tobytes())
    return digest.hexdigest()


def assess_evidence(samples):
    """Predeclared exploratory criteria; never authorize training automatically."""
    criteria = dict(teacher_auc=0.6, teacher_gap=0.02,
                    heldout_relative_mse_gain=0.01, score_epsilon=1e-6)
    heldout = [r for r in samples if r['role'] == 'heldout']
    if (len(heldout) != 4 or any(r.get('invalid_reason') for r in samples)
            or any(r.get('teacher_heldout') is None for r in heldout)):
        return dict(status='insufficient_evidence', criteria=criteria,
                    reasons=['Missing usable regions or held-out object tokens'],
                    automatic_training_authorized=False)
    domains = {}
    for domain in ('real', 'sim'):
        rows = [r for r in heldout if r['domain'] == domain]
        mean = lambda fn: sum(fn(r) for r in rows) / len(rows)
        auc = mean(lambda r: r['teacher_heldout']['auc'])
        gap = mean(lambda r: r['teacher_heldout']['gap'])
        control = mean(lambda r: r['control']['relation_mse'])
        distill = mean(lambda r: r['relation']['relation_mse'])
        mse_gain = (control - distill) / max(control, 1e-12)
        score_gain = mean(lambda r: r['relation']['classification']['score_gap']
                          - r['control']['classification']['score_gap'])
        student_auc_gain = mean(
            lambda r: r['relation']['student_heldout']['auc']
            - r['control']['student_heldout']['auc'])
        student_gap_gain = mean(
            lambda r: r['relation']['student_heldout']['gap']
            - r['control']['student_heldout']['gap'])
        object_drop = any(r['relation']['classification']['score_object'] <
                          r['control']['classification']['score_object'] - 1e-6
                          for r in rows)
        lost_riou = sum((r['baseline']['detection']['riou_hit'] or
                         r['control']['detection']['riou_hit']) and
                        not r['relation']['detection']['riou_hit'] for r in rows)
        lost_center = sum((r['baseline']['detection']['center_hit'] or
                           r['control']['detection']['center_hit']) and
                          not r['relation']['detection']['center_hit'] for r in rows)
        lost_output = sum((r['baseline']['detection']['output'] or
                           r['control']['detection']['output']) and
                          not r['relation']['detection']['output'] for r in rows)
        domains[domain] = dict(teacher_auc=auc, teacher_gap=gap,
            relative_mse_gain_over_control=mse_gain,
            score_gap_gain_over_control=score_gain,
            student_auc_gain_over_control=student_auc_gain,
            student_gap_gain_over_control=student_gap_gain,
            any_object_score_drop=object_drop,
            lost_riou_hits=lost_riou, lost_center_hits=lost_center,
            lost_outputs=lost_output)
    supported = all(d['teacher_auc'] >= 0.6 and d['teacher_gap'] >= 0.02
                    and d['relative_mse_gain_over_control'] >= 0.01
                    and d['score_gap_gain_over_control'] > 1e-6
                    and d['student_auc_gain_over_control'] >= -1e-6
                    and d['student_gap_gain_over_control'] >= -1e-6
                    and not d['any_object_score_drop']
                    and d['lost_riou_hits'] == 0
                    and d['lost_center_hits'] == 0
                    and d['lost_outputs'] == 0 for d in domains.values())
    adverse = any(d['teacher_auc'] <= 0.5 or d['lost_riou_hits'] > 0
                  or d['lost_center_hits'] > 0 or d['lost_outputs'] > 0
                  or d['score_gap_gain_over_control'] < -1e-6
                  for d in domains.values())
    return dict(status=('supports_next_controlled_experiment' if supported else
                        'no_support_in_this_probe' if adverse else
                        'insufficient_evidence'), criteria=criteria,
                domains=domains, automatic_training_authorized=False)


def probe(project_root, gpu):
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from mmcv.runner import load_checkpoint
    from mmrotate.datasets import build_dataset
    from mmrotate.models import build_detector

    root = Path(project_root).resolve()
    if Path.cwd().resolve() != root:
        raise ValueError('Run from the project root')
    cfg_path = root / CONFIG
    cfg = Config.fromfile(str(cfg_path))
    k1_cfg_path = root / 'crane_project/configs/crane_symeood_k1.py'
    k1_cfg = Config.fromfile(str(k1_cfg_path))
    checkpoint = (root / cfg.load_from).resolve()
    if sha256(checkpoint) != EXPECTED_BASELINE_SHA256:
        raise ValueError('K1 checkpoint identity mismatch')
    if (cfg.data.samples_per_gpu != 2
            or cfg.model.bbox_head.use_semantic_cls_adapter is not True):
        raise ValueError('Unexpected source data or classification adapter')
    if (k1_cfg.runner.max_epochs != 24
            or k1_cfg.model.backbone.frozen_stages != 1):
        raise ValueError('Unexpected original K1 training config')
    step_lr = float(k1_cfg.optimizer.lr)

    random.seed(1701)
    np.random.seed(1701)
    torch.manual_seed(1701)
    torch.cuda.manual_seed_all(1701)
    torch.cuda.set_device(gpu)
    dataset = build_dataset(cfg.data.train)
    if not hasattr(dataset, 'datasets') or len(dataset.datasets) != 2:
        raise ValueError('Expected real and sim source datasets')
    selection = []
    offset = 0
    for domain, subset in zip(('real', 'sim'), dataset.datasets):
        selection.extend(select_source_records(subset.data_infos, domain, offset))
        offset += len(subset)

    model_cfg = cfg.model.copy()
    model_cfg['semantic_distillation'] = None
    model = build_detector(model_cfg, train_cfg=cfg.get('train_cfg'),
                           test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    load_checkpoint(model, str(checkpoint), map_location='cpu', strict=False)
    model = model.cuda(gpu)
    adapter = model.bbox_head.semantic_cls_adapter
    if torch.count_nonzero(adapter.weight).item():
        raise ValueError('K1 load did not leave adapter at zero')
    # This is an adapter-only diagnostic in both arms, not a K1 training run.
    for param in model.parameters():
        param.requires_grad_(False)
    for param in adapter.parameters():
        param.requires_grad_(True)
    model.eval()
    torch.cuda.reset_peak_memory_stats(gpu)
    device = torch.device('cuda', gpu)
    records = []
    for item in selection:
        batch = scatter(collate([dataset[item['index']]], samples_per_gpu=1),
                        [gpu])[0]
        meta = batch['img_metas'][0]
        if Path(meta['filename']).name != item['image']:
            raise ValueError('Dataset replaced the preselected source image')
        teacher = batch['teacher_features'][0].float()
        if teacher.shape != (1024, 64, 64) or not torch.isfinite(teacher).all():
            raise ValueError('Invalid teacher cache tensor')
        record = dict(item, transformed_image=meta['filename'],
                      cache_transform=_check_cached_feature_transform(cfg, batch),
                      meta=meta, boxes=batch['gt_bboxes'][0].cpu(),
                      labels=batch['gt_labels'][0].cpu())
        try:
            obj, bg = object_background_masks(batch['gt_bboxes'][0], meta,
                                             64, 64, device)
        except ValueError as error:
            record['invalid_reason'] = str(error)
            records.append(record)
            continue
        with torch.no_grad():
            features = model.extract_feat(batch['img'])
            record.update(features=[v.cpu() for v in features],
                          obj=obj.cpu(), bg=bg.cpu(),
                          teacher_map=relation_map(teacher, obj).cpu(),
                          teacher_heldout=heldout_relation_stats(teacher, obj, bg),
                          object_tokens=int(obj.sum()), background_tokens=int(bg.sum()))
        records.append(record)

    from crane_project.tools.eval_crane_offline import compute_riou

    def device_record(record):
        return (tuple(v.to(device) for v in record['features']),
                record['obj'].to(device), record['bg'].to(device),
                record['teacher_map'].to(device))

    def evaluate(record):
        features, obj, bg, target = device_record(record)
        with torch.no_grad():
            student = model.bbox_head.forward_semantic_distillation_features(
                features, protect_geometry=True, feature_level=0)[0]
            student = F.interpolate(student.float(), size=obj.shape,
                                    mode='bilinear', align_corners=False)[0]
            relation = relation_map(student, obj)
            loss = balanced_relation_loss(relation, target, obj, bg)
            regression = tuple(model.bbox_head.forward_single(level)[1]
                               for level in features)
            prediction = model.simple_test_from_features(
                features, [record['meta']], rescale=False)[0][0]
            boxes = np.asarray(prediction)
            gt = record['boxes'][0].numpy()
            if len(boxes):
                pred = boxes[np.argmax(boxes[:, -1])]
                distance = float(np.linalg.norm(pred[:2] - gt[:2]))
                riou = compute_riou(pred[:5], gt)
                detection = dict(output=True, center_distance=distance,
                                 center_hit=distance < 15, riou=riou,
                                 riou_hit=riou >= 0.5, box=pred.tolist())
            else:
                detection = dict(output=False, center_distance=None,
                                 center_hit=False, riou=0., riou_hit=False, box=None)
            return dict(relation_mse=float(loss.item()),
                        student_heldout=heldout_relation_stats(student, obj, bg),
                        classification=_classification_stats(model, features, obj, bg),
                        regression_sha256=_tensor_sha256(regression),
                        detection=detection)

    usable = [r for r in records if 'invalid_reason' not in r]
    for record in usable:
        record['baseline'] = evaluate(record)
    fit = {domain: [r for r in usable if r['role'] == 'fit'
                    and r['domain'] == domain] for domain in ('real', 'sim')}
    # Save the changing parameters and all loss buffers/counters for matched arms.
    saved = [p.detach().clone() for p in adapter.parameters()]
    buffers = {name: value.detach().clone()
               for name, value in model.bbox_head.named_buffers()}
    assigner = model.bbox_head.assigner
    call_count = getattr(assigner, '_local_call_count', None)

    def restore():
        with torch.no_grad():
            for param, value in zip(adapter.parameters(), saved):
                param.copy_(value)
            for name, value in model.bbox_head.named_buffers():
                value.copy_(buffers[name])
        if call_count is not None:
            assigner._local_call_count = call_count
        model.eval()

    steps = 10
    relation_weight = 0.05
    histories = {}
    complete = len(usable) == 8 and all(len(fit[d]) == 2 for d in fit)
    try:
        if complete:
            for arm in ('control', 'relation'):
                restore()
                optimizer = torch.optim.SGD(adapter.parameters(), lr=step_lr,
                    momentum=k1_cfg.optimizer.momentum,
                    weight_decay=k1_cfg.optimizer.weight_decay)
                history = []
                for step in range(steps):
                    pair = [fit[d][step % 2] for d in ('real', 'sim')]
                    features = tuple(torch.cat([r['features'][level] for r in pair])
                                     .to(device) for level in range(5))
                    model.bbox_head.train()
                    optimizer.zero_grad()
                    outputs = model.bbox_head(features)
                    losses = model.bbox_head.loss(*outputs,
                        gt_bboxes=[r['boxes'].to(device) for r in pair],
                        gt_labels=[r['labels'].to(device) for r in pair],
                        img_metas=[r['meta'] for r in pair])
                    detection_loss = _loss_sum(losses)
                    student = model.bbox_head.forward_semantic_distillation_features(
                        features, protect_geometry=True, feature_level=0)[0]
                    student = F.interpolate(student.float(), size=(64, 64),
                                            mode='bilinear', align_corners=False)
                    rel_terms = []
                    for index, record in enumerate(pair):
                        obj, bg = record['obj'].to(device), record['bg'].to(device)
                        rel_terms.append(balanced_relation_loss(
                            relation_map(student[index], obj),
                            record['teacher_map'].to(device), obj, bg))
                    relation_loss = torch.stack(rel_terms).mean()
                    total = detection_loss + (relation_weight * relation_loss
                                               if arm == 'relation' else 0.)
                    if not torch.isfinite(total).item():
                        raise ValueError('Nonfinite short-probe loss')
                    relation_grad_norm = None
                    if arm == 'relation' and step == 0:
                        relation_grads = torch.autograd.grad(
                            relation_loss, tuple(adapter.parameters()),
                            retain_graph=True, allow_unused=True)
                        relation_terms_grad = [g.detach().float().square().sum()
                                               for g in relation_grads if g is not None]
                        relation_grad_norm = (torch.stack(relation_terms_grad).sum()
                                              .sqrt() if relation_terms_grad else
                                              torch.zeros((), device=device))
                        if (not torch.isfinite(relation_grad_norm).item()
                                or relation_grad_norm.item() == 0.):
                            raise ValueError('Relation loss did not reach adapter')
                    total.backward()
                    if any(p.grad is not None for name, p in model.named_parameters()
                           if 'semantic_cls_adapter' not in name):
                        raise ValueError('Gradient escaped the adapter')
                    norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(),
                        max_norm=k1_cfg.optimizer_config.grad_clip.max_norm,
                        norm_type=k1_cfg.optimizer_config.grad_clip.norm_type)
                    if not torch.isfinite(norm).item():
                        raise ValueError('Nonfinite adapter gradient')
                    optimizer.step()
                    model.eval()
                    entry = dict(step=step + 1, detection_loss=float(detection_loss.item()),
                                 relation_loss=float(relation_loss.item()),
                                 adapter_gradient_norm=float(norm.item()))
                    if relation_grad_norm is not None:
                        entry['relation_adapter_gradient_norm'] = float(
                            relation_grad_norm.item())
                    if step in (0, 4, 9):
                        entry['heldout'] = [dict(index=r['index'], **evaluate(r))
                                            for r in usable if r['role'] == 'heldout']
                    history.append(entry)
                histories[arm] = history
                for record in usable:
                    record[arm] = evaluate(record)
        if complete:
            for record in usable:
                hashes = {record[arm]['regression_sha256']
                          for arm in ('baseline', 'control', 'relation')}
                if len(hashes) != 1:
                    raise ValueError('Regression branch changed across probe arms')
    finally:
        restore()
    samples_report = [{k: v for k, v in record.items() if k not in
                      ('features', 'obj', 'bg', 'teacher_map', 'boxes', 'labels', 'meta')}
                     for record in records]
    assessment = (assess_evidence(samples_report) if complete else
                  dict(status='insufficient_evidence',
                       reasons=['Unusable preselected regions; no replacement sampling'],
                       automatic_training_authorized=False))
    return dict(protocol=PROTOCOL, revision='r2_matched_short_source',
        evidence_role='source_only_design_probe', fixed_seed=1701,
        checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
        probe_sha256=sha256(Path(__file__)),
        config=str(cfg_path), config_sha256=sha256(cfg_path),
        k1_config_sha256=sha256(k1_cfg_path),
        selection_rule='alternating sequences; 25/75 percentiles of GT short edge',
        sample_count=len(records), training_epochs=0,
        temporary_steps_per_arm=steps if complete else 0,
        total_temporary_optimizer_steps=2 * steps if complete else 0,
        trainable_parameters='classification adapter only in both arms',
        detection_supervision='unchanged K1 main-head detection loss',
        optimizer=dict(k1_cfg.optimizer), clip=dict(k1_cfg.optimizer_config.grad_clip),
        lr_schedule='constant K1 base LR for 10 diagnostic steps; no warmup',
        relation_weight=relation_weight,
        frozen_features=True, checkpoint_saved=False,
        source_detection_coordinates='resized/flipped image; center threshold 15px',
        assessment=assessment, samples=samples_report, learning_curves=histories,
        peak_allocated_mib=float(torch.cuda.max_memory_allocated(gpu) / 2**20))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--out-json', required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the source probe')
    out = Path(args.out_json)
    if out.exists():
        raise FileExistsError('Refusing to overwrite ' + str(out))
    report = probe(args.project_root, args.gpu)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n',
                   encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
