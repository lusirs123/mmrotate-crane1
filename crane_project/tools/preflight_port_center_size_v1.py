#!/usr/bin/env python3
"""Bounded TRAIN-only B/D config, loss and gradient check; no optimizer/TEST."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CONTROL = ROOT / 'crane_project/configs/crane_symeood_k1_port_day2night_aug_b_v1.py'
EXPERIMENT = ROOT / 'crane_project/configs/crane_symeood_k1_port_day2night_center_size_d_v1.py'
EXPECTED = dict(type='CenterSizeCompensationLoss', beta=0.1,
                loss_weight=0.25, eps=1e-6, reduction='mean')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check_configs():
    from mmcv import Config
    b, d = (Config.fromfile(str(p)) for p in (CONTROL, EXPERIMENT))
    base, changed = deepcopy(b.to_dict()), deepcopy(d.to_dict())
    if changed['model']['bbox_head'].pop('center_size_compensation') != EXPECTED:
        raise ValueError('D compensation differs from the registered fixed design')
    changed['work_dir'] = base['work_dir']
    if changed != base:
        raise ValueError('B/D differ beyond compensation and work_dir')
    if (b.load_from is not None or b.resume_from is not None
            or b.runner.max_epochs != 24 or b.data.samples_per_gpu != 2
            or b.model.backbone.frozen_stages != 1
            or dict(b.optimizer) != dict(type='SGD', lr=.0025,
                                       momentum=.9, weight_decay=.0001)
            or b.model.test_cfg.score_thr != .05
            or b.model.test_cfg.max_per_img != 1):
        raise ValueError('Unexpected B initialization/training/inference contract')
    if b.model.bbox_head.get('center_size_compensation') is not None:
        raise ValueError('B already has compensation')
    for entry, split in zip(d.data.train, ['train', 'train_sim']):
        if (entry.data_root != 'crane_project/data/crane_grab_port_day2night_v1/'
                or entry.ann_file != split + '/annfiles/'
                or entry.img_prefix != split + '/images/'):
            raise ValueError('Expected frozen seq06-restored TRAIN paths')
        shrink = [s for s in entry.pipeline if s['type'] == 'PortIsotropicShrink']
        if shrink != [dict(type='PortIsotropicShrink', prob=.5, scale_range=(.5, 1.))]:
            raise ValueError('Scale B augmentation changed')
    if len(d.data.train) != 2:
        raise ValueError('Expected real + sim TRAIN')
    return b, d


def fixed_train_specs(cfg, scale):
    """Keep the TRAIN GT loader; remove stochastic flip/shrink for this probe."""
    specs = deepcopy(cfg.data.train)
    for entry in specs:
        for step in entry.pipeline:
            if step['type'] == 'PortIsotropicShrink':
                step['prob'] = 0. if scale == 1. else 1.
                step['scale_range'] = (scale, scale)
            elif step['type'] == 'RRandomFlip':
                step['flip_ratio'] = [0., 0., 0.]
    return specs


def set_assignment_phase(head, phase):
    """The Python assigner counter is absent from checkpoint state_dict."""
    head.assigner._local_call_count = (
        0 if phase == 'warmup_o2m' else
        2 * max(2 * head.assigner.o2m_warmup_iters, head.assigner.warmup_iters))
    head.loss_cls._local_iter.fill_(
        0 if phase == 'warmup_o2m' else
        head.loss_cls.warmup_iters * head.loss_cls.loss_call_factor)


def norm(grads):
    return sum(float(g.detach().float().square().sum())
               for g in grads if g is not None) ** .5


def sum_loss(value):
    return sum(value) if isinstance(value, (list, tuple)) else value


def probe(cfg, gpu, sweep_path):
    import torch
    from mmcv.parallel import collate, scatter
    from mmcv.runner import load_checkpoint
    from mmrotate.datasets import build_dataset
    from mmrotate.models import build_detector
    from crane_project.tools.ckpt_sweep import annotation_set_sha256

    selection = json.loads((sweep_path / 'sweep_results.json').read_text())
    selected = selection['selected_checkpoint']
    record = selection['all_checkpoints'][selected]
    checkpoint = Path(record['checkpoint']).resolve()
    checkpoint_sha = sha(checkpoint)
    if (selection.get('evidence_role') != 'source_val_checkpoint_selection'
            or selected != 'epoch_24' or selection['config_sha256'] != sha(CONTROL)
            or Path(selection['selected_path']).resolve() != checkpoint
            or checkpoint_sha != record['checkpoint_sha256']):
        raise ValueError('Probe reference must be the frozen VAL-selected scale B epoch_24')
    random.seed(1701)
    np.random.seed(1701)
    torch.manual_seed(1701)
    torch.cuda.set_device(gpu)
    torch.cuda.manual_seed_all(1701)
    model = build_detector(deepcopy(cfg.model))
    load_checkpoint(model, str(checkpoint), map_location='cpu', strict=True)
    model.cuda(gpu).train()
    head = model.bbox_head
    reg = head.retina_reg.weight
    cls = head.retina_cls.weight
    fpn = [p for p in model.neck.parameters() if p.requires_grad]
    backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    if not fpn or not backbone:
        raise ValueError('Expected the inherited trainable FPN/backbone stages')
    parameters = [reg, cls] + fpn + backbone
    torch.cuda.reset_peak_memory_stats(gpu)
    rows, problems = [], []
    for scale in [1., .5]:
        dataset = build_dataset(fixed_train_specs(cfg, scale))
        if [len(part) for part in dataset.datasets] != [1810, 748]:
            raise ValueError('Expected frozen TRAIN real1810 + sim748')
        for phase, indices in [('warmup_o2m', [0, 1810]),
                               ('late_o2o', [905, 1810 + 374])]:
            set_assignment_phase(head, phase)
            batch = scatter(collate([dataset[i] for i in indices], samples_per_gpu=2), [gpu])[0]
            if any(meta.get('flip', False) for meta in batch['img_metas']):
                raise ValueError('Probe unexpectedly flipped an image')
            losses = model(return_loss=True, **batch)
            extra = sum_loss(losses['loss_center_size_compensation'])
            bbox = sum_loss(losses['loss_bbox'])
            classification = sum_loss(losses['loss_cls'])
            total = sum(sum_loss(v) for k, v in losses.items() if 'loss' in k)
            if not all(bool(torch.isfinite(v).all()) for v in (extra, bbox, classification, total)):
                raise RuntimeError('Nonfinite short-chain loss')
            grads = torch.autograd.grad(extra, parameters, retain_graph=True, allow_unused=True)
            bbox_grads = torch.autograd.grad(bbox, [reg] + fpn,
                                             retain_graph=True, allow_unused=True)
            if any(g is not None and not bool(torch.isfinite(g).all()) for g in grads + bbox_grads):
                raise RuntimeError('Nonfinite short-chain gradient')
            fpn_grad = norm(grads[2:2 + len(fpn)])
            ref_fpn_grad = norm(bbox_grads[1:])
            row = dict(phase=phase, shrink_scale=scale, source_indices=indices,
                images=[meta['filename'] for meta in batch['img_metas']],
                img_shapes=[list(meta['img_shape']) for meta in batch['img_metas']],
                positive_count=int(losses['center_size_positive_count']),
                loss_compensation=float(extra.detach()), loss_symkld=float(bbox.detach()),
                loss_classification=float(classification.detach()),
                compensation_to_symkld_loss_ratio=float(extra.detach()) / max(float(bbox.detach()), 1e-12),
                compensation_reg_gradient_l2=norm(grads[:1]),
                compensation_cls_gradient_l2=norm(grads[1:2]),
                compensation_fpn_gradient_l2=fpn_grad,
                compensation_backbone_gradient_l2=norm(grads[2 + len(fpn):]),
                compensation_to_symkld_fpn_gradient_ratio=fpn_grad / max(ref_fpn_grad, 1e-12),
                compensation_direct_angle_output_gradient_l2=(
                    norm([grads[0][4::5]]) if grads[0] is not None else 0.))
            if (row['positive_count'] <= 0 or row['loss_compensation'] <= 0
                    or row['compensation_reg_gradient_l2'] <= 0 or fpn_grad <= 0
                    or row['compensation_backbone_gradient_l2'] <= 0
                    or row['compensation_cls_gradient_l2'] != 0
                    or row['compensation_direct_angle_output_gradient_l2'] != 0):
                problems.append(dict(phase=phase, scale=scale,
                                     reason='Positive signal or designed gradient scope not verified'))
            model.zero_grad(set_to_none=True)
            total.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all())
                   for p in model.parameters()):
                raise RuntimeError('Nonfinite combined-training gradient')
            row['combined_gradient_norm_before_clip'] = float(torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg.optimizer_config.grad_clip.max_norm,
                norm_type=cfg.optimizer_config.grad_clip.norm_type))
            print(phase, 'scale', scale, 'compensation', row['loss_compensation'],
                  'FPN gradient', fpn_grad, flush=True)
            rows.append(row)
            model.zero_grad(set_to_none=True)
            del batch, losses, extra, bbox, classification, total, grads, bbox_grads
    return dict(status='READY_FOR_TRAINING' if not problems else 'NOT_READY',
        problems=problems, rows=rows,
        probe_checkpoint=str(checkpoint), probe_checkpoint_sha256=checkpoint_sha,
        reference_selection_sha256=sha(sweep_path / 'sweep_results.json'),
        training_epochs=0, optimizer_steps=0,
        formal_training_initialization='Inherited B ImageNet initialization; no load_from/resume_from',
        source_annotations_sha256={split:annotation_set_sha256(
            str(ROOT / cfg.data.train[0].data_root / split / 'annfiles'))
            for split in ['train', 'train_sim']},
        peak_allocated_mib=torch.cuda.max_memory_allocated(gpu) / 2**20)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpu', type=int, default=3)
    ap.add_argument('--config-only', action='store_true')
    ap.add_argument('--reference-sweep', default=
        'work_dirs/crane_symeood_k1_port_day2night_aug_b_v1/val_sweep_port_v1')
    ap.add_argument('--out-json', required=True)
    args = ap.parse_args()
    os.chdir(ROOT)
    output = Path(args.out_json).resolve()
    if output.exists():
        raise FileExistsError(output)
    _, cfg = check_configs()
    report = dict(protocol='port_center_size_compensation_preflight_v1',
                  evidence_role='source_train_only_implementation_check',
                  config_b_sha256=sha(CONTROL), config_d_sha256=sha(EXPERIMENT),
                  script_sha256=sha(__file__),
                  head_sha256=sha(ROOT / 'mmrotate/models/dense_heads/sym_eood_head.py'),
                  loss_sha256=sha(ROOT / 'mmrotate/models/losses/center_size_compensation.py'),
                  fixed_loss_config=EXPECTED)
    if args.config_only:
        report['status'] = 'CONFIG_ONLY_PASS_RUNTIME_NOT_CHECKED'
    else:
        report.update(probe(cfg, args.gpu, Path(args.reference_sweep).resolve()))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report['status'] == 'NOT_READY':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
