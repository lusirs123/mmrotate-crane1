"""Check matched K1 A/B configs and one short source gradient chain before training."""

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from crane_project.tools.preflight_k1_dino_warmstart_v2 import (
    EXPECTED_BASELINE_SHA256, sha256)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / 'crane_project/configs'
CONTROL = CONFIG_DIR / 'crane_symeood_k1_candidate_selection_a_v1.py'
EXPERIMENT = CONFIG_DIR / 'crane_symeood_k1_candidate_selection_b_v1.py'
BASE = CONFIG_DIR / 'crane_symeood_k1.py'


def check_configs():
    from mmcv import Config

    baseline, a, b = (Config.fromfile(str(path))
                      for path in (BASE, CONTROL, EXPERIMENT))
    shared = ('custom_imports', 'data', 'train_pipeline', 'test_pipeline',
              'optimizer', 'optimizer_config', 'lr_config', 'runner',
              'checkpoint_config', 'evaluation', 'load_from', 'resume_from')
    for key in shared:
        if a.get(key) != b.get(key):
            raise ValueError('A/B config mismatch: ' + key)
    a_model, b_model = copy.deepcopy(dict(a.model)), copy.deepcopy(dict(b.model))
    a_selection = a_model['bbox_head'].pop('candidate_selection', None)
    b_selection = b_model['bbox_head'].pop('candidate_selection', None)
    if a_selection is not None or a_model != b_model:
        raise ValueError('A/B model structures or losses differ beyond selection')
    expected = dict(score_threshold=0.05, iou_threshold=0.5,
                    margin=0.1, loss_weight=0.05)
    if dict(b_selection or {}) != expected:
        raise ValueError('B selection settings differ from preregistration')
    if a_model != dict(baseline.model):
        raise ValueError('A is not the native K1 model')
    if (a.runner.max_epochs != 24 or a.data.samples_per_gpu != 2
            or a.model.backbone.frozen_stages != 1
            or dict(a.optimizer) != dict(baseline.optimizer)
            or dict(a.lr_config) != dict(baseline.lr_config)
            or a.model.test_cfg.score_thr != 0.05
            or a.model.test_cfg.max_per_img != 1):
        raise ValueError('K1 training/inference budget changed')
    checkpoint = (ROOT / a.load_from).resolve()
    if sha256(checkpoint) != EXPECTED_BASELINE_SHA256:
        raise ValueError('K1 epoch_20 checkpoint SHA256 mismatch')
    for name, cfg in (('A', a), ('B', b)):
        if cfg.resume_from is not None:
            raise ValueError(name + ' unexpectedly resumes training')
        if any(step.get('type') == 'LoadDinoFeatureFromCache'
               for step in cfg.train_pipeline):
            raise ValueError(name + ' unexpectedly reads DINO cache')
        if cfg.model.bbox_head.get('use_semantic_cls_adapter', False):
            raise ValueError(name + ' unexpectedly enables semantic adapter')
    return a, b, checkpoint


def gradient_norm(grads):
    return sum(float(g.detach().float().square().sum())
               for g in grads if g is not None) ** 0.5


def one_source_probe(cfg, checkpoint, gpu):
    from mmcv.parallel import collate, scatter
    from mmcv.runner import load_checkpoint
    from mmrotate.datasets import build_dataset
    from mmrotate.models import build_detector

    random.seed(1701)
    np.random.seed(1701)
    torch.manual_seed(1701)
    torch.cuda.manual_seed_all(1701)
    torch.cuda.set_device(gpu)
    dataset = build_dataset(cfg.data.train)
    if not hasattr(dataset, 'datasets') or len(dataset.datasets) != 2:
        raise ValueError('Expected real + sim source train')
    real_count = len(dataset.datasets[0])
    indices = (0, real_count // 2, real_count,
               real_count + len(dataset.datasets[1]) // 2)
    model = build_detector(cfg.model,
                           train_cfg=cfg.get('train_cfg'),
                           test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, str(checkpoint), map_location='cpu', strict=True)
    model.cuda(gpu).train()
    cls_param = model.bbox_head.retina_cls.weight
    reg_param = model.bbox_head.retina_reg.weight
    fpn_params = [p for p in model.neck.parameters() if p.requires_grad]
    if not fpn_params:
        raise ValueError('Expected trainable FPN parameters')
    rows = []
    torch.cuda.reset_peak_memory_stats(gpu)
    for index in indices:
        start = time.perf_counter()
        batch = scatter(collate([dataset[index]], samples_per_gpu=1), [gpu])[0]
        losses = model(return_loss=True, **batch)
        select = losses['loss_candidate_selection']
        row = dict(index=index, image=batch['img_metas'][0]['filename'],
                   loss=float(select.detach()),
                   valid_images=int(losses['selection_valid_images']),
                   active_images=int(losses['selection_active_images']),
                   good_labeled_negative=int(
                       losses['selection_good_labeled_negative']),
                   any_good_labeled_negative=int(
                       losses['selection_any_good_labeled_negative']),
                   active_good_labeled_negative=int(
                       losses['selection_active_good_labeled_negative']),
                   no_good=int(losses['selection_no_good']),
                   empty_gt=int(losses['selection_empty_gt']))
        cls_loss = sum(losses['loss_cls'])
        bbox_loss = sum(losses['loss_bbox'])
        row.update(classification_loss=float(cls_loss.detach()),
                   regression_loss=float(bbox_loss.detach()),
                   selection_to_classification_loss_ratio=(
                       float(select.detach()) / max(float(cls_loss.detach()), 1e-12)),
                   selection_to_regression_loss_ratio=(
                       float(select.detach()) / max(float(bbox_loss.detach()), 1e-12)))
        if row['active_images']:
            grads = torch.autograd.grad(
                select, [cls_param] + fpn_params + [reg_param],
                retain_graph=True, allow_unused=True)
            row.update(selection_cls_gradient_l2=gradient_norm(grads[:1]),
                       selection_fpn_gradient_l2=gradient_norm(
                           grads[1:1 + len(fpn_params)]),
                       selection_reg_gradient_l2=gradient_norm(grads[-1:]))
            detection = sum(
                sum(v) if isinstance(v, (list, tuple)) else v
                for k, v in losses.items()
                if k.startswith('loss_') and k != 'loss_candidate_selection')
            det_grad = torch.autograd.grad(detection, cls_param,
                                           allow_unused=True)[0]
            sel_grad = grads[0]
            row['classification_gradient_cosine'] = float(
                torch.nn.functional.cosine_similarity(
                    sel_grad.flatten().float(),
                    det_grad.flatten().float(), dim=0))
            if (row['selection_cls_gradient_l2'] <= 0
                    or row['selection_fpn_gradient_l2'] <= 0
                    or row['selection_reg_gradient_l2'] != 0):
                raise RuntimeError('selection gradient scope differs from design')
        torch.cuda.synchronize(gpu)
        row['duration_seconds'] = time.perf_counter() - start
        rows.append(row)
        del batch, losses
    status = ('READY_FOR_FORMAL_TRAINING' if any(r['active_images'] for r in rows)
              else 'NO_ACTIVE_SELECTION_SIGNAL_ON_SAMPLED_SOURCE')
    valid_count = sum(r['valid_images'] for r in rows)
    conflict_count = sum(r['good_labeled_negative'] for r in rows)
    conflict_rate = (conflict_count / valid_count if valid_count else None)
    active_count = sum(r['active_images'] for r in rows)
    active_conflict_count = sum(r['active_good_labeled_negative'] for r in rows)
    active_conflict_rate = (active_conflict_count / active_count
                            if active_count else None)
    return dict(rows=rows, status=status,
                conflict_count=conflict_count,
                valid_candidate_image_count=valid_count,
                conflict_fraction_among_valid_images=conflict_rate,
                any_good_negative_image_count=sum(
                    r['any_good_labeled_negative'] for r in rows),
                active_selection_image_count=active_count,
                active_conflict_count=active_conflict_count,
                active_conflict_fraction=active_conflict_rate,
                conflict_over_30_percent_warning=(
                    active_conflict_rate is not None
                    and active_conflict_rate > 0.30),
                conflict_warning_is_diagnostic_only=True,
                peak_allocated_mib=torch.cuda.max_memory_allocated(gpu) / 2**20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--out-json', required=True)
    args = parser.parse_args()
    output = Path(args.out_json)
    if output.exists():
        raise FileExistsError(output)
    if Path.cwd().resolve() != ROOT:
        raise ValueError('Run from the project root')
    a, b, checkpoint = check_configs()
    report = dict(protocol='k1_candidate_selection_v1_preflight',
                  config_a_sha256=sha256(CONTROL),
                  config_b_sha256=sha256(EXPERIMENT),
                  checkpoint=str(checkpoint),
                  checkpoint_sha256=sha256(checkpoint),
                  source_only=True,
                  **one_source_probe(b, checkpoint, args.gpu))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
