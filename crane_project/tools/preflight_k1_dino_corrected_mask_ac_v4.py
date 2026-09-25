"""Verify the corrected-mask A/C comparison before source training.

Uses one temporary source optimization step per arm; saves no checkpoint and
does not read VAL or TEST. Run from the project root on the server.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from crane_project.tools.preflight_k1_dino_fpn_gradient_v3 import probe_arm
from crane_project.tools.preflight_k1_dino_warmstart_v2 import (
    EXPECTED_BASELINE_SHA256, sha256)
from crane_project.tools.probe_k1_dino_corrected_mask_source_v4 import (
    _verify_corrected_runtime)


def validate_pair(configs):
    """Require the distillation loss to be the only training difference."""
    if len(configs) != 2:
        raise ValueError('Exactly A and C configurations are required')
    a, c = configs
    keys = set(a.keys()) | set(c.keys())
    for key in keys - {'model', 'work_dir'}:
        if a.get(key) != c.get(key):
            raise ValueError('A/C differ in ' + key)
    if a.work_dir == c.work_dir or not all(
            cfg.work_dir.endswith('corrected_mask_{}_v4'.format(arm))
            for cfg, arm in zip(configs, 'ac')):
        raise ValueError('A/C need separate corrected-mask V4 work dirs')
    for cfg, arm in zip(configs, 'AC'):
        if cfg.model.backbone.frozen_stages != 4:
            raise ValueError('{} must freeze all ResNet stages'.format(arm))
        if not cfg.model.bbox_head.use_semantic_cls_adapter:
            raise ValueError('{} lacks the classification adapter'.format(arm))
        if (cfg.runner.max_epochs != 4 or cfg.optimizer.lr != 0.00025
                or cfg.data.samples_per_gpu != 2):
            raise ValueError('{} changes the matched training budget'.format(arm))
        if [(item.ann_file, item.img_prefix) for item in cfg.data.train] != [
                ('train/annfiles/', 'train/images/'),
                ('train_sim/annfiles/', 'train/images/')]:
            raise ValueError('{} changes source training data'.format(arm))
        if sum(step['type'] == 'LoadDinoFeatureFromCache'
               for step in cfg.train_pipeline) != 1:
            raise ValueError('{} changes the cache pipeline'.format(arm))
        if cfg.resume_from is not None:
            raise ValueError('{} unexpectedly resumes training'.format(arm))
    if a.model.get('semantic_distillation') is not None:
        raise ValueError('A must have no distillation loss')
    loss = c.model.get('semantic_distillation')
    if (not loss or not loss.get('enabled')
            or loss.get('mode') != 'feature'
            or loss.get('scope') != 'foreground'
            or loss.get('feature_level') != 0
            or loss.get('loss_weight') != 0.05
            or loss.get('max_tokens') != 4096
            or loss.get('teacher_channels') != 1024
            or loss.get('protect_geometry') is not False):
        raise ValueError('C has the wrong feature-distillation contract')
    a_model = dict(a.model)
    c_model = dict(c.model)
    a_model.pop('semantic_distillation', None)
    c_model.pop('semantic_distillation', None)
    if a_model != c_model:
        raise ValueError('A/C change model settings beyond distillation')
    checkpoint = Path(a.load_from).resolve()
    if sha256(checkpoint) != EXPECTED_BASELINE_SHA256:
        raise ValueError('K1 epoch_20 checkpoint SHA256 mismatch')
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-a', required=True)
    parser.add_argument('--config-c', required=True)
    parser.add_argument('--gpu', type=int, default=0,
                        help='Logical CUDA device after CUDA_VISIBLE_DEVICES')
    parser.add_argument('--out-json', required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('One CUDA GPU is required for the real-batch probe')
    from mmcv import Config
    from mmrotate.models import build_detector

    paths = [Path(args.config_a), Path(args.config_c)]
    cfgs = [Config.fromfile(str(path)) for path in paths]
    checkpoint = validate_pair(cfgs)
    for cfg, arm in zip(cfgs, 'AC'):
        if Path(cfg.work_dir).exists():
            raise FileExistsError('{} work dir already exists: {}'.format(
                arm, cfg.work_dir))
    torch.cuda.set_device(args.gpu)
    runtime_model = build_detector(cfgs[1].model,
                                   train_cfg=cfgs[1].get('train_cfg'),
                                   test_cfg=cfgs[1].get('test_cfg'))
    runtime_model.init_weights()
    runtime_model = runtime_model.cuda(args.gpu)
    _verify_corrected_runtime(runtime_model, args.gpu)
    del runtime_model

    results = [probe_arm(cfg, arm, checkpoint, args.gpu)
               for cfg, arm in zip(cfgs, 'AC')]
    if len({item['source_image'] for item in results}) != 1:
        raise ValueError('A/C source samples are not aligned')
    if len({item['cache_transform']['flip_direction'] for item in results}) != 1:
        raise ValueError('A/C source augmentations are not aligned')
    if len({item['cache_transform']['cache_path'] for item in results}) != 1:
        raise ValueError('A/C DINO cache entries are not aligned')
    if len({item['foreground_fpn_pixels'] for item in results}) != 1:
        raise ValueError('A/C foreground masks are not aligned')
    if results[0]['distillation_loss'] is not None:
        raise ValueError('A unexpectedly returned a distillation loss')
    if not results[1]['distillation_gradient_to_fpn']:
        raise ValueError('C distillation gradient did not reach the FPN')

    root = Path(__file__).resolve().parents[2]
    report = dict(
        protocol='k1_dino_corrected_mask_ac_preflight_v4',
        decision='CORRECTED_MASK_AC_READY', source_only=True,
        full_training_performed=False,
        temporary_probe_optimizer_steps_per_arm=1,
        corrected_mask_runtime_verified=True,
        corrected_downsample_runtime_verified=True,
        k1_checkpoint=str(checkpoint),
        k1_checkpoint_sha256=sha256(checkpoint),
        config_sha256={arm: sha256(path)
                       for arm, path in zip('AC', paths)},
        mask_implementation_sha256=sha256(
            root / 'mmrotate/models/detectors/sym_eood_detector.py'),
        loss_implementation_sha256=sha256(
            root / 'mmrotate/models/losses/semantic_feature_distill.py'),
        arms=results)
    out = Path(args.out_json)
    if out.exists():
        raise FileExistsError('Refusing to overwrite ' + str(out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('CORRECTED_MASK_AC_READY')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
