"""Check the local FAA comparison config; never launch training or TEST.

Default: stdlib source/config/data checks. --runtime adds bounded CPU-only model,
FAA-module backward, data-pipeline and optimizer-hook construction checks.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.utils.crane_frozen_pipeline_v1 import merge_config

CONFIG = 'configs/faa/crane_faa_oriented_rcnn_r50_fpn_24e_port_day2night_v1.py'
PROVENANCE = 'configs/faa/upstream_manifest.json'
EXPECTED_COUNTS = dict(train=1810, train_sim=748, val=887, test=1440)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(path, stack=()):
    """Execute trusted repository configs only; resolve their local inheritance."""
    path = Path(path).resolve()
    if (ROOT/'configs').resolve() not in path.parents:
        raise ValueError('FAA inheritance must stay inside root configs: '+str(path))
    if path in stack:
        raise ValueError('Config inheritance cycle')
    child = {k: v for k, v in runpy.run_path(str(path)).items()
             if not k.startswith('__')}
    bases = child.pop('_base_', [])
    if isinstance(bases, str):
        bases = [bases]
    merged, closure = {}, []
    for parent in bases:
        value, names = resolve(path.parent/parent, stack+(path,))
        if set(merged) & set(value):
            raise ValueError('Duplicate keys across base configs')
        merged.update(value)
        closure.extend(names)
    return merge_config(merged, child), closure+[str(path.relative_to(ROOT))]


def check_config():
    provenance = json.loads((ROOT/PROVENANCE).read_text())
    for name, spec in provenance['files'].items():
        if sha(ROOT/name) != spec['local_sha256']:
            raise ValueError('Source/config drift: '+name)
    for name, digest in provenance['references_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Comparison reference drift: '+name)
    cfg, closure = resolve(ROOT/CONFIG)
    upstream, _ = resolve(ROOT/'configs/faa/oriented_rcnn_r50_fpn_1x_dota_le90_faa.py')
    expected = deepcopy(upstream['model'])
    expected['roi_head']['bbox_head']['num_classes'] = 1
    if cfg['model'] != expected:
        raise ValueError('FAA model changed beyond the single-class output')
    assert cfg['model']['neck']['type'] == 'FAAFusionFPN'
    assert cfg['model']['roi_head']['bbox_head']['type'] == 'FAAHead'
    control = runpy.run_path(str(ROOT/'crane_project/configs/crane_eood_k1_port_day2night_aug_b_v1.py'))
    control_base = runpy.run_path(str(ROOT/'crane_project/configs/crane_eood_k1.py'))
    for index, split in enumerate(('train', 'train_sim')):
        ds = cfg['data']['train'][index]
        assert ds['pipeline'] == control['data']['train'][index]['pipeline']
        assert ds['ann_file'] == split+'/annfiles/'
        assert ds['img_prefix'] == split+'/images/'
    for split in ('val', 'test'):
        assert cfg['data'][split]['pipeline'] == control_base['test_pipeline']
        assert cfg['data'][split]['ann_file'] == split+'/annfiles/'
        assert cfg['data'][split]['img_prefix'] == split+'/images/'
    for ds in cfg['data']['train']+[cfg['data']['val'], cfg['data']['test']]:
        assert ds['type'] == 'CraneDataset' and ds['classes'] == ('grab',)
        assert ds['version'] == 'le90'
        assert ds['data_root'] == 'crane_project/data/crane_grab_port_day2night_v1/'
    assert cfg['optimizer'] == dict(type='AdamW', lr=1e-4,
                                   betas=(0.9, 0.999), weight_decay=0.05)
    assert cfg['data']['samples_per_gpu'] == 1 and cfg['gpu_ids'] == [0]
    assert cfg['optimizer_config']['type'] == 'GradientCumulativeOptimizerHook'
    assert cfg['optimizer_config']['cumulative_iters'] == 2
    assert cfg['runner']['max_epochs'] == 24 and cfg['lr_config']['step'] == [16, 22]
    assert cfg['lr_config']['warmup_iters'] == 1000
    assert cfg['evaluation']['save_best'] is None
    assert cfg['checkpoint_config']['max_keep_ckpts']*cfg['checkpoint_config']['interval'] >= 24
    assert cfg['seed'] == 0 and cfg['load_from'] is None and cfg['resume_from'] is None
    return cfg, dict(config=CONFIG, inherited_configs=closure,
                     source_sha256={name: sha(ROOT/name) for name in
                                    list(provenance['files'])+[PROVENANCE,
                                        'crane_project/tools/check_faa_adaptation_v1.py']},
                     references_sha256=provenance['references_sha256'],
                     upstream_commit=provenance['commit'],
                     model_delta=['roi_head.bbox_head.num_classes: 15 -> 1'],
                     effective_single_gpu_batch=2,
                     val_candidate_epochs=[16, 18, 20, 22, 24])


def check_dataset(cfg):
    root = ROOT/cfg['data_root']
    manifest = json.loads((root/'manifest.json').read_text())
    records = manifest['records']
    assert dict(Counter(r['split'] for r in records)) == EXPECTED_COUNTS
    image_hash_splits = {}
    counts = {}
    for split, expected in EXPECTED_COUNTS.items():
        selected = [r for r in records if r['split'] == split]
        ann = {p.stem: p for p in (root/split/'annfiles').glob('*.txt')
               if not p.name.startswith('._')}
        images = {p.stem for p in (root/split/'images').glob('*.jpg')
                  if not p.name.startswith('._')}
        assert set(ann) == images == {r['id'] for r in selected}
        assert len(ann) == expected
        for row in selected:
            path = root/row['annfiles']['path']
            assert sha(path) == row['annfiles']['sha256'], str(path)
            lines = [s.split() for s in path.read_text().splitlines() if s.strip()]
            assert len(lines) == 1, str(path)
            assert len(lines[0]) >= 9 and lines[0][8] == 'grab', str(path)
            assert all(math.isfinite(float(x)) for x in lines[0][:8]), str(path)
            image_hash_splits.setdefault(row['images']['sha256'], set()).add(split)
        counts[split] = len(ann)
    assert all(len(splits) == 1 for splits in image_hash_splits.values())
    return dict(counts=counts, annotations_match_manifest=True,
                image_names_match_manifest=True, image_bytes_rehashed=False,
                manifest_image_hash_cross_split_duplicates=0)


def runtime_check(cfg):
    # Set before importing Torch. This stage is deliberately CPU-only.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    torch.set_num_threads(2)
    from mmcv import Config
    from mmcv.runner.hooks import HOOKS
    from mmrotate.models import build_detector
    from mmrotate.datasets import build_dataset
    from mmrotate.utils import compat_cfg
    actual = Config.fromfile(str(ROOT/CONFIG))
    assert actual.to_dict() == cfg
    loader_cfg = compat_cfg(actual)
    assert loader_cfg.data.train_dataloader.samples_per_gpu == 1
    hook = HOOKS.build(actual.optimizer_config)
    assert hook.cumulative_iters == 2
    # Building does not download/load ImageNet weights or call init_weights.
    model = build_detector(actual.model)
    assert model.roi_head.bbox_head.num_classes == 1
    torch.manual_seed(0)
    features = [torch.randn(1, c, s, s, requires_grad=True)
                for c, s in zip((256, 512, 1024, 2048), (8, 4, 2, 1))]
    outputs = model.neck(features)
    assert [tuple(x.shape) for x in outputs] == [
        (1, 256, 8, 8), (1, 256, 4, 4), (1, 256, 2, 2),
        (1, 256, 1, 1), (1, 256, 1, 1)]
    sum(x.square().mean() for x in outputs).backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in features)
    roi = torch.randn(2, 256, 7, 7, requires_grad=True)
    cls, bbox = model.roi_head.bbox_head(roi)
    assert tuple(cls.shape) == (2, 2) and tuple(bbox.shape) == (2, 5)
    (cls.square().mean()+bbox.square().mean()).backward()
    assert torch.isfinite(roi.grad).all()
    pipelines = []
    for split, spec in [('train_real', actual.data.train[0]),
                        ('train_sim', actual.data.train[1]), ('val', actual.data.val)]:
        ds = build_dataset(spec)
        item = ds[0]
        img = item['img'][0].data if split == 'val' else item['img'].data
        assert tuple(img.shape) == (3, 1024, 1024)
        assert ds.CLASSES == ('grab',)
        pipelines.append(dict(split=split, dataset_frames=len(ds),
                              sample_image=ds.data_infos[0]['filename'],
                              image_shape=list(img.shape)))
    return dict(torch=torch.__version__, mmcv_config_matches=True,
                model_parameters=sum(p.numel() for p in model.parameters()),
                effective_loader_batch=loader_cfg.data.train_dataloader.samples_per_gpu,
                optimizer_hook=type(hook).__name__,
                faa_neck_backward_finite=True, faa_head_backward_finite=True,
                pipelines=pipelines, initialization_weights_loaded=False,
                cuda_used=False, full_detector_forward=False,
                optimizer_steps=0, test_images_forwarded=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', action='store_true', help='Add bounded CPU-only checks')
    parser.add_argument('--out', type=Path, help='New JSON receipt; existing files are rejected')
    args = parser.parse_args()
    os.chdir(ROOT)
    cfg, report = check_config()
    report['dataset'] = check_dataset(cfg)
    if args.runtime:
        report['runtime'] = runtime_check(cfg)
    report.update(passed=True, training_launched=False, test_inference=False,
                  gpu_memory_checked=False)
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if args.out:
        with args.out.open('x') as stream:
            stream.write(text)
    print(text, end='')


if __name__ == '__main__':
    main()
