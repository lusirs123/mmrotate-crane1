"""Validate standard Oriented R-CNN task adaptation without GPU inference."""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools.check_faa_adaptation_v1 import resolve
from crane_project.tools.prepare_external_obb_v1 import sha, read

CONFIG = 'configs/oriented_rcnn/crane_oriented_rcnn_r50_fpn_12e_port_day2night_v1.py'
NATIVE = 'configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90.py'
PROVENANCE = 'configs/oriented_rcnn/upstream_comparison_manifest_v1.json'
EPOCHS = [8, 9, 10, 11, 12]


def check_config():
    cfg, closure = resolve(ROOT / CONFIG)
    native, native_closure = resolve(ROOT / NATIVE)
    expected = deepcopy(native['model'])
    expected['roi_head']['bbox_head']['num_classes'] = 1
    if cfg['model'] != expected:
        raise ValueError('Model changed beyond single-class output')
    train = deepcopy(native['train_pipeline'])
    train[5] = dict(type='Pad', size=(1024, 1024))
    test = deepcopy(native['test_pipeline'])
    test[1]['transforms'][2] = dict(type='Pad', size=(1024, 1024))
    if cfg['train_pipeline'] != train or cfg['test_pipeline'] != test:
        raise ValueError('Native augmentation/preprocessing changed beyond fixed canvas')
    for split, ds in zip(('train', 'train_sim', 'val', 'test'),
                        cfg['data']['train'] + [cfg['data']['val'], cfg['data']['test']]):
        expected_ds = dict(type='CraneDataset', classes=('grab',),
            data_root='crane_project/data/crane_grab_port_day2night_v1/',
            ann_file=split+'/annfiles/', img_prefix=split+'/images/',
            pipeline=train if split.startswith('train') else test, version='le90')
        if ds != expected_ds:
            raise ValueError('Unexpected dataset adaptation: '+split)
    if cfg['custom_imports']['imports'] != [
            'mmrotate.datasets.crane_custom_dota',
            'crane_project.utils.external_training_observer_v1']:
        raise ValueError('Unexpected task/custom model import')
    if cfg['optimizer'] != dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001):
        raise ValueError('Optimizer differs from the author recipe')
    if cfg['optimizer_config'] != native['optimizer_config']:
        raise ValueError('Native optimizer hook changed')
    expected_lr = deepcopy(native['lr_config']); expected_lr['warmup_ratio'] = 0.001
    if cfg['lr_config'] != expected_lr or cfg['runner']['max_epochs'] != 12:
        raise ValueError('Schedule differs from the author 12-epoch recipe')
    if cfg['data']['samples_per_gpu'] != 1 or cfg['data']['workers_per_gpu'] != 2:
        raise ValueError('Unexpected batch or loader workers')
    if cfg['seed'] != 0 or cfg['load_from'] is not None or cfg['resume_from'] is not None:
        raise ValueError('Fresh seed-0 ImageNet training required')
    if cfg['evaluation']['save_best'] is not None or cfg['checkpoint_config'] != dict(interval=1, max_keep_ckpts=12):
        raise ValueError('Candidate retention or online selection changed')
    return cfg, sorted(set(closure + native_closure))


def check_sources():
    provenance = read(ROOT / PROVENANCE)
    for path, spec in provenance['files'].items():
        if sha(ROOT/path) != spec['sha256'] or spec['sha256'] != spec['upstream_sha256']:
            raise ValueError('Native upstream source drift: '+path)
    return provenance


def source_identity():
    _, closure = check_config()
    provenance = check_sources()
    names = set(closure) | set(provenance['files'])
    names.update(str(p.relative_to(ROOT)) for p in (ROOT/'mmrotate').rglob('*.py'))
    names.update([PROVENANCE, 'tools/train.py',
        'crane_project/tools/check_oriented_rcnn_comparison_v1.py',
        'crane_project/tools/run_oriented_rcnn_comparison_v1.py',
        'crane_project/tools/check_faa_adaptation_v1.py',
        'crane_project/tools/prepare_external_obb_v1.py',
        'crane_project/tools/smoke_faa_gpu_v1.py',
        'crane_project/tools/ckpt_sweep.py',
        'crane_project/utils/crane_frozen_pipeline_v1.py',
        'crane_project/utils/external_training_observer_v1.py',
        'crane_project/utils/external_training_monitor_io_v1.py'])
    return {name: sha(ROOT/name) for name in sorted(names)}


if __name__ == '__main__':
    cfg, closure = check_config()
    provenance = check_sources()
    print(json.dumps(dict(passed=True, config=CONFIG, model='OrientedRCNN',
        native_tag=provenance['tag'], native_files=len(provenance['files']),
        inherited_configs=closure, own_scale_augmentation=False,
        global_batch=2, optimizer=cfg['optimizer'], candidate_epochs=EPOCHS), indent=2))
