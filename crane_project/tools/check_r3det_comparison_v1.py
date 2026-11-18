"""Fail-closed R3Det recipe and provenance checks; no training or TEST."""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools.check_faa_adaptation_v1 import resolve
from crane_project.tools.prepare_external_obb_v1 import read, sha

CONFIG = 'configs/r3det/crane_r3det_r50_fpn_12e_single_gpu_v1.py'
NATIVE = 'configs/r3det/r3det_r50_fpn_1x_dota_oc.py'
PROVENANCE = 'configs/r3det/upstream_comparison_manifest_v1.json'
EPOCHS = [8, 9, 10, 11, 12]


def check_config():
    cfg, closure = resolve(ROOT/CONFIG)
    native, _ = resolve(ROOT/NATIVE)
    expected = deepcopy(native['model'])
    expected['bbox_head']['num_classes'] = 1
    expected['bbox_head']['type'] = 'R3DetNativeRetinaHeadV1'
    for head in expected['refine_heads']:
        head['num_classes'] = 1
        head['type'] = 'R3DetNativeRetinaRefineHeadV1'
    if cfg['model'] != expected:
        raise ValueError('Native model changed beyond single-class outputs')
    train, test = deepcopy(native['train_pipeline']), deepcopy(native['test_pipeline'])
    train[5] = dict(type='Pad', size=(1024, 1024))
    test[1]['transforms'][2] = dict(type='Pad', size=(1024, 1024))
    if cfg['train_pipeline'] != train or cfg['test_pipeline'] != test:
        raise ValueError('Augmentation changed beyond fixed canvas')
    for split, ds in zip(('train','train_sim','val','test'), cfg['data']['train']+[cfg['data']['val'],cfg['data']['test']]):
        expected_ds = dict(type='CraneDatasetOCV1', classes=('grab',),
            data_root='crane_project/data/crane_grab_port_day2night_v1/', version='oc',
            ann_file=split+'/annfiles/', img_prefix=split+'/images/',
            pipeline=train if split.startswith('train') else test)
        if ds != expected_ds:
            raise ValueError('Dataset or angle convention differs: '+split)
    if cfg['optimizer'] != native['optimizer']:
        raise ValueError('Native SGD parameters changed')
    hook = dict(native['optimizer_config'], type='GradientCumulativeOptimizerHook', cumulative_iters=2)
    if cfg['optimizer_config'] != hook:
        raise ValueError('Expected native gradient clipping after accumulation')
    lr = deepcopy(native['lr_config']); lr['warmup_iters'] *= 2
    if cfg['lr_config'] != lr or cfg['runner'] != native['runner']:
        raise ValueError('Native optimizer-step warmup or epoch schedule changed')
    if cfg['data']['samples_per_gpu'] != 1 or cfg['gpu_ids'] != [0] or cfg['seed'] != 0:
        raise ValueError('Single logical GPU, microbatch1, seed0 required')
    if cfg['load_from'] is not None or cfg['resume_from'] is not None:
        raise ValueError('Fresh ImageNet initialization required')
    if cfg['checkpoint_config'] != dict(interval=1,max_keep_ckpts=12) or cfg['evaluation']['save_best'] is not None:
        raise ValueError('Candidate retention or online selection differs')
    if cfg['custom_imports']['imports'] != ['crane_project.utils.r3det_native_retina_head_v1',
            'crane_project.utils.r3det_native_refine_head_v1','crane_project.utils.crane_dataset_oc_v1',
            'crane_project.utils.external_training_observer_v1']:
        raise ValueError('Unexpected model/augmentation import')
    return cfg, closure


def source_identity():
    _, closure = check_config()
    provenance = read(ROOT/PROVENANCE)
    for name, item in provenance['files'].items():
        if sha(ROOT/name) != item['sha256']:
            raise ValueError('Native R3Det source drift: '+name)
        import hashlib
        restored = (ROOT/name).read_text()
        for change in reversed(item.get('adaptations',[])):
            if restored.count(change['local'])!=1:raise ValueError('Non-unique native import alias')
            restored=restored.replace(change['local'],change['upstream'])
        if hashlib.sha256(restored.encode()).hexdigest()!=item['upstream_sha256']:
            raise ValueError('Changes beyond registry/import aliases: '+name)
    names = set(closure) | set(provenance['files'])
    names.update(str(p.relative_to(ROOT)) for p in (ROOT/'mmrotate').rglob('*.py'))
    names.update([PROVENANCE, 'tools/train.py',
        'crane_project/tools/check_r3det_comparison_v1.py',
        'crane_project/tools/run_r3det_comparison_v1.py',
        'crane_project/tools/run_oriented_rcnn_comparison_v1.py',
        'crane_project/tools/check_faa_adaptation_v1.py',
        'crane_project/tools/prepare_external_obb_v1.py',
        'crane_project/tools/smoke_faa_gpu_v1.py',
        'crane_project/tools/ckpt_sweep.py',
        'crane_project/utils/crane_dataset_oc_v1.py',
        'crane_project/utils/crane_frozen_pipeline_v1.py',
        'crane_project/utils/external_training_observer_v1.py',
        'crane_project/utils/external_training_monitor_io_v1.py'])
    return {name:sha(ROOT/name) for name in sorted(names)}


if __name__ == '__main__':
    cfg, _ = check_config()
    print(json.dumps(dict(passed=True, config=CONFIG, optimizer=cfg['optimizer'],
        microbatch=1, accumulation=2, effective_batch=2, epochs=cfg['runner']['max_epochs']),indent=2))
