"""CPU-only preparation of B2R0/B2R4; never launches a gate or training."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
REFERENCE = ROOT / 'work_dirs/symeood_110_retrain_seed0_20261026_v1'
BASE = ROOT / 'work_dirs/symeood_b2_20261030_v1'
SOURCES = (
    'mmrotate/models/losses/sym_iou_hybrid_loss_v1.py',
    'mmrotate/models/dense_heads/sym_eood_b2_head_v1.py',
    'crane_project/utils/symeood_b2_protocol_v1.py',
    'crane_project/utils/symeood_tower_trace_v1.py',
    'crane_project/tools/prepare_symeood_b2_control_v1.py',
    'tests/test_symeood_b2_v1.py',
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def prepare():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU preparation only; GPU work is held'
    import torch
    assert not torch.cuda.is_available()
    from mmcv import Config
    from mmcv.utils import import_modules_from_strings
    from mmrotate.models import build_detector
    from crane_project.utils.symeood_b2_protocol_v1 import ARMS, GEOMETRY_WEIGHT, candidate_model, check_model_delta
    from crane_project.utils.symeood_repro_trace_v1 import tensor_digest

    previous = read(REFERENCE / 'protocol.json')
    for relative, digest in previous['sources'].items():
        assert sha(ROOT / relative) == digest, relative
    assert sha(REFERENCE / 'configs/Coff_seed0.py') == previous['config_sha256']
    r4 = read(ROOT / 'work_dirs/symeood_110_reg_tower_20261030_v1/protocol.json')
    initial_path = REFERENCE / '110/complete_initial_model.pth'
    assert sha(initial_path) == r4['reference_initial_sha256']
    tower_path = ROOT / 'work_dirs/symeood_110_tower_control_20261029_v2/initial_T4.pth'
    assert sha(tower_path) == read(tower_path.parent / 'protocol.json')['initial_sha256']
    initial = torch.load(str(initial_path), map_location='cpu')['state_dict']
    tower_initial = torch.load(str(tower_path), map_location='cpu')['state_dict']
    audit = read(BASE / 'source_audit_v2/summary.json')
    assert audit['passed'] and audit['optimizer_updates'] == audit['model_forwards'] == 0
    hold = read(BASE / 'resource_hold.json')
    assert hold['formal_training_runs'] == 0
    new_sources = dict(previous['sources'])
    new_sources.update({name: sha(ROOT / name) for name in SOURCES})
    prepared = []
    for arm, layers in ARMS.items():
        destination = BASE / arm
        destination.mkdir(exist_ok=False)
        cfg = Config.fromfile(str(REFERENCE / 'configs/Coff_seed0.py'))
        old = deepcopy(cfg.to_dict())
        cfg.model = candidate_model(old['model'], arm)
        check_model_delta(old['model'], cfg.model.to_dict(), arm)
        cfg.custom_imports.imports += [
            'mmrotate.models.losses.sym_iou_hybrid_loss_v1',
            'mmrotate.models.dense_heads.sym_eood_b2_head_v1',
            'crane_project.utils.symeood_tower_trace_v1']
        import_modules_from_strings(**cfg.custom_imports)
        model_config = deepcopy(cfg.model)
        model_config.backbone.init_cfg = None
        model = build_detector(model_config)
        state = model.state_dict()
        extra = set(state) - set(initial)
        assert not (set(initial) - set(state))
        assert len(extra) == (8 if layers else 0)
        assert all(name.startswith('bbox_head.reg_convs.') for name in extra)
        for name in state:
            source = initial if name in initial else tower_initial
            assert state[name].shape == source[name].shape
            state[name] = source[name].clone()
        model.load_state_dict(state, strict=True)
        assert all(torch.equal(model.state_dict()[name], value) for name, value in initial.items())
        count = sum(p.numel() for p in model.parameters())
        assert count == 31560792 + (2360320 if layers else 0)
        init = destination / 'initial.pth'
        torch.save(dict(state_dict=state, meta=dict(purpose='before-update matched initialization', arm=arm)), str(init))
        cfg.work_dir = str(destination / 'train')
        cfg.load_from = str(init)
        assert cfg.resume_from is None
        hooks = deepcopy(cfg.custom_hooks)
        assert len(hooks) == 1
        hooks[0].update(type='SymEOODTowerTraceV1', reference_work=str(REFERENCE / '110'),
                        runtime_lock=str(destination / 'runtime_lock.json'))
        cfg.custom_hooks = hooks
        allowed = {'model','custom_imports','work_dir','load_from','custom_hooks'}
        for name in set(old) | set(cfg.to_dict()):
            if name not in allowed:
                assert old.get(name) == cfg.to_dict().get(name), name
        config = destination / 'config.py'
        cfg.dump(str(config))
        lock = read(REFERENCE / 'runtime_identity_gpus01.json')
        lock['physical_gpus'] = [0, 1]
        lock['runtime']['environment']['CUDA_VISIBLE_DEVICES'] = '0,1'
        lock['source_sha256'].update({str(ROOT / name): digest for name, digest in new_sources.items()})
        save(destination / 'runtime_lock.json', lock)
        save(destination / 'protocol.json', dict(protocol='symeood_b2_loss_and_regression_tower_control_v1',
            arm=arm, reg_tower_layers=layers, classification_tower_layers=0,
            geometry_weight=GEOMETRY_WEIGHT, overlap_weight=2.0, effective_geometry_weight=2.0 * GEOMETRY_WEIGHT,
            B_retained=True, epochs=24, candidate_epochs=[16,18,20,22,24], seed=0,
            world_size=2, per_rank_batch=2, physical_gpus=[0,1], allocator_limit_mib=6144,
            config=str(config), config_sha256=sha(config),
            initial_checkpoint=str(init), initial_sha256=sha(init), initial_state_sha256=tensor_digest(model.state_dict().items()),
            reference_initial_sha256=sha(initial_path), regression_initial_sha256=sha(tower_path),
            all_shared_initial_tensors_exact=True, parameter_count=count,
            sources=new_sources, runtime_lock_sha256=sha(destination / 'runtime_lock.json'),
            launch_environment=dict(previous['launch_environment'], CUDA_VISIBLE_DEVICES='0,1'),
            detection_reference=next(r for r in audit['metrics'] if r['side']=='E' and r['split']=='val' and r['domain']=='all'),
            selection='Complete VAL only: AP75 gain>=1pp over frozen E; Recall/center/mean RIoU protected. Sizes never select. No eligible checkpoint => null.',
            source_audit_sha256=sha(BASE / 'source_audit_v2/summary.json'),
            formal_training_started=False, GPU_gate_pending=True, resource_hold=True,
            training_updates=0, TEST=False, automatic_adoption=False, retry=False))
        prepared.append(dict(arm=arm, config_sha256=sha(config), parameter_count=count,
            added_state_keys=len(extra), B_retained=True, regression_layers=layers))
        del model, state
    save(BASE / 'CPU_prepare_receipt.json', dict(passed=True, arms=prepared,
        CPU_only=True, GPU_initialization=False, training_updates=0,
        formal_training_started=False, resource_hold=True,
        reason='User requests avoiding additional GPU memory pressure; wait for a new resource check before native gate/training.'))
    print(json.dumps(dict(CPU_PREPARE_PASS=True, arms=prepared, TRAINING_STARTED=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    prepare()
