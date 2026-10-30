"""Prepare one E+A configuration using stdlib only; never launches anything."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pprint
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.utils.eood_A_geometry_protocol_v1 import candidate_model, check_model_delta

REFERENCE = 'work_dirs/crane_eood_k1_port_day2night_aug_b_v1'
CFG = REFERENCE + '/crane_eood_k1_port_day2night_aug_b_v1.py'
SOURCES = (
    'mmrotate/core/bbox/assigners/pola_kld_geometry_v1.py',
    'crane_project/utils/eood_A_geometry_protocol_v1.py',
    'crane_project/tools/prepare_eood_A_geometry_v1.py',
    'crane_project/tools/check_eood_A_geometry_cpu_v1.py',
    'tests/test_eood_A_geometry_v1.py',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    index = json.loads((ROOT / 'docs/detection/证据索引/SymEOOD历史训练文件索引_20261021.json').read_text())
    assert sha(ROOT / CFG) == index['captured_files'][CFG]['sha256']
    sweep = json.loads((ROOT / REFERENCE / 'val_sweep_port_v1/sweep_results.json').read_text())
    assert sweep['selected_checkpoint'] == 'epoch_24'
    source_cfg = ROOT / 'crane_project/configs/crane_eood_k1_port_day2night_aug_b_v1.py'
    assert sha(source_cfg) == sweep['config_sha256']
    reference = {k: v for k, v in runpy.run_path(str(ROOT / CFG)).items() if not k.startswith('__')}
    cfg = deepcopy(reference)
    cfg['model'] = candidate_model(reference['model'])
    check_model_delta(reference['model'], cfg['model'])
    cfg['custom_imports']['imports'].append('mmrotate.core.bbox.assigners.pola_kld_geometry_v1')
    relative_out = args.out.resolve().relative_to(ROOT)
    cfg['work_dir'] = str(relative_out / 'train')
    # tools/train.py still requires --seed 0; the protocol locks that CLI arg.
    cfg['seed'] = 0
    for key in reference:
        if key not in ('model', 'custom_imports', 'work_dir'):
            assert cfg[key] == reference[key], key
    assert cfg['load_from'] is None and cfg['resume_from'] is None
    old_log = (ROOT / REFERENCE / '20260930_083040.log').read_text()
    assert 'Set random seed to 0, deterministic: False' in old_log
    audit_path = ROOT / 'work_dirs/symeood_b2_20261030_v1/downloaded_result/source_audit_v2/summary.json'
    audit = json.loads(audit_path.read_text())
    baseline = next(row for row in audit['metrics'] if row['side'] == 'E'
                    and row['split'] == 'val' and row['domain'] == 'all')
    assert baseline['frames'] == 887
    revision = index['historical_sources']['a074048']['full_commit']
    core_sources = {}
    for relative in (
        'mmrotate/models/dense_heads/eood_head.py',
        'mmrotate/models/dense_heads/rotated_eood_head.py',
        'mmrotate/models/dense_heads/rotated_atss_head.py',
        'mmrotate/models/dense_heads/rotated_anchor_head.py',
        'mmrotate/models/detectors/eood.py',
        'mmrotate/core/bbox/assigners/pola.py',
        'mmrotate/models/losses/rotated_iou_loss.py',
        'mmrotate/core/hooks/set_epoch_info_hook.py',
    ):
        old = subprocess.check_output(['git', '-C', str(ROOT), 'show', revision + ':' + relative])
        assert hashlib.sha256(old).hexdigest() == sha(ROOT / relative), relative
        core_sources[relative] = sha(ROOT / relative)
    args.out.mkdir(parents=True, exist_ok=False)
    config = args.out / 'candidate_config.py'
    with config.open('x') as stream:
        stream.write('# E+A geometry only; GPU/training held. Use --seed 0 when later authorized.\n')
        for key, value in cfg.items():
            stream.write(key + ' = ' + pprint.pformat(value, width=100, sort_dicts=False) + '\n')
    protocol = dict(protocol='eood_A_geometry_only_fixed_old_baseline_v1',
        planned_new_training_runs=1, control_retrain=False, arm='E_plus_A_geometry',
        baseline=dict(reference=REFERENCE, checkpoint='epoch_24',
            checkpoint_sha256=index['runs']['crane_eood_k1_port_day2night_aug_b_v1']['selected_sha_actual'],
            expanded_config_sha256=sha(ROOT / CFG), source_config_sha256=sha(source_cfg),
            baseline_VAL=baseline, result_audit_sha256=sha(audit_path)),
        model_change='Only main POLA geometry: 1-RIoU+center penalty -> existing raw symmetric KLD',
        geometry_coefficient=1.0, classification_cost_coefficient=1.0,
        temperature=False, progressive_positive_schedule=False,
        first_four_epochs='Original MaxIoU retained (epoch indices 0..3)',
        regression_B='Unchanged original rotated DIoU; no SymKLD regression loss or B2',
        seed=0, required_train_CLI_seed=0, deterministic=False,
        strict_initialization_pair=False, old_preupdate_initial_checkpoint_available=False,
        initialization_note='Old seed0 is verified; no certified old complete pre-update state. New training is exploratory versus a fixed historical baseline, not an exact paired initialization.',
        world_size=2, per_rank_batch=cfg['data']['samples_per_gpu'],
        epochs=24, candidate_epochs=[16,18,20,22,24],
        checkpoint_selection=dict(rule='Same historical EOOD selector, unchanged; only VAL epochs16/18/20/22/24',
            selection_config=sweep['selection_config'],
            selector_source='crane_project/tools/ckpt_sweep.py',
            selector_sha256=sha(ROOT / 'crane_project/tools/ckpt_sweep.py'),
            separate_detection_gate='Evaluate only original selector winner; AP75 gain>=1pp and Recall/center/mean RIoU protected vs frozen E; do not reselect for this gate.'),
        config=str(relative_out / config.name), config_sha256=sha(config),
        sources=dict(core_sources, **{name: sha(ROOT / name) for name in SOURCES}),
        historical_core_revision=revision,
        old_runtime_limit='Historical EOOD full uncommitted source tree not certified.',
        resource_hold=True, formal_training_started=False, optimizer_updates=0,
        CPU_preparation_only=True, native_CPU_tests_pending=True, GPU_gate_pending=True,
        GPU_training_resources=dict(physical_gpus=[0,1], allocator_limit_mib=6144,
            minimum_free_mib=6656, no_launch_while_hold=True,
            limit_scope='Planned allocator cap; not a total driver memory guarantee. Native GPU gate still pending.',
            on_OOM='Stop, no automatic retry or batch change'),
        TEST=False, downstream=False, automatic_launch=False, automatic_adoption=False)
    save(args.out / 'protocol.json', protocol)
    save(args.out / 'resource_hold.json', dict(state='TRAINING_NOT_STARTED_RESOURCE_HOLD',
        control_retrain=False, planned_new_training_runs=1, optimizer_updates=0,
        reason='User requests avoiding GPU memory pressure; CPU preparation only; no automatic launch.',
        GPU_gate_pending=True, TEST=False))
    print(json.dumps(dict(prepared=True, config=str(config), new_training_runs=0,
                         planned_new_training_runs=1, control_retrain=False, GPU_used=False)))


if __name__ == '__main__':
    main()
