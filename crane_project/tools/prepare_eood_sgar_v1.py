"""Write one independent EOOD+SGAR candidate and its source-locked protocol.

Explicit CPU preparation only; no native model, GPU, optimizer or subprocess
training. This file has not been executed as part of implementation.
"""
import argparse
import hashlib
from pathlib import Path
import pprint
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.utils.eood_sgar_protocol_v1 import (
    REFERENCE, REFERENCE_CONFIG, candidate_config, check_config_delta, read, save, sha)

OWN_SOURCES = (
    'crane_project/utils/eood_sgar_protocol_v1.py',
    'crane_project/tools/prepare_eood_sgar_v1.py',
    'crane_project/tools/check_eood_sgar_cpu_v1.py',
    'crane_project/tools/run_eood_sgar_v1.py',
    'crane_project/tools/native_eood_sgar_v1.py',
    'crane_project/tools/analyze_eood_sgar_v1.py',
    'tests/test_eood_sgar_v1.py',
    'mmrotate/core/bbox/assigners/pola_kld_geometry_v1.py',
    'mmrotate/models/losses/eood_sym_kld_positive_loss_v1.py',
    'mmrotate/models/losses/sym_kld_loss.py',
    'mmrotate/models/losses/sym_kld_calculator.py',
    'crane_project/utils/symeood_repro_trace_v1.py',
    'tools/train.py', 'tools/test.py', 'mmrotate/apis/train.py',
    'crane_project/tools/ckpt_sweep.py',
    'crane_project/tools/eval_crane_offline.py',
    'crane_project/tools/eval_port_geometry_ap_v1.py',
    'crane_project/tools/analyze_symeood_ablation_first4_v1.py',
    'crane_project/utils/port_geometry_midpoint_edge_residual_metrics_v1.py',
    'crane_project/utils/port_geometry_midpoint_edge_feasibility_v1.py',
)
CORE_SOURCES = (
    'mmrotate/models/dense_heads/eood_head.py',
    'mmrotate/models/dense_heads/rotated_eood_head.py',
    'mmrotate/models/dense_heads/rotated_atss_head.py',
    'mmrotate/models/dense_heads/rotated_anchor_head.py',
    'mmrotate/models/detectors/eood.py',
    'mmrotate/core/bbox/assigners/pola.py',
    'mmrotate/models/losses/rotated_iou_loss.py',
    'mmrotate/core/hooks/set_epoch_info_hook.py',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--distributed-backend', choices=('nccl', 'gloo'), default='nccl')
    args = parser.parse_args()
    base = args.out.resolve()
    relative = base.relative_to(ROOT)
    if relative.parts[0] != 'work_dirs' or not base.name.startswith('eood_sgar_'):
        raise ValueError('Use a new work_dirs/eood_sgar_* experiment directory')
    if base.exists():
        raise FileExistsError('Existing experiment must not be overwritten: ' + str(base))
    index = read(ROOT / 'docs/detection/证据索引/SymEOOD历史训练文件索引_20261021.json')
    assert sha(ROOT / REFERENCE_CONFIG) == index['captured_files'][REFERENCE_CONFIG]['sha256']
    sweep_path = ROOT / REFERENCE / 'val_sweep_port_v1/sweep_results.json'
    sweep = read(sweep_path)
    assert sweep['selected_checkpoint'] == 'epoch_24'
    old = {k: v for k, v in runpy.run_path(str(ROOT / REFERENCE_CONFIG)).items()
           if not k.startswith('__')}
    cfg = candidate_config(old, relative / 'train')
    check_config_delta(old, cfg)
    assert cfg['runner']['max_epochs'] == 24 and cfg['data']['samples_per_gpu'] == 2
    old_log = (ROOT / REFERENCE / '20260930_083040.log').read_text()
    assert 'Set random seed to 0, deterministic: False' in old_log
    audit_path = ROOT / 'work_dirs/symeood_b2_20261030_v1/downloaded_result/source_audit_v2/summary.json'
    baseline = next(r for r in read(audit_path)['metrics'] if r['side'] == 'E'
                    and r['split'] == 'val' and r['domain'] == 'all')
    assert baseline['frames'] == 887
    revision = index['historical_sources']['a074048']['full_commit']
    for name in CORE_SOURCES:
        blob = subprocess.check_output(['git', '-C', str(ROOT), 'show', revision + ':' + name])
        assert hashlib.sha256(blob).hexdigest() == sha(ROOT / name), name
    sources = {name: sha(ROOT / name) for name in CORE_SOURCES + OWN_SOURCES}
    references = {
        'sweep': str(sweep_path.relative_to(ROOT)),
        'inventory': 'work_dirs/symeood_110_retrain_seed0_20261026_v1/protocol.json',
        'VAL_GT_rows': 'work_dirs/symeood_110_retrain_seed0_20261026_v1/downstream/lvsr/fixed_b/epoch_01_val.rows.jsonl.gz',
        'baseline_VAL_audit': str(audit_path.relative_to(ROOT)),
    }
    reference_hashes = {name: sha(ROOT / path) for name, path in references.items()}
    base.mkdir(parents=True, exist_ok=False)
    config_path = base / 'candidate_config.py'
    with config_path.open('x') as stream:
        stream.write('# EOOD-background SGAR. Both geometry slots change; all EOOD settings retained.\n')
        for key, value in cfg.items():
            stream.write(key + ' = ' + pprint.pformat(value, width=100, sort_dicts=False) + '\n')
    protocol = dict(
        protocol='eood_sgar_three_arm_v1', arm='EOOD_plus_SGAR',
        comparison_arms=['EOOD_existing', 'EOOD_plus_SGAR_new', 'R4_existing'],
        planned_new_training_runs=1, control_retrain=False, R4_Ref=False,
        baseline=dict(reference=REFERENCE, checkpoint='epoch_24',
                      checkpoint_sha256=index['runs']['crane_eood_k1_port_day2night_aug_b_v1']['selected_sha_actual'],
                      expanded_config_sha256=sha(ROOT / REFERENCE_CONFIG), baseline_VAL=baseline),
        references=references, reference_sha256=reference_hashes,
        model_change='Only original POLA geometry and effective decoded positive-box regression',
        assignment=dict(class_coefficient=1., geometry_coefficient=1., temperature=False,
                        progressive_positive_schedule=False, first_four_epochs='Original MaxIoU',
                        later_epochs='POLA with raw symmetric KLD geometry'),
        regression=dict(weight=2., positive_only=True,
                        formula='2*weighted_reduce(min(sqrt(1+d_sym)-1,10),original_positive_normalizer)',
                        log_key='loss_0_loss_iou', log_key_semantics='SymKLD, not IoU; unused L1 remains excluded'),
        seed=0, required_train_CLI_seed=0, deterministic=False,
        strict_initialization_pair=False, old_preupdate_initial_checkpoint_available=False,
        initialization_note='Historical EOOD seed0 verified; complete old pre-update state not certified. Save this arm initial state; no claim of strict historical pairing.',
        epochs=24, world_size=2, per_rank_batch=2, expected_updates_per_rank=15360,
        candidate_epochs=[16, 18, 20, 22, 24], distributed_backend=args.distributed_backend,
        execution_override=None if args.distributed_backend == 'nccl' else
            'Gloo selected explicitly; historical backend arithmetic is not claimed identical.',
        checkpoint_selection=dict(selection_config=sweep['selection_config'],
                                  center_thresh_px=sweep.get('center_thresh_px', 15.),
                                  inference_schedule='Sequential on first explicitly selected GPU',
                                  rule='Original EOOD VAL selector; preserve null selection; no new AP gain/adoption gate'),
        config=str(config_path.relative_to(ROOT)), config_sha256=sha(config_path),
        sources=sources, historical_core_revision=revision,
        old_runtime_limit='Full historical uncommitted runtime source tree not certified.',
        allocator_limit_mib=6144, own_driver_limit_mib=6656,
        minimum_free_before_launch_mib=6656, minimum_free_during_run_mib=1024,
        memory_poll_interval_seconds=5,
        CPU_preparation_only=True, native_CPU_tests_pending=True, GPU_gate_pending=True,
        formal_training_started=False, optimizer_updates=0,
        TEST=False, downstream=False, automatic_adoption=False, automatic_resume=False, retry=False)
    save(base / 'protocol.json', protocol)
    print('EOOD_SGAR_CONFIG_PREPARED_NO_TRAINING', flush=True)


if __name__ == '__main__':
    main()
