"""CPU configuration preparation for one E+B run; no control retrain or launch."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pprint
import runpy
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.utils.eood_B_regression_protocol_v1 import candidate_model,check_model_delta

REFERENCE='work_dirs/crane_eood_k1_port_day2night_aug_b_v1'
CFG=REFERENCE+'/crane_eood_k1_port_day2night_aug_b_v1.py'
SOURCES=(
    'mmrotate/models/losses/eood_sym_kld_positive_loss_v1.py',
    'mmrotate/models/losses/sym_kld_loss.py',
    'mmrotate/models/losses/sym_kld_calculator.py',
    'crane_project/utils/eood_B_regression_protocol_v1.py',
    'crane_project/tools/prepare_eood_B_regression_v1.py',
    'crane_project/tools/check_eood_B_regression_cpu_v1.py',
    'tests/test_eood_B_regression_v1.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    with path.open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--distributed-backend',choices=('nccl','gloo'),default='nccl')
    args=parser.parse_args()
    index=json.loads((ROOT/'docs/detection/证据索引/SymEOOD历史训练文件索引_20261021.json').read_text())
    assert sha(ROOT/CFG)==index['captured_files'][CFG]['sha256']
    sweep=json.loads((ROOT/REFERENCE/'val_sweep_port_v1/sweep_results.json').read_text())
    assert sweep['selected_checkpoint']=='epoch_24'
    old={k:v for k,v in runpy.run_path(str(ROOT/CFG)).items() if not k.startswith('__')}
    cfg=deepcopy(old);cfg['model']=candidate_model(old['model'])
    check_model_delta(old['model'],cfg['model'])
    cfg['custom_imports']['imports'].append('mmrotate.models.losses.eood_sym_kld_positive_loss_v1')
    relative=args.out.resolve().relative_to(ROOT)
    cfg['work_dir']=str(relative/'train');cfg['seed']=0
    for key in old:
        if key not in ('model','custom_imports','work_dir'):assert cfg[key]==old[key],key
    assert cfg['load_from'] is None and cfg['resume_from'] is None
    ea=ROOT/'work_dirs/eood_A_geometry_20261030_v1/completed_VAL_downloaded_result/train/initial_rank0.json'
    initial=json.loads(ea.read_text())
    val_summary=ROOT/'work_dirs/symeood_b2_20261030_v1/downloaded_result/source_audit_v2/summary.json'
    baseline=next(r for r in json.loads(val_summary.read_text())['metrics'] if r['side']=='E' and r['split']=='val' and r['domain']=='all')
    core={}
    revision=index['historical_sources']['a074048']['full_commit']
    for name in ('mmrotate/models/dense_heads/eood_head.py','mmrotate/models/dense_heads/rotated_eood_head.py',
                 'mmrotate/models/dense_heads/rotated_atss_head.py','mmrotate/models/dense_heads/rotated_anchor_head.py',
                 'mmrotate/models/detectors/eood.py','mmrotate/core/bbox/assigners/pola.py',
                 'mmrotate/models/losses/rotated_iou_loss.py','mmrotate/core/hooks/set_epoch_info_hook.py'):
        blob=subprocess.check_output(['git','-C',str(ROOT),'show',revision+':'+name])
        assert hashlib.sha256(blob).hexdigest()==sha(ROOT/name),name
        core[name]=sha(ROOT/name)
    args.out.mkdir(parents=True,exist_ok=False)
    config=args.out/'candidate_config.py'
    with config.open('x') as f:
        f.write('# E+B only. Original POLA and auxiliary supervision retained.\n')
        for key,value in cfg.items():f.write(key+' = '+pprint.pformat(value,width=100,sort_dicts=False)+'\n')
    protocol=dict(protocol='eood_B_regression_only_fixed_old_baseline_v1',
        planned_new_training_runs=1,control_retrain=False,arm='E_plus_B_regression',
        baseline=dict(reference=REFERENCE,checkpoint='epoch_24',
            checkpoint_sha256=index['runs']['crane_eood_k1_port_day2night_aug_b_v1']['selected_sha_actual'],
            expanded_config_sha256=sha(ROOT/CFG),baseline_VAL=baseline),
        model_change='Only effective main decoded-box regression: original rotated DIoU -> original compressed symmetric KLD, positive boxes only',
        log_key_note='loss_0_loss_iou is retained for native interface compatibility; it contains B SymKLD, not IoU in this run. The unused original L1 return slot remains excluded.',
        A=False,B=True,B2=False,original_POLA=True,regression_weight=2.0,
        formula='2 * weighted_reduce(min(sqrt(1+d_sym)-1,10), original_positive_normalizer)',
        seed=0,required_train_CLI_seed=0,deterministic=False,
        strict_initialization_pair=False,old_preupdate_initial_checkpoint_available=False,
        expected_new_reference_initial_state_sha256=initial['state_sha256'],
        initialization_note='Verify new E+B initialization against saved pre-update E+A state; this does not certify the old historical EOOD initialization.',
        world_size=2,per_rank_batch=2,epochs=24,candidate_epochs=[16,18,20,22,24],
        distributed_backend=args.distributed_backend,
        execution_override=(None if args.distributed_backend=='nccl' else
            'CLI dist_params.backend=gloo only: NCCL/NVML blocked by unrelated GPU2 Xid79; same two ranks, batch, model, optimizer and data. Reduction arithmetic may differ from E+A NCCL.'),
        checkpoint_selection=dict(rule='Same historical EOOD selector; five fixed VAL epochs, no TEST',
            selection_config=sweep['selection_config'],selector_source='crane_project/tools/ckpt_sweep.py',
            selector_sha256=sha(ROOT/'crane_project/tools/ckpt_sweep.py'),
            separate_detection_gate='Evaluate only original winner: AP75 gain>=1pp over frozen E, Recall/center/mean RIoU protected; no reselection to pass.'),
        config=str(relative/config.name),config_sha256=sha(config),
        sources=dict(core,**{name:sha(ROOT/name) for name in SOURCES}),
        resource_hold=False,formal_training_started=False,optimizer_updates=0,
        CPU_preparation_only=True,native_CPU_tests_pending=True,GPU_gate_pending=True,
        GPU_training_resources=dict(physical_gpus=[0,1],allocator_limit_mib=6144,
            minimum_free_mib=6656,no_launch_while_hold=False,on_OOM='Stop, no automatic retry or batch change'),
        TEST=False,downstream=False,automatic_adoption=False,retry=False)
    save(args.out/'protocol.json',protocol)
    print(json.dumps(dict(prepared=True,planned_new_training_runs=1,control_retrain=False,GPU_used=False)))


if __name__=='__main__':main()
