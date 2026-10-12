#!/usr/bin/env python3
"""Fresh auxiliary A only, fixed 7000-update prevention window. No selection."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import run_port_reliability_oof_v1 as old
from crane_project.utils import port_reliability_oof_prevention_v1 as contract

FILES = ['crane_project/tools/run_port_reliability_oof_prevention_v1.py',
         'crane_project/tools/run_port_reliability_oof_prevention_v1.sh',
         'crane_project/utils/port_reliability_oof_prevention_v1.py',
         'mmrotate/core/hooks/port_oof_prevention_hook.py',
         'tests/test_port_reliability_oof_prevention_v1.py',
         'mmrotate/core/hooks/port_oof_regression_budget_hook.py',
         'mmrotate/core/hooks/port_oof_trace_hook.py',
         'crane_project/utils/port_reliability_oof_gradient_chain_v1.py',
         'crane_project/utils/port_reliability_oof_followup_v1.py',
         'crane_project/utils/port_reliability_oof_training_v2.py']


def prepare(out):
    from mmcv import Config
    rows, policy, plan, source, proof = old.checked()
    out.mkdir(parents=True, exist_ok=False)
    specification = plan['A']
    folder = out/'A'; folder.mkdir()
    data = folder/'data'
    for split in ('train', 'train_sim'):
        for field in ('images', 'annfiles'):
            (data/split/field).mkdir(parents=True)
    for name in specification['train_images']:
        item = source[name]
        for field in ('images', 'annfiles'):
            src = old.DATA/item[field]['path']
            (data/item['split']/field/src.name).symlink_to(src)
    cfg = Config.fromfile(str(old.CONFIG))
    old.training_repair.repair_config(cfg)
    for spec, split in zip(cfg.data.train, ('train', 'train_sim')):
        spec.data_root = str(data.resolve())+'/'
        spec.ann_file = split+'/annfiles/'
        spec.img_prefix = split+'/images/'
    cfg.data.val = dict(); cfg.data.test = dict()
    cfg.data.samples_per_gpu = old.core.SETTINGS['detector_batch']
    cfg.data.workers_per_gpu = 2
    cfg.data.pop('train_dataloader', None)
    cfg.workflow = [('train', 1)]
    cfg.runner.max_epochs = old.core.SETTINGS['detector_epochs']
    cfg.load_from = cfg.resume_from = None; cfg.auto_resume = False
    cfg.evaluation = None
    cfg.seed = contract.SETTINGS['seed']
    cfg.log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
    cfg.work_dir = str((folder/'detector').resolve())
    contract.configure(cfg, folder/'prevention')
    cfg.dump(str(folder/'detector_config.py'))
    old.write(folder/'membership.json', dict(specification,
        resolved_config_sha256=old.sha(folder/'detector_config.py')))
    old.write(out/'prepare.json', dict(proof=proof, plan=plan))
    old.write(out/'prevention.json', dict(protocol=contract.VERSION, settings=contract.SETTINGS,
        allocation_revision=contract.ALLOCATION_REVISION,
        memory_revision=contract.MEMORY_REVISION, memory_settings=contract.MEMORY_SETTINGS,
        sources={p:old.sha(ROOT/p) for p in FILES}, configuration_sha256=old.sha(folder/'detector_config.py'),
        initialization='fresh_ImageNet_only', inherited_optimizer_momentum=False,
        fold='A', TRAIN_images=len(specification['train_images']),
        VAL_scored=False, TEST_read=False, auxiliary_B_started=False,
        deployed_auxiliary_models=0, diagnostic_weights_usable_as_OOF=False))


def check(out):
    from mmcv import Config
    saved = json.loads((out/'prevention.json').read_text())
    if (saved['protocol'] != contract.VERSION or saved['settings'] != contract.SETTINGS or
            saved['allocation_revision'] != contract.ALLOCATION_REVISION or
            saved['memory_revision'] != contract.MEMORY_REVISION or saved['memory_settings'] != contract.MEMORY_SETTINGS):
        raise ValueError('Prevention contract changed')
    for path, pin in saved['sources'].items():
        if old.sha(ROOT/path) != pin:
            raise ValueError('Prevention source changed: '+path)
    if old.sha(out/'A/detector_config.py') != saved['configuration_sha256']:
        raise ValueError('Prevention config changed')
    old.check_fold(out, 'A')
    contract.check_config(Config.fromfile(str(out/'A/detector_config.py')))
    return saved


def run(out):
    check(out)
    proc = subprocess.run([sys.executable, str(ROOT/'tools/train.py'), str(out/'A/detector_config.py'),
                           '--no-validate', '--seed', '1701', '--deterministic'], cwd=str(ROOT))
    path = out/'A/prevention/completion.json'
    if not path.is_file():
        raise RuntimeError('Unexpected trainer failure; no bounded completion: %d'%proc.returncode)
    result = json.loads(path.read_text())
    allowed = ('PREVENTION_WINDOW_PASS', 'ANOMALY_QUARANTINED_BEFORE_OPTIMIZER',
               'FORWARD_FAILURE_STOP', 'UPDATE_OR_RESOURCE_FAILURE_STOP')
    if proc.returncode == 0 or result['status'] not in allowed:
        raise ValueError('Trainer did not terminate at registered boundary')
    if result['status'] == 'PREVENTION_WINDOW_PASS' and result['completed_updates'] != 7000:
        raise ValueError('Incomplete window cannot pass')
    check(out)
    old.write(out/'completion.json', dict(result, trainer_exit=proc.returncode,
        expected_bounded_stop=True, automatic_continuation=False,
        reliability_performance_measured=False, unique_root_cause_proven=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('prepare', 'check', 'train'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); out = args.out.resolve()
    if ROOT/'work_dirs' not in out.parents:
        raise ValueError('Outputs must remain under this project work_dirs')
    try:
        if args.mode == 'prepare': prepare(out)
        elif args.mode == 'check': check(out)
        else: run(out)
    except Exception as exc:
        if out.exists() and not (out/'failure.json').exists():
            old.write(out/'failure.json', dict(error=repr(exc), traceback=traceback.format_exc(),
                                              VAL_scored=False, TEST_read=False))
        raise


if __name__ == '__main__':
    main()
