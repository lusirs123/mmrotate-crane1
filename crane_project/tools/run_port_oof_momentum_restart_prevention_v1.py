#!/usr/bin/env python3
"""Independent fresh-A entry. Preparing it never automatically starts training."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import run_port_oof_task_budget_prevention_v1 as parent
from crane_project.utils import port_oof_momentum_restart_v1 as repair

FILES = ['crane_project/utils/port_oof_momentum_restart_v1.py',
         'mmrotate/core/hooks/port_oof_momentum_restart_hook.py',
         'crane_project/tools/run_port_oof_momentum_restart_prevention_v1.py',
         'tests/test_port_oof_momentum_restart_v1.py']


def prepare(out):
    from mmcv import Config
    parent.prepare(out)
    path = out/'A/detector_config.py'
    cfg = Config.fromfile(str(path))
    repair.configure(cfg)
    cfg.dump(str(path))
    pin = parent.parent.old.sha(path)
    member_path = out/'A/membership.json'
    member = json.loads(member_path.read_text())
    member['resolved_config_sha256'] = pin
    member_path.write_text(json.dumps(member, indent=2)+'\n')
    report = json.loads((out/'prevention.json').read_text())
    report.update(momentum_restart_revision=repair.VERSION, configuration_sha256=pin)
    report['sources'].update({p:parent.parent.old.sha(ROOT/p) for p in FILES})
    (out/'prevention.json').write_text(json.dumps(report, indent=2)+'\n')
    check(out)


def check(out):
    from mmcv import Config
    saved = json.loads((out/'prevention.json').read_text())
    c = parent.parent.contract
    task = parent.task
    if (saved.get('momentum_restart_revision') != repair.VERSION or
            saved.get('task_budget_revision') != task.VERSION or
            saved.get('task_memory_revision') != task.MEMORY_REVISION or
            saved.get('raw_gradient_measurement_enabled') is not False or
            saved.get('task_budgets') != {k:task.BUDGET for k in task.TASKS} or
            saved['protocol'] != c.VERSION or saved['settings'] != c.SETTINGS or
            saved['allocation_revision'] != c.ALLOCATION_REVISION or
            saved['memory_revision'] != c.MEMORY_REVISION or saved['memory_settings'] != c.MEMORY_SETTINGS):
        raise ValueError('Independent momentum-restart contract changed')
    for p, pin in saved['sources'].items():
        if parent.parent.old.sha(ROOT/p) != pin:
            raise ValueError('Source changed: '+p)
    if parent.parent.old.sha(out/'A/detector_config.py') != saved['configuration_sha256']:
        raise ValueError('Config changed')
    parent.parent.old.check_fold(out, 'A')
    repair.check_config(Config.fromfile(str(out/'A/detector_config.py')))
    return saved


def train(out):
    check(out)
    process = subprocess.run([sys.executable,str(ROOT/'tools/train.py'),
        str(out/'A/detector_config.py'),'--no-validate','--seed','1701','--deterministic'],cwd=str(ROOT))
    result = json.loads((out/'A/prevention/completion.json').read_text())
    if (process.returncode == 0 or result.get('momentum_restart_revision') != repair.VERSION or
            result.get('task_budget_revision') != parent.task.VERSION or
            result.get('task_memory_revision') != parent.task.MEMORY_REVISION or
            result['status'] not in ('PREVENTION_WINDOW_PASS','ANOMALY_QUARANTINED_BEFORE_OPTIMIZER',
                                    'FORWARD_FAILURE_STOP','UPDATE_OR_RESOURCE_FAILURE_STOP')):
        raise ValueError('Trainer did not stop at registered repaired boundary')
    if result['status'] == 'PREVENTION_WINDOW_PASS' and result['completed_updates'] != 7000:
        raise ValueError('Incomplete window cannot pass')
    check(out)
    parent.parent.old.write(out/'completion.json', dict(result,trainer_exit=process.returncode,
        expected_bounded_stop=True,automatic_continuation=False,reliability_performance_measured=False,
        unique_root_cause_proven=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('prepare','check','train'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if ROOT/'work_dirs' not in out.parents:
        raise ValueError('Project work_dirs only')
    {'prepare':prepare,'check':check,'train':train}[args.mode](out)


if __name__ == '__main__':
    main()
