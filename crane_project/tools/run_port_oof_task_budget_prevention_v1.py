#!/usr/bin/env python3
"""Separate fresh-A prevention entry; preparation is not training authorization."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.tools import run_port_reliability_oof_prevention_v1 as parent
from crane_project.utils import port_oof_task_budget_v1 as task

FILES=['crane_project/tools/run_port_oof_task_budget_prevention_v1.py',
       'crane_project/utils/port_oof_task_budget_v1.py',
       'mmrotate/core/hooks/port_oof_task_budget_hook.py','tests/test_port_oof_task_budget_v1.py']


def prepare(out):
    from mmcv import Config
    parent.prepare(out)
    path=out/'A/detector_config.py'
    cfg=Config.fromfile(str(path))
    task.configure(cfg)
    cfg.dump(str(path))
    pin=parent.old.sha(path)
    membership=json.loads((out/'A/membership.json').read_text())
    # parent.prepare created these metadata files in a new exclusive directory.
    # Replace only that newly prepared metadata, never an existing run.
    (out/'A/membership.json').write_text(json.dumps(dict(membership,resolved_config_sha256=pin),indent=2)+'\n')
    saved=json.loads((out/'prevention.json').read_text())
    saved.update(task_budget_revision=task.VERSION,task_budgets={k:task.BUDGET for k in task.TASKS},
        configuration_sha256=pin)
    saved['sources'].update({p:parent.old.sha(ROOT/p) for p in FILES})
    (out/'prevention.json').write_text(json.dumps(saved,indent=2)+'\n')
    check(out)


def check(out):
    from mmcv import Config
    saved=json.loads((out/'prevention.json').read_text())
    c=parent.contract
    if (saved.get('task_budget_revision')!=task.VERSION or
            saved.get('task_budgets')!={k:task.BUDGET for k in task.TASKS} or
            saved['protocol']!=c.VERSION or saved['settings']!=c.SETTINGS or
            saved['allocation_revision']!=c.ALLOCATION_REVISION or
            saved['memory_revision']!=c.MEMORY_REVISION or saved['memory_settings']!=c.MEMORY_SETTINGS):
        raise ValueError('Independent task-budget prevention contract changed')
    for p,pin in saved['sources'].items():
        if parent.old.sha(ROOT/p)!=pin:raise ValueError('Source changed: '+p)
    if parent.old.sha(out/'A/detector_config.py')!=saved['configuration_sha256']:
        raise ValueError('Config changed')
    parent.old.check_fold(out,'A')
    task.check_task_config(Config.fromfile(str(out/'A/detector_config.py')))
    return saved


def train(out):
    check(out)
    proc=subprocess.run([sys.executable,str(ROOT/'tools/train.py'),str(out/'A/detector_config.py'),
        '--no-validate','--seed','1701','--deterministic'],cwd=str(ROOT))
    result=json.loads((out/'A/prevention/completion.json').read_text())
    if (proc.returncode==0 or result.get('task_budget_revision')!=task.VERSION or
            result['status'] not in ('PREVENTION_WINDOW_PASS','ANOMALY_QUARANTINED_BEFORE_OPTIMIZER',
                                    'FORWARD_FAILURE_STOP','UPDATE_OR_RESOURCE_FAILURE_STOP')):
        raise ValueError('Trainer did not stop at registered task-budget boundary')
    if result['status']=='PREVENTION_WINDOW_PASS' and result['completed_updates']!=7000:
        raise ValueError('Incomplete window cannot pass')
    check(out)
    parent.old.write(out/'completion.json',dict(result,trainer_exit=proc.returncode,
        expected_bounded_stop=True,automatic_continuation=False,reliability_performance_measured=False,
        unique_root_cause_proven=False))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=('prepare','check','train'),required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();out=a.out.resolve()
    if ROOT/'work_dirs' not in out.parents:raise ValueError('Project work_dirs only')
    {'prepare':prepare,'check':check,'train':train}[a.mode](out)


if __name__=='__main__':main()
