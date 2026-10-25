"""Generate inspectable plans only; never starts GPU or formal training.

B and111 are naming aliases for the SAME B-derived execution config.
Native acceptance, repetition schedule/seeds and runtime finalization remain
separate gates. No --execute workaround is provided here.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
B_CONFIG=ROOT/'crane_project/configs/crane_symeood_k1_port_day2night_aug_b_v1.py'


def prepare(lock_path,out,alias,seed):
    from mmcv import Config
    lock=json.loads(lock_path.read_text())
    assert lock['canonical_config']==str(B_CONFIG)
    out.mkdir(parents=True,exist_ok=False)
    cfg=Config.fromfile(str(B_CONFIG));cfg.work_dir=str(out)
    imports=list(cfg.custom_imports.imports)
    if 'crane_project.utils.symeood_repro_trace_v1' not in imports:imports.append('crane_project.utils.symeood_repro_trace_v1')
    cfg.custom_imports=dict(imports=imports,allow_failed_imports=False)
    cfg.custom_hooks=[dict(type='SymEOODReproTraceV1',priority='LOWEST',runtime_lock=str(lock_path),engineering=False)]
    assert cfg.runner.max_epochs==24 and cfg.load_from is None and cfg.resume_from is None
    config_path=out/'canonical_execution.py';cfg.dump(str(config_path))
    plan=dict(alias=alias,method='complete_SymEOOD_scaleB',base_config=str(B_CONFIG),seed=seed,
        alias_changes_only=['work_dir','reporting name'],execution_config=str(config_path),
        execution_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        runtime_lock=str(lock_path),runtime_lock_state=lock['state'],
        formal_schedule_frozen=lock['formal_schedule_frozen'],launch_allowed=False,
        command_preview=[sys.executable,'-m','torch.distributed.launch','--nproc_per_node=2','--master_port=<allocate when scheduled>',
                         'tools/train.py',str(config_path),'--launcher','pytorch','--seed',str(seed)],
        prospective_native_environment={'CUDA_VISIBLE_DEVICES':'2,3','PYTHONPATH':str(ROOT)},
        blockers=lock['pending'],note='PLAN ONLY. No training, inference or GPU checks were started; no repetitions scheduled. CPU offload and limited loaders are NOT in the formal plan.')
    (out/'plan.json').write_text(json.dumps(plan,indent=2));return plan


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--alias',choices=['B','111'],required=True);p.add_argument('--seed',type=int,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.lock.resolve(),a.out.resolve(),a.alias,a.seed),indent=2))
