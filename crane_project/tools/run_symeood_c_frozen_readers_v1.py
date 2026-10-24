"""Resource scheduling only for the frozen observer; no model changes.

Candidate readers run in separate processes on the same physical GPU3, up to
four at once. Gradient readers remain serial. Handover waits for the original
100 full VAL/state receipt and terminates only this task's verified launcher.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
CORE=ROOT/'crane_project/tools/diagnose_symeood_c_frozen_v1.py'
spec=importlib.util.spec_from_file_location('frozen_observer',CORE)
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)


def worker(out,group,phase):
    import torch
    core.imports()
    draft=core.check_identity(out)
    protocol=json.loads((out/'protocol.json').read_text())
    assert protocol['script_sha256']==core.sha(CORE)
    model,cfg=core.build(group)
    before=core.model_sha(model)
    dest=out/group;dest.mkdir(exist_ok=True)
    if phase in ('val','train'):
        core.candidate_run(model,cfg,dest,phase,group)
    else:
        core.grad_run(model,cfg,dest,draft['training_views'])
        assert sum(1 for _ in (dest/'gradient_summary.jsonl').open())==128
    after=core.model_sha(model);assert before==after
    assert all(p.grad is None for p in model.parameters())
    core.write(out/('reader_%s_%s.json'%(phase,group)),dict(group=group,phase=phase,passed=True,
                before=before,after=after,peak_torch_MiB=torch.cuda.max_memory_allocated()/1024**2))


def run(out):
    core.check_identity(out)
    original=json.loads((out/'launch.json').read_text())
    pid=original['pid']
    proc=Path('/proc')/str(pid)
    command=(proc/'cmdline').read_bytes()
    assert b'diagnose_symeood_c_frozen_v1.py' in command and str(out).encode() in command
    assert proc.stat().st_uid==os.getuid()
    core.write(out/'execution_addendum.json',dict(core_sha256=core.sha(CORE),scheduler_sha256=core.sha(__file__),
               frozen_protocol_sha256=core.sha(out/'protocol.json'),physical_gpu=3,candidate_processes=4,
               gradient_processes=1,samples_changed=False,coefficients_changed=False,
               reason='overlap independent candidate compression/reads; retain original 100 VAL gate',
               handover_pid=pid))
    print('WAIT_FOR_100_FULL_VAL_AND_STATE',flush=True)
    while True:
        records=[json.loads(line) for line in (out/'state_checks.jsonl').read_text().splitlines()] if (out/'state_checks.jsonl').exists() else []
        if any(r['group']=='100' and r['phase']=='val' and r['passed'] for r in records):break
        if not proc.exists():raise RuntimeError('original reader stopped before VAL/state gate')
        time.sleep(.05)
    os.kill(pid,signal.SIGTERM)
    core.write(out/'handover.json',dict(pid=pid,at='after 100 complete VAL and unchanged-state receipt',
                                       time=time.strftime('%Y-%m-%d %H:%M:%S')))
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='3';env['PYTHONPATH']=str(ROOT)
    for phase in ('val','train','gradient'):
        groups=('101','110','111') if phase=='val' else core.GROUPS
        jobs=[]
        for group in groups:
            log=(out/('reader_%s_%s.log'%(phase,group))).open('x')
            cmd=[sys.executable,str(Path(__file__).resolve()),'--out',str(out),'--worker',group,'--phase',phase]
            p=subprocess.Popen(cmd,cwd=str(ROOT),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            jobs.append((group,p,log))
            if phase=='gradient':
                code=p.wait();log.close();assert code==0,(phase,group,code)
                print('COMPLETE',phase,group,flush=True)
        if phase!='gradient':
            for group,p,log in jobs:
                code=p.wait();log.close();assert code==0,(phase,group,code)
                print('COMPLETE',phase,group,flush=True)
            receipts=[json.loads((out/g/('candidates_'+phase+'_receipt.json')).read_text()) for g in core.GROUPS]
            assert len({r['input_tensor_sequence_sha256'] for r in receipts})==1
            assert all(r['frames']==(887 if phase=='val' else 2558) for r in receipts)
            print('ALL_INPUT_DIGESTS_MATCH',phase,flush=True)
    core.check_identity(out)
    core.write(out/'complete.json',dict(passed=True,groups=core.GROUPS,VAL_replay_frames=3548,
               candidate_frames=13780,gradient_batches=512,updates=0,physical_gpu=3,
               time=time.strftime('%Y-%m-%d %H:%M:%S')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--worker',choices=core.GROUPS)
    p.add_argument('--phase',choices=['val','train','gradient'])
    args=p.parse_args();os.chdir(ROOT)
    if args.worker:worker(args.out.resolve(),args.worker,args.phase)
    else:run(args.out.resolve())
