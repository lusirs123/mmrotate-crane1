"""One authorized inputs -> discarded verification -> fresh24 LVSR run."""
import argparse
from datetime import datetime,timedelta,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.utils import symeood_r4_lvsr_protocol_v1 as p


def command(stage,attempt,gpu):
    if stage not in ('inputs','verify','train'):raise ValueError('Only fixed LVSR stages')
    return [sys.executable,'-u','crane_project/tools/train_symeood_r4_lvsr_pair_v1.py',
        '--stage',stage,'--attempt',attempt,'--gpu',str(gpu)]


def main(args):
    paths=p.paths(args.attempt);identity=p.checked_sources()
    if paths['attempt'].exists():raise ValueError('Fresh attempt required; no automatic overwrite/retry')
    if json.loads((paths['preparation']/'protocol.json').read_text())['identity']!=identity:
        raise ValueError('Prepared source differs')
    from crane_project.tools.train_symeood_r4_midpoint_v1 import resource_check
    resource_check(args.gpu,before=True)
    if p.sha(ROOT/p.parent.R4_CHECKPOINT)!=p.parent.R4_SHA:
        raise ValueError('R4 checkpoint SHA differs')
    cp=p.midpoint_directory()/'train/head_epoch_20.pth'
    if p.sha(cp)!=p.MIDPOINT_SHA:raise ValueError('M20 checkpoint SHA differs')
    paths['attempt'].mkdir(parents=True)
    def status(state,**extra):
        tmp=paths['attempt']/'status.tmp'
        tmp.write_text(json.dumps(dict(state=state,time_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat(),
            controller_pid=os.getpid(),physical_gpu=args.gpu,TEST=False,depth=False,**extra),indent=2))
        tmp.replace(paths['attempt']/'status.json')
    (paths['attempt']/'launch.json').write_text(json.dumps(dict(identity=identity,controller_pid=os.getpid(),
        physical_gpu=args.gpu,stages=['inputs','verify','train'],TEST=False,depth=False,retry=False,
        continuation=False,automatic_adoption=False),indent=2))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONPATH=str(ROOT),
        OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    def interrupted(*args):raise RuntimeError('Controller interrupted')
    signal.signal(signal.SIGTERM,interrupted);proc=None
    try:
        for stage in ('inputs','verify','train'):
            if p.checked_sources()!=identity:raise ValueError('Source drift before '+stage)
            with (paths['attempt']/(stage+'.log')).open('x') as log:
                proc=subprocess.Popen(command(stage,args.attempt,args.gpu),cwd=str(ROOT),env=env,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                status('RUNNING',stage=stage,child_pid=proc.pid)
                while proc.poll() is None:
                    time.sleep(5)
                    row=resource_check(args.gpu)
                    text=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),
                        '--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
                    own=sum(int(line.split(',')[1]) for line in text.splitlines() if line.split(',')[0].strip()==str(proc.pid))
                    with (paths['attempt']/'resources.jsonl').open('a') as f:
                        f.write(json.dumps(dict(time=time.time(),stage=stage,child_pid=proc.pid,child_own_mib=own,**row))+'\n')
                    if own>=p.parent.MEMORY['process_stop_mib']:raise RuntimeError('Child memory guard')
                if proc.returncode!=0:raise RuntimeError(stage+' exited '+str(proc.returncode))
                status('STAGE_COMPLETE',stage=stage)
        complete=json.loads((paths['train']/'completion.json').read_text())
        if complete['epochs_completed']!=24 or complete['identity']!=identity:raise ValueError('Incomplete LVSR training')
        if p.sha(ROOT/p.parent.R4_CHECKPOINT)!=p.parent.R4_SHA or p.sha(cp)!=p.MIDPOINT_SHA:
            raise ValueError('Frozen frontend file changed')
        status('COMPLETE',epochs=24,updates=complete['updates'],selected_epoch=complete['selected_epoch'],
            representative_epoch=complete['representative_epoch'],automatic_adoption=False)
    except BaseException as error:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
        status('STOPPED_ON_ERROR',error=repr(error),retry=False);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,choices=(2,3),default=2)
    parser.add_argument('--attempt',default='attempt_001')
    main(parser.parse_args())
