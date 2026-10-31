"""Guarded zero-update audit only; deliberately no training CLI or optimizer."""
from datetime import datetime,timezone,timedelta
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.tools.diagnose_eood_AB_frozen_v2 import telemetry,enforce_resources,sha
from crane_project.tools.audit_eood_B_balance_v1 import CONFIG,CASES


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True)
    a=p.parse_args();base=a.base.resolve();base.mkdir(parents=True,exist_ok=False)
    def status(state,**kw):
        (base/'run_status.json').write_text(json.dumps(dict(state=state,
            time_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat(),**kw),indent=2))
    sources={str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'mmrotate').rglob('*.py')}
    for name in ('crane_project/tools/audit_eood_B_balance_v1.py','crane_project/tools/run_eood_B_balance_audit_v1.py',
                 'crane_project/tools/diagnose_eood_AB_frozen_v2.py','crane_project/tools/preflight_port_center_size_v1.py',
                 'tests/test_eood_B_balance_audit_v1.py',CONFIG):sources[name]=sha(ROOT/name)
    for _,epoch,_,digest in CASES:
        assert sha(ROOT/('work_dirs/eood_B_regression_20261031_v3/train/epoch_%d.pth'%epoch))==digest
    protocol=dict(sources=sources,physical_gpu=1,allocator_cap_mib=6144,own_driver_stop_mib=6656,
        minimum_free_mib=512,temperature_stop_C=None,default_driver_thermal_policy=True,
        TRAIN_only=True,TEST=False,VAL=False,training_updates=0,planned_training_runs=0,
        batch=2,cases=CASES,scales=[1.,.5],pairs_from_original_fixed_64_positions=[0,8,16,24],
        coefficient_scope='0 identity, .25 existing B2, 1 diagnostic unit weight; no performance-based coefficient selection.')
    (base/'protocol.json').write_text(json.dumps(protocol,indent=2))
    env=dict(os.environ,PYTHONPATH=str(ROOT),CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='1')
    tool='crane_project/tools/audit_eood_B_balance_v1.py'
    tests=subprocess.run([sys.executable,'-m','unittest','tests.test_eood_B_balance_audit_v1','-v'],
        env=dict(env,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,cwd=str(ROOT))
    (base/'CPU_tests.log').write_text(tests.stdout+tests.stderr);assert tests.returncode==0
    subprocess.run([sys.executable,tool,'--stage','offline','--out',str(base/'offline')],
        env=dict(env,CUDA_VISIBLE_DEVICES=''),cwd=str(ROOT),check=True)
    enforce_resources(telemetry(1),0,before=True)
    with (base/'diagnostic.log').open('x') as log:
        proc=subprocess.Popen([sys.executable,tool,'--stage','gpu','--gpu','1','--out',str(base/'gpu')],
            env=dict(env,CUDA_VISIBLE_DEVICES='1'),cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        status('RUNNING',pid=proc.pid,controller_pid=os.getpid(),training_updates=0)
        def terminate(signum,frame):raise RuntimeError('Audit controller terminated')
        signal.signal(signal.SIGTERM,terminate);start=time.time()
        try:
            while proc.poll() is None:
                time.sleep(3)
                if time.time()-start>900:raise RuntimeError('Audit time bound')
                row=telemetry(1)
                text=subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid,used_memory',
                    '--format=csv,noheader,nounits'],text=True)
                own={}
                for line in text.splitlines():
                    parts=line.split(',')
                    if len(parts)!=2 or not all(x.strip().isdigit() for x in parts):continue
                    pid,used=map(int,parts)
                    try:
                        if os.getpgid(pid)==proc.pid:own[pid]=used
                    except ProcessLookupError:pass
                with (base/'guard.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),GPU=row,own_process_mib=own))+'\n')
                enforce_resources(row,max(own.values(),default=0))
            assert proc.returncode==0,proc.returncode
            for name,h in sources.items():assert sha(ROOT/name)==h,name
            done=json.loads((base/'gpu/completion.json').read_text())
            assert done['passed'] and done['batches']==32 and done['training_updates']==0
            status('COMPLETE',training_updates=0,training_runs=0,TEST=False,VAL=False)
        except BaseException as error:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
            status('STOPPED_ON_ERROR',error=repr(error),retry=False,training_updates=0)
            raise


if __name__=='__main__':main()
