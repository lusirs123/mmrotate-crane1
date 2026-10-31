"""One bounded diagnostic process on GPU0, external guard, no retries."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.tools.diagnose_eood_AB_frozen_v1 import ARMS,telemetry,enforce_resources,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    a=p.parse_args();base=a.base.resolve()
    base.mkdir(parents=True,exist_ok=False)
    def status(state,**kw):
        value=dict(state=state,time_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat(),**kw)
        (base/'run_status.json').write_text(json.dumps(value,indent=2))
    sources={str(x.relative_to(ROOT)):sha(x) for x in (ROOT/'mmrotate').rglob('*.py')}
    for x in ('crane_project/tools/diagnose_eood_AB_frozen_v1.py',
              'crane_project/tools/run_eood_AB_diagnostic_v1.py',
              'crane_project/tools/preflight_port_center_size_v1.py',
              'tests/test_eood_AB_diagnostic_resources_v1.py'):
        sources[x]=sha(ROOT/x)
    configs={cfg:sha(ROOT/cfg) for _,cfg,_,_ in ARMS}
    for _,cfg,cp,h in ARMS:assert sha(ROOT/cp)==h,cp
    tests=subprocess.run([sys.executable,'-m','unittest','tests.test_eood_AB_diagnostic_resources_v1','-v'],
        cwd=str(ROOT),env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True)
    (base/'resource_tests.log').write_text(tests.stdout+tests.stderr)
    assert tests.returncode==0
    enforce_resources(telemetry(0),0,before=True)
    c=dict(sources=sources,configs=configs,physical_gpu=0,allocator_cap_mib=2048,
        own_driver_stop_mib=2560,minimum_free_during_mib=2048,temperature_stop_C=75,
        maximum_seconds=1200,TEST=False,VAL=False,training_updates=0,retry=False)
    (base/'protocol.json').write_text(json.dumps(c,indent=2))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',
             OMP_NUM_THREADS='1',PYTHONPATH=str(ROOT))
    with (base/'diagnostic.log').open('x') as log:
        proc=subprocess.Popen([sys.executable,'crane_project/tools/diagnose_eood_AB_frozen_v1.py',
            '--gpu','0','--out',str(base/'result')],cwd=str(ROOT),env=env,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        status('RUNNING',pid=proc.pid,controller_pid=os.getpid())
        start=time.time()
        def terminate(signum,frame):raise RuntimeError('Diagnostic controller terminated')
        signal.signal(signal.SIGTERM,terminate)
        try:
            while proc.poll() is None:
                time.sleep(3)
                if time.time()-start>1200:raise RuntimeError('Diagnostic time bound')
                row=telemetry(0)
                output=subprocess.check_output(['nvidia-smi','-i','0',
                    '--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
                own={}
                for line in output.splitlines():
                    parts=line.split(',')
                    if len(parts)!=2 or not all(v.strip().isdigit() for v in parts):continue
                    pid,used=map(int,parts)
                    try:
                        if os.getpgid(pid)==proc.pid:own[pid]=used
                    except ProcessLookupError:pass
                enforce_resources(row,max(own.values(),default=0))
                with (base/'guard.jsonl').open('a') as f:
                    f.write(json.dumps(dict(time=time.time(),GPU=row,own_process_mib=own))+'\n')
            assert proc.returncode==0,proc.returncode
            for name,h in dict(sources,**configs).items():assert sha(ROOT/name)==h,name
            result=json.loads((base/'result/completion.json').read_text())
            assert result['status']=='COMPLETE' and result['views']==384 and result['training_updates']==0
            status('COMPLETE',training_updates=0,TEST=False,VAL=False)
        except BaseException as e:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
            status('STOPPED_ON_ERROR',error=repr(e),retry=False,training_updates=0)
            raise


if __name__=='__main__':main()
