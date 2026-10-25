"""Finite low-VRAM audit queue; no formal training, VAL/TEST or cleanup."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(out,pid):
    out=out.resolve();contract=json.loads((out/'queue_contract.json').read_text())
    status=out/'queue_status.json'
    def write(**row):
        row['time']=time.strftime('%Y-%m-%d %H:%M:%S');tmp=status.with_suffix('.tmp')
        tmp.write_text(json.dumps(row,indent=2));tmp.replace(status)
    def verify_sources():
        for p,h in contract['source_sha256'].items():assert sha(ROOT/p)==h,('source changed',p)
    def completed(mode):
        records=[]
        for g in ('100','101','110','111'):
            for phase in ('early','mature'):
                for rank in (0,1):
                    candidates=[out/folder/mode/g/phase/('summary_rank%d.json'%rank) for folder in ('short_serial','short')]
                    found=next((p for p in candidates if p.exists()),None);assert found,('missing',mode,g,phase,rank)
                    row=json.loads(found.read_text());assert row['steps']==20
                    records.append(dict(path=str(found),all_steps_equal=row['all_steps_equal']))
        return records
    try:
        verify_sources();write(state='waiting_current_deterministic',pid=pid)
        deadline=time.monotonic()+6*3600
        while Path('/proc/%d/cmdline'%pid).exists():
            raw=Path('/proc/%d/cmdline'%pid).read_bytes()
            if not raw:break
            assert b'audit_symeood_cross_entry_serial_v1.py' in raw and b'29655' in raw,'PID identity changed'
            assert time.monotonic()<deadline,'wait timeout'
            time.sleep(15)
        det=completed('det')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,3',PYTHONPATH=str(ROOT),OMP_NUM_THREADS='1')
        jobs=[('nondeterministic','crane_project/tools/audit_symeood_cross_entry_serial_v1.py',['--mode','nondet']),
              ('boundaries','crane_project/tools/audit_symeood_cross_entry_boundaries_v1.py',['--stage','boundaries']),
              ('s_contract','crane_project/tools/audit_symeood_cross_entry_boundaries_v1.py',['--stage','s_contract'])]
        for i,(name,entry,extra) in enumerate(jobs):
            verify_sources();write(state='running',stage=name)
            command=[sys.executable,'-m','torch.distributed.launch','--nproc_per_node=2','--master_port='+str(29656+i),entry,'--out',str(out)]+extra
            with (out/(name+'_serial.log')).open('x') as log:
                code=subprocess.call(command,cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT)
            assert code==0,(name,'nonzero exit',code)
        nd=completed('nondet')
        boundaries=[json.loads(s) for p in (out/'boundaries').glob('comparisons_rank*.jsonl') for s in p.read_text().splitlines()]
        checks=[json.loads(s) for p in (out/'s_contract').glob('checks_rank*.jsonl') for s in p.read_text().splitlines()]
        assert len(boundaries)==160 and len(checks)==8
        assert all(r['passed'] for r in checks),'S contract check failed'
        verify_sources()
        write(state='complete',deterministic=det,nondeterministic=nd,boundary_records=len(boundaries),
              boundary_equal=sum(r['equal'] for r in boundaries),s_contract_checks=len(checks),analysis_pending=True,TEST=False,depth=False)
    except Exception:
        write(state='stopped_on_error',error=traceback.format_exc(),TEST=False,depth=False)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--wait-pid',type=int,required=True)
    args=p.parse_args();main(args.out,args.wait_pid)
