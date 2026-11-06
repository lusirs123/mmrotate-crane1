"""Bounded TRAIN-only allocator diagnosis; never starts formal training.

Official OBBTrainer/model/loss/optimizer are reused. Temporary updates are
discarded at a fixed batch bound, before epoch-end validation or saving.
"""
import argparse
from datetime import datetime, timezone, timedelta
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.tools import run_external_obb_training_v1 as native

PLAN = 'configs/yolo26/memory_probe_v1.json'


def check_plan(plan):
    if (plan['role'] != 'bounded_TRAIN_memory_diagnosis_only'
            or any(plan[k] for k in ('formal_training','VAL_inference','TEST_access',
                                     'save_checkpoints','automatic_retry'))
            or not 1 <= len(plan['cases']) <= 3):
        raise ValueError('Only a bounded TRAIN resource diagnosis is allowed')
    names=[]
    for case in plan['cases']:
        if (not 16 <= case['steps'] <= 64 or not 4096 <= case['allocator_mib'] <= 6144
                or not case['allocator_mib'] < case['driver_mib'] <= 6656
                or case['minimum_free_mib'] < 1536):
            raise ValueError('Invalid step bound or memory safety limits')
        names.append(case['name'])
    if len(set(names)) != len(names):raise ValueError('Duplicate probe case')
    return plan


def summarize_records(records):
    if len(records)<16:raise ValueError('Insufficient resource records')
    # Compare synchronized boundaries after initial allocator growth.
    tail=records[len(records)//2:]
    reservations=[r['reserved_mib'] for r in tail]
    return dict(steps=len(records),tail_steps=len(tail),
        tail_reserved_min_mib=min(reservations),tail_reserved_max_mib=max(reservations),
        tail_reserved_range_mib=max(reservations)-min(reservations),
        tail_allocation_retries=tail[-1]['allocation_retries']-tail[0]['allocation_retries'],
        total_allocation_retries=records[-1]['allocation_retries'],
        OOMs=records[-1]['OOMs'],
        train_peak_allocated_mib=max(r['step_peak_allocated_mib'] for r in records),
        train_peak_reserved_mib=max(r['step_peak_reserved_mib'] for r in records))


class ProbeComplete(Exception):
    pass


def worker(out, case_name):
    protocol=native.read(out.parent/'protocol.json');native.assert_sources(protocol)
    case=next(c for c in protocol['memory_probe']['cases'] if c['name']==case_name)
    rank=int(os.environ['LOCAL_RANK'])
    native.write_new(out/('rank_%d_identity.json'%rank),dict(pid=os.getpid(),rank=rank))
    import torch
    torch.cuda.set_device(rank);torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(case['allocator_mib']*2**20/
        torch.cuda.get_device_properties(rank).total_memory,rank)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0,str(ROOT/'third_party/ultralytics'))
    import ultralytics
    if Path(ultralytics.__file__).resolve().parent != ROOT/'third_party/ultralytics/ultralytics':
        raise ValueError('Wrong Ultralytics implementation')
    from ultralytics.models.yolo.obb.train import OBBTrainer
    from ultralytics.utils import YAML,callbacks
    callbacks.add_integration_callbacks=lambda instance:None
    args=YAML.load(str(ROOT/'configs/yolo26/crane_yolo26m_obb_train_v1.yaml'));args.pop('mode',None)
    if (args['epochs'],args['imgsz'],args['amp'],args['batch'],args['nbs']) != (24,1024,False,2,4):
        raise ValueError('Native training recipe changed')
    args.update(project=str(out/'native'),name='discarded',exist_ok=False,
        model=str(ROOT/args['model']),data=str(ROOT/args['data']),
        pretrained=str(ROOT/args['pretrained']),val=False,save=False,save_period=-1)
    class TrainOnlyProbe(OBBTrainer):
        def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
            # Construction of a validator is native even with val=False. Its
            # metadata comes from TRAIN, and validation itself is forbidden.
            return super().get_dataloader(self.data['train'] if mode!='train' else dataset_path,
                                           batch_size,rank,mode)
        def validate(self):raise RuntimeError('VAL forbidden in memory probe')
        def save_model(self):raise RuntimeError('Checkpoint forbidden in memory probe')
        def final_eval(self):raise RuntimeError('Final evaluation forbidden in memory probe')
    trainer=TrainOnlyProbe(overrides=args);records=[];updates=[0]
    def initialized(t):
        original=t.optimizer.step
        def step(*a,**kw):
            value=original(*a,**kw);updates[0]+=1
            if updates[0]>case['steps']:raise ValueError('Update bound exceeded')
            return value
        t.optimizer.step=step
    def begin(t):
        torch.cuda.synchronize(rank);torch.cuda.reset_peak_memory_stats(rank)
    log=(out/('rank_%d_memory.jsonl'%rank)).open('x')
    def end(t):
        torch.cuda.synchronize(rank);loss=float(t.loss.detach().cpu())
        if not math.isfinite(loss):raise ValueError('Nonfinite loss')
        st=torch.cuda.memory_stats(rank)
        row=dict(step=len(records)+1,updates=updates[0],loss=loss,
            allocated_mib=torch.cuda.memory_allocated(rank)/2**20,
            reserved_mib=torch.cuda.memory_reserved(rank)/2**20,
            step_peak_allocated_mib=torch.cuda.max_memory_allocated(rank)/2**20,
            step_peak_reserved_mib=torch.cuda.max_memory_reserved(rank)/2**20,
            inactive_split_mib=st.get('inactive_split_bytes.all.current',0)/2**20,
            allocation_retries=st.get('num_alloc_retries',0),OOMs=st.get('num_ooms',0))
        records.append(row);log.write(json.dumps(row)+'\n');log.flush()
        native.atomic(out/('rank_%d_heartbeat.json'%rank),row)
        if len(records)==case['steps']:raise ProbeComplete()
    trainer.add_callback('on_pretrain_routine_end',initialized)
    trainer.add_callback('on_train_batch_start',begin)
    trainer.add_callback('on_train_batch_end',end)
    try:
        try:trainer.train()
        except ProbeComplete:pass
        if len(records)!=case['steps']:raise ValueError('Wrong resource step count')
        if not all(torch.isfinite(p).all() for p in trainer.model.parameters()):
            raise ValueError('Nonfinite updated parameters')
        native.write_new(out/('rank_%d_result.json'%rank),dict(passed=True,
            rank=rank,summary=summarize_records(records),updates=updates[0],
            optimizer=trainer.optimizer.__class__.__name__,
            epochs_completed=0,parameters_discarded=True,checkpoint_saved=False,VAL=False,TEST=False))
        if torch.distributed.is_initialized():torch.distributed.barrier()
    finally:
        log.close()
        for loader in (getattr(trainer,'train_loader',None),getattr(trainer,'test_loader',None)):
            if hasattr(loader,'close'):loader.close()
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()


def one_case(out, protocol, case):
    native.assert_sources(protocol);native.idle_gpus()
    folder=out/case['name'];folder.mkdir()
    env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0,1',
        PYTHONPATH=str(ROOT),PYTORCH_CUDA_ALLOC_CONF=case['allocator_config'],
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true',WANDB_DISABLED='true',
        YOLO_CONFIG_DIR=str(folder/'settings'))
    command=[str(ROOT/'.venvs/yolo26_v1/bin/python'),'-m','torch.distributed.run',
        '--standalone','--nproc_per_node=2',str(Path(__file__).resolve()),
        '--stage','worker','--out',str(folder),'--case',case['name']]
    started=time.monotonic();reason=None;peaks={str(g):0 for g in native.GPUS}
    with (folder/'native.log').open('x') as log,(folder/'gpu_samples.jsonl').open('x') as samples:
        child=subprocess.Popen(command,cwd=str(ROOT),env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        native.write_new(folder/'launcher.json',dict(pid=child.pid,command=command))
        while child.poll() is None:
            try:
                rows={r['index']:r for r in native.gpu_rows() if r['index'] in native.GPUS}
                pids=native.worker_pids(folder);apps=native.processes()
                for g,row in rows.items():
                    own=sum(a['used_mib'] for a in apps if a['pid'] in pids and a['uuid']==row['uuid'])
                    peaks[str(g)]=max(peaks[str(g)],own)
                    if own>case['driver_mib'] or row['free_mib']<case['minimum_free_mib']:reason='MEMORY_GUARD'
                    if time.monotonic()-started>15 and any(a['uuid']==row['uuid'] and a['pid'] not in pids for a in apps):reason='GPU_CONTENTION'
                if time.monotonic()-started>protocol['memory_probe']['timeout_seconds_per_case']:reason='TIMEOUT'
                samples.write(json.dumps(dict(seconds=time.monotonic()-started,gpus=rows,
                    owned=[a for a in apps if a['pid'] in pids]))+'\n');samples.flush()
                if reason:native.stop_group(child,folder);break
                time.sleep(2)
            except BaseException:
                native.stop_group(child,folder);raise
        code=child.wait()
    reports=[native.read(p) for p in sorted(folder.glob('rank_*_result.json'))]
    result=dict(passed=code==0 and reason is None and len(reports)==2,
        exit_code=code,stop_reason=reason,process_peak_mib=peaks,workers=reports,
        formal_training=False,VAL=False,TEST=False,parameters_discarded=True,
        elapsed_seconds=time.monotonic()-started)
    native.write_new(folder/'result.json',result);return result


def run(out):
    protocol=native.read(out/'protocol.json');results={}
    for case in protocol['memory_probe']['cases']:
        native.atomic(out/'status.json',dict(state='BOUNDED_RESOURCE_PROBE',active_case=case['name']))
        results[case['name']]=one_case(out,protocol,case)
    native.assert_sources(protocol)
    native.write_new(out/'completion.json',dict(passed=all(r['passed'] for r in results.values()),
        results=results,formal_training=False,parameters_discarded=True,VAL=False,TEST=False))
    native.atomic(out/'status.json',dict(state='PROBE_FINISHED',formal_training=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('check','launch','run','worker'),default='check')
    p.add_argument('--out',type=Path);p.add_argument('--case')
    a=p.parse_args();os.chdir(ROOT);plan=check_plan(native.read(ROOT/PLAN))
    if a.stage=='check':print(json.dumps(dict(passed=True,plan=plan,GPU_used=False)));return
    if a.out is None:p.error('--out required')
    out=a.out.resolve()
    if a.stage=='worker':return worker(out,a.case)
    if a.stage=='run':return run(out)
    native.prepare(out)
    protocol=native.read(out/'protocol.json');protocol.update(
        protocol='bounded_yolo26_TRAIN_memory_probe_v1',queue=[],memory_probe=plan,
        formal_training=False,VAL_inference=False,TEST=False,checkpoint_saving=False)
    for n in (PLAN,'crane_project/tools/probe_yolo26_memory_v1.py'):
        protocol['sources'][n]=native.sha(ROOT/n)
    native.atomic(out/'protocol.json',protocol)
    with (out/'supervisor.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--stage','run','--out',str(out)],
            cwd=str(ROOT),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(pid=child.pid,out=str(out),formal_training=False,VAL=False,TEST=False,
        created_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat())
    native.write_new(out/'launch.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
