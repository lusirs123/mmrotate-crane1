"""Prepare/explicitly launch native YOLO26 resume. Default is CPU check only.

Restore the selected training checkpoint, not best.pt selection or weight-only
fine-tuning. Original files stay intact; output and logs use a new directory.
"""
import argparse
import csv
from datetime import datetime,timezone,timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.tools import run_external_obb_training_v1 as native

CONFIG='configs/yolo26/crane_yolo26m_obb_resume_v1.json'


def validate_metadata(meta,cfg):
    if (meta['sha256']!=cfg['checkpoint_sha256'] or meta['completed_epochs']!=cfg['expected_completed_epochs']
            or meta['epoch']+1!=meta['completed_epochs'] or not 0<meta['completed_epochs']<cfg['total_epochs']
            or not meta['optimizer_present'] or meta['optimizer_state_entries']<=0
            or not meta['ema_finite'] or meta['head_class']!='OBB26' or meta['classes']!=1):
        raise ValueError('Checkpoint is not the pinned complete, resumable YOLO state')
    expected=dict(task='obb',epochs=24,imgsz=1024,batch=2,nbs=4,amp=False,
                  optimizer='MuSGD',seed=0,deterministic=True,nms=False,save_period=1)
    if cfg['total_epochs']!=24 or any(meta['train_args'].get(k)!=v for k,v in expected.items()):
        raise ValueError('Resume must retain the original 24-epoch recipe')
    if meta['version']!=cfg['runtime']['ultralytics']:
        raise ValueError('Checkpoint implementation version differs')
    return dict(completed_epochs=meta['completed_epochs'],next_epoch=meta['completed_epochs']+1,
                total_epochs=24,remaining_epochs=24-meta['completed_epochs'])


def inspect_checkpoint(path):
    # Caller hides CUDA for the CPU check. Workers use map_location=cpu too.
    sys.path.insert(0,str(ROOT/'third_party/ultralytics'))
    import torch
    import ultralytics
    if Path(ultralytics.__file__).resolve().parent!=ROOT/'third_party/ultralytics/ultralytics':
        raise ValueError('Wrong Ultralytics package')
    c=torch.load(path,map_location='cpu',weights_only=False);ema=c.get('ema');opt=c.get('optimizer')
    return dict(sha256=native.sha(path),epoch=c.get('epoch',-1),completed_epochs=c.get('epoch',-1)+1,
        optimizer_present=isinstance(opt,dict),optimizer_state_entries=len(opt.get('state',{})) if opt else 0,
        optimizer_group_lrs=[g['lr'] for g in opt['param_groups']] if opt else [],
        ema_finite=ema is not None and all(torch.isfinite(v).all().item() for v in ema.state_dict().values() if isinstance(v,torch.Tensor)),
        head_class=ema.model[-1].__class__.__name__ if ema else None,classes=ema.model[-1].nc if ema else None,
        EMA_updates=c.get('updates'),scaler=c.get('scaler'),version=c.get('version'),
        train_args=c.get('train_args',{}),train_results_rows=len(next(iter(c['train_results'].values()))) if c.get('train_results') else 0,
        RNG_state_saved=any('rng' in k.lower() or 'random' in k.lower() for k in c),
        scheduler_serialized='scheduler' in c,
        runtime=dict(python=sys.version.split()[0],torch=torch.__version__,ultralytics=ultralytics.__version__))


def audit(cfg):
    parent=ROOT/cfg['parent_run'];protocol=native.read(parent/'protocol.json')
    if native.sha(parent/'protocol.json')!=cfg['parent_protocol_sha256']:
        raise ValueError('Parent training protocol changed')
    native.assert_sources(protocol);native.check_sources();native.check_configs()
    ckpt=ROOT/cfg['checkpoint']
    with tempfile.TemporaryDirectory(prefix='yolo26_resume_cpu_') as settings:
        os.environ['YOLO_CONFIG_DIR']=settings
        meta=inspect_checkpoint(ckpt)
    continuation=validate_metadata(meta,cfg)
    if meta['runtime']!=cfg['runtime']:raise ValueError('Use the verified existing YOLO runtime')
    results=parent/'yolo26/train/results.csv'
    rows=list(csv.DictReader(results.open()))
    if ([int(float(r['epoch'])) for r in rows]!=list(range(1,meta['completed_epochs']+1))
            or len(rows)!=meta['train_results_rows']):
        raise ValueError('Saved history is inconsistent with the last complete checkpoint')
    return dict(passed=True,checkpoint=str(ckpt),metadata=meta,continuation=continuation,
        original_results_csv=str(results),original_results_sha256=native.sha(results),
        GPU_used=False,trainer_constructed=False,optimizer_updates_executed=0,resume_started=False)


def retry_query(query,attempts=3,delay=1,pause=time.sleep,on_error=None):
    if not 1<=attempts<=3:raise ValueError('Bound GPU query attempts to at most three')
    for attempt in range(1,attempts+1):
        try:return query()
        except (subprocess.SubprocessError,OSError,ValueError) as error:
            if on_error:on_error(dict(attempt=attempt,error=repr(error),unix_time=time.time()))
            if attempt==attempts:raise RuntimeError('GPU_QUERY_UNAVAILABLE') from error
            pause(delay)


def snapshot(cfg):
    def query(option):
        return subprocess.check_output(['nvidia-smi','--id=0,1',option,
            '--format=csv,noheader,nounits'],text=True,timeout=cfg['gpu_query_timeout_seconds'])
    rows={}
    for line in query('--query-gpu=index,uuid,pci.bus_id,memory.total,memory.used,memory.free,utilization.gpu').splitlines():
        i,u,p,t,used,free,util=[s.strip() for s in line.split(',')]
        rows[int(i)]=dict(index=int(i),uuid=u,pci=p,total_mib=int(t),used_mib=int(used),free_mib=int(free),utilization_pct=int(util))
    if set(rows)!={0,1}:raise ValueError('Authorized GPU inventory differs')
    apps=[]
    for line in query('--query-compute-apps=gpu_uuid,pid,used_memory').splitlines():
        if not line.strip():continue
        u,p,m=[s.strip() for s in line.split(',')]
        if u in {r['uuid'] for r in rows.values()}:
            apps.append(dict(uuid=u,pid=int(p),used_mib=int(m)))
    return dict(gpus=rows,apps=apps)


def verify_protocol(p):
    native.assert_sources(p)
    if native.sha(p['resume_input'])!=p['resume_config']['checkpoint_sha256']:
        raise ValueError('Immutable resume input changed')


def worker(out):
    p=native.read(out/'protocol.json');verify_protocol(p);cfg=p['resume_config'];meta=p['checkpoint_audit']['metadata']
    rank=int(os.environ['LOCAL_RANK'])
    native.write_new(out/('rank_%d_identity.json'%rank),dict(pid=os.getpid(),rank=rank))
    import torch
    torch.cuda.set_device(rank);torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(cfg['allocator_mib']*2**20/torch.cuda.get_device_properties(rank).total_memory,rank)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0,str(ROOT/'third_party/ultralytics'))
    from ultralytics.models.yolo.obb.train import OBBTrainer
    from ultralytics.utils import callbacks
    callbacks.add_integration_callbacks=lambda instance:None
    trainer=OBBTrainer(overrides=dict(model=p['resume_input'],resume=p['resume_input'],
        data=meta['train_args']['data'],device='0,1',save_dir=str(out/'train')))
    updates=[0];batches=[0]
    def status(t,state):
        items={k:float(v.detach().cpu()) for k,v in t.loss_items.items()} if isinstance(getattr(t,'loss_items',None),dict) else {}
        native.atomic(out/('rank_%d_heartbeat.json'%rank),dict(state=state,rank=rank,
            epoch=int(getattr(t,'epoch',t.start_epoch))+1,total_epochs=t.epochs,start_epoch=t.start_epoch,
            resumed_microbatches=batches[0],resumed_optimizer_updates=updates[0],loss_terms=items,
            peak_allocated_mib=torch.cuda.max_memory_allocated(rank)/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved(rank)/2**20))
    def initialized(t):
        if (not t.resume or t.start_epoch!=cfg['expected_completed_epochs'] or t.epochs!=24
                or (t.args.imgsz,t.args.batch,t.args.nbs,t.args.amp)!=(1024,2,4,False)
                or t.optimizer.__class__.__name__!='MuSGD'):
            raise ValueError('Native resume did not restore the pinned epoch/recipe')
        state=t.optimizer.state_dict()
        if (len(state['state'])!=meta['optimizer_state_entries'] or t.ema.updates!=meta['EMA_updates']
                or [g['lr'] for g in state['param_groups']]!=meta['optimizer_group_lrs']):
            raise ValueError('Native optimizer/EMA/LR state was not restored')
        native.write_new(out/('rank_%d_restored_state.json'%rank),dict(passed=True,
            start_epoch=t.start_epoch,next_epoch=t.start_epoch+1,target_epochs=t.epochs,
            optimizer_states=len(state['state']),EMA_updates=t.ema.updates,
            scheduler_last_epoch=t.scheduler.last_epoch,exact_batch_or_RNG_replay=False))
        original=t.optimizer.step
        def counted(*a,**kw):
            value=original(*a,**kw);updates[0]+=1;return value
        t.optimizer.step=counted;status(t,'RESUMED_INITIALIZED')
    def batch_end(t):
        batches[0]+=1
        if not torch.isfinite(t.loss).all():raise RuntimeError('Nonfinite resumed loss')
        if batches[0]<=10 or batches[0]%10==0:status(t,'TRAINING')
    trainer.add_callback('on_pretrain_routine_end',initialized)
    trainer.add_callback('on_train_batch_end',batch_end)
    trainer.add_callback('on_train_epoch_end',lambda t:status(t,'EPOCH_FINISHED'))
    try:
        trainer.train()
        if trainer.start_epoch!=4 or int(trainer.epoch)+1!=24:raise ValueError('Incomplete native resume')
        native.write_new(out/('rank_%d_complete.json'%rank),dict(passed=True,
            start_epoch=4,completed_epochs=24,resumed_optimizer_updates=updates[0],resumed_microbatches=batches[0]))
    except BaseException as error:
        native.atomic(out/('rank_%d_failure.json'%rank),dict(error=repr(error)));raise


def controller(out):
    p=native.read(out/'protocol.json');verify_protocol(p);cfg=p['resume_config'];last_good=p['before']
    errors=[]
    def error(item):
        errors.append(item)
        with (out/'gpu_query_errors.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
    env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0,1',
        PYTHONPATH=str(ROOT),PYTORCH_CUDA_ALLOC_CONF=cfg['allocator_config'],
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true',YOLO_CONFIG_DIR=str(out/'settings'),WANDB_DISABLED='true')
    command=[str(ROOT/'.venvs/yolo26_v1/bin/python'),'-m','torch.distributed.run',
        '--standalone','--nproc_per_node=2',str(Path(__file__).resolve()),'--stage','worker','--out',str(out)]
    reason=None;start=time.monotonic();peaks={'0':0,'1':0}
    with (out/'native_train.log').open('x') as log,(out/'gpu_samples.jsonl').open('x') as samples:
        child=subprocess.Popen(command,cwd=str(ROOT),env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        native.write_new(out/'worker_launcher.json',dict(pid=child.pid,command=command))
        try:
            while child.poll() is None:
                last_good=retry_query(lambda:snapshot(cfg),cfg['gpu_query_attempts'],cfg['gpu_query_retry_seconds'],on_error=error)
                pids=native.worker_pids(out)
                for g,row in last_good['gpus'].items():
                    if row['uuid']!=p['expected_GPU_UUIDs'][str(g)]:raise RuntimeError('GPU_IDENTITY_CHANGED')
                    own=sum(a['used_mib'] for a in last_good['apps'] if a['pid'] in pids and a['uuid']==row['uuid'])
                    peaks[str(g)]=max(peaks[str(g)],own)
                    if own>cfg['driver_mib'] or row['free_mib']<cfg['minimum_free_mib']:raise RuntimeError('MEMORY_GUARD')
                    if time.monotonic()-start>15 and any(a['uuid']==row['uuid'] and a['pid'] not in pids for a in last_good['apps']):raise RuntimeError('GPU_CONTENTION')
                samples.write(json.dumps(dict(seconds=time.monotonic()-start,snapshot=last_good))+'\n');samples.flush()
                native.atomic(out/'status.json',dict(state='RUNNING',source_checkpoint=p['resume_input'],peaks=peaks,gpu_query_errors=len(errors)))
                time.sleep(2)
        except BaseException as exc:
            reason=repr(exc);native.stop_group(child,out)
        code=child.wait()
    reports=[native.read(f) for f in out.glob('rank_*_complete.json')]
    passed=code==0 and reason is None and len(reports)==2 and all(r['passed'] for r in reports)
    if passed:
        try:
            verify_protocol(p)
            if not all((out/'train/weights'/n).exists() for n in ('epoch15.pt','epoch17.pt','epoch19.pt','epoch21.pt','epoch23.pt')):raise ValueError('Missing fixed candidates')
        except BaseException as exc:passed=False;reason=repr(exc)
    # Reuse the last successful observation: do not let another failed query
    # destroy the completion/failure receipt, as happened in the original run.
    result=dict(passed=passed,state='COMPLETED' if passed else 'STOPPED_OR_FAILED',exit_code=code,
        reason=reason,elapsed_seconds=time.monotonic()-start,peaks=peaks,last_successful_snapshot=last_good,
        gpu_query_errors=len(errors),automatic_training_retry=False,auto_TEST=False,workers=reports)
    native.write_new(out/'completion.json',result);native.atomic(out/'status.json',result)


def launch(out,cfg):
    report=audit(cfg);before=retry_query(lambda:snapshot(cfg),cfg['gpu_query_attempts'],cfg['gpu_query_retry_seconds'])
    if any(r['free_mib']<7168 for r in before['gpus'].values()) or before['apps']:raise RuntimeError('GPU0/1 must be idle')
    if sorted(before['gpus'],key=lambda g:before['gpus'][g]['pci'])!=[0,1]:raise ValueError('PCI order differs')
    # Recheck all TRAIN/VAL bytes; no TEST access or selection.
    for row in native.dataset_manifest():
        if row['split']=='test':continue
        for k in ('images','annfiles'):
            if native.sha(native.DATA/row[k]['path'])!=row[k]['sha256']:raise ValueError('TRAIN/VAL bytes changed')
    converted=ROOT/native.read(ROOT/native.YOLO_CONFIG)['converted_data']
    conversion=native.read(converted/'conversion_receipt.json')
    if (conversion['manifest_sha256']!=native.MANIFEST_SHA or conversion['test_included']
            or conversion['counts']!=dict(train=2558,val=887)):
        raise ValueError('Converted TRAIN/VAL identity changed')
    for row in conversion['records']:
        if (native.sha(converted/row['label'])!=row['label_sha256']
                or native.sha(converted/row['image'])!=row['image_sha256']):
            raise ValueError('Converted TRAIN/VAL bytes changed')
    if ROOT/'work_dirs' not in out.parents:raise ValueError('Use a new work_dirs output')
    out.mkdir(parents=True,exist_ok=False);(out/'input').mkdir();(out/'train').mkdir()
    resume_input=out/'input/last.pt';shutil.copyfile(report['checkpoint'],resume_input)
    if native.sha(resume_input)!=cfg['checkpoint_sha256']:raise ValueError('Resume input copy differs')
    shutil.copyfile(report['original_results_csv'],out/'train/results.csv')
    parent=native.read(ROOT/cfg['parent_run']/'protocol.json')
    sources=dict(parent['sources'])
    for n in (CONFIG,'crane_project/tools/resume_yolo26_native_v1.py','crane_project/tools/prepare_external_obb_v1.py'):
        sources[n]=native.sha(ROOT/n)
    protocol=dict(protocol='native_YOLO_resume_v1',resume_config=cfg,checkpoint_audit=report,
        resume_input=str(resume_input),sources=sources,initialization=parent['initialization'],
        checkpoint_selection=parent['checkpoint_selection'],result_requirements=parent['result_requirements'],
        expected_GPU_UUIDs={str(g):r['uuid'] for g,r in before['gpus'].items()},before=before,
        prepared_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat(),auto_TEST=False)
    native.write_new(out/'protocol.json',protocol)
    with (out/'supervisor.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--stage','controller','--out',str(out)],
            cwd=str(ROOT),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(pid=child.pid,next_epoch=5,total_epochs=24,out=str(out),auto_TEST=False)
    native.write_new(out/'launch.json',receipt);return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('check','launch','controller','worker'),default='check')
    p.add_argument('--out',type=Path);p.add_argument('--report',type=Path)
    a=p.parse_args();os.chdir(ROOT)
    if a.stage not in ('worker',):os.environ['CUDA_VISIBLE_DEVICES']=''
    cfg=native.read(ROOT/CONFIG)
    if a.stage=='check':
        report=audit(cfg)
        if a.report:a.report.parent.mkdir(parents=True,exist_ok=True);native.write_new(a.report,report)
        print(json.dumps(report,indent=2));return
    if a.out is None:p.error('--out required for explicit execution')
    out=a.out.resolve()
    if a.stage=='worker':return worker(out)
    if a.stage=='controller':return controller(out)
    print(json.dumps(launch(out,cfg),indent=2))


if __name__=='__main__':main()
