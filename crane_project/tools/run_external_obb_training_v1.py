"""Queue guarded native GauCho then YOLO26 training on physical GPUs 0/1.

This orchestrates existing native trainers, identities and resource monitoring.
No model, loss, assignment, schedule or optimizer math is reimplemented.
No automatic resume/retry, offline selection or TEST.
"""
import argparse
from datetime import datetime, timezone, timedelta
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools.prepare_external_obb_v1 import (
    DATA, GAUCHO_CONFIG, YOLO_CONFIG, MANIFEST_SHA, check_sources,
    check_configs, dataset_manifest, read, sha, write_new)
from crane_project.tools.smoke_faa_gpu_v1 import gpu_rows, processes, worker_pids, stop_group

OBSERVED_CONFIG = 'configs/gaucho/crane_gaucho_fcos_r50_probiou_gpu01_observed_v1.py'
GPUS = (0, 1)


def atomic(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+'.%d.tmp' % os.getpid())
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(str(tmp), str(path))


def assert_sources(protocol):
    for name, digest in protocol['sources'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Source drift: '+name)
    for name, digest in protocol['initialization'].items():
        if sha(name) != digest:
            raise ValueError('Initialization drift: '+name)


def idle_gpus():
    rows = {r['index']: r for r in gpu_rows() if r['index'] in GPUS}
    if set(rows) != set(GPUS) or any(r['free_mib'] < 7168 for r in rows.values()):
        raise RuntimeError('Physical GPU0/1 need at least 7168 MiB free each')
    if any(p['uuid'] in {r['uuid'] for r in rows.values()} for p in processes()):
        raise RuntimeError('Another compute process is using GPU0/1')
    return rows


def prepare(out):
    check_sources()
    check_configs()
    from crane_project.tools.check_faa_adaptation_v1 import resolve
    cfg, closure = resolve(ROOT/OBSERVED_CONFIG)
    base, _ = resolve(ROOT/GAUCHO_CONFIG)
    for key in ('model', 'data', 'optimizer', 'optimizer_config', 'runner', 'lr_config',
                'checkpoint_config', 'evaluation', 'find_unused_parameters'):
        if cfg[key] != base[key]:
            raise ValueError('Observer changes training recipe: '+key)
    if not read(ROOT/'work_dirs/external_obb_smoke_20261104_v1/gaucho/attempt_001/result.json')['passed']:
        raise ValueError('GauCho trial did not pass')
    if not read(ROOT/'work_dirs/external_obb_smoke_20261105_v1/yolo26/attempt_001/result.json')['passed']:
        raise ValueError('YOLO trial did not pass')
    before = idle_gpus()
    verified = 0
    for row in dataset_manifest():
        if row['split'] == 'test':
            continue
        for kind in ('images', 'annfiles'):
            if sha(DATA/row[kind]['path']) != row[kind]['sha256']:
                raise ValueError('TRAIN/VAL bytes differ: '+row['id'])
            verified += 1
    conversion = read(ROOT/read(ROOT/YOLO_CONFIG)['converted_data']/'conversion_receipt.json')
    if conversion['manifest_sha256'] != MANIFEST_SHA or conversion['test_included']:
        raise ValueError('YOLO derived dataset identity differs')
    if conversion['counts'] != dict(train=2558, val=887):
        raise ValueError('YOLO derived dataset counts differ')
    for r in conversion['records']:
        folder = ROOT/read(ROOT/YOLO_CONFIG)['converted_data']
        if sha(folder/r['label']) != r['label_sha256'] or sha(folder/r['image']) != r['image_sha256']:
            raise ValueError('YOLO converted data changed')
    initialization = {}
    resnet = Path.home()/'.cache/torch/hub/checkpoints/resnet50-0676ba61.pth'
    if sha(resnet) != '0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a':
        raise ValueError('ResNet50 cache differs from FAA')
    initialization[str(resnet)] = sha(resnet)
    yolo = read(ROOT/YOLO_CONFIG)
    init = ROOT/yolo['initialization']['checkpoint']
    receipt = read(init.with_suffix('.initialization.json'))
    if receipt['source_sha256'] != '9f6546f33a70d910e2cd6dcca5265c4617b5670b19a1f287c39e99258bade01a':
        raise ValueError('ImageNet classification source differs')
    if sha(init) != receipt['checkpoint_sha256']:
        raise ValueError('YOLO derived initialization differs')
    initialization[str(init)] = sha(init)
    initialization[str(init.with_suffix('.initialization.json'))] = sha(init.with_suffix('.initialization.json'))
    names = set(closure)
    for name in ('gaucho', 'yolo26'):
        path = 'configs/%s/upstream_manifest.json'%name
        spec = read(ROOT/path)
        names.update(spec['files'])
        names.update(spec.get('shared_dependencies', {}))
        names.add(path)
    names.update(str(p.relative_to(ROOT)) for p in (ROOT/'mmrotate').rglob('*.py'))
    names.update([GAUCHO_CONFIG, YOLO_CONFIG, 'configs/yolo26/crane_yolo26m_obb_train_v1.yaml',
        'configs/yolo26/crane_grab_port_day2night_v1.yaml','configs/yolo26/yolo26m-obb.yaml',
        'configs/gaucho/experiment.json', 'tools/train.py','mmrotate/apis/train.py',
        'crane_project/tools/ckpt_sweep.py','crane_project/tools/smoke_faa_gpu_v1.py',
        'crane_project/tools/run_external_obb_training_v1.py',
        'crane_project/utils/external_training_observer_v1.py'])
    sources = {name: sha(ROOT/name) for name in sorted(names)}
    ordered = sorted(gpu_rows(), key=lambda r:r['pci'])
    ordinals = [next(i for i,r in enumerate(ordered) if r['uuid']==before[g]['uuid']) for g in GPUS]
    # Native YOLO selects relative device 0/1. Physical and PCI ordinals must
    # agree here, unlike a UUID-valued CUDA_VISIBLE_DEVICES on old Torch.
    if ordinals != [0,1]:
        raise ValueError('Unexpected GPU0/1 PCI ordering')
    out.mkdir(parents=True, exist_ok=False)
    from crane_project.utils.crane_frozen_pipeline_v1 import read_constants
    selection = read_constants(ROOT/'crane_project/tools/ckpt_sweep.py', ['SELECTION_CONFIG'])['SELECTION_CONFIG']
    write_new(out/'protocol.json', dict(protocol='native_external_formal_queue_v1',
        created_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        physical_gpus=list(GPUS), ordinals=ordinals, before=before,
        queue=['gaucho','yolo26'], sequential=True, sources=sources,
        initialization=initialization, manifest_sha256=MANIFEST_SHA,
        train_frames=2558,val_frames=887,verified_train_val_files=verified,
        gaucho=dict(config=OBSERVED_CONFIG,epochs=24,samples_per_gpu=1,global_batch=2,
                    optimizer=cfg['optimizer'],candidates=[16,18,20,22,24]),
        yolo26=dict(config='configs/yolo26/crane_yolo26m_obb_train_v1.yaml',epochs=24,
                    batch=2,nbs=4,optimizer_auto_expected='MuSGD',
                    candidates=[16,18,20,22,24],native_names=['epoch15.pt','epoch17.pt','epoch19.pt','epoch21.pt','epoch23.pt']),
        checkpoint_selection=dict(rule=selection,automatic_sweep=False,split='val'),
        dtype='FP32',seed=0,allocator_mib=6144,driver_mib=6656,
        minimum_free_mib=1536,auto_resume=False,auto_restart=False,auto_TEST=False))


def run_one(model_name, out, protocol):
    assert_sources(protocol)
    idle_gpus()
    folder = out/model_name
    folder.mkdir(exist_ok=False)
    python = (ROOT/'.venvs/yolo26_v1/bin/python' if model_name=='yolo26' else Path(sys.executable))
    env = dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0,1',
        EXTERNAL_RUN_DIR=str(folder),EXTERNAL_QUEUE_DIR=str(out),PYTHONPATH=str(ROOT),
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true',WANDB_DISABLED='true',
        YOLO_CONFIG_DIR=str(folder/'settings'),PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128',
        TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
    command = [str(python),'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
        str(Path(__file__).resolve()),'--stage','worker','--model',model_name,'--out',str(folder)]
    start = time.monotonic()
    peaks = {str(g):dict(process_mib=0,total_used_mib=0,minimum_free_mib=8192) for g in GPUS}
    reason = None
    with (folder/'native_train.log').open('x') as log, (folder/'gpu_samples.jsonl').open('x') as sample:
        child = subprocess.Popen(command,cwd=str(ROOT),env=env,stdin=subprocess.DEVNULL,
                                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write_new(folder/'launcher.json',dict(pid=child.pid,command=command))
        try:
            while child.poll() is None:
                rows = {r['index']:r for r in gpu_rows() if r['index'] in GPUS}
                pids = worker_pids(folder)
                apps = processes()
                owned = [a for a in apps if a['pid'] in pids]
                allowed = {r['uuid'] for r in rows.values()}
                if any(a['uuid'] not in allowed for a in owned):
                    reason = 'WRONG_GPU'
                if time.monotonic()-start>15 and any(a['uuid'] in allowed and a['pid'] not in pids for a in apps):
                    reason = 'OTHER_PROCESS_ON_GPU01'
                for g,row in rows.items():
                    memory = sum(a['used_mib'] for a in owned if a['uuid']==row['uuid'])
                    peak=peaks[str(g)]
                    peak['process_mib']=max(peak['process_mib'],memory)
                    peak['total_used_mib']=max(peak['total_used_mib'],row['used_mib'])
                    peak['minimum_free_mib']=min(peak['minimum_free_mib'],row['free_mib'])
                    if memory>6656 or row['free_mib']<1536:
                        reason='GPU%d_MEMORY_GUARD'%g
                item=dict(unix_time=time.time(),elapsed_seconds=time.monotonic()-start,gpus=rows,owned_processes=owned)
                sample.write(json.dumps(item)+'\n');sample.flush()
                heartbeats={p.stem:read(p) for p in folder.glob('rank_*_heartbeat.json')}
                atomic(folder/'status.json',dict(state='RUNNING',model=model_name,peaks=peaks,
                    latest=item,heartbeats=heartbeats,auto_restart=False,TEST=False))
                if reason:
                    stop_group(child,folder)
                    break
                time.sleep(2)
        except BaseException as e:
            reason='MONITOR_ERROR:'+repr(e)
            stop_group(child,folder)
        code=child.wait()
    reports=[read(p) for p in sorted(folder.glob('rank_*_complete.json'))]
    passed=code==0 and reason is None and len(reports)==2 and all(r['passed'] for r in reports)
    if passed:
        assert_sources(protocol)
        required=([folder/'train'/('epoch_%d.pth'%e) for e in protocol['gaucho']['candidates']]
                  if model_name=='gaucho' else [folder/'train/weights'/name for name in protocol['yolo26']['native_names']])
        if not all(p.is_file() for p in required):
            passed=False;reason='MISSING_CANDIDATE_CHECKPOINT'
    result=dict(passed=passed,state='COMPLETED' if passed else 'STOPPED_OR_FAILED',
        exit_code=code,reason=reason,elapsed_seconds=time.monotonic()-start,peaks=peaks,
        workers=reports,auto_restart=False,auto_TEST=False,after=gpu_rows())
    write_new(folder/'completion.json',result);atomic(folder/'status.json',result)
    return passed


def queue(out):
    protocol=read(out/'protocol.json')
    try:
        for model_name in protocol['queue']:
            atomic(out/'status.json',dict(state='RUNNING',active_model=model_name,auto_TEST=False))
            if not run_one(model_name,out,protocol):
                raise RuntimeError(model_name+' stopped. Queue will not continue or retry.')
        atomic(out/'status.json',dict(state='COMPLETED',auto_TEST=False))
    except BaseException as error:
        atomic(out/'status.json',dict(state='STOPPED_OR_FAILED',error=repr(error),auto_restart=False,auto_TEST=False))
        raise


def worker(model_name,folder):
    protocol=read(Path(os.environ['EXTERNAL_QUEUE_DIR'])/'protocol.json')
    assert_sources(protocol)
    rank=int(os.environ['LOCAL_RANK'])
    write_new(folder/('rank_%d_identity.json'%rank),dict(pid=os.getpid(),rank=rank,physical_gpu=GPUS[rank]))
    import torch
    torch.cuda.set_device(rank)
    torch.cuda.set_per_process_memory_fraction(6144*2**20/torch.cuda.get_device_properties(rank).total_memory,rank)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.cuda.reset_peak_memory_stats(rank)
    try:
        if model_name=='gaucho':
            sys.argv=['tools/train.py',str(ROOT/OBSERVED_CONFIG),'--launcher','pytorch',
                      '--seed','0','--deterministic','--work-dir',str(folder/'train')]
            runpy.run_path(str(ROOT/'tools/train.py'),run_name='__main__')
        else:
            sys.path.insert(0,str(ROOT/'third_party/ultralytics'))
            from ultralytics.models.yolo.obb.train import OBBTrainer
            from ultralytics.utils import YAML,callbacks
            callbacks.add_integration_callbacks=lambda instance:None
            args=YAML.load(str(ROOT/'configs/yolo26/crane_yolo26m_obb_train_v1.yaml'))
            args.pop('mode',None)
            args.update(project=str(folder),name='train',exist_ok=False,
                        model=str(ROOT/args['model']),data=str(ROOT/args['data']),
                        pretrained=str(ROOT/args['pretrained']))
            trainer=OBBTrainer(overrides=args)
            updates=[0]
            batches=[0]
            def write(t,state):
                items=({k:float(v.detach().cpu()) for k,v in t.loss_items.items()}
                       if isinstance(getattr(t,'loss_items',None),dict) else {})
                atomic(folder/('rank_%d_heartbeat.json'%rank),dict(state=state,rank=rank,
                    pid=os.getpid(),epoch=int(t.epoch)+1,max_epochs=t.epochs,
                    completed_microbatches=batches[0],optimizer_updates=updates[0],loss_terms=items,
                    peak_allocated_mib=torch.cuda.max_memory_allocated(rank)/2**20,
                    peak_reserved_mib=torch.cuda.max_memory_reserved(rank)/2**20,
                    optimizer=t.optimizer.__class__.__name__,optimizer_lr0=t.args.lr0))
            def initialized(t):
                step=t.optimizer.step
                def counted(*a,**kw):
                    result=step(*a,**kw);updates[0]+=1;return result
                t.optimizer.step=counted
                write(t,'INITIALIZED')
            def batch_end(t):
                batches[0]+=1
                if not math.isfinite(float(t.loss.detach())):
                    raise RuntimeError('Nonfinite YOLO loss. Stop without recovery/retry.')
                if batches[0]<=10 or batches[0]%10==0:write(t,'TRAINING')
            trainer.add_callback('on_pretrain_routine_end',initialized)
            trainer.add_callback('on_train_batch_end',batch_end)
            trainer.add_callback('on_train_epoch_end',lambda t:write(t,'EPOCH_FINISHED'))
            trainer.add_callback('on_val_start',lambda t:None)
            trainer.train()
            if int(trainer.epoch)+1!=protocol['yolo26']['epochs'] or trainer.start_epoch!=0:
                raise RuntimeError('YOLO completed epoch identity differs')
            write_new(folder/('rank_%d_complete.json'%rank),dict(passed=True,completed_epochs=24,
                optimizer_updates=updates[0],microbatches=batches[0],rank=rank))
    except BaseException as error:
        atomic(folder/('rank_%d_failure.json'%rank),dict(error=repr(error),rank=rank))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('launch','queue','worker'),required=True)
    parser.add_argument('--model',choices=('gaucho','yolo26'))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();os.chdir(ROOT);out=args.out.resolve()
    if args.stage=='worker':return worker(args.model,out)
    if args.stage=='queue':return queue(out)
    prepare(out)
    with (out/'supervisor.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--stage','queue','--out',str(out)],
            cwd=str(ROOT),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    value=dict(supervisor_pid=child.pid,queue=['gaucho','yolo26'],physical_gpus=list(GPUS),out=str(out),auto_TEST=False)
    write_new(out/'launch.json',value);print(json.dumps(value,indent=2))


if __name__=='__main__':main()
