"""Run the paired GauCho no-scale control using the existing native trainer."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import run_external_obb_training_v1 as previous
from crane_project.tools.prepare_external_obb_v1 import (
    DATA, MANIFEST_SHA, check_sources, dataset_manifest, read, sha, write_new)
from crane_project.tools.check_faa_adaptation_v1 import resolve
from crane_project.tools.smoke_faa_gpu_v1 import gpu_rows, processes, worker_pids, stop_group

OBSERVED_CONFIG = 'configs/gaucho/crane_gaucho_fcos_r50_probiou_gpu01_no_port_shrink_v1.py'
REFERENCE_CONFIG = 'configs/gaucho/crane_gaucho_fcos_r50_probiou_gpu01_observed_v1.py'
GPUS = (2, 3)
previous.GPUS = GPUS
previous.OBSERVED_CONFIG = OBSERVED_CONFIG
atomic = previous.atomic
assert_sources = previous.assert_sources
idle_gpus = previous.idle_gpus


def prepare(out):
    check_sources()
    cfg, closure = resolve(ROOT/OBSERVED_CONFIG)
    reference, _ = resolve(ROOT/REFERENCE_CONFIG)
    expected = deepcopy(reference)
    expected['work_dir'] = cfg['work_dir']

    def without_scale(pipeline):
        result = [step for step in pipeline if step['type'] != 'PortIsotropicShrink']
        if len(result) != len(pipeline)-1:
            raise ValueError('Expected exactly one scale operation')
        return result

    expected['train_pipeline'] = without_scale(expected['train_pipeline'])
    for dataset in expected['data']['train']:
        dataset['pipeline'] = without_scale(dataset['pipeline'])
    if cfg != expected:
        raise ValueError('Unexpected difference from the paired GauCho config')
    from mmcv import Config
    if dict(Config.fromfile(str(ROOT/OBSERVED_CONFIG), import_custom_modules=False).items()) != expected:
        raise ValueError('Native MMCV resolution differs')
    if cfg['seed'] != 0 or cfg['runner']['max_epochs'] != 24:
        raise ValueError('Training budget or seed differs')
    if cfg['load_from'] is not None or cfg['resume_from'] is not None:
        raise ValueError('Fresh training required')
    if (ROOT/cfg['work_dir']).resolve() != out/'gaucho/train':
        raise ValueError('Output directory differs from paired-control config')
    if not read(ROOT/'work_dirs/external_obb_smoke_20261104_v1/gaucho/attempt_001/result.json')['passed']:
        raise ValueError('Original GauCho engineering check required')
    before = idle_gpus()
    verified = 0
    for row in dataset_manifest():
        if row['split'] == 'test':
            continue
        for kind in ('images','annfiles'):
            if sha(DATA/row[kind]['path']) != row[kind]['sha256']:
                raise ValueError('TRAIN/VAL bytes changed: '+row['id'])
            verified += 1
    if verified != 6890:
        raise ValueError('Incomplete TRAIN/VAL check')
    resnet = Path.home()/'.cache/torch/hub/checkpoints/resnet50-0676ba61.pth'
    expected_sha = '0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a'
    if sha(resnet) != expected_sha:
        raise ValueError('ImageNet initialization changed')
    names = set(closure)
    provenance = 'configs/gaucho/upstream_manifest.json'
    spec = read(ROOT/provenance)
    names.update(spec['files'])
    names.update(spec.get('shared_dependencies',{}))
    names.update(str(p.relative_to(ROOT)) for p in (ROOT/'mmrotate').rglob('*.py'))
    names.update([provenance, 'tools/train.py', 'mmrotate/apis/train.py',
        'crane_project/tools/run_external_obb_training_v1.py',
        'crane_project/tools/run_gaucho_no_port_shrink_v1.py',
        'crane_project/tools/prepare_external_obb_v1.py',
        'crane_project/tools/check_faa_adaptation_v1.py',
        'crane_project/tools/ckpt_sweep.py',
        'crane_project/tools/smoke_faa_gpu_v1.py',
        'crane_project/utils/external_training_observer_v1.py'])
    sources = {name:sha(ROOT/name) for name in sorted(names)}
    ordered = sorted(gpu_rows(), key=lambda row:row['pci'])
    ordinals = [next(i for i,row in enumerate(ordered) if row['uuid']==before[g]['uuid']) for g in GPUS]
    if ordinals != [2,3]:
        raise ValueError('Unexpected GPU PCI ordering')
    out.mkdir(parents=True, exist_ok=False)
    from crane_project.utils.crane_frozen_pipeline_v1 import read_constants
    selection = read_constants(ROOT/'crane_project/tools/ckpt_sweep.py',['SELECTION_CONFIG'])['SELECTION_CONFIG']
    write_new(out/'protocol.json',dict(protocol='gaucho_no_port_scale_train_v1',
        created_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        physical_gpus=list(GPUS),ordinals=ordinals,before=before,queue=['gaucho'],
        sources=sources,initialization={str(resnet):expected_sha},manifest_sha256=MANIFEST_SHA,
        train_frames=2558,val_frames=887,verified_train_val_files=verified,
        reference_config=REFERENCE_CONFIG,paired_difference='remove PortIsotropicShrink',
        gaucho=dict(config=OBSERVED_CONFIG,epochs=24,samples_per_gpu=1,global_batch=2,
            optimizer=cfg['optimizer'],lr_config=cfg['lr_config'],candidates=[16,18,20,22,24]),
        checkpoint_selection=dict(rule=selection,automatic_sweep=False,split='val'),
        dtype='FP32',seed=0,allocator_mib=6144,driver_mib=6656,minimum_free_mib=1536,
        auto_resume=False,auto_restart=False,auto_TEST=False))


def run_one(model_name, out, protocol):
    assert_sources(protocol)
    idle_gpus()
    folder = out/model_name
    folder.mkdir(exist_ok=False)
    python = (ROOT/'.venvs/yolo26_v1/bin/python' if model_name=='yolo26' else Path(sys.executable))
    env = dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=','.join(map(str,protocol['ordinals'])),
        EXTERNAL_RUN_DIR=str(folder),EXTERNAL_QUEUE_DIR=str(out),PYTHONPATH=str(ROOT),
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true',WANDB_DISABLED='true',
        YOLO_CONFIG_DIR=str(folder/'settings'),
        PYTORCH_CUDA_ALLOC_CONF=protocol.get('allocator_config','max_split_size_mb:128'),
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
                    if memory>protocol['driver_mib'] or row['free_mib']<protocol['minimum_free_mib']:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('launch','queue','worker'), required=True)
    parser.add_argument('--model', choices=('gaucho',))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    out = args.out.resolve()
    if args.stage == 'worker':
        return previous.worker('gaucho',out)
    if args.stage == 'queue':
        return queue(out)
    prepare(out)
    with (out/'supervisor.log').open('x') as log:
        child = subprocess.Popen([sys.executable,str(Path(__file__).resolve()),
            '--stage','queue','--out',str(out)],cwd=str(ROOT),stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt = dict(supervisor_pid=child.pid,queue=['gaucho'],physical_gpus=list(GPUS),
        out=str(out),auto_TEST=False)
    write_new(out/'launch.json',receipt)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__ == '__main__':
    main()
