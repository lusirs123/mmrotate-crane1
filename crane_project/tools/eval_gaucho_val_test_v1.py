"""Fixed five-epoch VAL selection, one accuracy TEST, then common efficiency.

Only physical GPU0/1, one process per card. Common scoring/selection is reused.
No training, retries, threshold changes, expanded candidates or TEST selection.
"""
import argparse
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools.prepare_external_obb_v1 import (
    DATA, MANIFEST_SHA, read, sha, write_new, check_sources, check_configs)
from crane_project.tools.export_external_obb_v1 import config_identity, validate_test_lock
from crane_project.tools.smoke_faa_gpu_v1 import gpu_rows, processes

TRAIN = ROOT/'work_dirs/external_obb_formal_20261105_v1/attempt_001'
EPOCHS = (16,18,20,22,24)
TEST_MARKER = TRAIN/'gaucho/TEST_execution.json'


def static_check():
    """Local no-GPU check. Does not open weights or use remote paths."""
    sources = check_sources()
    check_configs()
    cfg = read(ROOT/'configs/gaucho/experiment.json')
    if cfg['selection']['candidate_completed_epochs'] != list(EPOCHS):
        raise ValueError('GauCho candidate set differs from the fixed five epochs')
    if cfg['selection']['native_checkpoint_names'] != ['epoch_%d.pth'%e for e in EPOCHS]:
        raise ValueError('GauCho checkpoint naming differs')
    return dict(passed=True, candidates=list(EPOCHS),physical_gpus=[0,1],
                VAL_frames=887,TEST_frames=1440,model='gaucho',sources=sources,
                configuration=config_identity('gaucho'),
                remote_contacted=False,GPU_used=False,training_updates=0,
                inference_runs=0,TEST_after_VAL_lock=True,efficiency_required=True)


def reserve_test(path, identity):
    """Exclusive per-training-run marker prevents an automatic second TEST."""
    write_new(path, dict(identity, state='RESERVED_BEFORE_INFERENCE',
        reserved_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        automatic_retry=False))


def status(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(str(tmp), str(path))


def checkpoint(epoch):
    return TRAIN/'gaucho/train'/('epoch_%d.pth'%epoch)


def prepare(out):
    static_check()
    if TEST_MARKER.exists():
        raise FileExistsError('TEST already reserved for this training run. Inspect the saved run instead of repeating it.')
    done = read(TRAIN/'gaucho/completion.json')
    if not done['passed'] or done['exit_code'] != 0:
        raise ValueError('GauCho training is incomplete')
    train = read(TRAIN/'protocol.json')
    for name, digest in train['sources'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Training source changed: '+name)
    for name, digest in train['initialization'].items():
        if sha(name) != digest:
            raise ValueError('Training initialization changed')
    check_sources()
    check_configs()
    config = config_identity('gaucho')
    rows = {r['index']:r for r in gpu_rows() if r['index'] in (0,1)}
    if set(rows) != {0,1} or any(r['free_mib']<7168 for r in rows.values()):
        raise ValueError('GPU0/1 must each have at least 7168 MiB free')
    if any(p['uuid'] in {r['uuid'] for r in rows.values()} for p in processes()):
        raise ValueError('Another compute process is on GPU0/1')
    ordered = sorted(gpu_rows(), key=lambda r:r['pci'])
    if [ordered[i]['uuid'] for i in (0,1)] != [rows[i]['uuid'] for i in (0,1)]:
        raise ValueError('Unexpected PCI order')
    from crane_project.tools.ckpt_sweep import SELECTION_CONFIG
    # Check the common TEST reference now, before any future TEST exposure.
    from crane_project.utils.crane_frozen_pipeline_v1 import read_constants
    reference = read_constants(ROOT/'crane_project/tools/eval_faa_val_test_v1.py', ['REF','REF_SHA'])
    if sha(ROOT/reference['REF']) != reference['REF_SHA']:
        raise ValueError('Shared frozen TEST reference is missing or changed')
    out.mkdir(parents=True, exist_ok=False)
    (out/'val').mkdir()
    write_new(out/'protocol.json', dict(
        protocol='gaucho_fixed_VAL_then_one_TEST_v1',
        created_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        training_completion_sha256=sha(TRAIN/'gaucho/completion.json'),
        training_protocol_sha256=sha(TRAIN/'protocol.json'), candidates=list(EPOCHS),
        weights={str(e):sha(checkpoint(e)) for e in EPOCHS},
        configuration=config, selection_config=SELECTION_CONFIG,
        training_sources=train['sources'],
        manifest_sha256=MANIFEST_SHA,
        test_reference=reference,
        physical_gpus=[0,1], before=rows,
        extra_sources={name:sha(ROOT/name) for name in (
            'crane_project/tools/eval_gaucho_val_test_v1.py',
            'crane_project/tools/prepare_external_obb_v1.py',
            'crane_project/tools/benchmark_external_obb_v1.py',
            'crane_project/tools/benchmark_crane_final_models_v1.py',
            'crane_project/tools/benchmark_port_detection_runtime_v1.py',
            'crane_project/utils/port_detection_runtime_v1.py',
            'configs/external_obb_efficiency_v1.json')},
        allocator_mib=6144,driver_mib=6656,minimum_free_mib=1536,
        automatic_retry=False, training_updates=0, TEST_after_VAL_lock=True,
        TEST_runs=1, TEST_reselection=False))


def verify(protocol):
    if config_identity('gaucho') != protocol['configuration']:
        raise ValueError('Evaluation configuration/source changed')
    for name,digest in protocol['extra_sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Controller source drift')
    for name,digest in protocol['training_sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Trained model source drift: '+name)
    if sha(DATA/'manifest.json') != protocol['manifest_sha256']:
        raise ValueError('Dataset manifest changed')


def stop(child):
    try:os.killpg(child.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        child.wait(timeout=10)


def infer(out, jobs, protocol):
    """Independent epochs can occupy GPU0 and GPU1 concurrently."""
    if (not 1 <= len(jobs) <= 2 or len({j[2] for j in jobs}) != len(jobs)
            or any(s not in ('val','test') or e not in EPOCHS or g not in (0,1) for s,e,g in jobs)
            or len({j[0] for j in jobs}) != 1):
        raise ValueError('Invalid split, candidate set or duplicate GPU assignment')
    if jobs[0][0] == 'test':
        lock=read(out/'selection/selected_identity.json')
        if len(jobs)!=1 or jobs[0][1]!=lock['selected_completed_epoch']:
            raise ValueError('Only the single VAL-selected checkpoint may enter TEST')
    children=[]
    started=time.monotonic()
    peaks={str(g):dict(process_mib=0,minimum_free_mib=8192) for _,_,g in jobs}
    env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONPATH=str(ROOT),
             OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    reason=None
    try:
        for split,epoch,gpu in jobs:
            if sha(checkpoint(epoch))!=protocol['weights'][str(epoch)]:
                raise ValueError('Checkpoint changed before inference')
            directory=out/split/('epoch_%d'%epoch)
            if directory.exists():raise FileExistsError(directory)
            command=[sys.executable,str(ROOT/'crane_project/tools/export_external_obb_v1.py'),
                '--model','gaucho','--split',split,'--checkpoint',str(checkpoint(epoch)),
                '--out',str(directory),'--device','cuda:0','--physical-gpus','0','1','--execute']
            if split=='test':command+=['--selection-lock',str(out/'selection/selected_identity.json')]
            log=(out/(split+'_epoch_%d.log'%epoch)).open('x')
            child=subprocess.Popen(command,cwd=str(ROOT),env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append((child,log,gpu,directory))
        with (out/('gpu_%s_%s.jsonl'%(jobs[0][0],jobs[0][1]))).open('x') as samples:
            while any(c.poll() is None for c,_,_,_ in children):
                rows={r['index']:r for r in gpu_rows() if str(r['index']) in peaks}
                if {str(g) for g in rows} != set(peaks):
                    raise RuntimeError('Assigned GPU inventory changed')
                app=processes()
                own=[a for a in app if a['pid'] in {c.pid for c,_,_,_ in children}]
                for g,row in rows.items():
                    used=sum(a['used_mib'] for a in own if a['uuid']==row['uuid'])
                    peaks[str(g)]['process_mib']=max(peaks[str(g)]['process_mib'],used)
                    peaks[str(g)]['minimum_free_mib']=min(peaks[str(g)]['minimum_free_mib'],row['free_mib'])
                    if used>6656 or row['free_mib']<1536:reason='MEMORY_GUARD'
                    if any(a['uuid']==row['uuid'] and a['pid'] not in {c.pid for c,_,_,_ in children} for a in app):
                        reason='OTHER_GPU_PROCESS'
                if time.monotonic()-started>1800:reason='INFERENCE_TIMEOUT'
                if any(c.poll() not in (None,0) for c,_,_,_ in children):reason='INFERENCE_FAILED'
                progress={str(d.relative_to(out)):read(d/'progress.json') for _,_,_,d in children if (d/'progress.json').exists()}
                item=dict(seconds=time.monotonic()-started,gpus=rows,owned_processes=own,progress=progress)
                samples.write(json.dumps(item)+'\n');samples.flush()
                status(out/'status.json',dict(state='INFERENCE',jobs=jobs,progress=progress,peaks=peaks))
                if reason:raise RuntimeError(reason)
                time.sleep(2)
        if any(c.wait()!=0 for c,_,_,_ in children):raise RuntimeError('Inference failed')
        for _,_,_,directory in children:
            receipt=read(directory/'export_receipt.json')
            if not receipt['passed'] or receipt['training_updates']!=0:
                raise ValueError('Missing frozen export')
        write_new(out/('resource_%s_%s.json'%(jobs[0][0],jobs[0][1])),dict(peaks=peaks,elapsed_seconds=time.monotonic()-started))
    except BaseException:
        for child,_,_,_ in children:
            if child.poll() is None:stop(child)
        raise
    finally:
        for _,log,_,_ in children:log.close()


def controller(out):
    # All parent metrics and selection are CPU-only. Child exports override
    # this visibility with exactly their assigned GPU ordinal.
    os.environ['CUDA_VISIBLE_DEVICES']=''
    protocol=read(out/'protocol.json')
    try:
        for jobs in ([('val',16,0),('val',18,1)],[('val',20,0),('val',22,1)],[('val',24,0)]):
            verify(protocol);infer(out,jobs,protocol)
        verify(protocol)
        from crane_project.tools.evaluate_external_obb_saved_v1 import evaluate
        for e in EPOCHS:
            status(out/'status.json',dict(state='CPU_METRICS',split='val',epoch=e))
            evaluate(out/'val'/('epoch_%d'%e))
        from crane_project.tools.select_external_obb_val_v1 import select
        select('gaucho',out/'val',out/'selection')
        lock=read(out/'selection/selected_identity.json')
        epoch=lock['selected_completed_epoch']
        if sha(lock['checkpoint'])!=protocol['weights'][str(epoch)]:
            raise ValueError('VAL-selected checkpoint identity differs')
        validate_test_lock(lock,'gaucho',protocol['weights'][str(epoch)],protocol['configuration'])
        verify(protocol)
        status(out/'status.json',dict(state='VAL_LOCKED',selected_epoch=epoch,
            locked_Taipei=lock['locked_Taipei'],TEST_started=False))
        (out/'test').mkdir()
        reserve_test(TEST_MARKER,dict(evaluation_dir=str(out),selected_epoch=epoch,
            checkpoint_sha256=protocol['weights'][str(epoch)],
            selection_lock_sha256=sha(out/'selection/selected_identity.json'),
            training_protocol_sha256=protocol['training_protocol_sha256']))
        infer(out,[('test',epoch,0)],protocol)
        status(out/'status.json',dict(state='CPU_METRICS',split='test',epoch=epoch))
        evaluate(out/'test'/('epoch_%d'%epoch))
        verify(protocol)
        write_new(out/'completion.json',dict(passed=True,selected_epoch=epoch,
            checkpoint_sha256=protocol['weights'][str(epoch)],TEST_runs=1,
            TEST_reselection=False,training_updates=0,automatic_retry=False,
            accuracy_complete=True,efficiency_required=True,
            metrics=str(out/'test'/('epoch_%d'%epoch)/'paper_metrics.json'),
            after=gpu_rows()))
        status(out/'status.json',dict(state='EFFICIENCY',selected_epoch=epoch,TEST_runs=1,training_updates=0))
        from crane_project.tools.benchmark_external_obb_v1 import run
        run(out,out/'efficiency',gpu=0)
        write_new(out/'pipeline_completion.json',dict(passed=True,accuracy_complete=True,
            efficiency_complete=True,selected_epoch=epoch,TEST_runs=1,training_updates=0,
            efficiency=str(out/'efficiency/completion.json')))
        status(out/'status.json',dict(state='COMPLETED',selected_epoch=epoch,TEST_runs=1,
            training_updates=0,accuracy_complete=True,efficiency_complete=True))
    except BaseException as error:
        status(out/'status.json',dict(state='FAILED',error=repr(error),automatic_retry=False))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('check','launch','controller'),default='check')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args();os.chdir(ROOT)
    if args.stage=='check':
        print(json.dumps(static_check(),indent=2));return
    if args.out is None:parser.error('--launch/--controller requires --out')
    out=args.out.resolve()
    if args.stage=='controller':return controller(out)
    prepare(out)
    with (out/'pipeline.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--stage','controller','--out',str(out)],
            cwd=str(ROOT),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_new(out/'launch.json',dict(pid=child.pid,out=str(out),VAL_candidates=list(EPOCHS),TEST_after_VAL_lock=True))
    print(json.dumps(read(out/'launch.json'),indent=2))


if __name__=='__main__':main()
