"""One monitored GPU2/3 E+A run, followed by fixed VAL; no retries or TEST."""
import argparse
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
EXTRA = ('crane_project/tools/run_eood_A_geometry_v1.py',
         'crane_project/tools/native_eood_A_geometry_v1.py',
         'crane_project/tools/analyze_eood_A_geometry_v1.py',
         'crane_project/utils/symeood_repro_trace_v1.py')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def gpus():
    result = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,memory.used',
                                      '--format=csv,noheader,nounits'], text=True)
    rows = {}
    for line in result.splitlines():
        index, free, used = map(int, line.split(','))
        rows[index] = dict(free_mib=free, used_mib=used)
    return {i: rows[i] for i in (2, 3)}


def event(base, state, **kwargs):
    row = dict(state=state, time_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat(), **kwargs)
    (base / 'run_status.json').write_text(json.dumps(row, indent=2))
    with (base / 'run_events.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')


def verify(base):
    c = read(base / 'launch_protocol_gpus23.json')
    assert sha(ROOT / c['config']) == c['config_sha256']
    for name, digest in c['sources'].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in c['source_tree'].items():
        assert sha(ROOT / name) == digest, name
    return c


def prepare(base):
    old = read(base / 'protocol.json')
    native = read(base / 'native_CPU_receipt.json')
    assert native['passed'] and native['native_tests'] == 6 and not native['skipped']
    assert native['protocol_sha256'] == sha(base / 'protocol.json')
    assert old['control_retrain'] is False and old['planned_new_training_runs'] == 1
    assert not (base / 'train').exists() and not (base / 'gate_work').exists()
    for name, digest in old['sources'].items():
        assert sha(ROOT / name) == digest, name
    assert sha(ROOT / old['config']) == old['config_sha256']
    snapshot_path = ROOT / 'work_dirs/symeood_110_retrain_seed0_20261026_v1/protocol.json'
    inventory = read(snapshot_path)['data_inventory']
    assert len(inventory) == 3445
    for row in inventory:
        assert row['split'] in ('train', 'val')
        for kind in ('image', 'annotation'):
            assert sha(ROOT / row[kind]) == row[kind + '_sha256'], row[kind]
    sweep = read(ROOT / old['baseline']['reference'] / 'val_sweep_port_v1/sweep_results.json')
    val_dir = ROOT / 'crane_project/data/crane_grab_port_day2night_v1/val/annfiles'
    digest = hashlib.sha256()
    anns = sorted(p for p in val_dir.glob('*.txt') if not p.name.startswith('._'))
    for p in anns:
        digest.update(p.name.encode()); digest.update(b'\0'); digest.update(p.read_bytes())
    assert len(anns) == 887 and digest.hexdigest() == sweep['source_val_annotations_sha256']
    assert sha(ROOT / old['baseline']['reference'] / 'epoch_24.pth') == old['baseline']['checkpoint_sha256']
    free = gpus()
    assert all(v['free_mib'] >= 6656 for v in free.values()), free
    previous_launch = read(snapshot_path)['launch_environment']
    c = dict(old)
    c.update(physical_gpus=[2,3], resource_hold=False, user_authorized=True,
        CPU_preparation_only=False, native_CPU_tests_pending=False,
        original_preparation_protocol_sha256=sha(base/'protocol.json'),
        allocator_limit_mib=6144, own_driver_limit_mib=6656,
        minimum_free_before_launch_mib=6656, minimum_free_during_run_mib=512,
        free_before=free, data_files_verified=2*len(inventory),
        snapshot_inventory_sha256=sha(snapshot_path),
        launch_environment=dict(previous_launch, CUDA_VISIBLE_DEVICES='2,3'),
        native_gate='Two-rank full TRAIN batch, epoch0 and epoch4 paths, backward only; zero updates, discarded.',
        observation_hook='Outside config: captures initial state/first epoch inputs/update count; checks finite loss before optimizer.',
        sources=dict(old['sources'], **{name:sha(ROOT/name) for name in EXTRA}),
        source_tree={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'mmrotate').rglob('*.py')},
        TEST=False, automatic_resume=False, retry=False, baseline_retrain=False)
    c['GPU_training_resources'] = dict(old['GPU_training_resources'],
        physical_gpus=[2,3], no_launch_while_hold=False,
        limit_scope='6144 MiB allocator cap enforced by native entry; own process driver memory monitored at 6656 MiB; stop on OOM or low free memory.')
    c['sources'].update({name:sha(ROOT/name) for name in (
        'tools/train.py','tools/test.py','mmrotate/apis/train.py',
        'crane_project/tools/ckpt_sweep.py','crane_project/tools/eval_crane_offline.py',
        'crane_project/tools/eval_port_geometry_ap_v1.py',
        'crane_project/utils/port_geometry_midpoint_edge_residual_metrics_v1.py')})
    save(base / 'launch_protocol_gpus23.json', c)
    event(base, 'PREPARED_GPU23', data_files_verified=c['data_files_verified'], GPU_memory=free)
    print('E_A_GPU23_PREPARED', flush=True)


def env_for(c, cpu=False):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    for key, value in c['launch_environment'].items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    if cpu:
        env['CUDA_VISIBLE_DEVICES'] = ''
    return env


def stop_group(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=15)


def execute(base, stage, command, cpu=False):
    c = verify(base)
    if not cpu:
        free = gpus()
        assert all(v['free_mib'] >= 6656 for v in free.values()), free
    log = base / (stage + '.log')
    with log.open('x') as stream:
        proc = subprocess.Popen(command, cwd=str(ROOT), env=env_for(c,cpu),
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        event(base, 'RUNNING', stage=stage, pid=proc.pid, controller_pid=os.getpid())
        try:
            while proc.poll() is None:
                time.sleep(10)
                free = gpus()
                if stage != 'VAL_analysis':
                    assert all(v['free_mib'] >= 512 for v in free.values()), ('Free memory guard',free)
                text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                    '--format=csv,noheader,nounits'],text=True)
                own = {}
                for line in text.splitlines():
                    parts = line.split(',')
                    if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
                        continue
                    pid, used = map(int, parts)
                    try:
                        if os.getpgid(pid) == proc.pid:
                            own[pid] = used
                    except ProcessLookupError:
                        pass
                assert all(v <= 6656 for v in own.values()), ('Own driver memory guard', own)
                with (base / 'memory_samples.jsonl').open('a') as f:
                    f.write(json.dumps(dict(stage=stage, time=time.time(), GPUs=free, own_process_mib=own))+'\n')
            assert proc.returncode == 0, (stage, proc.returncode)
        except BaseException:
            stop_group(proc)
            raise


def run(base):
    c = verify(base)
    assert not (base/'run_claim.json').exists()
    assert not (base/'train').exists() and not (base/'gate_work').exists()
    save(base/'run_claim.json',dict(pid=os.getpid(),physical_gpus=[2,3],retry=False))
    entry = 'crane_project/tools/native_eood_A_geometry_v1.py'
    def native(stage):
        return [sys.executable,'-m','torch.distributed.launch','--use_env','--nproc_per_node=2',
                '--master_port=29693',entry,stage,str(base)]
    try:
        execute(base,'gate',native('gate'))
        rows = [read(base/'gate_work'/('gate_rank%d.json'%rank)) for rank in (0,1)]
        assert all(r['passed'] and r['optimizer_updates']==0 and len(r['phases'])==2 for r in rows)
        save(base/'gate_acceptance.json',dict(passed=True,rows=rows,discarded=True))
        execute(base,'training',native('train'))
        for rank in (0,1):
            done = read(base/'train'/('complete_rank%d.json'%rank))
            assert done['passed'] and done['updates']==15360
        for name,digest in c['source_tree'].items():
            assert sha(ROOT/name)==digest,name
        save(base/'training_verification.json',dict(passed=True,epochs=24,updates=15360,TEST=False))
        execute(base,'VAL_sweep',[sys.executable,entry,'sweep',str(base)],cpu=True)
        execute(base,'VAL_analysis',[sys.executable,'crane_project/tools/analyze_eood_A_geometry_v1.py',
            '--base',str(base)],cpu=True)
        event(base,'COMPLETE',stage='TRAIN_VAL',TEST=False,automatic_adoption=False)
    except BaseException as e:
        event(base,'STOPPED_ON_ERROR',error=repr(e),retry=False)
        raise


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--stage',choices=['prepare','run'],required=True)
    a=p.parse_args()
    def terminated(signum, frame):
        raise RuntimeError('Controller terminated; stop its active process group')
    signal.signal(signal.SIGTERM, terminated)
    (prepare if a.stage=='prepare' else run)(a.base.resolve())
