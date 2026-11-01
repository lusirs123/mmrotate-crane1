"""Explicit two-GPU SGAR gate -> one training -> fixed VAL, with no TEST/retry.

Importing this file never launches work. GPU IDs must be supplied at launch
preparation; --execute is required for the separate run stage.
"""
import argparse
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.utils.eood_sgar_protocol_v1 import read, save, sha, verify


def gpu_memory(ids):
    result = subprocess.check_output([
        'nvidia-smi', '-i', ','.join(map(str, ids)),
        '--query-gpu=index,memory.free,memory.used', '--format=csv,noheader,nounits'], text=True)
    rows = {}
    for line in result.splitlines():
        index, free, used = map(int, line.split(','))
        if index not in ids or index in rows or min(free, used) < 0:
            raise ValueError('Invalid selected GPU memory row')
        rows[index] = dict(free_mib=free, used_mib=used)
    if set(rows) != set(ids):
        raise ValueError('Incomplete selected GPU query')
    return rows


def event(base, state, **values):
    row = dict(state=state, time_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat(), **values)
    (base / 'run_status.json').write_text(json.dumps(row, indent=2))
    with (base / 'run_events.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')


def prepare_launch(base, ids, master_port):
    c = verify(base)
    receipt = read(base / 'native_CPU_receipt.json')
    assert receipt['passed'] and receipt['skipped'] == 0
    assert receipt['protocol_sha256'] == sha(base / 'protocol.json')
    assert c['planned_new_training_runs'] == 1 and not c['control_retrain'] and not c['R4_Ref']
    if len(set(ids)) != 2 or any(i < 0 for i in ids):
        raise ValueError('Exactly two distinct physical GPU IDs required')
    if not 1024 <= master_port <= 65535:
        raise ValueError('Invalid distributed port')
    for name in ('train', 'gate_work', 'run_claim.json', 'launch_protocol.json'):
        assert not (base / name).exists(), name
    for key, path in c['references'].items():
        assert sha(ROOT / path) == c['reference_sha256'][key], path
    snapshot = read(ROOT / c['references']['inventory'])
    inventory = snapshot['data_inventory']
    assert len(inventory) == 3445
    for row in inventory:
        assert row['split'] in ('train', 'val')
        for kind in ('image', 'annotation'):
            assert sha(ROOT / row[kind]) == row[kind + '_sha256'], row[kind]
    sweep = read(ROOT / c['references']['sweep'])
    val_dir = ROOT / 'crane_project/data/crane_grab_port_day2night_v1/val/annfiles'
    import hashlib
    digest = hashlib.sha256()
    annotations = sorted(p for p in val_dir.glob('*.txt') if not p.name.startswith('._'))
    for p in annotations:
        digest.update(p.name.encode()); digest.update(b'\0'); digest.update(p.read_bytes())
    assert len(annotations) == 887 and digest.hexdigest() == sweep['source_val_annotations_sha256']
    baseline = c['baseline']
    assert sha(ROOT / baseline['reference'] / 'epoch_24.pth') == baseline['checkpoint_sha256']
    free = gpu_memory(ids)
    assert all(v['free_mib'] >= c['minimum_free_before_launch_mib'] for v in free.values()), free
    query = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id',
                                    '--format=csv,noheader'], text=True)
    devices = [tuple(part.strip() for part in line.split(',')) for line in query.splitlines()]
    pci_order = sorted(devices, key=lambda row: row[2])
    cuda_ordinals = [next(i for i, row in enumerate(pci_order) if int(row[0]) == physical)
                     for physical in ids]
    uuids = {str(int(row[0])): row[1] for row in devices if int(row[0]) in ids}
    assert set(uuids) == set(map(str, ids)) and len(set(uuids.values())) == 2
    c.update(physical_gpus=ids, GPU_UUIDs=uuids, visible_cuda_ordinals=cuda_ordinals,
             pci_device_inventory=devices, master_port=master_port,
             preparation_protocol_sha256=sha(base / 'protocol.json'),
             CPU_receipt_sha256=sha(base / 'native_CPU_receipt.json'),
             native_CPU_tests_pending=False, GPU_gate_pending=True,
             free_before=free, data_files_verified=2 * len(inventory),
             launch_environment=dict(snapshot['launch_environment'],
                 CUDA_VISIBLE_DEVICES=','.join(map(str, cuda_ordinals)),
                 CUDA_DEVICE_ORDER='PCI_BUS_ID'),
             source_tree={str(p.relative_to(ROOT)): sha(p)
                          for p in (ROOT / 'mmrotate').rglob('*.py')},
             val_annotations_sha256=digest.hexdigest())
    save(base / 'launch_protocol.json', c)
    event(base, 'LAUNCH_PREPARED_NO_MODEL_RUN', physical_gpus=ids)


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
    c = verify(base, launch=True)
    if not cpu:
        assert all(v['free_mib'] >= c['minimum_free_before_launch_mib']
                   for v in gpu_memory(c['physical_gpus']).values())
    with (base / (stage + '.log')).open('x') as stream:
        proc = subprocess.Popen(command, cwd=str(ROOT), env=env_for(c, cpu),
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        event(base, 'RUNNING', stage=stage, pid=proc.pid, controller_pid=os.getpid())
        try:
            while proc.poll() is None:
                time.sleep(c['memory_poll_interval_seconds'])
                if stage == 'VAL_analysis':
                    continue
                active_ids = c['physical_gpus'][:1] if stage == 'VAL_sweep' else c['physical_gpus']
                free = gpu_memory(active_ids)
                assert all(v['free_mib'] >= c['minimum_free_during_run_mib']
                           for v in free.values()), free
                apps = subprocess.check_output(['nvidia-smi', '-i', ','.join(map(str, active_ids)),
                    '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], text=True)
                own = {}
                for line in apps.splitlines():
                    parts = line.split(',')
                    if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
                        continue
                    pid, used = map(int, parts)
                    try:
                        if os.getpgid(pid) == proc.pid:
                            own[pid] = used
                    except ProcessLookupError:
                        pass
                assert all(v <= c['own_driver_limit_mib'] for v in own.values()), own
                with (base / 'memory_samples.jsonl').open('a') as out:
                    out.write(json.dumps(dict(stage=stage, time=time.time(),
                                              GPUs=free, own_process_mib=own)) + '\n')
            assert proc.returncode == 0, (stage, proc.returncode)
        except BaseException:
            stop_group(proc)
            raise


def run(base):
    c = verify(base, launch=True)
    assert sha(base / 'native_CPU_receipt.json') == c['CPU_receipt_sha256']
    assert not (base / 'train').exists() and not (base / 'gate_work').exists()
    save(base / 'run_claim.json', dict(pid=os.getpid(), physical_gpus=c['physical_gpus'], retry=False))
    entry = 'crane_project/tools/native_eood_sgar_v1.py'

    def native(stage):
        return [sys.executable, '-m', 'torch.distributed.launch', '--use_env',
                '--nproc_per_node=2', '--master_port=' + str(c['master_port']),
                entry, stage, str(base)]

    try:
        execute(base, 'gate', native('gate'))
        rows = [read(base / 'gate_work' / ('gate_rank%d.json' % r)) for r in (0, 1)]
        assert all(r['passed'] and r['optimizer_updates'] == 0 and len(r['phases']) == 2 for r in rows)
        save(base / 'gate_acceptance.json', dict(passed=True, rows=rows, discarded=True))
        execute(base, 'training', native('train'))
        for rank in (0, 1):
            done = read(base / 'train' / ('complete_rank%d.json' % rank))
            assert done['passed'] and done['updates'] == c['expected_updates_per_rank']
            records = [json.loads(s) for s in (base / 'train' / ('epochs_rank%d.jsonl' % rank)).read_text().splitlines()]
            assert [r['epoch'] for r in records] == list(range(1, 25))
            assert sum(r['batches'] for r in records) == c['expected_updates_per_rank']
        verify(base, launch=True)
        save(base / 'training_verification.json', dict(passed=True, epochs=24,
             updates_per_rank=c['expected_updates_per_rank'], TEST=False))
        execute(base, 'VAL_sweep', [sys.executable, entry, 'sweep', str(base)], cpu=True)
        execute(base, 'VAL_analysis', [sys.executable, 'crane_project/tools/analyze_eood_sgar_v1.py',
                                     '--base', str(base)], cpu=True)
        sweep = read(base / 'train/ckpt_sweep/sweep_results.json')
        chosen = sweep['selected_checkpoint']
        if chosen is not None:
            record = sweep['all_checkpoints'][chosen]
            source = Path(record['checkpoint'])
            assert sha(source) == record['checkpoint_sha256']
            target = base / 'selected_checkpoint.pth'
            with source.open('rb') as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst)
            assert sha(target) == record['checkpoint_sha256']
            save(base / 'selected_identity.json', dict(checkpoint=chosen,
                 sha256=sha(target), TEST=False, original_selection_unchanged=True))
        event(base, 'COMPLETE', stage='TRAIN_VAL', selected_checkpoint=chosen,
              TEST=False, automatic_adoption=False)
    except BaseException as error:
        event(base, 'STOPPED_ON_ERROR', error=repr(error), retry=False)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--stage', choices=('prepare', 'run'), required=True)
    p.add_argument('--gpus', nargs=2, type=int)
    p.add_argument('--master-port', type=int, default=29719)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    base = args.base.resolve()
    def terminated(signum, frame):
        raise RuntimeError('Controller terminated; stop its active process group')
    signal.signal(signal.SIGTERM, terminated)
    if args.stage == 'prepare':
        if args.gpus is None:
            p.error('Launch preparation requires two explicit --gpus')
        prepare_launch(base, args.gpus, args.master_port)
    else:
        if not args.execute:
            p.error('Running models requires explicit --execute')
        if args.gpus is not None:
            p.error('Run uses the GPU IDs frozen by launch preparation')
        run(base)


if __name__ == '__main__':
    main()
