"""Bounded monitoring I/O retries; persistent failures remain fatal."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

RETRY_DELAYS = (0.05, 0.15, 0.45)
TRANSIENT_ERRORS = (OSError, json.JSONDecodeError, subprocess.SubprocessError)


def retry_io(operation, target, function):
    for attempt in range(len(RETRY_DELAYS)+1):
        try:
            return function()
        except TRANSIENT_ERRORS as error:
            final = attempt == len(RETRY_DELAYS)
            event = dict(event='MONITOR_IO_FAILED' if final else 'MONITOR_IO_RETRY',
                         operation=operation, target=str(target), attempt=attempt+1,
                         error=repr(error), filename=getattr(error, 'filename', None),
                         traceback=traceback.format_exc(), unix_time=time.time())
            print(json.dumps(event), file=sys.stderr, flush=True)
            if final:
                raise
            time.sleep(RETRY_DELAYS[attempt])


def read_json(path):
    path = Path(path)
    return retry_io('read_json', path, lambda: json.loads(path.read_text()))


def atomic_json(path, value):
    path = Path(path)
    payload = json.dumps(value, indent=2, allow_nan=False)+'\n'

    def write():
        temporary = path.with_name(path.name+'.%d.%s.tmp'%(os.getpid(), uuid.uuid4().hex))
        try:
            temporary.write_text(payload)
            os.replace(str(temporary), str(path))
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    return retry_io('atomic_json', path, write)


def query_gpu(arguments):
    command = ['/usr/bin/nvidia-smi']+arguments+['--format=csv,noheader,nounits']
    return retry_io('nvidia-smi', command,
        lambda: subprocess.check_output(command, text=True, timeout=5))


def gpu_rows():
    value = query_gpu(['--query-gpu=index,uuid,pci.bus_id,memory.total,memory.used,memory.free,utilization.gpu'])
    rows = []
    for line in value.splitlines():
        index, identity, pci, total, used, free, util = [p.strip() for p in line.split(',')]
        rows.append(dict(index=int(index), uuid=identity, pci=pci, total_mib=int(total),
                         used_mib=int(used), free_mib=int(free), utilization_pct=int(util)))
    if len({row['index'] for row in rows}) != len(rows) or not rows:
        raise ValueError('Missing or duplicate GPU monitoring rows')
    return rows


def processes():
    value = query_gpu(['--query-compute-apps=gpu_uuid,pid,used_memory'])
    rows = []
    for line in value.splitlines():
        identity, pid, memory = [p.strip() for p in line.split(',')]
        rows.append(dict(uuid=identity, pid=int(pid), used_mib=int(memory)))
    return rows


def install(control):
    """Change only supervisor I/O, retaining its original stop and memory logic."""
    control.read = read_json
    control.atomic = atomic_json
    control.gpu_rows = gpu_rows
    control.processes = processes

    def worker_pids(folder):
        return {read_json(p)['pid'] for p in Path(folder).glob('rank_*_identity.json')}

    control.worker_pids = worker_pids
    original_stop = control.stop_group

    def stop_group(child, folder):
        error = sys.exc_info()[1]
        if error is not None:
            print(json.dumps(dict(event='MONITOR_FATAL_EXCEPTION', error=repr(error),
                filename=getattr(error, 'filename', None), traceback=traceback.format_exc(),
                unix_time=time.time())), file=sys.stderr, flush=True)
        return original_stop(child, folder)

    control.stop_group = stop_group
