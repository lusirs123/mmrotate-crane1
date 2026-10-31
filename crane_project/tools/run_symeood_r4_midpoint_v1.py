"""One authorized cache -> zero-update verification -> 24-epoch head run, no TEST."""
import argparse
from datetime import datetime, timedelta, timezone
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
from crane_project.utils import symeood_r4_midpoint_protocol_v1 as protocol


def command(stage, root, attempt, gpu):
    if stage not in ('cache', 'verify', 'train'):
        raise ValueError('Only the fixed adaptation stages are allowed')
    return [sys.executable, 'crane_project/tools/train_symeood_r4_midpoint_v1.py',
            '--stage', stage, '--root', str(root), '--attempt', attempt, '--gpu', str(gpu)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=protocol.DEFAULT_ROOT)
    parser.add_argument('--attempt', default='attempt_001')
    parser.add_argument('--gpu', type=int, choices=(2, 3), required=True)
    args = parser.parse_args()
    paths = protocol.artifact_paths(args.root, args.attempt)
    identity = protocol.checked_sources()
    if paths['attempt'].exists():
        raise ValueError('Fresh attempt child required; no resume or automatic retry')
    prepared = json.loads((paths['preparation']/'review.json').read_text())
    if prepared['identity'] != identity:
        raise ValueError('Preparation source identity differs')
    from crane_project.tools.train_symeood_r4_midpoint_v1 import resource_check
    resource_check(args.gpu, before=True)
    checkpoint = ROOT/protocol.R4_CHECKPOINT
    if protocol.sha(checkpoint) != protocol.R4_SHA:
        raise ValueError('R4 checkpoint SHA differs')
    paths['attempt'].mkdir(parents=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT), CUDA_DEVICE_ORDER='PCI_BUS_ID',
               CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    report = dict(protocol=protocol.VERSION, identity=identity, physical_gpu=args.gpu,
        stages=['cache', 'verify', 'train'], controller_pid=os.getpid(), retry=False,
        detector_updates=0, TEST=False, LVSR=False, automatic_adoption=False)
    (paths['attempt']/'launch.json').write_text(json.dumps(report, indent=2))
    def status(state, **details):
        value = dict(state=state, time_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                     physical_gpu=args.gpu, controller_pid=os.getpid(), TEST=False, LVSR=False, **details)
        tmp = paths['attempt']/'status.tmp'
        tmp.write_text(json.dumps(value, indent=2)); tmp.replace(paths['attempt']/'status.json')
    def interrupted(signum, frame):
        raise RuntimeError('Controller interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    proc = None
    try:
        for stage in ('cache', 'verify', 'train'):
            if protocol.checked_sources() != identity:
                raise ValueError('Source drift before '+stage)
            with (paths['attempt']/(stage+'.log')).open('x') as log:
                proc = subprocess.Popen(command(stage, paths['root'], args.attempt, args.gpu),
                    cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                status('RUNNING', stage=stage, child_pid=proc.pid)
                while proc.poll() is None:
                    time.sleep(5)
                    # Selected device only; driver default thermal protection stays unchanged.
                    text = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
                        '--query-gpu=index,memory.free,memory.used,temperature.gpu',
                        '--format=csv,noheader,nounits'], text=True, timeout=10)
                    index, free, used, temperature = map(int, text.strip().split(','))
                    apps = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
                        '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], text=True, timeout=10)
                    own = 0
                    for line in apps.splitlines():
                        fields = [v.strip() for v in line.split(',')]
                        if len(fields)==2 and fields[0]==str(proc.pid):
                            own += int(fields[1])
                    row = dict(time=time.time(), stage=stage, child_pid=proc.pid, physical_gpu=index,
                               free_mib=free, total_used_mib=used, own_mib=own, temperature_C=temperature)
                    with (paths['attempt']/'resources.jsonl').open('a') as f:
                        f.write(json.dumps(row)+'\n')
                    if index != args.gpu or free < protocol.MEMORY['minimum_free_mib'] or own >= protocol.MEMORY['process_stop_mib']:
                        raise RuntimeError('Selected-card memory/resource guard')
                if proc.returncode != 0:
                    raise RuntimeError(stage+' exited with code '+str(proc.returncode))
                status('STAGE_COMPLETE', stage=stage)
        selection = json.loads((paths['train']/'selection.json').read_text())
        selected = selection['selected_checkpoint']; source = paths['train']/selected['path']
        if protocol.sha(source) != selected['sha256']:
            raise ValueError('Selected head SHA differs')
        folder = paths['attempt']/'selected_backup'; folder.mkdir()
        with source.open('rb') as src, (folder/'selected_midpoint.pth').open('xb') as dst:
            shutil.copyfileobj(src, dst)
        if protocol.sha(folder/'selected_midpoint.pth') != selected['sha256']:
            raise ValueError('Selected head backup differs')
        if protocol.sha(checkpoint) != protocol.R4_SHA or protocol.checked_sources() != identity:
            raise ValueError('Frozen source/checkpoint changed')
        (paths['attempt']/'completion.json').write_text(json.dumps(dict(
            status='R4_MIDPOINT_TRAIN_VAL_COMPLETE_REVIEW_REQUIRED', selected=selected,
            detector_updates=0, epochs=24, TEST=False, LVSR=False, automatic_adoption=False), indent=2))
        status('COMPLETE', stage='train', selected=selected)
    except BaseException as error:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait(timeout=10)
        status('STOPPED_ON_ERROR', error=repr(error), retry=False)
        raise


if __name__ == '__main__':
    main()
