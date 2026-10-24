"""Frozen last-four queue: 24-epoch DDP training, VAL selection, fixed TEST.

No resume, deletion or overwrite. Run from the existing ljj repository.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'work_dirs/symeood_ablation_v1'
GROUPS = ('110', '101', '011', '111')
RUN = BASE / 'launch_last4_20261023'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_status(**kwargs):
    (RUN / 'status.json').write_text(json.dumps(dict(time=time.strftime('%Y-%m-%d %H:%M:%S'),
                                                   **kwargs), indent=2))


def main():
    os.chdir(ROOT)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '2,3'
    RUN.mkdir(exist_ok=False)
    for g in GROUPS:
        assert not (BASE / g).exists(), 'Existing formal directory: ' + g
        for rank in (0, 1):
            p = BASE / 'preflight_formal_20261023' / g / ('formal_preflight_rank%d.json' % rank)
            d = json.loads(p.read_text())
            assert d['passed'] and len(d['iterations']) == 10

    paths = set(ROOT.glob('mmrotate/**/*.py')) | set(ROOT.glob('configs/**/*.py'))
    paths |= set(ROOT.glob('crane_project/configs/*.py'))
    paths |= {ROOT/'tools/train.py', ROOT/'tools/test.py', ROOT/'tools/dist_train.sh',
              ROOT/'crane_project/tools/ckpt_sweep.py',
              ROOT/'crane_project/tools/eval_crane_offline.py', Path(__file__).resolve()}
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}
    manifest = dict(groups=GROUPS, physical_gpus=[2, 3], seed=0, epochs=24,
                    per_gpu_batch=2, selection_epochs=[16, 18, 20, 22, 24],
                    test_authorized=True, depth_authorized=False, sha256=hashes)
    (RUN/'manifest.json').write_text(json.dumps(manifest, indent=2))

    def command(g, stage, cmd, env):
        for p, h in hashes.items():
            assert digest(ROOT/p) == h, 'Source changed during queue: ' + p
        write_status(group=g, stage=stage, state='running', command=cmd)
        with (RUN / ('%s_%s.log' % (g, stage))).open('x') as log:
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)

    env = os.environ.copy()
    env['PORT'] = '29626'
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    for g in GROUPS:
        cfg = 'crane_project/configs/crane_symeood_k1_ablation_%s_v1.py' % g
        work = str(BASE/g)
        command(g, 'train', ['bash', 'tools/dist_train.sh', cfg, '2', '--work-dir', work], env)
        assert (BASE/g/'epoch_24.pth').is_file()
        val_epochs = set()
        for p in (BASE/g).glob('*.log.json'):
            for line in p.open():
                d = json.loads(line)
                if d.get('mode') == 'val':
                    val_epochs.add(d['epoch'])
        assert val_epochs == set(range(2, 25, 2)), ('Incomplete online VAL', g, val_epochs)

    for g in GROUPS:
        cfg = 'crane_project/configs/crane_symeood_k1_ablation_%s_v1.py' % g
        work = str(BASE/g)
        # sweep's --gpu uses physical IDs and sets visibility itself.
        args = [sys.executable, 'crane_project/tools/ckpt_sweep.py',
                '--config', cfg, '--work-dir', work, '--gpu', '3',
                '--center-thresh', '15', '--mcml-limit', '5']
        command(g, 'val_select', args + ['--epochs', '16', '18', '20', '22', '24'], env)
        selection = BASE/g/'ckpt_sweep/sweep_results.json'
        d = json.loads(selection.read_text())
        assert len(d['all_checkpoints']) == 5, 'Incomplete candidate VAL'
        command(g, 'test', args + ['--final-test-from', str(selection)], env)
        reports = list((BASE/g/'ckpt_sweep/final_test').glob('*/final_test_metrics_v2.json'))
        assert len(reports) == 1, 'Missing TEST report'
        write_status(group=g, stage='test', state='complete', report=str(reports[0]))
    write_status(state='complete', groups=GROUPS)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if RUN.is_dir():
            (RUN/'failure.txt').write_text(repr(exc))
        raise
