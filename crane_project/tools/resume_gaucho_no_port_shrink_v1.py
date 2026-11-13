"""Explicit native MMCV resume; reuse the GauCho controller and resource guards."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import run_gaucho_no_port_shrink_v1 as control
from crane_project.tools.prepare_external_obb_v1 import DATA, dataset_manifest, read, sha, write_new
from crane_project.utils import external_training_monitor_io_v1 as monitor_io


def prepare(parent, checkpoint, out):
    import torch
    from mmcv import Config
    if out.exists():
        raise ValueError('Use a new output directory; preserve interrupted artifacts')
    original = read(parent/'protocol.json')
    if original['protocol'] != 'gaucho_no_port_scale_train_v1':
        raise ValueError('Unexpected parent experiment')
    control.assert_sources(original)
    before = control.idle_gpus()
    checkpoint = checkpoint.resolve(strict=True)
    if checkpoint.parent != (parent/'gaucho/train').resolve():
        raise ValueError('Checkpoint must belong to the interrupted training')
    saved = torch.load(str(checkpoint), map_location='cpu')
    meta, optimizer = saved['meta'], saved['optimizer']
    epoch, iteration = int(meta['epoch']), int(meta['iter'])
    if not 0 < epoch < 24 or iteration != epoch*1279:
        raise ValueError('Incomplete or unexpected epoch boundary')
    if not optimizer.get('state') or not all('momentum_buffer' in s for s in optimizer['state'].values()):
        raise ValueError('Full SGD momentum state required')
    cfg = Config.fromfile(str(ROOT/control.OBSERVED_CONFIG), import_custom_modules=False)
    historical = Config.fromstring(meta['config'], '.py')
    expected = deepcopy(dict(cfg.items()))
    actual = deepcopy(dict(historical.items()))
    for key in ('work_dir', 'gpu_ids', 'seed', 'auto_resume'):
        expected.pop(key, None)
        actual.pop(key, None)
    if expected != actual:
        raise ValueError('Checkpoint training configuration differs')
    verified = 0
    for row in dataset_manifest():
        if row['split'] == 'test':
            continue
        for kind in ('images', 'annfiles'):
            if sha(DATA/row[kind]['path']) != row[kind]['sha256']:
                raise ValueError('TRAIN/VAL bytes changed: '+row['id'])
            verified += 1
    if verified != 6890:
        raise ValueError('Incomplete dataset validation')
    control.idle_gpus()
    out.mkdir(parents=True, exist_ok=False)
    config = out/'resume_config.py'
    hooks = list(cfg.custom_hooks)+[dict(type='GauChoResumeAuditV1', priority='LOWEST')]
    config.write_text('_base_ = '+repr(str(ROOT/control.OBSERVED_CONFIG))+'\n'
                      'resume_from = '+repr(str(checkpoint))+'\n'
                      'custom_hooks = '+repr(hooks)+'\n')
    protocol = deepcopy(original)
    protocol.update(protocol='gaucho_no_port_scale_native_resume_v1',
        created_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat(), before=before,
        parent=str(parent), parent_protocol_sha256=sha(parent/'protocol.json'),
        resume=dict(checkpoint=str(checkpoint), sha256=sha(checkpoint), epoch=epoch,
                    iteration=iteration, optimizer_states=len(optimizer['state']),
                    verified_train_val_files=verified), auto_resume=False)
    protocol['gaucho']['config'] = str(config.relative_to(ROOT))
    protocol['sources'][str(config.relative_to(ROOT))] = sha(config)
    protocol['sources'][str(Path(__file__).resolve().relative_to(ROOT))] = sha(__file__)
    helper = 'crane_project/utils/external_training_monitor_io_v1.py'
    protocol['sources'][helper] = sha(ROOT/helper)
    protocol['monitor_io'] = dict(attempts=4, delays_seconds=list(monitor_io.RETRY_DELAYS),
                                gpu_query_timeout_seconds=5, persistent_failure='stop',
                                memory_limits_changed=False)
    write_new(out/'protocol.json', protocol)


def register_resume_audit():
    import torch
    from mmcv.runner import HOOKS, Hook

    @HOOKS.register_module()
    class GauChoResumeAuditV1(Hook):
        def before_run(self, runner):
            protocol = read(Path(os.environ['EXTERNAL_QUEUE_DIR'])/'protocol.json')
            info = protocol['resume']
            if sha(info['checkpoint']) != info['sha256']:
                raise ValueError('Checkpoint identity changed')
            saved = torch.load(info['checkpoint'], map_location='cpu')['optimizer']
            restored = runner.optimizer.state_dict()
            if runner.epoch != info['epoch'] or runner.iter != info['iteration'] or runner.max_epochs != 24:
                raise ValueError('Native runner epoch/iteration was not restored')
            if restored['param_groups'] != saved['param_groups']:
                raise ValueError('Optimizer parameter groups or learning rate differ')
            if restored['state'].keys() != saved['state'].keys():
                raise ValueError('Optimizer state count differs')
            for key, state in saved['state'].items():
                for name, value in state.items():
                    got = restored['state'][key][name]
                    if torch.is_tensor(value):
                        if not torch.equal(got.detach().cpu(), value):
                            raise ValueError('Optimizer tensor was not restored')
                    elif got != value:
                        raise ValueError('Optimizer scalar was not restored')
            rank = int(os.environ['LOCAL_RANK'])
            write_new(Path(os.environ['EXTERNAL_RUN_DIR'])/('rank_%d_restored_state.json'%rank),
                dict(passed=True, rank=rank, start_epoch=runner.epoch, next_epoch=runner.epoch+1,
                     restored_iteration=runner.iter, target_epochs=runner.max_epochs,
                     optimizer_states=len(restored['state']), momentum_exact=True,
                     learning_rates=[g['lr'] for g in restored['param_groups']],
                     checkpoint_sha256=info['sha256'], physical_gpu=control.GPUS[rank]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('launch', 'queue', 'worker'), required=True)
    parser.add_argument('--model', choices=('gaucho',))
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    out = args.out.resolve()
    if args.stage == 'worker':
        protocol = read(Path(os.environ['EXTERNAL_QUEUE_DIR'])/'protocol.json')
        control.previous.OBSERVED_CONFIG = protocol['gaucho']['config']
        register_resume_audit()
        return control.previous.worker('gaucho', out)
    if args.stage == 'queue':
        # The reused controller dispatches workers through its module file path.
        # Redirect only that dispatch to this resume-aware wrapper.
        control.__file__ = str(Path(__file__).resolve())
        monitor_io.install(control)
        return control.queue(out)
    if args.parent is None or args.checkpoint is None:
        parser.error('Explicit parent and checkpoint are required for resume')
    prepare(args.parent.resolve(), args.checkpoint.resolve(), out)
    with (out/'supervisor.log').open('x') as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
            '--stage', 'queue', '--out', str(out)], cwd=str(ROOT), stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = dict(supervisor_pid=child.pid, out=str(out), physical_gpus=list(control.GPUS),
                   resumed_from=read(out/'protocol.json')['resume'], auto_TEST=False)
    write_new(out/'launch.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
