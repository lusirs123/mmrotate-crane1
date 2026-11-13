"""Continue the audited GauCho experiment on physical GPU0/1, preserving prior runs."""
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
from crane_project.tools.resume_gaucho_no_port_shrink_v1 import register_resume_audit
from crane_project.utils import external_training_monitor_io_v1 as monitor_io

control.GPUS = (0, 1)
control.previous.GPUS = control.GPUS


def check_training_config(baseline, historical, parent):
    expected, actual = deepcopy(baseline), deepcopy(historical)
    if expected.get('seed') != 0 or actual.get('seed') != 0:
        raise ValueError('Training seed differs')
    for key in ('work_dir', 'gpu_ids', 'auto_resume'):
        expected.pop(key, None)
        actual.pop(key, None)
    # MMCV resume replaces runner.meta with the saved metadata. Subsequent
    # checkpoints can therefore retain the original pre-resume config text.
    if actual == expected:
        return 'original_config_preserved_by_mmcv'
    if parent['protocol'] == 'gaucho_no_port_scale_native_resume_v1':
        if actual.get('resume_from') != parent['resume']['checkpoint']:
            raise ValueError('Checkpoint resume ancestry differs')
        hooks = list(expected.get('custom_hooks', []))+[
            dict(type='GauChoResumeAuditV1', priority='LOWEST')]
        if actual.get('custom_hooks') != hooks:
            raise ValueError('Checkpoint hooks differ from audited native resume')
        actual['custom_hooks'] = deepcopy(expected['custom_hooks'])
        actual['resume_from'] = expected['resume_from']
    if actual != expected:
        raise ValueError('Checkpoint training configuration differs')
    return 'immediate_parent_resume_config'


def prepare(parent, checkpoint, out):
    import torch
    from mmcv import Config
    if out.exists():
        raise ValueError('Use a new output directory')
    original = read(parent/'protocol.json')
    if original['protocol'] not in ('gaucho_no_port_scale_train_v1',
                                    'gaucho_no_port_scale_native_resume_v1'):
        raise ValueError('Unexpected parent experiment')
    control.assert_sources(original)
    before = control.idle_gpus()
    checkpoint = checkpoint.resolve(strict=True)
    if checkpoint.parent != (parent/'gaucho/train').resolve():
        raise ValueError('Checkpoint does not belong to the parent experiment')
    saved = torch.load(str(checkpoint), map_location='cpu')
    meta, optimizer = saved['meta'], saved['optimizer']
    epoch, iteration = int(meta['epoch']), int(meta['iter'])
    if not 0 < epoch < 24 or iteration != epoch*1279:
        raise ValueError('Unexpected checkpoint epoch/iteration')
    if not optimizer.get('state') or not all('momentum_buffer' in s for s in optimizer['state'].values()):
        raise ValueError('SGD momentum state is incomplete')
    cfg = Config.fromfile(str(ROOT/control.OBSERVED_CONFIG), import_custom_modules=False)
    historical = Config.fromstring(meta['config'], '.py')
    config_metadata = check_training_config(dict(cfg.items()), dict(historical.items()), original)
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
    ordered = sorted(monitor_io.gpu_rows(), key=lambda row: row['pci'])
    ordinals = [next(i for i, row in enumerate(ordered) if row['uuid']==before[g]['uuid'])
                for g in control.GPUS]
    if ordinals != [0, 1]:
        raise ValueError('Unexpected PCI ordering for physical GPU0/1')
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
        physical_gpus=list(control.GPUS), ordinals=ordinals,
        resume=dict(checkpoint=str(checkpoint), sha256=sha(checkpoint), epoch=epoch,
                    iteration=iteration, optimizer_states=len(optimizer['state']),
                    verified_train_val_files=verified, config_metadata=config_metadata), auto_resume=False)
    protocol['gaucho']['config'] = str(config.relative_to(ROOT))
    for name in [str(config.relative_to(ROOT)), str(Path(__file__).resolve().relative_to(ROOT)),
                 'crane_project/utils/external_training_monitor_io_v1.py']:
        protocol['sources'][name] = sha(ROOT/name)
    protocol['monitor_io'] = dict(attempts=4, delays_seconds=list(monitor_io.RETRY_DELAYS),
        gpu_query_timeout_seconds=5, persistent_failure='stop', memory_limits_changed=False)
    write_new(out/'protocol.json', protocol)


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
        control.__file__ = str(Path(__file__).resolve())
        monitor_io.install(control)
        return control.queue(out)
    if args.parent is None or args.checkpoint is None:
        parser.error('Explicit parent and checkpoint are required')
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
