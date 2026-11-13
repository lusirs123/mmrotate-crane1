"""Bind the existing VAL -> lock -> TEST -> timing pipeline to no-scale GauCho.

The native exporter, metric functions, selection rule and timing worker are
reused. Separate dispatch preserves the original augmented experiment identity.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import eval_gaucho_val_test_v1 as workflow
from crane_project.tools import export_external_obb_v1 as exporter
from crane_project.tools import benchmark_external_obb_v1 as efficiency
from crane_project.tools import benchmark_crane_final_models_v1 as timing
from crane_project.tools import smoke_faa_gpu_v1 as resources
from crane_project.utils import external_training_monitor_io_v1 as monitor
from crane_project.tools.prepare_external_obb_v1 import read, sha, write_new

TRAIN = ROOT/'work_dirs/gaucho_no_port_shrink_v1/resume_003_gpu01'
CONFIG = 'configs/gaucho/crane_gaucho_fcos_r50_probiou_gpu01_no_port_shrink_v1.py'
SELF = 'crane_project/tools/eval_gaucho_no_port_shrink_v1.py'
_native_identity = exporter.config_identity
_installed = False


def configuration(model):
    result = _native_identity(model)
    if model == 'gaucho':
        result[SELF] = sha(ROOT/SELF)
    return result


def route_command(command):
    command = list(command)
    if len(command) > 1 and command[1] == str(ROOT/'crane_project/tools/export_external_obb_v1.py'):
        return [command[0], str(ROOT/SELF), '--stage', 'export']+command[2:]
    if len(command) > 1 and command[1] == str(ROOT/'crane_project/tools/benchmark_crane_final_models_v1.py'):
        if command[2:4] != ['--stage', 'worker']:
            raise ValueError('Only the fixed timing worker can be redirected')
        return [command[0], str(ROOT/SELF), '--stage', 'timing_worker']+command[4:]
    raise ValueError('Unexpected child command; do not launch another experiment')


def spawn(command, **kwargs):
    return subprocess.Popen(route_command(command), **kwargs)


def child_process_api():
    return SimpleNamespace(Popen=spawn, TimeoutExpired=subprocess.TimeoutExpired,
                           DEVNULL=subprocess.DEVNULL, STDOUT=subprocess.STDOUT)


def install():
    global _installed
    if _installed:
        return
    _installed = True
    exporter.GAUCHO_CONFIG = CONFIG
    exporter.config_identity = configuration
    workflow.TRAIN = TRAIN
    workflow.TEST_MARKER = TRAIN/'gaucho/TEST_execution.json'
    workflow.config_identity = configuration
    workflow.read = monitor.read_json
    workflow.status = monitor.atomic_json
    workflow.gpu_rows = monitor.gpu_rows
    workflow.processes = monitor.processes
    workflow.subprocess = child_process_api()
    efficiency.GAUCHO_CONFIG = CONFIG
    efficiency.config_identity = configuration
    timing.read = monitor.read_json
    timing.subprocess = child_process_api()
    resources.gpu_rows = monitor.gpu_rows
    resources.processes = monitor.processes


def prepare(out):
    install()
    workflow.prepare(out)
    path = out/'protocol.json'
    protocol = read(path)
    protocol['protocol'] = 'gaucho_no_scale_fixed_VAL_then_one_TEST_v1'
    protocol['training_dir'] = str(TRAIN)
    protocol['training_variant'] = 'without PortIsotropicShrink'
    protocol['extra_sources'][SELF] = sha(ROOT/SELF)
    helper = 'crane_project/utils/external_training_monitor_io_v1.py'
    protocol['extra_sources'][helper] = sha(ROOT/helper)
    monitor.atomic_json(path, protocol)


def main():
    os.chdir(ROOT)
    stage = sys.argv[2] if len(sys.argv)>2 and sys.argv[1]=='--stage' else None
    if stage in ('export', 'timing_worker'):
        install()
        remaining = sys.argv[3:]
        if stage == 'export':
            if '--model' not in remaining or remaining[remaining.index('--model')+1] != 'gaucho':
                raise ValueError('This wrapper only exports GauCho')
            sys.argv = [str(ROOT/SELF)]+remaining
            return exporter.main()
        sys.argv = [str(ROOT/SELF), '--stage', 'worker']+remaining
        return timing.main()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('check','launch','controller'), default='check')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    install()
    if args.stage == 'check':
        result = workflow.static_check()
        result.update(training_dir=str(TRAIN), configuration=configuration('gaucho'))
        print(json.dumps(result, indent=2))
        return
    if args.out is None:
        parser.error('A new --out directory is required')
    out = args.out.resolve()
    if args.stage == 'controller':
        return workflow.controller(out)
    prepare(out)
    with (out/'pipeline.log').open('x') as log:
        child = subprocess.Popen([sys.executable,str(ROOT/SELF),'--stage','controller',
            '--out',str(out)],cwd=str(ROOT),stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
    receipt = dict(pid=child.pid,out=str(out),training_dir=str(TRAIN),
                   VAL_candidates=list(workflow.EPOCHS),TEST_after_VAL_lock=True,
                   efficiency_required=True,physical_gpus=[0,1])
    write_new(out/'launch.json',receipt)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__ == '__main__':
    main()
