"""Common TEST efficiency after completed, VAL-locked external accuracy.

Reuses the final/EOOD/FAA timing worker and resource guard. No accuracy
evaluation, checkpoint selection, model modifications or automatic retries.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import json
import os
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import benchmark_crane_final_models_v1 as common
from crane_project.tools.export_external_obb_v1 import config_identity, validate_test_lock
from crane_project.tools.prepare_external_obb_v1 import GAUCHO_CONFIG

TEMPLATE = 'configs/external_obb_efficiency_v1.json'


def completed_identity(evaluation):
    evaluation = Path(evaluation).resolve()
    done = common.read(evaluation/'completion.json')
    lock_path = evaluation/'selection/selected_identity.json'
    lock = common.read(lock_path)
    if (not done['passed'] or done['TEST_runs'] != 1 or done['training_updates'] != 0
            or done['TEST_reselection'] is not False or lock['model'] != 'gaucho'
            or done['selected_epoch'] != lock['selected_completed_epoch']
            or done['checkpoint_sha256'] != lock['checkpoint_sha256']):
        raise ValueError('Complete VAL-locked GauCho TEST is required')
    validate_test_lock(lock, 'gaucho', done['checkpoint_sha256'], config_identity('gaucho'))
    test = evaluation/'test'/('epoch_%d'%lock['selected_completed_epoch'])
    receipt = common.read(test/'export_receipt.json')
    metrics = common.read(test/'paper_metrics.json')
    if (not receipt['passed'] or receipt['model'] != 'gaucho' or receipt['split'] != 'test'
            or receipt['frames'] != 1440 or len(set(receipt['frame_ids'])) != 1440
            or receipt['checkpoint_sha256'] != lock['checkpoint_sha256']
            or receipt['selection_lock_sha256'] != common.frozen.sha(lock_path)
            or receipt['configuration'] != lock['configuration'] or receipt['training_updates'] != 0
            or receipt['initialized_state_sha256'] != receipt['final_state_sha256']
            or common.frozen.sha(test/'preds/results.pkl') != receipt['results_sha256']
            or metrics['split'] != 'test' or metrics['metrics'][0]['frames'] != 1440
            or not all(c['pass_check'] for c in metrics['native_checks'])):
        raise ValueError('Incomplete or inconsistent frozen TEST outputs')
    return lock, test


def make_profile(evaluation):
    evaluation = Path(evaluation).resolve()
    lock, test = completed_identity(evaluation)
    cfg = common.read(common.local(TEMPLATE))
    training = common.read(evaluation/'protocol.json')
    sources = dict(training['training_sources'], **lock['configuration'])
    for name in (TEMPLATE, cfg['runtime_reference'],
                 'crane_project/tools/benchmark_external_obb_v1.py',
                 'crane_project/tools/benchmark_crane_final_models_v1.py',
                 'crane_project/tools/benchmark_port_detection_runtime_v1.py',
                 'crane_project/utils/port_detection_runtime_v1.py',
                 'crane_project/tools/smoke_faa_gpu_v1.py'):
        sources[name] = common.frozen.sha(common.local(name))
    epoch = lock['selected_completed_epoch']
    cfg.update(evaluation_dir=str(evaluation), accuracy_evidence_sha256={
        str(p):common.frozen.sha(p) for p in (
            evaluation/'completion.json', evaluation/'selection/selected_identity.json',
            test/'export_receipt.json', test/'paper_metrics.json', test/'per_frame.jsonl')},
        source_sha256=sources, models=dict(gaucho=dict(
            label='Official GauCho FCOS R50 ProbIoU, VAL-selected ep%d'%epoch,
            config=GAUCHO_CONFIG, weights=[dict(epoch=epoch,sha256=lock['checkpoint_sha256'],
                paths=[lock['checkpoint']])], reference=dict(
                sha256=common.frozen.sha(test/'per_frame.jsonl'), paths=[str(test/'per_frame.jsonl')]),
            reference_key='prediction')))
    validate_profile(cfg)
    return cfg


def validate_profile(cfg):
    template = common.read(common.local(TEMPLATE))
    if any(cfg.get(k) != v for k,v in template.items()) or tuple(cfg['models']) != ('gaucho',):
        raise ValueError('External timing must retain the common TEST efficiency protocol')
    for name,digest in cfg['source_sha256'].items():
        if common.frozen.sha(common.local(name)) != digest:
            raise ValueError('Efficiency source drift: '+name)
    for name,digest in cfg['accuracy_evidence_sha256'].items():
        if common.frozen.sha(name) != digest:
            raise ValueError('Completed accuracy evidence changed: '+name)
    lock,test = completed_identity(cfg['evaluation_dir'])
    spec = cfg['models']['gaucho']
    expected_weight = dict(epoch=lock['selected_completed_epoch'],sha256=lock['checkpoint_sha256'],
                           paths=[lock['checkpoint']])
    expected_ref = dict(sha256=common.frozen.sha(test/'per_frame.jsonl'),paths=[str(test/'per_frame.jsonl')])
    if (spec['config'] != GAUCHO_CONFIG or spec['weights'] != [expected_weight]
            or spec['reference'] != expected_ref or spec['reference_key'] != 'prediction'):
        raise ValueError('Efficiency may only replay the fixed VAL-selected TEST checkpoint')
    train = common.read(Path(cfg['evaluation_dir'])/'protocol.json')
    required = dict(train['training_sources'], **lock['configuration'])
    if any(cfg['source_sha256'].get(n) != h for n,h in required.items()):
        raise ValueError('Efficiency sources omit frozen model/evaluator dependencies')
    for ancestor in common.config_closure(common.local(GAUCHO_CONFIG)):
        if str(ancestor.relative_to(ROOT)) not in cfg['source_sha256']:
            raise ValueError('Unpinned model config ancestor')


def prepare(evaluation, out):
    cfg = make_profile(evaluation)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    # Run-specific identity snapshot, not a model definition. The model and
    # shared efficiency settings stay under configs/.
    path = out/'execution_protocol.json'
    common.save(path, cfg)
    return path


def run_prepared(profile, out, gpu):
    try:
        common.benchmark(SimpleNamespace(config=str(profile),out_dir=Path(out)/'benchmark',gpu=gpu))
        report = common.read(Path(out)/'benchmark/runtime_compare.json')
        common.save(Path(out)/'completion.json', dict(passed=True,accuracy_and_efficiency_complete=True,
            evaluation_dir=common.read(profile)['evaluation_dir'],training_updates=0,
            accuracy_reselected=False,metrics=report['arms']['gaucho']))
    except BaseException as error:
        common.save(Path(out)/'failure.json',dict(passed=False,error=repr(error),automatic_retry=False))
        raise


def run(evaluation, out, gpu=0):
    profile = prepare(evaluation, out)
    return run_prepared(profile, out, gpu)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('check','launch','worker'),default='check')
    p.add_argument('--evaluation-dir',type=Path)
    p.add_argument('--out-dir',type=Path)
    p.add_argument('--gpu',type=int,choices=(0,1),default=0)
    p.add_argument('--profile',type=Path)
    a = p.parse_args();os.chdir(ROOT)
    if a.stage == 'check':
        cfg = make_profile(a.evaluation_dir) if a.evaluation_dir else common.read(common.local(TEMPLATE))
        print(json.dumps(dict(passed=True,protocol=cfg['schema'],warmup=50,repeats=3,
            TEST_frames=1440,model_forwards=0,GPU_used=False),indent=2));return
    if a.out_dir is None:p.error('A new --out-dir is required')
    out = a.out_dir.resolve()
    if a.stage == 'worker':
        if a.profile is None:p.error('--worker requires --profile')
        return run_prepared(a.profile,out,a.gpu)
    if a.evaluation_dir is None:p.error('--launch requires a completed --evaluation-dir')
    profile = prepare(a.evaluation_dir,out)
    command=[sys.executable,str(Path(__file__).resolve()),'--stage','worker',
             '--profile',str(profile),'--out-dir',str(out),'--gpu',str(a.gpu)]
    with (out/'supervisor.log').open('x') as log:
        child=subprocess.Popen(command,cwd=str(ROOT),stdin=subprocess.DEVNULL,stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
    result=dict(pid=child.pid,physical_gpu=a.gpu,evaluation_dir=str(a.evaluation_dir.resolve()),
                output=str(out),warmup=50,repeats=3,accuracy_reselected=False)
    common.save(out/'launch.json',result);print(json.dumps(result,indent=2))


if __name__ == '__main__':main()
