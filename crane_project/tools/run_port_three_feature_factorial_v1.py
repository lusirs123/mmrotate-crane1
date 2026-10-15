"""Train six tiny scorers and report archived linear TEST without retuning."""
import argparse
import csv
import gzip
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
import torch
from crane_project.utils import port_three_feature_factorial_v1 as core
from crane_project.utils import port_simple_pilot_increment_v1 as base
from crane_project.utils.port_reliability_paired_ranking_audit_v1 import Ranking, bad

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT/'crane_project/tools/port_three_feature_factorial_v1_manifest.json'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    with p.open('x') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def pins(values):
    for p, h in values.items():
        if sha(ROOT/p) != h:
            raise ValueError('SHA mismatch '+p)


def load(role, manifest):
    cfg = manifest['roles'][role]
    with gzip.open(ROOT/cfg['rows'], 'rt') as f:
        rows = [json.loads(s) for s in f]
    return base.checked_rows(rows, role, cfg['frames'])


def run(out):
    if out.exists() or Path.cwd().resolve() != ROOT:
        raise ValueError('Require new output and project root')
    m = json.loads(MANIFEST.read_text()); pins(m['development_pins'])
    prior = ROOT/m['linear_result']
    previous_models = json.loads((prior/'models.json').read_text())
    previous_freeze = json.loads((prior/'freeze.json').read_text())
    previous_report = json.loads((prior/'report.json').read_text())
    models = {a: previous_models[b] for a, b in (
        ('formal', 'formal'), ('linear_old', 'refit'), ('linear_augmented', 'augmented'))}
    cuts = {a: previous_freeze['cutoffs'][b] for a, b in (
        ('formal', 'formal'), ('linear_old', 'refit'), ('linear_augmented', 'augmented'))}
    train, pilot, val = (load(role, m) for role in ('TRAIN', 'PILOT', 'VAL'))
    if len({r['image'] for r in train+pilot+val}) != len(train+pilot+val):
        raise ValueError('Cross-role overlap')
    for rows, expected in ((train, (2558, 71)), (pilot, (172, 111)), (val, (886, 223))):
        if (sum(r['pred'] is not None for r in rows), sum(bad(r) is True for r in rows)) != expected:
            raise ValueError('Unexpected support')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    out.mkdir(parents=True); write(out/'protocol.json', m)
    materials = {'old': train, 'augmented': train+pilot}
    with (out/'train_log.jsonl').open('x') as f:
        for seed in core.SEEDS:
            checkpoint = {}
            for arm in core.ARMS:
                name = 'mlp_'+arm+'_'+str(seed)
                model, exported, records = core.fit(materials[arm], models['linear_'+arm], seed, m['fitting'])
                models[name] = exported; checkpoint[arm] = model.state_dict()
                for record in records:
                    f.write(json.dumps(dict(method=name, **record), allow_nan=False)+'\n')
                print(name, 'iterations', exported['optimizer_iterations'], 'evaluations', exported['function_evaluations'],
                      'TRAIN_gradient', exported['max_abs_gradient'], 'stationary', exported['converged'], flush=True)
            path = out/('seed_'+str(seed)+'.pth'); torch.save(checkpoint, path)
            restored = torch.load(path, map_location='cpu')
            if not all(torch.equal(v, restored[a][k]) for a, state in checkpoint.items() for k, v in state.items()):
                raise ValueError('Checkpoint reload mismatch')
    roles = dict(TRAIN=train, PILOT_TRAIN=pilot, COMBINED_TRAIN=train+pilot, VAL=val)
    for rows in (train, pilot, val):
        core.add_scores(rows, models)
    for name in models:
        if name.startswith('mlp_'):
            cuts[name] = Ranking(val, name).same_cr(653)['risk_le']
    results = {role: {name: base.diagnostics(rows, name, cuts) for name in models}
               for role, rows in roles.items()}
    for new, old in (('formal', 'formal'), ('linear_old', 'refit'), ('linear_augmented', 'augmented')):
        for role in roles:
            if results[role][new] != previous_report['results'][role][old]:
                raise ValueError('Archived linear comparison changed '+role+'/'+new)
    decision = core.gate(results['VAL'], models)
    write(out/'models.json', models)
    write(out/'freeze.json', dict(decision=decision, cutoffs=cuts,
        models_sha256=sha(out/'models.json'), protocol_sha256=sha(MANIFEST),
        checkpoint_sha256={str(s): sha(out/('seed_'+str(s)+'.pth')) for s in core.SEEDS},
        TEST_linear_supplement='Approved reporting extension to prior VAL_FAILED_STOP, no model/cutoff edits',
        all_training_complete_before_TEST=True))
    # TEST is opened only after the entire design, all trained models, gate and
    # cutoffs are frozen. Linear reporting proceeds regardless of the MLP gate.
    pins(m['test_pins']); test = load('TEST', m)
    test_models = {k: v for k, v in models.items() if k in ('formal', 'linear_old', 'linear_augmented')}
    if decision['passed']:
        test_models.update({k: models[k] for k in ('mlp_old_1701', 'mlp_augmented_1701')})
    core.add_scores(test, test_models); roles['TEST'] = test
    results['TEST'] = {name: base.diagnostics(test, name, cuts) for name in test_models}
    for role in ('TRAIN', 'PILOT_TRAIN', 'VAL', 'TEST'):
        with gzip.open(out/('scored_'+role+'.jsonl.gz'), 'xt') as f:
            for row in roles[role]: f.write(json.dumps(row, allow_nan=False)+'\n')
    runtime = dict(cwd=str(ROOT), environment=os.environ.get('CONDA_DEFAULT_ENV'),
        python=platform.python_version(), torch=torch.__version__, device='CPU float64',
        HEAD=subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip())
    report = dict(status='MLP_VAL_PASS_TEST_REPORTED' if decision['passed'] else 'MLP_VAL_FAILED_LINEAR_TEST_REPORTED',
        results=results, decision=decision, cutoffs=cuts, runtime=runtime,
        linear_TEST_reported=True, MLP_TEST_reported=decision['passed'],
        MLP_heads_fitted=6, detector_updates=0, detector_inference=0, feature_extraction=0,
        center_angle_refitted=False, test_repeatedly_exposed=True, protocol_sha256=sha(MANIFEST))
    write(out/'report.json', report)
    with (out/'metrics.csv').open('x', newline='') as f:
        keys = ['role', 'method', 'group', 'FA', 'FR', 'ED', 'CR', 'MISSING',
            'correct_retention', 'accepted_error_rate', 'acceptance_coverage', 'output_coverage',
            'full_frame_correct_coverage', 'longest_correct_FR', 'longest_FR_observations',
            'longest_without_CR_observations']
        writer = csv.DictWriter(f, keys); writer.writeheader()
        for role, methods in results.items():
            for name, values in methods.items():
                for group, d in values['at_frozen_VAL_cut'].items():
                    row = dict(role=role, method=name, group=group, **d['states'])
                    row.update({k: d[k] for k in keys if k in d}); writer.writerow(row)
    lines = ['# 数据×三维判断器形式对照', '', report['status'], '',
        '|角色|方法|FA|FR|ED|CR|Real CR|最长FR观测|', '|---|---|---:|---:|---:|---:|---:|---:|']
    for role in ('VAL', 'TEST'):
        for name, values in results[role].items():
            d = values['at_frozen_VAL_cut']; s = d['all']['states']
            lines.append('|%s|%s|%d|%d|%d|%d|%d|%d|' % (role, name,
                *(s[k] for k in ('FA', 'FR', 'ED', 'CR')), d['domain:real']['states']['CR'],
                d['all']['longest_FR_observations']))
    lines += ['', '41参数3→8→1；仅三维输入。固定3种子、相同TRAIN目标/L2系数，线性Newton与MLP L-BFGS不同，不称纯容量消融。',
        '各臂独立TRAIN归一化与原类别平衡；不添加图像/时序/视频ID。',
        '主点VAL统一CR>=653、Real CR>=189；增量MLP三个种子均需优于原数据MLP与正式simple，且TRAIN近似驻点。',
        'TEST只作已暴露报告；读取前全部模型/门限/决定已冻结，无后续训练或挑seed。',
        '线性TEST为本轮补充；MLP TEST读取：'+str(decision['passed']), json.dumps(decision, ensure_ascii=False)]
    (out/'analysis.md').write_text('\n'.join(lines)+'\n')
    pins(m['development_pins'])
    write(out/'completion.json', dict(status=report['status'], artifacts={p.name: sha(p) for p in out.iterdir() if p.is_file()}))
    print(report['status'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
    run(p.parse_args().out.resolve())
