"""Two deterministic CPU size-only fits from sealed predictions."""
import argparse
import csv
import gzip
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
import numpy as np
from crane_project.utils import port_simple_pilot_increment_v1 as core
from crane_project.utils.port_reliability_paired_ranking_audit_v1 import Ranking, bad

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT/'crane_project/tools/port_simple_pilot_increment_v1_manifest.json'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    with p.open('x') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def checkpins(pins):
    for p, h in pins.items():
        if sha(ROOT/p) != h:
            raise ValueError('Input/source SHA differs '+p)


def load(role, manifest):
    cfg = manifest['roles'][role]
    with gzip.open(ROOT/cfg['rows'], 'rt') as f:
        rows = [json.loads(line) for line in f]
    return core.checked_rows(rows, role, cfg['frames'])


def run(out):
    if out.exists() or Path.cwd().resolve() != ROOT:
        raise ValueError('Require new output and project working directory')
    m = json.loads(MANIFEST.read_text()); checkpins(m['development_pins'])
    policy = json.loads((ROOT/m['policy']).read_text())
    train, pilot, val = (load(role, m) for role in ('TRAIN', 'PILOT', 'VAL'))
    ids = [r['image'] for r in train+pilot+val]
    if len(set(ids)) != len(ids):
        raise ValueError('Cross-role image identity overlap')
    for rows, expected in ((train, (2558, 71)), (pilot, (172, 111)), (val, (886, 223))):
        if (sum(r['pred'] is not None for r in rows), sum(bad(r) is True for r in rows)) != expected:
            raise ValueError('Unexpected support')
    # Model architecture, objective and optimizer remain the original implementation.
    models = dict(formal=policy['simple_policy']['models']['size'],
                  refit=core.fit(train, m['fitting']),
                  augmented=core.fit(train+pilot, m['fitting']))
    replay = {k: float(np.max(np.abs(np.asarray(models['refit'][k])-np.asarray(models['formal'][k]))))
              for k in ('mean', 'scale', 'weights')}
    if max(replay.values()) > 1e-10:
        raise ValueError('Original TRAIN fit not reproduced')
    for rows in (train, pilot, val):
        core.add_scores(rows, models)
        for r in rows:
            d = r.get('original_simple_decision', r.get('decision'))
            if r['pred'] is not None:
                if abs(r['audit_risks']['formal']-d['risks']['size']) > 1e-12:
                    raise ValueError('Original frozen risk changed')
            flag = r['pred'] is not None and r['audit_risks']['formal'] <= m['formal_cutoff']
            if bool(flag) != d['size_accepted']:
                raise ValueError('Original frozen flag changed')
    cuts = {'formal': m['formal_cutoff']}
    for method in ('refit', 'augmented'):
        cuts[method] = Ranking(val, method).same_cr(m['primary_VAL_CR'])['risk_le']
    roles = dict(TRAIN=train, PILOT_TRAIN=pilot, COMBINED_TRAIN=train+pilot, VAL=val)
    results = {role: {name: core.diagnostics(rows, name, cuts) for name in core.METHODS}
               for role, rows in roles.items()}
    base = results['VAL']['formal']['at_frozen_VAL_cut']['all']['states']
    if base != dict(FA=190, FR=10, ED=33, CR=653, MISSING=1):
        raise ValueError('Formal VAL counts not reproduced')
    choice = core.decision(results['VAL'])
    out.mkdir(parents=True)
    write(out/'protocol.json', m); write(out/'models.json', models)
    # No TEST path is opened/hashed before this completed freeze record.
    write(out/'freeze.json', dict(decision=choice, cutoffs=cuts,
          model_sha256=sha(out/'models.json'), protocol_sha256=sha(MANIFEST),
          role='experimental size scorer only; formal policy unchanged'))
    if choice['passed']:
        checkpins(m['test_pins'])
        test = load('TEST', m); core.add_scores(test, models)
        roles['TEST'] = test
        results['TEST'] = {name: core.diagnostics(test, name, cuts) for name in core.METHODS}
    runtime = dict(cwd=str(ROOT), environment=os.environ.get('CONDA_DEFAULT_ENV'),
                   python=platform.python_version(), numpy=np.__version__, device='CPU',
                   HEAD=subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip())
    report = dict(status='VAL_PASS_TEST_REPORTED' if choice['passed'] else 'VAL_FAILED_STOP',
        results=results, decision=choice, cutoffs=cuts, refit_replay_max_abs=replay,
        TEST_read=choice['passed'], detector_inference=0, detector_updates=0,
        feature_extraction=0, size_models_fitted=2, center_angle_refitted=False,
        runtime=runtime, protocol_sha256=sha(MANIFEST))
    write(out/'report.json', report)
    with (out/'metrics.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, ['role', 'method', 'group', 'FA', 'FR', 'ED', 'CR', 'MISSING',
            'correct_retention', 'accepted_error_rate', 'acceptance_coverage',
            'output_coverage', 'full_frame_correct_coverage', 'longest_correct_FR',
            'longest_FR_observations', 'longest_without_CR_observations'])
        writer.writeheader()
        for role, methods in results.items():
            for name, d in methods.items():
                for group, v in d['at_frozen_VAL_cut'].items():
                    row = dict(role=role, method=name, group=group, **v['states'])
                    row.update({k: v[k] for k in writer.fieldnames if k in v})
                    writer.writerow(row)
    for role in ('TRAIN', 'PILOT_TRAIN', 'VAL', 'TEST'):
        if role not in roles:
            continue
        with gzip.open(out/('scored_'+role+'.jsonl.gz'), 'xt') as f:
            for row in roles[role]:
                f.write(json.dumps(row, allow_nan=False)+'\n')
    lines = ['# simple新增pilot训练材料对照', '', report['status'], '',
             '|角色|模型|FA|FR|ED|CR|MISSING|Real CR|最长FR观测|',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for role, methods in results.items():
        for name, d in methods.items():
            a = d['at_frozen_VAL_cut']; s = a['all']['states']
            lines.append('|%s|%s|%d|%d|%d|%d|%d|%d|%d|' % (
                role, name, *(s[k] for k in ('FA', 'FR', 'ED', 'CR', 'MISSING')),
                a['domain:real']['states']['CR'], a['all']['longest_FR_observations']))
    lines += ['', '相同3维、原拟合规则；各自TRAIN归一化和类别平衡随材料变化。',
        '完整VAL统一CR>=653门限；正式simple原门限保留。未按视频选门限。',
        '95%及97%辅助点/同接受数/同CR/完整曲线见report，不替代主保护点。',
        '新pilot是单视频相关采样，主要偏大错误；训练内改善不能当跨视频收益。',
        'TEST读取：'+str(choice['passed'])+'；通过前先冻结，TEST已多次暴露、不用于调参。',
        json.dumps(choice, ensure_ascii=False)]
    (out/'analysis.md').write_text('\n'.join(lines)+'\n')
    checkpins(m['development_pins'])
    write(out/'completion.json', dict(status=report['status'],
          artifacts={p.name: sha(p) for p in out.iterdir() if p.is_file()}))
    print(report['status'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out.resolve())
