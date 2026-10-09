#!/usr/bin/env python3
"""Hash-bound, CPU-only complementarity audit of frozen TRAIN/VAL scores."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.utils import port_reliability_complementarity_v1 as core
from crane_project.utils import port_reliability_state_continuity_v1 as continuity

CONTRACT = ROOT/'crane_project/tools/port_reliability_complementarity_v1_protocol.json'
SOURCES = ROOT/'crane_project/tools/port_reliability_complementarity_v1_sources.json'
FIELDS = ('image','sequence','domain','split','frame_id','image_size','gt','pred',
          'train_angle_eligible','angle_axis_well_defined','b_original','reliability_role',
          'size_risks','original_simple_decision')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def checked():
    contract = json.loads(CONTRACT.read_text())
    manifest = json.loads(SOURCES.read_text())
    if contract['protocol'] != core.VERSION or manifest['protocol'] != core.VERSION:
        raise ValueError('Protocol differs')
    for file, pin in dict(contract['input_pins'], **manifest['sources']).items():
        if sha(ROOT/file) != pin:
            raise ValueError('Pinned bytes changed: '+file)
    return contract, manifest


def jsonl(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def validate_rows(rows, role):
    continuity.checked_order(rows)
    if any(r['reliability_role'] != role.lower() for r in rows):
        raise ValueError('TRAIN/VAL identity changed')
    for r in rows:
        if r['domain'] not in ('real', 'sim'):
            raise ValueError('Unknown domain')
        if r['pred'] is not None:
            p, gt = r['pred'], r['gt']
            if len(p) != 6 or len(gt) != 5 or any(not math.isfinite(v) for v in p+gt+r['image_size']):
                raise ValueError('Invalid box/GT')
            if not 0 <= p[5] <= 1 or min(r['image_size']) <= 0:
                raise ValueError('Invalid score/image size')
            core.bad(r)


def load(contract):
    original = jsonl(ROOT/contract['primary_rows'])
    sources = {s['name']: json.loads((ROOT/s['directory']/'report.json').read_text())
               for s in contract['experiments']}
    for name, report in sources.items():
        if report['TEST_read'] or report['original_policy_changed'] or report['status'] != 'VAL_FAILED_STOP':
            raise ValueError('Unexpected historical status: '+name)
        if not report['boxes_scores_output_center_angle_unchanged']:
            raise ValueError('Historical front end differs')
    policy = json.loads((ROOT/contract['policy']).read_text())
    cutoffs = dict(formal_simple=policy['simple_policy']['cutoffs']['simple']['size']['risk_le'])
    simple95 = [report['VAL_cutoffs']['full_simple']['risk_le'] for report in sources.values()]
    if len(set(simple95)) != 1:
        raise ValueError('Shared frozen simple retention cutoff differs')
    cutoffs['simple_retention95'] = simple95[0]
    parts = {}; maximum_replay_difference = 0.
    for role in ('TRAIN', 'VAL'):
        rows = sorted([{k: r[k] for k in FIELDS} for r in original if r['reliability_role'] == role.lower()], key=lambda r: r['image'])
        validate_rows(rows, role)
        expected = contract['role_counts'][role]
        if len(rows) != expected['frames'] or Counter(r['domain'] for r in rows) != expected['domains']:
            raise ValueError('Complete split counts differ')
        if sum(core.bad(r) is True for r in rows) != expected['bad']:
            raise ValueError('Offline truth differs')
        by_id = {r['image']: r for r in rows}
        for r in rows:
            r['risks'] = dict(full_simple=r['size_risks']['full_simple'],
                             score_only=None if r['pred'] is None else 1-r['pred'][5])
            if r['pred'] is not None:
                p = r['pred']; long, short = sorted(p[2:4], reverse=True)
                score = max(1e-6, min(1-1e-6, p[5]))
                x = [math.log(score)-math.log1p(-score),
                    .5*(math.log(long)+math.log(short)-math.log(r['image_size'][0])-math.log(r['image_size'][1])),
                    math.log(long)-math.log(short)]
                m = policy['simple_policy']['models']['size']
                z = m['weights'][0]+sum((v-a)/b*w for v, a, b, w in zip(x, m['mean'], m['scale'], m['weights'][1:]))
                value = 1/(1+math.exp(-z)) if z >= 0 else math.exp(z)/(1+math.exp(z))
                error = abs(value-r['risks']['full_simple']); maximum_replay_difference = max(maximum_replay_difference, error)
                if error > 1e-12:
                    raise ValueError('Formal risk replay differs')
            d = r['original_simple_decision']
            if d['center_accepted'] != (r['pred'] is not None) or d['final_box_original'] != r['pred']:
                raise ValueError('Center/delivered box identity differs')
            keep = r['pred'] is not None and r['risks']['full_simple'] <= cutoffs['formal_simple']
            if d['size_accepted'] != keep:
                raise ValueError('Original policy threshold differs')
        for experiment in contract['experiments']:
            saved = jsonl(ROOT/experiment['directory']/('scored_'+role+'.jsonl.gz'))
            validate_rows(saved, role)
            if {r['image'] for r in saved} != set(by_id):
                raise ValueError('Historical full split membership differs')
            for r in saved:
                target = by_id[r['image']]
                if any(target[k] != r[k] for k in FIELDS):
                    raise ValueError('Historical labels/boxes/flags differ: '+r['image'])
                for method, arm in experiment['methods'].items():
                    value = r['experiment_risks'][arm]
                    if (value is None) != (r['pred'] is None) or (value is not None and (not math.isfinite(value) or not 0 <= value <= 1)):
                        raise ValueError('Invalid/missing candidate risk')
                    target['risks'][method] = value
                    threshold = sources[experiment['name']]['VAL_cutoffs'][arm]['risk_le']
                    cutoffs[method] = threshold
                    if r['candidate_decisions'][arm]['size_accepted'] != (value is not None and value <= threshold):
                        raise ValueError('Historical candidate cutoff replay differs')
                if any(r['experiment_risks'][m] != target['risks'][m] for m in core.CONTROLS):
                    raise ValueError('Historical shared controls differ')
        parts[role] = rows
    return parts, cutoffs, maximum_replay_difference


def analyze(parts, cutoffs, methods):
    result = {}
    for role, rows in parts.items():
        result[role] = dict(center={name: core.center(group) for name, group in core.grouped(rows).items()},
                            references={}, original_flag_identity_preserved=True)
        for reference in core.REFERENCES:
            baseline = core.accepted(rows, 'full_simple', cutoffs[reference])
            evaluations = {}
            for method in methods:
                candidate = core.accepted(rows, method, cutoffs[method])
                evaluations[method] = {name: core.complement(group, baseline, candidate, method)
                                      for name, group in core.grouped(rows).items()}
            oracle_union = {}
            for name, group in core.grouped(rows).items():
                ids = {r['image'] for r in group}; rescued = set(); restored = set()
                for values in evaluations.values():
                    rescued.update(values[name]['change_images']['rescued_FA'])
                    restored.update(values[name]['change_images']['restored_CR'])
                oracle_union[name] = dict(rescued_FA=len(rescued), restored_CR=len(restored),
                    rescued_images=sorted(rescued & ids), restored_images=sorted(restored & ids),
                    executable_policy=False, GT_selects_changes=True, methods_not_selected=True)
            result[role]['references'][reference] = dict(methods=evaluations, GT_oracle_union=oracle_union)
    gates = {method: core.gate(result['VAL']['references']['simple_retention95']['methods'][method]) for method in methods}
    return result, gates


def markdown(report):
    lines = ['**本次只核对已有评分的互补性；未训练、未重新推理、未读取TEST，正式policy保持。**', '',
        '主要参考为历史已冻结的simple各组正确保留≥95%工作点；正式policy原点单列，门限不重选。', '',
        '| 完整VAL，simple95基准 | 补充检出FA | 新增FR | 恢复CR | 新增FA | AND的FA/FR | 同CR simple FA | 证据门槛 |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    vals = report['results']['VAL']['references']['simple_retention95']['methods']
    for method, groups in vals.items():
        value = groups['all']; c = value['changes']; a = value['reject_only_AND']['states']
        match = value['AND_comparison']['full_simple']['same_CR']['states']
        lines.append('|%s|%d|%d|%d|%d|%d/%d|%d|%s|' % (method, c['rescued_FA'], c['new_FR'],
            c['restored_CR'], c['new_FA'], a['FA'], a['FR'], match['FA'],
            '通过必要条件' if report['evidence_gates'][method]['passed'] else '未通过'))
    baseline = next(iter(vals.values()))['all']['baseline']
    lines += ['', 'simple95基准：'+str(baseline['states'])+'。', '',
        '| VAL逐域/视频，固定AND诊断 | 方法 | 补充检出/新增FR | AND FA/FR | 正确保留 | 同CR simple FA |',
        '|---|---|---:|---:|---:|---:|']
    for method, groups in vals.items():
        for name, value in groups.items():
            if name == 'all': continue
            c = value['changes']; a = value['reject_only_AND']; match = value['AND_comparison']['full_simple']['same_CR']['states']
            lines.append('|%s|%s|%d/%d|%d/%d|%.2f%%|%d|' % (name, method, c['rescued_FA'], c['new_FR'],
                a['states']['FA'], a['states']['FR'], 100*a['good_retention'], match['FA']))
    lines += ['', '**解释边界。** AND仅是已有两条冻结规则的离线交集，未交付新policy。',
        '补充检出不等于净收益：它可能同时误拒正确框。同数/同CR对照以及连续正确FR已全部报告；各组不能相加当总体。',
        'GT oracle只展示能由真值挑出的最佳改动上限，不是在线可实现的收益；不据此通过继续条件。',
        'TRAIN评分均由在该TRAIN拟合过的模型产生，只是拟合诊断，不提供独立泛化证据。',
        'VAL及这些模型已多次用于开发，本次是历史评分的描述性核对，不构成独立验证或统计风险保证。',
        '固定AND门槛失败只能说明当前门限交集没有稳定收益，不能证明所有融合方案或图像信息都无效。', '',
        '**决定。** '+report['status']+'；没有模型选择或采用，不进入TEST，不自动训练融合头。']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    contract, manifest = checked()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    parts, cutoffs, maximum = load(contract)
    results, gates = analyze(parts, cutoffs, contract['methods'])
    report = dict(protocol=core.VERSION, contract=contract, sources=manifest,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT),text=True).strip(),
        input_pins=contract['input_pins'], cutoffs=cutoffs, results=results, evidence_gates=gates,
        status='COMPLEMENTARITY_SIGNAL_REQUIRES_NEW_DESIGN' if any(g['passed'] for g in gates.values()) else 'NO_STABLE_FROZEN_OVERLAY_KEEP_SIMPLE',
        selected_method=None, candidate_adopted=False, GT_online=False, TEST_read=False,
        TEST_repeatedly_exposed=True, training_updates_added=0, detector_inferences_added=0,
        feature_forwards_added=0, thresholds_fitted=0, original_policy_changed=False,
        maximum_simple_risk_replay_difference=maximum,
        boxes_scores_center_angle_output_unchanged=True, new_data_roles=False)
    write(out/'report.json', report)
    (out/'analysis.md').write_text(markdown(report))
    for role, rows in parts.items():
        with gzip.open(out/('joined_'+role+'.jsonl.gz'), 'xt') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    checked()
    write(out/'completion.json',dict(artifacts={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()},
                                    input_bytes_preserved=True))
    print(report['status'], 'TRAIN', len(parts['TRAIN']), 'VAL', len(parts['VAL']), flush=True)


if __name__ == '__main__':
    main()
