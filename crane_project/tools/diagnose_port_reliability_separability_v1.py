#!/usr/bin/env python3
"""CPU check/run of frozen simple and score discrimination on existing TRAIN/VAL.

No GPU, inference, fitting, TEST input, cutoff selection, or new deployment
policy. Defaults reuse the same reviewed numeric bundles as simple-v1 fitting.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import run_port_simple_reliability_v1 as parent
from crane_project.utils import port_reliability_separability_v1 as diagnostic
from crane_project.utils import port_simple_component_reliability_v1 as simple

PROTOCOL = ROOT/'crane_project/tools/port_reliability_separability_v1_protocol.json'
MANIFEST = ROOT/'crane_project/tools/port_reliability_separability_v1_sources.json'


def checked_sources():
    original, original_sources = parent.checked_sources()
    contract = json.loads(PROTOCOL.read_text())
    manifest = json.loads(MANIFEST.read_text())
    current = {name: parent.sha(ROOT/name) for name in manifest['sources']}
    if (manifest['protocol'] != diagnostic.VERSION or contract['protocol'] != diagnostic.VERSION
            or current != manifest['sources'] or contract['parent_protocol_sha256'] != parent.sha(parent.PROTOCOL)
            or contract['parent_manifest_sha256'] != original_sources['manifest_sha256']
            or contract['roles'] != ['train', 'val'] or contract['methods'] != list(diagnostic.METHODS)
            or contract['fit_or_select_cutoff'] is not False or contract['test_read'] is not False):
        raise ValueError('Reviewed diagnostic or unchanged parent sources/protocol differ')
    return contract, original, dict(manifest_sha256=parent.sha(MANIFEST), sources=current,
                                   parent=original_sources)


def checked_policy(path, contract, original, sources):
    if path.name != 'policy.json' or (path.parent/'failure.json').exists():
        raise ValueError('Use the completed original policy.json bundle')
    actual = {name: parent.sha(path.parent/name) for name in contract['reviewed_policy_files']}
    if actual != contract['reviewed_policy_files']:
        raise ValueError('Requires exact reviewed frozen policy bundle bytes')
    policy = parent.load_policy(path, sources['parent'], original)
    return policy, actual


def inputs(args, contract, original, sources):
    train, val, proof = parent.checked_inputs(args, original)
    policy, policy_hashes = checked_policy(args.policy, contract, original, sources)
    before = simple.fingerprint([train, val, policy])
    # Validate saved numeric GT qualifications even when a frame has no output.
    for role, rows in (('train', train), ('val', val)):
        diagnostic.prepare_rows(rows, policy)
        if role == 'train' and sum(r['train_angle_eligible'] for r in rows) != 2558:
            raise ValueError('Original all-frame TRAIN direction supervision differs')
    if before != simple.fingerprint([train, val, policy]):
        raise ValueError('Input validation changed frozen records')
    proof.update(policy_files_sha256=policy_hashes,
        cached_numeric_rows_and_policy_sha256=before,
        training_direction_eligible=sum(r['train_angle_eligible'] for r in train),
        evaluation_direction_eligible_train=sum(r['angle_axis_well_defined'] for r in train),
        evaluation_direction_eligible_val=sum(r['angle_axis_well_defined'] for r in val),
        actual_train_label_direction_stats_are_separate=True)
    return train, val, policy, proof


def replay_saved_val(prepared, policy_path, tolerance):
    saved = [json.loads(line) for line in (policy_path.parent/'val_decisions.jsonl').read_text().splitlines()]
    if len(saved) != len(prepared):
        raise ValueError('Saved VAL decision count differs')
    max_difference = 0.
    for source, reference in zip(prepared, saved):
        if source['image'] != reference['image']:
            raise ValueError('Saved VAL decision identity/order differs')
        for method in ('raw',)+diagnostic.METHODS:
            actual, old = source['methods'][method], reference['methods'][method]
            for key in ('raw_b_output', 'center_accepted', 'size_accepted', 'angle_accepted'):
                if actual[key] != old[key]:
                    raise ValueError('Frozen VAL box/flag replay differs: '+source['image'])
            for component in ('size', 'angle'):
                a, b = actual['risks'][component], old['risks'][component]
                if a is None or b is None:
                    if a is not None or b is not None:
                        raise ValueError('Missing-output risk differs')
                else:
                    difference = abs(a-b)
                    if difference > tolerance:
                        raise ValueError('Frozen VAL floating risk replay differs')
                    max_difference = max(max_difference, difference)
    return dict(frames=len(saved), original_boxes_and_all_flags_exact=True,
        max_abs_risk_difference=max_difference, risk_only_float64_tolerance=tolerance)


def flatten_point(point):
    flat = {k: v for k, v in point.items() if k != 'same_count_score'}
    flat.update({'same_count_score_'+k: v for k, v in point['same_count_score'].items()})
    return flat


def write_csv(path, records):
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)


def summary_rows(strata):
    records = []
    for role, groups in strata.items():
        for group, summary in groups.items():
            for component, result in summary['components'].items():
                for method, statistics in result['methods'].items():
                    records.append(dict(role=role, group=group, component=component, method=method,
                        frames=result['frames'], eligible_frames=result['eligible_frames'],
                        missing_eligible_outputs=result['missing_eligible_outputs'],
                        **statistics['rank'], **{k: v for k, v in statistics['fixed_workpoint'].items()
                                                if k != 'same_count_score'}))
    return records


def review_text(report):
    def number(x):
        return 'NA' if x is None else '%.4f' % x
    text = ['# 冻结TRAIN/VAL评分区分能力检查', '',
        '本页只描述已有simple与检测score；没有训练、重新推理、选择门限或生成新policy。',
        'TRAIN为拟合内描述；VAL此前参与前端选择和工作点校准，本次是开发诊断，均非独立确认。',
        'TEST已多次暴露；本入口不接收或读取TEST。', '',
        '错误为正类，风险越高越倾向拒绝。AP为按相同分数整组积分的非插值平均精确率；ROC面积按同分组计算。',
        '单类或没有输出的组不能支持区分能力结论；AP需同时参考错误占比。', '',
        '| 数据/域 | 分量 | 错误/可评输出 | simple AUROC/AP | score AUROC/AP | 冻结simple错误接受/正确误拒 |',
        '|---|---|---:|---:|---:|---:|']
    for role in ('train', 'val'):
        for domain in ('real', 'sim'):
            s = report['strata'][role]['domain:'+domain]
            for component in ('size', 'angle'):
                c = s['components'][component]; a = c['methods']['simple']; b = c['methods']['score_only']
                f = a['fixed_workpoint']
                text.append('| %s/%s | %s | %d/%d | %s/%s | %s/%s | %d/%d |' % (
                    role, domain, component, c['bad_outputs'], c['assessed_outputs'],
                    number(a['rank']['error_auroc']), number(a['rank']['error_average_precision']),
                    number(b['rank']['error_auroc']), number(b['rank']['error_average_precision']),
                    f['incorrect_accepted'], f['correct_rejected']))
    text.extend(['', '角度主表只评GT aspect≥1.2的输出；TRAIN全训练资格结果另列angle_training_qualification，不能混用计数。',
        '缺输出另报，不作为正确拒绝；不可评方向保留原在线标志，但不进入角度错误分类。',
        '各组保留原全局冻结门限，不使用分域/视频门限。曲线只在可评输出上整组接受同分值；各点不是部署方案。',
        'curve_points.csv保存错误接受、正确误拒、错误检出、正确保留、接受后正确率与不同覆盖分母。',
        '同接受数score按risk、image排序，仅作离线对照；另外报告切开同分组的错误数范围，不能据GT选择同分帧。',
        '输入角度变化不会改变现simple风险；未因此证明跨视频问题的唯一根因。风险不是校准错误概率。', '',
        '## 下一步判读', '',
        '先看VAL分域和逐视频区分是否一致，再看当前工作点相对于曲线的位置。不能仅凭pooled面积或TRAIN拟合内结果放行。',
        '若评分具有区分能力而工作点过宽，后续版本再预先固定误判代价和选择规则；本报告不输出最佳门限。',
        '若评分缺乏区分能力，单调校准或收紧门限不能增加排序信息，下一项才限定到尺寸的独立二维图像范围一致性。',
        '中心继续保留B输出；尺寸/方向拒绝不删除中心。始终分别报告输出覆盖、输出帧中心命中和全帧中心正确覆盖。',
        '视频帧相关、TRAIN错误稀少；没有新增独立视频、OOF、置信区间、显著性或深度精度结论。'])
    return '\n'.join(text)+'\n'


def run(args, train, val, policy, proof, contract, sources):
    before = simple.fingerprint([train, val, policy])
    strata, curves, val_prepared = {}, [], None
    for role, rows in (('train', train), ('val', val)):
        strata[role], points, prepared = diagnostic.analyze(rows, policy, role)
        curves.extend(points)
        if role == 'val':
            val_prepared = prepared
    parity = replay_saved_val(val_prepared, args.policy, contract['risk_replay_abs_tolerance'])
    if simple.fingerprint([train, val, policy]) != before:
        raise ValueError('Policy or original numeric records changed')
    report = dict(protocol=diagnostic.VERSION,
        status='FROZEN_TRAIN_VAL_SCORE_DIAGNOSTICS_COMPLETE_REVIEW_REQUIRED', sources=sources,
        fixed_protocol=contract, input_proof=proof, saved_val_replay=parity,
        strata=strata, curve_points=len(curves), detector_updates=0, reliability_updates=0,
        detector_inferences=0, GPU_used=False, test_read=False, new_policy_created=False,
        cutoff_or_checkpoint_selected=False, independent_confirmation=False,
        original_numeric_GT_and_B_unchanged=True,
        limitations=['TRAIN is in-sample; VAL is reused development/calibration evidence.',
            'Correlated video frames and sparse genuine TRAIN errors do not establish stable/significant gains.',
            'Whole-tie curves and exact-count image tie diagnostics have different deployment semantics.',
            'Risk values are not calibrated error probabilities; no architecture or threshold is selected.',
            'Current dataset/image bytes are not rehashed; exact previously reviewed numeric records are reused.',
            'TEST is repeatedly exposed but not read here; no new depth or geometry accuracy is established.'])
    parent.write_new(args.out_dir/'report.json', report)
    write_csv(args.out_dir/'curve_points.csv', [flatten_point(p) for p in curves])
    write_csv(args.out_dir/'summary.csv', summary_rows(strata))
    with (args.out_dir/'review.md').open('x', encoding='utf-8') as stream:
        stream.write(review_text(report))
    # Recheck artifact bytes and source seals before publishing completion.
    _, _, _, after = inputs(args, contract, contract_original(sources), sources)
    if after != proof or checked_sources()[2] != sources:
        raise ValueError('Execution sources or input artifact bytes changed during run')
    paths = ('input_check.json', 'report.json', 'curve_points.csv', 'summary.csv', 'review.md')
    parent.write_new(args.out_dir/'completion.json', dict(protocol=diagnostic.VERSION,
        status='CPU_FROZEN_TRAIN_VAL_DIAGNOSTICS_COMPLETE_REVIEW_REQUIRED',
        files_sha256={p: parent.sha(args.out_dir/p) for p in paths}, sources=sources,
        policy_unchanged=True, cached_numeric_inputs_unchanged=True, new_policy_created=False,
        parameter_updates=0, detector_inferences=0, GPU_used=False, test_read=False))
    for domain in ('real', 'sim'):
        s = strata['val']['domain:'+domain]
        print('VAL', domain, 'outputs', s['output_frames'], '/', s['frames'],
              'center/output', s['center_hit_rate_on_outputs'],
              'full-frame center', s['all_frame_center_correct_coverage'], flush=True)
        for component in ('size', 'angle'):
            c = s['components'][component]
            print(' ', component, 'bad/assessed', c['bad_outputs'], '/', c['assessed_outputs'],
                'simple AUC/AP', c['methods']['simple']['rank']['error_auroc'],
                c['methods']['simple']['rank']['error_average_precision'],
                'score AUC/AP', c['methods']['score_only']['rank']['error_auroc'],
                c['methods']['score_only']['rank']['error_average_precision'], flush=True)


def contract_original(sources):
    original, original_sources = parent.checked_sources()
    if original_sources != sources['parent']:
        raise ValueError('Parent contract changed during run')
    return original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('check', 'run'), required=True)
    parser.add_argument('--train-cache', type=Path, default=Path('work_dirs/port_reliability_train_support_v1_cache'))
    parser.add_argument('--input-snapshot', type=Path, default=Path('work_dirs/port_reliability_train_support_v1/train_input_snapshot.json'))
    parser.add_argument('--val-dir', type=Path, default=Path('work_dirs/port_reliability_branches_v1_val_cached_v1'))
    parser.add_argument('--policy', type=Path, default=Path('work_dirs/port_simple_reliability_v1_fit/policy.json'))
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args(); os.chdir(ROOT)
    if args.out_dir.exists():
        raise FileExistsError('Use a NEW --out-dir; preserve existing evidence: '+str(args.out_dir))
    contract, original, sources = checked_sources()
    train, val, policy, proof = inputs(args, contract, original, sources)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    try:
        parent.write_new(args.out_dir/'input_check.json', dict(protocol=diagnostic.VERSION,
            status='FROZEN_TRAIN_VAL_INPUT_CHECK_PASS_NO_FIT_NO_INFERENCE',
            sources=sources, fixed_protocol=contract, input_proof=proof,
            parameter_updates=0, detector_inferences=0, GPU_used=False, test_read=False))
        if args.mode == 'run':
            run(args, train, val, policy, proof, contract, sources)
    except Exception as error:
        parent.write_new(args.out_dir/'failure.json', dict(status='FAILED_PRESERVE_OUTPUTS',
            error=str(error), mode=args.mode))
        raise
    print('Saved', args.out_dir, 'mode', args.mode, 'CPU only; no fitting/inference/TEST', flush=True)


if __name__ == '__main__':
    main()
