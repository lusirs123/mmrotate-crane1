"""Hash-bound, offline-only focused paper archive. Compatible with Python 3.8."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from crane_project.tools import base_v3_obb_hybrid_fixed_test_diagnostic_v51 as diag
from crane_project.tools.base_v3_obb_reliability_baseline import _write_exact

PROTOCOL = 'base_v3_obb_focused_paper_result_archive_v31_r2'
CONTRACT_PROTOCOL = PROTOCOL + '_contract'
RECLASSIFIED_DIAGNOSTIC_FILENAME = (
    'fixed_test_obb_hybrid_policy_diagnostic_v51_v31_r1_reclassified.json')
ARCHIVE_FILENAME = 'fixed_test_focused_paper_result_archive_v31_r2.json'
MARKDOWN_FILENAME = 'fixed_test_focused_paper_result_archive_v31_r2.md'


def load_bound(path, expected):
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected['sha256']:
        raise ValueError('SHA256 mismatch: ' + str(path))
    data = json.loads(raw)
    if data['protocol'] != expected['protocol']:
        raise ValueError('Protocol mismatch: ' + str(path))
    return data, dict(sha256=digest, size_bytes=len(raw), filename=Path(path).name)


def revision_from_history(history, identity):
    for group in history['scale_matched_coverage'].values():
        for point in group['points']:
            score, risk = point['anchor_score'], point['learned_scale_risk']
            if score['retained_count'] != risk['retained_count']:
                raise ValueError('Unmatched retained counts')
            for metric in ('mean_error', 'failure_rate'):
                if point['learned_minus_score'][metric] != risk[metric] - score[metric]:
                    raise ValueError('Inconsistent historical raw delta')
    revised = copy.deepcopy(history)
    revised['scale_matched_coverage'] = diag.revise_scale_comparisons(
        revised['scale_matched_coverage'])
    revised.update(statistical_revision=diag.STATISTICAL_REVISION,
                   comparison_tolerance=diag.COMPARISON_TOLERANCE,
                   comparison_rules=diag.COMPARISON_RULES.copy(),
                   revision_source=identity,
                   revision_mode='reclassify_bound_historical_raw_deltas_no_frame_replay')
    return revised


def build(online, revised, identities):
    if revised.get('statistical_revision') != diag.STATISTICAL_REVISION:
        raise ValueError('Unrevised diagnostic')
    for source in (online, revised):
        if source.get('fixed_test_previously_exposed') is not True:
            raise ValueError('Missing exposure declaration')
        if source.get('parameter_tuning_authorized') is not False:
            raise ValueError('Tuning must remain forbidden')
    differences = {}
    for group in ('all', 'real', 'sim'):
        current = online['supporting_diagnostics']['angle_hold_pair_audit'][group]
        frozen = revised['angle_hold_pair_audit'][group]
        differences[group] = dict(
            online_mean_error_delta_vs_raw=current['mean_error_delta_vs_raw'],
            frozen_mean_error_delta_vs_raw=frozen['mean_error_delta_vs_raw'],
            explanation='unresolved_without_upstream_per_frame_comparison')
    return dict(
        protocol=PROTOCOL, inputs=identities,
        status='RESULT_ARCHIVE_GENERATED_WITH_EXPLICIT_LIMITATIONS',
        fixed_test_previously_exposed=True, parameter_tuning_authorized=False,
        overall_winner_claimed=False,
        pipeline=online['pipeline'], dataset_summary=online['dataset_summary'],
        metric_definitions={
            'available_obb_coverage': '完整 OBB 可用帧 / 全部帧',
            'jointly_correct_coverage': 'center、scale、angle 同时通过各自阈值的帧 / 全部帧',
            'riou_hit_coverage': '可用且 RIoU 达到阈值的帧 / 全部帧',
            'partial_valid_count': '仅一项或两项分量有效的帧数；不是完整框',
            'false_all_components_valid_rate': '至少一项分量错误的完整观测帧 / 完整观测帧'},
        online_performance=dict(
            source='online', thresholds=online['evaluation_thresholds'],
            tables=online['paper_tables'], front_end=online['front_end_summary'],
            upstream_inputs=online['inputs'],
            supporting_diagnostics=online['supporting_diagnostics'],
            runtime_measurement=online['runtime_measurement']),
        frozen_diagnostics=dict(source='revised_diagnostic', report=revised),
        source_differences=differences,
        supported=['完整在线链路与分量状态接口', '准确性、完整覆盖率与部分分量可用性须并列报告'],
        unsupported=['V5.1 总体最优', '未知序列泛化', '学习风险全面优越',
                     '角度保持提升平均精度', '物理状态能力', '标准稳态部署实时性'],
        outstanding=['本归档中的冻结诊断从绑定历史差值重分类；逐帧回放结果作为独立证据保存',
                     '两来源角度差异尚未完成上游逐帧核对',
                     '独立未知序列验证未完成',
                     '服务器源码一致性与 Python 3.8 实机验证待执行'])


def markdown(report):
    lines = ['# Base V3 + V5.1 结果归档 V3.1 R2', '',
             '固定 TEST 已暴露；不调参。V5.1 不是整体最优方法。', '',
             '真实在线性能与冻结诊断分别列示。角度来源差异未解释。', '']
    def render(value, title, depth=2):
        lines.extend(['#' * min(depth, 6) + ' ' + title, ''])
        if isinstance(value, list) and value and all(isinstance(x, dict) for x in value):
            keys = list(dict.fromkeys(k for x in value for k in x))
            lines.append('| ' + ' | '.join(keys) + ' |')
            lines.append('| ' + ' | '.join('---' for _ in keys) + ' |')
            for row in value:
                lines.append('| ' + ' | '.join(json.dumps(row.get(k), ensure_ascii=False).replace('|', '\\|') for k in keys) + ' |')
            lines.append('')
        elif isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, (dict, list)):
                    render(v, k, depth+1)
                else:
                    lines.extend(['- **{}**: {}'.format(k, v), ''])
        else:
            lines.extend(['```json', json.dumps(value, ensure_ascii=False, indent=2), '```', ''])
    # Avoid printing thousands of frame keys; full detail remains in JSON.
    visible = copy.deepcopy(report)
    frozen = visible['frozen_diagnostics']['report']
    scale_rows = []
    for group, g in frozen['scale_matched_coverage'].items():
        for p in g['points']:
            delta = p['learned_minus_score']
            scale_rows.append(dict(
                group=group, target_coverage=p['target_coverage'],
                retained_count=p['anchor_score']['retained_count'],
                score_mean=p['anchor_score']['mean_error'],
                risk_mean=p['learned_scale_risk']['mean_error'],
                mean_delta=delta['mean_error'], failure_delta=delta['failure_rate'],
                strict_win=delta['wins_mean_and_failure'],
                joint_tie=delta['tie_mean_and_failure']))
    frozen['scale_matched_coverage'] = scale_rows
    for g in frozen['angle_hold_pair_audit'].values():
        g.pop('prediction_cases', None)
    render(visible, '结果与证据目录')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--online', required=True)
    parser.add_argument('--historical-diagnostic', required=True)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    contract = json.loads(Path(args.contract).read_text())
    if contract['protocol'] != CONTRACT_PROTOCOL:
        raise ValueError('Invalid archive contract')
    online, online_id = load_bound(args.online, contract['online'])
    historical, historical_id = load_bound(args.historical_diagnostic, contract['historical_diagnostic'])
    if historical['input']['sha256'] != contract['frozen_eval_sha256']:
        raise ValueError('Frozen evaluation identity mismatch')
    revised = revision_from_history(historical, historical_id)
    out = Path(args.out_dir)
    # This is a deterministic reclassification of the bound historical report.
    # It must never share a filename with a diagnostic rebuilt from frame records.
    revised_id = _write_exact(out / RECLASSIFIED_DIAGNOSTIC_FILENAME,
                              revised)
    identities = dict(online=online_id, historical_diagnostic=historical_id,
                      revised_diagnostic={k:v for k,v in revised_id.items() if k != 'path'},
                      contract_sha256=hashlib.sha256(Path(args.contract).read_bytes()).hexdigest())
    report = build(online, revised, identities)
    result_id = _write_exact(out / ARCHIVE_FILENAME, report)
    text = markdown(report)
    md = out / MARKDOWN_FILENAME
    if md.exists() and md.read_text() != text:
        raise ValueError('Refusing to overwrite different Markdown')
    md.write_text(text, encoding='utf-8')
    print(json.dumps(dict(revised=revised_id, archive=result_id), indent=2))


if __name__ == '__main__':
    main()
