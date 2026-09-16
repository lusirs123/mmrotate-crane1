#!/usr/bin/env python3
"""Run or report the focused-paper RGB-to-component-observation pipeline.

The JSON config is the single entrypoint. ``infer`` produces GT-free online
observations, ``evaluate`` attaches GT afterwards and creates runtime-bound
contracts, and ``report`` consumes bound outputs without model inference.
``full`` executes those stages in order.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from crane_project.tools.base_v3_obb_reliability_baseline import (
    _identity, _write_exact)


CONFIG_PROTOCOL = 'base_v3_obb_focused_paper_pipeline_config_v1'
MODEL_REPORT_PROTOCOL = 'base_v3_obb_focused_paper_model_pipeline_v1'
RELIABILITY_REPORT_PROTOCOL = 'base_v3_obb_focused_paper_reliability_v1'
ONLINE_PROTOCOL = 'base_v3_obb_true_online_pipeline_v1'
FINALIZATION_PROTOCOL = 'base_v3_obb_true_online_finalization_v2'
PAPER_METRICS_PROTOCOL = 'base_v3_obb_true_online_paper_metrics_v3'
DIAGNOSTIC_PROTOCOL = 'base_v3_obb_hybrid_fixed_test_diagnostic_v51_v31'
METHODS = ('raw', 'score_rejection_only', 'v51_hybrid')
COMPONENTS = ('center', 'scale', 'angle')
PRIMARY_GROUPS = ('all', 'real', 'sim')
ANCHOR_GROUPS = ('anchor:k1', 'anchor:dino_fallback')


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(payload):
    raw = (json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':')) + '\n').encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def _resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def _relative_identity(root, path):
    path = Path(path).resolve()
    try:
        display = os.fspath(path.relative_to(root.resolve()))
    except ValueError:
        display = os.fspath(path)
    return dict(path=display, sha256=_sha256(path), size_bytes=path.stat().st_size)


def _load_bound(root, spec, role, expect_json=True):
    path = _resolve(root, spec['path'])
    if not path.is_file():
        raise RuntimeError('Missing {}: {}'.format(role, path))
    actual = _sha256(path)
    if actual != spec['sha256']:
        raise RuntimeError(
            '{} SHA256 mismatch: expected {}, got {}'.format(
                role, spec['sha256'], actual))
    identity = _relative_identity(root, path)
    if not expect_json:
        return None, identity
    payload = json.loads(path.read_text(encoding='utf-8'))
    expected_protocol = spec.get('protocol')
    if expected_protocol and payload.get('protocol') != expected_protocol:
        raise RuntimeError('{} protocol mismatch'.format(role))
    return payload, identity


def validate_config(config):
    if config.get('protocol') != CONFIG_PROTOCOL:
        raise ValueError('Unexpected focused-paper pipeline config')
    boundary = config.get('evidence_boundary', {})
    required_boundary = {
        'fixed_test_previously_exposed': True,
        'parameter_tuning_authorized': False,
        'unknown_sequence_claim_authorized': False,
        'physical_state_claim_authorized': False}
    for key, expected in required_boundary.items():
        if boundary.get(key) is not expected:
            raise ValueError('Evidence boundary changed: ' + key)
    inventory = config.get('metric_inventory', {})
    if inventory.get('methods') != list(METHODS):
        raise ValueError('Method inventory changed')
    if inventory.get('components') != list(COMPONENTS):
        raise ValueError('Component inventory changed')
    if inventory.get('primary_groups') != list(PRIMARY_GROUPS):
        raise ValueError('Primary group inventory changed')
    if inventory.get('anchor_groups') != list(ANCHOR_GROUPS):
        raise ValueError('Anchor group inventory changed')
    required_sources = {
        'online_pipeline', 'online_finalization', 'paper_metrics',
        'reliability_diagnostic', 'observation_csv'}
    if set(config.get('report_sources', {})) != required_sources:
        raise ValueError('Report source inventory changed')
    output_config = config.get('outputs', {})
    output_names = [output_config.get(key) for key in (
        'model_report_json', 'model_report_markdown',
        'reliability_report_json', 'reliability_report_markdown')]
    if len(output_names) != 4 or len(set(output_names)) != 4:
        raise ValueError('Four distinct report outputs are required')
    visualization = config.get('visualization', {})
    if visualization.get('method') != 'v51_hybrid':
        raise ValueError('Visualization must use the frozen V5.1 method')
    if visualization.get('selection_rule') != (
            'first_frame_per_component_state_tuple_without_gt_or_error_ranking'):
        raise ValueError('Visualization selection rule changed')
    if int(visualization.get('max_frames', 0)) <= 0:
        raise ValueError('Visualization max_frames must be positive')
    full_outputs = config.get('full_run', {}).get('outputs', {})
    required_full_outputs = {
        'online_pipeline', 'inference_receipt', 'online_finalization',
        'observation_csv', 'paper_metrics', 'reliability_diagnostic',
        'runtime_finalization_contract', 'runtime_paper_metrics_contract',
        'runtime_diagnostic_input', 'runtime_diagnostic_contract'}
    if set(full_outputs) != required_full_outputs:
        raise ValueError('Full-run output inventory changed')
    custom = inventory.get('custom_temporal_metrics', [])
    required_custom = {
        'R_center(%)', 'mean_RIoU', 'A-RMSE(deg)', 'DFR(%/frame)',
        'ACI', 'TDR_w10(%)', 'MCML_max(frames)',
        'MCML_mean(frames)', 'MCML_pass(limit=5)', 'MRF(frames)'}
    if set(custom) != required_custom:
        raise ValueError('Custom temporal metric inventory changed')


def load_report_sources(config, root):
    sources = config['report_sources']
    online, online_id = _load_bound(
        root, sources['online_pipeline'], 'online_pipeline')
    finalization, finalization_id = _load_bound(
        root, sources['online_finalization'], 'online_finalization')
    metrics, metrics_id = _load_bound(
        root, sources['paper_metrics'], 'paper_metrics')
    diagnostic, diagnostic_id = _load_bound(
        root, sources['reliability_diagnostic'], 'reliability_diagnostic')
    _unused, csv_id = _load_bound(
        root, sources['observation_csv'], 'observation_csv', expect_json=False)
    return dict(
        online=online, finalization=finalization, metrics=metrics,
        diagnostic=diagnostic,
        identities=dict(
            online_pipeline=online_id, online_finalization=finalization_id,
            paper_metrics=metrics_id, reliability_diagnostic=diagnostic_id,
            observation_csv=csv_id))


def load_run_sources(config, root, directory):
    """Load one generated run by its own embedded identities."""
    directory = Path(directory).resolve()
    names = config['full_run']['outputs']
    roles = {
        'online': ('online_pipeline', ONLINE_PROTOCOL),
        'finalization': ('online_finalization', FINALIZATION_PROTOCOL),
        'metrics': ('paper_metrics', PAPER_METRICS_PROTOCOL),
        'diagnostic': ('reliability_diagnostic', DIAGNOSTIC_PROTOCOL)}
    payloads = {}
    identities = {}
    identity_names = {
        'online': 'online_pipeline',
        'finalization': 'online_finalization',
        'metrics': 'paper_metrics',
        'diagnostic': 'reliability_diagnostic'}
    for role, (name_key, protocol) in roles.items():
        path = directory / names[name_key]
        if not path.is_file():
            raise RuntimeError('Missing generated {}: {}'.format(role, path))
        payload = json.loads(path.read_text(encoding='utf-8'))
        if payload.get('protocol') != protocol:
            raise RuntimeError('Unexpected generated {} protocol'.format(role))
        payloads[role] = payload
        identities[identity_names[role]] = _relative_identity(root, path)
    csv_path = directory / names['observation_csv']
    if not csv_path.is_file():
        raise RuntimeError('Missing generated observation CSV: ' + os.fspath(csv_path))
    identities['observation_csv'] = _relative_identity(root, csv_path)
    optional = {
        'inference_receipt': 'inference_receipt',
        'runtime_finalization_contract': 'runtime_finalization_contract',
        'runtime_paper_metrics_contract': 'runtime_paper_metrics_contract',
        'runtime_diagnostic_input': 'runtime_diagnostic_input',
        'runtime_diagnostic_contract': 'runtime_diagnostic_contract'}
    for role, name_key in optional.items():
        path = directory / names[name_key]
        if path.is_file():
            identities[role] = _relative_identity(root, path)
    receipt_path = directory / names['inference_receipt']
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if receipt.get('protocol') != (
                'base_v3_obb_focused_paper_inference_receipt_v1'):
            raise RuntimeError('Unexpected inference receipt protocol')
        if receipt['online_pipeline']['sha256'] != identities[
                'online_pipeline']['sha256']:
            raise RuntimeError('Inference receipt online SHA256 mismatch')
        if receipt['observation_csv']['sha256'] != identities[
                'observation_csv']['sha256']:
            raise RuntimeError('Inference receipt CSV SHA256 mismatch')
    return dict(
        online=payloads['online'], finalization=payloads['finalization'],
        metrics=payloads['metrics'], diagnostic=payloads['diagnostic'],
        identities=identities)


def validate_report_sources(bundle, config):
    online = bundle['online']
    finalization = bundle['finalization']
    metrics = bundle['metrics']
    diagnostic = bundle['diagnostic']
    expected = (
        (online, ONLINE_PROTOCOL),
        (finalization, FINALIZATION_PROTOCOL),
        (metrics, PAPER_METRICS_PROTOCOL),
        (diagnostic, DIAGNOSTIC_PROTOCOL))
    for payload, protocol in expected:
        if payload.get('protocol') != protocol:
            raise RuntimeError('Unexpected input protocol: ' + protocol)
    model_pipeline = config['pipeline'][:5]
    if online.get('pipeline') != model_pipeline:
        raise RuntimeError('Online model pipeline order changed')
    if metrics.get('pipeline') != model_pipeline:
        raise RuntimeError('Metric pipeline order changed')
    frame_count = metrics['dataset_summary']['frame_count']
    if frame_count != finalization['dataset_summary']['frame_count']:
        raise RuntimeError('Finalization and metric frame counts differ')
    if frame_count != online['summary']['frame_count']:
        raise RuntimeError('Online and metric frame counts differ')
    tables = metrics.get('paper_tables', {})
    if len(tables.get('complete_obb_metrics', [])) != 9:
        raise RuntimeError('Complete OBB table must contain 9 rows')
    if len(tables.get('component_metrics', [])) != 27:
        raise RuntimeError('Component table must contain 27 rows')
    if len(tables.get('anchor_source_metrics', [])) != 6:
        raise RuntimeError('Anchor-source table must contain 6 rows')
    if float(diagnostic.get('comparison_tolerance', -1)) != 1e-12:
        raise RuntimeError('Reliability comparison tolerance changed')
    if diagnostic.get('comparison_rules', {}).get('raw_deltas_preserved') is not True:
        raise RuntimeError('Reliability diagnostic does not preserve raw deltas')
    csv_sha = bundle['identities']['observation_csv']['sha256']
    if finalization.get('online_observation_export', {}).get('sha256') != csv_sha:
        raise RuntimeError('Finalization does not bind the observation CSV')
    bound = (
        (finalization.get('inputs', {}).get('true_online_report'),
         bundle['identities']['online_pipeline'],
         'Finalization online input'),
        (metrics.get('inputs', {}).get('true_online_finalization_v2'),
         bundle['identities']['online_finalization'],
         'Paper metrics finalization input'),
        (metrics.get('inputs', {}).get('true_online_report'),
         bundle['identities']['online_pipeline'],
         'Paper metrics online input'))
    for recorded, actual, role in bound:
        if recorded is not None and recorded.get('sha256') != actual['sha256']:
            raise RuntimeError(role + ' SHA256 mismatch')
    diagnostic_input = diagnostic.get('input', {})
    recorded_metric = diagnostic_input.get('paper_metrics')
    if (recorded_metric is not None and recorded_metric.get('sha256') !=
            bundle['identities']['paper_metrics']['sha256']):
        raise RuntimeError('Diagnostic paper metrics SHA256 mismatch')
    optional_bindings = (
        (finalization.get('inputs', {}).get('contract'),
         'runtime_finalization_contract', 'Finalization runtime contract'),
        (metrics.get('inputs', {}).get('contract'),
         'runtime_paper_metrics_contract', 'Paper-metrics runtime contract'),
        (diagnostic.get('runtime_artifacts', {}).get('input'),
         'runtime_diagnostic_input', 'Diagnostic runtime input'),
        (diagnostic.get('runtime_artifacts', {}).get('contract'),
         'runtime_diagnostic_contract', 'Diagnostic runtime contract'))
    for recorded, identity_key, role in optional_bindings:
        actual = bundle['identities'].get(identity_key)
        if recorded is not None and actual is not None:
            if recorded.get('sha256') != actual['sha256']:
                raise RuntimeError(role + ' SHA256 mismatch')


def _table_index(rows):
    return {(row['method'], row['group']): row for row in rows}


def _source_angle_differences(metrics, diagnostic):
    online = metrics['supporting_diagnostics']['angle_hold_pair_audit']
    frozen = diagnostic['angle_hold_pair_audit']
    result = {}
    for group in PRIMARY_GROUPS:
        a = online[group]['mean_error_delta_vs_raw']
        b = frozen[group]['mean_error_delta_vs_raw']
        same_full_run = diagnostic.get('generation_mode') == (
            'from_current_full_run_online_records')
        result[group] = dict(
            online_mean_error_delta_vs_raw=a,
            frozen_mean_error_delta_vs_raw=b,
            online_minus_frozen=None if a is None or b is None else a-b,
            explanation=(
                'same_full_run_records_reproduced'
                if same_full_run else
                'unresolved_without_upstream_per_frame_comparison'))
    return result


def build_reliability_report(bundle, config):
    metrics = bundle['metrics']
    finalization = bundle['finalization']
    diagnostic = bundle['diagnostic']
    complete = _table_index(metrics['paper_tables']['complete_obb_metrics'])
    score = complete[('score_rejection_only', 'all')]
    hybrid = complete[('v51_hybrid', 'all')]
    joint = diagnostic['joint_component_availability']
    angle = diagnostic['angle_hold_pair_audit']['all']
    scale = diagnostic['scale_matched_coverage']
    same_run_diagnostic = diagnostic.get('generation_mode') == (
        'from_current_full_run_online_records')
    interpretation = [
        dict(
            topic='complete_obb_tradeoff',
            finding=(
                '相对仅置信度拒绝，V5.1 的完整 OBB 覆盖率变化为 '
                '{:+.6f}，联合正确覆盖率变化为 {:+.6f}；两项必须并列报告。').format(
                    hybrid['available_obb_coverage']-
                    score['available_obb_coverage'],
                    hybrid['jointly_correct_coverage']-
                    score['jointly_correct_coverage'])),
        dict(
            topic='partial_component_availability',
            finding=(
                'V5.1 输出了 {} 个部分分量有效帧；这些帧只能作为分量观测，'
                '不能计为完整 OBB。').format(
                    joint['v51_hybrid']['all']['partial_valid_count'])),
        dict(
            topic='angle_hold',
            finding=(
                '一帧角度保持产生 {} 个预测，其中 {} 个改善、{} 个退化，'
                '平均误差变化为 {:+.6f} 度。').format(
                    angle['prediction_count'], angle['improved_count'],
                    angle['degraded_count'],
                    angle['mean_error_delta_vs_raw'])),
        dict(
            topic='scale_ranking',
            finding=(
                '在相同覆盖率下，全体帧尺度比较有 {} 个双指标严格胜出点和 '
                '{} 个双指标平局点，比较容差为 {}。').format(
                    scale['all']['risk_wins_both_count'],
                    scale['all']['matched_coverage_tie_count'],
                    diagnostic['comparison_tolerance']))]
    return dict(
        protocol=RELIABILITY_REPORT_PROTOCOL,
        config_protocol=config['protocol'],
        evidence_boundary=copy.deepcopy(config['evidence_boundary']),
        inputs=copy.deepcopy(bundle['identities']),
        metric_definitions=dict(
            valid=(
                '冻结在线规则允许输出该分量；它不是基于 GT 的正确性标签。'),
            measurement_coverage='测量状态帧数除以全部帧数',
            output_coverage=(
                '测量状态与有限保持预测状态帧数之和除以全部帧数'),
            complete_obb_coverage=(
                '中心、尺度和方向同时有效的帧数除以全部帧数'),
            jointly_correct_coverage=(
                '三个有效分量同时满足冻结误差阈值的帧数除以全部帧数'),
            false_all_components_valid_rate=(
                '至少一个分量错误的完整观测帧数除以完整观测帧数'),
            partial_valid_count=(
                '仅一个或两个分量有效的帧数；这些帧不构成完整 OBB'),
            riou_hit_coverage=(
                '具有完整 OBB 且 RIoU 达到阈值的帧数除以全部帧数')),
        policy_semantics=dict(
            center='按锚框置信度拒绝',
            scale='按冻结 V5.1 监督尺度风险拒绝',
            angle='按锚框置信度拒绝，并允许最多一帧的历史值保持',
            states=['measurement', 'prediction', 'unavailable'],
            angle_prediction_semantics='有限的最近测量值保持，不是运动预测'),
        evaluation_thresholds=copy.deepcopy(metrics['evaluation_thresholds']),
        component_metrics=copy.deepcopy(
            metrics['paper_tables']['component_metrics']),
        joint_component_availability=copy.deepcopy(joint),
        matched_coverage_scale=copy.deepcopy(scale),
        angle_hold=copy.deepcopy(diagnostic['angle_hold_pair_audit']),
        rejection_recovery=copy.deepcopy(
            metrics['supporting_diagnostics']['rejection_recovery']),
        source_angle_differences=_source_angle_differences(metrics, diagnostic),
        interpretation=interpretation,
        limits=[
            'V5.1 尚未被证实为整体最优方法。',
            '固定 TEST 已经暴露，不能据此调参。',
            ('本次角度诊断与论文指标来自同一逐帧运行记录。'
             if same_run_diagnostic else
             '完成逐帧比较前，两种角度报告来源之间的差异仍未解释。'),
            '当前不支持新的未知序列泛化、校准不确定性、物理状态或控制结论。'],
        runtime_measurement=copy.deepcopy(finalization['runtime_measurement']))


def build_model_report(bundle, config, reliability_identity, custom_temporal):
    metrics = bundle['metrics']
    online = bundle['online']
    return dict(
        protocol=MODEL_REPORT_PROTOCOL,
        config_protocol=config['protocol'],
        status='COMPLETE_BOUND_MODEL_AND_REPORTING_FLOW',
        evidence_boundary=copy.deepcopy(config['evidence_boundary']),
        inputs=copy.deepcopy(bundle['identities']),
        pipeline=copy.deepcopy(config['pipeline']),
        model_artifacts=copy.deepcopy(online['artifacts']),
        online_contracts=dict(
            history=copy.deepcopy(online['history_contract']),
            observation=copy.deepcopy(online['observation_contract']),
            leakage=copy.deepcopy(online['leakage_controls'])),
        dataset_summary=copy.deepcopy(metrics['dataset_summary']),
        evaluation_thresholds=copy.deepcopy(metrics['evaluation_thresholds']),
        front_end_summary=copy.deepcopy(metrics['front_end_summary']),
        metrics=dict(
            complete_obb=copy.deepcopy(
                metrics['paper_tables']['complete_obb_metrics']),
            components=copy.deepcopy(
                metrics['paper_tables']['component_metrics']),
            anchor_sources=copy.deepcopy(
                metrics['paper_tables']['anchor_source_metrics']),
            custom_diagnostics=copy.deepcopy(
                metrics['supporting_diagnostics']),
            custom_temporal=copy.deepcopy(custom_temporal)),
        runtime_measurement=copy.deepcopy(metrics['runtime_measurement']),
        observation_export=copy.deepcopy(metrics['online_observation_export']),
        reliability_report=copy.deepcopy(reliability_identity),
        completeness_audit=copy.deepcopy(metrics['completeness_audit']),
        conclusions=dict(
            supported=[
                'The chronological RGB-to-component-observation implementation is complete.',
                'Complete OBB, component validity, RIoU, partial availability, and missing runs are reported separately.'],
            unsupported=[
                'V5.1 is the overall best method.',
                'A valid output is guaranteed correct.',
                'Fresh unknown-sequence generalization or physical-state capability.']))


def _format(value):
    if value is None:
        return ''
    if isinstance(value, float):
        return '{:.6g}'.format(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    return str(value)


def _markdown_table(rows, fields):
    lines = [
        '| ' + ' | '.join(fields) + ' |',
        '| ' + ' | '.join('---' for _ in fields) + ' |']
    for row in rows:
        lines.append('| ' + ' | '.join(
            _format(row.get(field)).replace('|', '\\|') for field in fields) + ' |')
    return lines


def model_markdown(report):
    lines = [
        '# Base V3 + V5.1 小论文模型流程总报告', '',
        '状态：完整的、按时间顺序执行的 RGB 图像到分量观测流程。', '',
        '固定 TEST 已暴露；本报告只汇总冻结结果，不授权调参。', '',
        '## 模型流程', '',
        ' → '.join(report['pipeline']), '',
        '## 数据与运行范围', '',
        '- 帧数：{}'.format(report['dataset_summary']['frame_count']),
        '- 域计数：{}'.format(_format(report['dataset_summary']['domain_counts'])),
        '- 序列计数：{}'.format(_format(report['dataset_summary']['sequence_counts'])),
        '- 时间戳：未提供，不合成。', '',
        '## 完整 OBB 指标', '']
    complete_fields = [
        'method', 'group', 'frame_count', 'available_obb_coverage',
        'mean_available_riou', 'riou_hit_coverage',
        'jointly_correct_coverage', 'false_all_components_valid_rate',
        'longest_incomplete_obb_run']
    lines.extend(_markdown_table(report['metrics']['complete_obb'], complete_fields))
    lines.extend(['', '## 分量指标', ''])
    component_fields = [
        'method', 'group', 'component', 'measurement_coverage',
        'output_coverage', 'mean_available_output_error',
        'bad_available_output_rate', 'correct_output_coverage',
        'longest_unavailable_run']
    lines.extend(_markdown_table(report['metrics']['components'], component_fields))
    lines.extend(['', '## 锚来源指标', ''])
    anchor_fields = [
        'method', 'group', 'frame_count', 'available_obb_coverage',
        'mean_available_riou', 'riou_hit_coverage',
        'jointly_correct_coverage', 'false_all_components_valid_rate']
    lines.extend(_markdown_table(report['metrics']['anchor_sources'], anchor_fields))
    custom = report['metrics']['custom_temporal']
    lines.extend(['', '## 自定义检测与时序指标', '',
                  '完整 OBB 口径；部分分量观测在本表中计为缺少完整框。', ''])
    values = custom['values']
    custom_rows = []
    fields = sorted(set(
        key for method_values in values.values() for key in method_values))
    for metric in fields:
        custom_rows.append(dict(
            metric=metric, raw=values['raw'].get(metric),
            score_rejection_only=values['score_rejection_only'].get(metric),
            v51_hybrid=values['v51_hybrid'].get(metric)))
    lines.extend(_markdown_table(custom_rows, [
        'metric', 'raw', 'score_rejection_only', 'v51_hybrid']))
    lines.extend(['', '### 自定义指标口径', ''])
    lines.extend('- ' + note for note in custom['notes'])
    runtime = report['runtime_measurement']
    lines.extend(['', '## 运行时间', ''])
    if runtime is None:
        lines.append('未提供可绑定的运行时间。')
    else:
        lines.extend([
            '- 测量范围：{}'.format(runtime['scope']),
            '- 总时间：{} s'.format(_format(runtime['wall_clock_seconds'])),
            '- 平均：{} s/frame，{} frame/s'.format(
                _format(runtime['average_seconds_per_frame']),
                _format(runtime['average_frames_per_second']))])
    lines.extend([
        '', '## 解释边界', '',
        '- `valid` 表示规则允许输出，不是 GT 正确性标签。',
        '- 部分分量有效不能计为完整 OBB。',
        '- V5.1 不是整体最优方法。',
        '- 可靠性细节见独立可靠性 JSON 与 Markdown 报告。', ''])
    return '\n'.join(lines)


def reliability_markdown(report):
    lines = [
        '# Base V3 + V5.1 可靠性专项报告', '',
        '该报告单独解释分量有效性、完整框可用性和正确性，三者不能互相替代。', '',
        '## 指标定义', '']
    for key, value in report['metric_definitions'].items():
        lines.append('- {}：{}'.format(key, value))
    lines.extend(['', '## 关键解释', ''])
    for item in report['interpretation']:
        lines.append('- {}：{}'.format(item['topic'], item['finding']))
    lines.extend(['', '## 分量指标', ''])
    component_fields = [
        'method', 'group', 'component', 'measurement_count',
        'prediction_count', 'unavailable_count', 'measurement_coverage',
        'output_coverage', 'mean_available_output_error',
        'bad_available_output_rate', 'correct_output_coverage',
        'longest_unavailable_run']
    lines.extend(_markdown_table(report['component_metrics'], component_fields))
    lines.extend(['', '## 联合有效性与部分分量', ''])
    joint_rows = []
    for method in METHODS:
        for group, values in report['joint_component_availability'][method].items():
            if group in PRIMARY_GROUPS + ANCHOR_GROUPS:
                row = dict(method=method, group=group)
                row.update(values)
                joint_rows.append(row)
    joint_fields = [
        'method', 'group', 'frame_count', 'all_components_valid_coverage',
        'jointly_correct_coverage', 'false_all_components_valid_rate',
        'partial_valid_count', 'longest_incomplete_obb_run']
    lines.extend(_markdown_table(joint_rows, joint_fields))
    lines.extend(['', '## 尺度同覆盖率比较', ''])
    scale_rows = []
    for group, values in report['matched_coverage_scale'].items():
        for point in values['points']:
            delta = point['learned_minus_score']
            scale_rows.append(dict(
                group=group, target_coverage=point['target_coverage'],
                retained_count=point['anchor_score']['retained_count'],
                score_mean_error=point['anchor_score']['mean_error'],
                risk_mean_error=point['learned_scale_risk']['mean_error'],
                mean_error_delta=delta['mean_error'],
                failure_rate_delta=delta['failure_rate'],
                strict_win=delta['wins_mean_and_failure'],
                joint_tie=delta['tie_mean_and_failure']))
    scale_fields = [
        'group', 'target_coverage', 'retained_count', 'score_mean_error',
        'risk_mean_error', 'mean_error_delta', 'failure_rate_delta',
        'strict_win', 'joint_tie']
    lines.extend(_markdown_table(scale_rows, scale_fields))
    lines.extend(['', '## 角度保持', ''])
    angle_rows = []
    for group, values in report['angle_hold'].items():
        angle_rows.append(dict(
            group=group, prediction_count=values['prediction_count'],
            improved_count=values['improved_count'],
            degraded_count=values['degraded_count'],
            tied_count=values['tied_count'],
            mean_error_delta_vs_raw=values['mean_error_delta_vs_raw']))
    lines.extend(_markdown_table(angle_rows, [
        'group', 'prediction_count', 'improved_count', 'degraded_count',
        'tied_count', 'mean_error_delta_vs_raw']))
    lines.extend(['', '## 证据限制', ''])
    lines.extend('- ' + value for value in report['limits'])
    lines.append('')
    return '\n'.join(lines)


def _write_text_exact(path, text):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = text.encode('utf-8')
    if path.exists() and path.read_bytes() != raw:
        raise RuntimeError('Refusing to overwrite different output: ' + os.fspath(path))
    if not path.exists():
        path.write_bytes(raw)
    return dict(path=os.fspath(path), sha256=hashlib.sha256(raw).hexdigest(),
                size_bytes=len(raw))


def _copy_exact(source, destination):
    source = Path(source).resolve(); destination = Path(destination).resolve()
    raw = source.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() != raw:
        raise RuntimeError('Refusing to overwrite different copied input: '
                           + os.fspath(destination))
    if not destination.exists():
        destination.write_bytes(raw)
    return _identity(destination)


def _find_image(image_dir, frame_key):
    for suffix in ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'):
        path = image_dir / (frame_key + suffix)
        if path.is_file():
            return path
    raise RuntimeError('Missing visualization image: ' + frame_key)


def _draw_obb(cv2, image, box, color, thickness):
    if box is None:
        return
    cx, cy, width, height, angle = [float(value) for value in box[:5]]
    points = cv2.boxPoints(((cx, cy), (width, height),
                           angle * 180.0 / math.pi))
    points = points.round().astype('int32')
    cv2.polylines(image, [points], True, color, thickness, cv2.LINE_AA)


def write_visualizations(bundle, config, root, out_dir=None):
    """Render deterministic state examples without GT/error-based selection."""
    import cv2

    settings = config['visualization']
    image_dir = _resolve(root, settings['image_dir'])
    directory = (_resolve(root, settings['directory']) if out_dir is None
                 else Path(out_dir))
    directory.mkdir(parents=True, exist_ok=True)
    method = settings['method']
    selected = []
    seen = set()
    for row in bundle['online']['records']:
        observations = row['observations'][method]
        state_tuple = tuple(observations[name]['state'] for name in COMPONENTS)
        if state_tuple in seen:
            continue
        seen.add(state_tuple)
        selected.append((row, observations, state_tuple))
        if len(selected) >= int(settings['max_frames']):
            break
    rendered = []
    for row, observations, state_tuple in selected:
        source = _find_image(image_dir, row['frame_key'])
        image = cv2.imread(os.fspath(source), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError('Cannot read visualization image: ' + os.fspath(source))
        detector = row['detector_components']
        _draw_obb(cv2, image, detector.get('base_v3_box'), (160, 160, 160), 2)
        values = [observations[name].get('value') for name in COMPONENTS]
        if all(observations[name]['valid'] for name in COMPONENTS):
            _draw_obb(cv2, image, values[0] + values[1] + [values[2]],
                      (0, 210, 0), 3)
        elif observations['center']['valid']:
            center = tuple(int(round(value)) for value in values[0])
            cv2.circle(image, center, 6, (0, 220, 255), -1, cv2.LINE_AA)
        labels = ['{}={}'.format(name[0].upper(), observations[name]['state'])
                  for name in COMPONENTS]
        cv2.rectangle(image, (8, 8), (min(image.shape[1]-8, 760), 52),
                      (0, 0, 0), -1)
        cv2.putText(image, ' | '.join(labels), (18, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2,
                    cv2.LINE_AA)
        output = directory / (row['frame_key'] + '_v51_state.png')
        success, encoded = cv2.imencode('.png', image)
        if not success:
            raise RuntimeError('Failed to encode visualization: ' + row['frame_key'])
        raw = encoded.tobytes()
        if output.exists() and output.read_bytes() != raw:
            raise RuntimeError('Refusing to overwrite different visualization: '
                               + os.fspath(output))
        if not output.exists():
            output.write_bytes(raw)
        rendered.append(dict(
            frame_key=row['frame_key'], state_tuple=list(state_tuple),
            validity_mask=[name for name in COMPONENTS
                           if observations[name]['valid']],
            source_image=_relative_identity(root, source),
            output=_relative_identity(root, output),
            selection_uses_gt_or_error=False))
    manifest = dict(
        protocol='base_v3_obb_focused_paper_visualization_manifest_v1',
        online_report=copy.deepcopy(bundle['identities']['online_pipeline']),
        method=method, selection_rule=settings['selection_rule'],
        legend=dict(gray='Base V3 detector OBB', green='complete V5.1 OBB',
                    yellow='valid center when the complete OBB is unavailable'),
        selected_frame_count=len(rendered), records=rendered,
        gt_or_error_used_for_selection=False)
    manifest_identity = _write_exact(
        directory / settings['manifest'], manifest)
    return dict(manifest=manifest_identity, images=[row['output'] for row in rendered])


def write_reports(bundle, config, root, out_dir=None, config_identity=None):
    validate_report_sources(bundle, config)
    if config_identity is not None:
        bundle = dict(bundle)
        bundle['identities'] = copy.deepcopy(bundle['identities'])
        bundle['identities']['unified_config'] = copy.deepcopy(config_identity)
    outputs = config['outputs']
    directory = (_resolve(root, outputs['directory']) if out_dir is None
                 else Path(out_dir))
    directory.mkdir(parents=True, exist_ok=True)
    reliability = build_reliability_report(bundle, config)
    reliability_json = _write_exact(
        directory / outputs['reliability_report_json'], reliability)
    reliability_md = _write_text_exact(
        directory / outputs['reliability_report_markdown'],
        reliability_markdown(reliability))
    reliability_identity = dict(
        json={key: value for key, value in reliability_json.items()
              if key != 'path'},
        markdown={key: value for key, value in reliability_md.items()
                  if key != 'path'})
    from crane_project.tools.analyze_unified_full_run_v1 import (
        build_custom_metrics)
    custom_temporal = build_custom_metrics(
        bundle['online'], bundle['finalization'], bundle['metrics'])
    model = build_model_report(
        bundle, config, reliability_identity, custom_temporal)
    model_json = _write_exact(
        directory / outputs['model_report_json'], model)
    model_md = _write_text_exact(
        directory / outputs['model_report_markdown'], model_markdown(model))
    return dict(
        model_json=model_json, model_markdown=model_md,
        reliability_json=reliability_json,
        reliability_markdown=reliability_md)


def _validate_paths(config, root, roles):
    full = config['full_run']
    missing = []
    for role in roles:
        path = _resolve(root, full[role])
        if not path.exists():
            missing.append('{}={}'.format(role, path))
    if missing:
        raise RuntimeError('Missing pipeline inputs:\n' + '\n'.join(missing))


def validate_inference_inputs(config, root):
    _validate_paths(config, root, (
        'image_dir', 'dino_config', 'dino_checkpoint', 'base_v3_config',
        'runtime_calibration_v51', 'online_contract', 'paper_final_report'))


def validate_evaluation_inputs(config, root):
    _validate_paths(config, root, (
        'paper_final_report', 'fixed_test_attribution', 'v51_eval_contract',
        'finalization_contract', 'paper_metrics_contract', 'base_v3_results',
        'k1_results', 'all_lane_audit', 'annotation_dir'))


def validate_full_inputs(config, root):
    validate_inference_inputs(config, root)
    validate_evaluation_inputs(config, root)


def _require_sha(identity, expected, role):
    if identity['sha256'] != expected:
        raise RuntimeError(
            '{} SHA256 mismatch: expected {}, got {}'.format(
                role, expected, identity['sha256']))


def _runtime_contract(template, updates, template_identity, stage):
    contract = copy.deepcopy(template)
    contract['expected_inputs'].update(updates)
    contract['runtime_binding'] = dict(
        stage=stage, template_contract=copy.deepcopy(template_identity),
        generated_by=CONFIG_PROTOCOL)
    return contract


def build_current_diagnostic(online, final_payload, metric_contract,
                             metric_identity, return_runtime=False):
    """Rebuild reliability diagnostics from this run's own frame records."""
    from crane_project.tools import base_v3_obb_hybrid_fixed_test_diagnostic_v51 as diagnostic
    from crane_project.tools import base_v3_obb_true_online_paper_metrics_v3 as paper_metrics

    diagnostic_records = paper_metrics.build_diagnostic_records(
        online['records'], final_payload['records'])
    diagnostic_input = dict(
        protocol=diagnostic.INPUT_PROTOCOL,
        fixed_test_read=True, fixed_test_previously_exposed=True,
        frozen_calibration=True, threshold_fitting_performed=False,
        policy_selection_performed=False,
        eligible_for_parameter_tuning_from_this_report=False,
        method_inventory=list(METHODS), records=diagnostic_records)
    diagnostic_contract = dict(
        protocol=diagnostic.CONTRACT_PROTOCOL,
        fixed_test_read=True, fixed_test_previously_exposed=True,
        parameter_tuning_authorized=False,
        expected_input=dict(
            protocol=diagnostic.INPUT_PROTOCOL,
            sha256=_payload_sha256(diagnostic_input)),
        matched_coverage_targets=metric_contract['matched_coverage_targets'],
        fixed_test_scope=copy.deepcopy(metric_contract['fixed_test_scope']),
        evaluation=copy.deepcopy(metric_contract['evaluation']),
        evidence_boundary=metric_contract['evidence_boundary'],
        claim_limit=metric_contract['claim_limit'])
    payload = diagnostic.run(
        diagnostic_input, diagnostic_contract,
        input_identity=dict(
            source='current_full_run_online_and_post_inference_metrics',
            paper_metrics=metric_identity))
    payload['generation_mode'] = 'from_current_full_run_online_records'
    if return_runtime:
        return payload, diagnostic_input, diagnostic_contract
    return payload


def run_inference(config, root, out_dir, device):
    """Run the chronological model stage without reading annotations or GT."""
    validate_inference_inputs(config, root)
    from crane_project.tools import base_v3_obb_true_online_finalization_v2 as finalization
    full = config['full_run']
    run_outputs = full['outputs']
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    online_path = out_dir / run_outputs['online_pipeline']
    command = [
        sys.executable, '-m',
        'crane_project.tools.base_v3_obb_true_online_pipeline_v1',
        '--image-dir', os.fspath(_resolve(root, full['image_dir'])),
        '--dino-config', os.fspath(_resolve(root, full['dino_config'])),
        '--dino-checkpoint', os.fspath(_resolve(root, full['dino_checkpoint'])),
        '--base-v3-config', os.fspath(_resolve(root, full['base_v3_config'])),
        '--runtime-calibration-v51', os.fspath(_resolve(
            root, full['runtime_calibration_v51'])),
        '--contract', os.fspath(_resolve(root, full['online_contract'])),
        '--out-json', os.fspath(online_path),
        '--expected-frame-count', str(full['expected_frame_count']),
        '--paper-final-report', os.fspath(_resolve(
            root, full['paper_final_report'])), '--device', device]
    started = time.perf_counter()
    subprocess.run(command, cwd=os.fspath(root), check=True)
    wall_clock = time.perf_counter() - started
    online = json.loads(online_path.read_text(encoding='utf-8'))
    if online.get('leakage_controls', {}).get('annotations_or_gt_read') is not False:
        raise RuntimeError('Inference output does not prove GT isolation')
    if online.get('summary', {}).get('frame_count') != int(
            full['expected_frame_count']):
        raise RuntimeError('Inference output frame count changed')
    csv_path = out_dir / run_outputs['observation_csv']
    csv_identity = finalization.write_observation_csv(csv_path, online['records'])
    online_identity = _relative_identity(root, online_path)
    receipt = dict(
        protocol='base_v3_obb_focused_paper_inference_receipt_v1',
        stage='infer', gt_or_annotations_read=False,
        online_pipeline=online_identity,
        observation_csv=csv_identity,
        runtime_measurement=dict(
            scope='model_init_detector_image_io_and_online_json_generation',
            wall_clock_seconds=wall_clock,
            average_frames_per_second=len(online['records'])/wall_clock,
            average_seconds_per_frame=wall_clock/len(online['records']),
            multi_gpu_parallelism=False),
        command=command)
    receipt_identity = _write_exact(
        out_dir / run_outputs['inference_receipt'], receipt)
    return dict(
        online=online, online_identity=online_identity,
        observation_csv_identity=csv_identity,
        inference_receipt=receipt,
        inference_receipt_identity=receipt_identity)


def _load_inference(config, root, input_dir):
    full = config['full_run']; names = full['outputs']
    input_dir = Path(input_dir).resolve()
    online_path = input_dir / names['online_pipeline']
    if not online_path.is_file():
        raise RuntimeError('Missing inference output: ' + os.fspath(online_path))
    online = json.loads(online_path.read_text(encoding='utf-8'))
    if online.get('protocol') != ONLINE_PROTOCOL:
        raise RuntimeError('Unexpected inference output protocol')
    receipt_path = input_dir / names['inference_receipt']
    receipt = None
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if receipt.get('protocol') != (
                'base_v3_obb_focused_paper_inference_receipt_v1'):
            raise RuntimeError('Unexpected inference receipt protocol')
        if receipt['online_pipeline']['sha256'] != _sha256(online_path):
            raise RuntimeError('Inference receipt online SHA256 mismatch')
    return dict(
        online=online, online_path=online_path, receipt=receipt,
        input_observation_csv=input_dir / names['observation_csv'])


def run_evaluation(config, root, input_dir, out_dir=None):
    """Attach GT after inference and generate a fully bound result bundle."""
    validate_evaluation_inputs(config, root)
    from crane_project.tools import base_v3_obb_true_online_finalization_v2 as finalization
    from crane_project.tools import base_v3_obb_true_online_paper_metrics_v3 as paper_metrics
    from crane_project.tools.base_v3_obb_fixed_test_eval import build_records
    from crane_project.tools.base_v3_obb_hybrid_fixed_test_eval_v51 import (
        validate_contract as validate_v51_contract)

    full = config['full_run']; names = full['outputs']
    inferred = _load_inference(config, root, input_dir)
    online = inferred['online']; source_online_path = inferred['online_path']
    out_dir = Path(input_dir if out_dir is None else out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    online_path = out_dir / names['online_pipeline']
    online_identity = _copy_exact(source_online_path, online_path)
    receipt_identity = None
    source_receipt = Path(input_dir).resolve() / names['inference_receipt']
    if source_receipt.is_file():
        receipt_identity = _copy_exact(
            source_receipt, out_dir / names['inference_receipt'])

    final_contract_path = _resolve(root, full['finalization_contract'])
    final_template = json.loads(final_contract_path.read_text(encoding='utf-8'))
    final_template_id = _identity(final_contract_path)
    final_contract = _runtime_contract(
        final_template,
        {'true_online_sha256': online_identity['sha256']},
        final_template_id, 'evaluate')
    finalization.validate_contract(final_contract)
    paper_path = _resolve(root, full['paper_final_report'])
    attribution_path = _resolve(root, full['fixed_test_attribution'])
    v51_path = _resolve(root, full['v51_eval_contract'])
    expected = final_contract['expected_inputs']
    for identity, field, role in (
            (_identity(paper_path), 'paper_report_sha256', 'paper report'),
            (_identity(attribution_path), 'attribution_sha256', 'attribution'),
            (_identity(v51_path), 'v51_eval_contract_sha256', 'V5.1 contract')):
        _require_sha(identity, expected[field], role)
    paper = json.loads(paper_path.read_text(encoding='utf-8'))
    attribution = json.loads(attribution_path.read_text(encoding='utf-8'))
    v51 = json.loads(v51_path.read_text(encoding='utf-8'))
    validate_v51_contract(v51)
    finalization.validate_evaluation_alignment(final_contract, v51)
    final_contract_id = _write_exact(
        out_dir / names['runtime_finalization_contract'], final_contract)
    reconstruction_args = SimpleNamespace(
        base_v3_results=os.fspath(_resolve(root, full['base_v3_results'])),
        k1_results=os.fspath(_resolve(root, full['k1_results'])),
        all_lane_audit=os.fspath(_resolve(root, full['all_lane_audit'])),
        ann_dir=os.fspath(_resolve(root, full['annotation_dir'])))
    historical, reconstruction = build_records(
        attribution, reconstruction_args, v51)
    final_inputs = dict(
        true_online_report=_identity(online_path),
        paper_final_report=_identity(paper_path),
        fixed_test_attribution=_identity(attribution_path),
        v51_eval_contract=_identity(v51_path),
        contract=final_contract_id,
        template_contract=final_template_id,
        historical_reconstruction=reconstruction)
    runtime = None
    runtime_scope = 'inference_runtime_not_available'
    if inferred['receipt'] is not None:
        runtime_measurement = inferred['receipt']['runtime_measurement']
        runtime = runtime_measurement['wall_clock_seconds']
        runtime_scope = runtime_measurement['scope']
    final_payload = finalization.run(
        online, paper, historical, final_contract, final_inputs, runtime,
        runtime_scope=runtime_scope)
    csv_path = out_dir / names['observation_csv']
    csv_identity = finalization.write_observation_csv(csv_path, online['records'])
    input_csv = inferred['input_observation_csv']
    if input_csv.is_file() and _sha256(input_csv) != csv_identity['sha256']:
        raise RuntimeError('Evaluation observation CSV differs from inference export')
    final_payload['online_observation_export'] = csv_identity
    final_path = out_dir / names['online_finalization']
    final_identity = _write_exact(final_path, final_payload)

    metric_contract_path = _resolve(root, full['paper_metrics_contract'])
    metric_template = json.loads(metric_contract_path.read_text(encoding='utf-8'))
    metric_template_id = _identity(metric_contract_path)
    metric_contract = _runtime_contract(
        metric_template, dict(
            true_online_finalization_sha256=final_identity['sha256'],
            true_online_pipeline_sha256=online_identity['sha256'],
            online_observation_csv_sha256=csv_identity['sha256']),
        metric_template_id, 'evaluate')
    paper_metrics.validate_contract(metric_contract)
    metric_contract_id = _write_exact(
        out_dir / names['runtime_paper_metrics_contract'], metric_contract)
    metric_inputs = dict(
        true_online_finalization_v2=final_identity,
        true_online_report=_identity(online_path),
        contract=metric_contract_id, template_contract=metric_template_id,
        unified_config_runtime_binding=True)
    metric_payload = paper_metrics.run(
        final_payload, online, metric_contract, metric_inputs)
    metric_path = out_dir / names['paper_metrics']
    metric_identity = _write_exact(metric_path, metric_payload)
    _diagnostic_preview, diagnostic_input, diagnostic_contract = (
        build_current_diagnostic(
            online, final_payload, metric_contract, metric_identity,
            return_runtime=True))
    diagnostic_input_id = _write_exact(
        out_dir / names['runtime_diagnostic_input'], diagnostic_input)
    diagnostic_contract['expected_input']['sha256'] = (
        diagnostic_input_id['sha256'])
    diagnostic_contract['runtime_binding'] = dict(
        stage='evaluate', input=diagnostic_input_id,
        paper_metrics=metric_identity, generated_by=CONFIG_PROTOCOL)
    diagnostic_contract_id = _write_exact(
        out_dir / names['runtime_diagnostic_contract'], diagnostic_contract)
    from crane_project.tools import base_v3_obb_hybrid_fixed_test_diagnostic_v51 as diagnostic
    diagnostic_payload = diagnostic.run(
        diagnostic_input, diagnostic_contract, diagnostic_input_id)
    diagnostic_payload['generation_mode'] = (
        'from_current_full_run_online_records')
    diagnostic_payload['runtime_artifacts'] = dict(
        input=diagnostic_input_id, contract=diagnostic_contract_id)
    diagnostic_path = out_dir / names['reliability_diagnostic']
    diagnostic_identity = _write_exact(diagnostic_path, diagnostic_payload)
    identities = dict(
        online_pipeline=_relative_identity(root, online_path),
        online_finalization=_relative_identity(root, final_path),
        paper_metrics=_relative_identity(root, metric_path),
        reliability_diagnostic=_relative_identity(root, diagnostic_path),
        observation_csv=_relative_identity(root, csv_path),
        runtime_finalization_contract=final_contract_id,
        runtime_paper_metrics_contract=metric_contract_id,
        runtime_diagnostic_input=diagnostic_input_id,
        runtime_diagnostic_contract=diagnostic_contract_id)
    if receipt_identity is not None:
        identities['inference_receipt'] = receipt_identity
    bundle = dict(
        online=online, finalization=final_payload, metrics=metric_payload,
        diagnostic=diagnostic_payload,
        identities=identities)
    return bundle


def run_full(config, root, out_dir, device):
    """Run inference, post-inference evaluation, and bound reporting inputs."""
    validate_full_inputs(config, root)
    run_inference(config, root, out_dir, device)
    return run_evaluation(config, root, out_dir, out_dir)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument(
        '--mode', choices=(
            'validate', 'infer', 'evaluate', 'report', 'visualize', 'full'),
        default='report')
    parser.add_argument(
        '--input-dir', help=(
            'Generated run directory consumed by evaluate/report/validate'))
    parser.add_argument('--out-dir')
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    root = Path.cwd().resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    validate_config(config)
    if args.mode in ('infer', 'full'):
        if args.out_dir is None:
            raise ValueError(
                '--out-dir is required for {} mode so a fresh evidence '
                'directory is used'.format(args.mode))
    if args.mode == 'infer':
        inferred = run_inference(config, root, args.out_dir, args.device)
        result = dict(
            online_pipeline=inferred['online_identity'],
            observation_csv=inferred['observation_csv_identity'],
            inference_receipt=inferred['inference_receipt_identity'])
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.mode == 'evaluate':
        if args.input_dir is None:
            raise ValueError('--input-dir is required for evaluate mode')
        output_dir = args.input_dir if args.out_dir is None else args.out_dir
        bundle = run_evaluation(config, root, args.input_dir, output_dir)
    elif args.mode == 'full':
        bundle = run_full(config, root, args.out_dir, args.device)
    else:
        bundle = (load_run_sources(config, root, args.input_dir)
                  if args.input_dir is not None else
                  load_report_sources(config, root))
    validate_report_sources(bundle, config)
    if args.mode == 'validate':
        result = dict(
            status='VALID', config=_relative_identity(root, config_path),
            inputs=bundle['identities'])
    elif args.mode == 'visualize':
        result = write_visualizations(bundle, config, root, args.out_dir)
    else:
        result = write_reports(
            bundle, config, root, args.out_dir,
            config_identity=_relative_identity(root, config_path))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
