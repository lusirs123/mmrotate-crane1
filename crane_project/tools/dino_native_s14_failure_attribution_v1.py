#!/usr/bin/env python3
"""Build a read-only failure-attribution package for frozen native-S14 DINO.

This tool does not run inference or select a model.  It joins the retained
source-only classifier-selection result with the target-diagnosis-only RPN
coverage and RPN-to-ROI attrition audits.  The output is intended to decide
which model stage deserves the next experiment while preserving the exposed
target slices as diagnostic evidence only.
"""

import argparse
import csv
import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path


PROTOCOL = 'frozen_dino_native_s14_failure_attribution_v1'
EXPECTED_SELECTOR = 'Frozen DINO ROI Classifier Source Interpolation Selector V1'
EXPECTED_ATTRITION_AUDIT = 'DINO RPN-to-ROI Attrition and Latency Audit V1'
EXPECTED_COVERAGE_AUDIT = 'DINO Token-Scale and RPN Coverage Audit V1'
EXPECTED_ALPHA = 0.5
ALLOWED_CAUSES = {
    'ROI_TOP1_RESTORED',
    'RPN_GEOMETRY_MISSING',
    'ROI_REGRESSION_DESTROYS_RPN_GEOMETRY',
    'ROI_ORDERING_OR_NMS_REMOVES_GEOMETRY',
}


def parse_args():
    parser = argparse.ArgumentParser(description=PROTOCOL)
    parser.add_argument('--attrition', required=True)
    parser.add_argument('--coverage', required=True)
    parser.add_argument('--interpolation', required=True)
    parser.add_argument('--out-dir', required=True)
    return parser.parse_args()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _rows_by_key(group):
    rows = {}
    for row in group['rows']:
        key = (str(row['seq']), int(row['frame']))
        _require(key not in rows, 'Duplicate frame row: {}'.format(key))
        rows[key] = row
    return rows


def validate_inputs(attrition, coverage, interpolation):
    _require(attrition.get('audit') == EXPECTED_ATTRITION_AUDIT,
             'Unexpected attrition audit protocol')
    _require(int(attrition.get('protocol_version', -1)) == 2,
             'Attrition protocol version 2 is required')
    _require(coverage.get('audit') == EXPECTED_COVERAGE_AUDIT,
             'Unexpected coverage audit protocol')
    _require(int(coverage.get('protocol_version', -1)) == 1,
             'Coverage protocol version 1 is required')
    _require(interpolation.get('selector') == EXPECTED_SELECTOR,
             'Unexpected interpolation selector')
    _require(int(interpolation.get('protocol_version', -1)) == 1,
             'Interpolation selector protocol version 1 is required')
    _require(float(interpolation.get('selected_alpha', -1.0)) == EXPECTED_ALPHA,
             'The frozen native-S14 baseline requires alpha=0.5')

    checkpoint_hashes = {
        attrition.get('labeller_checkpoint_sha256'),
        coverage.get('labeller_checkpoint_sha256'),
        interpolation.get('selected_checkpoint_sha256'),
    }
    _require(None not in checkpoint_hashes and len(checkpoint_hashes) == 1,
             'DINO head checkpoint identities do not match')
    dino_hashes = {
        attrition.get('dinov2_checkpoint_sha256'),
        coverage.get('dinov2_checkpoint_sha256'),
        interpolation.get('dinov2_checkpoint_sha256'),
    }
    _require(None not in dino_hashes and len(dino_hashes) == 1,
             'DINOv2 checkpoint identities do not match')
    _require(float(attrition['protocol']['riou_thr']) ==
             float(coverage['protocol']['riou_thr']),
             'RIoU thresholds do not match')
    _require(not bool(attrition['protocol']['target_used_for_training']) and
             not bool(attrition['protocol']['target_used_for_checkpoint_selection']),
             'Attrition target slices must be diagnosis-only')
    _require(not bool(coverage['protocol']['target_used_for_training']) and
             not bool(coverage['protocol']['target_used_for_checkpoint_selection']),
             'Coverage target slices must be diagnosis-only')
    _require(not bool(interpolation['protocol']['target_data_read']) and
             not bool(interpolation['protocol']['target_used_for_selection']),
             'Interpolation selection must be source-only')
    selected = _selected_interpolation(interpolation)
    _require(bool(selected['source_gate']['passed']),
             'Selected interpolation did not pass its source gate')
    _require(int(selected['retention']['newly_incorrect_count']) == 0,
             'Selected interpolation did not retain old correct frames')

    attrition_groups = attrition.get('target_diagnoses', {})
    coverage_groups = coverage.get('target_diagnoses', {})
    _require(attrition_groups and set(attrition_groups) == set(coverage_groups),
             'Target diagnosis groups do not align')
    for name in sorted(attrition_groups):
        _require(attrition_groups[name]['specification'] ==
                 coverage_groups[name]['specification'],
                 'Group specification mismatch: {}'.format(name))
        attrition_rows = _rows_by_key(attrition_groups[name])
        coverage_rows = _rows_by_key(coverage_groups[name])
        _require(set(attrition_rows) == set(coverage_rows),
                 'Frame rows do not align: {}'.format(name))
        for key, row in attrition_rows.items():
            _require(len(row.get('objects', [])) == 1,
                     'Exactly one GT object is required: {}'.format(key))
            cause = row['objects'][0].get('attrition_cause')
            _require(cause in ALLOWED_CAUSES,
                     'Unknown attrition cause: {}'.format(cause))
            _require(len(coverage_rows[key].get('objects', [])) == 1,
                     'Exactly one coverage object is required: {}'.format(key))


def _selected_interpolation(interpolation):
    selected_alpha = float(interpolation['selected_alpha'])
    candidates = interpolation['source']['candidates']
    matches = [row for row in candidates
               if float(row['alpha']) == selected_alpha]
    _require(len(matches) == 1, 'Selected interpolation row is missing')
    return matches[0]


def _attribution_label(cause):
    mapping = {
        'ROI_TOP1_RESTORED': 'SUCCESS_TOP1',
        'RPN_GEOMETRY_MISSING': 'CANDIDATE_GENERATION',
        'ROI_REGRESSION_DESTROYS_RPN_GEOMETRY': 'ROI_REGRESSION',
        'ROI_ORDERING_OR_NMS_REMOVES_GEOMETRY': 'ROI_ORDERING_OR_NMS',
    }
    return mapping[cause]


def _flatten_rows(attrition, coverage):
    flat = []
    for group_name in sorted(attrition['target_diagnoses']):
        attr_group = attrition['target_diagnoses'][group_name]
        cov_by_key = _rows_by_key(coverage['target_diagnoses'][group_name])
        for attr_row in attr_group['rows']:
            key = (str(attr_row['seq']), int(attr_row['frame']))
            cov_row = cov_by_key[key]
            obj = attr_row['objects'][0]
            cov_obj = cov_row['objects'][0]
            cause = str(obj['attrition_cause'])
            post_valid = obj['post_valid_content']
            decoded = obj['roi_regression']
            flat.append(dict(
                group=group_name,
                split=str(attr_row['split']),
                sequence=key[0],
                frame=key[1],
                exposed_target_diagnosis_only=True,
                top1_hit=bool(post_valid['top1_hit']),
                attribution=_attribution_label(cause),
                original_attrition_cause=cause,
                review_required=not bool(post_valid['top1_hit']),
                review_priority=(0 if not bool(post_valid['top1_hit']) else 1),
                short_token=float(obj['token_scale']['short_token']),
                long_token=float(obj['token_scale']['long_token']),
                source_token_bin=str(obj['source_token_bin']),
                anchor_max_hbb_iou=float(cov_obj['anchor']['max_hbb_iou']),
                exact_assigned_positive_count=int(
                    cov_obj['anchor']['exact_assigned_positive_count']),
                rpn_best_riou=float(obj['rpn']['best_riou']),
                rpn_best_usable_rank=obj['rpn']['best_usable_rank'],
                decoded_best_riou=float(decoded['best_decoded_riou']),
                decoded_usable_count=int(decoded['decoded_usable_count']),
                decoded_usable_foreground_rank=(
                    decoded['decoded_usable_foreground_rank']),
                post_nms_best_usable_rank=obj['post_nms']['best_usable_rank'],
                post_valid_best_usable_rank=post_valid['best_usable_rank'],
                post_valid_best_riou=post_valid['best_riou'],
                candidate_counts=dict(attr_row['counts']),
            ))
    return sorted(flat, key=lambda row: (
        row['review_priority'], row['group'], row['sequence'], row['frame']))


def _summaries(rows):
    result = {}
    for group in sorted({row['group'] for row in rows}):
        selected = [row for row in rows if row['group'] == group]
        counts = Counter(row['attribution'] for row in selected)
        result[group] = dict(
            frame_count=len(selected),
            top1_hit_count=sum(row['top1_hit'] for row in selected),
            top1_hit_rate=sum(row['top1_hit'] for row in selected) / len(selected),
            failure_count=sum(not row['top1_hit'] for row in selected),
            attribution_counts=dict(sorted(counts.items())),
        )
    all_counts = Counter(row['attribution'] for row in rows)
    result['all'] = dict(
        frame_count=len(rows),
        top1_hit_count=sum(row['top1_hit'] for row in rows),
        top1_hit_rate=sum(row['top1_hit'] for row in rows) / len(rows),
        failure_count=sum(not row['top1_hit'] for row in rows),
        attribution_counts=dict(sorted(all_counts.items())),
    )
    return result


def build_report(attrition, coverage, interpolation, input_paths):
    validate_inputs(attrition, coverage, interpolation)
    rows = _flatten_rows(attrition, coverage)
    selected = _selected_interpolation(interpolation)
    baseline = interpolation['source']['baseline']
    return dict(
        protocol=PROTOCOL,
        protocol_version=1,
        evidence_boundary=dict(
            operation='READ_ONLY_JOIN_OF_EXISTING_REPORTS',
            runs_inference=False,
            trains_or_updates_parameters=False,
            selects_checkpoint=False,
            target_slices_are_exposed=True,
            target_results_authorize_tuning=False,
            ordering_and_nms_are_not_separable_from_retained_audit=True,
            raw_candidate_boxes_or_image_paths_available=False,
        ),
        comparison=dict(
            riou_threshold=float(attrition['protocol']['riou_thr']),
            top1_definition='post_valid_content top-ranked box has RIoU >= threshold',
        ),
        inputs={name: dict(path=str(path), sha256=_sha256(path))
                for name, path in input_paths.items()},
        model_identity=dict(
            frozen_dinov2_checkpoint_sha256=(
                attrition['dinov2_checkpoint_sha256']),
            dino_head_checkpoint_sha256=(
                attrition['labeller_checkpoint_sha256']),
            classifier_interpolation_alpha=float(
                interpolation['selected_alpha']),
            s7_enabled=False,
        ),
        source_selection_evidence=dict(
            baseline_full=baseline['source_full_summary'],
            baseline_small=baseline['source_small_summary'],
            selected_full=selected['source_full_summary'],
            selected_small=selected['source_small_summary'],
            selected_retention=selected['retention'],
            selected_source_gate=selected['source_gate'],
        ),
        target_diagnosis_summary=_summaries(rows),
        recommended_next_evidence=dict(
            primary_stage='ROI_ORDERING_OR_NMS',
            required_to_separate_ordering_from_nms=(
                'export pre-NMS and post-NMS candidate identities, scores, boxes, and GT RIoU'),
            required_for_visual_review=(
                'resolve sequence/frame identifiers to the immutable image manifest'),
            selection_rule=(
                'use new independent source/validation evidence; do not select with these target slices'),
        ),
        records=rows,
    )


def _json_bytes(report):
    return (json.dumps(report, ensure_ascii=False, indent=2,
                       sort_keys=True) + '\n').encode('utf-8')


def _csv_bytes(rows):
    fields = [
        'group', 'split', 'sequence', 'frame', 'top1_hit', 'attribution',
        'original_attrition_cause', 'review_required', 'short_token',
        'long_token', 'source_token_bin', 'anchor_max_hbb_iou',
        'exact_assigned_positive_count', 'rpn_best_riou',
        'rpn_best_usable_rank', 'decoded_best_riou',
        'decoded_usable_count', 'decoded_usable_foreground_rank',
        'post_nms_best_usable_rank', 'post_valid_best_usable_rank',
        'post_valid_best_riou',
    ]
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return stream.getvalue().encode('utf-8')


def _markdown_bytes(report):
    source = report['source_selection_evidence']
    lines = [
        '# Frozen DINO Native-S14 失败归因 V1', '',
        '本报告只读合并既有审计，不运行推理、不训练、不选择 checkpoint。',
        '目标困难切片已经暴露，只能用于失败诊断，不能用于调参或模型选择。', '',
        '## 模型身份', '',
        '- DINOv2 SHA256：`{}`'.format(
            report['model_identity']['frozen_dinov2_checkpoint_sha256']),
        '- DINO head SHA256：`{}`'.format(
            report['model_identity']['dino_head_checkpoint_sha256']),
        '- ROI 分类器插值系数：`{}`'.format(
            report['model_identity']['classifier_interpolation_alpha']), '',
        '## Source-only 选择证据', '',
        '| 范围 | 原始 top-1 | 选定 top-1 | 原始 MCML | 选定 MCML |',
        '|---|---:|---:|---:|---:|',
        '| full | {} / {} | {} / {} | {} | {} |'.format(
            source['baseline_full']['top1_hits'],
            source['baseline_full']['frame_count'],
            source['selected_full']['top1_hits'],
            source['selected_full']['frame_count'],
            source['baseline_full']['top1_mcml'],
            source['selected_full']['top1_mcml']),
        '| small | {} / {} | {} / {} | {} | {} |'.format(
            source['baseline_small']['top1_hits'],
            source['baseline_small']['frame_count'],
            source['selected_small']['top1_hits'],
            source['selected_small']['frame_count'],
            source['baseline_small']['top1_mcml'],
            source['selected_small']['top1_mcml']), '',
        '旧正确帧保留：`{}/{}`；新增正确：`{}`；新增错误：`{}`。'.format(
            source['selected_retention']['retained_correct_count'],
            source['selected_retention']['baseline_correct_count'],
            source['selected_retention']['newly_correct_count'],
            source['selected_retention']['newly_incorrect_count']), '',
        '## 暴露目标切片诊断', '',
        '| 分组 | 帧数 | top-1 命中 | 候选生成 | ROI 回归 | 排序或 NMS |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for group, summary in report['target_diagnosis_summary'].items():
        if group == 'all':
            continue
        counts = summary['attribution_counts']
        lines.append('| {} | {} | {} | {} | {} | {} |'.format(
            group, summary['frame_count'], summary['top1_hit_count'],
            counts.get('CANDIDATE_GENERATION', 0),
            counts.get('ROI_REGRESSION', 0),
            counts.get('ROI_ORDERING_OR_NMS', 0)))
    all_summary = report['target_diagnosis_summary']['all']
    all_counts = all_summary['attribution_counts']
    lines.append('| **合计** | **{}** | **{}** | **{}** | **{}** | **{}** |'.format(
        all_summary['frame_count'], all_summary['top1_hit_count'],
        all_counts.get('CANDIDATE_GENERATION', 0),
        all_counts.get('ROI_REGRESSION', 0),
        all_counts.get('ROI_ORDERING_OR_NMS', 0)))
    lines.extend([
        '', '## 结论与下一份证据', '',
        '当前保留审计把主要失败定位为 `ROI_ORDERING_OR_NMS`。它不能继续拆分排序与 NMS，',
        '因此下一次导出必须保存 pre-NMS 与 post-NMS 候选身份、分数、OBB 及其 GT RIoU。',
        'CSV 中 `review_required=true` 的记录是人工复查入口；现有审计没有原图路径或候选框坐标，',
        '需要通过不可变数据 manifest 将 `sequence + frame` 解析到图像。', '',
        '任何新方法必须使用新的独立 source/validation 证据选择；本报告中的目标切片只用于解释失败。',
        '', '## 输入绑定', '',
    ])
    for name, identity in sorted(report['inputs'].items()):
        lines.append('- {}：`{}`（SHA256 `{}`）'.format(
            name, identity['path'], identity['sha256']))
    return ('\n'.join(lines) + '\n').encode('utf-8')


def _write_exact(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError('Refusing to overwrite different output: {}'.format(path))
        return
    temporary = path.with_name(path.name + '.tmp')
    with open(temporary, 'wb') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def _preflight_outputs(outputs, payloads):
    for name, path in outputs.items():
        path = Path(path)
        if path.exists() and path.read_bytes() != payloads[name]:
            raise RuntimeError(
                'Refusing to overwrite different output: {}'.format(path))


def main():
    args = parse_args()
    paths = dict(attrition=args.attrition, coverage=args.coverage,
                 interpolation=args.interpolation)
    report = build_report(
        _load(args.attrition), _load(args.coverage),
        _load(args.interpolation), paths)
    out_dir = Path(args.out_dir)
    outputs = {
        'json': out_dir / 'frozen_dino_native_s14_failure_attribution_v1.json',
        'csv': out_dir / 'frozen_dino_native_s14_failure_attribution_v1.csv',
        'markdown': out_dir / 'frozen_dino_native_s14_failure_attribution_v1.md',
    }
    payloads = {
        'json': _json_bytes(report),
        'csv': _csv_bytes(report['records']),
        'markdown': _markdown_bytes(report),
    }
    _preflight_outputs(outputs, payloads)
    for name in ('json', 'csv', 'markdown'):
        _write_exact(outputs[name], payloads[name])
    print(json.dumps({
        name: dict(path=str(path), sha256=_sha256(path),
                   size_bytes=path.stat().st_size)
        for name, path in outputs.items()
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
