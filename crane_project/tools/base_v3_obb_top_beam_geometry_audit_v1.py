#!/usr/bin/env python3
"""Diagnose rigid top-beam OBB geometry on the exposed fixed TEST.

The audit reconstructs the already frozen Base V3, formal K1, DINO, and GT
boxes.  Operation labels and GT are used only after inference.  Full-set
statistics are accompanied by a contract-fixed, error-independent visual
sample; neither output authorizes policy tuning or phase-conditioned inference.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from crane_project.tools import base_v3_obb_hybrid_fixed_test_eval_v51 as v51
from crane_project.tools.base_v3_obb_component_reliability_continuous import (
    _geometry, _periodic_error_deg)
from crane_project.tools.base_v3_obb_fixed_test_eval import (
    ATTRIBUTION_PROTOCOL, _read_json, _require_identity, build_records)
from crane_project.tools.base_v3_obb_reliability_baseline import (
    _identity, _write_exact)


PROTOCOL = 'base_v3_obb_top_beam_geometry_audit_v1'
CONTRACT_PROTOCOL = 'base_v3_obb_top_beam_geometry_audit_contract_v1'
MANIFEST_PROTOCOL = 'fixed_test_operation_phase_manifest_v2'
INPUT_PROTOCOL = v51.PROTOCOL
SOURCES = ('base_v3', 'k1', 'dino')


def _valid_sha256(value):
    text = str(value)
    return len(text) == 64 and all(char in '0123456789abcdef' for char in text)


def validate_contract(contract):
    if contract.get('protocol') != CONTRACT_PROTOCOL:
        raise ValueError('Unexpected top-beam geometry audit contract')
    required_false = (
        'phase_labels_used_online', 'sample_selection_uses_model_errors',
        'parameter_tuning_authorized')
    if contract.get('fixed_test_read') is not True:
        raise ValueError('Top-beam audit must declare fixed TEST read')
    if contract.get('fixed_test_previously_exposed') is not True:
        raise ValueError('Top-beam audit must preserve prior TEST exposure')
    for field in required_false:
        if contract.get(field) is not False:
            raise ValueError(field + ' must be false')
    expected = contract.get('expected_inputs', {})
    if expected.get('fixed_test_report_protocol') != INPUT_PROTOCOL:
        raise ValueError('Contract must bind the frozen V5.1 TEST report')
    if expected.get('attribution_protocol') != ATTRIBUTION_PROTOCOL:
        raise ValueError('Contract must bind fixed-target attribution')
    if expected.get('phase_manifest_protocol') != MANIFEST_PROTOCOL:
        raise ValueError('Contract must bind operation manifest V2')
    for field in ('fixed_test_report_sha256', 'attribution_sha256',
                  'phase_manifest_sha256'):
        if not _valid_sha256(expected.get(field)):
            raise ValueError(field + ' must be a lowercase SHA256')
    scope = contract.get('fixed_test_scope', {})
    if int(scope.get('frame_count', 0)) <= 0:
        raise ValueError('Fixed TEST frame count must be positive')
    if sum(scope.get('domain_counts', {}).values()) != scope['frame_count']:
        raise ValueError('Domain counts must sum to fixed TEST frame count')
    if scope.get('annotation_status') != 'complete':
        raise ValueError('Top-beam audit requires complete TEST annotations')
    expected_groups = contract.get('expected_operation_group_counts', {})
    if sum(expected_groups.values()) != scope['frame_count']:
        raise ValueError('Operation group counts must sum to TEST frame count')
    if int(contract.get('expected_sample_count', 0)) <= 0:
        raise ValueError('Expected visual sample count must be positive')
    plans = contract.get('sample_plan')
    if not isinstance(plans, list) or not plans:
        raise ValueError('At least one fixed sample plan is required')
    if len({plan.get('sample_group') for plan in plans}) != len(plans):
        raise ValueError('Sample group names must be unique')
    for plan in plans:
        if plan.get('selection') not in (
                'uniform_ordered_index',
                'uniform_ordered_index_plus_required_frames'):
            raise ValueError('Unsupported sample selection rule')
        if int(plan.get('sample_count', 0)) <= 0:
            raise ValueError('Each sample group needs a positive sample count')
    rendering = contract.get('rendering', {})
    if rendering.get('require_all_images') is not True:
        raise ValueError('All fixed visual sample images must be required')
    if set(rendering.get('box_colors_bgr', {})) != {
            'gt', 'base_v3', 'k1', 'dino'}:
        raise ValueError('Rendering colors must cover every box source')


def validate_manifest(manifest):
    if manifest.get('protocol') != MANIFEST_PROTOCOL:
        raise ValueError('Unexpected operation-phase manifest')
    if manifest.get('target_definition') != 'grab_top_beam_obb':
        raise ValueError('Audit requires rigid top-beam OBB semantics')
    if manifest.get('online_input_authorized') is not False:
        raise ValueError('Operation labels must remain offline-only')
    rules = manifest.get('rules')
    if not isinstance(rules, list) or not rules:
        raise ValueError('Operation manifest needs rules')


def _rule_matches(row, rule):
    if (row['domain'], row['sequence']) != (
            rule['domain'], rule['sequence']):
        return False
    frame = int(row['frame'])
    start = rule.get('frame_start_inclusive')
    end = rule.get('frame_end_inclusive')
    return ((start is None or frame >= int(start)) and
            (end is None or frame <= int(end)))


def attach_operation_groups(rows, manifest):
    output = []
    for row in rows:
        matches = [rule for rule in manifest['rules']
                   if _rule_matches(row, rule)]
        if len(matches) != 1:
            raise ValueError('{} matches {} operation rules'.format(
                row['frame_key'], len(matches)))
        copied = dict(row)
        copied['operation_group'] = matches[0]['operation_group']
        copied['operation_interpretation'] = matches[0].get('interpretation')
        output.append(copied)
    return output


def _source_box(row, source):
    return row[source + '_box']


def geometry_error(box, gt_box):
    center, long_side, short_side, angle = _geometry(box)
    gt_center, gt_long, gt_short, gt_angle = _geometry(gt_box)
    long_error = abs(long_side-gt_long)/max(gt_long, 1e-9)
    short_error = abs(short_side-gt_short)/max(gt_short, 1e-9)
    if abs(long_error-short_error) <= 1e-12:
        dominant = 'tie'
    else:
        dominant = 'long_side' if long_error > short_error else 'short_side'
    return dict(
        center_error_px=float(np.linalg.norm(center-gt_center)),
        long_side_relative_error=float(long_error),
        short_side_relative_error=float(short_error),
        max_side_relative_error=float(max(long_error, short_error)),
        dominant_scale_error_component=dominant,
        angle_error_deg=float(_periodic_error_deg(angle, gt_angle)),
        predicted_geometry=dict(center=center.tolist(), long_side=long_side,
                                short_side=short_side,
                                angle_rad=float(angle)),
        gt_geometry=dict(center=gt_center.tolist(), long_side=gt_long,
                         short_side=gt_short, angle_rad=float(gt_angle)))


def add_geometry_errors(rows):
    output = []
    for row in rows:
        copied = dict(row)
        copied['geometry_errors'] = {}
        for source in SOURCES:
            box = _source_box(row, source)
            copied['geometry_errors'][source] = (
                None if box is None else geometry_error(box, row['gt_box']))
        output.append(copied)
    return output


def _metric_summary(values):
    if not values:
        return dict(mean=None, median=None, p90=None, maximum=None)
    data = np.asarray(values, dtype=np.float64)
    return dict(mean=float(np.mean(data)), median=float(np.median(data)),
                p90=float(np.quantile(data, .9)), maximum=float(np.max(data)))


def summarize(rows):
    metrics = ('center_error_px', 'long_side_relative_error',
               'short_side_relative_error', 'max_side_relative_error',
               'angle_error_deg')
    result = {}
    for source in SOURCES:
        available = [row['geometry_errors'][source] for row in rows
                     if row['geometry_errors'][source] is not None]
        result[source] = dict(
            frame_count=len(rows), available_count=len(available),
            availability=len(available)/len(rows),
            metrics={metric: _metric_summary(
                [item[metric] for item in available]) for metric in metrics},
            dominant_scale_error_counts=dict(Counter(
                item['dominant_scale_error_component'] for item in available)))
    return result


def full_set_summaries(rows):
    groups = {'all': rows}
    for group in sorted({row['operation_group'] for row in rows}):
        groups['operation:' + group] = [
            row for row in rows if row['operation_group'] == group]
    for source in ('k1', 'dino_fallback'):
        selected = [row for row in rows if row['anchor_source'] == source]
        if selected:
            groups['anchor:' + source] = selected
    return {name: summarize(selected) for name, selected in groups.items()}


def _in_plan(row, plan):
    if (row['domain'], row['sequence'], row['operation_group']) != (
            plan['domain'], plan['sequence'], plan['operation_group']):
        return False
    frame = int(row['frame'])
    start, end = (plan.get('frame_start_inclusive'),
                  plan.get('frame_end_inclusive'))
    return ((start is None or frame >= int(start)) and
            (end is None or frame <= int(end)))


def select_fixed_sample(rows, plans):
    selected = []
    used_keys = set()
    for plan in plans:
        candidates = sorted((row for row in rows if _in_plan(row, plan)),
                            key=lambda row: (row['frame'], row['frame_key']))
        count = int(plan['sample_count'])
        if len(candidates) < count:
            raise ValueError('{} has only {} candidates for {} samples'.format(
                plan['sample_group'], len(candidates), count))
        indices = np.rint(np.linspace(0, len(candidates)-1, count)).astype(int)
        chosen = [candidates[index] for index in indices]
        by_frame = {int(row['frame']): row for row in candidates}
        for frame in plan.get('required_frames', []):
            if int(frame) not in by_frame:
                raise ValueError('Required frame {} missing from {}'.format(
                    frame, plan['sample_group']))
            chosen.append(by_frame[int(frame)])
        for row in chosen:
            if row['frame_key'] in used_keys:
                continue
            copied = dict(row)
            copied['sample_group'] = plan['sample_group']
            copied['sample_selection'] = plan['selection']
            selected.append(copied)
            used_keys.add(row['frame_key'])
    return selected


def _obb_points(box):
    center = (float(box[0]), float(box[1]))
    size = (float(box[2]), float(box[3]))
    return np.rint(cv2.boxPoints(
        (center, size, math.degrees(float(box[4]))))).astype(np.int32)


def render_overlay(row, image_path, output_path, colors):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError('Cannot read sample image: ' + str(image_path))
    boxes = [('gt', row['gt_box'])]
    boxes.extend((source, _source_box(row, source)) for source in SOURCES)
    for name, box in boxes:
        if box is None:
            continue
        color = tuple(int(value) for value in colors[name])
        cv2.polylines(image, [_obb_points(box)], True, color, 3,
                      lineType=cv2.LINE_AA)
    lines = [row['frame_key'] + ' | ' + row['operation_group'],
             'anchor=' + str(row['anchor_source'])]
    for source in SOURCES:
        error = row['geometry_errors'][source]
        if error is not None:
            lines.append('{} C={:.1f}px L={:.3f} S={:.3f} A={:.1f}deg'.format(
                source, error['center_error_px'],
                error['long_side_relative_error'],
                error['short_side_relative_error'], error['angle_error_deg']))
    height = 26*len(lines)+12
    cv2.rectangle(image, (0, 0), (min(image.shape[1]-1, 900), height),
                  (0, 0, 0), -1)
    for index, line in enumerate(lines):
        cv2.putText(image, line, (10, 26*(index+1)),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2,
                    cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError('Failed to write overlay: ' + str(output_path))


def validate_v51_report(report, rows, contract):
    if report.get('protocol') != INPUT_PROTOCOL:
        raise ValueError('Unexpected frozen V5.1 TEST report')
    records = report.get('records')
    if not isinstance(records, list):
        raise ValueError('Frozen V5.1 TEST report has no records')
    report_by_key = {row['frame_key']: row for row in records}
    row_by_key = {row['frame_key']: row for row in rows}
    if set(report_by_key) != set(row_by_key):
        raise RuntimeError('Reconstructed and V5.1 TEST frame keys differ')
    for key, row in row_by_key.items():
        if report_by_key[key].get('anchor_source') != row['anchor_source']:
            raise RuntimeError('Anchor source changed for ' + key)
    if len(rows) != int(contract['fixed_test_scope']['frame_count']):
        raise RuntimeError('Reconstructed TEST frame count changed')


def run(rows, report, manifest, contract, image_dir, out_image_dir,
        identities=None):
    validate_contract(contract)
    validate_manifest(manifest)
    validate_v51_report(report, rows, contract)
    annotated = add_geometry_errors(attach_operation_groups(rows, manifest))
    operation_counts = dict(Counter(
        row['operation_group'] for row in annotated))
    if operation_counts != contract['expected_operation_group_counts']:
        raise RuntimeError('Operation group counts changed: {!r}'.format(
            operation_counts))
    sample = select_fixed_sample(annotated, contract['sample_plan'])
    if len(sample) != int(contract['expected_sample_count']):
        raise RuntimeError('Fixed visual sample count changed: {}'.format(
            len(sample)))
    colors = contract['rendering']['box_colors_bgr']
    rendered = []
    for row in sample:
        image_path = Path(image_dir)/(row['frame_key'] + '.jpg')
        output_path = Path(out_image_dir)/(row['frame_key'] + '_overlay.png')
        render_overlay(row, image_path, output_path, colors)
        rendered.append(dict(frame_key=row['frame_key'],
                             sample_group=row['sample_group'],
                             image_path=str(image_path.resolve()),
                             overlay_path=str(output_path.resolve())))
    sample_records = []
    for row in sample:
        sample_records.append({key: row[key] for key in (
            'frame_key', 'domain', 'sequence', 'frame', 'operation_group',
            'sample_group', 'sample_selection', 'anchor_source',
            'anchor_score', 'gt_box', 'base_v3_box', 'k1_box', 'dino_box',
            'geometry_errors')})
    return dict(
        protocol=PROTOCOL,
        evidence_boundary=contract['evidence_boundary'],
        claim_status='POST_EXPOSURE_TOP_BEAM_GEOMETRY_DIAGNOSTIC_ONLY',
        target_definition=manifest['target_definition'],
        target_geometry_note=manifest['target_geometry_note'],
        fixed_test_read=True, fixed_test_previously_exposed=True,
        operation_labels_attached_after_inference=True,
        operation_labels_used_in_online_decisions=False,
        sample_selection_uses_model_scores=False,
        sample_selection_uses_model_errors=False,
        sample_selection_uses_gt=False,
        parameter_tuning_authorized=False,
        inputs=identities,
        frame_count=len(annotated),
        operation_group_counts=operation_counts,
        anchor_source_counts=dict(Counter(
            row['anchor_source'] for row in annotated)),
        full_set_geometry_summary=full_set_summaries(annotated),
        sample_count=len(sample_records), sample_records=sample_records,
        rendered_overlays=rendered,
        eligible_for_parameter_tuning_from_this_report=False,
        eligible_for_phase_conditioned_online_policy_claim=False,
        eligible_for_fresh_sequence_generalization_claim=False,
        claim_limit=contract['claim_limit'])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixed-test-attribution', required=True)
    parser.add_argument('--fixed-test-v51-report', required=True)
    parser.add_argument('--phase-manifest', required=True)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--base-v3-results')
    parser.add_argument('--k1-results')
    parser.add_argument('--all-lane-audit')
    parser.add_argument('--ann-dir')
    parser.add_argument('--image-dir', default=(
        'crane_project/data/crane_grab/test/images'))
    parser.add_argument('--out-image-dir', required=True)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    contract_id, contract = _read_json(args.contract, 'contract')
    validate_contract(contract)
    attribution_id, attribution = _read_json(
        args.fixed_test_attribution, 'fixed-test attribution')
    report_id, report = _read_json(
        args.fixed_test_v51_report, 'frozen V5.1 TEST report')
    manifest_id, manifest = _read_json(args.phase_manifest, 'phase manifest')
    expected = contract['expected_inputs']
    _require_identity(attribution_id, expected['attribution_sha256'],
                      'fixed-test attribution')
    _require_identity(report_id, expected['fixed_test_report_sha256'],
                      'frozen V5.1 TEST report')
    _require_identity(manifest_id, expected['phase_manifest_sha256'],
                      'phase manifest V2')
    rows, reconstruction = build_records(attribution, args, contract)
    identities = dict(contract=contract_id, attribution=attribution_id,
                      fixed_test_v51_report=report_id,
                      phase_manifest=manifest_id,
                      reconstruction=reconstruction)
    payload = run(rows, report, manifest, contract, args.image_dir,
                  args.out_image_dir, identities)
    output = _write_exact(args.out_json, payload)
    all_summary = payload['full_set_geometry_summary']['all']
    print(json.dumps(dict(
        output=output, frame_count=payload['frame_count'],
        operation_group_counts=payload['operation_group_counts'],
        anchor_source_counts=payload['anchor_source_counts'],
        sample_count=payload['sample_count'],
        base_v3_scale_dominance=all_summary['base_v3'][
            'dominant_scale_error_counts'],
        claim_status=payload['claim_status'],
        operation_labels_used_in_online_decisions=False,
        sample_selection_uses_model_errors=False,
        parameter_tuning_authorized=False), indent=2))


if __name__ == '__main__':
    main()
