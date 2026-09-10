import copy

import pytest

from crane_project.tools import base_v3_obb_top_beam_geometry_audit_v1 as v1
from crane_project.tools import base_v3_obb_top_beam_paired_diagnostic_v2 as v2
from crane_project.tools.base_v3_obb_component_reliability_continuous import (
    _error_from_value, _measurement_value)


def _contract():
    return dict(
        protocol=v2.CONTRACT_PROTOCOL,
        evidence_boundary='post-exposure diagnostic',
        fixed_test_read=True, fixed_test_previously_exposed=True,
        operation_labels_used_online=False,
        parameter_tuning_authorized=False,
        expected_inputs=dict(
            fixed_test_report_protocol=v2.V51_PROTOCOL,
            fixed_test_report_sha256='a'*64,
            geometry_v1_report_protocol=v2.GEOMETRY_V1_PROTOCOL,
            geometry_v1_report_sha256='b'*64,
            attribution_protocol=v2.ATTRIBUTION_PROTOCOL,
            attribution_sha256='c'*64,
            phase_manifest_protocol=v2.MANIFEST_PROTOCOL,
            phase_manifest_sha256='d'*64),
        fixed_test_scope=dict(
            frame_count=2, domain_counts={'real': 2},
            annotation_status='complete'),
        expected_operation_group_counts={'transport': 1, 'grabbing': 1},
        expected_anchor_source_counts={'k1': 1, 'dino_fallback': 1},
        evaluation=dict(
            center_error_threshold_px=5.0,
            scale_relative_error_threshold=.1,
            angle_error_threshold_deg=3.0,
            comparison_tolerance=1e-12,
            identity_tolerance=1e-9),
        claim_limit='diagnostic only')


def _manifest():
    return dict(
        protocol=v2.MANIFEST_PROTOCOL,
        target_definition='grab_top_beam_obb',
        target_geometry_note='rigid top beam',
        online_input_authorized=False,
        rules=[
            dict(domain='real', sequence='seq', frame_start_inclusive=1,
                 frame_end_inclusive=1, operation_group='transport'),
            dict(domain='real', sequence='seq', frame_start_inclusive=2,
                 frame_end_inclusive=2, operation_group='grabbing')])


def _rows():
    gt = [0.0, 0.0, 20.0, 10.0, 0.0]
    return [
        dict(frame_key='real_seq_00001', domain='real', sequence='seq',
             frame=1, anchor_source='k1', anchor_score=.9,
             base_v3_box=[1.0, 0.0, 21.0, 11.0, .05],
             k1_box=[2.0, 0.0, 22.0, 12.0, .10], dino_box=None,
             gt_box=copy.deepcopy(gt)),
        dict(frame_key='real_seq_00002', domain='real', sequence='seq',
             frame=2, anchor_source='dino_fallback', anchor_score=.6,
             base_v3_box=[2.0, 0.0, 25.0, 15.0, .20], k1_box=None,
             dino_box=[.5, 0.0, 18.0, 8.0, .05],
             gt_box=copy.deepcopy(gt))]


def _v51_report(rows):
    records = []
    for row in rows:
        values = {component: _measurement_value(row, component)
                  for component in ('center', 'scale', 'angle')}
        errors = {component: _error_from_value(
            value, row, component) for component, value in values.items()}
        records.append(dict(
            frame_key=row['frame_key'], anchor_source=row['anchor_source'],
            online_observations={'raw': {
                component: {'value': value}
                for component, value in values.items()}},
            offline_errors={'raw': errors}))
    return dict(protocol=v2.V51_PROTOCOL, records=records)


def _geometry_v1_report(rows, manifest):
    annotated = v1.attach_operation_groups(rows, manifest)
    return dict(
        protocol=v2.GEOMETRY_V1_PROTOCOL, frame_count=len(rows),
        operation_group_counts={'transport': 1, 'grabbing': 1},
        anchor_source_counts={'k1': 1, 'dino_fallback': 1},
        full_set_geometry_summary=v1.full_set_summaries(
            v1.add_geometry_errors(annotated)))


def test_same_frame_pairing_reports_improvement_degradation_and_bias():
    rows, manifest, contract = _rows(), _manifest(), _contract()
    result = v2.run(
        rows, _v51_report(rows), _geometry_v1_report(rows, manifest),
        manifest, contract)
    all_summary = result['paired_summaries']['all']

    assert all_summary['component_comparison']['center']['outcome_counts'] == {
        'improved': 1, 'degraded': 1, 'tie': 0}
    assert all_summary['component_comparison']['short_side'][
        'outcome_counts'] == {'improved': 1, 'degraded': 1, 'tie': 0}
    assert all_summary['signed_bias']['base_v3'][
        'short_side_relative_bias']['mean'] == pytest.approx(.3)
    assert result['paired_summaries'][
        'operation:transport|anchor:k1']['frame_count'] == 1
    assert result['input_reverification']['v51_report'][
        'maximum_raw_error_difference'] == pytest.approx(0.0)


def test_v51_error_mismatch_is_rejected_for_any_reconstructed_frame():
    rows, report, contract = _rows(), _v51_report(_rows()), _contract()
    report['records'][1]['offline_errors']['raw']['scale'] += .01
    annotated = v1.attach_operation_groups(rows, _manifest())
    with pytest.raises(RuntimeError, match='raw observation mismatch'):
        v2.verify_v51_report(report, annotated, contract)


def test_online_operation_labels_and_tuning_remain_forbidden():
    contract = _contract()
    contract['operation_labels_used_online'] = True
    with pytest.raises(ValueError, match='offline-only'):
        v2.validate_contract(contract)
