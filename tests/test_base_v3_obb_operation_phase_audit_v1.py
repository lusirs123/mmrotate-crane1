import copy

import pytest

from crane_project.tools import base_v3_obb_operation_phase_audit_v1 as audit


def _contract():
    return dict(
        protocol=audit.CONTRACT_PROTOCOL, evidence_boundary='diagnostic',
        fixed_test_read=True, fixed_test_previously_exposed=True,
        phase_labels_used_online=False, parameter_tuning_authorized=False,
        expected_inputs=dict(
            fixed_test_report_protocol=audit.INPUT_PROTOCOL,
            fixed_test_report_sha256='a'*64,
            phase_manifest_protocol=audit.MANIFEST_PROTOCOL,
            phase_manifest_sha256='b'*64),
        fixed_test_scope=dict(frame_count=4, domain_counts={'real': 4}),
        expected_operation_group_counts={'transport': 2, 'grabbing': 2},
        reported_operation_groups=['transport', 'grabbing'],
        matched_coverage_targets=[1.0, 0.5],
        evaluation=dict(center_error_threshold_px=5.0,
                        scale_relative_error_threshold=0.1,
                        angle_error_threshold_deg=3.0),
        claim_limit='diagnostic only')


def _manifest():
    return dict(
        protocol=audit.MANIFEST_PROTOCOL,
        target_definition='grab_top_beam_obb',
        target_geometry_note='fixed top beam',
        online_input_authorized=False,
        rules=[
            dict(domain='real', sequence='seq', frame_start_inclusive=1,
                 frame_end_inclusive=2, operation_group='transport'),
            dict(domain='real', sequence='seq', frame_start_inclusive=3,
                 frame_end_inclusive=4, operation_group='grabbing')])


def _observation(state, value, risk=None):
    return dict(state=state, valid=state != 'unavailable',
                value=None if state == 'unavailable' else copy.deepcopy(value),
                risk=risk)


def _report():
    rows = []
    scale_errors = [0.01, 0.20, 0.02, 0.30]
    risks = [0.1, 0.9, 0.2, 0.8]
    for frame in range(1, 5):
        values = dict(center=[1.0, 2.0], scale=[10.0, 5.0], angle=0.1)
        errors = dict(center=1.0, scale=scale_errors[frame-1], angle=1.0)
        observations = {method: {
            component: _observation(
                'measurement', values[component],
                risks[frame-1] if method == 'v51_hybrid' and
                component == 'scale' else None)
            for component in ('center', 'scale', 'angle')}
            for method in audit.METHODS}
        offline = {method: copy.deepcopy(errors) for method in audit.METHODS}
        rows.append(dict(
            frame_key='real_seq_{:05d}'.format(frame), domain='real',
            sequence='seq', frame=frame, anchor_source='k1',
            anchor_score=1.0-frame*.1,
            online_observations=observations, offline_errors=offline))
    return dict(
        protocol=audit.INPUT_PROTOCOL,
        fixed_test_read=True, fixed_test_previously_exposed=True,
        frozen_calibration=True, threshold_fitting_performed=False,
        policy_selection_performed=False,
        eligible_for_parameter_tuning_from_this_report=False,
        method_inventory=list(audit.METHODS), records=rows)


def test_phase_labels_are_offline_and_all_records_match_once():
    result = audit.run(_report(), _manifest(), _contract())

    assert result['operation_group_counts'] == {
        'transport': 2, 'grabbing': 2}
    assert result['operation_labels_used_in_online_decisions'] is False
    assert result['parameter_tuning_authorized'] is False
    assert result['scale_matched_coverage']['transport'][
        'comparison_counts']['tie'] == 2
    assert result['component_performance']['grabbing']['raw']['scale'][
        'frame_count'] == 2


def test_overlapping_phase_rules_are_rejected():
    manifest = _manifest()
    manifest['rules'].append(dict(
        domain='real', sequence='seq', frame_start_inclusive=2,
        frame_end_inclusive=3, operation_group='overlap'))
    with pytest.raises(ValueError, match='matches 2 phase rules'):
        audit.assign_operation_groups(_report()['records'], manifest)


def test_phase_labels_cannot_be_enabled_online():
    contract = _contract()
    contract['phase_labels_used_online'] = True
    with pytest.raises(ValueError, match='offline-only'):
        audit.validate_contract(contract)
