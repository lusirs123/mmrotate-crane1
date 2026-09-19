import json

from crane_project.tools.base_v3_obb_measurement_validity_audit_v1 import (
    build_audit)


def test_measurement_validity_keeps_raw_and_excludes_explicit_interval():
    report = {
        'protocol': 'example', 'fixed_test_read': True,
        'records': [
            {'frame_key': 'real_seq03_00129',
             'offline_errors': {'raw': {'center': 1.0}},
             'offline_riou': {'raw': 0.5}},
            {'frame_key': 'real_seq03_00130',
             'offline_errors': {'raw': {'center': None}},
             'offline_riou': {'raw': None}},
            {'frame_key': 'real_seq03_00188',
             'offline_errors': {'raw': {'center': 2.0}},
             'offline_riou': {'raw': 0.8}},
        ]}
    audit = build_audit(report, {'real_seq03': [[130, 187]]})
    assert audit['excluded_interval_count'] == 1
    assert audit['methods']['raw']['raw']['frame_count'] == 3
    assert audit['methods']['raw']['measurement_valid']['frame_count'] == 2
    assert audit['methods']['raw']['excluded_interval']['center_valid_count'] == 0
    assert audit['prediction_or_threshold_selection_performed'] is False


def test_measurement_validity_rejects_non_fixed_or_overlapping_intervals():
    report = {'fixed_test_read': False, 'records': []}
    try:
        build_audit(report, {})
    except RuntimeError as exc:
        assert 'fixed-test' in str(exc)
    else:
        raise AssertionError('expected fixed-test validation failure')
