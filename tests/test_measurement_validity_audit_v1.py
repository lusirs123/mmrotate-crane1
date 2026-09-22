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


def test_measurement_validity_can_slice_online_component_states():
    report = {
        'fixed_test_read': True,
        'evaluation_thresholds': {'center_error_threshold_px': 5.0},
        'records': [{'frame_key': 'seq_00001',
                     'offline_errors': {'raw': {'center': 1.0}},
                     'offline_riou': {'raw': 0.5}}]}
    online = {'records': [{'frame_key': 'seq_00001', 'observations': {
        'raw': {component: {'state': 'measurement'}
                for component in ('center', 'scale', 'angle')}}}]}
    audit = build_audit(report, {}, online_report=online)
    assert audit['protocol'] == 'base_v3_obb_measurement_validity_audit_v2'
    assert audit['online_component_states_included'] is True
    assert audit['methods']['raw']['component_raw']['center'][
        'output_coverage'] == 1.0


def test_intervals_are_validated_before_sorting_and_then_normalized():
    report = {
        'fixed_test_read': True,
        'records': [{'frame_key': 'seq_00001',
                     'offline_errors': {'raw': {'center': 1.0}},
                     'offline_riou': {'raw': 0.5}}]}
    audit = build_audit(report, {'seq': [[8, 9], [2, 3]]})
    assert audit['intervals'] == {'seq': [[2, 3], [8, 9]]}

    try:
        build_audit(report, {'seq': [[2, 3], 'bad']})
    except RuntimeError as exc:
        assert 'sequence=seq' in str(exc)
        assert 'index=1' in str(exc)
    else:
        raise AssertionError('expected interval structure failure')


def test_online_structure_error_has_full_context():
    report = {
        'fixed_test_read': True,
        'records': [{'frame_key': 'seq_00001',
                     'offline_errors': {'raw': {'center': 1.0}},
                     'offline_riou': {'raw': 0.5}}]}
    online = {'records': [{'frame_key': 'seq_00001', 'observations': {
        'raw': {'center': {'state': 'broken'}}}}]}
    try:
        build_audit(report, {}, online_report=online)
    except RuntimeError as exc:
        message = str(exc)
        assert 'frame=seq_00001' in message
        assert 'method=raw' in message
        assert 'component=center' in message
        assert "state='broken'" in message
    else:
        raise AssertionError('expected online observation failure')
