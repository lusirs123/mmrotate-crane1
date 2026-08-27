import json
import math
from pathlib import Path

import pytest

from crane_project.tools import (
    symeood_dino_application_domain_v4_audit as audit)


def _box(cx=10.0, cy=10.0, score=0.9):
    return [cx, cy, 8.0, 4.0, 0.0, score]


def _annotation(path, cx=10.0, cy=10.0):
    # Axis-aligned 8x4 rectangle in DOTA polygon format.
    values = [
        cx - 4.0, cy - 2.0,
        cx + 4.0, cy - 2.0,
        cx + 4.0, cy + 2.0,
        cx - 4.0, cy + 2.0,
    ]
    path.write_text(' '.join(str(value) for value in values) + ' grab 0\n')


def _record(filename, frame, sym, dino, sym_key='sym_eood_box'):
    domain, seq_id, _parsed = audit.parse_seq_frame(filename)
    return {
        'filename': '/old/root/' + filename,
        'sequence': '{}_{}'.format(domain, seq_id),
        'frame': frame,
        sym_key: sym,
        'dino_native_box': dino,
        'dino_invoked': True,
    }


def _payload_and_split(tmp_path, records, split='test'):
    split_root = tmp_path / split
    annotation_dir = split_root / 'annfiles'
    annotation_dir.mkdir(parents=True)
    for record in records:
        _annotation(annotation_dir / (Path(record['filename']).stem + '.txt'))
    payload = {
        'protocol': audit.INPUT_PROTOCOL,
        'records': records,
    }
    raw = json.dumps(payload).encode()
    return payload, raw, split_root


def _measurement_validity(sequences, status=None):
    payload = {
        'protocol': audit.MEASUREMENT_VALIDITY_PROTOCOL,
        'selection_basis': audit.MEASUREMENT_VALIDITY_SELECTION_BASIS,
        'sequences': sequences,
    }
    if status is not None:
        payload['status'] = status
    return payload


def test_v4_uses_fixed_application_domain_policy_and_legacy_sym_key(tmp_path):
    records = [
        _record('real_seq02_00001.jpg', 1, _box(cx=100), _box()),
        _record('real_seq02_00002.jpg', 2, _box(), None),
        _record('sim_seq09_00001.jpg', 1, _box(), _box(cx=100)),
        _record('sim_seq09_00002.jpg', 2, None, _box()),
    ]
    payload, raw, split_root = _payload_and_split(tmp_path, records)
    result = audit.audit_payload(
        payload, raw, split_root, 'fixed-target',
        required_frame_count=4)

    assert result['input']['sym_eood_box_key'] == 'sym_eood_box'
    assert result['policy']['sequence_frame_slice_routing'] is False
    assert result['policy']['detector_forward'] is False
    decisions = result['v4']['decisions']
    assert [row['selected_source'] for row in decisions] == [
        'dino_native', 'sym_eood_fallback', 'sym_eood', 'missing']
    assert decisions[0]['selected_box'] == pytest.approx(_box())
    assert decisions[1]['selected_box'] == pytest.approx(_box())
    assert decisions[2]['selected_box'] == pytest.approx(_box())
    assert decisions[3]['selected_box'] is None
    assert result['documented_gate']['checks'][
        'sim_stream_is_exact_symeood_primary'] is True
    assert result['v4_vs_sym_eood']['gained_hit_frame_keys'] == [
        'real_seq02|1']


def test_v4_prefers_complete_original_sym_key(tmp_path):
    records = [
        _record(
            'real_seq02_00001.jpg', 1, _box(), _box(),
            sym_key='sym_eood_original_box'),
        _record(
            'sim_seq09_00001.jpg', 1, _box(), _box(),
            sym_key='sym_eood_original_box'),
    ]
    payload, raw, split_root = _payload_and_split(tmp_path, records)
    result = audit.audit_payload(
        payload, raw, split_root, 'fixed-target',
        required_frame_count=2)
    assert result['input']['sym_eood_box_key'] == 'sym_eood_original_box'


def test_v4_rejects_any_frame_where_dino_was_not_computed(tmp_path):
    records = [
        _record('real_seq02_00001.jpg', 1, _box(), None),
        _record('sim_seq09_00001.jpg', 1, _box(), None),
    ]
    records[0]['dino_invoked'] = False
    payload, raw, split_root = _payload_and_split(tmp_path, records)
    with pytest.raises(RuntimeError, match='computed on every input frame'):
        audit.audit_payload(
            payload, raw, split_root, 'fixed-target',
            required_frame_count=2)


def test_v4_rejects_routed_input_protocol(tmp_path):
    records = [
        _record('real_seq02_00001.jpg', 1, _box(), _box()),
        _record('sim_seq09_00001.jpg', 1, _box(), _box()),
    ]
    payload, raw, split_root = _payload_and_split(tmp_path, records)
    payload['protocol'] = 'lane_isolated_conditional_dino_v3'
    raw = json.dumps(payload).encode()
    with pytest.raises(RuntimeError, match='unrouted all-lane audit'):
        audit.audit_payload(
            payload, raw, split_root, 'fixed-target',
            required_frame_count=2)


def test_v4_rejects_unknown_domain_instead_of_routing_on_identity(tmp_path):
    records = [
        _record('real_seq02_00001.jpg', 1, _box(), _box()),
        {
            'filename': '/old/root/camera_x_00001.jpg',
            'sequence': 'camera_x',
            'frame': 1,
            'sym_eood_box': _box(),
            'dino_native_box': _box(),
            'dino_invoked': True,
        },
    ]
    payload, raw, split_root = _payload_and_split(tmp_path, records)
    with pytest.raises(RuntimeError, match='unknown application domain'):
        audit.audit_payload(
            payload, raw, split_root, 'fixed-target',
            required_frame_count=2)


def test_v4_rejects_frame_metadata_misalignment(tmp_path):
    records = [
        _record('real_seq02_00001.jpg', 99, _box(), _box()),
        _record('sim_seq09_00001.jpg', 1, _box(), _box()),
    ]
    payload, raw, split_root = _payload_and_split(tmp_path, records)
    with pytest.raises(RuntimeError, match='frame metadata disagrees'):
        audit.audit_payload(
            payload, raw, split_root, 'fixed-target',
            required_frame_count=2)


def test_source_val_uses_full_metric_v2_without_fixed_target_claim(tmp_path):
    records = [
        _record('real_seq07_00001.jpg', 1, _box(), _box()),
        _record('sim_seq10_00001.jpg', 1, _box(), _box()),
    ]
    payload, raw, split_root = _payload_and_split(
        tmp_path, records, split='val')
    result = audit.audit_payload(
        payload, raw, split_root, 'source-val',
        required_frame_count=2)
    assert result['metric_protocol_version'] == 2
    assert result['evidence_boundary'] == 'source_only_gate'
    assert result['documented_gate'][
        'fixed_test_sim_a_rmse_reference_deg'] is None
    assert math.isfinite(result['v4']['metrics']['sim/A-RMSE(deg)'])
    assert result['decision'] == 'PASS_SOURCE_VAL_DOCUMENTED_GATE'
    assert result['eligible_for_formal_config_from_this_report_alone'] is False


def test_fixed_target_output_keeps_diagnostic_boundary(tmp_path):
    records = [
        _record('real_seq02_00001.jpg', 1, _box(), _box()),
        _record('sim_seq09_00001.jpg', 1, _box(), _box()),
    ]
    payload, raw, split_root = _payload_and_split(tmp_path, records)
    result = audit.audit_payload(
        payload, raw, split_root, 'fixed-target',
        required_frame_count=2)
    assert result['evidence_boundary'] == (
        'fixed_target_diagnostic_not_unknown_sequence')
    assert result['documented_gate']['checks'][
        'sim_a_rmse_le_1_5487_deg'] is True
    assert result['decision'] == 'PASS_FIXED_TARGET_DIAGNOSTIC_GATE'
    assert result['eligible_for_unknown_sequence_claim'] is False


def test_six_present_wrong_dino_frames_close_missing_only_holder(tmp_path):
    records = [
        _record(
            'real_seq03_{:05d}.jpg'.format(frame), frame,
            _box(), _box(cx=100.0))
        for frame in range(1, 7)
    ]
    records.append(
        _record('sim_seq09_00001.jpg', 1, _box(), _box()))
    payload, raw, split_root = _payload_and_split(tmp_path, records)
    result = audit.audit_payload(
        payload, raw, split_root, 'fixed-target',
        required_frame_count=7)

    attribution = result['failure_attribution']
    assert attribution['max_run_length'] == 6
    assert len(attribution['over_limit_runs']) == 1
    run = attribution['over_limit_runs'][0]
    assert run['dino_missing_count'] == 0
    assert run['dino_present_wrong_count'] == 6
    assert run['sym_eood_hit_count'] == 6
    assert run['selected_missing_count'] == 0
    assert attribution['missing_only_observation_holder'] == {
        'eligible_for_followup_counterfactual': False,
        'reason': 'over_limit_run_has_no_selected_missing_frame',
        'necessary_condition': 'failed',
    }


def test_failure_attribution_does_not_bridge_frame_id_gap(tmp_path):
    records = [
        _record(
            'real_seq02_{:05d}.jpg'.format(frame), frame,
            _box(), _box(cx=100.0))
        for frame in (1, 2, 3, 10, 11, 12)
    ]
    records.append(
        _record('sim_seq09_00001.jpg', 1, _box(), _box()))
    payload, raw, split_root = _payload_and_split(tmp_path, records)
    result = audit.audit_payload(
        payload, raw, split_root, 'fixed-target',
        required_frame_count=7)

    attribution = result['failure_attribution']
    assert attribution['max_run_length'] == 3
    assert [run['length'] for run in attribution['runs']] == [3, 3]
    assert attribution['missing_only_observation_holder']['reason'] == (
        'no_over_limit_failure_run')


def test_measurement_validity_adds_separate_gate_without_overriding_raw(
        tmp_path):
    records = [
        _record(
            'real_seq03_{:05d}.jpg'.format(frame), frame,
            _box(), _box(cx=100.0))
        for frame in range(1, 7)
    ]
    records.append(_record('sim_seq09_00001.jpg', 1, _box(), _box()))
    payload, raw, split_root = _payload_and_split(tmp_path, records)
    validity = _measurement_validity({
        'real_seq03': {
            'default_valid': True,
            'invalid_intervals': [{
                'start_frame': 2,
                'end_frame': 3,
                'reason': 'material_contact',
            }],
        },
    }, status='POST_HOC_OPERATIONAL_VALIDITY_DIAGNOSTIC')

    result = audit.audit_payload(
        payload, raw, split_root, 'fixed-target',
        required_frame_count=7,
        measurement_validity=validity)

    assert result['documented_gate']['passed'] is False
    assert result['decision'] == 'STOP_V4_GATE_FAILED'
    operational = result['measurement_validity']
    assert operational['original_documented_gate_overridden'] is False
    assert operational['eligible_for_original_gate_override'] is False
    assert operational['operational_gate']['passed'] is True
    assert operational['scope']['all_frame_count'] == 7
    assert operational['scope']['sim_frame_count'] == 1
    assert operational['scope']['raw_real_frame_count'] == 6
    assert operational['scope']['measurement_valid_real_frame_count'] == 4
    assert operational['scope']['measurement_invalid_real_frame_count'] == 2
    assert operational['scope']['excluded_hit_count'] == 0
    assert operational['scope']['excluded_miss_count'] == 2
    assert operational['metrics']['real/MCML_max(frames)'] == 3
    assert operational['failure_attribution']['max_run_length'] == 3
    assert [row['frame_key'] for row in operational['excluded_frames']] == [
        'real_seq03|2', 'real_seq03|3']


def test_measurement_validity_counts_excluded_successes(tmp_path):
    records = [
        _record('real_seq02_00001.jpg', 1, _box(), _box()),
        _record('real_seq02_00002.jpg', 2, _box(), _box()),
        _record('sim_seq09_00001.jpg', 1, _box(), _box()),
    ]
    payload, raw, split_root = _payload_and_split(tmp_path, records)
    validity = _measurement_validity({
        'real_seq02': {
            'default_valid': True,
            'invalid_intervals': [{
                'start_frame': 1,
                'end_frame': 1,
                'reason': 'grabbing_or_closing',
            }],
        },
    })

    result = audit.audit_payload(
        payload, raw, split_root, 'fixed-target',
        required_frame_count=3,
        measurement_validity=validity)
    scope = result['measurement_validity']['scope']
    assert scope['excluded_hit_count'] == 1
    assert scope['excluded_miss_count'] == 0
    assert scope['per_sequence']['real_seq02'][
        'measurement_valid_frame_count'] == 1


def test_measurement_validity_requires_every_real_sequence(tmp_path):
    records = [
        _record('real_seq02_00001.jpg', 1, _box(), _box()),
        _record('real_seq03_00001.jpg', 1, _box(), _box()),
        _record('sim_seq09_00001.jpg', 1, _box(), _box()),
    ]
    payload, raw, split_root = _payload_and_split(tmp_path, records)
    validity = _measurement_validity({
        'real_seq03': {
            'default_valid': True,
            'invalid_intervals': [],
        },
    })

    with pytest.raises(RuntimeError, match='exactly match real input'):
        audit.audit_payload(
            payload, raw, split_root, 'fixed-target',
            required_frame_count=3,
            measurement_validity=validity)


def test_measurement_validity_rejects_source_val_masking(tmp_path):
    records = [
        _record('real_seq07_00001.jpg', 1, _box(), _box()),
        _record('sim_seq10_00001.jpg', 1, _box(), _box()),
    ]
    payload, raw, split_root = _payload_and_split(
        tmp_path, records, split='val')
    validity = _measurement_validity({
        'real_seq07': {
            'default_valid': True,
            'invalid_intervals': [],
        },
    })

    with pytest.raises(RuntimeError, match='fixed-target diagnostic only'):
        audit.audit_payload(
            payload, raw, split_root, 'source-val',
            required_frame_count=2,
            measurement_validity=validity)
