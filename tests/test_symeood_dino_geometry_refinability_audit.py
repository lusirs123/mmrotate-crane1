import json
import math
from pathlib import Path

import pytest

from crane_project.tools import (
    symeood_dino_geometry_refinability_audit as audit)


def _box(cx=10.0, cy=10.0, width=8.0, height=4.0,
         angle=0.0, score=0.9):
    return [cx, cy, width, height, angle, score]


def _annotation(path, cx=10.0, cy=10.0, width=8.0, height=4.0):
    values = [
        cx - width / 2.0, cy - height / 2.0,
        cx + width / 2.0, cy - height / 2.0,
        cx + width / 2.0, cy + height / 2.0,
        cx - width / 2.0, cy + height / 2.0,
    ]
    path.write_text(
        ' '.join(str(value) for value in values) + ' grab 0\n')


def _record(filename, frame, sym, dino):
    domain, seq_id, _ = audit.parse_seq_frame(filename)
    return dict(
        filename='/old/root/' + filename,
        sequence='{}_{}'.format(domain, seq_id),
        frame=frame,
        sym_eood_box=sym,
        dino_native_box=dino,
        dino_invoked=True)


def _payload(tmp_path, records, split='test'):
    split_root = tmp_path / split
    annotations = split_root / 'annfiles'
    annotations.mkdir(parents=True)
    for record in records:
        _annotation(annotations / (
            Path(record['filename']).stem + '.txt'))
    payload = dict(protocol=audit.INPUT_PROTOCOL, records=records)
    return payload, json.dumps(payload).encode(), split_root


def _validity(real_sequences):
    return dict(
        protocol=audit.base.MEASUREMENT_VALIDITY_PROTOCOL,
        status=audit.base.MEASUREMENT_VALIDITY_STATUS,
        selection_basis=audit.base.MEASUREMENT_VALIDITY_SELECTION_BASIS,
        sequences=real_sequences)


def test_geometry_audit_is_domain_agnostic_and_marks_oracles_non_deployable(
        tmp_path):
    records = [
        _record(
            'real_seq02_00001.jpg', 1,
            _box(), _box(cx=12.0, width=12.0, height=8.0)),
        _record(
            'sim_seq09_00001.jpg', 1,
            _box(), _box(cx=12.0, width=12.0, height=8.0)),
    ]
    payload, raw, split_root = _payload(tmp_path, records)
    result = audit.audit_payload(
        payload, raw, split_root, 'fixed-target',
        required_frame_count=2)

    assert result['audit_contract']['domain_specific_routing'] is False
    assert result['audit_contract']['sequence_frame_slice_routing'] is False
    assert result['audit_contract'][
        'gt_component_replacement_is_non_deployable_oracle'] is True
    assert result['audit_contract']['eligible_for_runtime_policy'] is False
    decomposition = result['both_present_component_decomposition']
    assert decomposition['real']['mean_riou'][
        'dino_center_sym_geometry'] > decomposition['real'][
            'mean_riou']['dino_native']
    assert decomposition['real']['mean_riou']['sym_eood'] == pytest.approx(1.0)
    assert decomposition['sim']['candidate_win_counts'] == {
        'sym_eood_better': 1}
    assert result['frame_rows'][0]['candidate_oracle_source'] == 'sym_eood'


def test_gt_component_oracles_isolate_dino_size_error(tmp_path):
    records = [
        _record(
            'real_seq02_00001.jpg', 1,
            _box(), _box(width=16.0, height=8.0)),
        _record(
            'sim_seq09_00001.jpg', 1,
            _box(), _box(width=16.0, height=8.0)),
    ]
    payload, raw, split_root = _payload(tmp_path, records)
    result = audit.audit_payload(
        payload, raw, split_root, 'fixed-target',
        required_frame_count=2)

    real = result['capacity_summary']['real']
    assert real['dino_mean_riou'] == pytest.approx(0.25)
    assert real['dino_center_gt_geometry_mean_riou'] == pytest.approx(1.0)
    assert real['geometry_oracle_mean_riou_gain'] == pytest.approx(0.75)
    refinability = result['dino_refinability']['real']
    assert refinability['gt_center_inside_dino_rate'] == pytest.approx(1.0)
    width = refinability['residual_quantiles']['log_width_ratio']
    assert width['median'] == pytest.approx(math.log(0.5))
    assert result['frame_rows'][0]['riou'][
        'dino_gt_center_oracle'] == pytest.approx(0.25)
    assert result['frame_rows'][0]['riou'][
        'dino_gt_size_oracle'] == pytest.approx(1.0)


def test_candidate_oracle_transition_analysis_separates_source_switches(
        tmp_path):
    records = [
        _record(
            'real_seq02_00001.jpg', 1,
            _box(), _box(cx=100.0)),
        _record(
            'real_seq02_00002.jpg', 2,
            _box(cx=100.0), _box()),
        _record('sim_seq09_00001.jpg', 1, _box(), _box()),
    ]
    payload, raw, split_root = _payload(tmp_path, records)
    result = audit.audit_payload(
        payload, raw, split_root, 'fixed-target',
        required_frame_count=3)

    transition = result['candidate_oracle_transition_analysis']['real']
    assert transition['same_source']['transition_count'] == 0
    assert transition['source_switch']['transition_count'] == 1
    assert transition['source_switch']['mean_dfr_fraction'] == pytest.approx(0.0)
    assert transition['source_switch']['mean_aci'] == pytest.approx(1.0)
    counts = result['candidate_counts']['real'][
        'candidate_oracle_source_counts']
    assert counts == {'sym_eood': 1, 'dino_native': 1}


def test_measurement_validity_is_a_separate_evaluation_slice(tmp_path):
    records = [
        _record('real_seq02_00001.jpg', 1, _box(), _box()),
        _record('real_seq02_00002.jpg', 2, _box(), _box(cx=100.0)),
        _record('sim_seq09_00001.jpg', 1, _box(), _box()),
    ]
    payload, raw, split_root = _payload(tmp_path, records)
    validity = _validity({
        'real_seq02': dict(
            default_valid=True,
            invalid_intervals=[dict(
                start_frame=2,
                end_frame=2,
                reason='material_contact')]),
    })
    result = audit.audit_payload(
        payload, raw, split_root, 'fixed-target',
        required_frame_count=3,
        measurement_validity=validity)

    assert result['input']['frame_count'] == 3
    assert result['candidate_counts']['all']['frame_count'] == 3
    measurement = result['measurement_validity']
    assert measurement['use'] == 'evaluation_scope_only_never_model_input'
    assert measurement['scope']['kept_frame_count'] == 2
    assert measurement['scope']['excluded_real_frame_count'] == 1
    assert measurement['eligible_for_original_gate_override'] is False
    assert result['audit_contract'][
        'measurement_validity_used_for_selection'] is False


def test_geometry_audit_rejects_routed_or_conditionally_computed_input(
        tmp_path):
    records = [
        _record('real_seq02_00001.jpg', 1, _box(), _box()),
        _record('sim_seq09_00001.jpg', 1, _box(), _box()),
    ]
    records[0]['dino_invoked'] = False
    payload, raw, split_root = _payload(tmp_path, records)
    with pytest.raises(RuntimeError, match='computed on every input frame'):
        audit.audit_payload(
            payload, raw, split_root, 'fixed-target',
            required_frame_count=2)

    records[0]['dino_invoked'] = True
    payload['protocol'] = 'lane_isolated_conditional_dino_v3'
    raw = json.dumps(payload).encode()
    with pytest.raises(RuntimeError, match='unrouted all-lane audit'):
        audit.audit_payload(
            payload, raw, split_root, 'fixed-target',
            required_frame_count=2)


def test_source_val_capacity_audit_has_no_runtime_or_target_claim(tmp_path):
    records = [
        _record('real_seq07_00001.jpg', 1, _box(), _box()),
        _record('sim_seq10_00001.jpg', 1, _box(), _box()),
    ]
    payload, raw, split_root = _payload(tmp_path, records, split='val')
    result = audit.audit_payload(
        payload, raw, split_root, 'source-val',
        required_frame_count=2)

    assert result['evidence_boundary'] == (
        'source_only_refiner_hypothesis_audit')
    assert result['decision'] == (
        'SOURCE_VAL_GEOMETRY_CAPACITY_AUDIT_COMPLETE')
    assert result['next_stage']['geometry_refiner_trained'] is False
    assert result['next_stage']['eligible_for_unknown_sequence_claim'] is False

    validity = _validity({
        'real_seq07': dict(default_valid=True, invalid_intervals=[]),
    })
    with pytest.raises(RuntimeError, match='fixed-target diagnostic only'):
        audit.audit_payload(
            payload, raw, split_root, 'source-val',
            required_frame_count=2,
            measurement_validity=validity)
