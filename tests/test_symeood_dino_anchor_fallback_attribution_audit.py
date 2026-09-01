import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from crane_project.tools import (
    symeood_dino_anchor_fallback_attribution_audit as audit)


def _box(cx=10.0, cy=10.0, width=8.0, height=4.0,
         angle=0.0, score=0.9):
    return np.asarray(
        [cx, cy, width, height, angle, score], dtype=np.float64)


def _annotation(path, box):
    cx, cy, width, height = box[:4]
    values = [
        cx - width / 2.0, cy - height / 2.0,
        cx + width / 2.0, cy - height / 2.0,
        cx + width / 2.0, cy + height / 2.0,
        cx - width / 2.0, cy + height / 2.0]
    path.write_text(
        ' '.join(str(value) for value in values) + ' grab 0\n')


def _record(filename, sym, dino):
    domain, seq_id, frame = audit.parse_seq_frame(filename)
    return dict(
        filename='/old/root/' + filename,
        sequence='{}_{}'.format(domain, seq_id), frame=frame,
        sym_eood_box=None if sym is None else sym.tolist(),
        dino_native_box=None if dino is None else dino.tolist(),
        dino_invoked=True)


def _result(box):
    if box is None:
        return [np.zeros((0, 6), dtype=np.float32)]
    return [np.asarray(box, dtype=np.float32).reshape(1, 6)]


def _fixture(tmp_path, rows):
    split = tmp_path / 'test'
    ann = split / 'annfiles'
    ann.mkdir(parents=True)
    records = []
    streams = {name: [] for name in ('k1_reference', *audit.MODE_NAMES)}
    for row in rows:
        filename = row['filename']
        _annotation(ann / (Path(filename).stem + '.txt'), row['gt'])
        records.append(_record(filename, row['k1'], row['dino']))
        streams['k1_reference'].append(row['k1'])
        streams['k1_identity'].append(
            row['k1'] if row['k1'] is not None else row['dino'])
        streams['center_only'].append(row.get(
            'center_only', streams['k1_identity'][-1]))
        streams['full'].append(row.get(
            'full', streams['center_only'][-1]))
        streams['current_only'].append(row.get(
            'current_only', streams['full'][-1]))
    payload = dict(protocol=base_protocol(), records=records)
    raw = json.dumps(payload).encode()
    result_payloads = {}
    for name, boxes in streams.items():
        serialized = pickle.dumps([_result(box) for box in boxes])
        result_payloads[name] = (
            str(tmp_path / (name + '.pkl')), serialized, boxes)
    return payload, raw, split, result_payloads


def base_protocol():
    return 'source_owned_geometry_union_v2'


def test_transition_audit_attributes_k1_to_dino_geometry_jump(tmp_path):
    rows = [
        dict(filename='real_seq02_00001.jpg', gt=_box(),
             k1=_box(), dino=_box()),
        dict(filename='real_seq02_00002.jpg', gt=_box(width=16.0),
             k1=None, dino=_box(width=16.0)),
        dict(filename='sim_seq09_00001.jpg', gt=_box(),
             k1=_box(), dino=_box()),
    ]
    payload, raw, split, streams = _fixture(tmp_path, rows)
    result = audit.audit_payload(
        payload, raw, split, 'fixed-target', streams,
        required_frame_count=3)

    transition = result['identity_transition_attribution']['real'][
        'k1_anchor->dino_fallback']
    assert transition['transition_count'] == 1
    assert transition['mean_dfr_fraction'] > 0.0
    assert transition['share_of_identity_dfr_sum'] == pytest.approx(1.0)
    assert result['component_mode_contract']['passed'] is True
    assert result['audit_contract']['detector_forward'] is False


def test_present_wrong_anchor_lock_and_gt_oracle_are_separated(tmp_path):
    rows = [
        dict(filename='real_seq02_00001.jpg', gt=_box(),
             k1=_box(), dino=_box()),
        dict(filename='real_seq02_00002.jpg', gt=_box(),
             k1=_box(cx=100.0), dino=_box()),
        dict(filename='real_seq02_00003.jpg', gt=_box(),
             k1=_box(cx=100.0), dino=_box()),
        dict(filename='sim_seq09_00001.jpg', gt=_box(),
             k1=_box(), dino=_box()),
    ]
    payload, raw, split, streams = _fixture(tmp_path, rows)
    result = audit.audit_payload(
        payload, raw, split, 'fixed-target', streams,
        required_frame_count=4)

    support = result['support_counts']['real']
    assert support['k1_present_wrong_dino_hit'] == 2
    failure = result['failure_attribution']['k1_identity']
    assert failure['max_run_length'] == 2
    longest = failure['longest_runs'][0]
    assert longest['cause_counts'] == {
        'k1_present_wrong_anchor_lock_dino_hit': 2}
    oracle = result['diagnostic_capacity'][
        'present_wrong_dino_rescue_oracle']
    assert oracle['non_deployable_gt_oracle'] is True
    assert oracle['rescued_frame_count'] == 2
    assert oracle['metrics']['real/MCML_max(frames)'] == 0
    assert result['next_stage']['eligible_for_runtime_policy'] is False


def test_component_contract_rejects_non_identity_k1_mode(tmp_path):
    rows = [
        dict(filename='real_seq02_00001.jpg', gt=_box(),
             k1=_box(), dino=_box()),
        dict(filename='sim_seq09_00001.jpg', gt=_box(),
             k1=_box(), dino=_box()),
    ]
    payload, raw, split, streams = _fixture(tmp_path, rows)
    streams['k1_identity'][2][0] = _box(cx=30.0)
    with pytest.raises(RuntimeError, match='Component-mode contract failed'):
        audit.audit_payload(
            payload, raw, split, 'fixed-target', streams,
            required_frame_count=2)


def test_causal_geometry_fallback_retains_history_across_missing_run(
        tmp_path):
    rows = [
        dict(filename='real_seq02_00001.jpg', gt=_box(),
             k1=_box(), dino=_box()),
        dict(filename='real_seq02_00002.jpg', gt=_box(),
             k1=None, dino=_box(width=16.0)),
        dict(filename='real_seq02_00003.jpg', gt=_box(),
             k1=None, dino=_box(width=16.0)),
        dict(filename='sim_seq09_00001.jpg', gt=_box(),
             k1=_box(), dino=_box()),
    ]
    payload, raw, split, streams = _fixture(tmp_path, rows)
    result = audit.audit_payload(
        payload, raw, split, 'fixed-target', streams,
        required_frame_count=4)

    fallback = result['diagnostic_capacity'][
        'causal_geometry_preserving_fallback_h4']
    assert fallback['dino_fallback_frame_count'] == 2
    assert fallback['recent_k1_geometry_used_count'] == 2
    assert fallback['native_dino_geometry_used_count'] == 0
    assert fallback['metrics']['real/DFR(%/frame)'] == pytest.approx(0.0)


def test_measurement_validity_remains_secondary(tmp_path):
    rows = [
        dict(filename='real_seq02_00001.jpg', gt=_box(),
             k1=_box(), dino=_box()),
        dict(filename='real_seq02_00002.jpg', gt=_box(),
             k1=_box(cx=100.0), dino=_box()),
        dict(filename='sim_seq09_00001.jpg', gt=_box(),
             k1=_box(), dino=_box()),
    ]
    payload, raw, split, streams = _fixture(tmp_path, rows)
    manifest = dict(
        protocol='crane_measurement_validity_v1',
        status='POST_HOC_OPERATIONAL_VALIDITY_DIAGNOSTIC',
        selection_basis='manual_video_operation_phase_review',
        sequences=dict(
            real_seq02=dict(
                default_valid=True,
                invalid_intervals=[dict(
                    start_frame=2, end_frame=2,
                    reason='material_contact')])) )
    manifest_raw = json.dumps(manifest).encode()
    result = audit.audit_payload(
        payload, raw, split, 'fixed-target', streams,
        required_frame_count=3,
        measurement_validity=manifest,
        measurement_validity_bytes=manifest_raw)

    measurement = result['measurement_validity']
    assert measurement['excluded_real_frame_count'] == 1
    assert measurement['eligible_for_primary_decision_override'] is False
    assert result['input']['frame_count'] == 3
