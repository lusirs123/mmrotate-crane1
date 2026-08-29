import json
from pathlib import Path

import pytest

from crane_project.tools import (
    symeood_dino_causal_history_support_audit as audit)


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


def _record(sequence, frame, sym, dino):
    filename = '{}_{:05d}.jpg'.format(sequence, frame)
    return dict(
        filename='/old/root/' + filename,
        sequence=sequence,
        frame=frame,
        sym_eood_box=sym,
        dino_native_box=dino,
        dino_invoked=True)


def _payload(tmp_path, records, centers=None, split='val'):
    annotations = tmp_path / split / 'annfiles'
    annotations.mkdir(parents=True)
    centers = centers or {}
    for record in records:
        center = centers.get(
            (record['sequence'], int(record['frame'])), (10.0, 10.0))
        _annotation(
            annotations / (Path(record['filename']).stem + '.txt'),
            cx=center[0], cy=center[1])
    payload = dict(protocol=audit.INPUT_PROTOCOL, records=records)
    raw = json.dumps(payload).encode()
    return payload, raw, tmp_path / split


def test_history_audit_reports_good_and_all_bad_history(tmp_path):
    records = [
        _record('real_seq07', 1, _box(), _box()),
        _record('real_seq07', 2, _box(), _box()),
        _record('real_seq07', 3, _box(cx=100.0), _box(cx=100.0)),
        _record('real_seq07', 4, _box(cx=100.0), _box(cx=100.0)),
        _record('real_seq07', 5, _box(cx=100.0), _box(cx=100.0)),
        _record('sim_seq10', 1, _box(), _box()),
    ]
    payload, raw, split_root = _payload(tmp_path, records)
    result = audit.audit_payload(
        payload, raw, split_root, 'source-val', required_frame_count=6)

    real = result['streams']['dino_native']['real']
    horizon_two = real['history_horizons']['2']
    assert horizon_two['current_miss_count'] == 3
    assert horizon_two['history_own_hit_support_count'] == 2
    assert horizon_two['full_window_all_own_bad_count'] == 1
    assert horizon_two['current_and_history_all_own_bad_runs'][
        'max_run_length'] == 1
    assert result['audit_contract']['eligible_for_runtime_policy'] is False
    assert result['next_stage']['temporal_model_training_authorized'] is False


def test_history_never_crosses_sequence_or_frame_gap(tmp_path):
    records = [
        _record('real_seq07', 1, _box(), _box()),
        _record('real_seq07', 3, _box(cx=100.0), _box(cx=100.0)),
        _record('sim_seq10', 1, _box(cx=100.0), _box(cx=100.0)),
    ]
    payload, raw, split_root = _payload(tmp_path, records)
    result = audit.audit_payload(
        payload, raw, split_root, 'source-val', required_frame_count=3)

    real = result['streams']['dino_native']['real']
    assert real['history_horizons']['1'][
        'history_own_hit_support_count'] == 0
    diagnostic = real['miss_diagnostics'][0]
    assert diagnostic['available_history_count'] == 0
    assert real['last_previous_hit_age']['no_previous_hit_count'] == 1


def test_constant_velocity_support_uses_only_two_previous_frames(tmp_path):
    records = [
        _record('real_seq07', 1, _box(cx=0.0), _box(cx=0.0)),
        _record('real_seq07', 2, _box(cx=10.0), _box(cx=10.0)),
        _record('real_seq07', 3, _box(cx=100.0), _box(cx=100.0)),
        _record('sim_seq10', 1, _box(), _box()),
    ]
    centers = {
        ('real_seq07', 1): (0.0, 10.0),
        ('real_seq07', 2): (10.0, 10.0),
        ('real_seq07', 3): (20.0, 10.0),
    }
    payload, raw, split_root = _payload(
        tmp_path, records, centers=centers)
    result = audit.audit_payload(
        payload, raw, split_root, 'source-val', required_frame_count=4)

    velocity = result['streams']['dino_native']['real'][
        'constant_velocity']
    assert velocity['current_miss_count'] == 1
    assert velocity['available_miss_count'] == 1
    assert velocity['support_count'] == 1
    assert velocity['support_rate'] == pytest.approx(1.0)


def test_source_and_fixed_target_evidence_boundaries_are_explicit(tmp_path):
    source_records = [
        _record('real_seq07', 1, _box(), _box()),
        _record('sim_seq10', 1, _box(), _box()),
    ]
    payload, raw, split_root = _payload(tmp_path, source_records)
    source = audit.audit_payload(
        payload, raw, split_root, 'source-val', required_frame_count=2)
    assert source['decision'] == (
        'SOURCE_VAL_CAUSAL_HISTORY_SUPPORT_AUDIT_COMPLETE')
    assert source['next_stage']['eligible_for_unknown_sequence_claim'] is False

    target_records = [
        _record('real_seq02', 1, _box(), _box()),
        _record('sim_seq09', 1, _box(), _box()),
    ]
    payload, raw, split_root = _payload(
        tmp_path, target_records, split='test')
    target = audit.audit_payload(
        payload, raw, split_root, 'fixed-target', required_frame_count=2)
    assert target['decision'] == (
        'FIXED_TARGET_CAUSAL_HISTORY_DIAGNOSTIC_ONLY')
    assert target['evidence_boundary'] == (
        'fixed_target_posthoc_temporal_diagnostic_not_model_selection')


def test_rejects_conditionally_computed_dino_input(tmp_path):
    records = [
        _record('real_seq07', 1, _box(), _box()),
        _record('sim_seq10', 1, _box(), _box()),
    ]
    records[0]['dino_invoked'] = False
    payload, raw, split_root = _payload(tmp_path, records)
    with pytest.raises(RuntimeError, match='computed on every input frame'):
        audit.audit_payload(
            payload, raw, split_root, 'source-val',
            required_frame_count=2)
