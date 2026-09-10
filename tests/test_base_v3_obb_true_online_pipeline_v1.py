import json
from pathlib import Path

import numpy as np
import pytest

from crane_project.tools import base_v3_obb_true_online_pipeline_v1 as online


def _identity(frame, sequence='seq02'):
    return dict(
        frame_key='real_{}_{:05d}'.format(sequence, frame),
        domain='real', sequence=sequence, frame=frame)


def test_history_is_strictly_previous_and_resets_on_gap():
    buffer = online.OnlineDinoHistoryBuffer(horizon=4)
    image = np.full((3, 4, 3), 7, dtype=np.uint8)
    first = buffer.prepare(_identity(1), image)
    assert first['causal_history_valid_mask'].tolist() == [
        False, False, False, False]
    buffer.commit(_identity(1), image, [1, 2, 3, 4, 0.1])
    second = buffer.prepare(_identity(2), image)
    assert second['causal_history_valid_mask'].tolist() == [
        True, False, False, False]
    assert second['causal_history_frame_keys'][0] == 'real_seq02_00001'
    assert np.allclose(
        second['causal_history_proposals_raw'][0],
        [[1, 2, 3, 4, 0.1]])
    gap = buffer.prepare(_identity(4), image)
    assert gap['causal_history_valid_mask'].tolist() == [
        False, False, False, False]


def test_missing_dino_history_is_invalid_not_extrapolated():
    buffer = online.OnlineDinoHistoryBuffer(horizon=2)
    image = np.ones((2, 2, 3), dtype=np.uint8)
    buffer.prepare(_identity(1), image)
    buffer.commit(_identity(1), image, None)
    second = buffer.prepare(_identity(2), image)
    assert second['causal_history_valid_mask'].tolist() == [False, False]
    assert second['causal_history_proposals_raw'][0].tolist() == [
        [0.0, 0.0, 1.0, 1.0, 0.0]]


def test_assemble_detector_row_uses_formal_anchor_score():
    audit = dict(
        coordinate_space='original_image', anchor_source='k1',
        k1_box=[1, 2, 3, 4, 0.1], k1_score=0.8,
        dino_box=[1, 2, 3, 4, 0.1],
        base_v3_box=[1, 2, 3, 4, 0.1])
    dino = [1, 2, 3, 4, 0.1, 0.3]
    row = online.assemble_detector_row(
        _identity(1), (10, 20, 3), dino, audit)
    assert row['anchor_source'] == 'k1'
    assert row['anchor_score'] == 0.8
    assert row['image_size'] == [20, 10]
    audit['anchor_source'] = 'dino_fallback'
    audit['k1_box'] = None
    audit['k1_score'] = None
    row = online.assemble_detector_row(
        _identity(1), (10, 20, 3), dino, audit)
    assert row['anchor_score'] == 0.3


def test_dino_coordinate_mismatch_fails_closed():
    audit = dict(
        coordinate_space='original_image', anchor_source='dino_fallback',
        k1_box=None, k1_score=None, dino_box=[9, 2, 3, 4, 0.1],
        base_v3_box=[1, 2, 3, 4, 0.1])
    with pytest.raises(RuntimeError, match='coordinate mapping'):
        online.assemble_detector_row(
            _identity(1), (10, 20, 3),
            [1, 2, 3, 4, 0.1, 0.3], audit)


def test_contract_rejects_cache_or_future():
    path = Path(
        'crane_project/data_contracts/'
        'base_v3_obb_true_online_pipeline_v1.json')
    contract = json.loads(path.read_text())
    online.validate_contract(contract)
    changed = dict(contract)
    changed['cached_dino_predictions_used'] = True
    with pytest.raises(ValueError, match='cached_dino_predictions_used'):
        online.validate_contract(changed)
    changed = dict(contract)
    changed['future_frames_used'] = True
    with pytest.raises(ValueError, match='future_frames_used'):
        online.validate_contract(changed)


def test_online_config_contains_no_cached_audit_loader():
    text = Path(
        'crane_project/configs/'
        'crane_symeood_dino_k1_retentive_causal_phase_online_v1.py'
    ).read_text()
    assert 'LoadDinoProposalFromAudit' not in text
    assert 'LoadCausalHistoryFromAudit' not in text
    assert "cached_dino_loader=False" in text
    assert "cached_history_loader=False" in text


def test_top1_detection_selects_score_and_validates_shape():
    result = [np.asarray([
        [1, 2, 3, 4, 0.1, 0.2],
        [5, 6, 7, 8, 0.2, 0.9]], dtype=np.float32)]
    selected = online.top1_detection(result)
    assert selected[0] == 5.0
    assert selected[5] == pytest.approx(0.9)


def test_paper_report_comparison_is_read_only_and_detects_state_drift():
    component = dict(
        value=[1.0, 2.0], valid=True, state='measurement',
        source='base_v3_measurement')
    observations = {
        method: {name: dict(component) for name in ('center', 'scale', 'angle')}
        for method in online.METHODS}
    record = dict(
        frame_key='real_seq02_00001',
        detector=dict(raw_obb=[1, 2, 3, 4, 0.1]),
        observations=observations)
    report = {'protocol': online.PROTOCOL.replace('true_online_pipeline', 'paper_final_report'), 'records': [dict(record)]}
    audit = online.compare_paper_report([record], report)
    assert audit['passed'] is True
    assert audit['reference_used_in_online_decisions'] is False
    changed = {'protocol': online.PROTOCOL.replace('true_online_pipeline', 'paper_final_report'), 'records': [dict(record)]}
    changed['records'][0] = dict(record)
    changed['records'][0]['observations'] = {
        method: {name: dict(value) for name, value in values.items()}
        for method, values in observations.items()}
    changed['records'][0]['observations']['v51_hybrid']['angle'][
        'state'] = 'unavailable'
    assert online.compare_paper_report([record], changed)[
        'component_state_mismatch_count'] == 1
