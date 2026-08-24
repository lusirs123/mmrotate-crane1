import numpy as np
import pytest

from crane_project.tools import (
    symeood_dino_lane_isolated_conditional_calibrate as calibrate)
from crane_project.utils.lane_isolated_conditional_dino import (
    LaneIsolatedConditionalDinoSelector, normalized_diagonal)


def _box(score=0.8, width=20.0, height=10.0, angle=0.0):
    return np.asarray(
        [10.0, 10.0, width, height, angle, score], dtype=np.float32)


def _selector(small=0.0, sym_diag=0.2, dino_diag=0.2):
    return LaneIsolatedConditionalDinoSelector(
        small_diag_ratio=small,
        max_sym_diag_change=sym_diag,
        max_sym_angle_change_deg=15.0,
        max_dino_diag_change=dino_diag,
        max_dino_angle_change_deg=15.0)


def test_normalized_diagonal_uses_image_diagonal():
    value = normalized_diagonal(_box(width=30.0, height=40.0), (300, 400))
    assert value == pytest.approx(0.1)


def test_missing_or_small_symeood_triggers_dino():
    missing = _selector().select(None, _box(), (100, 100), 'real_seq02', 1)
    assert missing['invoke_dino'] is True
    assert missing['selected_source'] == 'dino_native'
    small = _selector(small=0.2).select(
        _box(width=5.0, height=5.0), _box(),
        (100, 100), 'real_seq02', 1)
    assert small['invoke_dino'] is True
    assert 'sym_eood_small_geometry' in small['trigger_reasons']


def test_untriggered_frame_ignores_precollected_dino_box():
    sym = _box(score=0.7, width=40.0)
    result = _selector().select(
        sym, _box(score=0.99), (100, 100), 'sim_seq09', 1)
    assert result['invoke_dino'] is False
    assert result['selected_source'] == 'sym_eood'
    assert np.array_equal(result['selected'], sym)


def test_first_dino_rescue_is_not_compared_with_previous_symeood_lane():
    selector = _selector(sym_diag=0.1, dino_diag=0.1)
    first = selector.select(
        _box(width=100.0), _box(width=10.0),
        (200, 200), 'real_seq03', 1)
    second = selector.select(
        _box(width=50.0), _box(width=10.0),
        (200, 200), 'real_seq03', 2)
    assert first['selected_source'] == 'sym_eood'
    assert second['invoke_dino'] is True
    assert second['selected_source'] == 'dino_native'
    assert second['dino_geometry_stable'] is True
    assert second['measurement_valid'] is True


def test_dino_self_discontinuity_marks_risk_without_switching_lane():
    selector = _selector(small=1.0, dino_diag=0.1)
    selector.select(
        _box(width=5.0), _box(width=20.0),
        (100, 100), 'real_seq03', 1)
    result = selector.select(
        _box(width=5.0), _box(width=50.0),
        (100, 100), 'real_seq03', 2)
    assert result['selected_source'] == 'dino_native'
    assert result['dino_geometry_stable'] is False
    assert result['measurement_valid'] is False
    assert result['risk_reasons'] == ['dino_self_geometry_discontinuity']


def test_two_phase_api_requires_one_finish_per_frame():
    selector = _selector()
    selector.begin_frame(_box(), (100, 100), 'sim_seq09', 1)
    with pytest.raises(RuntimeError, match='finish_frame'):
        selector.begin_frame(_box(), (100, 100), 'sim_seq09', 2)
    selector.finish_frame(None)
    with pytest.raises(RuntimeError, match='begin_frame'):
        selector.finish_frame(None)


def _summary(hit_keys, invocation_count, **metrics):
    return dict(
        hit_frame_keys=list(hit_keys),
        dino_invocation_count=invocation_count,
        metrics=metrics)


def test_source_gate_requires_exact_retention_of_both_lanes():
    metrics = {
        'sim/A-RMSE(deg)': 1.0,
        'sim/mean_RIoU': 0.9,
        'real/DFR(%/frame)': 2.0,
        'sim/DFR(%/frame)': 2.0,
        'real/MCML_max(frames)': 1,
        'sim/MCML_max(frames)': 0,
    }
    sym = _summary({'a', 'b'}, 0, **metrics)
    dino = _summary({'a', 'c'}, 3, **metrics)
    candidate = _summary({'a', 'b', 'c'}, 2, **metrics)
    passed = calibrate._source_gate(candidate, sym, dino, 4)
    assert passed['passed'] is True
    lost = _summary({'a', 'b'}, 2, **metrics)
    failed = calibrate._source_gate(lost, sym, dino, 4)
    assert failed['passed'] is False
    assert failed['lost_vs_native_dino_frame_keys'] == ['c']


def test_source_record_validation_rejects_routed_audit(tmp_path):
    image_dir = tmp_path / 'images'
    image_dir.mkdir()
    (image_dir / 'real_seq01_00001.jpg').write_bytes(b'x')
    record = dict(
        filename='/old/root/real_seq01_00001.jpg',
        sequence='real_seq01', frame=1,
        sym_eood_original_box=_box().tolist())
    payload = dict(
        protocol='lane_isolated_conditional_dino_v3',
        frame_count=1,
        metadata=dict(
            fusion_policy='sym_eood_proposal_dino_roi_union'))
    with pytest.raises(RuntimeError, match='all-lane source collection'):
        calibrate._validate_source_records(payload, [record], image_dir, 1)
