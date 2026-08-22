import numpy as np
import pytest

from crane_project.utils.conservative_takeover import (
    ConservativeTakeoverSelector, geometry_change)


def _box(score, width=20.0, angle=0.0):
    return np.asarray([10.0, 10.0, width, 10.0, angle, score], np.float32)


def _selector(confirmations=2):
    return ConservativeTakeoverSelector(
        enter_margin=0.10,
        exit_margin=0.02,
        min_confirmations=confirmations,
        max_diag_change=0.20,
        max_angle_change_deg=15.0)


def test_takeover_requires_confirmation_then_uses_hysteresis():
    selector = _selector(confirmations=2)
    first = selector.select(_box(0.60), _box(0.80), 'real_seq02', 1)
    second = selector.select(_box(0.60), _box(0.80), 'real_seq02', 2)
    hold = selector.select(_box(0.70), _box(0.73), 'real_seq02', 3)
    exit_result = selector.select(
        _box(0.75), _box(0.74), 'real_seq02', 4)
    assert first['selected_source'] == 'sym_eood'
    assert first['takeover_reason'] == 'awaiting_dino_confirmation'
    assert second['selected_source'] == 'dino_native'
    assert second['takeover_reason'] == 'confirmed_score_takeover'
    assert hold['selected_source'] == 'dino_native'
    assert hold['takeover_reason'] == 'dino_hysteresis_hold'
    assert exit_result['selected_source'] == 'sym_eood'


def test_geometry_gate_rejects_unstable_dino_takeover():
    selector = _selector(confirmations=1)
    selector.select(_box(0.80), _box(0.70), 'sim_seq09', 1)
    result = selector.select(
        _box(0.60), _box(0.95, width=40.0), 'sim_seq09', 2)
    assert result['selected_source'] == 'sym_eood'
    assert result['geometry_allowed'] is False
    assert result['takeover_reason'] == 'dino_geometry_rejected'


def test_sequence_gap_resets_lane_state():
    selector = _selector(confirmations=1)
    selected = selector.select(
        _box(0.60), _box(0.90), 'real_seq02', 1)
    after_gap = selector.select(
        _box(0.60), _box(0.90), 'real_seq02', 3)
    assert selected['selected_source'] == 'dino_native'
    assert after_gap['previous_source'] == 'sym_eood'


def test_geometry_change_is_pi_periodic():
    first = _box(0.8, angle=0.01)
    equivalent = _box(0.8, angle=np.pi + 0.01)
    change = geometry_change(first, equivalent)
    assert change['diag_change'] == pytest.approx(0.0)
    assert change['angle_change_deg'] == pytest.approx(0.0, abs=1e-4)
