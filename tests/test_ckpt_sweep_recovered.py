"""Regression checks for the restored historical K1 selection rule."""

import pickle

import pytest

from crane_project.tools import ckpt_sweep


def _metrics(center, angle, aci=0.95, mcml=2):
    return {
        'real/TDR_w10(%)': 100.0,
        'sim/TDR_w10(%)': 100.0,
        'real/R_center(%)': center,
        'sim/R_center(%)': center,
        'sim/ACI': aci,
        'real/MCML_max(frames)': mcml,
        'sim/MCML_max(frames)': mcml,
        'sim/A-RMSE(deg)': angle,
    }


def test_historical_constraint_and_score_are_preserved():
    candidates = {
        'epoch_16': {'metrics': _metrics(100.0, 9.0)},
        'epoch_18': {'metrics': _metrics(90.0, 0.0, aci=1.0)},
        'epoch_20': {'metrics': _metrics(100.0, 0.0, mcml=6)},
        'epoch_22': {'metrics': _metrics(99.8, 3.0)},
        'epoch_24': {'metrics': _metrics(99.8, 2.0)},
    }
    selected, _, info = ckpt_sweep.select_best_checkpoint(
        candidates, dict(ckpt_sweep.SELECTION_CONFIG))
    assert selected == 'epoch_24'
    assert info['selection'] == 'constraint_pass'
    assert info['total'] == 5


def test_missing_metric_does_not_silently_become_zero():
    metrics = _metrics(100.0, 2.0)
    del metrics['sim/ACI']
    with pytest.raises(ValueError, match='sim/ACI'):
        ckpt_sweep.extract_metrics(metrics, ckpt_sweep.SELECTION_CONFIG)


def test_cached_prediction_count_must_match_annotations(tmp_path):
    pred = tmp_path / 'pred.pkl'
    with pred.open('wb') as stream:
        pickle.dump([], stream)
    with pytest.raises(ValueError, match='count mismatch'):
        ckpt_sweep.pkl_to_dota(str(pred), ['real_seq07_00001'], str(tmp_path))
