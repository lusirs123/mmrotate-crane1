"""Regression checks for the restored historical K1 selection rule."""

import os
import pickle

import numpy as np
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


def test_val_subprocess_imports_this_checkout_before_installed_mmrotate(
        tmp_path, monkeypatch):
    monkeypatch.setenv('PYTHONPATH', '/other/python/packages')
    monkeypatch.setattr(ckpt_sweep, 'check_or_record_prediction',
                        lambda *args, **kwargs: args[2])
    seen = {}

    def fake_run(cmd, **kwargs):
        seen.update(cmd=cmd, kwargs=kwargs)
        return type('Result', (), {'returncode': 0})()

    monkeypatch.setattr(ckpt_sweep.subprocess, 'run', fake_run)
    ckpt_sweep.run_test_on_val(
        'student.py', 'epoch_1.pth', str(tmp_path), 'epoch_1', gpu=0)
    env = seen['kwargs']['env']
    assert env['PYTHONPATH'].split(os.pathsep)[0] == ckpt_sweep.PROJ_ROOT
    assert env['CUDA_VISIBLE_DEVICES'] == '0'
    assert seen['kwargs']['cwd'] == ckpt_sweep.PROJ_ROOT
    assert seen['cmd'][1] == os.path.join(ckpt_sweep.PROJ_ROOT,
                                         'tools/test.py')


def test_cached_prediction_requires_generation_provenance(tmp_path):
    config = tmp_path / 'student.py'
    checkpoint = tmp_path / 'epoch_1.pth'
    pkl = tmp_path / 'results.pkl'
    ann = tmp_path / 'annfiles'
    ann.mkdir()
    (ann / 'real_seq01_00001.txt').write_text('GT')
    config.write_text('config')
    checkpoint.write_bytes(b'weights')
    pkl.write_bytes(b'prediction')
    with pytest.raises(RuntimeError, match='lacks generation provenance'):
        ckpt_sweep.check_or_record_prediction(
            str(config), str(checkpoint), str(pkl), str(ann), 'fixed_test')
    ckpt_sweep.check_or_record_prediction(
        str(config), str(checkpoint), str(pkl), str(ann), 'fixed_test',
        generated=True)
    assert ckpt_sweep.check_or_record_prediction(
        str(config), str(checkpoint), str(pkl), str(ann), 'fixed_test') == str(pkl)
    pkl.write_bytes(b'different prediction')
    with pytest.raises(RuntimeError, match='provenance mismatch'):
        ckpt_sweep.check_or_record_prediction(
            str(config), str(checkpoint), str(pkl), str(ann), 'fixed_test')


def test_cached_dota_export_rejects_changed_prediction_pkl(tmp_path):
    pkl = tmp_path / 'results.pkl'
    with pkl.open('wb') as stream:
        pickle.dump([[np.array([[10., 10., 4., 2., 0., .8]])]], stream)
    ckpt_sweep.pkl_to_dota(str(pkl), ['real_seq01_00001'], str(tmp_path))
    ckpt_sweep.pkl_to_dota(str(pkl), ['real_seq01_00001'], str(tmp_path))
    with pkl.open('wb') as stream:
        pickle.dump([[np.array([[11., 10., 4., 2., 0., .8]])]], stream)
    with pytest.raises(RuntimeError, match='export provenance mismatch'):
        ckpt_sweep.pkl_to_dota(str(pkl), ['real_seq01_00001'], str(tmp_path))
