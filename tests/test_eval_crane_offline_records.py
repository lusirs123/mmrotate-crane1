import numpy as np
import pytest

from crane_project.tools.eval_crane_offline import (
    CraneOfflineEvaluator, compute_riou)


def _box(cx, width=10.0, height=20.0, angle=0.0):
    return np.asarray([cx, 0.0, width, height, angle], dtype=np.float64)


def test_direct_record_evaluation_matches_ordered_stream_semantics():
    evaluator = CraneOfflineEvaluator(
        mode='test', center_thresh_px=15.0, ekf_window=2,
        mcml_limit=1, iou_thresh=0.5)
    metrics = evaluator.evaluate_records([
        dict(domain='real', seq_id='seq02', frame_id=1,
             pred_box=_box(0.0), gt_box=_box(0.0), score=0.9),
        dict(domain='real', seq_id='seq02', frame_id=2,
             pred_box=None, gt_box=_box(0.0), score=0.0),
        dict(domain='real', seq_id='seq02', frame_id=3,
             pred_box=_box(1.0), gt_box=_box(0.0), score=0.8),
    ])
    assert metrics['real/R_center(%)'] == pytest.approx(66.67, abs=0.01)
    assert metrics['real/MCML_max(frames)'] == 1
    assert metrics['real/MCML_pass(limit=1)'] == 1
    assert metrics['real/TDR_w2(%)'] == 100.0


def test_rotated_iou_respects_angle_and_containment():
    horizontal = _box(0.0, width=100.0, height=10.0, angle=0.0)
    vertical = _box(0.0, width=100.0, height=10.0, angle=np.pi / 2.0)
    assert compute_riou(horizontal, vertical) == pytest.approx(
        100.0 / 1900.0, rel=1e-5)

    outer = _box(0.0, width=100.0, height=100.0)
    inner = _box(0.0, width=10.0, height=10.0)
    assert compute_riou(outer, inner) == pytest.approx(0.01, rel=1e-5)


def test_temporal_metrics_do_not_bridge_frame_gaps():
    evaluator = CraneOfflineEvaluator(
        mode='test', center_thresh_px=15.0, ekf_window=2,
        mcml_limit=1, iou_thresh=0.5)
    metrics = evaluator.evaluate_records([
        dict(domain='real', seq_id='seq02', frame_id=1,
             pred_box=_box(0.0), gt_box=_box(0.0)),
        dict(domain='real', seq_id='seq02', frame_id=2,
             pred_box=None, gt_box=_box(0.0)),
        dict(domain='real', seq_id='seq02', frame_id=10,
             pred_box=None, gt_box=_box(0.0)),
        dict(domain='real', seq_id='seq02', frame_id=11,
             pred_box=_box(0.0), gt_box=_box(0.0)),
    ])
    assert metrics['real/MCML_max(frames)'] == 1
    assert metrics['real/MCML_mean(frames)'] == 1.0
    assert metrics['real/TDR_w2(%)'] == 100.0


def test_mrf_counts_the_full_recovered_miss_run():
    evaluator = CraneOfflineEvaluator(
        mode='test', center_thresh_px=15.0, ekf_window=2,
        mcml_limit=2, iou_thresh=0.5)
    metrics = evaluator.evaluate_records([
        dict(domain='real', seq_id='seq03', frame_id=1,
             pred_box=_box(0.0), gt_box=_box(0.0)),
        dict(domain='real', seq_id='seq03', frame_id=2,
             pred_box=None, gt_box=_box(0.0)),
        dict(domain='real', seq_id='seq03', frame_id=3,
             pred_box=None, gt_box=_box(0.0)),
        dict(domain='real', seq_id='seq03', frame_id=4,
             pred_box=_box(0.0), gt_box=_box(0.0)),
    ])
    assert metrics['real/MRF(frames)'] == 2.0


def test_sim_angle_rmse_penalizes_center_miss():
    evaluator = CraneOfflineEvaluator(
        mode='test', center_thresh_px=15.0,
        sim_angle_center_thresh_px=10.0, ekf_window=1)
    metrics = evaluator.evaluate_records([
        dict(domain='sim', seq_id='seq09', frame_id=1,
             pred_box=_box(20.0), gt_box=_box(0.0)),
    ])
    assert metrics['sim/A-RMSE(deg)'] == pytest.approx(90.0)
    assert metrics['sim/R_center(%)'] == 0.0
