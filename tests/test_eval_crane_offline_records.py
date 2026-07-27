import numpy as np

from crane_project.tools.eval_crane_offline import CraneOfflineEvaluator


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
    # R_center remains the historical conditional metric over output frames;
    # silence affects MCML/TDR instead.
    assert metrics['real/R_center(%)'] == 100.0
    assert metrics['real/MCML_max(frames)'] == 1
    assert metrics['real/MCML_pass(limit=1)'] == 1
    assert metrics['real/TDR_w2(%)'] == 100.0
