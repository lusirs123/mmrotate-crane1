import math
import numpy as np
import pytest

from crane_project.tools.analyze_unified_full_run_v1 import frozen_metrics
from crane_project.tools.eval_crane_offline import CraneOfflineEvaluator, compute_riou, angle_diff


def test_frozen_errors_match_geometry_evaluator_with_missing_and_gaps():
    original, frozen = [], []
    for domain in ('real', 'sim'):
        for i in list(range(1, 26)) + list(range(100, 114)):
            gt = np.array([0., 0., 20., 10., 0.])
            pred = None if i % 7 in (0, 1) else np.array([float(i % 17), 0., 21., 10., (i % 5)*.1])
            row = dict(domain=domain,seq_id='s',frame_id=i,pred_box=pred)
            original.append(dict(row,gt_box=gt))
            frozen.append(dict(row,server_iou=None if pred is None else compute_riou(pred,gt),
                server_errors=dict(center=None if pred is None else float(np.linalg.norm(pred[:2]-gt[:2])),
                    angle=None if pred is None else math.degrees(abs(float(angle_diff(pred[4:5],gt[4:5])[0]))))))
    assert frozen_metrics(list(reversed(frozen))) == CraneOfflineEvaluator().evaluate_records(original)


def test_frozen_metrics_reject_missing_error_and_duplicate_frame():
    row = dict(domain='real',seq_id='s',frame_id=1,pred_box=[0,0,20,10,0],
               server_iou=None,server_errors=dict(center=0,angle=0))
    with pytest.raises(ValueError,match='missing server metrics'):
        frozen_metrics([row])
    row['server_iou'] = 1.
    with pytest.raises(ValueError,match='Duplicate frame'):
        frozen_metrics([row,row])
