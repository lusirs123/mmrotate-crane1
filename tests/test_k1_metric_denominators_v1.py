import math

import numpy as np

from crane_project.tools.audit_k1_metric_denominators_v1 import (
    historical_approx_riou, markdown)
from crane_project.tools.eval_crane_offline import compute_riou


def test_historical_overlap_ignores_angle_but_exact_overlap_does_not():
    first = np.array([50, 50, 40, 10, 0], dtype=float)
    rotated = np.array([50, 50, 40, 10, math.pi / 4], dtype=float)
    assert abs(historical_approx_riou(first, rotated) - 1) < 1e-8
    assert compute_riou(first, rotated) < 0.6


def test_report_labels_conditional_and_all_frame_denominators():
    row = dict(gt_positive_frame_count=10, output_frame_count=5,
               output_coverage=0.5, center_hit_given_output=0.8,
               center_hit_all_gt_frames=0.4,
               historical_approx_riou_given_output=0.9,
               exact_riou_given_output=0.7,
               exact_riou_all_gt_frames=0.35)
    report = dict(by_domain={'real': row},
                  min_pkl_to_dota_geometry_iou=0.999,
                  sources=dict(
        pred_pkl_sha256='a' * 64, gt_annotations_sha256='b' * 64))
    output = markdown(report)
    assert '中心命中/有输出' in output
    assert '中心命中/全部' in output
    assert '旋转 IoU/全部' in output
