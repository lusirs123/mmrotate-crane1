"""Output-format adaptation only. Native detectors own decoding and postprocess."""
import math

from crane_project.utils.port_geometry_midpoint_edge_residual_metrics_v1 import canonical


def yolo_to_common(xywhr, scores, classes):
    """Original-pixel, radian YOLO results -> one-class native Kx6 results.

    Keep every delivered box. Canonicalization changes representation only.
    Sorting makes the old offline selector's first-box rule equal score Top1.
    """
    import numpy as np
    xywhr = np.asarray(xywhr)
    scores = np.asarray(scores).reshape(-1)
    classes = np.asarray(classes).reshape(-1)
    if xywhr.shape != (len(scores), 5) or len(classes) != len(scores):
        raise ValueError('YOLO output shape differs')
    if not np.all(classes == 0):
        raise ValueError('Unexpected non-grab output class')
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('Invalid confidence')
    values = [canonical(box) + [float(score)] for box, score in zip(xywhr, scores)]
    return common_result(np.asarray(values, dtype=np.float32).reshape(-1, 6))


def common_result(boxes):
    import numpy as np
    values = np.asarray(boxes, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError('Expected native Kx6 one-class OBB result')
    if not np.isfinite(values).all() or np.any(values[:, 2:4] <= 0):
        raise ValueError('Invalid native box geometry')
    if np.any((values[:, 5] < 0) | (values[:, 5] > 1)):
        raise ValueError('Invalid native confidence')
    order = np.argsort(-values[:, 5], kind='stable')
    return [values[order].copy()]
