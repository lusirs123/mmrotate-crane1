"""Geometry-level tests for frozen detector-OBB depth evaluation."""

import importlib.util
import math
import pathlib
import sys

import numpy as np
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEPTH_TOOLS = ROOT / "tools/depth"
sys.path.insert(0, str(DEPTH_TOOLS))
try:
    spec = importlib.util.spec_from_file_location(
        "evaluate_detector_obb_depth",
        DEPTH_TOOLS / "evaluate_detector_obb_depth.py")
    MODULE = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(MODULE)
finally:
    sys.path.pop(0)


def test_obb_round_trip_preserves_long_short_and_periodic_angle():
    box = MODULE.normalize_obb(300, 400, 120, 40, math.radians(27))
    recovered = MODULE.corners_to_obb(MODULE.obb_corners(box))
    assert recovered[:4] == pytest.approx(box[:4], abs=1e-4)
    assert MODULE.angle_error_deg(recovered[4], box[4]) < 1e-4


def test_identity_homography_keeps_obb_geometry():
    box = MODULE.normalize_obb(512, 550, 130, 45, math.radians(-7))
    transformed = MODULE.transform_obb(box, np.eye(3))
    assert transformed[:4] == pytest.approx(box[:4], abs=1e-4)
    assert MODULE.angle_error_deg(transformed[4], box[4]) < 1e-4
    assert MODULE.rotated_iou(box, transformed) == pytest.approx(1.0, abs=1e-5)


def test_six_percent_short_edge_shrink_increases_pinhole_depth():
    parameters = {"c": 1.0, "beta": 0.0, "b_m": 0.0}
    truth = MODULE.normalize_obb(0, 0, 120, 40, 0)
    shrunk = MODULE.normalize_obb(0, 0, 112.8, 37.6, 0)
    z_truth, _ = MODULE.depth_from_plumb_box(
        truth, 900, 900, 1.8, 0.6, parameters)
    z_shrunk, _ = MODULE.depth_from_plumb_box(
        shrunk, 900, 900, 1.8, 0.6, parameters)
    assert z_shrunk / z_truth == pytest.approx(1.0 / 0.94)


def test_raw_opt_calibration_contract_is_required():
    valid = {
        "coordinate_contract": {"id": "raw_opt_v1", "target": "z_cg_opt_m"},
        "deployable_inputs": [
            "obb_geometry.w_px",
            "obb_geometry.h_px",
            "obb_geometry.gamma_deg",
        ],
    }
    MODULE.validate_raw_opt_calibration(valid)

    invalid = {
        "coordinate_contract": {
            "id": "legacy_plumb_opt_v1", "target": "z_cg_opt_m"},
        "deployable_inputs": [
            "obb_geometry.plumb_w_px",
            "obb_geometry.plumb_h_px",
            "obb_geometry.plumb_gamma_deg",
        ],
    }
    with pytest.raises(ValueError, match="raw_opt_v1"):
        MODULE.validate_raw_opt_calibration(invalid)
