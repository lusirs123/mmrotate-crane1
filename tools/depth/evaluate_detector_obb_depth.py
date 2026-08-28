#!/usr/bin/env python3
"""Evaluate frozen detector OBBs against one frozen Webots fixed-dev sequence.

The script never refits the monocular formula or changes detector boxes.  It
reports (1) raw OBB geometry against the four-point Webots truth and (2) depth
obtained after applying the stored raw-to-plumb homography and the frozen M2S
formula.  Unknown-test sequences are intentionally rejected by this entry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np

from fit_obb_dual_scale import _metrics, _stratified_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--sequence", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--reference-predictions", type=Path,
        help="Optional frozen component output for paired metric deltas")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    if not rows:
        raise ValueError(f"No records in {path}")
    return rows


def normalize_obb(
    cx: float, cy: float, width: float, height: float, angle_rad: float
) -> np.ndarray:
    box = np.asarray([cx, cy, width, height, angle_rad], dtype=np.float64)
    if not np.isfinite(box).all() or width <= 0.0 or height <= 0.0:
        raise ValueError(f"Invalid OBB: {box.tolist()}")
    if box[2] < box[3]:
        box[2], box[3] = box[3], box[2]
        box[4] += math.pi / 2.0
    box[4] = (box[4] + math.pi / 2.0) % math.pi - math.pi / 2.0
    return box


def obb_corners(box: np.ndarray) -> np.ndarray:
    cx, cy, width, height, angle = map(float, box)
    return cv2.boxPoints(
        ((cx, cy), (width, height), math.degrees(angle))).astype(np.float64)


def corners_to_obb(corners: np.ndarray) -> np.ndarray:
    (cx, cy), (width, height), angle_deg = cv2.minAreaRect(
        np.asarray(corners, dtype=np.float32))
    return normalize_obb(cx, cy, width, height, math.radians(angle_deg))


def transform_obb(box: np.ndarray, homography: Iterable[Iterable[float]]) -> np.ndarray:
    points = obb_corners(box).reshape(1, 4, 2).astype(np.float64)
    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Invalid raw-to-plumb homography")
    transformed = cv2.perspectiveTransform(points, matrix)[0]
    return corners_to_obb(transformed)


def rotated_iou(first: np.ndarray, second: np.ndarray) -> float:
    def rect(box: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
        return ((float(box[0]), float(box[1])),
                (float(box[2]), float(box[3])),
                math.degrees(float(box[4])))
    kind, polygon = cv2.rotatedRectangleIntersection(rect(first), rect(second))
    intersection = 0.0 if polygon is None else abs(float(cv2.contourArea(polygon)))
    union = float(first[2] * first[3] + second[2] * second[3] - intersection)
    return intersection / union if union > 0.0 else 0.0


def angle_error_deg(first: float, second: float) -> float:
    delta = 0.5 * math.atan2(
        math.sin(2.0 * (first - second)),
        math.cos(2.0 * (first - second)))
    return abs(math.degrees(delta))


def summary(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def prediction_manifest_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


def validate_prediction_manifest(
    path: Path, manifest: Dict[str, Any], expected_frames: int
) -> None:
    if manifest.get("limit") is not None:
        raise ValueError(f"Refusing partial/smoke predictions: {path}")
    if int(manifest.get("processed_frame_count", -1)) != expected_frames:
        raise ValueError("Prediction manifest does not cover the full sequence")
    if manifest.get("metric_truth_read") is not False:
        raise ValueError("Detector inference manifest must state metric_truth_read=false")
    if manifest.get("threshold_tuning_performed") is not False:
        raise ValueError("Detector inference performed threshold tuning")


def truth_box(row: Dict[str, Any]) -> np.ndarray:
    obb = row["obb_geometry"]
    return normalize_obb(
        float(obb["cx_px"]), float(obb["cy_px"]),
        float(obb["w_px"]), float(obb["h_px"]),
        math.radians(float(obb["gamma_deg"])))


def predicted_box(record: Dict[str, Any]) -> np.ndarray | None:
    top1 = record.get("top1")
    if not record.get("detected") or top1 is None:
        return None
    return normalize_obb(
        float(top1["cx_px"]), float(top1["cy_px"]),
        float(top1["width_px"]), float(top1["height_px"]),
        float(top1["angle_rad"]))


def depth_from_plumb_box(
    box: np.ndarray, fx: float, fy: float, long_m: float, short_m: float,
    parameters: Dict[str, Any]
) -> Tuple[float, Dict[str, float]]:
    _, _, width, height, gamma = map(float, box)
    s_long = width * math.sqrt(
        (math.cos(gamma) / fx) ** 2 + (math.sin(gamma) / fy) ** 2)
    s_short = height * math.sqrt(
        (math.sin(gamma) / fx) ** 2 + (math.cos(gamma) / fy) ** 2)
    z_long = long_m / s_long
    z_short = short_m / s_short
    q_signed = math.log(z_long / z_short)
    prediction = (
        float(parameters["c"]) * z_short
        * math.exp(float(parameters["beta"]) * q_signed**2)
        + float(parameters.get("b_m", 0.0)))
    return prediction, {
        "z_long_m": z_long,
        "z_short_m": z_short,
        "q_signed": q_signed,
    }


def evaluate_predictions(
    predictions_path: Path, truth_rows: List[Dict[str, Any]],
    sequence_manifest: Dict[str, Any], calibration: Dict[str, Any]
) -> Dict[str, Any]:
    predictions = load_jsonl(predictions_path)
    manifest_path = prediction_manifest_path(predictions_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing prediction manifest: {manifest_path}")
    prediction_manifest = load_json(manifest_path)
    validate_prediction_manifest(predictions_path, prediction_manifest, len(truth_rows))
    if len(predictions) != len(truth_rows):
        raise ValueError("Prediction/truth row count mismatch")

    intrinsics = sequence_manifest["camera"]["intrinsics"]
    reference = sequence_manifest["obb_reference_geometry"]
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    long_m = float(reference["long_edge_mean_m"])
    short_m = float(reference["short_edge_mean_m"])
    parameters = calibration["parameters"]

    center_errors: List[float] = []
    rious: List[float] = []
    angle_errors: List[float] = []
    long_rel: List[float] = []
    short_rel: List[float] = []
    diag_rel: List[float] = []
    scores: List[float] = []
    depth_prediction: List[float] = []
    depth_truth: List[float] = []
    theta: List[float] = []
    oracle_prediction: List[float] = []
    missing_frames: List[str] = []

    for pred, truth in zip(predictions, truth_rows):
        expected_frame = Path(str(truth["image_file"])).stem
        if pred.get("frame_id") != expected_frame:
            raise ValueError(
                f"Frame mismatch: prediction={pred.get('frame_id')} truth={expected_frame}")
        gt = truth_box(truth)
        gt_plumb = normalize_obb(
            float(truth["obb_geometry"]["cx_px"]),
            float(truth["obb_geometry"]["cy_px"]),
            float(truth["obb_geometry"]["plumb_w_px"]),
            float(truth["obb_geometry"]["plumb_h_px"]),
            math.radians(float(truth["obb_geometry"]["plumb_gamma_deg"])))
        oracle_z, _ = depth_from_plumb_box(
            gt_plumb, fx, fy, long_m, short_m, parameters)
        oracle_prediction.append(oracle_z)

        box = predicted_box(pred)
        if box is None:
            missing_frames.append(expected_frame)
            continue
        plumb = transform_obb(box, truth["homography_raw_to_plumb"])
        predicted_z, _ = depth_from_plumb_box(
            plumb, fx, fy, long_m, short_m, parameters)
        gt_diag = math.hypot(float(gt[2]), float(gt[3]))
        pred_diag = math.hypot(float(box[2]), float(box[3]))
        center_errors.append(float(np.linalg.norm(box[:2] - gt[:2])))
        rious.append(rotated_iou(box, gt))
        angle_errors.append(angle_error_deg(float(box[4]), float(gt[4])))
        long_rel.append(float(box[2] / gt[2] - 1.0))
        short_rel.append(float(box[3] / gt[3] - 1.0))
        diag_rel.append(pred_diag / gt_diag - 1.0)
        scores.append(float(pred["top1"]["score"]))
        depth_prediction.append(predicted_z)
        depth_truth.append(float(truth["camera_geometry"]["z_cg_opt_m"]))
        theta.append(float(truth["pivot_relative"]["theta_total_deg"]))

    all_gt = np.asarray([
        float(row["camera_geometry"]["z_cg_opt_m"]) for row in truth_rows
    ], dtype=np.float64)
    oracle = np.asarray(oracle_prediction, dtype=np.float64)
    detected_depth = np.asarray(depth_prediction, dtype=np.float64)
    detected_gt = np.asarray(depth_truth, dtype=np.float64)
    detected_theta = np.asarray(theta, dtype=np.float64)
    metrics = _metrics(detected_depth, detected_gt) if detected_depth.size else None
    if metrics is not None:
        metrics["by_theta"] = _stratified_metrics(
            detected_depth, detected_gt, detected_theta)

    return {
        "prediction_file": str(predictions_path.resolve()),
        "prediction_sha256": sha256_file(predictions_path),
        "prediction_manifest": prediction_manifest,
        "frame_count": len(truth_rows),
        "detected_frame_count": len(depth_prediction),
        "detection_rate": len(depth_prediction) / len(truth_rows),
        "missing_frames": missing_frames,
        "geometry": {
            "center_error_px": summary(center_errors),
            "rotated_iou": summary(rious),
            "angle_abs_error_deg": summary(angle_errors),
            "long_edge_relative_error": summary(long_rel),
            "short_edge_relative_error": summary(short_rel),
            "diagonal_relative_error": summary(diag_rel),
            "score": summary(scores),
        },
        "detector_obb_depth_metrics": metrics,
        "truth_obb_oracle_depth_metrics": _metrics(oracle, all_gt),
    }


def metric_delta(current: Dict[str, Any], reference: Dict[str, Any]) -> Dict[str, float]:
    result = {}
    first = current.get("detector_obb_depth_metrics") or {}
    second = reference.get("detector_obb_depth_metrics") or {}
    for key in ("mae_m", "rmse_m", "abs_rel", "bias_m", "p95_abs_error_m"):
        if key in first and key in second:
            result[key] = float(first[key] - second[key])
    for key in ("detection_rate",):
        result[key] = float(current[key] - reference[key])
    return result


def main() -> None:
    args = parse_args()
    sequence_dir = args.sequence.resolve()
    sequence_manifest = load_json(sequence_dir / "metadata" / "manifest.json")
    if sequence_manifest.get("split") != "fixed_dev":
        raise ValueError(
            "This evaluator is fixed-dev only; unknown-sequence evaluation "
            "must remain a separate protocol")
    calibration = load_json(args.calibration.resolve())
    truth_rows = load_jsonl(sequence_dir / "metadata" / "frames.jsonl")
    current = evaluate_predictions(
        args.predictions.resolve(), truth_rows, sequence_manifest, calibration)

    gates = calibration.get("preregistered_unknown_test", {}).get(
        "performance_gate", {})
    metrics = current.get("detector_obb_depth_metrics") or {}
    checks = {
        "complete_detection": current["detected_frame_count"] == current["frame_count"],
        "mae_m": metrics.get("mae_m", math.inf) <= float(gates.get("mae_m_max", math.inf)),
        "rmse_m": metrics.get("rmse_m", math.inf) <= float(gates.get("rmse_m_max", math.inf)),
        "abs_rel": metrics.get("abs_rel", math.inf) <= float(gates.get("abs_rel_max", math.inf)),
        "p95_abs_error_m": metrics.get("p95_abs_error_m", math.inf) <= float(
            gates.get("p95_abs_error_m_max", math.inf)),
    }
    report: Dict[str, Any] = {
        "schema_version": "fixed_dev_detector_obb_depth_eval_v1",
        "protocol_role": "fixed_target_dev_component_diagnostic",
        "sequence_id": sequence_manifest["sequence_id"],
        "calibration_id": calibration["calibration_id"],
        "calibration_sha256": sha256_file(args.calibration.resolve()),
        "parameter_refit_performed": False,
        "threshold_tuning_performed": False,
        "unknown_sequence_read": False,
        "real_metric_depth_claim_allowed": False,
        "current": current,
        "diagnostic_reference_gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "thresholds_reused_from_frozen_unknown_protocol": gates,
            "deployment_gate": False,
        },
    }
    if args.reference_predictions is not None:
        reference = evaluate_predictions(
            args.reference_predictions.resolve(), truth_rows,
            sequence_manifest, calibration)
        report["reference"] = reference
        report["paired_metric_delta_current_minus_reference"] = metric_delta(
            current, reference)

    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
