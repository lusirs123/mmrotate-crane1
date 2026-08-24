#!/usr/bin/env python3
"""Audit or evaluate a frozen OBB-only depth calibration without refitting.

Use ``--coverage-only`` immediately after collection.  That stage intentionally
does not compute or reveal depth residuals.  Run the full evaluation only after
the preregistered coverage gate passes and the sequence directory is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np

from fit_obb_dual_scale import _load_sequence, _metrics, _stratified_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--eval", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--coverage-only",
        action="store_true",
        help="audit collection coverage without exposing prediction errors",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_frames(
    sequence_dir: Path, expected_sequence_id: str, expected_run_id: str
) -> Dict[str, Any]:
    frames_path = sequence_dir / "metadata" / "frames.jsonl"
    total = valid = audit_passed = 0
    run_ids = set()
    with frames_path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            if row.get("sequence_id") != expected_sequence_id:
                raise ValueError(
                    f"Frame sequence_id mismatch at {frames_path}:{line_number}")
            run_id = row.get("run_instance_id")
            if run_id != expected_run_id:
                raise ValueError(
                    f"Frame run_instance_id mismatch at {frames_path}:{line_number}")
            if run_id:
                run_ids.add(str(run_id))
            if row.get("obb_valid") and row.get("truth_valid"):
                valid += 1
                audit = row.get("frame_truth_audit", {})
                if audit.get("passed"):
                    audit_passed += 1
            elif row.get("truth_valid"):
                raise ValueError(
                    f"truth_valid frame has no valid OBB at {frames_path}:{line_number}")
    return {
        "stored_frame_count": total,
        "valid_frame_count": valid,
        "runtime_truth_audit_pass_count": audit_passed,
        "runtime_truth_audit_pass_rate": (
            audit_passed / valid if valid else 0.0),
        "run_instance_ids": sorted(run_ids),
        "single_run_instance": len(run_ids) == 1,
    }


def theta_bin_counts(theta_deg: np.ndarray) -> Dict[str, int]:
    bins = ((0.0, 3.0), (3.0, 5.0), (5.0, 8.0), (8.0, math.inf))
    result: Dict[str, int] = {}
    for lower, upper in bins:
        label = "inf" if math.isinf(upper) else f"{upper:g}"
        result[f"theta_{lower:g}_{label}_deg"] = int(
            np.count_nonzero((theta_deg >= lower) & (theta_deg < upper)))
    return result


def evaluate_gate(actual: Dict[str, Any], required: Dict[str, Any]) -> Dict[str, Any]:
    checks = {
        "valid_frames": actual["valid_frame_count"] >= int(
            required["min_valid_frames"]),
        "audit_pass_rate": actual["runtime_truth_audit_pass_rate"] >= float(
            required["required_runtime_truth_audit_pass_rate"]),
        "single_run_instance": bool(actual["single_run_instance"]),
        "z_span": actual["z_cg_opt_span_m"] >= float(
            required["min_z_cg_opt_span_m"]),
    }
    for key, minimum in required["min_theta_bin_counts"].items():
        checks[f"count_{key}"] = actual["theta_bin_counts"].get(key, 0) >= int(minimum)
    return {"passed": all(checks.values()), "checks": checks}


def main() -> None:
    args = parse_args()
    calibration_path = args.calibration.resolve()
    sequence_dir = args.eval.resolve()
    calibration = load_json(calibration_path)
    manifest = load_json(sequence_dir / "metadata" / "manifest.json")

    expected = calibration["preregistered_unknown_test"]
    if manifest.get("split") != "unknown_test":
        raise ValueError(
            f"Refusing unknown-test evaluation for split={manifest.get('split')!r}")
    if manifest.get("sequence_id") != expected["sequence_id"]:
        raise ValueError("Sequence ID does not match the preregistered unknown test")
    if manifest.get("scenario") != expected["scenario"]:
        raise ValueError("Scenario does not match the preregistered unknown test")
    seed = manifest.get("reproducibility", {}).get("python_random_seed")
    if seed != expected["random_seed"]:
        raise ValueError("Random seed does not match the preregistered unknown test")
    nominal_fps = manifest.get("sampling", {}).get("nominal_fps")
    if (nominal_fps is None or
            abs(float(nominal_fps) - float(expected["nominal_capture_fps"])) > 1e-6):
        raise ValueError("Sampling rate does not match the preregistered unknown test")
    if manifest.get("disturbance", {}).get("yaw_enabled"):
        raise ValueError("First unknown test must not enable active yaw disturbance")

    boundary = manifest.get("protocol_boundary", {})
    if boundary.get("scale_fit_allowed") or boundary.get("fit_obb_dual_scale_allowed"):
        raise ValueError("Unknown-test manifest incorrectly allows parameter fitting")
    if boundary.get("frozen_scale_calibration_id") != calibration["calibration_id"]:
        raise ValueError("Unknown-test manifest does not bind the frozen calibration ID")

    data = _load_sequence(sequence_dir, require_train=False)
    frame_audit = audit_frames(
        sequence_dir,
        expected_sequence_id=manifest["sequence_id"],
        expected_run_id=manifest["run_instance_id"],
    )
    z_gt = data["z_gt"]
    coverage = {
        **frame_audit,
        "z_cg_opt_min_m": float(np.min(z_gt)),
        "z_cg_opt_max_m": float(np.max(z_gt)),
        "z_cg_opt_span_m": float(np.max(z_gt) - np.min(z_gt)),
        "theta_total_min_deg": float(np.min(data["theta_deg"])),
        "theta_total_max_deg": float(np.max(data["theta_deg"])),
        "theta_bin_counts": theta_bin_counts(data["theta_deg"]),
        "q_signed_min": float(np.min(data["q_signed"])),
        "q_signed_max": float(np.max(data["q_signed"])),
    }
    coverage_gate = evaluate_gate(coverage, expected["coverage_gate"])
    report: Dict[str, Any] = {
        "schema_version": "frozen_obb_depth_unknown_eval_v1",
        "stage": "coverage_only" if args.coverage_only else "frozen_evaluation",
        "calibration_id": calibration["calibration_id"],
        "calibration_sha256": sha256_file(calibration_path),
        "sequence_id": manifest["sequence_id"],
        "run_instance_id": manifest.get("run_instance_id"),
        "coverage": coverage,
        "coverage_gate": coverage_gate,
        "parameter_refit_performed": False,
    }

    if not args.coverage_only:
        if not coverage_gate["passed"]:
            raise RuntimeError(
                "Coverage gate failed; refusing to expose unknown-test model errors")
        params = calibration["parameters"]
        prediction = (
            float(params["c"])
            * data["z_short"]
            * np.exp(float(params["beta"]) * data["q_signed"] ** 2)
        )
        metrics = _metrics(prediction, z_gt)
        metrics["by_theta"] = _stratified_metrics(
            prediction, z_gt, data["theta_deg"])
        gates = calibration["preregistered_unknown_test"]["performance_gate"]
        performance_checks = {
            "mae_m": metrics["mae_m"] <= float(gates["mae_m_max"]),
            "rmse_m": metrics["rmse_m"] <= float(gates["rmse_m_max"]),
            "abs_rel": metrics["abs_rel"] <= float(gates["abs_rel_max"]),
            "p95_abs_error_m": metrics["p95_abs_error_m"] <= float(
                gates["p95_abs_error_m_max"]),
        }
        report["metrics"] = metrics
        report["performance_gate"] = {
            "passed": all(performance_checks.values()),
            "checks": performance_checks,
            "thresholds": gates,
        }

    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
