#!/usr/bin/env python3
"""Fit and audit compact OBB-only monocular depth formulas.

The script deliberately consumes only deployable OBB geometry plus camera
intrinsics and the known rigid grab dimensions.  Webots metric truth is used
as the supervised target, never as a model input.

Candidate models:
  M0: Z = k * l_diag ** alpha                         (legacy baseline)
  M1: Z = c * Z_diag + b                              (pinhole baseline)
  M1L/M1S: Z = c * Z_long_or_short + b                (axis ablations)
  M2S: Z = c * Z_short * exp(beta * q**2) + b         (minimal correction)
  M2: Z = c * Z_long**lambda * Z_short**(1-lambda)
          * exp(beta * q**2) + b                      (dual-scale model)

where q = log(Z_long / Z_short).  The paper-facing minimal M2 sets b=0;
``--fit-offset`` enables b only as an ablation.

Coordinate contracts are explicit.  The deployable default is ``raw_opt_v1``:
raw-image OBB geometry predicts the physical camera optical-axis depth.  The
legacy plumb-OBB/optical-depth pairing is retained only so historical artifacts
remain reproducible; it must not be promoted as a geometrically closed model.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
from scipy.optimize import least_squares


THETA_BINS_DEG = ((0.0, 3.0), (3.0, 5.0), (5.0, 8.0), (8.0, math.inf))

COORDINATE_CONTRACTS = {
    "raw_opt_v1": {
        "width": "w_px",
        "height": "h_px",
        "angle": "gamma_deg",
        "diagonal": "l_diag_raw_px",
        "target": "z_cg_opt_m",
        "description": "raw-image OBB -> physical-camera optical-axis depth",
        "geometrically_closed": True,
    },
    "plumb_plumb_v1": {
        "width": "plumb_w_px",
        "height": "plumb_h_px",
        "angle": "plumb_gamma_deg",
        "diagonal": "l_diag_plumb_px",
        "target": "z_cg_plumb_m",
        "description": "virtual-plumb OBB -> virtual-plumb axis depth",
        "geometrically_closed": True,
    },
    "legacy_plumb_opt_v1": {
        "width": "plumb_w_px",
        "height": "plumb_h_px",
        "angle": "plumb_gamma_deg",
        "diagonal": "l_diag_plumb_px",
        "target": "z_cg_opt_m",
        "description": "historical plumb-OBB -> physical optical-depth pairing",
        "geometrically_closed": False,
    },
}


def _sequence_files(sequence_dir: Path) -> Tuple[Path, Path]:
    manifest = sequence_dir / "metadata" / "manifest.json"
    frames = sequence_dir / "metadata" / "frames.jsonl"
    if not manifest.is_file() or not frames.is_file():
        raise FileNotFoundError(
            f"Expected metadata/manifest.json and metadata/frames.jsonl under {sequence_dir}")
    return manifest, frames


def _load_sequence(
    sequence_dir: Path,
    require_train: bool,
    coordinate_contract: str = "legacy_plumb_opt_v1",
) -> Dict[str, Any]:
    if coordinate_contract not in COORDINATE_CONTRACTS:
        raise ValueError(f"Unknown coordinate contract: {coordinate_contract}")
    contract = COORDINATE_CONTRACTS[coordinate_contract]
    manifest_path, frames_path = _sequence_files(sequence_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if require_train and manifest.get("split") != "calibration_train":
        raise ValueError(
            f"Refusing to fit on split={manifest.get('split')!r}; "
            "training input must be calibration_train")

    intrinsics = manifest["camera"]["intrinsics"]
    reference = manifest["obb_reference_geometry"]
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    long_m = float(reference["long_edge_mean_m"])
    short_m = float(reference["short_edge_mean_m"])

    rows = []
    with frames_path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not (row.get("obb_valid") and row.get("truth_valid")):
                continue
            obb = row.get("obb_geometry", {})
            camera = row.get("camera_geometry", {})
            pivot = row.get("pivot_relative", {})
            required = (
                obb.get(contract["width"]),
                obb.get(contract["height"]),
                obb.get(contract["angle"]),
                obb.get(contract["diagonal"]),
                camera.get(contract["target"]),
                pivot.get("theta_total_deg"),
            )
            if any(value is None for value in required):
                raise ValueError(
                    f"Missing compact depth field at {frames_path}:{line_number}")
            rows.append(required)

    if len(rows) < 20:
        raise ValueError(f"Only {len(rows)} valid rows in {sequence_dir}; need at least 20")

    values = np.asarray(rows, dtype=np.float64)
    w_px, h_px, gamma_deg, diag_px, z_gt, theta_deg = values.T
    if np.any(w_px <= 0.0) or np.any(h_px <= 0.0) or np.any(z_gt <= 0.0):
        raise ValueError(f"Non-positive scale/depth value found in {sequence_dir}")

    gamma = np.deg2rad(gamma_deg)
    # Convert pixel edge lengths into normalized image-plane angular lengths.
    s_long = w_px * np.sqrt((np.cos(gamma) / fx) ** 2 + (np.sin(gamma) / fy) ** 2)
    s_short = h_px * np.sqrt((np.sin(gamma) / fx) ** 2 + (np.cos(gamma) / fy) ** 2)
    z_long = long_m / s_long
    z_short = short_m / s_short
    z_diag = math.hypot(long_m, short_m) / np.sqrt(s_long**2 + s_short**2)
    q_signed = np.log(z_long / z_short)

    return {
        "manifest": manifest,
        "sequence_dir": str(sequence_dir),
        "coordinate_contract": coordinate_contract,
        "coordinate_contract_definition": contract,
        "diag_px": diag_px,
        "z_gt": z_gt,
        "theta_deg": theta_deg,
        "z_long": z_long,
        "z_short": z_short,
        "z_diag": z_diag,
        "q_signed": q_signed,
    }


def _metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    residual = prediction - target
    absolute = np.abs(residual)
    return {
        "count": int(target.size),
        "mae_m": float(np.mean(absolute)),
        "rmse_m": float(np.sqrt(np.mean(residual**2))),
        "abs_rel": float(np.mean(absolute / target)),
        "bias_m": float(np.mean(residual)),
        "max_abs_error_m": float(np.max(absolute)),
        "p95_abs_error_m": float(np.quantile(absolute, 0.95)),
    }


def _stratified_metrics(
    prediction: np.ndarray, target: np.ndarray, theta_deg: np.ndarray
) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for lower, upper in THETA_BINS_DEG:
        mask = (theta_deg >= lower) & (theta_deg < upper)
        if not np.any(mask):
            continue
        upper_label = "inf" if math.isinf(upper) else f"{upper:g}"
        result[f"theta_{lower:g}_{upper_label}_deg"] = _metrics(
            prediction[mask], target[mask])
    return result


def _fit_m0(data: Dict[str, Any]) -> Dict[str, Any]:
    x = np.log(data["diag_px"])
    y = np.log(data["z_gt"])
    alpha, log_k = np.polyfit(x, y, 1)
    return {"k": float(np.exp(log_k)), "alpha": float(alpha)}


def _predict_m0(params: Dict[str, Any], data: Dict[str, Any]) -> np.ndarray:
    return params["k"] * data["diag_px"] ** params["alpha"]


def _fit_m1(data: Dict[str, Any], huber_scale_m: float, fit_offset: bool) -> Dict[str, Any]:
    z_diag = data["z_diag"]
    target = data["z_gt"]

    def residual(params: np.ndarray) -> np.ndarray:
        log_c = params[0]
        offset = params[1] if fit_offset else 0.0
        return np.exp(log_c) * z_diag + offset - target

    initial = np.array([0.0, 0.0] if fit_offset else [0.0])
    lower = np.array([-2.0, -2.0] if fit_offset else [-2.0])
    upper = np.array([2.0, 2.0] if fit_offset else [2.0])
    fit = least_squares(
        residual, initial, bounds=(lower, upper), loss="huber", f_scale=huber_scale_m)
    return {
        "c": float(np.exp(fit.x[0])),
        "b_m": float(fit.x[1]) if fit_offset else 0.0,
        "fit_offset": bool(fit_offset),
        "optimizer_success": bool(fit.success),
    }


def _predict_m1(params: Dict[str, Any], data: Dict[str, Any]) -> np.ndarray:
    return params["c"] * data["z_diag"] + params["b_m"]


def _fit_single_axis(
    data: Dict[str, Any], axis_key: str, huber_scale_m: float, fit_offset: bool
) -> Dict[str, Any]:
    scale = data[axis_key]
    target = data["z_gt"]

    def residual(params: np.ndarray) -> np.ndarray:
        offset = params[1] if fit_offset else 0.0
        return np.exp(params[0]) * scale + offset - target

    initial = np.array([0.0, 0.0] if fit_offset else [0.0])
    lower = np.array([-2.0, -2.0] if fit_offset else [-2.0])
    upper = np.array([2.0, 2.0] if fit_offset else [2.0])
    fit = least_squares(
        residual, initial, bounds=(lower, upper), loss="huber", f_scale=huber_scale_m)
    return {
        "axis_key": axis_key,
        "c": float(np.exp(fit.x[0])),
        "b_m": float(fit.x[1]) if fit_offset else 0.0,
        "fit_offset": bool(fit_offset),
        "optimizer_success": bool(fit.success),
    }


def _predict_single_axis(params: Dict[str, Any], data: Dict[str, Any]) -> np.ndarray:
    return params["c"] * data[params["axis_key"]] + params["b_m"]


def _fit_short_q2(
    data: Dict[str, Any], huber_scale_m: float, fit_offset: bool
) -> Dict[str, Any]:
    z_short = data["z_short"]
    q = data["q_signed"]
    target = data["z_gt"]

    def residual(params: np.ndarray) -> np.ndarray:
        log_c, beta = params[:2]
        offset = params[2] if fit_offset else 0.0
        return np.exp(log_c) * z_short * np.exp(beta * q**2) + offset - target

    initial = np.array([0.0, 0.0, 0.0] if fit_offset else [0.0, 0.0])
    lower = np.array([-2.0, -100.0, -2.0] if fit_offset else [-2.0, -100.0])
    upper = np.array([2.0, 100.0, 2.0] if fit_offset else [2.0, 100.0])
    fit = least_squares(
        residual, initial, bounds=(lower, upper), loss="huber",
        f_scale=huber_scale_m, max_nfev=20000)
    return {
        "c": float(np.exp(fit.x[0])),
        "beta": float(fit.x[1]),
        "b_m": float(fit.x[2]) if fit_offset else 0.0,
        "fit_offset": bool(fit_offset),
        "optimizer_success": bool(fit.success),
        "q_signed_min": float(np.min(q)),
        "q_signed_max": float(np.max(q)),
        "q_signed_p01": float(np.quantile(q, 0.01)),
        "q_signed_p99": float(np.quantile(q, 0.99)),
    }


def _predict_short_q2(params: Dict[str, Any], data: Dict[str, Any]) -> np.ndarray:
    q = data["q_signed"]
    return (
        params["c"] * data["z_short"] * np.exp(params["beta"] * q**2)
        + params["b_m"])


def _fit_m2(data: Dict[str, Any], huber_scale_m: float, fit_offset: bool) -> Dict[str, Any]:
    z_long = data["z_long"]
    z_short = data["z_short"]
    q = data["q_signed"]
    target = data["z_gt"]

    def residual(params: np.ndarray) -> np.ndarray:
        log_c, lambda_logit, beta = params[:3]
        lam = 1.0 / (1.0 + np.exp(-lambda_logit))
        offset = params[3] if fit_offset else 0.0
        prediction = (
            np.exp(log_c)
            * z_long**lam
            * z_short ** (1.0 - lam)
            * np.exp(beta * q**2)
            + offset
        )
        return prediction - target

    initial = np.array([0.0, 0.0, 0.0, 0.0] if fit_offset else [0.0, 0.0, 0.0])
    lower = np.array([-2.0, -8.0, -100.0, -2.0] if fit_offset else [-2.0, -8.0, -100.0])
    upper = np.array([2.0, 8.0, 100.0, 2.0] if fit_offset else [2.0, 8.0, 100.0])
    fit = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        loss="huber",
        f_scale=huber_scale_m,
        max_nfev=20000,
    )
    lam = 1.0 / (1.0 + math.exp(-float(fit.x[1])))
    return {
        "c": float(math.exp(float(fit.x[0]))),
        "lambda": float(lam),
        "beta": float(fit.x[2]),
        "b_m": float(fit.x[3]) if fit_offset else 0.0,
        "fit_offset": bool(fit_offset),
        "optimizer_success": bool(fit.success),
        "q_signed_min": float(np.min(q)),
        "q_signed_max": float(np.max(q)),
    }


def _predict_m2(params: Dict[str, Any], data: Dict[str, Any]) -> np.ndarray:
    lam = params["lambda"]
    q = data["q_signed"]
    return (
        params["c"]
        * data["z_long"] ** lam
        * data["z_short"] ** (1.0 - lam)
        * np.exp(params["beta"] * q**2)
        + params["b_m"]
    )


def _evaluate_models(models: Dict[str, Dict[str, Any]], data: Dict[str, Any]) -> Dict[str, Any]:
    predictors = {
        "M0_power": _predict_m0,
        "M1_inverse": _predict_m1,
        "M1_long_only": _predict_single_axis,
        "M1_short_only": _predict_single_axis,
        "M2_short_q2": _predict_short_q2,
        "M2_dual_scale": _predict_m2,
    }
    result: Dict[str, Any] = {}
    for name, params in models.items():
        prediction = predictors[name](params, data)
        result[name] = {
            "overall": _metrics(prediction, data["z_gt"]),
            "by_theta": _stratified_metrics(
                prediction, data["z_gt"], data["theta_deg"]),
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path, help="calibration_train sequence directory")
    parser.add_argument("--eval", type=Path, help="independent fixed-dev or test sequence directory")
    parser.add_argument("--output", type=Path, help="write JSON report/model to this path")
    parser.add_argument("--huber-scale-m", type=float, default=0.05)
    parser.add_argument(
        "--coordinate-contract",
        choices=tuple(COORDINATE_CONTRACTS),
        default="raw_opt_v1",
        help=(
            "pixel/depth coordinate pairing; raw_opt_v1 is the deployable "
            "default, while legacy_plumb_opt_v1 is historical only"
        ),
    )
    parser.add_argument(
        "--fit-offset",
        action="store_true",
        help="fit additive b term as an ablation; the paper-facing minimal M2 keeps b=0",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.huber_scale_m <= 0.0:
        raise ValueError("--huber-scale-m must be positive")

    train = _load_sequence(
        args.train.resolve(), require_train=True,
        coordinate_contract=args.coordinate_contract)
    q_span = float(np.quantile(train["q_signed"], 0.95) - np.quantile(train["q_signed"], 0.05))
    theta_max = float(np.max(train["theta_deg"]))
    identifiability_warnings = []
    if q_span < 0.005:
        identifiability_warnings.append(
            "dual-scale q coverage is too narrow; M2 correction is not identifiable")
    if theta_max < 5.0:
        identifiability_warnings.append(
            "training sequence has no >=5 deg tilt coverage; do not freeze M2")
    models = {
        "M0_power": _fit_m0(train),
        "M1_inverse": _fit_m1(train, args.huber_scale_m, args.fit_offset),
        "M1_long_only": _fit_single_axis(
            train, "z_long", args.huber_scale_m, args.fit_offset),
        "M1_short_only": _fit_single_axis(
            train, "z_short", args.huber_scale_m, args.fit_offset),
        "M2_short_q2": _fit_short_q2(
            train, args.huber_scale_m, args.fit_offset),
        "M2_dual_scale": _fit_m2(train, args.huber_scale_m, args.fit_offset),
    }
    fitted_lambda = models["M2_dual_scale"]["lambda"]
    if fitted_lambda < 0.02 or fitted_lambda > 0.98:
        identifiability_warnings.append(
            "M2 lambda is near a [0,1] boundary; compare the simpler axis "
            "ablations before freezing the formula")
    report: Dict[str, Any] = {
        "schema_version": "obb_dual_scale_depth_fit_v2",
        "coordinate_contract": {
            "id": args.coordinate_contract,
            **COORDINATE_CONTRACTS[args.coordinate_contract],
        },
        "target": (
            "camera_geometry."
            + COORDINATE_CONTRACTS[args.coordinate_contract]["target"]),
        "deployable_inputs_only": True,
        "fit_loss": {"name": "Huber", "scale_m": args.huber_scale_m},
        "train_sequence": train["manifest"].get("sequence_id"),
        "training_coverage": {
            "q_p05_to_p95_span": q_span,
            "theta_total_max_deg": theta_max,
            "dual_scale_identifiability_passed": not identifiability_warnings,
            "warnings": identifiability_warnings,
        },
        "models": models,
        "train_metrics_for_diagnostic_only": _evaluate_models(models, train),
        "paper_facing_formula": (
            "Z_hat = c * Z_long^lambda * Z_short^(1-lambda) "
            "* exp(beta * log(Z_long/Z_short)^2)"
        ),
        "minimal_formula_candidate": (
            "Z_hat = c * Z_short * exp(beta * log(Z_long/Z_short)^2)"
        ),
    }

    if args.eval is not None:
        evaluation = _load_sequence(
            args.eval.resolve(), require_train=False,
            coordinate_contract=args.coordinate_contract)
        if evaluation["manifest"].get("sequence_id") == train["manifest"].get("sequence_id"):
            raise ValueError("Train and evaluation sequence IDs must differ")
        report["eval_sequence"] = evaluation["manifest"].get("sequence_id")
        report["eval_split"] = evaluation["manifest"].get("split")
        report["eval_metrics"] = _evaluate_models(models, evaluation)

    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
