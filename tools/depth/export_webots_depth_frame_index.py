#!/usr/bin/env python3
"""Export and validate the formal Webots monocular-depth frame index.

The exporter is read-only with respect to the Webots dataset.  It writes a
flat CSV for downstream analysis and a manifest that records the source
hashes, constant calibration values, sequence coverage, and validation
results.  Detector predictions are deliberately not joined here: GT geometry
and detector outputs remain separate evidence layers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


DEFAULT_SEQUENCES = (
    "calibration_train_03_tilt_obb",
    "fixed_dev_01_unseen_depth_swing",
    "unknown_test_02_coverage_retry",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--sequences", nargs="+", default=DEFAULT_SEQUENCES)
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
    rows: List[Dict[str, Any]] = []
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


def finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite value for {name}: {value!r}")
    return number


def flatten_matrix(matrix: Sequence[Sequence[Any]], prefix: str) -> Dict[str, float]:
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError(f"{prefix} must be 3x3")
    return {
        f"{prefix}_{row_index}{column_index}": finite(
            matrix[row_index][column_index],
            f"{prefix}[{row_index}][{column_index}]",
        )
        for row_index in range(3)
        for column_index in range(3)
    }


def theta_bin(theta_deg: float) -> str:
    if theta_deg < 3.0:
        return "0_to_3"
    if theta_deg < 5.0:
        return "3_to_5"
    if theta_deg < 8.0:
        return "5_to_8"
    return "8_or_more"


def relative_path(sequence_id: str, value: str) -> str:
    return str(Path(sequence_id) / Path(value))


def validate_constant_contract(
    manifests: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    manifests = list(manifests)
    first = manifests[0]
    expected_camera = first["camera"]
    expected_reference = first["obb_reference_geometry"]
    for manifest in manifests[1:]:
        if manifest["camera"] != expected_camera:
            raise ValueError("Camera contract differs across formal sequences")
        keys = ("long_edge_mean_m", "short_edge_mean_m", "reference_ar_3d")
        for key in keys:
            if not math.isclose(
                float(manifest["obb_reference_geometry"][key]),
                float(expected_reference[key]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"OBB reference geometry differs for {key}")
    return {
        "camera": expected_camera,
        "obb_reference_geometry": {
            key: expected_reference[key]
            for key in (
                "long_edge_mean_m",
                "short_edge_mean_m",
                "reference_ar_3d",
                "planarity_error_m",
                "grab_reference_center_error_m",
            )
        },
    }


def record_for_csv(
    sequence_id: str,
    role: str,
    sequence_dir: Path,
    row: Dict[str, Any],
) -> Dict[str, Any]:
    image_file = str(row["image_file"])
    label_file = str(row["label_file"])
    image_path = sequence_dir / image_file
    label_path = sequence_dir / label_file
    obb = row["obb_geometry"]
    camera = row["camera_geometry"]
    camera_xyz = camera["grab_reference_xyz_camera_cv_m"]
    pivot = row["pivot_relative"]
    self_rotation = row["grab_self_rotation"]
    camera_world = camera["position_world_xyz_m"]
    theta_total = finite(pivot["theta_total_deg"], "theta_total_deg")
    output: Dict[str, Any] = {
        "protocol_role": role,
        "sequence_id": sequence_id,
        "run_instance_id": row["run_instance_id"],
        "frame_index": int(row["frame_index"]),
        "simulation_time_s": finite(row["simulation_time_s"], "simulation_time_s"),
        "image_relpath": relative_path(sequence_id, image_file),
        "label_relpath": relative_path(sequence_id, label_file),
        "image_exists": int(image_path.is_file()),
        "label_exists": int(label_path.is_file()),
        "obb_valid": int(row["obb_valid"]),
        "truth_valid": int(row["truth_valid"]),
        "truth_audit_passed": int(row["frame_truth_audit"]["passed"]),
        "gt_cx_px": finite(obb["cx_px"], "cx_px"),
        "gt_cy_px": finite(obb["cy_px"], "cy_px"),
        "gt_raw_w_px": finite(obb["w_px"], "w_px"),
        "gt_raw_h_px": finite(obb["h_px"], "h_px"),
        "gt_raw_gamma_deg": finite(obb["gamma_deg"], "gamma_deg"),
        "gt_l_diag_raw_px": finite(obb["l_diag_raw_px"], "l_diag_raw_px"),
        "gt_plumb_w_px": finite(obb["plumb_w_px"], "plumb_w_px"),
        "gt_plumb_h_px": finite(obb["plumb_h_px"], "plumb_h_px"),
        "gt_plumb_gamma_deg": finite(obb["plumb_gamma_deg"], "plumb_gamma_deg"),
        "gt_l_diag_plumb_px": finite(obb["l_diag_plumb_px"], "l_diag_plumb_px"),
        "gt_pixel_corners_json": json.dumps(
            obb["pixel_corners"], ensure_ascii=False, separators=(",", ":")
        ),
        "projected_ar_warning": int(obb["projected_ar_warning"]),
        "grab_x_camera_cv_m": finite(camera_xyz[0], "grab_x_camera_cv_m"),
        "grab_y_camera_cv_m": finite(camera_xyz[1], "grab_y_camera_cv_m"),
        "z_cg_opt_gt_m": finite(camera["z_cg_opt_m"], "z_cg_opt_m"),
        "z_cg_plumb_gt_m": finite(camera["z_cg_plumb_m"], "z_cg_plumb_m"),
        "x_pg_gt_m": finite(pivot["x_m"], "x_pg_gt_m"),
        "y_pg_gt_m": finite(pivot["y_m"], "y_pg_gt_m"),
        "h_pg_vertical_gt_m": finite(pivot["z_pg_vertical_m"], "z_pg_vertical_m"),
        "l_pg_euclidean_gt_m": finite(
            pivot["geometric_rope_length_m"], "geometric_rope_length_m"
        ),
        "theta_x_gt_deg": finite(pivot["theta_x_deg"], "theta_x_deg"),
        "theta_y_gt_deg": finite(pivot["theta_y_deg"], "theta_y_deg"),
        "theta_total_gt_deg": theta_total,
        "theta_total_bin": theta_bin(theta_total),
        "psi_yaw_relative_deg": (
            ""
            if self_rotation.get("psi_yaw_relative_to_mechanical_zero_deg") is None
            else finite(
                self_rotation["psi_yaw_relative_to_mechanical_zero_deg"],
                "psi_yaw_relative_deg",
            )
        ),
        "camera_world_x_m": finite(camera_world[0], "camera_world_x_m"),
        "camera_world_y_m": finite(camera_world[1], "camera_world_y_m"),
        "camera_world_z_m": finite(camera_world[2], "camera_world_z_m"),
        "raw_to_plumb_homography_json": json.dumps(
            row["homography_raw_to_plumb"],
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }
    output.update(
        flatten_matrix(
            camera["rotation_world_from_camera_webots"], "camera_r_wc"
        )
    )
    return output


def sequence_summary(
    sequence_id: str,
    manifest: Dict[str, Any],
    rows: List[Dict[str, Any]],
    sequence_dir: Path,
) -> Dict[str, Any]:
    frame_indices = [int(row["frame_index"]) for row in rows]
    expected_indices = list(range(len(rows)))
    image_count = len(list((sequence_dir / "images").glob("*.jpg")))
    label_count = len(list((sequence_dir / "labels").glob("*.txt")))
    theta = [float(row["pivot_relative"]["theta_total_deg"]) for row in rows]
    z_opt = [float(row["camera_geometry"]["z_cg_opt_m"]) for row in rows]
    z_pg = [float(row["pivot_relative"]["z_pg_vertical_m"]) for row in rows]
    all_paths_exist = all(
        (sequence_dir / row["image_file"]).is_file()
        and (sequence_dir / row["label_file"]).is_file()
        for row in rows
    )
    bin_counts = {
        name: sum(theta_bin(value) == name for value in theta)
        for name in ("0_to_3", "3_to_5", "5_to_8", "8_or_more")
    }
    return {
        "sequence_id": sequence_id,
        "protocol_role": manifest["split"],
        "scenario": manifest["scenario"],
        "run_instance_id": manifest["run_instance_id"],
        "schema_version": manifest["schema_version"],
        "nominal_fps": manifest["sampling"]["nominal_fps"],
        "frame_count": len(rows),
        "image_count": image_count,
        "label_count": label_count,
        "frame_indices_contiguous_zero_based": frame_indices == expected_indices,
        "all_image_and_label_paths_exist": all_paths_exist,
        "obb_valid_count": sum(int(row["obb_valid"]) == 1 for row in rows),
        "truth_valid_count": sum(int(row["truth_valid"]) == 1 for row in rows),
        "truth_audit_pass_count": sum(
            int(row["frame_truth_audit"]["passed"]) == 1 for row in rows
        ),
        "z_cg_opt_range_m": [min(z_opt), max(z_opt)],
        "h_pg_vertical_range_m": [min(z_pg), max(z_pg)],
        "theta_total_range_deg": [min(theta), max(theta)],
        "theta_total_bin_counts": bin_counts,
        "manifest_sha256": sha256_file(sequence_dir / "metadata" / "manifest.json"),
        "frames_jsonl_sha256": sha256_file(sequence_dir / "metadata" / "frames.jsonl"),
    }


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    sequence_manifests: List[Dict[str, Any]] = []
    sequence_summaries: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []

    for sequence_id in args.sequences:
        sequence_dir = dataset_root / sequence_id
        manifest_path = sequence_dir / "metadata" / "manifest.json"
        frames_path = sequence_dir / "metadata" / "frames.jsonl"
        if not manifest_path.is_file() or not frames_path.is_file():
            raise FileNotFoundError(f"Incomplete sequence directory: {sequence_dir}")
        manifest = load_json(manifest_path)
        rows = load_jsonl(frames_path)
        if manifest["sequence_id"] != sequence_id:
            raise ValueError(f"Sequence ID mismatch for {sequence_dir}")
        if any(row["sequence_id"] != sequence_id for row in rows):
            raise ValueError(f"Frame sequence ID mismatch for {sequence_dir}")
        if any(row["run_instance_id"] != manifest["run_instance_id"] for row in rows):
            raise ValueError(f"Run ID mismatch for {sequence_dir}")
        summary = sequence_summary(sequence_id, manifest, rows, sequence_dir)
        checks = (
            summary["frame_count"] == summary["image_count"] == summary["label_count"],
            summary["frame_indices_contiguous_zero_based"],
            summary["all_image_and_label_paths_exist"],
            summary["obb_valid_count"] == summary["frame_count"],
            summary["truth_valid_count"] == summary["frame_count"],
            summary["truth_audit_pass_count"] == summary["frame_count"],
        )
        if not all(checks):
            raise ValueError(f"Sequence validation failed: {summary}")
        sequence_manifests.append(manifest)
        sequence_summaries.append(summary)
        csv_rows.extend(
            record_for_csv(sequence_id, manifest["split"], sequence_dir, row)
            for row in rows
        )

    constant_contract = validate_constant_contract(sequence_manifests)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    output_manifest = {
        "schema_version": "webots_depth_formal_frame_index_v1",
        "dataset_root": str(dataset_root),
        "included_sequences": list(args.sequences),
        "excluded_by_protocol": {
            "calibration_train_01": "historical power-law calibration only",
            "calibration_train_02_yaw_small_step": "training-side yaw audit only",
            "unknown_test_01_mixed_depth_swing": "coverage gate failed before model errors were exposed",
            "depth_pilot_04": "pipeline pilot",
            "depth_pilot_05_compact": "storage pilot",
        },
        "frame_count": len(csv_rows),
        "constant_contract": constant_contract,
        "sequence_summaries": sequence_summaries,
        "validation": {
            "all_sequences_passed": True,
            "detector_predictions_joined": False,
            "gps_used_as_model_input": False,
            "plc_or_encoder_used_as_model_input": False,
            "real_metric_depth_claim_allowed": False,
        },
        "csv_sha256": sha256_file(args.output_csv),
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
