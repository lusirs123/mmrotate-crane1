#!/usr/bin/env python3
"""Run a frozen MMRotate detector on one Webots image sequence.

This entrypoint deliberately performs detection only.  It writes one JSONL
record per input frame and does not read Webots metric truth, fit depth
parameters, select thresholds, or modify detector outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from mmdet.apis import inference_detector, init_detector

import mmrotate  # noqa: F401


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen OBB inference for a Webots sequence")
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("sequence_dir", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N sorted frames (smoke test only)")
    parser.add_argument(
        "--expected-frames", type=int, default=None,
        help="Fail unless the discovered frame count matches this value")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace an existing output file")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_images(sequence_dir: Path) -> List[Path]:
    image_dir = sequence_dir / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    images = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise RuntimeError(f"No images found under {image_dir}")
    return images


def flatten_obb_result(result: Any) -> List[Tuple[int, np.ndarray]]:
    if isinstance(result, tuple):
        result = result[0]
    if not isinstance(result, (list, tuple)):
        raise TypeError(f"Unexpected detector result type: {type(result)!r}")

    detections: List[Tuple[int, np.ndarray]] = []
    for class_id, class_result in enumerate(result):
        array = np.asarray(class_result)
        if array.size == 0:
            continue
        if array.ndim != 2 or array.shape[1] < 6:
            raise ValueError(
                f"Expected rotated detections shaped Nx6+, got {array.shape}")
        for row in array:
            detections.append((class_id, row.astype(np.float64, copy=False)))
    detections.sort(key=lambda item: float(item[1][-1]), reverse=True)
    return detections


def prediction_record(image_path: Path, result: Any) -> Dict[str, Any]:
    detections = flatten_obb_result(result)
    record: Dict[str, Any] = {
        "frame_id": image_path.stem,
        "image_name": image_path.name,
        "detected": bool(detections),
        "num_detections": len(detections),
        "top1": None,
    }
    if not detections:
        return record

    class_id, row = detections[0]
    cx, cy, width, height, angle_rad = map(float, row[:5])
    score = float(row[-1])
    values = (cx, cy, width, height, angle_rad, score)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Non-finite top-1 prediction for {image_path}")
    record["top1"] = {
        "class_id": int(class_id),
        "cx_px": cx,
        "cy_px": cy,
        "width_px": width,
        "height_px": height,
        "angle_rad": angle_rad,
        "angle_deg": math.degrees(angle_rad),
        "score": score,
    }
    return record


def main() -> None:
    args = parse_args()
    config = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    sequence_dir = args.sequence_dir.resolve()
    output_jsonl = args.output_jsonl.resolve()

    if not config.is_file():
        raise FileNotFoundError(f"Config not found: {config}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if output_jsonl.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_jsonl}; pass --overwrite to replace")

    all_images = discover_images(sequence_dir)
    if args.expected_frames is not None and len(all_images) != args.expected_frames:
        raise RuntimeError(
            f"Expected {args.expected_frames} images, found {len(all_images)}")
    images = all_images if args.limit is None else all_images[:args.limit]
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    model = init_detector(str(config), str(checkpoint), device=args.device)

    detected_count = 0
    with output_jsonl.open("w", encoding="utf-8") as output_file:
        for index, image_path in enumerate(images, start=1):
            result = inference_detector(model, str(image_path))
            record = prediction_record(image_path, result)
            detected_count += int(record["detected"])
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            if index == 1 or index % 50 == 0 or index == len(images):
                print(
                    f"[infer] {index}/{len(images)} "
                    f"detected={detected_count} image={image_path.name}",
                    flush=True)

    manifest = {
        "schema_version": "webots_frozen_obb_predictions_v1",
        "sequence_id": sequence_dir.name,
        "sequence_dir": str(sequence_dir),
        "config": str(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "device": args.device,
        "discovered_frame_count": len(all_images),
        "processed_frame_count": len(images),
        "detected_frame_count": detected_count,
        "limit": args.limit,
        "model_selection_performed": False,
        "threshold_tuning_performed": False,
        "metric_truth_read": False,
    }
    manifest_path = output_jsonl.with_suffix(output_jsonl.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"[done] predictions={output_jsonl}")
    print(f"[done] manifest={manifest_path}")


if __name__ == "__main__":
    main()
