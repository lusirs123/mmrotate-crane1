#!/usr/bin/env python3
"""Build a diagnosis-only contact sheet for proxy and target-atlas previews."""

from __future__ import annotations

import argparse
import glob
import os
from typing import List, Sequence, Tuple

import cv2
import numpy as np


DEFAULT_PROXY_VARIANTS = (
    'industrial_edges_d0p45_x1p00_overlay',
    'source_background_d0p45_x1p00_overlay',
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare source proxy candidate overlays with the '
                    'diagnosis-only target hard-negative atlas.')
    parser.add_argument('--proxy-preview-dir', required=True)
    parser.add_argument('--atlas-preview-dir', required=True)
    parser.add_argument('--proxy-variants', nargs='+',
                        default=list(DEFAULT_PROXY_VARIANTS))
    parser.add_argument('--proxy-count', type=int, default=3)
    parser.add_argument('--atlas-count', type=int, default=6)
    parser.add_argument('--cell-width', type=int, default=360)
    parser.add_argument('--cell-height', type=int, default=240)
    parser.add_argument('--out', required=True)
    return parser.parse_args()


def discover_images(directory: str) -> List[str]:
    paths = []
    for extension in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        paths.extend(glob.glob(os.path.join(directory, extension)))
    return sorted(set(os.path.realpath(path) for path in paths))


def evenly_select(paths: Sequence[str], count: int) -> List[str]:
    paths = list(paths)
    if not paths or count <= 0:
        return []
    sample_count = min(int(count), len(paths))
    positions = np.linspace(
        0, len(paths) - 1, sample_count, dtype=np.int64)
    return [paths[int(position)] for position in positions]


def _render_cell(path: str, width: int, height: int) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f'Failed to read preview: {path}')
    footer = 24
    content_height = height - footer
    scale = min(
        width / max(float(image.shape[1]), 1.0),
        content_height / max(float(image.shape[0]), 1.0))
    resized = cv2.resize(
        image,
        (max(1, int(round(image.shape[1] * scale))),
         max(1, int(round(image.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (content_height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    cv2.putText(
        canvas, os.path.basename(path), (6, height - 7),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (225, 225, 225), 1,
        cv2.LINE_AA)
    return canvas


def build_contact_sheet(rows: Sequence[Tuple[str, Sequence[str]]],
                        cell_width: int = 360,
                        cell_height: int = 240) -> np.ndarray:
    if cell_width < 120 or cell_height < 100:
        raise ValueError('Contact-sheet cells are too small')
    if not rows or any(not paths for _, paths in rows):
        raise ValueError('Every comparison row requires at least one image')
    columns = max(len(paths) for _, paths in rows)
    label_width = 300
    title_height = 54
    sheet = np.full(
        (title_height + len(rows) * cell_height,
         label_width + columns * cell_width, 3),
        18, dtype=np.uint8)
    cv2.putText(
        sheet,
        'DIAGNOSIS ONLY - target atlas must not enter training',
        (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78,
        (255, 255, 255), 2, cv2.LINE_AA)
    for row_index, (label, paths) in enumerate(rows):
        top = title_height + row_index * cell_height
        cv2.rectangle(
            sheet, (0, top), (label_width, top + cell_height),
            (35, 35, 35), -1)
        wrapped = [label[index:index + 34]
                   for index in range(0, len(label), 34)]
        for line_index, line in enumerate(wrapped[:5]):
            cv2.putText(
                sheet, line, (12, top + 34 + 25 * line_index),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (235, 235, 235), 1, cv2.LINE_AA)
        for column, path in enumerate(paths):
            cell = _render_cell(path, cell_width, cell_height)
            left = label_width + column * cell_width
            sheet[top:top + cell_height, left:left + cell_width] = cell
    return sheet


def main():
    args = parse_args()
    if args.proxy_count <= 0 or args.atlas_count <= 0:
        raise ValueError('preview counts must be positive')

    rows = []
    for variant in args.proxy_variants:
        directory = os.path.join(args.proxy_preview_dir, variant)
        selected = evenly_select(discover_images(directory), args.proxy_count)
        if not selected:
            raise RuntimeError(f'No proxy previews found in {directory}')
        rows.append((f'SOURCE PROXY: {variant}', selected))

    atlas_paths = evenly_select(
        discover_images(args.atlas_preview_dir), args.atlas_count)
    if not atlas_paths:
        raise RuntimeError(
            f'No target atlas previews found in {args.atlas_preview_dir}')
    rows.append((
        'TARGET-DEV ATLAS: green=GT, red=false peak, cyan=usable',
        atlas_paths))

    sheet = build_contact_sheet(
        rows, cell_width=args.cell_width, cell_height=args.cell_height)
    output_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not cv2.imwrite(output_path, sheet):
        raise RuntimeError(f'Failed to write comparison: {output_path}')
    print(f'[out] wrote {output_path}')
    print('[policy] TARGET-DEV DIAGNOSIS ONLY; DO NOT USE ATLAS IMAGES '
          'OR CROPS FOR TRAINING')


if __name__ == '__main__':
    main()
