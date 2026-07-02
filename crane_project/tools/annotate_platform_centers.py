#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
annotate_platform_centers.py - local click annotator for platform centers.

This is intentionally a local-only helper.  It writes a portable JSON file with
sequence/frame ids and image-coordinate centers; the server can consume that
file with platform_context_probe.py --manual-platform-json without needing a GUI
or matching local image paths.

Example:
  python3 crane_project/tools/annotate_platform_centers.py \
    --split test --seq real_seq02 \
    --frames 137,144,150,156,162,169 \
    --out work_dirs/crane_symeood_k1/manual_platform_centers_real_seq02.json
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import cv2
import numpy as np

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from crane_project.tools import mcml_diag as diag  # noqa: E402
from crane_project.tools.platform_context_probe import ann_to_poly  # noqa: E402


HELP_TEXT = (
    'left click: set platform center | n/space: next | p: prev | '
    'u/backspace: clear | s: save | q/esc: save+quit'
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Locally click platform centers and export portable JSON.')
    parser.add_argument('--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--split', default='test')
    parser.add_argument('--seq', required=True)
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--frames', default='',
                        help='Comma-separated frame ids; overrides start/end.')
    parser.add_argument('--out', required=True,
                        help='Portable JSON consumed by --manual-platform-json.')
    parser.add_argument('--resume', default='',
                        help='Existing annotation JSON to continue editing.')
    parser.add_argument('--window-scale', type=float, default=0.9,
                        help='Initial display scale for large images.')
    return parser.parse_args()


def parse_frame_ids(args) -> List[int]:
    if args.frames.strip():
        return [int(x) for x in args.frames.split(',') if x.strip()]
    if args.start is None or args.end is None:
        raise RuntimeError('Use either --frames or both --start/--end.')
    return list(range(int(args.start), int(args.end) + 1))


def load_existing(path: str) -> Dict[int, Dict]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        data = json.load(f)
    frames = data.get('frames', data)
    rows: Dict[int, Dict] = {}
    if isinstance(frames, dict):
        for key, value in frames.items():
            rows[int(key)] = value
    elif isinstance(frames, list):
        for item in frames:
            rows[int(item['frame'])] = item
    return rows


def write_json(path: str, seq: str, split: str, rows: Dict[int, Dict]):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    payload = {
        'format': 'manual_platform_centers_v1',
        'coordinate_space': 'original_image_pixels',
        'seq': seq,
        'split': split,
        'frames': {
            str(fid): {
                'frame': int(fid),
                'seq': seq,
                'split': split,
                'center': [
                    float(rows[fid]['center'][0]),
                    float(rows[fid]['center'][1]),
                ],
            }
            for fid in sorted(rows)
            if rows[fid].get('center') is not None
        },
    }
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
        f.write('\n')


def draw_frame(img, beam, center: Optional[List[float]], frame_id: int, scale: float):
    vis = img.copy()
    if beam is not None:
        cv2.polylines(vis, [np.round(beam).astype(np.int32)], True, (255, 0, 0), 2)
        c = np.round(beam.mean(axis=0)).astype(int)
        cv2.putText(vis, 'beam', tuple(c), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 0, 0), 2, cv2.LINE_AA)
    if center is not None:
        c = tuple(np.round(center).astype(int))
        cv2.drawMarker(vis, c, (0, 255, 0), markerType=cv2.MARKER_CROSS,
                       markerSize=28, thickness=2)
        cv2.circle(vis, c, 8, (0, 255, 0), 2)
        cv2.putText(vis, 'platform center', (c[0] + 8, c[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, f'frame {frame_id}', (20, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, HELP_TEXT, (20, max(64, vis.shape[0] - 24)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    if scale != 1.0:
        vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return vis


def main():
    args = parse_args()
    frame_ids = parse_frame_ids(args)
    annotations = load_existing(args.resume or args.out)
    window = 'annotate platform center'
    clicked = {'xy': None}
    scale = float(args.window_scale)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked['xy'] = [float(x) / scale, float(y) / scale]

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    idx = 0
    while 0 <= idx < len(frame_ids):
        fid = int(frame_ids[idx])
        img_path, ann_path = diag.find_files(args.data_root, args.split, args.seq, fid)
        if img_path is None:
            print(f'[skip] missing image: {args.split}/{args.seq}/{fid:05d}')
            idx += 1
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f'[skip] unreadable image: {img_path}')
            idx += 1
            continue
        beam = ann_to_poly(ann_path) if ann_path else None
        clicked['xy'] = None

        while True:
            center = annotations.get(fid, {}).get('center')
            if clicked['xy'] is not None:
                center = clicked['xy']
            cv2.imshow(window, draw_frame(img, beam, center, fid, scale))
            key = cv2.waitKey(30) & 0xFF
            if clicked['xy'] is not None:
                annotations[fid] = {'center': clicked['xy']}
                clicked['xy'] = None
            if key in (ord('n'), ord(' '), 13):
                idx += 1
                break
            if key == ord('p'):
                idx = max(0, idx - 1)
                break
            if key in (ord('u'), 8):
                annotations.pop(fid, None)
            if key == ord('s'):
                write_json(args.out, args.seq, args.split, annotations)
                print(f'[save] {args.out}')
            if key in (ord('q'), 27):
                write_json(args.out, args.seq, args.split, annotations)
                print(f'[save] {args.out}')
                cv2.destroyAllWindows()
                return

    write_json(args.out, args.seq, args.split, annotations)
    print(f'[save] {args.out}')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
