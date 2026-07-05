#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
annotate_platform_polygons.py - local four-point platform annotator.

Use this on a local machine with image access and a GUI.  It exports portable
image-coordinate JSON that can be copied to a server and consumed by
platform_context_probe.py --manual-platform-json.

Example:
  python3 crane_project/tools/annotate_platform_polygons.py \
    --split test --seq real_seq02 \
    --frames 137,144,150,156,162,169 \
    --out work_dirs/crane_symeood_k1/manual_platform_polygons_real_seq02.json

Controls:
  left click       add platform corner
  right click/u    undo last point
  r                reset current frame
  s                save current four-point platform
  n/space/enter    next frame
  p                previous frame
  d                delete current frame annotation
  +/-              zoom in/out
  q/esc            save and quit
"""

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional

import cv2
import numpy as np

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)


HELP_TEXT = (
    'L-click add | R-click/u undo | r reset | s save | n next | p prev | '
    'd delete | +/- zoom | q quit'
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Locally annotate full platform polygons and export JSON.')
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


def normalize_polygon(value) -> Optional[List[List[float]]]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 8:
        arr = arr.reshape(4, 2)
    if arr.shape != (4, 2):
        return None
    return [[float(x), float(y)] for x, y in arr.tolist()]


def polygon_center(poly) -> np.ndarray:
    return np.asarray(poly, dtype=np.float32).reshape(-1, 2).mean(axis=0)


def ann_to_poly(ann_path: str) -> Optional[np.ndarray]:
    if not ann_path or not os.path.exists(ann_path):
        return None
    with open(ann_path, 'r') as f:
        line = f.readline().strip()
    parts = line.split()
    if len(parts) < 8:
        return None
    return np.asarray([float(x) for x in parts[:8]], dtype=np.float32).reshape(4, 2)


def find_files(data_root: str, split: str, seq: str, frame_id: int):
    img_split = 'train' if split == 'train_sim' else split
    ann_split = 'train_sim' if split == 'train_sim' else split
    img_dir = os.path.join(data_root, img_split, 'images')
    ann_dir = os.path.join(data_root, ann_split, 'annfiles')
    stem = f'{seq}_{int(frame_id):05d}'

    img_path = None
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
        p = os.path.join(img_dir, stem + ext)
        if os.path.exists(p):
            img_path = p
            break
    if img_path is None:
        matches = sorted(glob.glob(os.path.join(img_dir, stem + '.*')))
        img_path = matches[0] if matches else None

    ann_path = os.path.join(ann_dir, stem + '.txt')
    if not os.path.exists(ann_path):
        ann_path = None
    return img_path, ann_path


def annotation_key(split: str, seq: str, frame_id: int) -> str:
    return f'{split}/{seq}/{int(frame_id):05d}'


def row_key(row: Dict, default_split: str, default_seq: str) -> Optional[str]:
    if row is None:
        return None
    frame = row.get('frame', row.get('frame_id'))
    if frame is None:
        return None
    split = str(row.get('split', default_split))
    seq = str(row.get('seq', row.get('sequence', default_seq)))
    return annotation_key(split, seq, int(frame))


def load_existing(path: str, split: str, seq: str) -> Dict[str, Dict]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        data = json.load(f)
    frames = data.get('frames', data)
    rows: Dict[str, Dict] = {}
    if isinstance(frames, dict):
        iterator = frames.items()
    else:
        iterator = ((item.get('frame'), item) for item in frames)

    for key, value in iterator:
        if value is None:
            continue
        row = dict(value)
        if 'frame' not in row:
            try:
                row['frame'] = int(str(key).split('/')[-1])
            except ValueError:
                continue
        row.setdefault('split', split)
        row.setdefault('seq', seq)
        poly = normalize_polygon(
            row.get('platform_corners')
            or row.get('polygon')
            or row.get('corners'))
        if poly is None:
            continue
        k = row_key(row, split, seq)
        if k is None:
            continue
        rows[k] = {
            'frame': int(row['frame']),
            'seq': str(row.get('seq', seq)),
            'split': str(row.get('split', split)),
            'platform_corners': poly,
        }
    return rows


def write_json(path: str, rows: Dict[str, Dict]):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    payload = {
        'format': 'manual_platform_polygons_v2',
        'coordinate_space': 'original_image_pixels',
        'frames': {},
    }
    for key in sorted(rows):
        row = rows[key]
        poly = normalize_polygon(row.get('platform_corners'))
        if poly is None:
            continue
        center = np.asarray(poly, dtype=np.float32).mean(axis=0).tolist()
        split = str(row['split'])
        seq = str(row['seq'])
        fid = int(row['frame'])
        payload['frames'][annotation_key(split, seq, fid)] = {
            'frame': fid,
            'seq': seq,
            'split': split,
            'platform_corners': poly,
            'center': [float(center[0]), float(center[1])],
        }

    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
        f.write('\n')


def save_current_points(args, annotations: Dict[str, Dict], fid: int,
                        points: List[List[float]]) -> bool:
    if len(points) != 4:
        return False
    key = annotation_key(args.split, args.seq, fid)
    annotations[key] = {
        'frame': int(fid),
        'seq': args.seq,
        'split': args.split,
        'platform_corners': normalize_polygon(points),
    }
    write_json(args.out, annotations)
    print(f'[save] {args.split}/{args.seq}/{fid:05d} -> {os.path.abspath(args.out)}')
    return True


def draw_poly(vis: np.ndarray, points: List[List[float]], color, closed: bool):
    pts = np.asarray(points, dtype=np.float32)
    pts_i = np.round(pts).astype(np.int32)
    for i, pt in enumerate(pts_i):
        cv2.circle(vis, tuple(pt), 5, color, -1)
        cv2.putText(vis, str(i + 1), (int(pt[0]) + 8, int(pt[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    if len(pts_i) >= 2:
        cv2.polylines(vis, [pts_i], isClosed=closed, color=color, thickness=2)


def draw_frame(img: np.ndarray,
               beam: Optional[np.ndarray],
               saved_poly: Optional[List[List[float]]],
               current_points: List[List[float]],
               frame_id: int,
               scale: float):
    vis = img.copy()

    if beam is not None:
        cv2.polylines(vis, [np.round(beam).astype(np.int32)], True,
                      (255, 0, 0), 2)
        c = np.round(polygon_center(beam)).astype(int)
        cv2.putText(vis, 'beam GT', tuple(c), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 0, 0), 2, cv2.LINE_AA)

    if saved_poly:
        draw_poly(vis, saved_poly, (0, 200, 0), closed=True)
        c = np.round(np.asarray(saved_poly, dtype=np.float32).mean(axis=0)).astype(int)
        cv2.putText(vis, 'saved platform', (int(c[0]) + 8, int(c[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 0), 2, cv2.LINE_AA)

    if current_points:
        draw_poly(vis, current_points, (0, 255, 255), closed=len(current_points) == 4)
        if len(current_points) == 4:
            cv2.putText(vis, 'press s to save this platform polygon',
                        (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(vis, f'frame {frame_id}', (20, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, f'points {len(current_points)}/4', (20, 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, HELP_TEXT, (20, max(112, vis.shape[0] - 24)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

    if scale != 1.0:
        vis = cv2.resize(vis, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    return vis


def main():
    args = parse_args()
    frame_ids = parse_frame_ids(args)
    annotations = load_existing(args.resume or args.out, args.split, args.seq)
    window = 'annotate full platform polygon'
    state = {'points': [], 'scale': float(args.window_scale)}

    def on_mouse(event, x, y, flags, param):
        scale = max(float(state['scale']), 1e-6)
        if event == cv2.EVENT_LBUTTONDOWN and len(state['points']) < 4:
            state['points'].append([float(x) / scale, float(y) / scale])
        elif event == cv2.EVENT_RBUTTONDOWN and state['points']:
            state['points'].pop()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    idx = 0
    while 0 <= idx < len(frame_ids):
        fid = int(frame_ids[idx])
        img_path, ann_path = find_files(args.data_root, args.split, args.seq, fid)
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
        state['points'] = []

        while True:
            key = annotation_key(args.split, args.seq, fid)
            saved_poly = annotations.get(key, {}).get('platform_corners')
            vis = draw_frame(
                img, beam, saved_poly, state['points'], fid, state['scale'])
            cv2.imshow(window, vis)
            key = cv2.waitKey(30) & 0xFF

            if key in (ord('u'), 8):
                if state['points']:
                    state['points'].pop()
            elif key == ord('r'):
                state['points'] = []
            elif key == ord('d'):
                annotations.pop(annotation_key(args.split, args.seq, fid), None)
                state['points'] = []
            elif key == ord('s'):
                if not save_current_points(args, annotations, fid, state['points']):
                    print(f'[warn] frame {fid}: need 4 points, got {len(state["points"])}')
            elif key in (ord('+'), ord('=')):
                state['scale'] = min(state['scale'] * 1.15, 4.0)
            elif key in (ord('-'), ord('_')):
                state['scale'] = max(state['scale'] / 1.15, 0.1)
            elif key in (ord('n'), ord(' '), 13):
                save_current_points(args, annotations, fid, state['points'])
                idx += 1
                break
            elif key == ord('p'):
                save_current_points(args, annotations, fid, state['points'])
                idx = max(0, idx - 1)
                break
            elif key in (ord('q'), 27):
                save_current_points(args, annotations, fid, state['points'])
                write_json(args.out, annotations)
                print(f'[save] {os.path.abspath(args.out)}')
                cv2.destroyAllWindows()
                return

    write_json(args.out, annotations)
    print(f'[save] {os.path.abspath(args.out)}')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
