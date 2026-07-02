#!/usr/bin/env python3
"""
input_enhance_probe_sweep.py

Real hard-slice input-enhancement sweep for §14.4:
  - runs subthreshold_peak_probe.py under multiple deterministic preproc modes;
  - keeps the exact same ROI/background/angle gate implementation;
  - writes one compact sweep_summary.csv/json for deciding whether real
    138..161 can be revived by test-time input enhancement.

Example:
PYTHONPATH=. python3 crane_project/tools/input_enhance_probe_sweep.py \
  --config crane_project/configs/crane_symeood_degraded_cls_k1.py \
  --checkpoint work_dirs/crane_symeood_degraded_cls_k1/epoch_24.pth \
  --seq real_seq02 --start 138 --end 161 \
  --heads aux1 --require-angle --device cuda:0 \
  --out-root work_dirs/subthreshold_peak_probe/input_enhance_degraded_cls_k1_ep24_138_161

Note:
  crane_symeood_degraded_cls_k1.py is still a pending branch until its training
  checkpoint and metrics are available.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from typing import Dict, List, Sequence


DEFAULT_MODES = [
    'none',
    'linear-brighten',
    'clahe',
    'linear-clahe',
    'linear-clahe-gray',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run subthreshold peak probes across input enhancements')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--split', default='test')
    parser.add_argument('--seq', default='real_seq02')
    parser.add_argument('--start', type=int, default=138)
    parser.add_argument('--end', type=int, default=161)
    parser.add_argument('--ok-radius', type=int, default=0)
    parser.add_argument('--extra-ok-frames', default='')
    parser.add_argument('--heads', nargs='+', default=['aux1'],
                        choices=['main', 'aux1'])
    parser.add_argument('--modes', nargs='+', default=DEFAULT_MODES,
                        choices=DEFAULT_MODES)
    parser.add_argument('--linear-gain', type=float, default=2.0)
    parser.add_argument('--clahe-clip-limit', type=float, default=2.0)
    parser.add_argument('--clahe-tile-grid', type=int, default=8)
    parser.add_argument('--roi-scale', type=float, default=1.75)
    parser.add_argument('--min-roi-cells', type=int, default=3)
    parser.add_argument('--guard-cells', type=int, default=2)
    parser.add_argument('--bg-samples', type=int, default=4096)
    parser.add_argument('--neg-samples', type=int, default=64)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--dist-scale', type=float, default=0.75)
    parser.add_argument('--min-dist-px', type=float, default=12.0)
    parser.add_argument('--angle-thr-deg', type=float, default=30.0)
    parser.add_argument('--require-angle', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--python', default=sys.executable,
                        help='Python executable used to launch subthreshold_peak_probe.py')
    parser.add_argument('--out-root', required=True)
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def mode_label(mode: str) -> str:
    return 'raw' if mode == 'none' else mode.replace('-', '_')


def probe_script_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'subthreshold_peak_probe.py')


def build_command(args, mode: str, out_dir: str) -> List[str]:
    cmd = [
        args.python, probe_script_path(),
        '--config', args.config,
        '--checkpoint', args.checkpoint,
        '--data-root', args.data_root,
        '--split', args.split,
        '--seq', args.seq,
        '--start', str(args.start),
        '--end', str(args.end),
        '--ok-radius', str(args.ok_radius),
        '--heads', *args.heads,
        '--preproc', mode,
        '--linear-gain', str(args.linear_gain),
        '--clahe-clip-limit', str(args.clahe_clip_limit),
        '--clahe-tile-grid', str(args.clahe_tile_grid),
        '--roi-scale', str(args.roi_scale),
        '--min-roi-cells', str(args.min_roi_cells),
        '--guard-cells', str(args.guard_cells),
        '--bg-samples', str(args.bg_samples),
        '--neg-samples', str(args.neg_samples),
        '--alpha', str(args.alpha),
        '--dist-scale', str(args.dist_scale),
        '--min-dist-px', str(args.min_dist_px),
        '--angle-thr-deg', str(args.angle_thr_deg),
        '--device', args.device,
        '--seed', str(args.seed),
        '--out-dir', out_dir,
    ]
    if args.extra_ok_frames:
        cmd += ['--extra-ok-frames', args.extra_ok_frames]
    if args.require_angle:
        cmd.append('--require-angle')
    return cmd


def read_json(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in ('true', '1', 'yes')


def summarize_mode(mode: str, out_dir: str) -> List[Dict]:
    summary_path = os.path.join(out_dir, 'summary.json')
    frame_path = os.path.join(out_dir, 'per_frame.csv')
    summary = read_json(summary_path)
    frame_rows = read_csv_rows(frame_path)
    rows = []
    for key, group in sorted(summary.get('groups', {}).items()):
        head, role = key.split('/', 1)
        subset = [
            row for row in frame_rows
            if row.get('head') == head and row.get('role') == role
        ]
        passed = [row for row in subset if as_bool(row.get('pass_all'))]
        passed_frames = ','.join(row['frame'] for row in passed)
        best_levels = ','.join(row['best_level'] for row in passed)
        neg_samples = group.get('neg_samples') or 0
        neg_pass = group.get('neg_pass') or 0
        rows.append(dict(
            mode=mode,
            mode_label=mode_label(mode),
            out_dir=out_dir,
            head=head,
            role=role,
            n=group.get('n'),
            k=group.get('k'),
            pass_rate=group.get('pass_rate'),
            neg_samples=neg_samples,
            neg_pass=neg_pass,
            neg_pass_rate=(
                float(neg_pass) / float(neg_samples)
                if float(neg_samples) > 0 else ''),
            binom_p_nominal_alpha=group.get('binom_p_nominal_alpha'),
            binom_p_empirical_neg=group.get('binom_p_empirical_neg'),
            mean_best_roi_max=group.get('mean_best_roi_max'),
            mean_best_p=group.get('mean_best_p'),
            mean_best_loc_dist=group.get('mean_best_loc_dist'),
            passed_frames=passed_frames,
            passed_best_levels=best_levels,
        ))
    return rows


def write_csv(path: str, rows: Sequence[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        'mode', 'mode_label', 'head', 'role', 'n', 'k', 'pass_rate',
        'neg_samples', 'neg_pass', 'neg_pass_rate',
        'binom_p_nominal_alpha', 'binom_p_empirical_neg',
        'mean_best_roi_max', 'mean_best_p', 'mean_best_loc_dist',
        'passed_frames', 'passed_best_levels', 'out_dir',
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    os.makedirs(args.out_root, exist_ok=True)
    commands_path = os.path.join(args.out_root, 'commands.txt')
    all_rows = []
    commands = []
    env = os.environ.copy()
    cwd = os.getcwd()
    env['PYTHONPATH'] = cwd + os.pathsep + env.get('PYTHONPATH', '')
    env.setdefault('MPLCONFIGDIR', '/tmp/mplconfig')

    for mode in args.modes:
        out_dir = os.path.join(args.out_root, mode_label(mode))
        cmd = build_command(args, mode, out_dir)
        commands.append(' '.join(cmd))
        summary_path = os.path.join(out_dir, 'summary.json')
        if args.dry_run:
            print('[dry-run]', ' '.join(cmd))
            continue
        if args.skip_existing and os.path.exists(summary_path):
            print(f'[skip-existing] {mode}: {summary_path}')
        else:
            print(f'[run] mode={mode} out={out_dir}')
            subprocess.run(cmd, check=True, env=env)
        all_rows.extend(summarize_mode(mode, out_dir))

    with open(commands_path, 'w') as f:
        for command in commands:
            f.write(command + '\n')

    if not args.dry_run:
        write_csv(os.path.join(args.out_root, 'sweep_summary.csv'), all_rows)
        sweep_json = dict(
            config=args.config,
            checkpoint=args.checkpoint,
            split=args.split,
            seq=args.seq,
            frame_range=[args.start, args.end],
            heads=args.heads,
            modes=args.modes,
            require_angle=args.require_angle,
            angle_thr_deg=args.angle_thr_deg,
            alpha=args.alpha,
            results=all_rows,
        )
        with open(os.path.join(args.out_root, 'sweep_summary.json'), 'w') as f:
            json.dump(sweep_json, f, indent=2, ensure_ascii=False)
        print(f'[done] wrote {args.out_root}/sweep_summary.csv and .json')
        for row in all_rows:
            if row['role'] == 'dead':
                print(f"  {row['mode_label']}/{row['head']}: "
                      f"k={row['k']}/{row['n']} "
                      f"neg={row['neg_pass']}/{row['neg_samples']} "
                      f"frames={row['passed_frames']}")


if __name__ == '__main__':
    main()
