#!/usr/bin/env python3
"""Run full test evaluation with optional test-time normalization.

This script follows the existing final-test path:
  tools/test.py -> results.pkl -> DOTA txt -> eval_crane_offline.py

It is for one-shot diagnosis of whether test-time normalization improves
standard test metrics such as TDR_w10 and MCML_mean.
"""

import argparse
import os
import shutil
import subprocess
import sys

from mmcv import Config

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

from ckpt_sweep import get_val_img_ids, pkl_to_dota, run_offline_eval  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description='Full test eval with TTA normalization')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--variant', default='norm')
    parser.add_argument(
        '--preproc', default='linear-clahe-gray',
        choices=[
            'none', 'linear-brighten', 'clahe', 'gray-world',
            'linear-clahe', 'linear-clahe-gray',
        ])
    parser.add_argument('--linear-gain', type=float, default=2.0)
    parser.add_argument('--clahe-clip-limit', type=float, default=2.0)
    parser.add_argument('--clahe-tile-grid', type=int, default=8)
    parser.add_argument('--center-thresh', type=float, default=15.0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--force', action='store_true',
                        help='Rerun inference even if results.pkl exists')
    return parser.parse_args()


def _insert_tta_transform(pipeline, args):
    if args.preproc == 'none':
        return pipeline

    tta = dict(
        type='TestTimeNormalize',
        mode=args.preproc,
        linear_gain=args.linear_gain,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_grid=args.clahe_tile_grid,
    )

    new_pipeline = []
    inserted = False
    for step in pipeline:
        new_pipeline.append(step)
        if step.get('type') == 'LoadImageFromFile':
            new_pipeline.append(tta)
            inserted = True
    if not inserted:
        new_pipeline.insert(0, tta)
    return new_pipeline


def make_eval_config(args, out_dir):
    cfg = Config.fromfile(args.config)

    if args.preproc != 'none':
        cfg.test_pipeline = _insert_tta_transform(cfg.test_pipeline, args)
        cfg.data.test.pipeline = _insert_tta_transform(cfg.data.test.pipeline, args)

    cfg.work_dir = out_dir
    cfg_path = os.path.join(out_dir, f'{args.variant}_eval_config.py')
    os.makedirs(out_dir, exist_ok=True)
    cfg.dump(cfg_path)
    return cfg_path


def main():
    args = parse_args()

    out_dir = os.path.join(args.work_dir, args.variant)
    preds_dir = os.path.join(out_dir, 'preds')
    os.makedirs(preds_dir, exist_ok=True)

    cfg_path = make_eval_config(args, out_dir)
    pkl_path = os.path.join(preds_dir, 'results.pkl')

    if args.force and os.path.exists(pkl_path):
        os.remove(pkl_path)
    if args.force:
        task_dir = os.path.join(preds_dir, 'Task1_grab')
        if os.path.isdir(task_dir):
            shutil.rmtree(task_dir)

    if not os.path.exists(pkl_path):
        cmd = [
            sys.executable,
            os.path.join(PROJ_ROOT, 'tools/test.py'),
            cfg_path,
            args.checkpoint,
            '--work-dir',
            out_dir,
            '--out',
            pkl_path,
            '--gpu-ids',
            str(args.gpu),
        ]
        print('[inference] ' + ' '.join(cmd))
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        result = subprocess.run(
            cmd, cwd=PROJ_ROOT, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            raise SystemExit(result.returncode)
    else:
        print(f'[skip] existing inference result: {pkl_path}')

    gt_dir = os.path.join(PROJ_ROOT, 'crane_project/data/crane_grab/test/annfiles')
    img_ids = get_val_img_ids(gt_dir)
    task_dir = pkl_to_dota(pkl_path, img_ids, preds_dir)

    print('\n' + '=' * 64)
    print(f'  FULL TEST EVAL [{args.variant}]')
    print(f'  config:     {args.config}')
    print(f'  checkpoint: {args.checkpoint}')
    print(f'  preproc:    {args.preproc}')
    print('=' * 64)

    metrics = run_offline_eval(
        task_dir, gt_dir, mode='test', center_thresh=args.center_thresh)

    print('\n  [TEST metric dict]')
    for k, v in sorted(metrics.items()):
        print(f'    {k}: {v}')

    return metrics


if __name__ == '__main__':
    main()
