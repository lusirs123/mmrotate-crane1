#!/usr/bin/env python3
"""Run or reuse full test evaluation outputs.

Default path:
  tools/test.py -> results.pkl -> DOTA txt -> eval_crane_offline.py

If --pred-dir is given, the script skips inference and reuses an existing
ckpt_sweep final_test prediction directory, then only calls eval_crane_offline.
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
    parser.add_argument('--config', default=None)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--work-dir', default='work_dirs/re_eval_existing_test')
    parser.add_argument('--variant', default='norm')
    parser.add_argument(
        '--pred-dir',
        default=None,
        help='Reuse existing prediction dir. Accepts .../preds or .../preds/Task1_grab')
    parser.add_argument(
        '--preproc', default='linear-clahe-gray',
        choices=[
            'none', 'linear-brighten', 'clahe', 'gray-world',
            'linear-clahe', 'linear-clahe-gray',
        ])
    parser.add_argument('--linear-gain', type=float, default=2.0)
    parser.add_argument('--clahe-clip-limit', type=float, default=2.0)
    parser.add_argument('--clahe-tile-grid', type=int, default=8)
    parser.add_argument(
        '--brightness-thr',
        type=float,
        default=None,
        help='Apply TTA only when grayscale mean brightness is below this threshold')
    parser.add_argument('--center-thresh', type=float, default=15.0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--force', action='store_true',
                        help='Rerun inference even if results.pkl exists')
    return parser.parse_args()


def resolve_existing_task_dir(pred_dir):
    pred_dir = os.path.abspath(pred_dir)
    if not os.path.isdir(pred_dir):
        raise FileNotFoundError(f'--pred-dir does not exist: {pred_dir}')

    if os.path.basename(pred_dir) == 'Task1_grab':
        return pred_dir, os.path.dirname(pred_dir)

    task_dir = os.path.join(pred_dir, 'Task1_grab')
    if os.path.isdir(task_dir):
        return task_dir, pred_dir

    pkl_path = os.path.join(pred_dir, 'results.pkl')
    if os.path.exists(pkl_path):
        gt_dir = os.path.join(PROJ_ROOT, 'crane_project/data/crane_grab/test/annfiles')
        img_ids = get_val_img_ids(gt_dir)
        task_dir = pkl_to_dota(pkl_path, img_ids, pred_dir)
        return task_dir, pred_dir

    raise FileNotFoundError(
        f'No Task1_grab/ or results.pkl found under --pred-dir: {pred_dir}')


def _insert_tta_transform(pipeline, args):
    if args.preproc == 'none':
        return pipeline

    tta = dict(
        type='TestTimeNormalize',
        mode=args.preproc,
        linear_gain=args.linear_gain,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_grid=args.clahe_tile_grid,
        brightness_thr=args.brightness_thr,
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

    if args.pred_dir is not None:
        task_dir, preds_dir = resolve_existing_task_dir(args.pred_dir)
        gt_dir = os.path.join(PROJ_ROOT, 'crane_project/data/crane_grab/test/annfiles')

        print('\n' + '=' * 64)
        print(f'  FULL TEST RE-EVAL [{args.variant}]')
        print(f'  pred_dir: {preds_dir}')
        print(f'  task_dir: {task_dir}')
        print('  inference: skipped (reuse existing ckpt_sweep final_test outputs)')
        print('=' * 64)

        metrics = run_offline_eval(
            task_dir, gt_dir, mode='test', center_thresh=args.center_thresh)

        print('\n  [TEST metric dict]')
        for k, v in sorted(metrics.items()):
            print(f'    {k}: {v}')
        return metrics

    if args.config is None or args.checkpoint is None:
        raise SystemExit(
            'Either provide --pred-dir to reuse existing predictions, or provide '
            '--config and --checkpoint to run inference.')

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
    print(f'  gate_thr:   {args.brightness_thr}')
    print('=' * 64)

    metrics = run_offline_eval(
        task_dir, gt_dir, mode='test', center_thresh=args.center_thresh)

    print('\n  [TEST metric dict]')
    for k, v in sorted(metrics.items()):
        print(f'    {k}: {v}')

    return metrics


if __name__ == '__main__':
    main()
