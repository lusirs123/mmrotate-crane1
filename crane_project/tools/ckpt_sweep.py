#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ckpt_sweep.py — 训练后 checkpoint 离线扫描选权工具

【核心方案：两阶段约束式选择】

  第一阶段（硬约束筛选）：
    1. Weighted_R_center >= W_max - delta   （位置精度不显著退化）
    2. MCML_max <= MCML_limit               （无长时间连续漏检）
    不满足任一条的 checkpoint 直接排除。

  第二阶段（软指标排序）：
    在可行域内计算控制质量评分：
      Score = 0.35 * TDR + 0.25 * R_center + 0.20 * ACI + 0.20 * AngleScore
    选择 Score 最大的 checkpoint。

  Fallback（两级降级）：
    1. 若无 checkpoint 同时满足 R_center 近最优约束与 MCML 约束，
       但存在 MCML 达标 checkpoint，则保持 MCML 生存红线，放宽 R_center 近最优约束，
       在 MCML 达标集合内按软评分选择。
    2. 若所有 checkpoint 均 MCML 超限，才退化为选 MCML 最小者；
       若并列选 R_center 最高者，若仍并列选 A_RMSE 最低者。

【指标跨域说明】
  TDR 和 R_center 均采用跨域加权：0.7 * sim + 0.3 * real，与训练配置一致。
  ACI 仅使用 sim 域（sim 有可靠角度标注，ACI 在 sim 域才有对照基准）。
  A-RMSE 仅使用 sim 域（real 域无角度真值）。

【Checkpoint 扫描范围】
  默认只扫描 epoch_*.pth（固定间隔保存的 checkpoint）。
  avg_*.pth 和 best_*.pth 默认不纳入，需显式传入 --include-avg / --include-best。

【输出文件】
  {work_dir}/ckpt_sweep/sweep_results.json       — 全部 checkpoint 原始指标 + 派生选择指标
  {work_dir}/ckpt_sweep/selected_checkpoint.txt   — 最优权重绝对路径（一行）
  {work_dir}/ckpt_sweep/final_test_{ckpt_name}/   — （可选）test 集最终评估结果

【工作流程】
  1. 遍历 work_dir 下所有 checkpoint（可指定 epoch 列表）
  2. 对每个 checkpoint：
     a. 调用 test.py 在 val 集上推理，导出 pickle
     b. 将 pickle 转换为 DOTA 多边形文本格式
     c. 调用 eval_crane_offline 计算时序指标（mode=test 获取全部指标）
  3. 硬约束筛选 + 软评分排序，推荐最优权重
  4. 保存 sweep_results.json 和 selected_checkpoint.txt
  5. （可选）用最优权重在 test 集上跑最终评估

【用法示例】
  # 扫描全部 epoch checkpoint（默认）
  python crane_project/tools/ckpt_sweep.py \\
      --config crane_project/configs/crane_symeood_m2.py \\
      --work-dir work_dirs/crane_symeood_m2

  # 只扫描指定 epoch
  python crane_project/tools/ckpt_sweep.py \
      --config crane_project/configs/crane_symeood_m2.py \
      --work-dir work_dirs/crane_symeood_m2 \
      --epochs 16 18 20 22 24 \
      --gpus 0 1 2

python crane_project/tools/ckpt_sweep.py \
      --config crane_project/configs/crane_symeood_baseline.py \
      --work-dir work_dirs/crane_symeood_baseline \
      --epochs 16 18 20 22 24 \
      --gpus 0 1 2
      
  # 也纳入 avg 和 best 权重
  python crane_project/tools/ckpt_sweep.py \\
      --config crane_project/configs/crane_symeood_m2.py \\
      --work-dir work_dirs/crane_symeood_m2 \\
      --include-avg --include-best

  # 扫描 BrightAug 并自动在 test 集上跑最终评估
  python crane_project/tools/ckpt_sweep.py \
      --config crane_project/configs/crane_symeood_k1_brightaug.py \
      --work-dir work_dirs/crane_symeood_k1_brightaug \
      --epochs 16 18 20 22 24 \
      --gpus 0 1 2 \
      --run-final-test
"""

import argparse
import glob
import hashlib
import json
import math
import os
import pickle
import re
import subprocess
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# 项目根目录
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))
from eval_crane_offline import CraneOfflineEvaluator, METRIC_PROTOCOL_VERSION


def inference_subprocess_env(gpu=None):
    """Resolve MMRotate from this checkout, as tools/dist_train.sh does."""
    env = os.environ.copy()
    existing = env.get('PYTHONPATH')
    env['PYTHONPATH'] = (PROJ_ROOT + os.pathsep + existing
                         if existing else PROJ_ROOT)
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    return env


# =====================================================================
# 0. 选择协议配置
# =====================================================================

SELECTION_CONFIG = {
    # ---- 硬约束 ----
    'r_center_delta': 0.005,     # 允许比最优 R_center 低 0.5%
    'mcml_limit': 5,             # 最大连续漏检帧数上限
    # ---- 跨域加权（与训练配置一致） ----
    'weight_sim': 0.7,
    'weight_real': 0.3,
    # ---- 软评分权重 ----
    'soft_w_tdr': 0.35,
    'soft_w_r_center': 0.25,
    'soft_w_aci': 0.20,
    'soft_w_angle': 0.20,
}


# =====================================================================
# 1. Checkpoint 发现
# =====================================================================

def find_checkpoints(work_dir, epochs=None, include_avg=False, include_best=False):
    """查找 work_dir 下的 checkpoint 文件。

    返回 OrderedDict[名称, 绝对路径]，按 epoch 升序。
    avg_* 和 best_* 默认不纳入扫描（需显式开启），避免混入旧指标选择产生的权重。
    """
    ckpts = OrderedDict()

    for f in sorted(glob.glob(os.path.join(work_dir, 'epoch_*.pth'))):
        m = re.match(r'epoch_(\d+)\.pth', os.path.basename(f))
        if m:
            ep = int(m.group(1))
            if epochs is None or ep in epochs:
                ckpts[f'epoch_{ep}'] = os.path.abspath(f)

    if include_avg:
        for f in sorted(glob.glob(os.path.join(work_dir, 'avg_*.pth'))):
            name = os.path.splitext(os.path.basename(f))[0]
            ckpts[name] = os.path.abspath(f)

    if include_best:
        for f in sorted(glob.glob(os.path.join(work_dir, 'best_*.pth'))):
            name = os.path.splitext(os.path.basename(f))[0]
            ckpts[name] = os.path.abspath(f)

    return ckpts


# =====================================================================
# 2. 推理 + 格式转换
# =====================================================================

def get_val_img_ids(val_ann_dir):
    """从 val 标注目录获取有序 img_id 列表（与 CraneDataset 一致）。"""
    txt_files = sorted(path for path in glob.glob(
        os.path.join(val_ann_dir, '*.txt'))
        if not os.path.basename(path).startswith('._'))
    return [os.path.splitext(os.path.basename(f))[0] for f in txt_files]


def prediction_provenance_path(pkl_path):
    return pkl_path + '.provenance.json'


def dota_export_sha256(task_dir, img_ids):
    digest = hashlib.sha256()
    for img_id in img_ids:
        path = os.path.join(task_dir, img_id + '.txt')
        digest.update((img_id + '.txt').encode('utf-8'))
        digest.update(b'\0')
        with open(path, 'rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


def prediction_provenance(config, checkpoint, pkl_path, ann_dir, split):
    """Describe the exact inputs and output of one completed inference."""
    return dict(
        protocol='crane_prediction_provenance_v1', split=split,
        config=os.path.abspath(config), config_sha256=sha256_file(config),
        checkpoint=os.path.abspath(checkpoint),
        checkpoint_sha256=sha256_file(checkpoint),
        annotations_sha256=annotation_set_sha256(ann_dir),
        results_pkl=os.path.abspath(pkl_path),
        results_pkl_sha256=sha256_file(pkl_path))


def check_or_record_prediction(config, checkpoint, pkl_path, ann_dir, split,
                               generated=False):
    """Never infer a cached PKL's generating checkpoint from its directory."""
    provenance_path = prediction_provenance_path(pkl_path)
    if not os.path.isfile(pkl_path):
        raise RuntimeError('Inference produced no prediction PKL: ' + pkl_path)
    expected = prediction_provenance(
        config, checkpoint, pkl_path, ann_dir, split)
    if generated:
        # An old sidecar beside a new output is an identity conflict.
        with open(provenance_path, 'x', encoding='utf-8') as stream:
            json.dump(expected, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
    else:
        if not os.path.isfile(provenance_path):
            raise RuntimeError('Cached prediction lacks generation provenance: '
                               + pkl_path)
        with open(provenance_path, 'r', encoding='utf-8') as stream:
            observed = json.load(stream)
        if observed != expected:
            raise RuntimeError('Cached prediction provenance mismatch: '
                               + pkl_path)
    return pkl_path


def run_test_on_val(config, checkpoint, sweep_dir, ckpt_name, gpu=None):
    """调用 test.py 在 val 集上推理，返回 pickle 路径。"""
    preds_dir = os.path.join(sweep_dir, ckpt_name, 'preds')
    os.makedirs(preds_dir, exist_ok=True)
    pkl_path = os.path.join(preds_dir, 'results.pkl')
    val_ann_dir = os.path.join(
        PROJ_ROOT, 'crane_project/data/crane_grab/val/annfiles')

    if os.path.exists(pkl_path):
        check_or_record_prediction(
            config, checkpoint, pkl_path, val_ann_dir, 'source_val')
        print(f'  [跳过] 已有推理结果: {pkl_path}')
        return pkl_path
    if os.path.exists(prediction_provenance_path(pkl_path)):
        raise RuntimeError('Prediction provenance exists without PKL: '
                           + pkl_path)

    tmp_work_dir = os.path.join(sweep_dir, ckpt_name)

    cmd = [
        sys.executable,
        os.path.join(PROJ_ROOT, 'tools/test.py'),
        config, checkpoint,
        '--work-dir', tmp_work_dir,
        '--out', pkl_path,
        '--cfg-options',
        'data.test.ann_file=val/annfiles/',
        'data.test.img_prefix=val/images/',
    ]

    print(f'  [推理] test.py -> {pkl_path}')
    env = inference_subprocess_env(gpu)
    if gpu is not None:
        print(f'  [GPU] {ckpt_name} 使用 GPU {gpu}')
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=PROJ_ROOT, timeout=1200, env=env,
    )

    if result.returncode != 0:
        print(f'  [错误] test.py 失败 (exit {result.returncode})')
        for line in result.stderr.strip().split('\n')[-5:]:
            print(f'    {line}')
        return None

    return check_or_record_prediction(
        config, checkpoint, pkl_path, val_ann_dir, 'source_val',
        generated=True)


def pkl_to_dota(pkl_path, img_ids, output_dir):
    """将 pickle 预测结果转换为 DOTA 多边形文本格式。

    pickle 内容：list[per_image]，每个元素为 list[per_class_ndarray]。
    单类 grab 对应 results[i][0]，shape=(K, 6)：[cx, cy, w, h, theta, score]。
    """
    import cv2
    task_dir = os.path.join(output_dir, 'Task1_grab')
    export_identity_path = task_dir + '.provenance.json'
    source_identity = dict(
        protocol='crane_dota_export_provenance_v1',
        results_pkl_sha256=sha256_file(pkl_path),
        frame_ids=img_ids)
    if os.path.isdir(task_dir) and glob.glob(os.path.join(task_dir, '*.txt')):
        observed = {os.path.splitext(os.path.basename(path))[0]
                    for path in glob.glob(os.path.join(task_dir, '*.txt'))
                    if not os.path.basename(path).startswith('._')}
        if observed != set(img_ids):
            raise RuntimeError('Incomplete cached DOTA predictions: '
                               + task_dir)
        if not os.path.isfile(export_identity_path):
            raise RuntimeError('Cached DOTA export lacks provenance: '
                               + task_dir)
        with open(export_identity_path, 'r', encoding='utf-8') as stream:
            recorded = json.load(stream)
        expected = dict(source_identity,
                        dota_sha256=dota_export_sha256(task_dir, img_ids))
        if recorded != expected:
            raise RuntimeError('Cached DOTA export provenance mismatch: '
                               + task_dir)
        print(f'  [跳过] 已有 DOTA 预测: {task_dir} ({len(observed)} 文件)')
        return task_dir
    if os.path.exists(export_identity_path):
        raise RuntimeError('DOTA provenance exists without complete export: '
                           + task_dir)

    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
    if len(results) != len(img_ids):
        raise ValueError(
            'Prediction/annotation count mismatch: {} versus {}'.format(
                len(results), len(img_ids)))

    os.makedirs(task_dir, exist_ok=True)
    count = 0

    for idx, img_id in enumerate(img_ids):
        if idx >= len(results):
            break
        pred_bboxes = results[idx][0]
        out_path = os.path.join(task_dir, f'{img_id}.txt')

        with open(out_path, 'w') as f:
            if pred_bboxes.shape[0] > 0:
                for bbox in pred_bboxes:
                    cx, cy, w, h, theta, score = bbox[:6]
                    rect = (
                        (float(cx), float(cy)),
                        (float(w), float(h)),
                        float(np.degrees(theta)),
                    )
                    pts = cv2.boxPoints(rect).flatten()
                    coords = ' '.join([f'{p:.2f}' for p in pts])
                    f.write(f'{coords} {float(score):.4f}\n')
                    count += 1

    print(f'  [转换] {count} 个预测框 -> {task_dir}')
    source_identity['dota_sha256'] = dota_export_sha256(task_dir, img_ids)
    with open(export_identity_path, 'x', encoding='utf-8') as stream:
        json.dump(source_identity, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    return task_dir


def process_checkpoint(ckpt_name, ckpt_path, args, sweep_dir, img_ids, val_ann_dir, gpu):
    """处理单个 checkpoint：val 推理、格式转换、离线评估。"""
    print(f'\n{"─" * 60}')
    print(f'  [{ckpt_name}] 开始处理...')
    print(f'{"─" * 60}')

    if args.reuse_preds_from:
        preds_dir = os.path.join(args.reuse_preds_from, ckpt_name, 'preds')
        pkl_path = os.path.join(preds_dir, 'results.pkl')
        task_dir = os.path.join(preds_dir, 'Task1_grab')
        if not os.path.isfile(pkl_path):
            raise FileNotFoundError(pkl_path)
        with open(pkl_path, 'rb') as stream:
            if len(pickle.load(stream)) != len(img_ids):
                raise ValueError('Cached result count does not match VAL: '
                                 + ckpt_name)
        observed = {os.path.splitext(name)[0] for name in os.listdir(
            task_dir) if name.endswith('.txt') and not name.startswith('._')}
        if observed != set(img_ids):
            raise ValueError('Cached prediction frame keys do not match VAL: '
                             + ckpt_name)
    else:
        pkl_path = run_test_on_val(
            args.config, ckpt_path, sweep_dir, ckpt_name, gpu=gpu)
        if pkl_path is None:
            raise RuntimeError('VAL inference failed: ' + ckpt_name)
        preds_dir = os.path.join(sweep_dir, ckpt_name, 'preds')
        task_dir = pkl_to_dota(pkl_path, img_ids, preds_dir)

    metrics = run_offline_eval(
        task_dir, val_ann_dir,
        mode='test',
        center_thresh=args.center_thresh,
    )

    return ckpt_name, {
        'checkpoint': ckpt_path,
        'checkpoint_sha256': sha256_file(ckpt_path),
        'results_pkl': os.path.abspath(pkl_path),
        'results_pkl_sha256': sha256_file(pkl_path),
        'metrics': metrics,
        'gpu': gpu,
    }


# =====================================================================
# 3. 离线评估（直接调用，不走 subprocess）
# =====================================================================

def run_offline_eval(pred_dir, gt_dir, mode='test', center_thresh=15.0):
    """运行离线评估，返回指标字典。

    使用 mode='test' 获取全部时序指标（TDR/MCML/MRF/DFR/ACI），
    即使数据来源是 val 集也无影响——evaluator 只关心帧间时序关系。
    """
    evaluator = CraneOfflineEvaluator(
        mode=mode,
        center_thresh_px=center_thresh,
    )
    evaluator.extract_from_dirs(gt_dir=gt_dir, pred_dir=pred_dir)
    metrics = evaluator.compute_metrics()
    return metrics


# =====================================================================
# 4. 指标提取辅助函数
# =====================================================================

def extract_metrics(metrics, config):
    """从离线指标字典中提取归一化的各项数值。"""
    required = (
        'real/TDR_w10(%)', 'sim/TDR_w10(%)',
        'real/R_center(%)', 'sim/R_center(%)',
        'sim/ACI', 'real/MCML_max(frames)',
        'sim/MCML_max(frames)', 'sim/A-RMSE(deg)')
    missing = [key for key in required if key not in metrics]
    if missing:
        raise ValueError('Incomplete offline metrics: ' + ', '.join(missing))
    nonfinite = [key for key in required
                 if not math.isfinite(float(metrics[key]))]
    if nonfinite:
        raise ValueError('Non-finite offline metrics: ' + ', '.join(nonfinite))
    tdr_real = metrics.get('real/TDR_w10(%)', None)
    tdr_sim = metrics.get('sim/TDR_w10(%)', None)
    if tdr_real is not None and tdr_sim is not None:
        tdr = (config['weight_sim'] * tdr_sim + config['weight_real'] * tdr_real) / 100.0
    elif tdr_real is not None:
        tdr = tdr_real / 100.0
    elif tdr_sim is not None:
        tdr = tdr_sim / 100.0
    else:
        tdr = 0.0

    r_real = metrics.get('real/R_center(%)', 0) / 100.0
    r_sim = metrics.get('sim/R_center(%)', 0) / 100.0
    r_center = (config['weight_sim'] * r_sim + config['weight_real'] * r_real)
    aci = metrics.get('sim/ACI', 0.0)
    mcml = max(
        metrics.get('real/MCML_max(frames)', 0),
        metrics.get('sim/MCML_max(frames)', 0),
    )
    armse = metrics.get('sim/A-RMSE(deg)', 90.0)
    angle_score = 1.0 - min(armse, 90.0) / 90.0
    return {
        'tdr': tdr,
        'r_center': r_center,
        'aci': aci,
        'mcml': mcml,
        'armse': armse,
        'angle_score': angle_score,
    }


# =====================================================================
# 5. 两阶段约束式选择
# =====================================================================

def select_best_checkpoint(all_results, config):
    """两阶段约束式 checkpoint 选择。

    第一阶段：硬约束筛选
      1. Weighted_R_center >= W_max - delta
      2. MCML_max <= mcml_limit

    第二阶段：可行域内软评分
      Score = 0.35 * TDR + 0.25 * R_center + 0.20 * ACI + 0.20 * AngleScore

    Fallback：
    1) 若无 checkpoint 同时满足 R_center 近最优约束与 MCML 约束，
       但存在 MCML 达标 checkpoint，则保持 MCML 生存红线，放宽 R_center 近最优约束，
       在 MCML 达标集合内按软评分选择；
    2) 若所有 checkpoint 均 MCML 超限，才退化为选 MCML 最小者，
       若并列选 R_center 最高者，若仍并列选 A_RMSE 最低者。

    返回 (best_name, best_data, selection_info)。
    """
    mcml_limit = config['mcml_limit']
    delta = config['r_center_delta']

    # ---- 提取所有 checkpoint 的归一化指标 ----
    enriched = {}
    for name, data in all_results.items():
        vals = extract_metrics(data['metrics'], config)
        vals['soft_score'] = (
            config['soft_w_tdr'] * vals['tdr']
            + config['soft_w_r_center'] * vals['r_center']
            + config['soft_w_aci'] * vals['aci']
            + config['soft_w_angle'] * vals['angle_score']
        )
        enriched[name] = vals

    if not enriched:
        return None, None, {'reason': '无有效 checkpoint'}

    # ---- 第一阶段：硬约束 ----
    w_max = max(v['r_center'] for v in enriched.values())

    feasible = {}
    mcml_rejected = {}
    for name, vals in enriched.items():
        r_ok = vals['r_center'] >= w_max - delta
        mcml_ok = vals['mcml'] <= mcml_limit
        vals['r_constraint_ok'] = r_ok
        vals['mcml_ok'] = mcml_ok
        vals['feasible'] = r_ok and mcml_ok
        if vals['feasible']:
            feasible[name] = vals
        elif not mcml_ok:
            mcml_rejected[name] = vals

    info = {
        'w_max': w_max,
        'mcml_limit': mcml_limit,
        'delta': delta,
        'total': len(enriched),
        'feasible': len(feasible),
        'mcml_rejected': len(mcml_rejected),
    }

    # ---- 第二阶段：可行域内选最优 ----
    if feasible:
        best_name = max(feasible, key=lambda n: feasible[n]['soft_score'])
        info['selection'] = 'constraint_pass'
        return best_name, all_results[best_name], info

    # ---- Fallback 1：保持 MCML 生存红线，放宽 R_center 近最优约束 ----
    mcml_feasible = {
        name: vals for name, vals in enriched.items()
        if vals['mcml_ok']
    }
    if mcml_feasible:
        print('\n  [警告] 无 checkpoint 同时满足 R_center 近最优约束与 MCML 约束。')
        print('  [Fallback] 保持 MCML 生存红线，放宽 R_center 近最优约束，在 MCML 达标集合内按软评分选择。')
        best_name = max(mcml_feasible, key=lambda n: mcml_feasible[n]['soft_score'])
        info['selection'] = 'fallback_mcml_pass_soft_score'
        info['mcml_feasible'] = len(mcml_feasible)
        return best_name, all_results[best_name], info

    # ---- Fallback 2：所有 checkpoint 均 MCML 超限，选 MCML 最小者 ----
    print('\n  [警告] 所有 checkpoint 均超过 MCML 生存红线，进入最终 Fallback 选择。')
    # 按 MCML 升序 → R_center 降序 → A_RMSE 升序 排序
    fallback_sorted = sorted(
        enriched.items(),
        key=lambda x: (x[1]['mcml'], -x[1]['r_center'], x[1]['armse']),
    )
    best_name = fallback_sorted[0][0]
    info['selection'] = 'fallback_mcml_min'
    info['fallback_mcml'] = enriched[best_name]['mcml']
    return best_name, all_results[best_name], info


# =====================================================================
# 6. 对比表输出
# =====================================================================

def print_comparison_table(all_results, selection_info, config):
    """打印所有 checkpoint 的指标对比表，标注可行性。"""
    if not all_results:
        print('无有效结果。')
        return

    mcml_limit = config['mcml_limit']
    sep = '=' * 110
    print(f'\n{sep}')
    print(f'  Checkpoint 离线扫描对比表  [VAL 集, 两阶段约束式选择]')
    print(f'  硬约束: R_center >= W_max - {config["r_center_delta"]}, '
          f'MCML <= {mcml_limit}')
    print(f'  软评分: 0.35*TDR + 0.25*Rc + 0.20*ACI + 0.20*AngleScore')
    print(sep)

    header = (
        f'{"Checkpoint":<25s} '
        f'{"TDR":>6s} {"Rc_w":>6s} {"ACI":>6s} {"MCML":>5s} '
        f'{"A-RMSE":>7s} {"Angle":>6s} '
        f'{"软评分":>7s} {"状态":>8s}'
    )
    print(header)
    print('-' * 110)

    # 重新计算所有指标用于显示
    enriched = {}
    for name, data in all_results.items():
        vals = extract_metrics(data['metrics'], config)
        vals['soft_score'] = (
            config['soft_w_tdr'] * vals['tdr']
            + config['soft_w_r_center'] * vals['r_center']
            + config['soft_w_aci'] * vals['aci']
            + config['soft_w_angle'] * vals['angle_score']
        )
        enriched[name] = vals

    w_max = selection_info.get('w_max', 0)
    delta = config['r_center_delta']

    # 按软评分降序排列
    sorted_items = sorted(
        enriched.items(),
        key=lambda x: x[1]['soft_score'],
        reverse=True,
    )

    for name, vals in sorted_items:
        r_ok = vals['r_center'] >= w_max - delta
        mcml_ok = vals['mcml'] <= mcml_limit
        feasible = r_ok and mcml_ok

        if feasible:
            status = '  OK'
        elif not mcml_ok:
            status = 'MCML!'
        else:
            status = 'Rc!'

        print(
            f'{name:<25s} '
            f'{vals["tdr"]:>6.3f} {vals["r_center"]:>6.3f} '
            f'{vals["aci"]:>6.3f} {vals["mcml"]:>5d} '
            f'{vals["armse"]:>7.2f} {vals["angle_score"]:>6.3f} '
            f'{vals["soft_score"]:>7.4f} {status:>8s}'
        )

    print(sep)

    sel = selection_info.get('selection', 'unknown')
    if sel == 'constraint_pass':
        print(f'\n  选择方式: 硬约束通过 + 软评分最优')
    elif sel == 'fallback_mcml_pass_soft_score':
        print(f'\n  选择方式: Fallback-1（保持 MCML 生存红线，放宽 R_center 近最优约束）')
        print(f'  MCML 达标 checkpoint: {selection_info.get("mcml_feasible", "?")}')
    elif sel == 'fallback_mcml_min':
        print(f'\n  选择方式: Fallback-2（所有 checkpoint 均 MCML 超限，选 MCML 最小）')
        print(f'  Fallback MCML: {selection_info.get("fallback_mcml", "?")}')

    print(f'  可行 checkpoint: {selection_info.get("feasible", 0)}'
          f' / {selection_info.get("total", 0)}')
    print(f'{sep}\n')


# =====================================================================
# 7. 最终 test 评估（可选）
# =====================================================================

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def annotation_set_sha256(directory):
    digest = hashlib.sha256()
    for path in sorted(glob.glob(os.path.join(directory, '*.txt'))):
        if os.path.basename(path).startswith('._'):
            continue
        digest.update(os.path.basename(path).encode('utf-8'))
        digest.update(b'\0')
        with open(path, 'rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


def run_final_test(config, best_ckpt, sweep_dir, center_thresh):
    """用最优权重在 test 集上跑最终离线评估。"""
    print(f'\n{"=" * 60}')
    print(f'  最终评估: 在 TEST 集上使用最优权重')
    print(f'{"=" * 60}')

    final_dir = os.path.join(
        sweep_dir,
        'final_test',
        os.path.splitext(os.path.basename(best_ckpt))[0],
    )
    preds_dir = os.path.join(final_dir, 'preds')
    os.makedirs(preds_dir, exist_ok=True)
    pkl_path = os.path.join(preds_dir, 'results.pkl')
    gt_dir = os.path.join(PROJ_ROOT, 'crane_project/data/crane_grab/test/annfiles')

    if os.path.exists(pkl_path):
        check_or_record_prediction(
            config, best_ckpt, pkl_path, gt_dir, 'fixed_test')
    else:
        if os.path.exists(prediction_provenance_path(pkl_path)):
            raise RuntimeError('Prediction provenance exists without PKL: '
                               + pkl_path)
        cmd = [
            sys.executable,
            os.path.join(PROJ_ROOT, 'tools/test.py'),
            config, best_ckpt,
            '--work-dir', final_dir,
            '--out', pkl_path,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=PROJ_ROOT, timeout=1200, env=inference_subprocess_env(),
        )
        if result.returncode != 0:
            print('  [错误] test 集推理失败')
            for line in result.stderr.strip().split('\n')[-5:]:
                print(f'    {line}')
            return None
        check_or_record_prediction(
            config, best_ckpt, pkl_path, gt_dir, 'fixed_test',
            generated=True)

    img_ids = get_val_img_ids(gt_dir)
    if len(img_ids) != 992:
        raise RuntimeError('Expected the fixed 992-frame TEST; found '
                           + str(len(img_ids)))
    task_dir = pkl_to_dota(pkl_path, img_ids, preds_dir)
    metrics = run_offline_eval(task_dir, gt_dir, mode='test', center_thresh=center_thresh)
    report = dict(
        protocol='crane_ckpt_sweep_final_test_v2',
        metric_protocol_version=METRIC_PROTOCOL_VERSION,
        evidence_role='fixed_test_after_source_val_selection',
        config=os.path.abspath(config),
        config_sha256=sha256_file(config),
        checkpoint=os.path.abspath(best_ckpt),
        checkpoint_sha256=sha256_file(best_ckpt),
        results_pkl=os.path.abspath(pkl_path),
        results_pkl_sha256=sha256_file(pkl_path),
        gt_dir=os.path.abspath(gt_dir),
        gt_annotations_sha256=annotation_set_sha256(gt_dir),
        frame_count=len(img_ids),
        center_thresh_px=center_thresh,
        metrics=metrics)
    report_path = os.path.join(final_dir, 'final_test_metrics_v2.json')
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + '\n'
    if os.path.exists(report_path):
        with open(report_path, 'r', encoding='utf-8') as stream:
            if stream.read() != encoded:
                raise RuntimeError(
                    'Refusing to overwrite different TEST report: '
                    + report_path)
    else:
        with open(report_path, 'w', encoding='utf-8') as stream:
            stream.write(encoded)

    print('\n  [TEST 集最终结果]')
    for k, v in sorted(metrics.items()):
        print(f'    {k}: {v}')

    return metrics


# =====================================================================
# 主入口
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='训练后 checkpoint 离线扫描选权工具',
    )
    parser.add_argument('--config', required=True, help='训练配置文件路径')
    parser.add_argument('--work-dir', required=True, help='训练输出目录')
    parser.add_argument(
        '--sweep-dir', default=None,
        help='独立输出目录；不指定时沿用 work-dir/ckpt_sweep')
    parser.add_argument(
        '--reuse-preds-from', default=None,
        help='从历史 sweep 逐帧预测重算指标，不重新运行 VAL 推理')
    parser.add_argument(
        '--final-test-from', default=None,
        help='从已完成的 source-VAL sweep_results.json 读取选中权重，只运行最终 TEST')
    parser.add_argument(
        '--epochs', nargs='+', type=int, default=None,
        help='只扫描指定 epoch（默认扫描全部 checkpoint）',
    )
    parser.add_argument(
        '--center-thresh', type=float, default=15.0,
        help='质心命中阈值（像素），默认 15.0',
    )
    parser.add_argument(
        '--mcml-limit', type=int, default=5,
        help='MCML 硬约束上限（帧数），默认 5',
    )
    parser.add_argument(
        '--run-final-test', action='store_true',
        help='扫描完成后自动在 test 集上跑最终评估',
    )
    parser.add_argument(
        '--include-avg', action='store_true',
        help='扫描时纳入 avg_*.pth 权重（默认不纳入）',
    )
    parser.add_argument(
        '--include-best', action='store_true',
        help='扫描时纳入 best_*.pth 权重（默认不纳入）',
    )
    parser.add_argument('--gpu', type=int, default=0, help='GPU 编号（单 GPU 串行模式）')
    parser.add_argument(
        '--gpus', nargs='+', type=int, default=None,
        help='多 GPU 并行扫描使用的 GPU 编号列表，例如 --gpus 0 1 2；指定后会覆盖 --gpu',
    )
    args = parser.parse_args()

    if args.final_test_from and (args.reuse_preds_from or args.run_final_test):
        raise ValueError('--final-test-from is a separate stage')
    if args.reuse_preds_from and args.run_final_test:
        raise ValueError('Finish the source-VAL reuse stage before TEST')

    gpu_list = args.gpus if args.gpus else [args.gpu]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in gpu_list)

    # 合并配置：命令行参数覆盖默认值
    config = dict(SELECTION_CONFIG)
    config['mcml_limit'] = args.mcml_limit

    work_dir = os.path.abspath(args.work_dir)
    sweep_dir = os.path.abspath(args.sweep_dir or os.path.join(
        work_dir, 'ckpt_sweep'))
    if args.final_test_from:
        with open(args.final_test_from, 'r', encoding='utf-8') as stream:
            selection = json.load(stream)
        if (selection.get('metric_protocol_version') !=
                METRIC_PROTOCOL_VERSION
                or selection.get('evidence_role') !=
                'source_val_checkpoint_selection'
                or selection.get('center_thresh_px') !=
                args.center_thresh
                or selection.get('config_sha256') !=
                sha256_file(args.config)
                or os.path.abspath(args.final_test_from) !=
                os.path.join(sweep_dir, 'sweep_results.json')):
            raise ValueError('Invalid source-VAL selection for TEST')
        chosen = selection['selected_checkpoint']
        checkpoint = selection['all_checkpoints'][chosen]
        if (os.path.abspath(selection['selected_path']) !=
                os.path.abspath(checkpoint['checkpoint'])
                or sha256_file(checkpoint['checkpoint']) !=
                checkpoint['checkpoint_sha256']):
            raise ValueError('Selected checkpoint identity mismatch')
        run_final_test(args.config, checkpoint['checkpoint'], sweep_dir,
                       args.center_thresh)
        return
    summary_path = os.path.join(sweep_dir, 'sweep_results.json')
    if os.path.exists(summary_path):
        raise RuntimeError('Refusing to overwrite existing sweep: '
                           + summary_path)
    os.makedirs(sweep_dir, exist_ok=True)

    # ---- 1. 发现 checkpoint ----
    ckpts = find_checkpoints(
        work_dir, args.epochs,
        include_avg=args.include_avg,
        include_best=args.include_best,
    )
    print(f'\n发现 {len(ckpts)} 个 checkpoint:')
    for name, path in ckpts.items():
        print(f'  {name}: {path}')
    if not ckpts:
        print('未找到任何 checkpoint，退出。')
        return
    if args.epochs is not None:
        missing = sorted(set(args.epochs) - {
            int(name.split('_')[-1]) for name in ckpts
            if re.fullmatch(r'epoch_\d+', name)})
        if missing:
            raise RuntimeError('Missing requested epochs: ' + repr(missing))

    # ---- 2. 获取 val 集 img_id ----
    val_ann_dir = os.path.join(PROJ_ROOT, 'crane_project/data/crane_grab/val/annfiles')
    img_ids = get_val_img_ids(val_ann_dir)
    if len(img_ids) != 738:
        raise RuntimeError('Expected the fixed 738-frame source VAL; found '
                           + str(len(img_ids)))
    print(f'\nVal 集: {len(img_ids)} 帧')

    # ---- 3. 逐 checkpoint 扫描 ----
    all_results = OrderedDict()
    ckpt_items = list(ckpts.items())

    if len(gpu_list) == 1:
        gpu = gpu_list[0]
        for ckpt_name, ckpt_path in ckpt_items:
            name, result = process_checkpoint(
                ckpt_name, ckpt_path, args, sweep_dir,
                img_ids, val_ann_dir, gpu,
            )
            if result is not None:
                all_results[name] = result
    else:
        print(f'\n启用多 GPU 并行扫描: {gpu_list}')
        max_workers = len(gpu_list)
        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for idx, (ckpt_name, ckpt_path) in enumerate(ckpt_items):
                gpu = gpu_list[idx % len(gpu_list)]
                print(f'  分配 {ckpt_name} -> GPU {gpu}')
                fut = executor.submit(
                    process_checkpoint,
                    ckpt_name, ckpt_path, args, sweep_dir,
                    img_ids, val_ann_dir, gpu,
                )
                futures[fut] = ckpt_name

            completed = {}
            for fut in as_completed(futures):
                ckpt_name = futures[fut]
                try:
                    name, result = fut.result()
                except Exception as e:
                    print(f'  [{ckpt_name}] 处理异常，跳过: {e}')
                    continue
                if result is not None:
                    completed[name] = result

        # 恢复 checkpoint 发现顺序，保证 sweep_results.json 稳定可复现
        for ckpt_name, _ in ckpt_items:
            if ckpt_name in completed:
                all_results[ckpt_name] = completed[ckpt_name]

    if set(all_results) != set(ckpts):
        raise RuntimeError('Incomplete checkpoint sweep: missing '
                           + repr(sorted(set(ckpts) - set(all_results))))

    # ---- 4. 两阶段约束式选择 ----
    best_name, best_data, selection_info = select_best_checkpoint(
        all_results, config,
    )

    # ---- 5. 对比表 ----
    print_comparison_table(all_results, selection_info, config)

    if best_name is None:
        print('选择失败：', selection_info.get('reason', '未知'))
        return

    print(f'  推荐最优权重: {best_name}')
    print(f'  路径: {best_data["checkpoint"]}')

    # ---- 6. 保存结果 ----
    save_data = {
        'selection_config': config,
        'metric_protocol_version': METRIC_PROTOCOL_VERSION,
        'evidence_role': 'source_val_checkpoint_selection',
        'candidate_epochs': list(args.epochs or sorted(
            int(name.split('_')[-1]) for name in ckpts
            if re.fullmatch(r'epoch_\d+', name))),
        'center_thresh_px': args.center_thresh,
        'config_sha256': sha256_file(args.config),
        'source_val_annotations_sha256': annotation_set_sha256(
            val_ann_dir),
        'selection_info': {k: v for k, v in selection_info.items()},
        'selected_checkpoint': best_name,
        'selected_path': best_data['checkpoint'],
        'all_checkpoints': {},
    }
    for k, v in all_results.items():
        derived = extract_metrics(v['metrics'], config)
        derived['soft_score'] = (
            config['soft_w_tdr'] * derived['tdr']
            + config['soft_w_r_center'] * derived['r_center']
            + config['soft_w_aci'] * derived['aci']
            + config['soft_w_angle'] * derived['angle_score']
        )
        derived['r_constraint_ok'] = derived['r_center'] >= selection_info['w_max'] - config['r_center_delta']
        derived['mcml_ok'] = derived['mcml'] <= config['mcml_limit']
        derived['feasible'] = derived['r_constraint_ok'] and derived['mcml_ok']
        save_data['all_checkpoints'][k] = {
            'checkpoint': v['checkpoint'],
            'checkpoint_sha256': v['checkpoint_sha256'],
            'results_pkl': v['results_pkl'],
            'results_pkl_sha256': v['results_pkl_sha256'],
            'metrics': v['metrics'],
            'derived_selection_metrics': derived,
        }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f'结果已保存至: {summary_path}')

    # ---- 6b. 保存最优权重路径文本文件 ----
    selected_txt = os.path.join(sweep_dir, 'selected_checkpoint.txt')
    with open(selected_txt, 'w', encoding='utf-8') as f:
        f.write(best_data['checkpoint'] + '\n')
    print(f'最优权重路径已保存至: {selected_txt}')

    # ---- 7. 可选：最终 test 评估 ----
    if args.run_final_test:
        print(f'\n最优权重 [{best_name}] 将在 TEST 集上进行最终评估...')
        run_final_test(args.config, best_data['checkpoint'], sweep_dir,
                       args.center_thresh)


if __name__ == '__main__':
    main()
