"""
eval_crane_offline.py
港口门座起重机抓斗 OBB 检测 —— 纯静态离线时空评估基座

核心机制：
1. 完全剥离 MMEngine 与 MMCV 依赖。
2. 继承原版 crane_metrics.py 的所有数学计算流形与指标体系。
3. 动态解析 DOTA 文本并重建绝对时序。
"""

import math
import os
import re
import glob
import logging
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2

# =====================================================================
# 原版工具函数（数学逻辑 100% 保持不变）
# =====================================================================

def parse_seq_frame(img_path: str) -> Tuple[str, str, int]:
    basename = os.path.splitext(os.path.basename(img_path))[0]
    m = re.match(r'^(real|sim)_(.+)_(\d+)$', basename)
    if m:
        return m.group(1), m.group(2), int(m.group(3))

    m = re.match(r'^(.+)_(\d+)$', basename)
    if m:
        return 'unknown', m.group(1), int(m.group(2))

    return 'unknown', 'default', abs(hash(basename)) % (10 ** 8)

def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a - b
    diff = np.arctan2(np.sin(diff), np.cos(diff))
    diff = np.where(diff >  np.pi / 2, diff - np.pi, diff)
    diff = np.where(diff < -np.pi / 2, diff + np.pi, diff)
    return diff

def obb_diag(box: np.ndarray) -> float:
    return float(math.sqrt(float(box[2]) ** 2 + float(box[3]) ** 2))

def obb_center(box: np.ndarray) -> np.ndarray:
    return box[:2].copy()

def compute_riou(box1: np.ndarray, box2: np.ndarray) -> float:
    # 离线环境默认降级为高效的轴对齐近似计算，避免依赖底层 CUDA 算子
    area1   = float(box1[2]) * float(box1[3])
    area2   = float(box2[2]) * float(box2[3])
    cx_diff = abs(float(box1[0]) - float(box2[0]))
    cy_diff = abs(float(box1[1]) - float(box2[1]))
    ow = max(0.0, (float(box1[2]) + float(box2[2])) / 2 - cx_diff)
    oh = max(0.0, (float(box1[3]) + float(box2[3])) / 2 - cy_diff)
    inter = ow * oh
    union = area1 + area2 - inter + 1e-6
    return float(np.clip(inter / union, 0.0, 1.0))

# =====================================================================
# DOTA 文本离线解析器
# =====================================================================

def dota2obb_le90(poly: List[float]) -> np.ndarray:
    """将 DOTA 格式的多边形顶点严格转换为 le90 规范的 (cx, cy, w, h, theta)"""
    pts = np.array(poly, dtype=np.float32).reshape(4, 2)
    rect = cv2.minAreaRect(pts)
    (cx, cy), (w, h), angle = rect
    
    # le90 定义：w 与 x 轴锐角夹角为 theta，范围 [-pi/2, pi/2)
    if w < h:
        w, h = h, w
        angle += 90.0
    if angle >= 90.0:
        angle -= 180.0
    if angle < -90.0:
        angle += 180.0
        
    return np.array([cx, cy, w, h, math.radians(angle)], dtype=np.float64)

def parse_dota_txt(txt_path: str) -> List[np.ndarray]:
    bboxes = []
    if not os.path.exists(txt_path):
        return bboxes
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8:
                poly = [float(x) for x in parts[:8]]
                bboxes.append(dota2obb_le90(poly))
    return bboxes

# =====================================================================
# 核心指标类 (剥离 MMEngine 架构)
# =====================================================================

class CraneOfflineEvaluator:
    def __init__(
        self,
        mode: str = 'test',
        center_thresh_px: float = 15.0,
        ekf_window: int = 10,
        mcml_limit: int = 5,
        angle_limit_deg: float = 35.0,
        depth_k: float = 1000.0,
        depth_alpha: float = -1.5,
        iou_thresh: float = 0.5,
    ) -> None:
        self.mode             = mode
        self.center_thresh_px = center_thresh_px
        self.ekf_window       = ekf_window
        self.mcml_limit       = mcml_limit
        self.angle_limit_rad  = math.radians(angle_limit_deg)
        self.depth_k          = depth_k
        self.depth_alpha      = depth_alpha
        self.iou_thresh       = iou_thresh
        self.results          = []
        
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        self.logger = logging.getLogger(__name__)

    def extract_from_dirs(self, gt_dir: str, pred_dir: str) -> None:
        """从物理目录加载真值与预测流形，重构原版 process() 数据结构"""
        txt_files = glob.glob(os.path.join(gt_dir, '*.txt'))
        if not txt_files:
            self.logger.error(f"严重错误：在 {gt_dir} 未发现 GT 文件。")
            return

        for gt_path in txt_files:
            filename = os.path.basename(gt_path)
            pred_path = os.path.join(pred_dir, filename)
            
            domain, seq_id, frame_id = parse_seq_frame(filename)
            
            gt_boxes = parse_dota_txt(gt_path)
            pred_boxes = parse_dota_txt(pred_path)
            
            # 单目标假设对齐
            gt_box = gt_boxes[0] if gt_boxes else None
            pred_box = pred_boxes[0] if pred_boxes else None
            
            # TODO: 若后续需要接入深度指标，可在此处加载外部 PLC 字典
            plc_rope = None 

            self.results.append({
                'domain':   domain,
                'seq_id':   seq_id,
                'frame_id': frame_id,
                'pred_box': pred_box,
                'gt_box':   gt_box,
                'score':    1.0 if pred_box is not None else 0.0,
                'plc_rope': plc_rope,
            })

    def compute_metrics(self) -> Dict[str, float]:
        self.logger.info(f'CraneOfflineEvaluator [{self.mode.upper()} 模式]: 开始计算指标...')

        results_sorted = sorted(
            self.results,
            key=lambda x: (x['domain'], x['seq_id'], x['frame_id']),
        )

        seq_dict: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
        for r in results_sorted:
            seq_dict[(r['domain'], r['seq_id'])].append(r)

        self._diagnose_gaps(seq_dict)

        domain_buckets: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))

        for (domain, seq_id), frames in seq_dict.items():
            sm = self._compute_sequence_metrics(frames)
            b  = domain_buckets[domain]

            if domain == 'sim':
                b['angle_errors'].extend(sm['angle_errors'])

            b['center_hits'].extend(sm['center_hits'])
            b['riou_vals'].extend(sm['riou_vals'])
            b['dfr_vals'].extend(sm['dfr_vals'])
            b['aci_vals'].extend(sm['aci_vals'])
            b['dep_vals'].extend(sm['dep_vals'])
            b['tdr_hits'].extend(sm['tdr_hits'])
            b['mcml_list'].append(sm['mcml'])
            b['mrf_vals'].extend(sm['mrf_vals'])

        metrics: Dict[str, float] = {}
        all_domains = sorted(domain_buckets.keys())

        for domain in all_domains:
            b   = domain_buckets[domain]
            if self.mode == 'val':
                self._aggregate_val(metrics, b, domain)
            else:
                self._aggregate_test(metrics, b, domain)

        self._log_metrics(metrics, all_domains)
        return metrics

    # =================================================================
    # 以下为原版 crane_metrics.py 计算与聚合逻辑的绝对复用
    # =================================================================

    def _aggregate_val(self, metrics, b, pfx):
        if pfx == 'sim' and b['angle_errors']:
            a_rmse = math.degrees(math.sqrt(float(np.mean(np.array(b['angle_errors']) ** 2))))
            metrics[f'{pfx}/A-RMSE(deg)'] = round(a_rmse, 4)
        if pfx == 'sim' and b['aci_vals']:
            metrics[f'{pfx}/ACI'] = round(float(np.mean(b['aci_vals'])), 4)
        if b['center_hits']:
            metrics[f'{pfx}/R_center(%)'] = round(float(np.mean(b['center_hits'])) * 100, 2)

    def _aggregate_test(self, metrics, b, pfx):
        if pfx == 'sim' and b['angle_errors']:
            a_rmse = math.degrees(math.sqrt(float(np.mean(np.array(b['angle_errors']) ** 2))))
            metrics[f'{pfx}/A-RMSE(deg)'] = round(a_rmse, 4)
        if b['center_hits']:
            metrics[f'{pfx}/R_center(%)'] = round(float(np.mean(b['center_hits'])) * 100, 2)
        if b['riou_vals']:
            metrics[f'{pfx}/mean_RIoU'] = round(float(np.mean(b['riou_vals'])), 4)
        if b['dfr_vals']:
            metrics[f'{pfx}/DFR(%/frame)'] = round(float(np.mean(b['dfr_vals'])) * 100, 4)
        if b['aci_vals']:
            metrics[f'{pfx}/ACI'] = round(float(np.mean(b['aci_vals'])), 4)
        if b['dep_vals']:
            metrics[f'{pfx}/DEP(%)'] = round(float(np.mean(b['dep_vals'])) * 100, 4)
        if b['tdr_hits']:
            metrics[f'{pfx}/TDR_w{self.ekf_window}(%)'] = round(float(np.mean(b['tdr_hits'])) * 100, 2)
        if b['mcml_list']:
            max_mcml  = int(max(b['mcml_list']))
            mean_mcml = float(np.mean(b['mcml_list']))
            metrics[f'{pfx}/MCML_max(frames)']  = max_mcml
            metrics[f'{pfx}/MCML_mean(frames)'] = round(mean_mcml, 2)
            metrics[f'{pfx}/MCML_pass(limit={self.mcml_limit})'] = 1 if max_mcml <= self.mcml_limit else 0
        if b['mrf_vals']:
            metrics[f'{pfx}/MRF(frames)'] = round(float(np.mean(b['mrf_vals'])), 2)

    def _compute_sequence_metrics(self, frames: List[dict]) -> dict:
        angle_errors, center_hits = [], []
        dfr_vals, aci_vals, dep_vals, riou_vals, hit_flags = [], [], [], [], []
        prev_diag, prev_gamma, prev_frame_id = None, None, None

        for frame in frames:
            pred    = frame['pred_box']
            gt      = frame['gt_box']
            plc     = frame['plc_rope']
            cur_fid = int(frame['frame_id'])

            if gt is None:
                hit_flags.append(True)
                prev_frame_id = cur_fid
                continue

            is_hit = False
            if pred is not None:
                riou   = compute_riou(pred, gt)
                riou_vals.append(riou)
                is_hit = riou >= self.iou_thresh

                err = float(angle_diff(np.array([pred[4]]), np.array([gt[4]]))[0])
                angle_errors.append(err)

                dist = float(np.linalg.norm(obb_center(pred) - obb_center(gt)))
                center_hits.append(float(dist < self.center_thresh_px))

                cur_diag  = obb_diag(pred)
                cur_gamma = float(pred[4])

                if prev_diag is not None and prev_frame_id is not None and prev_diag > 1e-6:
                    gap = cur_fid - prev_frame_id
                    if gap > 0:
                        dfr_val = abs(cur_diag - prev_diag) / (prev_diag * gap)
                        dfr_vals.append(dfr_val)

                if prev_gamma is not None and prev_frame_id is not None:
                    gap = cur_fid - prev_frame_id
                    if gap > 0:
                        d_gamma = abs(float(angle_diff(np.array([cur_gamma]), np.array([prev_gamma]))[0]))
                        aci_val = 1.0 - d_gamma / (self.angle_limit_rad + 1e-9)
                        aci_vals.append(float(np.clip(aci_val, 0.0, 1.0)))

                prev_diag, prev_gamma, prev_frame_id = cur_diag, cur_gamma, cur_fid

                if plc is not None and float(plc) > 0:
                    z_est   = self.depth_k * (cur_diag ** self.depth_alpha)
                    dep_val = abs(z_est - float(plc)) / float(plc)
                    dep_vals.append(dep_val)
            else:
                prev_diag, prev_gamma, prev_frame_id = None, None, cur_fid

            hit_flags.append(is_hit)

        w = self.ekf_window
        tdr_hits = [any(hit_flags[i: i + w]) for i in range(max(0, len(hit_flags) - w + 1))]

        mcml = cur_miss = 0
        for h in hit_flags:
            if not h:
                cur_miss += 1
                mcml = max(mcml, cur_miss)
            else:
                cur_miss = 0

        mrf_vals, in_miss, miss_end = [], False, 0
        for i, h in enumerate(hit_flags):
            if not h:
                in_miss, miss_end = True, i
            elif in_miss:
                mrf_vals.append(i - miss_end)
                in_miss = False

        return {
            'angle_errors': angle_errors, 'center_hits': center_hits,
            'dfr_vals': dfr_vals, 'aci_vals': aci_vals, 'dep_vals': dep_vals,
            'riou_vals': riou_vals, 'tdr_hits': tdr_hits, 'mcml': mcml, 'mrf_vals': mrf_vals,
        }

    def _diagnose_gaps(self, seq_dict):
        for (domain, seq_id), frames in seq_dict.items():
            fids = np.array([f['frame_id'] for f in frames], dtype=np.int64)
            tag  = f"[{domain}] {seq_id}"
            if len(fids) < 2:
                continue
            gaps = np.diff(fids)
            bad = np.where(gaps <= 0)[0]
            if len(bad) > 0:
                self.logger.error(f"{tag}：存在 {len(bad)} 处帧号非单调，请检查文件命名。")

    def _log_metrics(self, metrics, all_domains):
        sep = '═' * 64
        self.logger.info(f'\n{sep}')
        self.logger.info(f'  CraneOBBMetric 评估结果  [{self.mode.upper()} 模式]')
        self.logger.info(sep)

        layer_defs = [
            ('第一层  静态精度（单帧）', ['A-RMSE(deg)', 'R_center(%)', 'mean_RIoU']),
            ('第二层  时序稳定性（帧间）', ['DFR(%/frame)', 'ACI', 'DEP(%)']),
            ('第三层  控制适用性（系统级）', [f'TDR_w{self.ekf_window}(%)', 'MCML_max(frames)', 'MCML_mean(frames)', f'MCML_pass(limit={self.mcml_limit})', 'MRF(frames)']),
        ]

        for domain in all_domains:
            self.logger.info(f'\n  ┌─ [{domain} 域] {"─"*46}')
            for layer_name, keys in layer_defs:
                self.logger.info(f'  │  {layer_name}')
                found = False
                for k in keys:
                    full_k = f'{domain}/{k}'
                    if full_k in metrics:
                        note = '（仅 sim 域输出）' if k == 'A-RMSE(deg)' and domain != 'sim' else ''
                        self.logger.info(f'  │    {full_k:<52s} {metrics[full_k]}{note}')
                        found = True
                if not found:
                    self.logger.info(f'  │    （{domain} 域本层无有效数据）')
            self.logger.info(f'  └{"─"*60}')
        self.logger.info(f'\n{sep}\n')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CraneOBB 离线评估器')
    parser.add_argument(
        '--gt_dir',
        default='crane_project/data/crane_grab/test/annfiles',
        help='GT 标注目录')
    parser.add_argument(
        '--pred_dir',
        default='/home/omnisky/workspace/symEOOD/work_dirs/crane_baseline/preds_thr001/Task1_grab/',
        help='预测结果目录')
    parser.add_argument(
        '--mode',
        default='test',
        choices=['test', 'val'])
    parser.add_argument(
        '--center_thresh',
        type=float,
        default=15.0)
    args = parser.parse_args()

    evaluator = CraneOfflineEvaluator(
        mode=args.mode,
        center_thresh_px=args.center_thresh,
    )
    evaluator.extract_from_dirs(gt_dir=args.gt_dir, pred_dir=args.pred_dir)
    evaluator.compute_metrics()