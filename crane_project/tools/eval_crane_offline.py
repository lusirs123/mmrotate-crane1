"""
eval_crane_offline.py
港口门座起重机抓斗 OBB 检测 —— 纯静态离线时空评估基座

核心机制：
1. 完全剥离 MMEngine 与 MMCV 依赖。
2. 保留 crane_metrics.py 的指标名称，修正旋转 IoU 与时序边界实现。
3. 动态解析 DOTA 文本并重建绝对时序。
4. 评估结果自动保存至对应训练目录，方便跨实验对比。
"""

import json
import math
import os
import re
import glob
import logging
import warnings
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2


METRIC_PROTOCOL_VERSION = 2

# =====================================================================
# 通用几何与时序工具函数
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
    """Compute exact CPU rotated IoU for le90 OBBs.

    OpenCV keeps the offline evaluator independent of MMCV CUDA operators,
    while still respecting orientation and unequal box sizes.  The previous
    axis-aligned shortcut was not a valid IoU (even for contained boxes) and
    therefore also corrupted the TDR/MCML hit flags derived from it.
    """
    first = np.asarray(box1, dtype=np.float64).reshape(-1)
    second = np.asarray(box2, dtype=np.float64).reshape(-1)
    if (first.size < 5 or second.size < 5
            or not np.isfinite(first[:5]).all()
            or not np.isfinite(second[:5]).all()
            or np.any(first[2:4] <= 0.0)
            or np.any(second[2:4] <= 0.0)):
        return 0.0

    def _rect(box):
        return ((float(box[0]), float(box[1])),
                (float(box[2]), float(box[3])),
                math.degrees(float(box[4])))

    area1 = float(first[2] * first[3])
    area2 = float(second[2] * second[3])
    kind, points = cv2.rotatedRectangleIntersection(
        _rect(first), _rect(second))
    if kind == cv2.INTERSECT_NONE:
        intersection = 0.0
    elif points is None:
        # OpenCV may omit vertices for a numerically exact full containment.
        intersection = min(area1, area2)
    else:
        hull = cv2.convexHull(np.asarray(points, dtype=np.float32))
        intersection = abs(float(cv2.contourArea(hull)))
        intersection = min(intersection, area1, area2)
    union = area1 + area2 - intersection
    if union <= 1e-6:
        return 0.0
    return float(np.clip(intersection / union, 0.0, 1.0))

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
        sim_angle_center_thresh_px: float = 10.0,
    ) -> None:
        self.mode             = mode
        self.center_thresh_px = center_thresh_px
        self.ekf_window       = ekf_window
        self.mcml_limit       = mcml_limit
        self.angle_limit_rad  = math.radians(angle_limit_deg)
        self.sim_angle_center_thresh_px = float(
            sim_angle_center_thresh_px)
        self.depth_k          = depth_k
        self.depth_alpha      = depth_alpha
        self.iou_thresh       = iou_thresh
        self.results          = []
        
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        self.logger = logging.getLogger(__name__)

    def evaluate_records(self, records: List[dict]) -> Dict[str, float]:
        """Evaluate already decoded per-frame OBB records.

        This is the in-model/config counterpart of ``extract_from_dirs``.  It
        avoids a temporary DOTA export when MMRotate already owns the ordered
        prediction stream.
        """
        required = ('domain', 'seq_id', 'frame_id', 'pred_box', 'gt_box')
        normalized = []
        for record in records:
            if any(key not in record for key in required):
                raise ValueError('Temporal evaluation record is incomplete')
            normalized.append(dict(
                domain=str(record['domain']),
                seq_id=str(record['seq_id']),
                frame_id=int(record['frame_id']),
                pred_box=(None if record['pred_box'] is None else
                          np.asarray(record['pred_box'], dtype=np.float64)),
                gt_box=(None if record['gt_box'] is None else
                        np.asarray(record['gt_box'], dtype=np.float64)),
                score=float(record.get('score', 0.0)),
                plc_rope=record.get('plc_rope')))
        self.results = normalized
        return self.compute_metrics()

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
            # A frame-id gap denotes a new physical clip.  In particular,
            # test/real_seq02 contains 2..180 and 960..1000; temporal windows
            # must never bridge that discontinuity.
            segments = self._split_contiguous_frames(frames)
            for segment in segments:
                sm = self._compute_sequence_metrics(segment)
                b = domain_buckets[domain]

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
    # 保留原指标名称的修正后计算与聚合逻辑
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
                # Target-presence metrics are undefined on a negative frame.
                # Exclude it and break temporal continuity instead of treating
                # it as a successful detection.
                prev_diag, prev_gamma, prev_frame_id = None, None, None
                continue

            is_hit = False
            if pred is not None:
                riou   = compute_riou(pred, gt)
                riou_vals.append(riou)
                is_hit = riou >= self.iou_thresh

                dist = float(np.linalg.norm(obb_center(pred) - obb_center(gt)))
                center_hits.append(float(dist < self.center_thresh_px))
                if dist < self.sim_angle_center_thresh_px:
                    err = float(angle_diff(
                        np.array([pred[4]]), np.array([gt[4]]))[0])
                    angle_errors.append(err)
                else:
                    angle_errors.append(math.pi / 2.0)

                cur_diag  = obb_diag(pred)
                cur_gamma = float(pred[4])

                if prev_diag is not None and prev_frame_id is not None and prev_diag > 1e-6:
                    gap = cur_fid - prev_frame_id
                    if gap == 1:
                        dfr_val = abs(cur_diag - prev_diag) / (prev_diag * gap)
                        dfr_vals.append(dfr_val)

                if prev_gamma is not None and prev_frame_id is not None:
                    gap = cur_fid - prev_frame_id
                    if gap == 1:
                        d_gamma = abs(float(angle_diff(np.array([cur_gamma]), np.array([prev_gamma]))[0]))
                        aci_val = 1.0 - d_gamma / (self.angle_limit_rad + 1e-9)
                        aci_vals.append(float(np.clip(aci_val, 0.0, 1.0)))

                prev_diag, prev_gamma, prev_frame_id = cur_diag, cur_gamma, cur_fid

                if plc is not None and float(plc) > 0:
                    z_est   = self.depth_k * (cur_diag ** self.depth_alpha)
                    dep_val = abs(z_est - float(plc)) / float(plc)
                    dep_vals.append(dep_val)
            else:
                # Static metrics use all positive-GT frames.  A missing output
                # contributes zero overlap/center recall and the maximal angle
                # penalty, while also breaking temporal box continuity.
                riou_vals.append(0.0)
                center_hits.append(0.0)
                angle_errors.append(math.pi / 2.0)
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

        mrf_vals, miss_start = [], None
        for i, h in enumerate(hit_flags):
            if not h and miss_start is None:
                miss_start = i
            elif h and miss_start is not None:
                mrf_vals.append(i - miss_start)
                miss_start = None

        return {
            'angle_errors': angle_errors, 'center_hits': center_hits,
            'dfr_vals': dfr_vals, 'aci_vals': aci_vals, 'dep_vals': dep_vals,
            'riou_vals': riou_vals, 'tdr_hits': tdr_hits, 'mcml': mcml, 'mrf_vals': mrf_vals,
        }

    @staticmethod
    def _split_contiguous_frames(frames: List[dict]) -> List[List[dict]]:
        segments = []
        current = []
        for frame in frames:
            # Negative/unknown-presence frames are not part of target-present
            # metrics and form a hard temporal boundary.
            if frame.get('gt_box') is None:
                if current:
                    segments.append(current)
                    current = []
                continue
            if (current and int(frame['frame_id']) !=
                    int(current[-1]['frame_id']) + 1):
                segments.append(current)
                current = [frame]
            else:
                current.append(frame)
        if current:
            segments.append(current)
        return segments

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

    # =================================================================
    # 结果持久化
    # =================================================================

    def save_results(
        self,
        metrics: Dict[str, float],
        save_dir: str,
        config: str = '',
        checkpoint: str = '',
    ) -> str:
        """将本次评估指标写入 JSON 文件，同时维护一份可追加的 summary。

        文件布局（与 avg_eval/ 风格对齐）：
            {save_dir}/offline_eval_<YYYYmmdd_HHMMSS>.json
            {save_dir}/offline_eval_summary.json

        返回本次写入的 JSON 文件路径。
        """
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        entry = {
            'config': config,
            'checkpoint': checkpoint,
            'mode': self.mode,
            'metric_protocol_version': METRIC_PROTOCOL_VERSION,
            'metric': metrics,
        }

        # ---- 单次文件 ----
        single_path = os.path.join(save_dir, f'offline_eval_{timestamp}.json')
        with open(single_path, 'w', encoding='utf-8') as f:
            json.dump(entry, f, indent=2, ensure_ascii=False)
        self.logger.info(f'评估结果已保存至: {single_path}')

        # ---- 累积 summary ----
        summary_path = os.path.join(save_dir, 'offline_eval_summary.json')
        if os.path.exists(summary_path):
            with open(summary_path, 'r', encoding='utf-8') as f:
                try:
                    summary = json.load(f)
                except json.JSONDecodeError:
                    summary = []
        else:
            summary = []

        summary.append(entry)
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        self.logger.info(f'累积摘要已更新至: {summary_path}')

        return single_path

def _infer_save_dir(pred_dir: str) -> str:
    """从预测目录向上回溯，自动定位对应的训练 work_dir。

    典型目录结构：
        work_dirs/<exp_name>/preds*/Task1_grab/  →  返回 work_dirs/<exp_name>
        work_dirs/<exp_name>/preds*/              →  返回 work_dirs/<exp_name>
    若回溯至 work_dirs/ 本身或文件系统根目录仍未命中，则回退到 pred_dir。
    """
    cur = os.path.abspath(pred_dir)
    work_dirs_root = os.path.abspath('work_dirs')

    for _ in range(5):
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        # 如果当前层已包含 .pth 或 .py 配置，视为 work_dir
        if any(f.endswith('.pth') for f in os.listdir(cur)):
            return cur
        # 如果父级就是 work_dirs/ 且当前层是实验目录
        if parent == work_dirs_root:
            return cur
        cur = parent

    return os.path.abspath(pred_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CraneOBB 离线评估器')
    parser.add_argument(
        '--gt_dir',
        default='crane_project/data/crane_grab/test/annfiles',
        help='GT 标注目录')
    parser.add_argument(
        '--pred_dir',
        default='work_dirs/crane_baseline/preds/Task1_grab/',
        help='预测结果目录')
    parser.add_argument(
        '--mode',
        default='test',
        choices=['test', 'val'])
    parser.add_argument(
        '--center_thresh',
        type=float,
        default=15.0)
    parser.add_argument(
        '--save_dir',
        default=None,
        help='评估结果保存目录；未指定时自动从 pred_dir 推断对应训练目录')
    parser.add_argument(
        '--config',
        default='',
        help='关联的训练配置文件路径（仅用于记录）')
    parser.add_argument(
        '--checkpoint',
        default='',
        help='关联的权重文件路径（仅用于记录）')
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='不保存评估结果 JSON 文件（仅终端输出）')
    args = parser.parse_args()

    # ---- 确定保存目录 ----
    if args.save_dir is not None:
        save_dir = args.save_dir
    else:
        save_dir = _infer_save_dir(args.pred_dir)

    evaluator = CraneOfflineEvaluator(
        mode=args.mode,
        center_thresh_px=args.center_thresh,
    )
    evaluator.extract_from_dirs(gt_dir=args.gt_dir, pred_dir=args.pred_dir)
    metrics = evaluator.compute_metrics()

    # ---- 持久化评估结果 ----
    if metrics and not args.no_save:
        evaluator.save_results(
            metrics,
            save_dir=save_dir,
            config=args.config,
            checkpoint=args.checkpoint,
        )
    elif metrics:
        print('（--no-save 模式：结果未写入文件）')
