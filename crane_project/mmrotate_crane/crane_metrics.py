"""
crane_metrics.py
港口门座起重机抓斗 OBB 检测 —— 完整时空评估指标

版本：v3.0（域感知 + 非等间隔抽帧 + 训练/测试模式分离）
兼容：MMRotate 1.x（OpenMMLab 2.0）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
命名协议（必须严格遵守）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  格式：{domain}_{seqID}_{frameID}.jpg
  示例：
    real_seq04_00221.jpg  →  domain='real', seq='seq04', frame=221
    sim_seq01_00050.jpg   →  domain='sim',  seq='seq01', frame=50

  frameID：视频原始帧号（绝对值），非等间隔抽帧时保持原始值
  domain ：必须为 'real' 或 'sim'，否则归入 'unknown' 域

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
三层指标体系
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  第一层（静态精度，单帧）：
    A-RMSE(deg)      角度绝对均方根误差     仅 sim 域
    R_center(%)      中心点召回率           两域
    mean_RIoU        平均旋转 IoU           仅 test 模式

  第二层（时序稳定性，帧间，按帧间距归一化）：
    DFR(%/frame)     对角线相对抖动率       仅 test 模式
    ACI              角度时序一致性         val 仅 sim，test 两域
    DEP(%)           深度估算误差传播       仅 test 模式

  第三层（控制适用性，系统级）：
    TDR_w(%)         滑动窗口有效检测率     仅 test 模式
    MCML             最大/均值连续漏检帧数  仅 test 模式
    MRF(frames)      平均漏检恢复帧数       仅 test 模式

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
val 模式（训练期，驱动 Early Stopping）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  计算：sim/A-RMSE, sim/ACI, sim/R_center, real/R_center
  Early Stopping 基准：crane/sim/ACI（越高越好）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
test 模式（测试期，完整工程报告）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  计算：所有指标，两域分别报告

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
目录结构
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  your_project/
    ├── mmrotate_crane/
    │   ├── __init__.py
    │   └── crane_metrics.py   ← 本文件
    └── configs/
        └── crane_eood.py
"""

import math
import os
import re
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmrotate.registry import METRICS


# ═══════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════

def parse_seq_frame(img_path: str) -> Tuple[str, str, int]:
    """
    从文件名解析域标签、序列 ID、绝对帧号。

    命名协议：{domain}_{seqID}_{frameID}.ext
      有域前缀：real_seq04_00221.jpg → ('real', 'seq04', 221)
      无域前缀：seq04_00221.jpg      → ('unknown', 'seq04', 221)
      不符合  ：任意名               → ('unknown', 'default', hash)

    Returns:
        (domain, seq_id, frame_id)
    """
    basename = os.path.splitext(os.path.basename(img_path))[0]

    # 带域前缀：{real|sim}_{seqID}_{frameID}
    m = re.match(r'^(real|sim)_(.+)_(\d+)$', basename)
    if m:
        return m.group(1), m.group(2), int(m.group(3))

    # 无域前缀旧格式：{seqID}_{frameID}
    m = re.match(r'^(.+)_(\d+)$', basename)
    if m:
        warnings.warn(
            f"文件名 '{basename}' 缺少域前缀（real_/sim_），"
            f"归入 'unknown' 域。A-RMSE 等强 GT 依赖指标将跳过此域。"
            f"建议重命名为 real_seqXX_YYYYYYY.jpg 格式。",
            UserWarning,
            stacklevel=2,
        )
        return 'unknown', m.group(1), int(m.group(2))

    # 完全不符合协议
    warnings.warn(
        f"文件名 '{basename}' 不符合命名协议，退化为单序列模式。",
        UserWarning,
        stacklevel=2,
    )
    return 'unknown', 'default', abs(hash(basename)) % (10 ** 8)


def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    计算两角度序列的差值，处理 OBB 的 180° 周期性边界。

    OBB 旋转角具有 180° 等价性（旋转 180° 框不变），
    因此差值映射到 (-π/2, π/2] 而非 (-π, π]。

    Args:
        a, b: 角度数组，单位弧度
    Returns:
        差值数组，范围 (-π/2, π/2]
    """
    diff = a - b
    # 先映射到 (-π, π]
    diff = np.arctan2(np.sin(diff), np.cos(diff))
    # 再利用 180° 等价性映射到 (-π/2, π/2]
    diff = np.where(diff >  np.pi / 2, diff - np.pi, diff)
    diff = np.where(diff < -np.pi / 2, diff + np.pi, diff)
    return diff


def obb_diag(box: np.ndarray) -> float:
    """
    计算 OBB 对角线像素长度。

    Args:
        box: shape (5,)，格式 (cx, cy, w, h, theta_rad)
    Returns:
        对角线长度（像素）
    """
    return float(math.sqrt(float(box[2]) ** 2 + float(box[3]) ** 2))


def obb_center(box: np.ndarray) -> np.ndarray:
    """提取 OBB 中心点坐标，返回 shape (2,)"""
    return box[:2].copy()


def compute_riou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    计算两个 OBB 的旋转 IoU。

    优先使用 mmrotate 内置实现（精确多边形交集）。
    环境缺失时退化为轴对齐近似（仅用于调试）。

    Args:
        box1, box2: shape (5,)，格式 (cx, cy, w, h, theta_rad)
    Returns:
        IoU 值，范围 [0, 1]
    """
    try:
        import torch
        from mmrotate.structures.bbox import rbbox_overlaps
        b1 = torch.tensor(box1, dtype=torch.float32).unsqueeze(0)
        b2 = torch.tensor(box2, dtype=torch.float32).unsqueeze(0)
        iou = rbbox_overlaps(b1, b2, mode='iou').item()
        return float(np.clip(iou, 0.0, 1.0))
    except Exception:
        # 轴对齐近似退化
        area1   = float(box1[2]) * float(box1[3])
        area2   = float(box2[2]) * float(box2[3])
        cx_diff = abs(float(box1[0]) - float(box2[0]))
        cy_diff = abs(float(box1[1]) - float(box2[1]))
        ow = max(0.0, (float(box1[2]) + float(box2[2])) / 2 - cx_diff)
        oh = max(0.0, (float(box1[3]) + float(box2[3])) / 2 - cy_diff)
        inter = ow * oh
        union = area1 + area2 - inter + 1e-6
        return float(np.clip(inter / union, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════
# 主指标类
# ═══════════════════════════════════════════════════════════════════════

@METRICS.register_module(force=True)
class CraneOBBMetric(BaseMetric):
    """
    港口门座起重机抓斗 OBB 检测综合时空评估指标。

    核心设计原则：
      1. 域感知：real 域和 sim 域指标分开计算，A-RMSE 仅在 sim 域计算
      2. 时序重构：多 GPU 下通过绝对帧号重建帧序，不依赖 dataloader 顺序
      3. 非等间隔支持：DFR/ACI 按实际帧间距 gap 归一化
      4. 模式分离：val 模式只算 Early Stopping 必需的最小指标集

    Args:
        mode (str):
            'val'  → 训练期验证，计算最小指标集，驱动 Early Stopping
            'test' → 测试期评估，计算完整指标集
        center_thresh_px (float):
            中心点召回容差 δ_px（像素）。
            从控制精度要求反推：摆角精度 ±1° 对应约 15px（需按实际标定）。
        ekf_window (int):
            EKF 最大可靠预测窗口 w（帧数）。30fps 下 10 帧 = 333ms。
            含义：每连续 w 帧内至少需要 1 次有效检测。
        mcml_limit (int):
            最大允许连续漏检帧数硬性上限。超过则 MCML_pass=0。
        angle_limit_deg (float):
            抓斗物理约束角度范围（度），用于 ACI 归一化。
            来自起重机防摇控制的允许偏航范围，通常为 35°。
        depth_k (float):
            深度估算系数 k，Z_v = k × l_diag^alpha。需 Webots 标定。
        depth_alpha (float):
            深度估算指数 alpha，Z_v = k × l_diag^alpha。需 Webots 标定。
        iou_thresh (float):
            单帧 TP 判定的 RIoU 阈值。
        collect_device (str):
            分布式评估时的结果收集设备。
        prefix (str):
            日志和指标名称前缀，结果以 {prefix}/{domain}/{metric} 形式记录。
    """

    default_prefix: str = 'crane'

    def __init__(
        self,
        mode: str = 'val',
        center_thresh_px: float = 15.0,
        ekf_window: int = 10,
        mcml_limit: int = 5,
        angle_limit_deg: float = 35.0,
        depth_k: float = 1000.0,
        depth_alpha: float = -1.5,
        iou_thresh: float = 0.5,
        collect_device: str = 'cpu',
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        assert mode in ('val', 'test'), (
            f"mode 必须为 'val' 或 'test'，得到 '{mode}'"
        )
        self.mode             = mode
        self.center_thresh_px = center_thresh_px
        self.ekf_window       = ekf_window
        self.mcml_limit       = mcml_limit
        self.angle_limit_rad  = math.radians(angle_limit_deg)
        self.depth_k          = depth_k
        self.depth_alpha      = depth_alpha
        self.iou_thresh       = iou_thresh

    # ───────────────────────────────────────────────────────────────────
    # MMEngine 标准接口：process
    # ───────────────────────────────────────────────────────────────────

    def process(
        self,
        data_batch: dict,
        data_samples: Sequence[dict],
    ) -> None:
        """
        逐帧收集预测结果，存入 self.results。

        data_samples[i].metainfo 需包含：
          img_path (str):
              图像文件路径，文件名须符合命名协议，用于解析域/序列/帧号。
          plc_rope_length (float, 可选):
              PLC 编码器绳长，用于 DEP 计算。无此字段时 DEP 自动跳过。
          domain / seq_id / frame_id (可选):
              手动指定，优先级高于文件名解析，用于特殊情况覆盖。

        data_samples[i].pred_instances 需包含：
          bboxes (Tensor N×5): OBB (cx, cy, w, h, theta_rad)
          scores (Tensor N,):  置信度

        data_samples[i].gt_instances 需包含：
          bboxes (Tensor M×5): OBB，格式同上
        """
        for data_sample in data_samples:
            meta     = data_sample.get('metainfo', {})
            img_path = meta.get('img_path', '')

            # ── 时序信息解析（文件名为主，metainfo 显式字段可覆盖）──
            domain_f, seq_id_f, frame_id_f = parse_seq_frame(img_path)
            domain   = str(meta.get('domain',   domain_f))
            seq_id   = str(meta.get('seq_id',   seq_id_f))
            frame_id = int(meta.get('frame_id', frame_id_f))
            plc_rope = meta.get('plc_rope_length', None)

            # ── GT 解析（单目标场景，取第一个框）──
            gt_instances = data_sample.get('gt_instances', None)
            if (gt_instances is not None
                    and hasattr(gt_instances, 'bboxes')
                    and len(gt_instances.bboxes) > 0):
                gt_box = gt_instances.bboxes[0].cpu().numpy().astype(np.float64)
            else:
                gt_box = None

            # ── 预测解析（取置信度最高的框）──
            pred_instances = data_sample.get('pred_instances', None)
            if (pred_instances is not None
                    and hasattr(pred_instances, 'bboxes')
                    and len(pred_instances.bboxes) > 0):
                scores     = pred_instances.scores.cpu().numpy()
                best_idx   = int(scores.argmax())
                pred_box   = pred_instances.bboxes[best_idx].cpu().numpy().astype(np.float64)
                pred_score = float(scores[best_idx])
            else:
                pred_box   = None
                pred_score = 0.0

            # ── 存入 self.results（BaseMetric 标准接口）──
            # frame_id 为绝对帧号，collect_results() 汇总后
            # 在 compute_metrics() 按 (domain, seq_id, frame_id) 重构时序
            self.results.append({
                'domain':   domain,
                'seq_id':   seq_id,
                'frame_id': frame_id,
                'pred_box': pred_box,
                'gt_box':   gt_box,
                'score':    pred_score,
                'plc_rope': plc_rope,
            })

    # ───────────────────────────────────────────────────────────────────
    # MMEngine 标准接口：compute_metrics
    # ───────────────────────────────────────────────────────────────────

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """
        collect_results() 汇总后调用。

        执行流程：
          1. 时序重构：按 (domain, seq_id, frame_id) 排序
          2. 域分流：按 (domain, seq_id) 建立序列字典
          3. 帧间距诊断：打印分布，检测非法帧号
          4. 逐序列计算中间量
          5. 按域聚合，按 mode 输出对应指标子集
        """
        logger = MMLogger.get_current_instance()
        logger.info(
            f'CraneOBBMetric [{self.mode.upper()} 模式]: 开始计算指标...'
        )

        # ── Step 1：时序重构 ──────────────────────────────────────────
        results_sorted = sorted(
            results,
            key=lambda x: (x['domain'], x['seq_id'], x['frame_id']),
        )

        # ── Step 2：域感知分流 ────────────────────────────────────────
        # key: (domain, seq_id)，确保 real_seq05 和 sim_seq05 严格隔离
        seq_dict: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
        for r in results_sorted:
            seq_dict[(r['domain'], r['seq_id'])].append(r)

        # ── Step 3：帧间距诊断 ────────────────────────────────────────
        self._diagnose_gaps(seq_dict, logger)

        # ── Step 4：逐序列计算中间量，按域分桶 ──────────────────────
        # domain_buckets[domain][metric_key] = List[float | bool | int]
        domain_buckets: Dict[str, Dict[str, list]] = defaultdict(
            lambda: defaultdict(list)
        )

        for (domain, seq_id), frames in seq_dict.items():
            sm = self._compute_sequence_metrics(frames)
            b  = domain_buckets[domain]

            # 角度误差：仅 sim 域 GT 可信
            if domain == 'sim':
                b['angle_errors'].extend(sm['angle_errors'])

            # 中心点命中：两域的中心点标注手抖影响较小，均可信
            b['center_hits'].extend(sm['center_hits'])

            # RIoU：两域
            b['riou_vals'].extend(sm['riou_vals'])

            # 帧间指标：两域
            b['dfr_vals'].extend(sm['dfr_vals'])
            b['aci_vals'].extend(sm['aci_vals'])

            # DEP：依赖 PLC 绳长，两域按数据可用性
            b['dep_vals'].extend(sm['dep_vals'])

            # 控制适用性：两域
            b['tdr_hits'].extend(sm['tdr_hits'])
            b['mcml_list'].append(sm['mcml'])
            b['mrf_vals'].extend(sm['mrf_vals'])

        # ── Step 5：按 mode 聚合输出 ─────────────────────────────────
        metrics: Dict[str, float] = {}
        all_domains = sorted(domain_buckets.keys())

        for domain in all_domains:
            b   = domain_buckets[domain]
            pfx = domain    # 'sim', 'real', 'unknown'

            if self.mode == 'val':
                self._aggregate_val(metrics, b, pfx)
            else:
                self._aggregate_test(metrics, b, pfx)

        self._log_metrics(metrics, all_domains, logger)
        return metrics

    # ───────────────────────────────────────────────────────────────────
    # 指标聚合：val 模式（最小集）
    # ───────────────────────────────────────────────────────────────────

    def _aggregate_val(
        self,
        metrics: Dict[str, float],
        b: Dict[str, list],
        pfx: str,
    ) -> None:
        """
        val 模式：只计算驱动 Early Stopping 的最小指标集。
          sim 域：A-RMSE, ACI, R_center
          real 域：R_center（监控 Sim2Real 泛化）
          unknown 域：R_center
        """
        # A-RMSE：仅 sim 域，GT 角度可信
        if pfx == 'sim' and b['angle_errors']:
            a_rmse = math.degrees(math.sqrt(
                float(np.mean(np.array(b['angle_errors']) ** 2))
            ))
            metrics[f'{pfx}/A-RMSE(deg)'] = round(a_rmse, 4)

        # ACI：仅 sim 域，作为 save_best 的基准
        if pfx == 'sim' and b['aci_vals']:
            metrics[f'{pfx}/ACI'] = round(
                float(np.mean(b['aci_vals'])), 4
            )

        # R_center：所有域，用于监控 Sim2Real 泛化差距
        if b['center_hits']:
            metrics[f'{pfx}/R_center(%)'] = round(
                float(np.mean(b['center_hits'])) * 100, 2
            )

    # ───────────────────────────────────────────────────────────────────
    # 指标聚合：test 模式（完整集）
    # ───────────────────────────────────────────────────────────────────

    def _aggregate_test(
        self,
        metrics: Dict[str, float],
        b: Dict[str, list],
        pfx: str,
    ) -> None:
        """
        test 模式：计算所有指标，两域分别报告。
        A-RMSE 仅在 sim 域输出，real 域不输出（GT 手抖不可信）。
        """
        # ── 第一层：静态精度 ──────────────────────────────────────
        # A-RMSE：仅 sim
        if pfx == 'sim' and b['angle_errors']:
            a_rmse = math.degrees(math.sqrt(
                float(np.mean(np.array(b['angle_errors']) ** 2))
            ))
            metrics[f'{pfx}/A-RMSE(deg)'] = round(a_rmse, 4)

        # R_center：两域
        if b['center_hits']:
            metrics[f'{pfx}/R_center(%)'] = round(
                float(np.mean(b['center_hits'])) * 100, 2
            )

        # mean_RIoU：两域
        if b['riou_vals']:
            metrics[f'{pfx}/mean_RIoU'] = round(
                float(np.mean(b['riou_vals'])), 4
            )

        # ── 第二层：时序稳定性 ────────────────────────────────────
        # DFR：两域，单位 %/frame
        if b['dfr_vals']:
            metrics[f'{pfx}/DFR(%/frame)'] = round(
                float(np.mean(b['dfr_vals'])) * 100, 4
            )

        # ACI：两域（test 期 real 域也报告，作为泛化参考）
        if b['aci_vals']:
            metrics[f'{pfx}/ACI'] = round(
                float(np.mean(b['aci_vals'])), 4
            )

        # DEP：按 PLC 数据可用性，两域
        if b['dep_vals']:
            metrics[f'{pfx}/DEP(%)'] = round(
                float(np.mean(b['dep_vals'])) * 100, 4
            )

        # ── 第三层：控制适用性 ────────────────────────────────────
        # TDR_w：两域
        if b['tdr_hits']:
            metrics[f'{pfx}/TDR_w{self.ekf_window}(%)'] = round(
                float(np.mean(b['tdr_hits'])) * 100, 2
            )

        # MCML：两域
        if b['mcml_list']:
            max_mcml  = int(max(b['mcml_list']))
            mean_mcml = float(np.mean(b['mcml_list']))
            metrics[f'{pfx}/MCML_max(frames)']  = max_mcml
            metrics[f'{pfx}/MCML_mean(frames)'] = round(mean_mcml, 2)
            metrics[f'{pfx}/MCML_pass(limit={self.mcml_limit})'] = (
                1 if max_mcml <= self.mcml_limit else 0
            )

        # MRF：两域
        if b['mrf_vals']:
            metrics[f'{pfx}/MRF(frames)'] = round(
                float(np.mean(b['mrf_vals'])), 2
            )

    # ───────────────────────────────────────────────────────────────────
    # 单序列计算（核心逻辑）
    # ───────────────────────────────────────────────────────────────────

    def _compute_sequence_metrics(self, frames: List[dict]) -> dict:
        """
        对单个视频序列计算所有中间量。
        frames 已按 frame_id 升序排列（由 compute_metrics 保证）。

        非等间隔抽帧处理：
          gap = frame_id[t] - frame_id[t-1]（实际帧间距）

          DFR 归一化：
            每帧相对变化率 = |Δdiag| / (diag_prev × gap)
            单位：1/frame，跨不同 gap 可比

          ACI 归一化：
            ACI = 1 - |Δγ| / angle_limit
            （gap 在分子分母约分，等价于用物理角度上限直接归一化）

        漏检处理：
          漏检时重置 prev_diag / prev_gamma / prev_frame_id，
          避免跨漏检段计算帧间指标（两段之间的差值无物理意义）。
        """
        angle_errors: List[float] = []
        center_hits:  List[float] = []
        dfr_vals:     List[float] = []
        aci_vals:     List[float] = []
        dep_vals:     List[float] = []
        riou_vals:    List[float] = []
        hit_flags:    List[bool]  = []

        prev_diag:     Optional[float] = None
        prev_gamma:    Optional[float] = None
        prev_frame_id: Optional[int]   = None

        for frame in frames:
            pred    = frame['pred_box']   # np.ndarray (5,) or None
            gt      = frame['gt_box']     # np.ndarray (5,) or None
            plc     = frame['plc_rope']   # float or None
            cur_fid = int(frame['frame_id'])

            # 无 GT 帧：不计漏检，时序继续推进
            if gt is None:
                hit_flags.append(True)
                prev_frame_id = cur_fid
                continue

            is_hit = False

            if pred is not None:
                # RIoU（单帧）
                riou   = compute_riou(pred, gt)
                riou_vals.append(riou)
                is_hit = riou >= self.iou_thresh

                # A-RMSE（单帧，GT 可信性由域决定，此处只收集）
                err = float(angle_diff(
                    np.array([pred[4]]), np.array([gt[4]])
                )[0])
                angle_errors.append(err)

                # R_center（单帧）
                dist = float(np.linalg.norm(
                    obb_center(pred) - obb_center(gt)
                ))
                center_hits.append(float(dist < self.center_thresh_px))

                cur_diag  = obb_diag(pred)
                cur_gamma = float(pred[4])

                # DFR（帧间，按 gap 归一化，漏检后跳过）
                if (prev_diag is not None
                        and prev_frame_id is not None
                        and prev_diag > 1e-6):
                    gap = cur_fid - prev_frame_id
                    if gap > 0:
                        dfr_val = (
                            abs(cur_diag - prev_diag) / (prev_diag * gap)
                        )
                        dfr_vals.append(dfr_val)
                    else:
                        warnings.warn(
                            f"frame_id 不单调：cur={cur_fid}，"
                            f"prev={prev_frame_id}，跳过此帧 DFR。"
                            f"请检查文件命名。",
                            UserWarning, stacklevel=2,
                        )

                # ACI（帧间，gap 约分后等价于 1 - |Δγ|/angle_limit）
                if (prev_gamma is not None
                        and prev_frame_id is not None):
                    gap = cur_fid - prev_frame_id
                    if gap > 0:
                        d_gamma = abs(float(angle_diff(
                            np.array([cur_gamma]),
                            np.array([prev_gamma])
                        )[0]))
                        aci_val = 1.0 - d_gamma / (
                            self.angle_limit_rad + 1e-9
                        )
                        aci_vals.append(float(np.clip(aci_val, 0.0, 1.0)))

                # 更新时序状态
                prev_diag     = cur_diag
                prev_gamma    = cur_gamma
                prev_frame_id = cur_fid

                # DEP（单帧，需 PLC 绳长）
                if plc is not None and float(plc) > 0:
                    z_est   = self.depth_k * (cur_diag ** self.depth_alpha)
                    dep_val = abs(z_est - float(plc)) / float(plc)
                    dep_vals.append(dep_val)

            else:
                # 漏检：重置时序状态，避免跨漏检段计算
                prev_diag     = None
                prev_gamma    = None
                prev_frame_id = cur_fid   # 帧号仍推进，用于后续 gap 计算

            hit_flags.append(is_hit)

        # ── TDR_w：滑动窗口有效检测率 ────────────────────────────
        w        = self.ekf_window
        tdr_hits = [
            any(hit_flags[i: i + w])
            for i in range(max(0, len(hit_flags) - w + 1))
        ]

        # ── MCML：最大连续漏检长度 ───────────────────────────────
        mcml = cur_miss = 0
        for h in hit_flags:
            if not h:
                cur_miss += 1
                mcml = max(mcml, cur_miss)
            else:
                cur_miss = 0

        # ── MRF：漏检恢复帧数 ────────────────────────────────────
        mrf_vals = []
        in_miss  = False
        miss_end = 0
        for i, h in enumerate(hit_flags):
            if not h:
                in_miss  = True
                miss_end = i
            elif in_miss:
                mrf_vals.append(i - miss_end)
                in_miss = False

        return {
            'angle_errors': angle_errors,
            'center_hits':  center_hits,
            'dfr_vals':     dfr_vals,
            'aci_vals':     aci_vals,
            'dep_vals':     dep_vals,
            'riou_vals':    riou_vals,
            'tdr_hits':     tdr_hits,
            'mcml':         mcml,
            'mrf_vals':     mrf_vals,
        }

    # ───────────────────────────────────────────────────────────────────
    # 帧间距诊断
    # ───────────────────────────────────────────────────────────────────

    def _diagnose_gaps(
        self,
        seq_dict: Dict[Tuple[str, str], List[dict]],
        logger,
    ) -> None:
        """
        输出每个序列的帧间距分布，检测非法帧号（非单调）。
        非连续帧（非等间隔抽帧）属正常情况，输出 INFO 而非 WARNING。
        """
        for (domain, seq_id), frames in seq_dict.items():
            fids = np.array([f['frame_id'] for f in frames], dtype=np.int64)
            tag  = f"[{domain}] {seq_id}"

            if len(fids) < 2:
                logger.warning(
                    f"{tag}：只有 {len(fids)} 帧，时序指标无法计算。"
                )
                continue

            gaps = np.diff(fids)
            logger.info(
                f"{tag}：共 {len(fids)} 帧 | "
                f"帧间距 min={int(gaps.min())}, max={int(gaps.max())}, "
                f"mean={float(gaps.mean()):.1f}, std={float(gaps.std()):.1f}"
            )

            # 非法帧号（gap ≤ 0）：错误
            bad = np.where(gaps <= 0)[0]
            if len(bad) > 0:
                logger.error(
                    f"{tag}：存在 {len(bad)} 处帧号非单调，"
                    f"首处：帧 {fids[bad[0]]} → {fids[bad[0]+1]}。"
                    f"请检查文件命名是否符合协议。"
                )

            # 非连续帧（gap > 1）：正常的非等间隔抽帧
            jumps = np.where(gaps > 1)[0]
            if len(jumps) > 0:
                logger.info(
                    f"{tag}：存在 {len(jumps)} 处非连续帧"
                    f"（非等间隔抽帧），DFR/ACI 已按帧间距归一化，结果有效。"
                )

    # ───────────────────────────────────────────────────────────────────
    # 日志格式化
    # ───────────────────────────────────────────────────────────────────

    def _log_metrics(
        self,
        metrics: Dict[str, float],
        all_domains: List[str],
        logger,
    ) -> None:
        """按域和层次打印指标，val/test 模式输出不同子集。"""
        sep = '═' * 64
        logger.info(f'\n{sep}')
        logger.info(
            f'  CraneOBBMetric 评估结果  [{self.mode.upper()} 模式]'
        )
        logger.info(sep)

        if self.mode == 'val':
            # val 模式：按域打印最小指标集
            for domain in all_domains:
                pfx = domain
                logger.info(f'\n  [{domain} 域]')
                logger.info('  ' + '─' * 56)
                val_keys = [
                    f'{pfx}/A-RMSE(deg)',
                    f'{pfx}/ACI',
                    f'{pfx}/R_center(%)',
                ]
                found = False
                for k in val_keys:
                    if k in metrics:
                        logger.info(f'    {k:<48s} {metrics[k]}')
                        found = True
                if not found:
                    logger.info('    （本域无有效数据）')

        else:
            # test 模式：按域和层次打印完整指标
            layer_defs = [
                ('第一层  静态精度（单帧）', [
                    'A-RMSE(deg)', 'R_center(%)', 'mean_RIoU',
                ]),
                ('第二层  时序稳定性（帧间，已按帧间距归一化）', [
                    'DFR(%/frame)', 'ACI', 'DEP(%)',
                ]),
                ('第三层  控制适用性（系统级）', [
                    f'TDR_w{self.ekf_window}(%)',
                    'MCML_max(frames)',
                    'MCML_mean(frames)',
                    f'MCML_pass(limit={self.mcml_limit})',
                    'MRF(frames)',
                ]),
            ]

            for domain in all_domains:
                pfx = domain
                logger.info(f'\n  ┌─ [{domain} 域] {"─"*46}')
                for layer_name, keys in layer_defs:
                    logger.info(f'  │  {layer_name}')
                    found = False
                    for k in keys:
                        full_k = f'{pfx}/{k}'
                        if full_k in metrics:
                            note = ''
                            if k == 'A-RMSE(deg)' and domain != 'sim':
                                note = '（仅 sim 域输出）'
                            logger.info(
                                f'  │    {full_k:<52s} '
                                f'{metrics[full_k]}{note}'
                            )
                            found = True
                    if not found:
                        logger.info(f'  │    （{domain} 域本层无有效数据）')
                logger.info(f'  └{"─"*60}')

        logger.info(f'\n{sep}\n')


# ═══════════════════════════════════════════════════════════════════════
# 离线评估工具
# ═══════════════════════════════════════════════════════════════════════

def evaluate_from_pkl(
    result_pkl: str,
    mode: str = 'test',
    center_thresh_px: float = 15.0,
    ekf_window: int = 10,
    mcml_limit: int = 5,
    angle_limit_deg: float = 35.0,
    depth_k: float = 1000.0,
    depth_alpha: float = -1.5,
    iou_thresh: float = 0.5,
) -> Dict[str, float]:
    """
    从 MMRotate 保存的 result.pkl 离线计算指标。
    pkl 中的数据格式须与 process() 期望的 data_samples 格式一致。

    用法：
        from mmrotate_crane.crane_metrics import evaluate_from_pkl
        metrics = evaluate_from_pkl(
            'work_dirs/crane_eood/test_results.pkl',
            mode='test',
            depth_k=850.0,      # 填入 Webots 标定值
            depth_alpha=-1.42,
        )
        for k, v in metrics.items():
            print(f'{k}: {v}')
    """
    import pickle
    with open(result_pkl, 'rb') as f:
        data_samples = pickle.load(f)

    metric = CraneOBBMetric(
        mode=mode,
        center_thresh_px=center_thresh_px,
        ekf_window=ekf_window,
        mcml_limit=mcml_limit,
        angle_limit_deg=angle_limit_deg,
        depth_k=depth_k,
        depth_alpha=depth_alpha,
        iou_thresh=iou_thresh,
    )

    if isinstance(data_samples, list) and len(data_samples) > 0:
        metric.process({}, data_samples)

    return metric.compute_metrics(metric.results)