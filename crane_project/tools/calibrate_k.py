#!/usr/bin/env python3
"""
离线标定脚本：测量不同抓斗的平台:顶梁尺寸比（k值）

功能：
1. 交互式标注：读取顶梁框，允许用户手动标注平台框
2. 自动计算每个样本的k值（平台框尺寸 / 顶梁框尺寸）
3. 统计分析：按抓斗类型/序列分组，输出k的范围和中位数
4. 输出k-jitter训练区间：[k_min, k_max]

使用方法：
    # 标注所有序列（train + train_sim），每个序列10张
    python tools/calibrate_k.py \
        --data-root crane_project/data/crane_grab \
        --samples-per-seq 10 \
        --output calibration_results.json

    # 仅标注指定序列
    python tools/calibrate_k.py \
        --sequences real_seq01 real_seq02 sim_seq09 \
        --samples-per-seq 10 \
        --output calibration_subset.json

    # 仅标注real域
    python tools/calibrate_k.py \
        --splits train \
        --samples-per-seq 10 \
        --output calibration_real.json

    # 继续之前的标注
    python tools/calibrate_k.py \
        --resume calibration_results.json

按键：
    - 鼠标拖拽：标注平台OBB（4个角点）
    - s: 保存当前标注
    - n: 跳过当前样本
    - q: 退出并保存结果
"""

import argparse
import json
import os
import os.path as osp
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='标定平台:顶梁尺寸比k')
    parser.add_argument(
        '--data-root',
        type=str,
        default='crane_project/data/crane_grab',
        help='数据根目录')
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=['train', 'train_sim'],
        help='数据集划分，可指定多个（默认包含train和train_sim）')
    parser.add_argument(
        '--samples-per-seq',
        type=int,
        default=10,
        help='每个序列标注的样本数量')
    parser.add_argument(
        '--output',
        type=str,
        default='calibration_results.json',
        help='输出文件路径')
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='从已有标定结果继续')
    parser.add_argument(
        '--sequences',
        type=str,
        nargs='+',
        default=None,
        help='仅标注指定序列（如 real_seq01 sim_seq09），默认标注所有')
    return parser.parse_args()


def load_obb_annotation(ann_path: str) -> np.ndarray:
    """加载顶梁OBB标注

    Returns:
        corners: (4, 2) array of corner points [x, y]
    """
    with open(ann_path, 'r') as f:
        line = f.readline().strip()

    parts = line.split()
    coords = [float(x) for x in parts[:8]]
    corners = np.array(coords).reshape(4, 2)
    return corners


def compute_obb_size(corners: np.ndarray) -> Tuple[float, float, float]:
    """计算OBB的宽、高和对角线长度

    Args:
        corners: (4, 2) corner points

    Returns:
        width: OBB宽度（短边）
        height: OBB高度（长边）
        diagonal: 对角线长度 sqrt(w^2 + h^2)
    """
    # 计算4条边的长度
    edge_lengths = []
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        edge_lengths.append(np.linalg.norm(p2 - p1))

    # 相对边长度
    w1, h1 = edge_lengths[0], edge_lengths[1]
    w2, h2 = edge_lengths[2], edge_lengths[3]

    # 取平均并确定宽高
    w = (w1 + w2) / 2
    h = (h1 + h2) / 2

    # 宽度是短边
    width = min(w, h)
    height = max(w, h)
    diagonal = np.sqrt(width**2 + height**2)

    return width, height, diagonal


def draw_obb(img: np.ndarray, corners: np.ndarray, color: Tuple[int, int, int],
             thickness: int = 2, label: str = None):
    """绘制OBB"""
    corners_int = corners.astype(np.int32)
    cv2.polylines(img, [corners_int], isClosed=True, color=color, thickness=thickness)

    # 绘制角点
    for pt in corners_int:
        cv2.circle(img, tuple(pt), 3, color, -1)

    # 绘制标签
    if label:
        cx = int(np.mean(corners[:, 0]))
        cy = int(np.mean(corners[:, 1]))
        cv2.putText(img, label, (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)


class PlatformAnnotator:
    """平台框标注工具"""

    def __init__(self, window_name: str = "Calibrate K"):
        self.window_name = window_name
        self.points = []
        self.current_img = None
        self.result = None

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append([x, y])
                self.draw()

    def draw(self):
        """绘制当前状态"""
        if self.current_img is None:
            return

        img_show = self.current_img.copy()

        # 绘制已标注的点
        for i, pt in enumerate(self.points):
            cv2.circle(img_show, tuple(pt), 5, (0, 255, 0), -1)
            cv2.putText(img_show, str(i+1), (pt[0]+10, pt[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 绘制连线
        if len(self.points) > 1:
            pts = np.array(self.points, dtype=np.int32)
            cv2.polylines(img_show, [pts], isClosed=False,
                         color=(0, 255, 0), thickness=2)

        # 如果有4个点，闭合
        if len(self.points) == 4:
            pts = np.array(self.points, dtype=np.int32)
            cv2.polylines(img_show, [pts], isClosed=True,
                         color=(0, 255, 0), thickness=2)

        cv2.imshow(self.window_name, img_show)

    def annotate(self, img: np.ndarray, beam_corners: np.ndarray) -> np.ndarray:
        """标注平台框

        Args:
            img: 输入图像
            beam_corners: 顶梁框角点 (4, 2)

        Returns:
            platform_corners: 平台框角点 (4, 2)，如果跳过则返回None
        """
        self.current_img = img.copy()
        self.points = []
        self.result = None

        # 绘制顶梁框
        draw_obb(self.current_img, beam_corners, (255, 0, 0), 2, "Beam")

        # 显示指引
        h, w = img.shape[:2]
        cv2.putText(self.current_img, "Click 4 corners for Platform OBB",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(self.current_img, "s: save | n: skip | q: quit | r: reset",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.draw()

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):  # 保存
                if len(self.points) == 4:
                    self.result = np.array(self.points, dtype=np.float32)
                    break
                else:
                    print(f"需要标注4个角点，当前只有{len(self.points)}个")

            elif key == ord('n'):  # 跳过
                self.result = None
                break

            elif key == ord('q'):  # 退出
                self.result = 'quit'
                break

            elif key == ord('r'):  # 重置
                self.points = []
                self.current_img = img.copy()
                draw_obb(self.current_img, beam_corners, (255, 0, 0), 2, "Beam")
                cv2.putText(self.current_img, "Click 4 corners for Platform OBB",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(self.current_img, "s: save | n: skip | q: quit | r: reset",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                self.draw()

        return self.result


def collect_sequences_from_splits(data_root: str, splits: List[str],
                                  filter_sequences: List[str] = None) -> Dict[str, Dict]:
    """收集所有split中的序列信息

    Args:
        data_root: 数据根目录
        splits: 数据集划分列表 ['train', 'train_sim', ...]
        filter_sequences: 仅保留指定序列（可选）

    Returns:
        {
            'real_seq01': {
                'split': 'train',
                'ann_dir': '...',
                'img_dir': '...',
                'files': ['...', ...]
            },
            ...
        }
    """
    sequences = {}

    for split in splits:
        ann_dir = osp.join(data_root, split, 'annfiles')

        # train_sim的图像在train/images中，其他split在自己的images中
        if split == 'train_sim':
            img_dir = osp.join(data_root, 'train', 'images')
        else:
            img_dir = osp.join(data_root, split, 'images')

        if not osp.exists(ann_dir):
            print(f"警告：跳过不存在的目录 {ann_dir}")
            continue

        # 按序列分组
        seq_files = defaultdict(list)
        for fname in os.listdir(ann_dir):
            if fname.endswith('.txt'):
                parts = fname.split('_')
                if len(parts) >= 2:
                    seq_name = f"{parts[0]}_{parts[1]}"
                    seq_files[seq_name].append(fname)

        # 添加到总字典
        for seq_name, files in seq_files.items():
            if filter_sequences and seq_name not in filter_sequences:
                continue

            if seq_name in sequences:
                print(f"警告：序列 {seq_name} 在多个split中出现")
                continue

            sequences[seq_name] = {
                'split': split,
                'ann_dir': ann_dir,
                'img_dir': img_dir,
                'files': sorted(files)
            }

    return sequences


def sample_files_per_sequence(sequences: Dict[str, Dict],
                               samples_per_seq: int) -> List[Tuple[str, str, str]]:
    """每个序列均匀采样固定数量文件

    Args:
        sequences: 序列信息字典
        samples_per_seq: 每个序列采样数量

    Returns:
        List of (seq_name, ann_path, img_path)
    """
    sampled = []

    for seq_name, seq_info in sorted(sequences.items()):
        files = seq_info['files']
        ann_dir = seq_info['ann_dir']
        img_dir = seq_info['img_dir']

        n = len(files)
        n_samples = min(samples_per_seq, n)

        # 均匀采样索引
        if n_samples == n:
            indices = list(range(n))
        else:
            indices = np.linspace(0, n-1, n_samples, dtype=int)

        for idx in indices:
            fname = files[idx]
            ann_path = osp.join(ann_dir, fname)
            img_name = fname.replace('.txt', '.jpg')
            img_path = osp.join(img_dir, img_name)
            sampled.append((seq_name, ann_path, img_path))

    return sampled


def main():
    args = parse_args()

    # 收集所有序列
    print(f"\n收集序列信息...")
    sequences = collect_sequences_from_splits(
        args.data_root,
        args.splits,
        filter_sequences=args.sequences
    )

    if len(sequences) == 0:
        print("错误：未找到任何序列")
        return

    print(f"\n找到 {len(sequences)} 个序列:")
    for seq_name, seq_info in sorted(sequences.items()):
        print(f"  {seq_name}: {len(seq_info['files'])} 帧 (split: {seq_info['split']})")

    # 采样文件：每个序列固定数量
    print(f"\n每个序列采样 {args.samples_per_seq} 个样本...")
    sampled_files = sample_files_per_sequence(sequences, args.samples_per_seq)
    print(f"采样完成，共 {len(sampled_files)} 个样本\n")

    # 加载已有结果
    calibration_data = []
    if args.resume and osp.exists(args.resume):
        with open(args.resume, 'r') as f:
            result = json.load(f)
            calibration_data = result.get('samples', [])
        print(f"从 {args.resume} 加载了 {len(calibration_data)} 个已标注样本\n")

    # 标注工具
    annotator = PlatformAnnotator()

    # 标注循环
    for i, (seq_name, ann_path, img_path) in enumerate(sampled_files):
        fname = osp.basename(ann_path)

        # 检查是否已标注
        if any(s['filename'] == fname for s in calibration_data):
            print(f"[{i+1}/{len(sampled_files)}] 跳过已标注: {fname}")
            continue

        if not osp.exists(img_path):
            print(f"[{i+1}/{len(sampled_files)}] 图像不存在: {img_path}")
            continue

        print(f"\n[{i+1}/{len(sampled_files)}] 序列: {seq_name} | 标注: {fname}")

        # 加载数据
        img = cv2.imread(img_path)
        beam_corners = load_obb_annotation(ann_path)

        # 标注平台框
        platform_corners = annotator.annotate(img, beam_corners)

        if platform_corners == 'quit':
            print("用户退出")
            break

        if platform_corners is None:
            print("跳过")
            continue

        # 计算k值
        beam_w, beam_h, beam_diag = compute_obb_size(beam_corners)
        plat_w, plat_h, plat_diag = compute_obb_size(platform_corners)

        k_w = plat_w / beam_w
        k_h = plat_h / beam_h
        k_diag = plat_diag / beam_diag

        # 提取序列信息
        parts = fname.split('_')
        domain = parts[0]  # real or sim
        seq_name = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else "unknown"

        # 保存数据
        sample_data = {
            'filename': fname,
            'domain': domain,
            'sequence': seq_name,
            'beam_corners': beam_corners.tolist(),
            'platform_corners': platform_corners.tolist(),
            'beam_size': {'w': float(beam_w), 'h': float(beam_h), 'diag': float(beam_diag)},
            'platform_size': {'w': float(plat_w), 'h': float(plat_h), 'diag': float(plat_diag)},
            'k_values': {'k_w': float(k_w), 'k_h': float(k_h), 'k_diag': float(k_diag)}
        }
        calibration_data.append(sample_data)

        print(f"  顶梁尺寸: w={beam_w:.1f}, h={beam_h:.1f}, diag={beam_diag:.1f}")
        print(f"  平台尺寸: w={plat_w:.1f}, h={plat_h:.1f}, diag={plat_diag:.1f}")
        print(f"  k值: k_w={k_w:.2f}, k_h={k_h:.2f}, k_diag={k_diag:.2f}")

    cv2.destroyAllWindows()

    if len(calibration_data) == 0:
        print("\n没有标注数据")
        return

    # 统计分析
    print(f"\n{'='*60}")
    print("标定结果统计")
    print(f"{'='*60}")

    # 按序列分组统计
    seq_stats = defaultdict(lambda: {'k_w': [], 'k_h': [], 'k_diag': [], 'domain': None})
    for sample in calibration_data:
        seq = sample['sequence']
        seq_stats[seq]['k_w'].append(sample['k_values']['k_w'])
        seq_stats[seq]['k_h'].append(sample['k_values']['k_h'])
        seq_stats[seq]['k_diag'].append(sample['k_values']['k_diag'])
        seq_stats[seq]['domain'] = sample['domain']

    # 输出各序列统计（按序列分组，识别不同抓斗）
    print("\n各序列（抓斗）统计:")
    for seq in sorted(seq_stats.keys()):
        stats = seq_stats[seq]
        k_w_arr = np.array(stats['k_w'])
        k_h_arr = np.array(stats['k_h'])
        k_diag_arr = np.array(stats['k_diag'])
        domain = stats['domain']

        print(f"\n序列 {seq} [{domain}] (n={len(k_w_arr)}):")
        print(f"  k_w:    中位数={np.median(k_w_arr):.2f}, 范围=[{k_w_arr.min():.2f}, {k_w_arr.max():.2f}]")
        print(f"  k_h:    中位数={np.median(k_h_arr):.2f}, 范围=[{k_h_arr.min():.2f}, {k_h_arr.max():.2f}]")
        print(f"  k_diag: 中位数={np.median(k_diag_arr):.2f}, 范围=[{k_diag_arr.min():.2f}, {k_diag_arr.max():.2f}]")

    # 全局统计
    all_k_w = [s['k_values']['k_w'] for s in calibration_data]
    all_k_h = [s['k_values']['k_h'] for s in calibration_data]
    all_k_diag = [s['k_values']['k_diag'] for s in calibration_data]

    k_w_global = np.array(all_k_w)
    k_h_global = np.array(all_k_h)
    k_diag_global = np.array(all_k_diag)

    print(f"\n{'='*60}")
    print(f"全局统计（覆盖所有抓斗，n={len(calibration_data)}）:")
    print(f"  k_w:    中位数={np.median(k_w_global):.2f}, 范围=[{k_w_global.min():.2f}, {k_w_global.max():.2f}]")
    print(f"  k_h:    中位数={np.median(k_h_global):.2f}, 范围=[{k_h_global.min():.2f}, {k_h_global.max():.2f}]")
    print(f"  k_diag: 中位数={np.median(k_diag_global):.2f}, 范围=[{k_diag_global.min():.2f}, {k_diag_global.max():.2f}]")

    # 推荐的k-jitter区间（加20%余量，覆盖所有抓斗）
    safety_margin = 1.2
    k_w_min, k_w_max = k_w_global.min() / safety_margin, k_w_global.max() * safety_margin
    k_h_min, k_h_max = k_h_global.min() / safety_margin, k_h_global.max() * safety_margin
    k_diag_min, k_diag_max = k_diag_global.min() / safety_margin, k_diag_global.max() * safety_margin

    # 使用对角线k作为各向同性外扩的默认值
    k_median = float(np.median(k_diag_global))
    k_range = [float(k_diag_min), float(k_diag_max)]

    print(f"\n{'='*60}")
    print("推荐的k-jitter训练区间（含20%安全余量，覆盖所有抓斗）:")
    print(f"  k_w:    [{k_w_min:.2f}, {k_w_max:.2f}]")
    print(f"  k_h:    [{k_h_min:.2f}, {k_h_max:.2f}]")
    print(f"  k_diag: [{k_diag_min:.2f}, {k_diag_max:.2f}]  <-- 推荐用于各向同性外扩")
    print(f"\n推理时默认k值: {k_median:.2f} (全局中位数)")
    print(f"\n说明:")
    print(f"  - k-jitter区间覆盖了所有标定的抓斗类型")
    print(f"  - 训练时每个iteration从区间随机采样k，实现跨抓斗鲁棒性")
    print(f"  - 推理时：已知抓斗型号用其序列中位数k，未知用全局中位数")
    print(f"{'='*60}\n")

    # 保存结果
    result = {
        'metadata': {
            'data_root': args.data_root,
            'splits': args.splits,
            'samples_per_seq': args.samples_per_seq,
            'num_samples': len(calibration_data),
            'num_sequences': len(seq_stats)
        },
        'k_jitter_range': {
            'k_diag': k_range,
            'k_w': [float(k_w_min), float(k_w_max)],
            'k_h': [float(k_h_min), float(k_h_max)]
        },
        'k_default': k_median,
        'sequence_stats': {
            seq: {
                'domain': stats['domain'],
                'n_samples': len(stats['k_w']),
                'k_w_median': float(np.median(stats['k_w'])),
                'k_h_median': float(np.median(stats['k_h'])),
                'k_diag_median': float(np.median(stats['k_diag'])),
                'k_w_range': [float(np.min(stats['k_w'])), float(np.max(stats['k_w']))],
                'k_h_range': [float(np.min(stats['k_h'])), float(np.max(stats['k_h']))],
                'k_diag_range': [float(np.min(stats['k_diag'])), float(np.max(stats['k_diag']))]
            }
            for seq, stats in seq_stats.items()
        },
        'samples': calibration_data
    }

    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"结果已保存到: {args.output}")
    print(f"标注样本数: {len(calibration_data)}")
    print(f"\n可使用 --resume {args.output} 继续标注更多样本")


if __name__ == '__main__':
    main()
