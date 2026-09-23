"""Explain historical K1 metric changes from bound, existing predictions.

This audit never runs inference or selects a checkpoint.  Historical
axis-aligned overlap is reproduced only as a clearly labelled comparison.
"""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from crane_project.tools.eval_crane_offline import (
    compute_riou, obb_center, parse_dota_txt)


PROTOCOL = 'k1_metric_denominator_compatibility_audit_v1'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(paths):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode('utf-8'))
        digest.update(b'\0')
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


def historical_approx_riou(box1, box2):
    """Exact historical approximation; not a geometrically valid RIoU."""
    area1 = float(box1[2]) * float(box1[3])
    area2 = float(box2[2]) * float(box2[3])
    cx_diff = abs(float(box1[0]) - float(box2[0]))
    cy_diff = abs(float(box1[1]) - float(box2[1]))
    ow = max(0.0, (float(box1[2]) + float(box2[2])) / 2 - cx_diff)
    oh = max(0.0, (float(box1[3]) + float(box2[3])) / 2 - cy_diff)
    inter = ow * oh
    return float(np.clip(inter / (area1 + area2 - inter + 1e-6), 0, 1))


def _mean(values):
    return float(np.mean(values)) if values else None


def audit(gt_dir, pred_dir, pred_pkl):
    gt_paths = sorted(path for path in Path(gt_dir).glob('*.txt')
                      if not path.name.startswith('._'))
    if len(gt_paths) != 992:
        raise ValueError('Expected 992 fixed-TEST annotations; got %d'
                         % len(gt_paths))
    with open(pred_pkl, 'rb') as stream:
        results = pickle.load(stream)
    if len(results) != len(gt_paths):
        raise ValueError('Prediction pickle does not match GT frame count')
    pred_paths = [Path(pred_dir) / path.name for path in gt_paths]
    if not all(path.is_file() for path in pred_paths):
        raise ValueError('Missing or mismatched DOTA prediction files')
    buckets = {}
    max_center_export_delta_px = 0.0
    conversion_ious = []
    for gt_path, pred_path, result in zip(gt_paths, pred_paths, results):
        domain = gt_path.name.split('_', 1)[0]
        if domain not in ('real', 'sim'):
            raise ValueError('Unknown domain: ' + gt_path.name)
        gt_boxes = parse_dota_txt(str(gt_path))
        pred_boxes = parse_dota_txt(str(pred_path))
        raw_boxes = np.asarray(result[0])
        if len(pred_boxes) != len(raw_boxes):
            raise ValueError('PKL/DOTA count mismatch: ' + gt_path.name)
        if len(pred_boxes):
            delta = np.linalg.norm(raw_boxes[0, :2] - pred_boxes[0][:2])
            max_center_export_delta_px = max(
                max_center_export_delta_px, float(delta))
            if delta > 0.02:
                raise ValueError('PKL/DOTA center mismatch: ' + gt_path.name)
            conversion_iou = compute_riou(
                raw_boxes[0, :5], pred_boxes[0])
            if conversion_iou < 0.99:
                raise ValueError('PKL/DOTA geometry mismatch: '
                                 + gt_path.name)
            conversion_ious.append(conversion_iou)
        b = buckets.setdefault(domain, dict(gt=0, output=0, center_hit=0,
                                            exact=[], approx=[]))
        if not gt_boxes:
            continue
        b['gt'] += 1
        if not pred_boxes:
            continue
        b['output'] += 1
        if float(np.linalg.norm(obb_center(pred_boxes[0]) -
                                obb_center(gt_boxes[0]))) < 15.0:
            b['center_hit'] += 1
        b['exact'].append(compute_riou(pred_boxes[0], gt_boxes[0]))
        b['approx'].append(historical_approx_riou(
            pred_boxes[0], gt_boxes[0]))
    by_domain = {}
    for domain, b in sorted(buckets.items()):
        if not b['gt'] or not b['output']:
            raise ValueError('No GT or output in domain ' + domain)
        by_domain[domain] = dict(
            gt_positive_frame_count=b['gt'],
            output_frame_count=b['output'],
            missing_output_frame_count=b['gt'] - b['output'],
            output_coverage=b['output'] / b['gt'],
            center_hit_count=b['center_hit'],
            center_hit_given_output=b['center_hit'] / b['output'],
            center_hit_all_gt_frames=b['center_hit'] / b['gt'],
            historical_approx_riou_given_output=_mean(b['approx']),
            exact_riou_given_output=_mean(b['exact']),
            exact_riou_all_gt_frames=sum(b['exact']) / b['gt'])
    return dict(
        protocol=PROTOCOL,
        evidence_role='fixed_test_metric_compatibility_diagnosis_only',
        model_inference_performed=False,
        checkpoint_selection_performed=False,
        sources=dict(
            gt_dir=str(Path(gt_dir).resolve()),
            gt_annotations_sha256=directory_sha256(gt_paths),
            pred_dir=str(Path(pred_dir).resolve()),
            pred_dota_sha256=directory_sha256(pred_paths),
            pred_pkl=str(Path(pred_pkl).resolve()),
            pred_pkl_sha256=sha256(pred_pkl)),
        center_threshold_px=15.0,
        max_pkl_to_dota_center_delta_px=max_center_export_delta_px,
        min_pkl_to_dota_geometry_iou=min(conversion_ious),
        mean_pkl_to_dota_geometry_iou=_mean(conversion_ious),
        by_domain=by_domain)


def markdown(report):
    lines = [
        '# K1 指标口径核对', '',
        '该报告只重算已保存的固定 TEST 预测，不选择权重。', '',
        '| 域 | GT 帧 | 有输出 | 输出覆盖率 | 中心命中/有输出 | 中心命中/全部 | 历史近似 IoU/有输出 | 旋转 IoU/有输出 | 旋转 IoU/全部 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for domain, b in report['by_domain'].items():
        lines.append('| {} | {} | {} | {:.2%} | {:.2%} | {:.2%} | {:.4f} | {:.4f} | {:.4f} |'.format(
            domain, b['gt_positive_frame_count'], b['output_frame_count'],
            b['output_coverage'], b['center_hit_given_output'],
            b['center_hit_all_gt_frames'],
            b['historical_approx_riou_given_output'],
            b['exact_riou_given_output'], b['exact_riou_all_gt_frames']))
    lines.extend([
        '',
        '历史近似 IoU 只为解释旧记录而复现，不能作为修订版旋转 IoU。'
        '中心命中率和 IoU 应始终注明分母；缺测帧不应被当成有效输出。',
        '',
        '预测 PKL SHA256：`{}`。'.format(
            report['sources']['pred_pkl_sha256']),
        'PKL 到 DOTA 的最小几何 IoU：`{:.6f}`。'.format(
            report['min_pkl_to_dota_geometry_iou']),
        'GT 标注集 SHA256：`{}`。'.format(
            report['sources']['gt_annotations_sha256']),
        ''
    ])
    return '\n'.join(lines)


def write_exact(path, content):
    path = Path(path)
    if path.exists():
        if path.read_text(encoding='utf-8') != content:
            raise RuntimeError('Refusing to overwrite different output: '
                               + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt-dir', required=True)
    parser.add_argument('--pred-dir', required=True)
    parser.add_argument('--pred-pkl', required=True)
    parser.add_argument('--out-json', required=True)
    parser.add_argument('--out-md', required=True)
    args = parser.parse_args()
    report = audit(args.gt_dir, args.pred_dir, args.pred_pkl)
    write_exact(args.out_json,
                json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    write_exact(args.out_md, markdown(report))
    print('K1_METRIC_DENOMINATOR_AUDIT_OK')
    print(json.dumps(report['by_domain'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
