"""Source-VAL-only audit of K1's 0.05 score gate using paired predictions.

The zero-threshold arm is diagnostic.  It does not choose a new threshold,
checkpoint, or inference policy and never reads fixed TEST.
"""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from crane_project.tools.eval_crane_offline import (
    compute_riou, parse_dota_txt, parse_seq_frame)


PROTOCOL = 'k1_source_val_score_gate_audit_v1'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def annotation_set_sha256(paths):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode('utf-8'))
        digest.update(b'\0')
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


def _read(path):
    with open(path, 'rb') as stream:
        return pickle.load(stream)


def _mean(values):
    return None if not values else float(np.mean(values))


def _median(values):
    return None if not values else float(np.median(values))


def build_report(gt_dir, baseline_pkl, permissive_pkl,
                 permissive_identity=None):
    gt_paths = sorted(p for p in Path(gt_dir).glob('*.txt')
                      if not p.name.startswith('._'))
    baseline = _read(baseline_pkl)
    permissive = _read(permissive_pkl)
    if len(gt_paths) != 738 or len(baseline) != 738 or len(permissive) != 738:
        raise ValueError('Expected 738 paired source-VAL frames')
    if permissive_identity is not None:
        identity = json.loads(Path(permissive_identity).read_text(
            encoding='utf-8'))
        observed_keys = [row.get('frame_key') for row in identity.get(
            'runtime_dataset_order', [])]
        if (identity.get('result_count') != 738
                or identity.get('results_sha256') !=
                sha256(permissive_pkl)
                or observed_keys != [path.stem for path in gt_paths]):
            raise ValueError('Permissive result identity does not match VAL')
    groups = {}
    for gt_path, base_row, open_row in zip(gt_paths, baseline, permissive):
        domain, sequence, frame = parse_seq_frame(gt_path.name)
        if domain not in ('real', 'sim'):
            raise ValueError('Unexpected source-VAL domain: ' + gt_path.name)
        gt_boxes = parse_dota_txt(str(gt_path))
        if len(gt_boxes) != 1:
            raise ValueError('Expected one GT box: ' + gt_path.name)
        base = np.asarray(base_row[0])
        opened = np.asarray(open_row[0])
        if len(base) > 1 or len(opened) > 1:
            raise ValueError('K1 max_per_img=1 contract violated: '
                             + gt_path.name)
        if len(base):
            if not len(opened) or not np.allclose(
                    base[0], opened[0], rtol=1e-4, atol=1e-3):
                raise ValueError('Paired inference changed an existing box: '
                                 + gt_path.name)
            if float(base[0, 5]) <= 0.05:
                raise ValueError('Baseline score gate mismatch: '
                                 + gt_path.name)
        elif len(opened) and float(opened[0, 5]) > 0.05 + 1e-6:
            raise ValueError('Permissive rescue exceeds baseline threshold: '
                             + gt_path.name)
        for key in (domain, domain + '/' + sequence):
            b = groups.setdefault(key, dict(frame_count=0, baseline_output=0,
                baseline_missing=0, rescued=0, rescued_center_hit=0,
                rescued_riou_hit=0, rescued_scores=[], rescued_rious=[],
                rescued_gt_short_edges=[]))
            b['frame_count'] += 1
            if len(base):
                b['baseline_output'] += 1
            else:
                b['baseline_missing'] += 1
                if len(opened):
                    b['rescued'] += 1
                    b['rescued_scores'].append(float(opened[0, 5]))
                    b['rescued_gt_short_edges'].append(float(
                        min(gt_boxes[0][2:4])))
                    if float(np.linalg.norm(
                            opened[0, :2] - gt_boxes[0][:2])) < 15.0:
                        b['rescued_center_hit'] += 1
                    riou = compute_riou(opened[0, :5], gt_boxes[0])
                    b['rescued_rious'].append(riou)
                    if riou >= 0.5:
                        b['rescued_riou_hit'] += 1
    summarized = {}
    for key, b in sorted(groups.items()):
        summarized[key] = dict(
            frame_count=b['frame_count'],
            baseline_output_count=b['baseline_output'],
            baseline_missing_count=b['baseline_missing'],
            permissive_output_count=b['baseline_output'] + b['rescued'],
            rescued_output_count=b['rescued'],
            rescued_center_hit_count=b['rescued_center_hit'],
            rescued_riou_hit_count=b['rescued_riou_hit'],
            rescued_center_hit_rate_given_rescue=(
                None if not b['rescued'] else
                b['rescued_center_hit'] / b['rescued']),
            rescued_riou_hit_rate_given_rescue=(
                None if not b['rescued'] else
                b['rescued_riou_hit'] / b['rescued']),
            rescued_score_median=_median(b['rescued_scores']),
            rescued_riou_mean=_mean(b['rescued_rious']),
            rescued_gt_short_edge_median_px=_median(
                b['rescued_gt_short_edges']))
    return dict(
        protocol=PROTOCOL,
        evidence_role='source_validation_diagnosis_only',
        fixed_test_read=False,
        threshold_or_checkpoint_selected=False,
        baseline_score_threshold=0.05,
        diagnostic_score_threshold=0.0,
        center_hit_threshold_px=15.0,
        riou_hit_threshold=0.5,
        sources=dict(
            gt_dir=str(Path(gt_dir).resolve()),
            gt_annotations_sha256=annotation_set_sha256(gt_paths),
            baseline_pkl=str(Path(baseline_pkl).resolve()),
            baseline_pkl_sha256=sha256(baseline_pkl),
            permissive_pkl=str(Path(permissive_pkl).resolve()),
            permissive_pkl_sha256=sha256(permissive_pkl),
            permissive_identity_sha256=(
                None if permissive_identity is None else
                sha256(permissive_identity))),
        groups=summarized)


def markdown(report):
    lines = ['# K1 source VAL 分数门槛诊断', '',
        '固定对照：原 `score_thr=0.05`；诊断臂 `score_thr=0`。'
        '本报告不选择新门槛，也不读取固定 TEST。', '',
        '| 分组 | 帧数 | 原缺测 | 新补框 | 补框中心命中 | 补框 RIoU≥0.5 | 补框分数中位数 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name, row in report['groups'].items():
        lines.append('| {} | {} | {} | {} | {} | {} | {} |'.format(
            name, row['frame_count'], row['baseline_missing_count'],
            row['rescued_output_count'], row['rescued_center_hit_count'],
            row['rescued_riou_hit_count'],
            ('—' if row['rescued_score_median'] is None else
             '{:.6f}'.format(row['rescued_score_median']))))
    lines.append('')
    return '\n'.join(lines)


def write_exact(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding='utf-8') != content:
            raise RuntimeError('Refusing to overwrite different output: '
                               + str(path))
    else:
        path.write_text(content, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt-dir', required=True)
    parser.add_argument('--baseline-pkl', required=True)
    parser.add_argument('--permissive-pkl', required=True)
    parser.add_argument('--permissive-identity', required=True)
    parser.add_argument('--out-json', required=True)
    parser.add_argument('--out-md', required=True)
    args = parser.parse_args()
    report = build_report(args.gt_dir, args.baseline_pkl,
                          args.permissive_pkl, args.permissive_identity)
    write_exact(args.out_json,
                json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    write_exact(args.out_md, markdown(report))
    print('K1_SOURCE_VAL_SCORE_GATE_AUDIT_OK')
    print(json.dumps(report['groups'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
