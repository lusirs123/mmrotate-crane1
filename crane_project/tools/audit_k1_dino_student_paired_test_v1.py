"""Read-only paired fixed-TEST audit of K1 and its DINO feature student.

The report describes an already exposed TEST set. It never selects a model,
threshold, checkpoint, or training sample, and it performs no inference.
"""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from crane_project.tools.eval_crane_offline import (
    METRIC_PROTOCOL_VERSION, compute_riou, parse_dota_txt, parse_seq_frame)


PROTOCOL = 'k1_dino_student_paired_fixed_test_audit_v1'
CENTER_THRESHOLD_PX = 15.0
RIOU_HIT_THRESHOLD = 0.5
GEOMETRY_TIE_TOLERANCE = 1e-6


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
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


def _load_arm(label, spec, gt_paths, gt_sha, expected_frame_count):
    report_path, pkl_path, pred_dir = (Path(part) for part in spec)
    report = json.loads(report_path.read_text(encoding='utf-8'))
    actual_pkl_sha = sha256(pkl_path)
    if (report.get('protocol') != 'crane_ckpt_sweep_final_test_v2'
            or report.get('metric_protocol_version') !=
            METRIC_PROTOCOL_VERSION
            or report.get('frame_count') != expected_frame_count
            or report.get('center_thresh_px') != CENTER_THRESHOLD_PX
            or report.get('results_pkl_sha256') != actual_pkl_sha
            or report.get('gt_annotations_sha256') != gt_sha):
        raise ValueError(label + ' final TEST report identity mismatch')
    pred_paths = [pred_dir / path.name for path in gt_paths]
    if not all(path.is_file() for path in pred_paths):
        raise ValueError(label + ' missing per-frame DOTA predictions')
    with pkl_path.open('rb') as stream:
        predictions = pickle.load(stream)
    if len(predictions) != expected_frame_count:
        raise ValueError(label + ' PKL frame count mismatch')
    boxes = []
    for gt_path, pred_path, result in zip(gt_paths, pred_paths, predictions):
        dota = parse_dota_txt(str(pred_path))
        if len(result) != 1:
            raise ValueError(label + ' expected one prediction class: '
                             + gt_path.name)
        raw = np.asarray(result[0])
        if raw.ndim != 2 or raw.shape[1] < 6 or len(raw) > 1:
            raise ValueError(label + ' expected at most one OBB: '
                             + gt_path.name)
        if len(dota) != len(raw):
            raise ValueError(label + ' PKL/DOTA count mismatch: '
                             + gt_path.name)
        if len(raw):
            if (not np.isfinite(raw[0, :6]).all()
                    or np.any(raw[0, 2:4] <= 0)):
                raise ValueError(label + ' invalid OBB: ' + gt_path.name)
            if np.linalg.norm(raw[0, :2] - dota[0][:2]) > 0.02:
                raise ValueError(label + ' PKL/DOTA center mismatch: '
                                 + gt_path.name)
            if compute_riou(raw[0, :5], dota[0]) < 0.99:
                raise ValueError(label + ' PKL/DOTA geometry mismatch: '
                                 + gt_path.name)
        # The official offline evaluator reads these DOTA polygons.  Use the
        # same geometry for paired metrics after verifying the PKL export.
        boxes.append(dota[0].astype(float).tolist() if dota else None)
    return dict(boxes=boxes, report=report, sources=dict(
        final_report=str(report_path.resolve()),
        final_report_sha256=sha256(report_path),
        checkpoint_sha256=report.get('checkpoint_sha256'),
        config_sha256=report.get('config_sha256'),
        results_pkl=str(pkl_path.resolve()),
        results_pkl_sha256=actual_pkl_sha,
        pred_dir=str(pred_dir.resolve()),
        pred_dota_sha256=directory_sha256(pred_paths)))


def _metric(box, gt):
    if box is None:
        return dict(output=False, center_hit=False, riou_hit=False,
                    center_error_px=None, riou=None)
    error = float(np.linalg.norm(np.asarray(box[:2]) - gt[:2]))
    riou = compute_riou(box[:5], gt)
    return dict(output=True, center_hit=error < CENTER_THRESHOLD_PX,
                riou_hit=riou >= RIOU_HIT_THRESHOLD,
                center_error_px=error, riou=riou)


def _summarize(rows):
    count = len(rows)
    if count == 0:
        raise ValueError('Cannot summarize an empty group')
    paired = [row for row in rows if row['baseline']['output']
              and row['student']['output']]
    result = dict(frame_count=count, methods={})
    for name in ('baseline', 'student'):
        output = [row[name] for row in rows if row[name]['output']]
        hits = sum(row[name]['center_hit'] for row in rows)
        result['methods'][name] = dict(
            output_frame_count=len(output),
            missing_output_frame_count=count - len(output),
            output_coverage=len(output) / count,
            center_hit_count=hits,
            center_hit_given_output=(hits / len(output) if output else None),
            center_hit_all_gt_frames=hits / count,
            riou_hit_count=sum(row[name]['riou_hit'] for row in rows),
            exact_riou_given_output=(
                sum(item['riou'] for item in output) / len(output)
                if output else None),
            exact_riou_all_gt_frames=(
                sum(item['riou'] for item in output) / count))
    result['paired'] = dict(
        both_output_count=len(paired),
        baseline_only_output_count=sum(
            row['baseline']['output'] and not row['student']['output']
            for row in rows),
        student_only_output_count=sum(
            row['student']['output'] and not row['baseline']['output']
            for row in rows),
        neither_output_count=sum(
            not row['baseline']['output'] and not row['student']['output']
            for row in rows),
        baseline_only_center_hit_count=sum(
            row['baseline']['center_hit'] and not row['student']['center_hit']
            for row in rows),
        student_only_center_hit_count=sum(
            row['student']['center_hit'] and not row['baseline']['center_hit']
            for row in rows),
        both_center_hit_count=sum(
            row['baseline']['center_hit'] and row['student']['center_hit']
            for row in rows),
        neither_center_hit_count=sum(
            not row['baseline']['center_hit']
            and not row['student']['center_hit'] for row in rows),
        common_output=dict(
            frame_count=len(paired),
            baseline_mean_riou=(sum(row['baseline']['riou'] for row in paired)
                                / len(paired) if paired else None),
            student_mean_riou=(sum(row['student']['riou'] for row in paired)
                               / len(paired) if paired else None),
            student_riou_better_count=sum(
                row['student']['riou'] - row['baseline']['riou'] >
                GEOMETRY_TIE_TOLERANCE for row in paired),
            baseline_riou_better_count=sum(
                row['baseline']['riou'] - row['student']['riou'] >
                GEOMETRY_TIE_TOLERANCE for row in paired),
            riou_tie_count=sum(
                abs(row['student']['riou'] - row['baseline']['riou'])
                <= GEOMETRY_TIE_TOLERANCE for row in paired)))
    return result


def build_report(gt_dir, baseline_spec, student_spec,
                 expected_frame_count=992):
    gt_paths = sorted(path for path in Path(gt_dir).glob('*.txt')
                      if not path.name.startswith('._'))
    if len(gt_paths) != expected_frame_count:
        raise ValueError('Expected %d fixed-TEST annotations; found %d' % (
            expected_frame_count, len(gt_paths)))
    gt_sha = directory_sha256(gt_paths)
    baseline = _load_arm('baseline', baseline_spec, gt_paths, gt_sha,
                         expected_frame_count)
    student = _load_arm('student', student_spec, gt_paths, gt_sha,
                        expected_frame_count)
    rows = []
    groups = {}
    for path, base_box, student_box in zip(
            gt_paths, baseline['boxes'], student['boxes']):
        gt_boxes = parse_dota_txt(str(path))
        if len(gt_boxes) != 1:
            raise ValueError('Expected one GT OBB: ' + path.name)
        domain, sequence, frame = parse_seq_frame(path.name)
        if domain not in ('real', 'sim'):
            raise ValueError('Unexpected TEST domain: ' + path.name)
        row = dict(frame_key=path.stem, domain=domain, sequence=sequence,
                   frame=frame, gt_short_edge_px=float(
                       min(gt_boxes[0][2:4])),
                   baseline=_metric(base_box, gt_boxes[0]),
                   student=_metric(student_box, gt_boxes[0]))
        rows.append(row)
        for key in (domain, domain + '/' + sequence):
            groups.setdefault(key, []).append(row)
    return dict(protocol=PROTOCOL,
                evidence_role='post_exposure_fixed_test_diagnosis_only',
                model_inference_performed=False,
                checkpoint_or_threshold_selection_performed=False,
                center_threshold_px=CENTER_THRESHOLD_PX,
                riou_hit_threshold=RIOU_HIT_THRESHOLD,
                common_frame_riou_tolerance=GEOMETRY_TIE_TOLERANCE,
                gt_source=dict(path=str(Path(gt_dir).resolve()),
                               annotation_count=len(gt_paths),
                               annotations_sha256=gt_sha),
                baseline_source=baseline['sources'],
                student_source=student['sources'],
                groups={key: _summarize(value)
                        for key, value in sorted(groups.items())},
                frames=rows)


def markdown(report):
    lines = ['# K1 与 DINO 语义蒸馏学生：固定 TEST 配对审计', '',
             '本报告只分析既有预测；TEST 已暴露，不能据此选权、调阈值或设计训练样本。',
             '中心命中：误差 < 15 像素。无输出在全帧指标中计为未命中；'
             '有输出时的误差只在相应条件样本上计算。', '',
             '| 分组 | 方法 | GT 帧 | 输出覆盖率 | 输出后中心命中率 |'
             ' 全帧中心命中率 | 全帧 RIoU |',
             '|---|---|---:|---:|---:|---:|---:|']
    for key, group in report['groups'].items():
        for name, label in (('baseline', 'K1'), ('student', '蒸馏学生')):
            row = group['methods'][name]
            conditional = row['center_hit_given_output']
            lines.append('| {} | {} | {} | {:.2%} | {} | {:.2%} | {:.4f} |'.format(
                key, label, group['frame_count'], row['output_coverage'],
                '—' if conditional is None else '{:.2%}'.format(conditional),
                row['center_hit_all_gt_frames'],
                row['exact_riou_all_gt_frames']))
    lines += ['', '| 分组 | 双方均输出 | 仅 K1 输出 | 仅学生输出 | 双方均缺测 |'
              ' K1 独有中心命中 | 学生独有中心命中 | 共同输出 K1 RIoU |'
              ' 共同输出学生 RIoU |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key, group in report['groups'].items():
        pair = group['paired']
        common = pair['common_output']
        fmt = lambda value: '—' if value is None else '{:.4f}'.format(value)
        lines.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            key, pair['both_output_count'],
            pair['baseline_only_output_count'],
            pair['student_only_output_count'],
            pair['neither_output_count'],
            pair['baseline_only_center_hit_count'],
            pair['student_only_center_hit_count'],
            fmt(common['baseline_mean_riou']),
            fmt(common['student_mean_riou'])))
    lines += ['', '共同输出帧的 RIoU 才是配对几何比较；各自有输出帧的平均值'
              '具有不同样本集合，不能直接解释为几何改善。', '',
              'GT SHA256：`{}`'.format(report['gt_source']['annotations_sha256']),
              'K1 TEST 报告 SHA256：`{}`'.format(
                  report['baseline_source']['final_report_sha256']),
              '学生 TEST 报告 SHA256：`{}`'.format(
                  report['student_source']['final_report_sha256']), '']
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gt-dir', required=True)
    for arm in ('baseline', 'student'):
        parser.add_argument('--' + arm + '-report', required=True)
        parser.add_argument('--' + arm + '-pkl', required=True)
        parser.add_argument('--' + arm + '-pred-dir', required=True)
    parser.add_argument('--out-json', required=True)
    parser.add_argument('--out-md', required=True)
    args = parser.parse_args()
    report = build_report(
        args.gt_dir,
        (args.baseline_report, args.baseline_pkl, args.baseline_pred_dir),
        (args.student_report, args.student_pkl, args.student_pred_dir))
    write_exact(args.out_json,
                json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    write_exact(args.out_md, markdown(report))
    print('K1_DINO_STUDENT_PAIRED_TEST_AUDIT_OK')
    print(json.dumps(report['groups'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
