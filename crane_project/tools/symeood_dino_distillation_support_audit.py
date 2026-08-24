"""Audit source-only support for a unified DINO-to-SymEOOD student.

The input is an all-lane audit collected on source ``train + train_sim``.
No model is trained here.  Ground truth is used only to measure whether the
frozen DINO teacher provides complementary classification/localization
signals, especially when SymEOOD emits a present-but-wrong box.  Complete
sequence counts determine whether a later learned router is identifiable;
the broader instance-level signal determines whether distillation is worth
attempting.
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from crane_project.tools.eval_crane_offline import (
    METRIC_PROTOCOL_VERSION, compute_riou, parse_dota_txt, parse_seq_frame)


PROTOCOL = 'source_only_symeood_dino_distillation_support_v1'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-audit-json', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab')
    parser.add_argument('--out-json', required=True)
    parser.add_argument('--require-frame-count', type=int, default=2781)
    parser.add_argument('--hit-iou', type=float, default=0.5)
    parser.add_argument('--teacher-gain-delta', type=float, default=0.10)
    parser.add_argument('--min-distill-gain-frames', type=int, default=64)
    parser.add_argument('--min-distill-real-sequences', type=int, default=2)
    parser.add_argument(
        '--min-distill-gain-per-real-sequence', type=int, default=8)
    parser.add_argument('--min-present-wrong-frames', type=int, default=32)
    parser.add_argument('--min-present-wrong-per-sequence', type=int, default=8)
    parser.add_argument('--min-router-real-sequences', type=int, default=2)
    return parser.parse_args()


def _box(record, key):
    if key not in record:
        raise RuntimeError('Source audit record is missing ' + key)
    value = record.get(key)
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if (array.size < 6 or not np.isfinite(array[:6]).all()
            or np.any(array[2:4] <= 0.0)):
        raise RuntimeError('Source audit has invalid ' + key)
    return array[:6].copy()


def _riou(box, gt):
    if box is None or gt is None:
        return 0.0
    return float(compute_riou(box[:5], gt[:5]))


def _annotation_path(record, data_root):
    domain, _sequence, _frame = parse_seq_frame(record['filename'])
    split = 'train_sim' if domain == 'sim' else 'train'
    return data_root / split / 'annfiles' / (
        Path(record['filename']).stem + '.txt')


def _ground_truth(record, data_root):
    path = _annotation_path(record, data_root)
    if not path.is_file():
        raise RuntimeError('Source annotation does not exist: ' + os.fspath(path))
    boxes = parse_dota_txt(os.fspath(path))
    if not boxes:
        return None
    return np.asarray(boxes[0], dtype=np.float64)


def _validate(payload, records, data_root, required_frame_count):
    if payload.get('protocol') != 'source_owned_geometry_union_v2':
        raise RuntimeError('Support audit requires an unrouted all-lane audit')
    metadata = dict(payload.get('metadata') or {})
    if metadata.get('fusion_policy') != 'sym_eood_proposal_dino_roi_union':
        raise RuntimeError('Unexpected source fusion policy')
    if bool(metadata.get('conditional_dino_enabled', False)):
        raise RuntimeError('Source collection already used conditional DINO')
    if int(payload.get('frame_count', len(records))) != len(records):
        raise RuntimeError('Source audit frame_count does not match records')
    if len(records) != int(required_frame_count):
        raise RuntimeError(
            'Expected {} source frames, got {}'.format(
                required_frame_count, len(records)))
    if not data_root.is_dir():
        raise RuntimeError('Source data root does not exist')

    seen = set()
    observed_domains = Counter()
    for record in records:
        for key in ('filename', 'sequence', 'frame',
                    'sym_eood_original_box', 'dino_native_box'):
            if key not in record:
                raise RuntimeError('Source audit record is missing ' + key)
        normalized = os.fspath(record['filename']).replace('\\', '/')
        if '/test/' in normalized or '/val/' in normalized:
            raise RuntimeError('Support audit may only read source train data')
        domain, sequence, frame = parse_seq_frame(record['filename'])
        if domain not in ('real', 'sim'):
            raise RuntimeError('Unexpected source domain: ' + domain)
        expected_sequence = '{}_{}'.format(domain, sequence)
        if str(record['sequence']) != expected_sequence:
            raise RuntimeError('Sequence provenance mismatch')
        frame_key = (expected_sequence, int(frame))
        if frame_key in seen:
            raise RuntimeError('Duplicate source frame: {}'.format(frame_key))
        seen.add(frame_key)
        observed_domains[domain] += 1
        _box(record, 'sym_eood_original_box')
        _box(record, 'dino_native_box')
        _ground_truth(record, data_root)
    if not observed_domains['real'] or not observed_domains['sim']:
        raise RuntimeError('Support audit must include real and sim source data')


def _classify_frame(sym, dino, gt, hit_iou, gain_delta):
    sym_iou = _riou(sym, gt)
    dino_iou = _riou(dino, gt)
    sym_hit = bool(sym is not None and sym_iou >= hit_iou)
    dino_hit = bool(dino is not None and dino_iou >= hit_iou)
    delta = float(dino_iou - sym_iou)
    hard_teacher_gain = bool(dino_hit and not sym_hit)
    present_wrong_teacher_gain = bool(
        hard_teacher_gain and sym is not None)
    teacher_gain = bool(hard_teacher_gain or delta >= gain_delta)
    student_preserve = bool(sym_hit and not dino_hit)
    if hard_teacher_gain:
        category = 'hard_teacher_gain'
    elif student_preserve:
        category = 'student_preserve'
    elif not sym_hit and not dino_hit:
        category = 'both_wrong'
    elif abs(delta) < gain_delta:
        category = 'similar'
    elif delta > 0.0:
        category = 'soft_teacher_gain'
    else:
        category = 'soft_student_gain'
    return dict(
        sym_riou=float(sym_iou),
        dino_riou=float(dino_iou),
        delta_riou=delta,
        sym_hit=sym_hit,
        dino_hit=dino_hit,
        teacher_gain=teacher_gain,
        hard_teacher_gain=hard_teacher_gain,
        present_wrong_teacher_gain=present_wrong_teacher_gain,
        student_preserve=student_preserve,
        category=category)


def _summarize(records, data_root, hit_iou, gain_delta):
    per_sequence = defaultdict(lambda: dict(
        domain=None, frame_count=0, categories=Counter(),
        teacher_gain=0, hard_teacher_gain=0,
        present_wrong_teacher_gain=0, student_preserve=0,
        sym_rious=[], dino_rious=[], delta_rious=[]))
    global_categories = Counter()
    frame_rows = []
    for record in records:
        sym = _box(record, 'sym_eood_original_box')
        dino = _box(record, 'dino_native_box')
        gt = _ground_truth(record, data_root)
        domain, sequence_name, frame = parse_seq_frame(record['filename'])
        sequence = '{}_{}'.format(domain, sequence_name)
        result = _classify_frame(sym, dino, gt, hit_iou, gain_delta)
        bucket = per_sequence[sequence]
        bucket['domain'] = domain
        bucket['frame_count'] += 1
        bucket['categories'][result['category']] += 1
        for key in ('teacher_gain', 'hard_teacher_gain',
                    'present_wrong_teacher_gain', 'student_preserve'):
            bucket[key] += int(result[key])
        bucket['sym_rious'].append(result['sym_riou'])
        bucket['dino_rious'].append(result['dino_riou'])
        bucket['delta_rious'].append(result['delta_riou'])
        global_categories[result['category']] += 1
        frame_rows.append(dict(
            frame_key='{}|{}'.format(sequence, int(frame)),
            domain=domain,
            sequence=sequence,
            frame=int(frame),
            sym_present=bool(sym is not None),
            dino_present=bool(dino is not None),
            **result))

    sequence_rows = []
    for sequence in sorted(per_sequence):
        bucket = per_sequence[sequence]
        sequence_rows.append(dict(
            sequence=sequence,
            domain=bucket['domain'],
            frame_count=int(bucket['frame_count']),
            category_counts=dict(bucket['categories']),
            teacher_gain_count=int(bucket['teacher_gain']),
            hard_teacher_gain_count=int(bucket['hard_teacher_gain']),
            present_wrong_teacher_gain_count=int(
                bucket['present_wrong_teacher_gain']),
            student_preserve_count=int(bucket['student_preserve']),
            mean_sym_riou=float(np.mean(bucket['sym_rious'])),
            mean_dino_riou=float(np.mean(bucket['dino_rious'])),
            mean_delta_riou=float(np.mean(bucket['delta_rious']))))
    return frame_rows, sequence_rows, global_categories


def _support_gate(frame_rows, sequence_rows, args):
    teacher_gain_count = sum(int(row['teacher_gain']) for row in frame_rows)
    present_wrong_count = sum(
        int(row['present_wrong_teacher_gain']) for row in frame_rows)
    distill_real_sequences = [
        row['sequence'] for row in sequence_rows
        if (row['domain'] == 'real'
            and row['teacher_gain_count']
            >= args.min_distill_gain_per_real_sequence)]
    router_real_sequences = [
        row['sequence'] for row in sequence_rows
        if (row['domain'] == 'real'
            and row['present_wrong_teacher_gain_count']
            >= args.min_present_wrong_per_sequence)]
    distillation_checks = dict(
        enough_teacher_gain_frames=(
            teacher_gain_count >= args.min_distill_gain_frames),
        enough_real_sequences=(
            len(distill_real_sequences)
            >= args.min_distill_real_sequences))
    router_checks = dict(
        enough_present_wrong_frames=(
            present_wrong_count >= args.min_present_wrong_frames),
        enough_real_sequences=(
            len(router_real_sequences)
            >= args.min_router_real_sequences))
    return dict(
        teacher_gain_count=int(teacher_gain_count),
        present_wrong_teacher_gain_count=int(present_wrong_count),
        distill_real_support_sequences=distill_real_sequences,
        router_real_support_sequences=router_real_sequences,
        distillation_checks=distillation_checks,
        router_checks=router_checks,
        eligible_for_instance_distillation=all(
            distillation_checks.values()),
        eligible_for_learned_router=all(router_checks.values()))


def main():
    args = parse_args()
    source_path = Path(args.source_audit_json)
    payload = json.loads(source_path.read_text())
    records = list(payload.get('records') or [])
    if not records:
        raise RuntimeError('Source all-lane audit has no records')
    data_root = Path(args.data_root)
    _validate(payload, records, data_root, args.require_frame_count)
    frame_rows, sequence_rows, category_counts = _summarize(
        records, data_root, args.hit_iou, args.teacher_gain_delta)
    gate = _support_gate(frame_rows, sequence_rows, args)
    if gate['eligible_for_instance_distillation']:
        recommendation = 'TRAIN_SOURCE_ONLY_INSTANCE_DISTILLATION'
    else:
        recommendation = 'STOP_NO_VERIFIABLE_SOURCE_TEACHER_SUPPORT'
    output = dict(
        protocol=PROTOCOL,
        metric_protocol_version=METRIC_PROTOCOL_VERSION,
        source_splits=['train', 'train_sim'],
        target_data_read=False,
        parameter_update_count=0,
        source_audit_json=os.fspath(source_path),
        source_frame_count=len(records),
        thresholds=dict(
            hit_iou=float(args.hit_iou),
            teacher_gain_delta=float(args.teacher_gain_delta),
            min_distill_gain_frames=int(args.min_distill_gain_frames),
            min_distill_real_sequences=int(
                args.min_distill_real_sequences),
            min_distill_gain_per_real_sequence=int(
                args.min_distill_gain_per_real_sequence),
            min_present_wrong_frames=int(args.min_present_wrong_frames),
            min_present_wrong_per_sequence=int(
                args.min_present_wrong_per_sequence),
            min_router_real_sequences=int(
                args.min_router_real_sequences)),
        category_counts=dict(category_counts),
        support_gate=gate,
        recommended_next_stage=recommendation,
        sequence_summaries=sequence_rows,
        frame_rows=frame_rows)
    output_path = Path(args.out_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + '.tmp')
    temporary.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    temporary.replace(output_path)
    print('[distill-support] frames={}'.format(len(records)))
    print('[distill-support] categories={}'.format(dict(category_counts)))
    print('[distill-support] teacher_gain={} present_wrong={}'.format(
        gate['teacher_gain_count'],
        gate['present_wrong_teacher_gain_count']))
    print('[distill-support] distillation_eligible={}'.format(
        gate['eligible_for_instance_distillation']))
    print('[distill-support] router_eligible={}'.format(
        gate['eligible_for_learned_router']))
    print('[distill-support] next={}'.format(recommendation))


if __name__ == '__main__':
    main()
