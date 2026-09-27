"""Read-only teacher/student capability-gap audit on source VAL.

This audit compares one frozen teacher frame-outcome report with one K1
student prediction PKL on the same 738 source-VAL frames.  It is deliberately
frame-level: a teacher report containing only final top-1 outcomes cannot prove
anything about teacher background responses or student candidate stages.

The tool never reads fixed TEST, runs no model inference, changes no threshold,
and does not select a checkpoint.  The native-DINO output report and the
DINOv2 feature-cache teacher are different evidence roles; the report records
the supplied ``teacher_role`` explicitly.
"""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from crane_project.tools.eval_crane_offline import compute_riou, parse_dota_txt


PROTOCOL = 'teacher_student_source_val_gap_audit_v1'
FRAME_COUNT = 738
RIoU_THRESHOLD = 0.5
CENTER_THRESHOLD_PX = 15.0


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def frame_key(seq, frame):
    return '{}|{}'.format(seq, int(frame))


def load_teacher_outcomes(path, alpha=None):
    """Load a source ``frame_outcomes`` list from a teacher JSON report."""
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    selected_alpha = payload.get('selected_alpha') if alpha is None else alpha
    source = payload.get('source', payload)
    candidates = source.get('candidates') if isinstance(source, dict) else None
    if candidates is not None:
        if selected_alpha is None:
            raise ValueError('Teacher report has candidates but no alpha')
        matches = [row for row in candidates
                   if abs(float(row.get('alpha')) - float(selected_alpha)) < 1e-12]
        if len(matches) != 1:
            raise ValueError('Expected one teacher candidate for alpha {}'.format(
                selected_alpha))
        node = matches[0]
    else:
        node = source
    rows = node.get('frame_outcomes')
    if not isinstance(rows, list) or len(rows) != FRAME_COUNT:
        raise ValueError('Teacher report must contain 738 frame_outcomes')
    result = {}
    for row in rows:
        if not {'seq', 'frame', 'top1_hit'}.issubset(row):
            raise ValueError('Teacher frame outcome lacks seq/frame/top1_hit')
        key = frame_key(row['seq'], row['frame'])
        if key in result:
            raise ValueError('Duplicate teacher frame: ' + key)
        result[key] = dict(
            key=key, seq=str(row['seq']), frame=int(row['frame']),
            correct=bool(row['top1_hit']),
            riou=(None if row.get('top1_riou') is None else
                  float(row['top1_riou'])),
            score=(None if row.get('top1_score') is None else
                   float(row['top1_score'])),
            output_observed=('top1_score' in row and row.get('top1_score') is not None),
            best_usable_rank=row.get('best_usable_rank'))
    return result, dict(
        path=str(Path(path).resolve()), sha256=sha256(path),
        source_protocol=payload.get('protocol'),
        selected_alpha=selected_alpha,
        selected_checkpoint=payload.get('selected_checkpoint'),
        selected_checkpoint_sha256=payload.get('selected_checkpoint_sha256'))


def load_student_predictions(path):
    with Path(path).open('rb') as stream:
        rows = pickle.load(stream)
    if not isinstance(rows, list) or len(rows) != FRAME_COUNT:
        raise ValueError('Student PKL must contain 738 frames')
    provenance_path = Path(str(path) + '.provenance.json')
    provenance = None
    identity_status = 'order_assumed_from_sorted_source_val_annotations'
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
        observed_hash = provenance.get('results_sha256',
                                      provenance.get('results_pkl_sha256'))
        if observed_hash not in (None, sha256(path)):
            raise ValueError('Student provenance does not match prediction PKL')
        identity_status = 'provenance_sidecar_present'
    return rows, provenance, identity_status


def student_outcome(prediction, gt):
    """Normalize K1 one-class max-per-image output and score it against GT."""
    if not isinstance(prediction, (list, tuple)) or len(prediction) != 1:
        raise ValueError('Expected one-class prediction list per frame')
    boxes = np.asarray(prediction[0])
    if boxes.size == 0:
        return dict(output=False, correct=False, riou=None, score=None,
                    center_distance_px=None)
    if boxes.ndim != 2 or boxes.shape[1] != 6 or len(boxes) > 1:
        raise ValueError('Student prediction violates max_per_img=1 contract')
    box = boxes[0].astype(float)
    riou = float(compute_riou(box[:5], gt))
    distance = float(np.linalg.norm(box[:2] - gt[:2]))
    return dict(output=True, correct=riou >= RIoU_THRESHOLD, riou=riou,
                score=float(box[5]), center_distance_px=distance)


def build_report(gt_dir, teacher_json, student_pkl, teacher_role,
                 student_config=None, student_checkpoint=None, alpha=None):
    gt_paths = sorted(p for p in Path(gt_dir).glob('*.txt')
                      if not p.name.startswith('._'))
    if len(gt_paths) != FRAME_COUNT:
        raise ValueError('Expected 738 source-VAL annotations')
    if {p.stem.split('_', 1)[0] for p in gt_paths} != {'real', 'sim'}:
        raise ValueError('Unexpected source-VAL frame identities')
    teacher, teacher_meta = load_teacher_outcomes(teacher_json, alpha)
    student_rows, student_provenance, student_identity_status = \
        load_student_predictions(student_pkl)
    rows = []
    for gt_path, prediction in zip(gt_paths, student_rows):
        parts = gt_path.stem.rsplit('_', 2)
        if len(parts) != 3:
            raise ValueError('Cannot parse source frame: ' + gt_path.name)
        seq = parts[0] + '_' + parts[1]
        frame = int(parts[2])
        key = frame_key(seq, frame)
        if key not in teacher:
            raise ValueError('Teacher frame missing: ' + key)
        gt_boxes = parse_dota_txt(str(gt_path))
        if len(gt_boxes) != 1:
            raise ValueError('Expected one GT box: ' + gt_path.name)
        student = student_outcome(prediction, np.asarray(gt_boxes[0], dtype=float))
        t = teacher[key]
        rows.append(dict(frame=key, domain=seq.split('_', 1)[0],
                         teacher=t, student=student,
                         comparison=(
                             'teacher_correct_student_wrong'
                             if t['correct'] and not student['correct'] else
                             'teacher_wrong_student_correct'
                             if not t['correct'] and student['correct'] else
                             'both_correct' if t['correct'] else 'both_wrong')))
    counts = {}
    for row in rows:
        counts[row['comparison']] = counts.get(row['comparison'], 0) + 1
    domains = {}
    for domain in ('real', 'sim'):
        subset = [row for row in rows if row['domain'] == domain]
        domains[domain] = {k: sum(r['comparison'] == k for r in subset)
                           for k in ('teacher_correct_student_wrong',
                                     'teacher_wrong_student_correct',
                                     'both_correct', 'both_wrong')}
    teacher_correct = sum(r['teacher']['correct'] for r in rows)
    student_correct = sum(r['student']['correct'] for r in rows)
    return dict(
        protocol=PROTOCOL, evidence_role='source_val_only',
        fixed_test_read=False, inference_executed=False,
        thresholds=dict(riou=RIoU_THRESHOLD, center_px=CENTER_THRESHOLD_PX),
        teacher_role=teacher_role, teacher=teacher_meta,
        student=dict(path=str(Path(student_pkl).resolve()),
                     sha256=sha256(student_pkl), config=student_config,
                     checkpoint=student_checkpoint,
                     provenance=student_provenance,
                     identity_status=student_identity_status),
        frame_count=len(rows), teacher_correct_count=teacher_correct,
        student_correct_count=student_correct, comparison_counts=counts,
        comparison_by_domain=domains, frames=rows,
        limitations=[
            'Teacher input contains final frame outcomes only; it does not expose teacher features, background responses, or student candidate stages.',
            'When no student provenance sidecar is present, PKL order is paired with sorted source-VAL annotation stems by contract and is not independently image-verified.',
            'Teacher output_observed means a top-1 outcome was recorded, not necessarily deployment output after an unavailable teacher threshold.',
            'A native-DINO detector report is not the same artifact as the frozen DINOv2 feature-cache teacher used by V1-V5.',
            'Frame-level complementarity does not prove a distillation loss will improve the student.'
        ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gt-dir', required=True)
    parser.add_argument('--teacher-json', required=True)
    parser.add_argument('--student-pkl', required=True)
    parser.add_argument('--teacher-role', required=True)
    parser.add_argument('--student-config')
    parser.add_argument('--student-checkpoint')
    parser.add_argument('--teacher-alpha', type=float)
    parser.add_argument('--out-json', required=True)
    args = parser.parse_args()
    output = Path(args.out_json)
    if output.exists():
        raise FileExistsError(output)
    report = build_report(args.gt_dir, args.teacher_json, args.student_pkl,
                          args.teacher_role, args.student_config,
                          args.student_checkpoint, args.teacher_alpha)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(dict(protocol=PROTOCOL,
                          comparison_counts=report['comparison_counts'],
                          comparison_by_domain=report['comparison_by_domain']),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
