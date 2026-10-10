"""Fixed B24 / sigma1.5-ep03 full-VAL OBB AP; no inference or selection.

CPU reference uses the existing offline rotated IoU and MMRotate's VOC07
matching/integration convention. --backend mmrotate additionally verifies every
PR point against the repository native evaluator. No TEST entry point exists.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from crane_project.tools.eval_crane_offline import compute_riou, parse_dota_txt

VERSION = 'port_geometry_ap_v1_fixed_val'
B_SHA = '8f8008c4944807a65ed0f2ee0cc348ea78690d54a4176944b2c9b0ebc83cec23'
HEAD_SHA = '16c2fb448ac4e1c53530b8086d547f6f6ccb8d6b0763a42391c34f9b337982d7'
ANN_SHA = '8bd2e2da86d52b555e2cac5e1098361bd01a4ca09f9df9cb0642e07accc1f6be'
ROWS_SHA = '9faf8e146776b4f7394b3498d04ef25876a52d2bc0637504f9e0195efb7c7981'
COUNTS = dict(real_seq07=226, real_seq14=149, sim_seq10=512)
REQUIRED = ('completion.json', 'protocol.json', 'frozen_selection.json', 'val_rows.jsonl')
THRESHOLDS = (.5, .75)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError('Non-finite JSON: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject)


def box(value, score=False):
    n = 6 if score else 5
    if (not isinstance(value, list) or len(value) != n or
            any(isinstance(x, bool) or not isinstance(x, (float, int)) for x in value)):
        raise ValueError('Expected finite numeric OBB of length %d' % n)
    a = np.asarray(value, dtype=np.float64)
    if (not np.isfinite(a).all() or min(a[2:4]) <= 0 or
            not -math.pi / 2 <= a[4] < math.pi / 2 or
            (score and not 0 <= a[5] <= 1)):
        raise ValueError('Invalid OBB dimensions/angle/score')
    return a


def validate_rows(rows):
    expected = {'%s_%05d' % (seq, i) for seq, n in COUNTS.items()
                for i in range(1 if seq == 'real_seq07' else 0,
                               n + (1 if seq == 'real_seq07' else 0))}
    if (len(rows) != 887 or {r['image'] for r in rows} != expected or
            Counter(r['sequence'] for r in rows) != Counter(COUNTS)):
        raise ValueError('Full VAL identity/count mismatch (missing/duplicate/extra frame)')
    # Native np.argsort uses original image order to resolve score ties.
    if [r['image'] for r in rows] != sorted(expected):
        raise ValueError('Frozen image order differs')
    for r in rows:
        if (r['scale'] != 1.0 or r['domain'] != r['sequence'].split('_')[0] or
                r['image'] != '%s_%05d' % (r['sequence'], r['frame_id']) or
                r['original_b_raw_exact_before_after'] is not True):
            raise ValueError('VAL row identity/scale/frozen state differs')
        box(r['gt'])
        b, m = r['b'], r['midpoint']
        if (b is None) != (m is None):
            raise ValueError('Output coverage differs')
        if (b is None and r['accepted'] is not None) or (b is not None and type(r['accepted']) is not bool):
            raise ValueError('Acceptance decision/missing-frame semantics differ')
        if b is not None:
            box(b, True); box(m, True)
            if b[5] != m[5] or (not r['accepted'] and b != m):
                raise ValueError('Inherited score/fallback differs')


def checked_inputs(eval_dir, ann_dir):
    p, a = Path(eval_dir), Path(ann_dir)
    manifest = read_json(p / 'artifacts.json')
    for name in REQUIRED:
        if sha(p / name) != manifest['files'][name]:
            raise ValueError('Saved artifact SHA differs: ' + name)
    if sha(p / 'val_rows.jsonl') != ROWS_SHA:
        raise ValueError('Not the sealed sigma1.5 online VAL rows')
    c, protocol, selection = [read_json(p / n) for n in REQUIRED[:3]]
    if (c['status'] != 'FROZEN_SIGMA15_VAL_COMPLETE_REVIEW_REQUIRED' or
            c['split'] != 'val' or c['frames'] != 887 or
            c['test_access'] is not False or c['head_updates'] != 0 or
            c['detector_updates'] != 0 or
            c['val_replay']['selected_epoch_box_decision_replay_pass'] is not True or
            protocol['frozen_b']['checkpoint_sha256'] != B_SHA or
            protocol['frozen_b']['selected_epoch'] != 'epoch_24' or
            protocol['counts']['val'] != COUNTS or
            c['data_identity']['annotation_sha256'] != ANN_SHA or
            selection['split'] != 'val' or selection['test_access'] is not False or
            selection['selection_on_test'] is not False):
        raise ValueError('Fixed VAL provenance/selection differs')
    for checkpoint in (c['fixed_checkpoint'], protocol['fixed_checkpoint'],
                       selection['selected_checkpoint']):
        if (checkpoint['sha256'] != HEAD_SHA or checkpoint['epoch'] != 3 or
                checkpoint['sigma_cells'] != 1.5 or checkpoint['updates'] != 2706):
            raise ValueError('Fixed head differs')
    rows = [read_json_line(s) for s in (p / 'val_rows.jsonl').read_text().splitlines() if s.strip()]
    validate_rows(rows)
    anns = sorted(x for x in a.glob('*.txt') if not x.name.startswith('._'))
    digest = hashlib.sha256()
    for path in anns:
        digest.update(path.name.encode()); digest.update(b'\0'); digest.update(path.read_bytes())
    if len(anns) != 887 or digest.hexdigest() != ANN_SHA:
        raise ValueError('Original VAL annotation set SHA/count differs')
    if {x.stem for x in anns} != {r['image'] for r in rows}:
        raise ValueError('Annotation identities differ')
    # CraneDataset does not create difficult/ignored GT; every grab is counted.
    # Reject unexpected annotation semantics instead of silently skipping it.
    conversion_errors, conversion_ious = [], []
    for r in rows:
        path = a / (r['image'] + '.txt')
        lines = [s.split() for s in path.read_text().splitlines() if s.strip()]
        if len(lines) != 1 or len(lines[0]) != 10 or lines[0][8:] != ['grab', '0']:
            raise ValueError('One non-difficult grab GT required: ' + path.name)
        parsed = parse_dota_txt(str(path))
        if len(parsed) != 1:
            raise ValueError('Original GT parse failed: ' + path.name)
        poly = np.asarray([float(x) for x in lines[0][:8]]).reshape(4, 2)
        validate_saved_gt(r['gt'], poly, parsed[0])
        conversion_errors.append(np.abs(np.asarray(r['gt']) - parsed[0]))
        conversion_ious.append(compute_riou(r['gt'], parsed[0]))
    return rows, dict(inputs={n: sha(p / n) for n in REQUIRED + ('artifacts.json',)},
                      annotation_set_sha256=digest.hexdigest(),
                      b_checkpoint_sha256=B_SHA, head_checkpoint_sha256=HEAD_SHA,
                      gt_conversion=dict(evaluation_gt='sealed row GT, never regenerated',
                          max_reparse_abs_error_xywha=np.max(conversion_errors, axis=0).tolist(),
                          min_reparse_riou=min(conversion_ious),
                          validation='both SHA pinned; polygon enclosed within 0.001px; minimum area relative error <=1e-5',
                          note='OpenCV minAreaRect may choose different near-equal-area polygon edges across versions'),
                      checkpoint_identity='saved evaluation metadata; no local weight reload')


def validate_saved_gt(gt, poly, reparsed):
    """Validate the sealed minimum rectangle geometrically, not angle byte equality.

    Near-tied polygon edges can produce different minimum rectangles in OpenCV
    releases. Artifact/annotation SHAs provide exact identity; this independent
    sanity check tests enclosure and minimum area, preserving the saved OBB.
    """
    b = box(gt)
    delta = np.asarray(poly, dtype=np.float64) - b[:2]
    c, s = math.cos(b[4]), math.sin(b[4])
    local = delta @ np.asarray([[c, -s], [s, c]])
    outside = np.max(np.abs(local) - b[2:4] / 2)
    area_error = abs(b[2] * b[3] / (reparsed[2] * reparsed[3]) - 1)
    if outside > .001 or area_error > 1e-5:
        raise ValueError('Saved original GT is not an enclosing minimum rectangle')


def read_json_line(line):
    def reject(value):
        raise ValueError('Non-finite JSON: ' + value)
    return json.loads(line, parse_constant=reject)


def ap11(recalls, precisions):
    """Match mmdet average_precision(..., mode='11points'), float32 accumulator."""
    ap = np.zeros(1, dtype=np.float32)
    for threshold in np.arange(0, 1 + 1e-3, .1):
        eligible = precisions[recalls >= threshold]
        ap[0] += eligible.max() if eligible.size else 0.
    ap /= 11
    return float(ap[0])


def evaluate_images(detections, ground_truth, threshold):
    """One class, max-IoU one-to-one matching, duplicates are FP; no ignored GT.

    General matching is tested with duplicates. Production contract is Top-1
    per image. Native NumPy quicksort ordering (including ties) is retained.
    """
    if len(detections) != len(ground_truth):
        raise ValueError('Image count mismatch')
    all_tp, all_fp, all_scores, matches = [], [], [], []
    for dets, gts in zip(detections, ground_truth):
        tp, fp = np.zeros(len(dets), np.float32), np.zeros(len(dets), np.float32)
        covered = np.zeros(len(gts), bool)
        ious = np.asarray([[compute_riou(d[:5], g) for g in gts] for d in dets])
        for i in np.argsort(-np.asarray([d[5] for d in dets], dtype=np.float32)):
            if not gts:
                fp[i] = 1; continue
            j = int(ious[i].argmax())
            if ious[i, j] >= threshold and not covered[j]:
                tp[i] = 1; covered[j] = True
            else:
                fp[i] = 1
        all_tp.extend(tp); all_fp.extend(fp); all_scores.extend(d[5] for d in dets)
        matches.append(dict(tp=tp.astype(int).tolist(), fp=fp.astype(int).tolist()))
    order = np.argsort(-np.asarray(all_scores, dtype=np.float32))
    tp = np.asarray(all_tp, np.float32)[order].cumsum()
    fp = np.asarray(all_fp, np.float32)[order].cumsum()
    ngt = sum(len(g) for g in ground_truth)
    if ngt == 0:
        raise ValueError('AP undefined without GT')
    # MMRotate divides the float32 cumulative TP by an int->float64 GT array.
    # Keep this precision explicitly across NumPy 1.x/2.x promotion rules.
    recall = tp.astype(np.float64) / ngt
    precision = tp / np.maximum(tp + fp, np.finfo(np.float32).eps)
    ap = ap11(recall, precision)
    if not math.isfinite(ap) or not 0 <= ap <= 1:
        raise ValueError('Invalid AP')
    return dict(ap=ap, ap_percent=100 * ap, num_gts=ngt, num_dets=len(all_scores),
                tp=int(tp[-1]) if len(tp) else 0, fp=int(fp[-1]) if len(fp) else 0,
                recall=recall.tolist(), precision=precision.tolist(),
                score_order=order.tolist(), image_matches=matches)


def verify_native(detections, ground_truth, threshold, reference):
    # Purposefully fail on unavailable/broken native evaluator: never skip mAP.
    from mmrotate.core.evaluation.eval_map import eval_rbbox_map
    det_results = [[np.asarray(ds, dtype=np.float32).reshape(-1, 6)] for ds in detections]
    annotations = [dict(bboxes=np.asarray(gs, dtype=np.float32).reshape(-1, 5),
                        labels=np.zeros(len(gs), dtype=np.int64),
                        bboxes_ignore=np.zeros((0, 5), dtype=np.float32),
                        labels_ignore=np.zeros(0, dtype=np.int64)) for gs in ground_truth]
    mean_ap, result = eval_rbbox_map(det_results, annotations, iou_thr=threshold,
                                   use_07_metric=True, logger='silent', nproc=1)
    r = result[0]
    if (r['num_gts'] != reference['num_gts'] or r['num_dets'] != reference['num_dets'] or
            np.shape(r['recall']) != np.shape(reference['recall']) or
            np.shape(r['precision']) != np.shape(reference['precision']) or
            not np.allclose(r['recall'], reference['recall'], atol=1e-7, rtol=0) or
            not np.allclose(r['precision'], reference['precision'], atol=1e-7, rtol=0) or
            not math.isfinite(mean_ap) or not 0 <= mean_ap <= 1 or
            abs(mean_ap - reference['ap']) > 2e-6):
        raise ValueError('Native/reference AP or PR mismatch; inspect IoU boundary, do not promote')
    return dict(pass_check=True, ap=float(mean_ap), ap_percent=100 * float(mean_ap))


def coverage(rows, method):
    output = [r for r in rows if r[method] is not None]
    correct = sum(np.linalg.norm(np.asarray(r[method][:2]) - r['gt'][:2]) < 15 for r in output)
    return dict(total_frames=len(rows), output_frames=len(output), center_correct_frames=int(correct),
                center_threshold_px=15, center_comparison='strict <',
                output_center_hit_percent=100 * correct / len(output) if output else None,
                output_coverage_percent=100 * len(output) / len(rows),
                full_frame_center_correct_percent=100 * correct / len(rows))


def compute(rows, backend):
    groups, details = {}, {}
    for domain in ('real', 'sim'):
        subset = [r for r in rows if r['domain'] == domain]
        groups[domain] = {}
        for method in ('b', 'midpoint'):
            detections = [[r[method]] if r[method] is not None else [] for r in subset]
            gts = [[r['gt']] for r in subset]
            entry = dict(coverage=coverage(subset, method))
            for threshold in THRESHOLDS:
                label = 'AP%d' % int(threshold * 100)
                reference = evaluate_images(detections, gts, threshold)
                native = verify_native(detections, gts, threshold, reference) if backend == 'mmrotate' else None
                entry[label] = {k: reference[k] for k in ('ap', 'ap_percent', 'num_gts', 'num_dets', 'tp', 'fp')}
                entry[label]['native_verification'] = native
                if native is not None:
                    entry[label]['cpu_reference_ap'] = reference['ap']
                    entry[label]['ap'] = native['ap']
                    entry[label]['ap_percent'] = native['ap_percent']
                details['%s/%s/%s' % (domain, method, label)] = dict(
                    images=[r['image'] for r in subset], **reference)
            # Publish margin diagnostics, not a new threshold sweep.
            ious = [compute_riou(r[method][:5], r['gt']) for r in subset if r[method] is not None]
            entry['min_abs_iou_margin'] = {str(t): min(abs(x - t) for x in ious) for t in THRESHOLDS}
            groups[domain][method] = entry
        groups[domain]['delta_percentage_points'] = {
            label: groups[domain]['midpoint'][label]['ap_percent'] - groups[domain]['b'][label]['ap_percent']
            for label in ('AP50', 'AP75')}
    return groups, details


def run(args):
    rows, provenance = checked_inputs(args.eval_dir, args.ann_dir)
    report = dict(protocol=VERSION, split='val', frames=887, provenance=provenance,
                  head_updates=0, detector_updates=0, inference_calls=0, test_access=False,
                  test_repeatedly_exposed=True, selection_changed=False,
                  metric=dict(class_name='grab', coordinates='original image le90 radians',
                              output_protocol='fixed final Top-1, inherited score, missing frames retained',
                              iou_thresholds=list(THRESHOLDS), integration='VOC2007 11points',
                              matching='per-image max IoU, greedy descending score, duplicates FP',
                              ignored_gt=0, tie_order='NumPy default argsort; sealed image order',
                              iou_backend='eval_crane_offline.compute_riou / OpenCV CPU',
                              native_backend=args.backend, coco_ap_50_95=False),
                  software=dict(python=platform.python_version(), numpy=np.__version__, opencv=cv2.__version__))
    if args.check_only:
        report['status'] = 'STATIC_AP_INPUTS_PASS_NO_INFERENCE_NO_UPDATES'
        details = None
    else:
        report['groups'], details = compute(rows, args.backend)
        report['status'] = ('VAL_AP_NATIVE_VERIFIED' if args.backend == 'mmrotate'
                            else 'VAL_AP_CPU_COMPLETE_NATIVE_CHECK_PENDING')
        if args.backend == 'mmrotate':
            import mmcv, mmdet, mmrotate
            report['software'].update(mmcv=mmcv.__version__, mmdet=mmdet.__version__, mmrotate=mmrotate.__version__)
    # Validate and compute BEFORE creating output; failures leave no empty directory.
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'ap_report.json', report)
    if details is not None:
        write_json(out / 'pr_details.json', details)
    source_paths = [Path(__file__).resolve(), ROOT / 'crane_project/tools/eval_crane_offline.py',
                    ROOT / 'mmrotate/core/evaluation/eval_map.py', ROOT / 'tests/test_port_geometry_ap_v1.py']
    write_json(out / 'artifacts.json', dict(protocol=VERSION,
        files={p.name: sha(p) for p in out.iterdir() if p.is_file()},
        sources={str(p.relative_to(ROOT)): sha(p) for p in source_paths}))
    for domain, value in report.get('groups', {}).items():
        for method in ('b', 'midpoint'):
            print('%s %s AP50=%.4f%% AP75=%.4f%% output=%d/%d' % (
                domain, method, value[method]['AP50']['ap_percent'], value[method]['AP75']['ap_percent'],
                value[method]['coverage']['output_frames'], value[method]['coverage']['total_frames']))
    print('Saved', out / 'ap_report.json', report['status'])


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval-dir', default='work_dirs/port_geometry_midpoint_sigma15_v1_val_eval')
    p.add_argument('--ann-dir', default='crane_project/data/crane_grab_port_day2night_v1/val/annfiles')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--backend', choices=('cpu', 'mmrotate'), default='mmrotate')
    p.add_argument('--check-only', action='store_true')
    return p


if __name__ == '__main__':
    run(parser().parse_args())
