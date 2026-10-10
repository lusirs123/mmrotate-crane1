"""Explicitly authorized exposed TEST AP for fixed B24 / sigma1.5-ep03.

Reuse sealed full1440 predictions and the unchanged VAL AP arithmetic. Never
infer, fit, scan checkpoints, alter output scores or use TEST for selection.
The original VAL-only entry point remains unchanged.
"""
import argparse
from collections import Counter
import hashlib
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from crane_project.tools import eval_port_geometry_ap_v1 as ap

VERSION = 'port_geometry_ap_test_v1_fixed_exposed_test'
COUNTS = dict(real_seq03=200, real_seq04=668, sim_seq09=572)
ROWS_SHA = '13c4cfa7f1f67a7bc2cad36da1ec067adbeae74080b387d39efe39fdc41a788d'
ANN_SHA = 'e0dbb1bd8aea7209314d8ed60bc44e965550ed606135cc0016e1075d717de13e'
VAL_AP_SHA = '6e682f64824fe0b68bbe6b49276ebf7d147fe9891a70b0fe3d7f7757b4bc7d34'
REQUIRED = ('completion.json', 'protocol.json', 'frozen_selection.json', 'test_rows.jsonl')
DEFAULT_EVAL_DIR = 'docs/detection/ap_evidence_20261010/test_source'


def checked_eval_dir(eval_dir):
    # Explicit archive paths remain supported; never guess another experiment.
    p = Path(eval_dir)
    missing = [name for name in REQUIRED + ('artifacts.json',) if not (p / name).is_file()]
    if missing:
        raise FileNotFoundError(
            'Incomplete sealed TEST AP input directory: %s; missing: %s. '
            'Sync %s and use --eval-dir %s, or specify the complete original '
            'archived directory. No inference or input reconstruction is needed.' %
            (p, ', '.join(missing), DEFAULT_EVAL_DIR, DEFAULT_EVAL_DIR))
    return p


def checked_val_gate(path):
    # Gate checked BEFORE any TEST file is opened.
    if ap.sha(path) != VAL_AP_SHA:
        raise ValueError('Not the sealed native-verified VAL AP report')
    r = ap.read_json(path)
    if (r['status'] != 'VAL_AP_NATIVE_VERIFIED' or r['protocol'] != ap.VERSION or
            r['split'] != 'val' or r['frames'] != 887 or r['test_access'] is not False or
            r['selection_changed'] is not False or
            r['metric']['integration'] != 'VOC2007 11points' or
            r['metric']['iou_thresholds'] != list(ap.THRESHOLDS) or
            r['metric']['native_backend'] != 'mmrotate' or
            r['provenance']['b_checkpoint_sha256'] != ap.B_SHA or
            r['provenance']['head_checkpoint_sha256'] != ap.HEAD_SHA or
            r['provenance']['annotation_set_sha256'] != ap.ANN_SHA or
            r['provenance']['inputs']['val_rows.jsonl'] != ap.ROWS_SHA):
        raise ValueError('Native VAL gate identity/protocol differs')
    for domain in ('real', 'sim'):
        for method in ('b', 'midpoint'):
            for label in ('AP50', 'AP75'):
                entry = r['groups'][domain][method][label]
                if (entry['native_verification']['pass_check'] is not True or
                        not math.isfinite(entry['ap']) or not 0 <= entry['ap'] <= 1 or
                        entry['ap'] != entry['native_verification']['ap'] or
                        entry['ap'] != entry['cpu_reference_ap']):
                    raise ValueError('VAL native verification incomplete')
    return r


def validate_rows(rows):
    expected = {'%s_%05d' % (seq, i) for seq, n in COUNTS.items()
                for i in range(int(seq == 'real_seq03'), n + int(seq == 'real_seq03'))}
    if (len(rows) != 1440 or {r['image'] for r in rows} != expected or
            Counter(r['sequence'] for r in rows) != Counter(COUNTS) or
            [r['image'] for r in rows] != sorted(expected)):
        raise ValueError('Complete standard-scale TEST image identities/order differ')
    for r in rows:
        if (r['domain'] != r['sequence'].split('_')[0] or r['scale'] != 1. or
                r['image'] != '%s_%05d' % (r['sequence'], r['frame_id']) or
                r['original_b_raw_exact_before_after'] is not True):
            raise ValueError('TEST row identity/scale/frozen state differs')
        ap.box(r['gt'])
        b, m = r['b'], r['midpoint']
        if (b is None) != (m is None):
            raise ValueError('TEST output count differs')
        if b is None:
            if r['accepted'] is not None:
                raise ValueError('Missing frame acceptance must be null')
        else:
            ap.box(b, True); ap.box(m, True)
            if (type(r['accepted']) is not bool or b[5] != m[5] or
                    (not r['accepted'] and b != m)):
                raise ValueError('TEST score/fallback/acceptance differs')


def checked_inputs(eval_dir, ann_dir, val_ap_report):
    val = checked_val_gate(val_ap_report)
    p, a = checked_eval_dir(eval_dir), Path(ann_dir)
    manifest = ap.read_json(p / 'artifacts.json')
    for name in REQUIRED:
        if ap.sha(p / name) != manifest['files'][name]:
            raise ValueError('Saved TEST artifact SHA differs: ' + name)
    if ap.sha(p / 'test_rows.jsonl') != ROWS_SHA:
        raise ValueError('Not the sealed sigma1.5 full TEST rows')
    c, protocol, selection = [ap.read_json(p / n) for n in REQUIRED[:3]]
    if (c['status'] != 'FROZEN_SIGMA15_TEST_COMPLETE_REVIEW_REQUIRED' or
            c['split'] != 'test' or c['frames'] != 1440 or
            c['test_access'] is not True or c['selection_on_test'] is not False or
            c['head_updates'] != 0 or c['detector_updates'] != 0 or
            c['state_before'] != c['state_after'] or
            c['data_identity']['annotation_sha256'] != ANN_SHA or
            c['data_identity']['sequence_counts'] != COUNTS or
            protocol['counts']['test'] != COUNTS or
            protocol['test_annotation_sha256'] != ANN_SHA or
            protocol['frozen_b']['checkpoint_sha256'] != ap.B_SHA or
            protocol['frozen_b']['selected_epoch'] != 'epoch_24' or
            selection['split'] != 'val' or selection['test_access'] is not False or
            selection['selection_on_test'] is not False or
            c['online_val_gate']['pass_online_val'] is not True or
            c['online_val_gate']['frames'] != 887 or
            c['online_val_gate']['val_rows_sha256'] != ap.ROWS_SHA or
            c['online_val_gate']['completion_sha256'] != val['provenance']['inputs']['completion.json'] or
            ap.sha(p / 'frozen_selection.json') != val['provenance']['inputs']['frozen_selection.json'] or
            ap.sha(p / 'protocol.json') != val['provenance']['inputs']['protocol.json']):
        raise ValueError('Frozen TEST and native VAL provenance differ')
    for ckpt in (c['fixed_checkpoint'], protocol['fixed_checkpoint'], selection['selected_checkpoint']):
        if (ckpt['sha256'] != ap.HEAD_SHA or ckpt['epoch'] != 3 or
                ckpt['updates'] != 2706 or ckpt['sigma_cells'] != 1.5):
            raise ValueError('Fixed head differs')
    rows = [ap.read_json_line(s) for s in (p / 'test_rows.jsonl').read_text().splitlines() if s.strip()]
    validate_rows(rows)
    anns = sorted(x for x in a.glob('*.txt') if not x.name.startswith('._'))
    digest = hashlib.sha256()
    for path in anns:
        digest.update(path.name.encode()); digest.update(b'\0'); digest.update(path.read_bytes())
    if (len(anns) != 1440 or digest.hexdigest() != ANN_SHA or
            {x.stem for x in anns} != {r['image'] for r in rows}):
        raise ValueError('Original complete TEST annotation SHA/identities differ')
    errors, ious = [], []
    for r in rows:
        path = a / (r['image'] + '.txt')
        lines = [s.split() for s in path.read_text().splitlines() if s.strip()]
        if len(lines) != 1 or len(lines[0]) != 10 or lines[0][8:] != ['grab', '0']:
            raise ValueError('One non-difficult grab GT required: ' + path.name)
        parsed = ap.parse_dota_txt(str(path))
        if len(parsed) != 1:
            raise ValueError('Original TEST GT parse failed')
        poly = ap.np.asarray([float(x) for x in lines[0][:8]]).reshape(4, 2)
        ap.validate_saved_gt(r['gt'], poly, parsed[0])
        errors.append(ap.np.abs(ap.np.asarray(r['gt']) - parsed[0]))
        ious.append(ap.compute_riou(r['gt'], parsed[0]))
    return rows, dict(inputs={n: ap.sha(p / n) for n in REQUIRED + ('artifacts.json',)},
        eval_directory=str(p.resolve()),
        native_val_ap_report_sha256=ap.sha(val_ap_report), annotation_set_sha256=digest.hexdigest(),
        b_checkpoint_sha256=ap.B_SHA, head_checkpoint_sha256=ap.HEAD_SHA,
        checkpoint_identity='sealed evaluation metadata, no weight reload',
        gt_conversion=dict(evaluation_gt='sealed row GT, never regenerated',
            max_reparse_abs_error_xywha=ap.np.max(errors, axis=0).tolist(), min_reparse_riou=min(ious),
            validation='same VAL enclosure 0.001px/minimum area relative error 1e-5 checks'))


def run(args):
    rows, provenance = checked_inputs(args.eval_dir, args.ann_dir, args.val_ap_report)
    report = dict(protocol=VERSION, split='test', frames=1440, provenance=provenance,
        head_updates=0, detector_updates=0, inference_calls=0, test_access=True,
        test_repeatedly_exposed=True, selection_changed=False, selection_on_test=False,
        metric=dict(class_name='grab', coordinates='original image le90 radians',
            output_protocol='fixed final Top-1, inherited score, missing frames retained',
            iou_thresholds=list(ap.THRESHOLDS), integration='VOC2007 11points',
            matching='per-image max IoU, greedy descending score, duplicates FP',
            ignored_gt=0, tie_order='NumPy default argsort; sealed image order',
            iou_backend='eval_crane_offline.compute_riou / OpenCV CPU',
            native_backend=args.backend, coco_ap_50_95=False),
        software=dict(python=platform.python_version(), numpy=ap.np.__version__, opencv=ap.cv2.__version__))
    if args.check_only:
        report['status'] = 'STATIC_TEST_AP_INPUTS_PASS_NO_INFERENCE_NO_UPDATES'
        details = None
    else:
        report['groups'], details = ap.compute(rows, args.backend)
        report['status'] = ('TEST_AP_NATIVE_VERIFIED' if args.backend == 'mmrotate'
                            else 'TEST_AP_CPU_COMPLETE_NATIVE_CHECK_PENDING')
        if args.backend == 'mmrotate':
            import mmcv, mmdet, mmrotate
            report['software'].update(mmcv=mmcv.__version__, mmdet=mmdet.__version__, mmrotate=mmrotate.__version__)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    ap.write_json(out / 'ap_report.json', report)
    if details is not None:
        ap.write_json(out / 'pr_details.json', details)
    sources = [Path(__file__).resolve(), Path(ap.__file__).resolve(),
               ROOT / 'crane_project/tools/eval_crane_offline.py',
               ROOT / 'mmrotate/core/evaluation/eval_map.py', ROOT / 'tests/test_port_geometry_ap_test_v1.py']
    ap.write_json(out / 'artifacts.json', dict(protocol=VERSION,
        files={p.name: ap.sha(p) for p in out.iterdir() if p.is_file()},
        sources={str(p.relative_to(ROOT)): ap.sha(p) for p in sources}))
    for domain, value in report.get('groups', {}).items():
        for method in ('b', 'midpoint'):
            print('TEST %s %s AP50=%.4f%% AP75=%.4f%% output=%d/%d' % (
                domain, method, value[method]['AP50']['ap_percent'], value[method]['AP75']['ap_percent'],
                value[method]['coverage']['output_frames'], value[method]['coverage']['total_frames']))
    print('Saved', out / 'ap_report.json', report['status'])


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval-dir', default=DEFAULT_EVAL_DIR,
                   help='Complete sealed TEST input directory; explicit archived paths are supported')
    p.add_argument('--ann-dir', default='crane_project/data/crane_grab_port_day2night_v1/test/annfiles')
    p.add_argument('--val-ap-report', default='docs/detection/ap_evidence_20261010/val_native/ap_report.json')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--backend', choices=('cpu', 'mmrotate'), default='mmrotate')
    p.add_argument('--check-only', action='store_true')
    return p


if __name__ == '__main__':
    run(parser().parse_args())
