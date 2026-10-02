#!/usr/bin/env python3
"""CPU-only frozen-B VAL cache and TRAIN structure-label readiness checks.

No detector inference, fitting, optimizer, TEST access, or cache writes. The
score curve is a descriptive comparator, not a chosen deployment policy.
Only allowlisted TRAIN axis folders are opened for structure supervision.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'crane_project/data/crane_grab_port_day2night_v1'
B_CONFIG = ROOT / 'crane_project/configs/crane_symeood_k1_port_day2night_aug_b_v1.py'
# Copy of the already retained B identity; independent of the running F-S job.
FROZEN_B = dict(
    selected_epoch='epoch_24',
    checkpoint_sha256='8f8008c4944807a65ed0f2ee0cc348ea78690d54a4176944b2c9b0ebc83cec23',
    config_sha256='9da972b7010540e12b2501d1c400db13c159381f495f539d6315e7adb498e345',
    pkl_sha256='cbfa341c53faf859a4eb96ab62a2470611cde0493896f50fcc7f805f53020725',
    selection_sha256='ccaa55376d3d40175c86386fbb7c6e2e8db54a1e13b9b1611783c7255fb54352',
    annotation_sha256='8bd2e2da86d52b555e2cac5e1098361bd01a4ca09f9df9cb0642e07accc1f6be')
VAL_COUNTS = dict(real_seq07=226, real_seq14=149, sim_seq10=512)
TRAIN_SPLITS = dict(real_seq01='train', real_seq05='train', real_seq06='train',
                    real_seq12='train', real_seq13='train', sim_seq08='train_sim')
TRAIN_COUNTS = dict(real_seq01=339, real_seq05=560, real_seq06=466,
                    real_seq12=141, real_seq13=304, sim_seq08=748)
NATIVE_K = dict(real_seq01=2.7, real_seq05=1.2, real_seq06=1.5,
                real_seq12=2.1, real_seq13=2.1)
LEGACY_TRAIN = ('real_seq01', 'real_seq05', 'real_seq06')
COVERAGES = (1.0, .99, .98, .95, .90, .80)
ERROR_LIMITS = dict(center_px=15., size_max_relative=.10, angle_deg=3.)
# Conversion tolerances cover the native polygon's 0.1px rounding, not accuracy.
AXIS_TOL = dict(center_px=.15, endpoint_px=.20, length_px=.20,
                short_px=.20, angle_deg=.20)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def files(directory, suffix):
    return sorted(p for p in Path(directory).glob('*' + suffix)
                  if not p.name.startswith('._'))


def set_sha(paths):
    """Same filename/NUL/content convention as ckpt_sweep annotation SHA."""
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode('utf-8')); h.update(b'\0')
        with p.open('rb') as f:
            for b in iter(lambda: f.read(1024 * 1024), b''):
                h.update(b)
    return h.hexdigest()


def canonical(box):
    b = np.asarray(box, dtype=float).copy()
    if b.shape != (5,) or not np.isfinite(b).all() or min(b[2:4]) <= 0:
        raise ValueError('Expected finite positive [cx,cy,w,h,theta]')
    if b[2] < b[3]:
        b[2], b[3] = b[3], b[2]; b[4] += math.pi / 2
    b[4] = (b[4] + math.pi / 2) % math.pi - math.pi / 2
    return b


def angle_error(a, b):
    return abs((a - b + math.pi / 2) % math.pi - math.pi / 2) * 180 / math.pi


def box_axis(box):
    b = canonical(box)
    v = .5 * b[2] * np.array([math.cos(b[4]), math.sin(b[4])])
    return np.stack((b[:2] - v, b[:2] + v))


def box_polygon(box):
    b = canonical(box)
    local = np.array([[-b[2]/2,-b[3]/2], [b[2]/2,-b[3]/2],
                      [b[2]/2,b[3]/2], [-b[2]/2,b[3]/2]])
    c, s = math.cos(b[4]), math.sin(b[4])
    return local @ np.array([[c,s],[-s,c]]) + b[:2]


def corner_difference(first, second):
    return min(float(np.linalg.norm(first-np.roll(order, shift, axis=0), axis=1).max())
               for order in (second, second[::-1]) for shift in range(4))


def polygon_box(path):
    lines = [s.split() for s in Path(path).read_text().splitlines() if s.strip()]
    if len(lines) != 1 or len(lines[0]) != 10 or lines[0][8] != 'grab' or lines[0][9] not in ('0','1'):
        raise ValueError('Expected one grab polygon, retaining difficulty 0/1: ' + str(path))
    p = np.asarray([float(x) for x in lines[0][:8]], dtype=np.float32).reshape(4, 2)
    if not np.isfinite(p).all():
        raise ValueError('Nonfinite polygon: ' + str(path))
    edges = np.roll(p, -1, axis=0) - p
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    if not (np.all(cross > 0) or np.all(cross < 0)):
        raise ValueError('Degenerate or nonconvex polygon: ' + str(path))
    (x, y), (w, h), a = cv2.minAreaRect(p)
    if min(w, h) < 2:
        raise ValueError('Polygon outside existing le90 >=2px contract: ' + str(path))
    # Match dataset poly2obb_np_le90 and its float32 storage, without MMRotate.
    b = canonical(np.asarray([x, y, w, h, math.radians(a)], dtype=np.float32))
    return p.astype(float), b


def geometry_errors(gt, pred):
    g, p = canonical(gt), canonical(pred)
    return dict(center_px=float(np.linalg.norm(g[:2] - p[:2])),
                long_relative=float(abs(p[2] / g[2] - 1)),
                short_relative=float(abs(p[3] / g[3] - 1)),
                size_max_relative=float(max(abs(p[2] / g[2] - 1), abs(p[3] / g[3] - 1))),
                long_signed_log=float(math.log(p[2] / g[2])),
                short_signed_log=float(math.log(p[3] / g[3])),
                angle_deg=float(angle_error(g[4], p[4])))


def describe(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return dict(n=0, mean=None, median=None, p90=None, p95=None, rmse=None, max=None)
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p90=float(np.percentile(a, 90)), p95=float(np.percentile(a, 95)),
                rmse=float(np.sqrt(np.mean(a*a))), max=float(a.max()))


def risk_stats(accepted, all_rows):
    """Conditional errors plus full-frame correct/error acceptance and rejection."""
    n, k = len(all_rows), len(accepted)
    out = [r for r in all_rows if r['pred'] is not None]
    result = dict(frames=n, output_frames=len(out), accepted_frames=k,
                  output_coverage=len(out)/n if n else None,
                  accepted_coverage=k/n if n else None,
                  output_center_hit_rate=sum(r['errors']['center_px'] < 15 for r in out)/len(out) if out else None,
                  all_frame_center_correct_coverage=sum(r['errors']['center_px'] < 15 for r in out)/n if n else None,
                  accepted_center_hit_rate=sum(r['errors']['center_px'] < 15 for r in accepted)/k if k else None)
    accepted_names = {r['image'] for r in accepted}
    longest = run = 0; previous = None
    for r in sorted(all_rows, key=lambda r: r['image']):
        sequence, frame = r['image'].rsplit('_',1); key = (sequence,int(frame))
        if previous is None or key[0] != previous[0] or key[1] != previous[1]+1:
            run = 0
        run = run+1 if r['image'] not in accepted_names else 0
        longest = max(longest,run); previous = key
    result['longest_unaccepted_run'] = longest
    result['continuous_edge_errors'] = {key: describe([r['errors'][key] for r in accepted])
                                        for key in ('long_relative','short_relative')}
    for component, limit in ERROR_LIMITS.items():
        # Ambiguous GT axis is unassessed, not an angle failure/success.
        eligible = lambda r: component != 'angle_deg' or r['angle_axis_well_defined']
        a = [r['errors'][component] for r in accepted if eligible(r)]
        correct = lambda v: v < limit if component == 'center_px' else v <= limit
        good = sum(correct(v) for v in a)
        raw_good = sum(correct(r['errors'][component]) for r in out if eligible(r))
        result[component] = dict(error=describe(a), assessable_accepted=len(a),
            unassessed_accepted=k-len(a), incorrect_accepted=len(a)-good,
            conditional_error_rate=(len(a)-good)/len(a) if a else None,
            all_frame_correct_acceptance=good/n if n else None,
            all_frame_incorrect_acceptance=(len(a)-good)/n if n else None,
            correct_outputs_rejected=raw_good-good,
            incorrect_outputs_rejected=sum(not correct(r['errors'][component]) for r in out if eligible(r))-(len(a)-good))
    return result


def score_curve(rows, coverages=COVERAGES):
    """Only whole equal-score groups; no GT-dependent or filename tie cutting."""
    if not rows:
        return dict(points=[], fixed_coverage_grid=[dict(requested_coverage=t, status='NO_FRAMES', point=None)
                                                    for t in coverages])
    out = sorted((r for r in rows if r['pred'] is not None), key=lambda r: -r['pred'][5])
    points, accepted = [], []
    i = 0
    while i < len(out):
        score = out[i]['pred'][5]
        j = i + 1
        while j < len(out) and out[j]['pred'][5] == score:
            j += 1
        accepted.extend(out[i:j])
        points.append(dict(score_ge=score, stats=risk_stats(accepted, rows), tie_group_size=j-i))
        i = j
    grid = []
    for target in coverages:
        if target > len(out)/len(rows) + 1e-12:
            grid.append(dict(requested_coverage=target, status='UNREACHABLE_MISSING_OUTPUT', point=None))
            continue
        below = [p for p in points if p['stats']['accepted_coverage'] <= target + 1e-12]
        above = [p for p in points if p['stats']['accepted_coverage'] > target + 1e-12]
        chosen = below[-1] if below else dict(score_ge=None, stats=risk_stats([], rows))
        grid.append(dict(requested_coverage=target, status='WHOLE_SCORE_TIES_ONLY', point=chosen,
                         next_attainable_coverage=above[0]['stats']['accepted_coverage'] if above else None))
    return dict(points=points, fixed_coverage_grid=grid,
                note='Thresholds enumerate cached scores only. No optimal threshold or fitted policy is selected.')


def rank_association(x, y):
    """Spearman descriptive association, average ranks for ties; no iid p-value."""
    def ranks(a):
        a = np.asarray(a)
        order = np.argsort(a, kind='mergesort'); r = np.empty(len(a), dtype=float)
        i = 0
        while i < len(a):
            j = i + 1
            while j < len(a) and a[order[j]] == a[order[i]]:
                j += 1
            r[order[i:j]] = (i+j-1)/2.; i = j
        return r
    if len(x) < 3 or len(set(x)) == 1 or len(set(y)) == 1:
        return None
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def validate_cached_rows(rows, ann_paths, image_dir, expected_counts=VAL_COUNTS):
    expected = {p.stem: p for p in ann_paths}
    if Counter(r['sequence'] for r in rows) != Counter(expected_counts):
        raise ValueError('Unexpected VAL sequence counts')
    if len({r['image'] for r in rows}) != len(rows) or set(expected) != {r['image'] for r in rows}:
        raise ValueError('Duplicate/missing cache frames or current VAL annotation mismatch')
    clean = []
    for r in rows:
        name = r['image']; seq, number = name.rsplit('_', 1)
        domain = seq.split('_')[0]
        if r['split'] != 'val' or r['sequence'] != seq or r['domain'] != domain or r['frame_id'] != int(number):
            raise ValueError('Cache role/frame identity mismatch: ' + name)
        _, actual_gt = polygon_box(expected[name]); gt = canonical(r['gt'])
        # minAreaRect can pick another almost-equal enclosing edge across
        # OpenCV versions. Compare corners at the 0.1px label precision,
        # after exact annotation SHA verification, not exact angle equality.
        corner_delta = corner_difference(box_polygon(gt), box_polygon(actual_gt))
        if corner_delta > .20:
            raise ValueError('Cached GT differs from current le90 annotation: ' + name)
        image = image_dir / (name + '.jpg')
        if sha(image) != r['image_sha256']:
            raise ValueError('Cached image identity mismatch: ' + name)
        with Image.open(image) as im:
            width, height = im.size
        pred = r['pred']; m = r['metrics']
        if type(m['output']) is not bool or m['output'] != (pred is not None):
            raise ValueError('Output presence mismatch: ' + name)
        item = dict(image=name, sequence=seq, domain=domain, gt=gt.tolist(), pred=pred,
                    angle_axis_well_defined=bool(gt[2]/gt[3] >= 1.2), gt_conversion_corner_delta_px=corner_delta)
        if pred is None:
            if m['center_hit'] is not False:
                raise ValueError('Missing output cannot hit center: ' + name)
        else:
            p = np.asarray(pred, dtype=float)
            if p.shape != (6,) or not np.isfinite(p).all() or not .05 < p[5] <= 1:
                raise ValueError('Invalid frozen B output: ' + name)
            e = geometry_errors(gt, p[:5]); item['errors'] = e
            fields = dict(center_error_px='center_px', long_edge_relative_error='long_relative',
                          short_edge_relative_error='short_relative', long_edge_signed_log_ratio='long_signed_log',
                          short_edge_signed_log_ratio='short_signed_log', angle_error_deg='angle_deg')
            if any(not math.isclose(m[a], e[b], rel_tol=1e-6, abs_tol=1e-6) for a, b in fields.items()):
                raise ValueError('Cached component metric mismatch: ' + name)
            if m['center_hit'] != (e['center_px'] < 15) or m['angle_axis_well_defined'] != item['angle_axis_well_defined']:
                raise ValueError('Cached component convention mismatch: ' + name)
            pc = canonical(p[:5])
            item['features'] = dict(score=float(p[5]), pred_log_long=math.log(pc[2]), pred_log_short=math.log(pc[3]),
                                    pred_aspect=pc[2]/pc[3], pred_abs_angle_deg=abs(math.degrees(pc[4])),
                                    pred_cx_over_width=pc[0]/width, pred_cy_over_height=pc[1]/height)
        clean.append(item)
    return sorted(clean, key=lambda r: r['image'])


def native_cache_checks(identity, rows, require_native=False):
    """Verify original hashes if available; PKL correspondence needs no model."""
    paths = dict(checkpoint=Path(identity['checkpoint']), pkl=Path(identity['pkl']),
                 selection=Path(identity['pkl']).parents[2] / 'sweep_results.json')
    availability = {key: p.is_file() for key, p in paths.items()}
    if require_native and not all(availability.values()):
        raise FileNotFoundError('Native B checkpoint/PKL/selection unavailable; do not fall back in strict mode')
    result = {}
    for key, p in paths.items():
        if p.is_file():
            if sha(p) != FROZEN_B[key + '_sha256']:
                raise ValueError('Native B ' + key + ' hash mismatch')
            result[key] = dict(path=str(p), status='HASH_VERIFIED')
        else:
            result[key] = dict(path=str(p), status='UNAVAILABLE_REPORT_DERIVED')
    if availability['pkl']:
        # Only deserialize the allowlisted SHA-verified project prediction file.
        with paths['pkl'].open('rb') as f:
            predictions = pickle.load(f)
        ordered = sorted(rows, key=lambda r: r['image'])
        if len(predictions) != len(ordered):
            raise ValueError('Native PKL frame count mismatch')
        for prediction, r in zip(predictions, ordered):
            if not isinstance(prediction, (list, tuple)) or len(prediction) != 1:
                raise ValueError('Native PKL must be single class')
            a = np.asarray(prediction[0])
            if a.ndim != 2 or a.shape[1] != 6 or len(a) > 1:
                raise ValueError('Native PKL must obey B max_per_img=1')
            pred = a[0].tolist() if len(a) else None
            if pred != r['pred']:
                raise ValueError('Native PKL/report correspondence mismatch: ' + r['image'])
        result['pkl']['row_correspondence'] = 'ALL_ROWS_VERIFIED'
        result['pkl']['verified_frames'] = len(ordered)
    return result


def check_cache(report_path, data_root, require_native=False):
    d = json.loads(report_path.read_text())
    if (d.get('protocol') != 'port_shape_e_h_v1_val_compare'
            or d.get('evidence_role') != 'source_val_only_fixed_experiment_comparison'
            or d.get('metric_consistency_review_required') is not False):
        raise ValueError('Requires the reviewed source-VAL E-H comparison with embedded frozen B rows')
    identity = d['identities']['b']
    if any(identity.get(k) != v for k, v in FROZEN_B.items()) or sha(B_CONFIG) != FROZEN_B['config_sha256']:
        raise ValueError('Frozen B identity/config mismatch')
    ann_paths = files(data_root/'val/annfiles', '.txt')
    if set_sha(ann_paths) != FROZEN_B['annotation_sha256']:
        raise ValueError('Current VAL annotation SHA differs from retained B')
    rows = validate_cached_rows(d['geometry']['rows']['b'], ann_paths, data_root/'val/images')
    native = native_cache_checks(identity, rows, require_native)
    groups = {'all': rows}
    groups.update({'domain:'+domain: [r for r in rows if r['domain'] == domain] for domain in ('real', 'sim')})
    groups.update({'sequence:'+s: [r for r in rows if r['sequence'] == s] for s in VAL_COUNTS})
    result = dict(protocol='port_b_reliability_cache_check_v1', status='CACHE_CHECK_COMPLETE_REVIEW_REQUIRED',
                  evidence_role='source_val_descriptive_no_fit', source_report=str(report_path.resolve()),
                  source_report_sha256=sha(report_path), b_identity=identity, native_cache_verification=native,
                  limits=ERROR_LIMITS, limits_role='center existing protocol; size/angle descriptive references only',
                  coordinates='original pixels; canonical long/short; pi-periodic angle; GT aspect>=1.2 angle assessability',
                  current_gt_conversion_check=dict(corner_tolerance_px=.20,
                      differences=describe([r['gt_conversion_corner_delta_px'] for r in rows]),
                      note='Exact annotation SHA + corner equivalence; OpenCV minAreaRect edge ties can differ across versions.'),
                  groups={})
    for name, group in groups.items():
        out = [r for r in group if r['pred'] is not None]
        curve = score_curve(group)
        associations = {}
        for feature in out[0]['features'] if out else []:
            associations[feature] = {}
            for component in ERROR_LIMITS:
                eligible = [r for r in out if component != 'angle_deg' or r['angle_axis_well_defined']]
                associations[feature][component] = dict(n=len(eligible), spearman=rank_association(
                    [r['features'][feature] for r in eligible], [r['errors'][component] for r in eligible]))
        cut = float(np.percentile([r['pred'][5] for r in out], 75)) if out else None
        high = [r for r in out if r['pred'][5] >= cut] if out else []
        result['groups'][name] = dict(raw=risk_stats(out, group), score_only=curve,
            score_on_correct_and_incorrect={component: {
                label: describe([r['pred'][5] for r in out if (component != 'angle_deg' or r['angle_axis_well_defined'])
                    and ((r['errors'][component] < limit if component == 'center_px' else r['errors'][component] <= limit) == good)])
                for label, good in [('correct', True), ('incorrect', False)]} for component, limit in ERROR_LIMITS.items()},
            relative_high_score=dict(definition='score>=within-group score p75, includes whole ties; descriptive, not runtime gate',
                                     cut=cut, stats=risk_stats(high, group), images=[r['image'] for r in high]),
            feature_error_association=associations,
            signed_log_sizes={k: describe([r['errors'][k] for r in out]) for k in ('long_signed_log','short_signed_log')})
    result['limitations'] = [
        'TEST repeatedly exposed; no TEST input, tuning, checkpoint reselection, or inference here.',
        'VAL already selected the detector and participated in development; exploratory source evidence only.',
        'Video frames are correlated. Associations have no iid significance claim and do not choose features.',
        'Report-derived identity is distinguished from native PKL verification; local weights are unavailable.',
        'Missing detections remain in all-frame denominators; ambiguous angles are explicitly unassessed.',
        'No fitted reliability head, calibrated probabilities, modified boxes, depth accuracy, or deployment threshold.']
    return result


def read_axis(path, expected_stem, image_size, legacy=False):
    d = json.loads(path.read_text())
    shapes = d.get('shapes', [])
    if len(shapes) != 1 or shapes[0].get('label') != 'axis' or shapes[0].get('shape_type') != 'line':
        raise ValueError('Expected exactly one native axis line: ' + str(path))
    a = np.asarray(shapes[0].get('points'), dtype=float)
    if a.shape != (2, 2) or not np.isfinite(a).all() or np.linalg.norm(a[1]-a[0]) <= 0:
        raise ValueError('Invalid native axis points: ' + str(path))
    allowed_stems = {expected_stem}
    if legacy and expected_stem.rsplit('_', 1)[0] in LEGACY_TRAIN:
        allowed_stems.add(expected_stem[5:])
    if (d.get('imageWidth'), d.get('imageHeight')) != image_size or Path(d.get('imagePath', '')).stem not in allowed_stems:
        raise ValueError('Native axis image identity/resolution mismatch: ' + str(path))
    width, height = image_size
    flags = dict(axis_outside_image=bool(np.any(a < 0) or np.any(a[:,0] >= width) or np.any(a[:,1] >= height)),
                 annotation_difficult=bool(shapes[0].get('difficult', False)),
                 visibility_supervision_present=any(k in shapes[0] for k in ('visibility', 'occluded', 'keypoint_visibility')))
    return a, flags


def axis_consistency(axis, box, conversion_k=2.1):
    b = canonical(box); a = np.asarray(axis, dtype=float)
    if (a.shape != (2, 2) or not np.isfinite(a).all() or np.linalg.norm(a[1]-a[0]) <= 0
            or not math.isfinite(conversion_k) or conversion_k <= 1):
        raise ValueError('Invalid axis')
    length = float(np.linalg.norm(a[1]-a[0])); target = box_axis(b)
    endpoint = min(float(np.linalg.norm(a-order, axis=1).max()) for order in (target, target[::-1]))
    errors = dict(center_px=float(np.linalg.norm(a.mean(axis=0)-b[:2])), endpoint_px=endpoint,
                  length_px=abs(length-b[2]), short_px=abs(length/conversion_k-b[3]),
                  angle_deg=angle_error(math.atan2(*(a[1]-a[0])[::-1]), b[4]))
    return dict(errors=errors, conversion_consistent=all(errors[k] <= AXIS_TOL[k] for k in errors),
                native_length_px=length, native_axis=a.tolist(), obb_derived_axis=target.tolist())


def check_structure(data_root):
    records, samples, identities = [], [], {}
    legacy_root = data_root/'provenance/axis_legacy_train_v1'
    legacy_manifest = json.loads((legacy_root/'manifest.json').read_text()) if (legacy_root/'manifest.json').is_file() else None
    if legacy_manifest is not None:
        if legacy_manifest.get('protocol') != 'port_legacy_train_axis_snapshot_v1' or legacy_manifest.get('sequences') != list(LEGACY_TRAIN):
            raise ValueError('Unexpected legacy TRAIN axis snapshot protocol/roles')
        identities['legacy_snapshot_manifest_sha256'] = sha(legacy_root/'manifest.json')
        expected_names = {p.stem for p in files(data_root/'train/annfiles','.txt')
                          if p.stem.rsplit('_',1)[0] in LEGACY_TRAIN}
        if set(legacy_manifest.get('rows',{})) != expected_names or any(
                r.get('split') != 'train' or r.get('sequence') != name.rsplit('_',1)[0]
                for name,r in legacy_manifest['rows'].items()):
            raise ValueError('Legacy snapshot must bind exactly current allowlisted TRAIN frames')
    for split in ('train', 'train_sim'):
        anns = files(data_root/split/'annfiles', '.txt')
        counts = Counter(p.stem.rsplit('_',1)[0] for p in anns)
        expected = {s: n for s, n in TRAIN_COUNTS.items() if TRAIN_SPLITS[s] == split}
        if counts != Counter(expected):
            raise ValueError('Unexpected TRAIN role/sequence counts: ' + split)
        identities[split+'_annotation_sha256'] = set_sha(anns)
        for sequence in sorted(expected):
            seq_anns = [p for p in anns if p.stem.rsplit('_', 1)[0] == sequence]
            raw = {}
            if sequence in NATIVE_K and (sequence not in LEGACY_TRAIN or legacy_manifest is not None):
                axis_root = legacy_root if sequence in LEGACY_TRAIN else data_root/'provenance/axis_k2p1'
                axis_files = files(axis_root/sequence/'axis_json', '.json')
                raw = {p.stem: p for p in axis_files}
                if set(raw) != {p.stem for p in seq_anns}:
                    raise ValueError('Native TRAIN axis missing/extra frame: ' + sequence)
                identities[sequence+'_axis_json_sha256'] = set_sha(axis_files)
            chosen = {seq_anns[0].stem, seq_anns[len(seq_anns)//2].stem}
            for path in seq_anns:
                image = data_root/split/'images'/(path.stem+'.jpg')
                with Image.open(image) as im:
                    size = im.size
                poly, box = polygon_box(path)
                row = dict(image=path.stem, sequence=sequence, split=split,
                    image_size=list(size), image_path=str(image.resolve()), annotation_sha256=sha(path),
                    source='native_axis' if raw else 'obb_derived_weak_axis', gt=box.tolist(), polygon=poly.tolist(),
                    obb_derived_axis=box_axis(box).tolist(),
                    polygon_outside_image=bool(np.any(poly < 0) or np.any(poly[:,0] >= size[0]) or np.any(poly[:,1] >= size[1])),
                    gt_short_px=float(box[3]), gt_aspect=float(box[2]/box[3]),
                    dota_difficulty=int(path.read_text().split()[9]))
                if raw:
                    axis, flags = read_axis(raw[path.stem], path.stem, size, legacy=sequence in LEGACY_TRAIN)
                    row.update(flags); row.update(axis_consistency(axis, box, NATIVE_K[sequence]))
                    row['axis_json_sha256'] = sha(raw[path.stem])
                    row['axis_source_group'] = 'legacy_snapshot' if sequence in LEGACY_TRAIN else 'axis_k2p1'
                    row['conversion_k'] = NATIVE_K[sequence]
                    if sequence in LEGACY_TRAIN:
                        binding = legacy_manifest['rows'][path.stem]
                        if (binding['axis_json_sha256'] != row['axis_json_sha256']
                                or binding['annotation_sha256'] != row['annotation_sha256']
                                or binding['image_sha256'] != sha(image)):
                            raise ValueError('Legacy snapshot/current TRAIN source identity mismatch: ' + path.stem)
                records.append(row)
                if path.stem in chosen:
                    samples.append(dict(row, image_sha256=sha(image), sample_policy='first and sorted middle in each TRAIN sequence; fixed before image review',
                                        semantic_review='PENDING_HUMAN_REVIEW'))
    native = [r for r in records if r['source'] == 'native_axis']
    summaries = {}
    for sequence in TRAIN_COUNTS:
        group = [r for r in records if r['sequence'] == sequence]
        summaries[sequence] = dict(frames=len(group), source=group[0]['source'],
            gt_short_px=describe([r['gt_short_px'] for r in group]),
            gt_aspect=describe([r['gt_aspect'] for r in group]),
            dota_difficult_frames=sum(r['dota_difficulty'] for r in group),
            polygon_outside_image=sum(r['polygon_outside_image'] for r in group))
        if group[0]['source'] == 'native_axis':
            summaries[sequence].update(conversion_consistent=sum(r['conversion_consistent'] for r in group),
                axis_outside_image=sum(r['axis_outside_image'] for r in group),
                annotation_difficult=sum(r['annotation_difficult'] for r in group),
                conversion_errors={k: describe([r['errors'][k] for r in group]) for k in AXIS_TOL})
    return dict(protocol='port_train_structure_supervision_check_v1', status='NUMERIC_CHECK_COMPLETE_SEMANTIC_REVIEW_REQUIRED',
        evidence_role='train_only_annotation_readiness_not_branch_training', data_root=str(data_root.resolve()), source_identities=identities,
        train_frames=len(records), native_axis_frames=len(native), obb_derived_weak_axis_frames=len(records)-len(native),
        native_conversion_consistent=sum(r['conversion_consistent'] for r in native), conversion_tolerances=AXIS_TOL,
        legacy_snapshot_available=legacy_manifest is not None,
        visibility_supervision_present=sum(r['visibility_supervision_present'] for r in native),
        excluded_native_axis_sequence='real_seq14 belongs to VAL; not opened for TRAIN supervision',
        sequences=summaries, review_samples=samples, rows=records,
        suitability=dict(center_and_pi_periodic_axis='conditional candidate after semantic review; native TRAIN labels available',
            endpoint_visibility='not supervised by the verified schema',
            short_edge_physical_boundary='not independently supervised: per-sequence conversion ratios, never an inference prior',
            other_sequences='Without a legacy snapshot, missing originals means unavailable locally, not never axis-annotated; sim axes remain OBB-derived'),
        limitations=['Numerical conversion agreement does not establish visible landmarks or independent label information.',
                     'Native supervision spans the available allowlisted real sequences; frames are not independent events.',
                     'No inferred visibility labels, no automatic removal of difficult/cropped examples.',
                     'No TEST, no global 2.1 aspect prior, no training or change to B/coordinate/depth contracts.'])


def make_gallery(samples, out_dir):
    """Isotropic thumbnails + original-coordinate overlays; no source editing."""
    pages = []
    for page_index, start in enumerate(range(0, len(samples), 6)):
        sheet = Image.new('RGB', (1440, 990), (24, 24, 24)); draw = ImageDraw.Draw(sheet)
        draw.text((12, 8), 'TRAIN only | blue OBB | yellow OBB-derived axis | red native axis | endpoints unordered', fill='white')
        for i, r in enumerate(samples[start:start+6]):
            with Image.open(r['image_path']) as src:
                im = src.convert('RGB')
            # Full image gives context. Separate isotropic ROI makes small axes visible.
            b = canonical(r['gt']); side = max(1., b[2]*2.5)
            left, top = math.floor(b[0]-side/2), math.floor(b[1]-side/2)
            right, bottom = math.ceil(b[0]+side/2), math.ceil(b[1]+side/2)
            views = [(im, 0, 0), (im.crop((left, top, right, bottom)), left, top)]
            x0, y0 = (i%2)*720, 32+(i//2)*316
            draw.text((x0+8, y0), r['image']+' | '+r['source'], fill='white')
            for j, (view, ox, oy) in enumerate(views):
                original_w = r['image_size'][0] if j == 0 else right-left
                original_h = r['image_size'][1] if j == 0 else bottom-top
                # Enlarge the small ROI for review only; it adds no pixel
                # information. One isotropic scale, with integer display rounding.
                scale = min(348/original_w, 278/original_h)
                view = view.resize((max(1,round(original_w*scale)), max(1,round(original_h*scale))), Image.LANCZOS)
                px, py = x0+6+j*354, y0+22
                sheet.paste(view, (px, py))
                xy = lambda p: (px+(p[0]-ox)*scale, py+(p[1]-oy)*scale)
                polygon = [xy(p) for p in r['polygon']]; draw.line(polygon+[polygon[0]], fill=(80,160,255), width=2)
                draw.line([xy(p) for p in r['obb_derived_axis']], fill='yellow', width=2)
                if r['source'] == 'native_axis':
                    points = [xy(p) for p in r['native_axis']]
                    draw.line(points, fill=(255,75,75), width=2)
                    for x,y in points:
                        draw.ellipse((x-3,y-3,x+3,y+3), outline=(255,75,75), width=2)
        path = out_dir/('train_structure_review_%02d.png' % (page_index+1))
        with path.open('xb') as stream:
            sheet.save(stream, format='PNG')
        pages.append(dict(path=str(path.resolve()), sha256=sha(path)))
    return pages


def write_new(path, data):
    payload = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    with path.open('x', encoding='utf-8') as f:
        f.write(payload+'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('all','cache','structure'), default='all')
    parser.add_argument('--b-report', type=Path, default=ROOT/'work_dirs/port_shape_e_h_v1_val_compare.json')
    parser.add_argument('--data-root', type=Path, default=DATA)
    parser.add_argument('--require-native-cache', action='store_true', help='Require original checkpoint/PKL/sweep JSON, and verify all PKL rows')
    parser.add_argument('--out-dir', type=Path, default=ROOT/'work_dirs/port_reliability_readiness_v1')
    args = parser.parse_args(argv)
    out = args.out_dir.resolve(); data = args.data_root.resolve()
    if out.exists():
        raise FileExistsError('Refuse to overwrite output directory: ' + str(out))
    if data == out or data in out.parents:
        raise ValueError('Output directory must be outside the source dataset')
    reports = {}
    if args.mode in ('all','cache'):
        reports['b_val_cache_check.json'] = check_cache(args.b_report, data, args.require_native_cache)
    if args.mode in ('all','structure'):
        reports['train_structure_check.json'] = check_structure(data)
    out.mkdir(parents=True, exist_ok=False)
    if 'train_structure_check.json' in reports:
        reports['train_structure_check.json']['review_pages'] = make_gallery(reports['train_structure_check.json']['review_samples'], out)
    for filename, report in reports.items():
        report['tool_sha256'] = sha(Path(__file__))
        report['runtime'] = dict(python=sys.version.split()[0], numpy=np.__version__, opencv=cv2.__version__)
        write_new(out/filename, report)
        print(filename, report['status'])
    if 'b_val_cache_check.json' in reports:
        for name in ('domain:real','domain:sim'):
            raw = reports['b_val_cache_check.json']['groups'][name]['raw']
            print(name, 'frames', raw['frames'], 'outputs', raw['output_frames'],
                  'center_bad', raw['center_px']['incorrect_accepted'], 'size_bad', raw['size_max_relative']['incorrect_accepted'],
                  'angle_bad', raw['angle_deg']['incorrect_accepted'])
    if 'train_structure_check.json' in reports:
        r = reports['train_structure_check.json']
        print('TRAIN structure', r['train_frames'], 'native', r['native_axis_frames'],
              'conversion_consistent', r['native_conversion_consistent'], 'weak', r['obb_derived_weak_axis_frames'])
    print('Saved', out, '| no fitting, training, GPU, or TEST')


if __name__ == '__main__':
    main()
