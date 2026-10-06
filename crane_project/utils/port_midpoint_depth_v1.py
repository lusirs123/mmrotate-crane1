"""Frozen Raw-opt diagnostics. Standard library only; never fits or filters OBBs."""
import hashlib
import json
import math
from pathlib import Path
import sys

VERSION = 'port_midpoint_depth_v1_train03_frozen_diagnostic'
SEQUENCE = 'calibration_train_03_tilt_obb'
COUNT = 981
CALIBRATION = 'crane_project/configs/depth_scale_calibration_obb_raw_opt_short_q2_train03_v1.json'
CAL_SHA = 'c9c14fc1b09ba8223996121425e302888ac7d44213192225acd5365fb2f1ffe2'
B_CONFIG = 'crane_project/configs/crane_symeood_k1_port_day2night_aug_b_v1.py'
B_SHA = '8f8008c4944807a65ed0f2ee0cc348ea78690d54a4176944b2c9b0ebc83cec23'
HEAD_SHA = '16c2fb448ac4e1c53530b8086d547f6f6ccb8d6b0763a42391c34f9b337982d7'
CACHE_SHA = '046c5998dee0ba3703f1ae4e08fc6a02e9804d216ea12241357f69f2a4afd1e3'
HEAD_DIGEST = dict(parameters='f84d99dac7cb0f2dcb86b5eaf33c21c1d360bdca5d0193b7b3da6c98f92bdc63',
    buffers='44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a',
    parameter_count=17696, buffer_count=0)
MANIFEST_SHA = '4dbc6c8dcb039c73cca0311da1a648fa08ad807d6b29a1e23124d6c659a4c0d6'
TRUTH_SHA = '3441b0e1dad135ee7d49b0ee2c243fc603c5193284aa69a8ab67d71b52db7aad'
IMAGE_SHA = '4eda8c70f634023fd607fec1db8ed1b5c5e959c84a742ed9a6ab305cd171f97b'
ORACLE = dict(count=981, mae_m=0.040899054277329, rmse_m=0.0534267060329009,
    abs_rel=0.002473595701081284, bias_m=-0.01135319901640459,
    p95_abs_error_m=0.11329915424724213, max_abs_error_m=0.15337788871866742)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, value):
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n'
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(rendered)


def protocol_document():
    return dict(protocol=VERSION, stages=['check', 'oracle', 'audit'],
        sequence=SEQUENCE, split='calibration_train', frame_count=COUNT,
        manifest_sha256=MANIFEST_SHA, truth_sha256=TRUTH_SHA, image_identity_sha256=IMAGE_SHA,
        calibration=dict(path=CALIBRATION, sha256=CAL_SHA, contract='raw_opt_v1', fit_offset=False),
        detector=dict(config=B_CONFIG, epoch=24, sha256=B_SHA),
        midpoint=dict(sigma_cells=1.5, epoch=3, updates=2706, sha256=HEAD_SHA, digest=HEAD_DIGEST),
        reference_oracle=ORACLE,
        inference='Unchanged frozen B native pipeline, one P3 extraction and three native head calls/frame; reuse original capture and sigma head. Actual sx/sy, raw w/h-angle and whole-box fallback preserved. No metric truth loaded until prediction JSONL is closed.',
        metrics='All 981 frames retained. All numerically computable outputs, including q outside GT-fit support, enter the main depth metrics. Numeric failures and missing outputs reported with full-frame denominators. q support is diagnostic, never a rejection gate or a depth_valid claim.',
        comparison='Same-run B24 vs B24+sigma1.5/epoch03, with GT-OBB Raw-opt reference; optional validated historical report is labelled unpaired, never substituted for fresh B.',
        scope=dict(detector_updates=0, midpoint_updates=0, parameter_refit=False,
            reliability_filter=False, fixed_dev_access=False, unknown_access=False,
            detection_test_access=False, deployment_validity_gate_frozen=False))


def canonical(box):
    if len(box) not in (5, 6) or not all(math.isfinite(float(v)) for v in box):
        raise ValueError('Expected finite raw OBB with optional score')
    cx, cy, w, h, angle = map(float, box[:5])
    if min(w, h) <= 0:
        raise ValueError('OBB edges must be positive')
    if w < h:
        w, h, angle = h, w, angle+math.pi/2
    return [cx, cy, w, h, (angle+math.pi/2) % math.pi-math.pi/2]


def depth(box, intrinsics, geometry, parameters):
    """Equivalent to the original formula; detect overflow without clamping q."""
    _, _, w, h, angle = canonical(box)
    fx, fy = float(intrinsics['fx']), float(intrinsics['fy'])
    long_m, short_m = float(geometry['long_edge_mean_m']), float(geometry['short_edge_mean_m'])
    if not all(math.isfinite(v) and v > 0 for v in (fx, fy, long_m, short_m)):
        raise ValueError('Invalid camera or physical geometry')
    sl = w*math.hypot(math.cos(angle)/fx, math.sin(angle)/fy)
    ss = h*math.hypot(math.sin(angle)/fx, math.cos(angle)/fy)
    if min(sl, ss) <= 0 or not all(math.isfinite(v) for v in (sl, ss)):
        return dict(z_m=None, q_signed=None, status='nonfinite_scale')
    log_long, log_short = math.log(long_m)-math.log(sl), math.log(short_m)-math.log(ss)
    q = log_long-log_short
    if not all(math.isfinite(v) for v in (q, log_short)):
        return dict(z_m=None, q_signed=None, status='nonfinite_scale')
    if parameters.get('b_m', 0.0) != 0.0:
        raise ValueError('Frozen zero-offset formula required')
    c, beta = float(parameters['c']), float(parameters['beta'])
    if not math.isfinite(c) or c <= 0 or not math.isfinite(beta):
        raise ValueError('Invalid frozen calibration parameters')
    log_z = math.log(c)+log_short+beta*q*q
    if not math.isfinite(log_z) or log_z > math.log(sys.float_info.max):
        return dict(z_m=None, q_signed=q, status='formula_overflow')
    z = math.exp(log_z)
    if not math.isfinite(z) or z <= 0:
        return dict(z_m=None, q_signed=q, status='formula_nonpositive')
    return dict(z_m=z, q_signed=q, status='finite_formula')


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    index = (len(values)-1)*fraction
    lo, hi = int(index), min(int(index)+1, len(values)-1)
    portion = index-lo
    return (1-portion)*values[lo]+portion*values[hi]


def describe(values):
    if not values:
        return dict(count=0, mean=None, p95=None, max=None)
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Nonfinite values cannot be hidden in a summary')
    return dict(count=len(values), mean=math.fsum(v/len(values) for v in values),
                p95=percentile(values, .95), max=max(values))


def error_metrics(predictions, truths):
    if len(predictions) != len(truths):
        raise ValueError('Metric lengths differ')
    if not predictions:
        return dict(count=0, mae_m=None, rmse_m=None, abs_rel=None,
                    bias_m=None, p95_abs_error_m=None, max_abs_error_m=None)
    if not all(math.isfinite(z) and z > 0 for z in predictions+truths):
        raise ValueError('Only explicitly computable positive depths allowed')
    errors = [p-t for p, t in zip(predictions, truths)]
    absolute = [abs(e) for e in errors]
    maximum = max(absolute)
    n = len(errors)
    rms = maximum*math.sqrt(math.fsum((v/maximum)**2/n for v in absolute)) if maximum else 0.0
    return dict(count=n, mae_m=math.fsum(v/n for v in absolute), rmse_m=rms,
        abs_rel=math.fsum(abs(e)/t/n for e, t in zip(errors, truths)),
        bias_m=math.fsum(e/n for e in errors), p95_abs_error_m=percentile(absolute, .95),
        max_abs_error_m=maximum)


def truth_box(row):
    obb = row['obb_geometry']
    return canonical([obb['cx_px'], obb['cy_px'], obb['w_px'], obb['h_px'],
                      math.radians(obb['gamma_deg'])])


def validate_truth(rows, sequence_manifest):
    if sequence_manifest.get('sequence_id') != SEQUENCE or sequence_manifest.get('split') != 'calibration_train':
        raise ValueError('Train-03 only; fixed-dev and unknown are not admitted')
    if len(rows) != COUNT:
        raise ValueError('Full 981-frame truth required')
    run_ids = set()
    for index, row in enumerate(rows):
        expected = 'images/frame_%05d.jpg' % index
        if (row.get('sequence_id') != SEQUENCE or row.get('image_file') != expected
                or not row.get('obb_valid') or not row.get('truth_valid')
                or not row.get('frame_truth_audit', {}).get('passed')):
            raise ValueError('Truth identity/validity/order differs at '+expected)
        run_ids.add(row.get('run_instance_id'))
        truth_box(row)
        z = row['camera_geometry']['z_cg_opt_m']
        if not math.isfinite(z) or z <= 0:
            raise ValueError('Invalid independent optical-axis truth')
    if run_ids != {sequence_manifest['run_instance_id']}:
        raise ValueError('Run identity differs')


def validate_head_header(payload, frozen_b):
    if (payload.get('protocol') != 'port_geometry_midpoint_sigma_v1'
            or payload.get('sigma_cells') != 1.5 or payload.get('epoch') != 3
            or payload.get('updates') != 2706 or payload.get('head_digest') != HEAD_DIGEST
            or payload.get('frozen_b') != frozen_b
            or payload.get('context', {}).get('manifest_sha256') != CACHE_SHA
            or not payload.get('context', {}).get('detector_state')
            or 'head_state' not in payload):
        raise ValueError('Requires the sealed sigma1.5/epoch03 head and B context')


def checked_weight(root, explicit, filename, digest):
    if explicit is not None:
        path = Path(explicit).resolve()
        if not path.is_file() or sha(path) != digest:
            raise ValueError('Explicit weight has wrong identity: '+str(path))
        return path
    paths = sorted(p.resolve() for p in (Path(root)/'work_dirs').rglob(filename))
    matches = [p for p in paths if p.is_file() and sha(p) == digest]
    if not matches:
        raise FileNotFoundError('No weight with the fixed SHA found; pass the explicit '+filename+' path')
    # All admissible copies are byte-identical, never selected by performance.
    return matches[0]


def image_sources(root, expected_identity=IMAGE_SHA):
    folder = Path(root)/SEQUENCE/'images'
    paths = sorted(folder.glob('*.jpg'))
    if [p.name for p in paths] != ['frame_%05d.jpg'%i for i in range(COUNT)]:
        raise ValueError('Full Train-03 image names/count required')
    h = hashlib.sha256()
    records = []
    for path in paths:
        digest = sha(path)
        h.update(path.name.encode()); h.update(b'\0'); h.update(bytes.fromhex(digest))
        records.append(dict(frame_id=path.stem, image_path=str(path.resolve()), sha256=digest))
    if h.hexdigest() != expected_identity:
        raise ValueError('Original Train-03 JPEG bytes differ')
    return records


def geometry_row(box, gt, predicted_components, gt_components):
    pred, target = canonical(box), canonical(gt)
    center = math.hypot(pred[0]-target[0], pred[1]-target[1])
    angle = abs((pred[4]-target[4]+math.pi/2) % math.pi-math.pi/2)*180/math.pi
    long_e, short_e = pred[2]/target[2]-1, pred[3]/target[3]-1
    q_delta = (None if predicted_components['q_signed'] is None else
               predicted_components['q_signed']-gt_components['q_signed'])
    return dict(center_error_px=center, angle_error_deg=angle,
        long_relative_error=long_e, short_relative_error=short_e,
        log_aspect_residual=math.log(pred[2])-math.log(pred[3])-math.log(target[2])+math.log(target[3]),
        q_prediction_minus_gt=q_delta,
        center_correct=center < 15., both_edges_correct=abs(long_e) <= .10 and abs(short_e) <= .10)


def summarize(rows, method):
    outputs = [r for r in rows if r[method]['box'] is not None]
    finite = [r for r in rows if r[method]['depth']['status'] == 'finite_formula']
    supported = [r for r in outputs if r[method]['q_in_fit_support']]
    n = len(rows)
    if not n:
        raise ValueError('Empty depth sequence')
    center = sum(r[method]['geometry']['center_correct'] for r in outputs)
    edges = sum(r[method]['geometry']['both_edges_correct'] for r in outputs)
    geoms = [r[method]['geometry'] for r in outputs]
    metrics = error_metrics([r[method]['depth']['z_m'] for r in finite], [r['truth_z_m'] for r in finite])
    steps, step_errors = [], []
    nonfinite_steps = 0
    for previous, current in zip(rows, rows[1:]):
        a, b = previous[method]['depth']['z_m'], current[method]['depth']['z_m']
        if a is None or b is None or current['frame_index'] != previous['frame_index']+1:
            continue
        relative = abs(b-a)/a
        residual = abs((b-a)-(current['truth_z_m']-previous['truth_z_m']))
        if not math.isfinite(relative) or not math.isfinite(residual):
            nonfinite_steps += 1
        else:
            steps.append(relative); step_errors.append(residual)
    return dict(frame_count=n, output_frame_count=len(outputs), output_coverage=len(outputs)/n,
        numeric_depth_count=len(finite), numeric_depth_coverage=len(finite)/n,
        numeric_failure_count=len(outputs)-len(finite),
        depth_metric_denominator='all numerically computable outputs; q is never filtered',
        depth_metrics=metrics, q_in_fit_support_count=len(supported),
        q_out_of_fit_support_count=sum(r[method]['depth']['q_signed'] is not None and not r[method]['q_in_fit_support'] for r in outputs),
        q_uncomputable_count=sum(r[method]['depth']['q_signed'] is None for r in outputs),
        q_fit_support_coverage_all_frames=len(supported)/n,
        q_fit_support_rate_output_frames=len(supported)/len(outputs) if outputs else None,
        deployment_validity_gate_frozen=False,
        center_correct_output_frames=center, center_correct_rate_output_frames=center/len(outputs) if outputs else None,
        center_correct_coverage_all_frames=center/n, both_edges_correct_coverage_all_frames=edges/n,
        geometry={k:dict(signed=describe([g[k] for g in geoms if g[k] is not None]),
                          absolute=describe([abs(g[k]) for g in geoms if g[k] is not None]))
                  for k in ('center_error_px', 'angle_error_deg', 'long_relative_error',
                            'short_relative_error', 'log_aspect_residual', 'q_prediction_minus_gt')},
        depth_relative_step=describe(steps), depth_step_error_against_truth_m=describe(step_errors),
        numeric_step_failure_count=nonfinite_steps)


def evaluate(predictions, truths, manifest, calibration):
    validate_truth(truths, manifest)
    if len(predictions) != len(truths):
        raise ValueError('Full paired B/M prediction stream required')
    intr, geom = manifest['camera']['intrinsics'], manifest['obb_reference_geometry']
    params = calibration['parameters']
    rows = []
    for index, (prediction, truth) in enumerate(zip(predictions, truths)):
        frame = 'frame_%05d'%index
        if prediction.get('frame_id') != frame:
            raise ValueError('Prediction/truth frame identity differs')
        b, midpoint = prediction['b'], prediction['midpoint']
        if (b is None) != (midpoint is None):
            raise ValueError('Midpoint changed output presence')
        if b is not None and (len(b) != 6 or len(midpoint) != 6 or b[5] != midpoint[5]):
            raise ValueError('Raw output/score identity differs')
        gt = truth_box(truth); oracle = depth(gt, intr, geom, params)
        if oracle['status'] != 'finite_formula':
            raise ValueError('Frozen GT oracle cannot be computed')
        row = dict(frame_id=frame, frame_index=index, truth_z_m=truth['camera_geometry']['z_cg_opt_m'],
            theta_total_deg=truth['pivot_relative']['theta_total_deg'], gt_box=gt,
            midpoint_accepted=prediction.get('accepted'), failed_checks=prediction.get('failed_checks', []))
        for method, box in (('b', b), ('midpoint', midpoint), ('gt_obb', gt)):
            comp = depth(box, intr, geom, params) if box is not None else dict(z_m=None, q_signed=None, status='missing_output')
            q = comp['q_signed']
            row[method] = dict(box=box, depth=comp, q_in_fit_support=q is not None and params['q_signed_min'] <= q <= params['q_signed_max'],
                geometry=geometry_row(box, gt, comp, oracle) if box is not None else None)
        rows.append(row)
    groups = {method:summarize(rows, method) for method in ('b', 'midpoint', 'gt_obb')}
    for key, expected in ORACLE.items():
        actual = groups['gt_obb']['depth_metrics'][key]
        if actual is None or abs(actual-expected) > 1e-10:
            raise ValueError('Existing frozen Train-03 GT oracle differs: '+key)
    paired = [r for r in rows if r['b']['depth']['z_m'] is not None and r['midpoint']['depth']['z_m'] is not None]
    changes = []
    for r in paired:
        before = abs(r['b']['depth']['z_m']-r['truth_z_m'])
        after = abs(r['midpoint']['depth']['z_m']-r['truth_z_m'])
        changes.append(after-before)
    summary = dict(protocol=VERSION, sequence_id=SEQUENCE, coordinate_contract='raw_opt_v1',
        groups=groups, paired_numeric_depth_count=len(paired),
        paired_absolute_error_delta_m=describe(changes),
        paired_improved=sum(v < -1e-12 for v in changes), paired_worsened=sum(v > 1e-12 for v in changes),
        paired_tied=sum(abs(v) <= 1e-12 for v in changes),
        midpoint_accepted_count=sum(r['midpoint_accepted'] is True for r in rows),
        midpoint_fallback_count=sum(r['midpoint_accepted'] is False for r in rows),
        parameter_refit_performed=False, reliability_used=False, unknown_sequence_read=False,
        fixed_dev_read=False, deployment_validated=False)
    summary['by_theta'] = {label:{method:summarize(selected, method) if selected else None for method in groups}
        for label, low, high in [('theta_0_3', 0., 3.), ('theta_3_5', 3., 5.), ('theta_5_8', 5., 8.), ('theta_8_inf', 8., math.inf)]
        for selected in [[r for r in rows if low <= r['theta_total_deg'] < high]]}
    return rows, summary


def historical_reference(path, calibration_sha=CAL_SHA):
    """Reported old-model observation only; never a fresh paired baseline."""
    if path is None:
        return dict(status='not_provided', same_run=False, note='Old detector per-frame results were not recovered locally')
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    current = data.get('current', {})
    receipt = current.get('prediction_manifest', {})
    if (data.get('coordinate_contract') != 'raw_opt_v1' or data.get('sequence_id') != SEQUENCE
            or data.get('stage') != 'calibration_train_audit' or data.get('calibration_sha256') != calibration_sha
            or data.get('parameter_refit_performed') is not False or data.get('unknown_sequence_read') is not False
            or current.get('frame_count') != COUNT or receipt.get('processed_frame_count') != COUNT
            or receipt.get('limit') is not None or receipt.get('metric_truth_read') is not False
            or receipt.get('threshold_tuning_performed') is not False
            or not receipt.get('checkpoint_sha256') or not current.get('prediction_sha256')):
        raise ValueError('Historical report is not the matched frozen Raw-opt Train-03 protocol')
    oracle = current.get('truth_obb_oracle_depth_metrics', {})
    if any(oracle.get(k) is None or abs(oracle[k]-v) > 1e-10 for k, v in ORACLE.items()):
        raise ValueError('Historical GT oracle differs from the original Train-03 reference')
    return dict(status='unpaired_historical_report_only', path=str(Path(path).resolve()), sha256=sha(path),
        same_run=False, truth_bytes_verified=False, calibration_sha256=calibration_sha,
        checkpoint_sha256=receipt['checkpoint_sha256'], prediction_sha256=current['prediction_sha256'],
        model_type=receipt.get('model_type'), detector_obb_depth_metrics=current.get('detector_obb_depth_metrics'),
        geometry=current.get('geometry'), detection_rate=current.get('detection_rate'),
        note='Named sequence, frozen calibration and oracle match; old report lacks truth-byte receipt. No causal paired deltas are claimed.')
