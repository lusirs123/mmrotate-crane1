#!/usr/bin/env python3
"""Read-only selected formal midpoint VAL size/motion residual diagnosis.

Standard library only, Python 3.8+. No weights, images, inference, fitting,
TEST, new checkpoint selection, smoothing or output-box modification.
Original-coordinate canonical edge lengths are an OFFLINE metric view only.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VERSION = 'port_geometry_midpoint_size_temporal_v1'
TRAIN_VERSION = 'port_geometry_midpoint_formal_v1'
VAL_COUNTS = {'real_seq07': 226, 'real_seq14': 149, 'sim_seq10': 512}
FIELDS = ('long', 'short', 'diagonal')
METHODS = ('b', 'midpoint')
FORMAL = ROOT / 'work_dirs/crane_symeood_k1_port_day2night_midpoint_formal_v1'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_read(data):
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(data, parse_constant=invalid)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close(a, b, tolerance=1e-9):
    return finite(a) and finite(b) and abs(a-b) <= tolerance


def quantile(values, q):
    ordered = sorted(values)
    position = (len(ordered)-1)*q
    lo = int(position)
    hi = min(lo+1, len(ordered)-1)
    return ordered[lo] + (ordered[hi]-ordered[lo])*(position-lo)


def stats(values):
    values = list(values)
    if not values:
        return dict(n=0, mean=None, median=None, mae=None, rmse=None,
                    std=None, p10=None, p90=None, abs_p90=None)
    if not all(finite(v) for v in values):
        raise ValueError('Invalid statistic input')
    n = len(values)
    mean = sum(values)/n
    return dict(n=n, mean=mean, median=quantile(values, .5),
        mae=sum(abs(v) for v in values)/n,
        rmse=math.sqrt(sum(v*v for v in values)/n),
        std=math.sqrt(sum((v-mean)**2 for v in values)/n),
        p10=quantile(values, .1), p90=quantile(values, .9),
        abs_p90=quantile([abs(v) for v in values], .9))


def correlation(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
    vx = sum((v-mx)**2 for v in xs)
    vy = sum((v-my)**2 for v in ys)
    if vx <= 1e-20 or vy <= 1e-20:
        return None
    return sum((x-mx)*(y-my) for x, y in zip(xs, ys))/math.sqrt(vx*vy)


def wrap(angle):
    return (angle+math.pi/2) % math.pi-math.pi/2


def geometry(box):
    if (not isinstance(box, list) or len(box) not in (5, 6) or
            not all(finite(v) for v in box) or min(box[2:4]) <= 0):
        raise ValueError('Expected finite positive original-coordinate OBB')
    width, height = box[2:4]
    return dict(long=max(width, height), short=min(width, height),
        diagonal=math.hypot(width, height),
        angle=wrap(box[4]+(math.pi/2 if width < height else 0)),
        x=box[0], y=box[1])


def existing_selection_rule():
    """Execute only the existing trusted local selector functions, no imports.

    Never execute artifact text. This is a consistency check of the original
    selection, not selection on new residuals or reading other epochs' boxes.
    """
    source = ROOT / 'crane_project/tools/ckpt_sweep.py'
    tree = ast.parse(source.read_text())
    config = next(ast.literal_eval(n.value) for n in tree.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and
            t.id == 'SELECTION_CONFIG' for t in n.targets))
    wanted = {'extract_metrics', 'select_best_checkpoint'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    if {n.name for n in nodes} != wanted:
        raise ValueError('Existing selector unavailable')
    namespace = {'math': math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    return config, namespace['select_best_checkpoint']


def checked_identity(identity):
    manifest_path = ROOT / 'crane_project/tools/port_geometry_midpoint_formal_v1_sources.json'
    manifest = json_read(manifest_path.read_bytes())
    if (identity.get('sources') != manifest['sources'] or
            identity.get('sources_sha256') != sha(manifest_path.read_bytes())):
        raise ValueError('Formal source identity differs')
    for name, expected in manifest['sources'].items():
        if sha((ROOT/name).read_bytes()) != expected:
            raise ValueError('Formal source SHA differs: ' + name)
    protocol_path = ROOT / 'crane_project/tools/port_geometry_midpoint_formal_v1_protocol.json'
    protocol = json_read(protocol_path.read_bytes())
    if (identity.get('protocol_sha256') != sha(protocol_path.read_bytes()) or
            identity.get('frozen_b') != protocol['frozen_b']):
        raise ValueError('Frozen B/protocol identity differs')


def checked_rows(rows, comparison):
    if (len(rows) != 887 or len({r['image'] for r in rows}) != 887 or
            Counter(r['sequence'] for r in rows) != Counter(VAL_COUNTS)):
        raise ValueError('Requires fixed 887-frame VAL only')
    for row in rows:
        name = row['image']
        if (row['domain'] != row['sequence'].split('_')[0] or
                not isinstance(row['frame_id'], int) or isinstance(row['frame_id'], bool) or
                name.rsplit('_', 1)[0] != row['sequence'] or
                int(name.rsplit('_', 1)[1]) != row['frame_id'] or row['scale'] != 1.):
            raise ValueError('VAL frame metadata differs: ' + name)
        gt = geometry(row['gt'])
        if len(row['gt']) != 5:
            raise ValueError('GT must be Nx5')
        if (row['b'] is None) != (row['midpoint'] is None):
            raise ValueError('Output coverage changed: ' + name)
        for method in METHODS:
            box, value = row[method], row['metrics'][method]
            if value['output'] != (box is not None):
                raise ValueError('Output flag differs')
            if box is None:
                if value['center_hit'] or value['riou'] != 0 or value['angle_penalty_reason'] != 'no_output':
                    raise ValueError('Missing-output metric differs')
                continue
            pred = geometry(box)
            center = math.hypot(pred['x']-gt['x'], pred['y']-gt['y'])
            angle = abs(wrap(pred['angle']-gt['angle']))*180/math.pi
            expected = dict(center_error_px=center, angle_error_deg=angle,
                short_edge_relative_error=abs(pred['short']/gt['short']-1),
                long_edge_relative_error=abs(pred['long']/gt['long']-1),
                short_edge_signed_log_ratio=math.log(pred['short']/gt['short']),
                long_edge_signed_log_ratio=math.log(pred['long']/gt['long']),
                protocol_angle_error_deg=angle if center < 10 else 90.)
            if (len(box) != 6 or not .05 < box[5] <= 1 or
                    value['center_hit'] != (center < 15) or
                    value['angle_penalty_reason'] != ('center_ge_10px' if center >= 10 else None) or
                    not all(close(value[k], v) for k, v in expected.items()) or
                    not finite(value['riou']) or not 0 <= value['riou'] <= 1):
                raise ValueError('Stored VAL geometry metric differs: ' + name)
        if row['b'] is not None:
            if row['b'][5] != row['midpoint'][5] or not isinstance(row['accepted'], bool):
                raise ValueError('Score/acceptance differs')
            if row['accepted']:
                if row['candidate'] != row['midpoint']:
                    raise ValueError('Accepted candidate differs')
            elif row['midpoint'] != row['b']:
                raise ValueError('Full-B fallback differs')
    for group in ('real', 'sim'):
        subset = [r for r in rows if r['domain'] == group]
        for method in METHODS:
            value = comparison['groups'][group][method]
            output = [r for r in subset if r[method] is not None]
            correct = sum(r['metrics'][method]['center_hit'] for r in subset)
            expected = dict(output_coverage_fraction=dict(numerator=len(output), denominator=len(subset)),
                conditional_center_correct_fraction=dict(numerator=correct, denominator=len(output)),
                all_frame_center_correct_fraction=dict(numerator=correct, denominator=len(subset)))
            if any(value[k] != v for k, v in expected.items()):
                raise ValueError('Saved coverage denominator differs')
            for field, key in [('short', 'short_edge'), ('long', 'long_edge')]:
                residuals = [geometry(r[method])[field]/geometry(r['gt'])[field]-1 for r in output]
                if not close(stats(residuals)['mae'], value[key+'_relative_error']['mean']):
                    raise ValueError('Saved size aggregation differs')


def validate_inputs(read, online=False):
    """read(role/name) returns bytes. Only named VAL files are opened."""
    blobs = {}
    def get(name):
        if name not in blobs:
            blobs[name] = read(name)
        return blobs[name]
    def document(name):
        return json_read(get(name))
    selection = document('training/selection.json')
    completion = document('training/completion.json')
    artifacts = document('training/artifacts.json')['files']
    if (selection.get('protocol') != TRAIN_VERSION or selection.get('split') != 'val' or
            selection.get('test_access') is not False or selection.get('selection_on_test') is not False or
            completion.get('status') != 'FORMAL_MIDPOINT_TRAIN_VAL_COMPLETE_REVIEW_REQUIRED' or
            completion.get('epochs_completed') != 24 or completion.get('detector_updates') != 0 or
            completion.get('test_access') is not False or completion.get('selection') != selection):
        raise ValueError('Requires completed formal training and unchanged VAL selection')
    checked_identity(selection['identity'])
    selected = selection['selected_checkpoint']
    epoch = selected['epoch']
    if (not isinstance(epoch, int) or isinstance(epoch, bool) or not 1 <= epoch <= 24 or
            selected['path'] != 'head_epoch_%02d.pth' % epoch or
            set(selection['all_checkpoints']) != {'head_epoch_%02d.pth' % e for e in range(1, 25)}):
        raise ValueError('Formal checkpoint metadata differs')
    config, selector = existing_selection_rule()
    if selection['selection_config'] != config:
        raise ValueError('Original selection config differs')
    chosen, _, info = selector(selection['all_checkpoints'], config)
    if (chosen != selected['path'] or info != selection['selection_info'] or
            selection['all_checkpoints'][chosen]['checkpoint'] != selected or
            artifacts.get(chosen) != selected['sha256']):
        raise ValueError('Original full-VAL selection differs')
    row_name = 'val_epoch_%02d.rows.jsonl' % epoch
    summary_name = 'val_epoch_%02d.json' % epoch
    for name in ('selection.json', 'completion.json', 'selected_val_compare.json', row_name, summary_name):
        if artifacts.get(name) != sha(get('training/'+name)):
            raise ValueError('Training artifact SHA differs: ' + name)
    comparison = document('training/selected_val_compare.json')
    if comparison.get('split') != 'val' or comparison != document('training/'+summary_name):
        raise ValueError('Selected VAL summary differs')
    for e in range(1, 25):
        name = 'val_epoch_%02d.json' % e
        value = document('training/'+name)
        if (artifacts.get(name) != sha(get('training/'+name)) or value['split'] != 'val' or
                value['groups']['overall']['midpoint']['metric_protocol_v2'] !=
                selection['all_checkpoints']['head_epoch_%02d.pth' % e]['metrics']):
            raise ValueError('Original epoch VAL summary/selection metrics differ')
    rows = [json_read(line) for line in get('training/'+row_name).splitlines() if line.strip()]
    checked_rows(rows, comparison)
    proof = dict(selected_checkpoint=selected, selection_info=info,
        source_identity=selection['identity'], independent_online_val=None,
        weight_bytes_read=False, dataset_image_annotation_bytes_read=False, selected_rows=row_name)
    if online:
        online_artifacts = document('online/artifacts.json')['files']
        for name in ('completion.json', 'val_compare.json', 'val_rows.jsonl'):
            if online_artifacts.get(name) != sha(get('online/'+name)):
                raise ValueError('Online VAL artifact SHA differs: ' + name)
        report = document('online/completion.json')
        if (report.get('split') != 'val' or report.get('status') != 'FROZEN_FORMAL_MIDPOINT_VAL_COMPLETE_REVIEW_REQUIRED' or
                report.get('test_access') is not False or report.get('selection_on_test') is not False or
                report.get('frames') != 887 or report.get('head_updates') != 0 or report.get('detector_updates') != 0 or
                report.get('feature_extractions') != 887 or report.get('native_head_calls') != 2661 or
                report.get('state_before') != report.get('state_after') or
                report['state_before']['head'] != selected['head_digest'] or
                report['identity']['training_identity'] != selection['identity'] or
                report['selection']['selected_checkpoint'] != selected or
                report['selection']['selection_sha256'] != sha(get('training/selection.json')) or
                report['selection']['completion_sha256'] != sha(get('training/completion.json')) or
                report['data_identity']['annotation_sha256'] != selection['identity']['frozen_b']['annotation_sha256'] or
                document('online/val_compare.json') != comparison):
            raise ValueError('Independent online VAL identity/replay differs')
        fresh = [json_read(line) for line in get('online/val_rows.jsonl').splitlines() if line.strip()]
        checked_rows(fresh, comparison)
        indexed = {r['image']: r for r in rows}
        if {r['image'] for r in fresh} != set(indexed):
            raise ValueError('Online VAL frame identities differ')
        for row in fresh:
            previous = indexed[row['image']]
            for key in ('sequence', 'domain', 'frame_id', 'scale', 'gt', 'b', 'midpoint',
                        'candidate', 'accepted', 'candidate_valid', 'failed_checks', 'metrics'):
                if row[key] != previous[key]:
                    raise ValueError('Online VAL row differs: ' + row['image']+'/'+key)
        rows = fresh
        proof['independent_online_val'] = dict(frames=887, exact_rows_and_summary=True)
    proof['input_sha256'] = {name: sha(data) for name, data in blobs.items()}
    return rows, comparison, proof


def static_frame(row):
    gt = geometry(row['gt'])
    value = dict(image=row['image'], sequence=row['sequence'], domain=row['domain'],
        frame_id=row['frame_id'], accepted=row['accepted'], gt=gt)
    for method in METHODS:
        box = row[method]
        if box is None:
            value[method] = None
            continue
        pred = geometry(box)
        value[method] = dict(pred=pred,
            error_px={f: pred[f]-gt[f] for f in FIELDS},
            relative_error={f: pred[f]/gt[f]-1 for f in FIELDS},
            log_ratio={f: math.log(pred[f]/gt[f]) for f in FIELDS},
            center_error_px=row['metrics'][method]['center_error_px'],
            pure_angle_error_deg=row['metrics'][method]['angle_error_deg'],
            riou=row['metrics'][method]['riou'])
    value['log_correction'] = ({f: math.log(value['midpoint']['pred'][f]/value['b']['pred'][f])
        for f in FIELDS} if value['b'] is not None else None)
    return value


def temporal_pairs(frames):
    pairs = []
    counts = Counter(sequence_or_gap_boundaries=0, adjacent_identity_pairs=0,
                     missing_output_pairs=0, used_pairs=0)
    previous = None
    for current in sorted(frames, key=lambda r: (r['sequence'], r['frame_id'])):
        if previous is not None:
            if current['sequence'] != previous['sequence'] or current['frame_id'] != previous['frame_id']+1:
                counts['sequence_or_gap_boundaries'] += 1
            else:
                counts['adjacent_identity_pairs'] += 1
                if previous['b'] is None or current['b'] is None:
                    counts['missing_output_pairs'] += 1
                else:
                    pair = dict(sequence=current['sequence'], domain=current['domain'],
                        previous=previous['image'], image=current['image'],
                        previous_frame_id=previous['frame_id'], frame_id=current['frame_id'],
                        acceptance_switch=previous['accepted'] != current['accepted'],
                        gt={}, b={}, midpoint={}, log_correction_change={})
                    for f in FIELDS:
                        gt_rel = current['gt'][f]/previous['gt'][f]-1
                        gt_log = math.log(current['gt'][f]/previous['gt'][f])
                        pair['gt'][f] = dict(relative_increment=gt_rel, log_increment=gt_log)
                        for method in METHODS:
                            a, b = previous[method]['pred'][f], current[method]['pred'][f]
                            inc, log_inc = b/a-1, math.log(b/a)
                            pair[method][f] = dict(relative_increment=inc, log_increment=log_inc,
                                relative_increment_error=inc-gt_rel,
                                log_increment_error=log_inc-gt_log)
                        pair['log_correction_change'][f] = current['log_correction'][f]-previous['log_correction'][f]
                        if not close(pair['midpoint'][f]['log_increment_error']-pair['b'][f]['log_increment_error'],
                                     pair['log_correction_change'][f]):
                            raise ValueError('Temporal residual decomposition differs')
                    gt_angle = wrap(current['gt']['angle']-previous['gt']['angle'])
                    pair['gt']['angle_increment_rad'] = gt_angle
                    for method in METHODS:
                        angle = wrap(current[method]['pred']['angle']-previous[method]['pred']['angle'])
                        pair[method]['angle_increment_deg'] = angle*180/math.pi
                        pair[method]['angle_increment_error_deg'] = wrap(angle-gt_angle)*180/math.pi
                        pair[method]['aci'] = max(0., min(1., 1-abs(angle)/(math.radians(35)+1e-9)))
                    pairs.append(pair)
                    counts['used_pairs'] += 1
        previous = current
    return pairs, dict(counts)


def fraction(numerator, denominator):
    return dict(numerator=numerator, denominator=denominator,
                pct=100*numerator/denominator if denominator else None)


def group_summary(frames, pairs):
    out = [r for r in frames if r['b'] is not None]
    result = dict(frames=len(frames), output_frames=len(out), temporal_pairs=len(pairs),
        output_coverage=fraction(len(out), len(frames)),
        fallbacks=sum(r['accepted'] is False for r in frames),
        acceptance_switch_pairs=sum(r['acceptance_switch'] for r in pairs),
        gt_motion={}, b={}, midpoint={}, paired={})
    for f in FIELDS:
        result['gt_motion'][f] = dict(
            relative_increment=stats(p['gt'][f]['relative_increment'] for p in pairs),
            log_increment=stats(p['gt'][f]['log_increment'] for p in pairs))
    for method in METHODS:
        hits = sum(r[method]['center_error_px'] < 15 for r in out)
        result[method] = dict(conditional_center_correct=fraction(hits, len(out)),
            all_frame_center_correct=fraction(hits, len(frames)), static={}, temporal={},
            center_error_px=stats(r[method]['center_error_px'] for r in out),
            pure_angle_error_deg=stats(r[method]['pure_angle_error_deg'] for r in out),
            all_frame_mean_riou=sum(r[method]['riou'] for r in out)/len(frames) if frames else None)
        for f in FIELDS:
            result[method]['static'][f] = dict(
                error_px=stats(r[method]['error_px'][f] for r in out),
                relative_error=stats(r[method]['relative_error'][f] for r in out),
                log_ratio=stats(r[method]['log_ratio'][f] for r in out),
                under_gt_count=sum(r[method]['relative_error'][f] < 0 for r in out),
                over_gt_count=sum(r[method]['relative_error'][f] > 0 for r in out))
            result[method]['temporal'][f] = dict(
                relative_increment=stats(p[method][f]['relative_increment'] for p in pairs),
                relative_increment_error=stats(p[method][f]['relative_increment_error'] for p in pairs),
                log_increment_error=stats(p[method][f]['log_increment_error'] for p in pairs),
                gt_prediction_increment_correlation=correlation(
                    [p['gt'][f]['log_increment'] for p in pairs],
                    [p[method][f]['log_increment'] for p in pairs]))
        result[method]['dfr_pct_per_frame'] = (100*stats(p[method]['diagonal']['relative_increment'] for p in pairs)['mae']
                                              if pairs else None)
        result[method]['aci'] = stats(p[method]['aci'] for p in pairs)['mean']
        result[method]['angle_increment_error_deg'] = stats(p[method]['angle_increment_error_deg'] for p in pairs)
    result['gt_dfr_same_pairs_pct_per_frame'] = (100*stats(p['gt']['diagonal']['relative_increment'] for p in pairs)['mae']
                                               if pairs else None)
    for f in FIELDS:
        result['paired'][f] = dict(
            log_correction=stats(r['log_correction'][f] for r in out),
            log_correction_change=stats(p['log_correction_change'][f] for p in pairs),
            abs_relative_error_delta=stats(abs(r['midpoint']['relative_error'][f])-abs(r['b']['relative_error'][f]) for r in out),
            abs_log_increment_error_delta=stats(abs(p['midpoint'][f]['log_increment_error'])-abs(p['b'][f]['log_increment_error']) for p in pairs),
            crossed_gt_count=sum(r['b']['relative_error'][f]*r['midpoint']['relative_error'][f] < 0 for r in out))
    return result


def size_bin(length):
    for upper in (16, 24, 32, 48):
        if length < upper:
            return 'lt_%d' % upper
    return 'ge_48'


def analyze(rows, comparison):
    frames = [static_frame(r) for r in rows]
    pairs, support = temporal_pairs(frames)
    groups = {'overall': frames}
    groups.update({name: [r for r in frames if r['domain'] == name or r['sequence'] == name]
                   for name in ('real', 'sim')+tuple(VAL_COUNTS)})
    summaries = {}
    for name, subset in groups.items():
        ids = {r['image'] for r in subset}
        ps = [p for p in pairs if p['image'] in ids and p['previous'] in ids]
        summaries[name] = group_summary(subset, ps)
        if name != 'overall':
            for method in METHODS:
                original = comparison['groups'][name][method]['metric_protocol_v2']
                domain = subset[0]['domain']
                for field, key in [('dfr_pct_per_frame', 'DFR(%/frame)'), ('aci', 'ACI')]:
                    if not close(round(summaries[name][method][field], 4), original[domain+'/'+key]):
                        raise ValueError('Original temporal metric differs: '+name+'/'+method+'/'+key)
    strata = {}
    for ref in ('gt', 'b'):
        strata[ref+'_short_original_px'] = {}
        for domain in ('real', 'sim'):
            for label in ('lt_16', 'lt_24', 'lt_32', 'lt_48', 'ge_48'):
                subset = [r for r in frames if r['domain'] == domain and r['b'] is not None and
                          size_bin(r['gt']['short'] if ref == 'gt' else r['b']['pred']['short']) == label]
                ids = {r['image'] for r in subset}
                ps = [p for p in pairs if p['previous'] in ids and p['image'] in ids]
                strata[ref+'_short_original_px'][domain+'/'+label] = group_summary(subset, ps)
    # Use existing center thresholds only, without suppressing the all-output
    # primary result. Wrong-centre boxes can otherwise confound size diagnosis.
    strata['center_support_existing_15px'] = {}
    for domain in ('real', 'sim'):
        for label in ('b_center_lt15', 'b_center_ge15', 'both_centers_lt15'):
            subset = [r for r in frames if r['domain'] == domain and r['b'] is not None and
                (r['b']['center_error_px'] < 15 if label == 'b_center_lt15' else
                 r['b']['center_error_px'] >= 15 if label == 'b_center_ge15' else
                 r['b']['center_error_px'] < 15 and r['midpoint']['center_error_px'] < 15)]
            ids = {r['image'] for r in subset}
            ps = [p for p in pairs if p['previous'] in ids and p['image'] in ids]
            strata['center_support_existing_15px'][domain+'/'+label] = group_summary(subset, ps)
    summary = dict(protocol=VERSION, split='val', status='VAL_SIZE_TEMPORAL_RESIDUALS_COMPLETE_REVIEW_REQUIRED',
        groups=summaries, strata=strata, temporal_support=support,
        definitions=dict(
            signed_relative_error='pred_length/gt_length-1; negative=undersized',
            log_ratio='log(pred_length/gt_length); signed static bias',
            relative_increment='current/previous-1, separately for GT and predictions',
            log_increment_error='delta log(pred)-delta log(GT)=delta static log_ratio',
            log_correction_change='delta log(midpoint/B); equals midpoint minus B log increment errors',
            dfr='100*mean(abs(predicted diagonal relative increment)); lower does not prove better GT tracking',
            aci='Original periodic-angle continuity proxy, 35deg limit; separate from GT-relative angle increment errors',
            support='Same sequence and consecutive frame IDs; both frames B/midpoint present; missing outputs and gaps break pairs',
            bins='Original-image px, descriptive disjoint bins; not feature/input pixels or chosen training thresholds'),
        limitations=['GT changes are annotation-derived; not physical motion/depth ground truth.',
            'Correlations and residual changes are descriptive, not unique causal proof or significance.',
            'No smoothed boxes, extra thresholds, loss choices, checkpoint reselection or automatic promotion.'])
    return summary, frames, pairs


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def run(args):
    directories = {'training': Path(args.training_dir)}
    if args.val_eval_dir:
        directories['online'] = Path(args.val_eval_dir)
    def read(name):
        role, file = name.split('/', 1)
        if Path(file).name != file:
            raise ValueError('Only fixed flat VAL artifact names allowed')
        return (directories[role]/file).read_bytes()
    rows, comparison, proof = validate_inputs(read, online=bool(args.val_eval_dir))
    out = Path(args.out_dir)
    # Existing output directories are never overwritten.
    out.mkdir(parents=True, exist_ok=False)
    completion = dict(protocol=VERSION, split='val', head_updates=0, detector_updates=0,
        test_access=False, selection_on_test=False, automatic_promotion=False,
        inference_calls=0, cuda_imported=False, proof=proof,
        analyzer_sha256=sha(Path(__file__).read_bytes()))
    if args.check_only:
        completion['status'] = 'STATIC_VAL_RESIDUAL_INPUTS_PASS_NO_GPU_NO_UPDATES'
    else:
        summary, frames, pairs = analyze(rows, comparison)
        write_json(out/'summary.json', summary)
        for name, values in [('static_frames.jsonl', frames), ('temporal_pairs.jsonl', pairs)]:
            with (out/name).open('x') as stream:
                for value in values:
                    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')
        completion.update(status=summary['status'], frames=len(frames), temporal_pairs=len(pairs))
        for domain in ('real', 'sim'):
            value = summary['groups'][domain]
            print('VAL', domain, 'frames', value['frames'], 'pairs', value['temporal_pairs'])
            for method in METHODS:
                print(method, 'short MAE', value[method]['static']['short']['relative_error']['mae'],
                    'short log bias', value[method]['static']['short']['log_ratio']['mean'],
                    'DFR', value[method]['dfr_pct_per_frame'],
                    'diagonal motion residual RMSE', value[method]['temporal']['diagonal']['log_increment_error']['rmse'],
                    'ACI', value[method]['aci'])
    write_json(out/'completion.json', completion)
    write_json(out/'artifacts.json', dict(files={p.name: sha(p.read_bytes()) for p in out.iterdir() if p.is_file()}))
    print('Saved', out/'completion.json', 'status', completion['status'])


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-dir', default=str(FORMAL))
    p.add_argument('--val-eval-dir', help='Optional completed independent online VAL directory; checks exact replay')
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--out-dir', required=True)
    return p


if __name__ == '__main__':
    run(parser().parse_args())
