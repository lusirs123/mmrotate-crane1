"""Read-only enclosing-OBB readout diagnostics, not GT or training approval.

Native runs the repository's exact le90 function with float32/OpenCV. Local
reference is a float64 geometric calculation and never claimed native output.
Independent same-frame axes/repeat annotations remain optional human inputs.
"""
import argparse
import ast
import copy
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('packet_tool', ROOT/'tools/data/prepare_port_independent_size_labels_v1.py')
packet = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(packet)
VERSION = 'port_size_annotation_readout_v1'
TARGET = 'top_beam_body_including_trapezoidal_protrusions_along_long_edges'
REPEAT_IMAGES = ('candidate08_f00585.png', 'candidate08_f01048.png', 'candidate11_f00008.png')
PACKET_SHA = 'ec1b4cc0d53da8b0762c0163ecd28c6c3f284a5ac986367cc4f83bb8af1653c7'
NATIVE_SHA = 'e133b1cfd3f4fe8e6aeb05d4d6d93d16b0d52a89ebcdeff43d46887729f1f53c'


def angle(theta):
    return (theta+math.pi/2) % math.pi-math.pi/2


def angle_delta(a, b):
    return abs(angle(a-b))*180/math.pi


def project(points, theta):
    u = (math.cos(theta), math.sin(theta)); n = (-u[1], u[0])
    x = [p[0]*u[0]+p[1]*u[1] for p in points]
    y = [p[0]*n[0]+p[1]*n[1] for p in points]
    cx, cy = (min(x)+max(x))/2, (min(y)+max(y))/2
    return dict(center_px=[cx*u[0]+cy*n[0], cx*u[1]+cy*n[1]],
                along_axis_px=max(x)-min(x), transverse_px=max(y)-min(y),
                angle_rad=angle(theta))


def proxy_axis(points):
    edges = [(points[(i+1)%4][0]-points[i][0], points[(i+1)%4][1]-points[i][1]) for i in range(4)]
    means = [(math.hypot(*edges[i])+math.hypot(*edges[i+2]))/2 for i in (0, 1)]
    i = 0 if means[0] >= means[1] else 1
    dx, dy = edges[i][0]-edges[i+2][0], edges[i][1]-edges[i+2][1]
    return angle(math.atan2(dy, dx))


def reference_obb(points):
    candidates = []
    for i in range(4):
        q = points[(i+1)%4]; p = points[i]
        theta = math.atan2(q[1]-p[1], q[0]-p[0]); r = project(points, theta)
        w, h = r['along_axis_px'], r['transverse_px']
        if w < h: w, h, theta = h, w, theta+math.pi/2
        candidates.append(dict(center_px=r['center_px'], long_px=w, short_px=h, angle_rad=angle(theta)))
    return min(candidates, key=lambda r: (r['long_px']*r['short_px'], abs(r['angle_rad'])))


def native_loader():
    import cv2
    import numpy as np
    path = ROOT/'mmrotate/core/bbox/transforms.py'
    if packet.sha(path) != NATIVE_SHA: raise ValueError('Native repository source identity differs')
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'poly2obb_np_le90']
    if len(nodes) != 1: raise ValueError('Native le90 function missing/ambiguous')
    # Execute only the existing standalone function, avoiding Torch/MMCV imports.
    module = ast.Module(body=nodes, type_ignores=[]); ast.fix_missing_locations(module)
    namespace = dict(np=np, cv2=cv2)
    exec(compile(module, str(path), 'exec'), namespace)
    def read(points):
        result = namespace['poly2obb_np_le90'](np.asarray(points, dtype=np.float32).reshape(8))
        if result is None: raise ValueError('Native le90 rejected sides below 2px')
        x, y, w, h, theta = map(float, result)
        return dict(center_px=[x, y], long_px=w, short_px=h, angle_rad=theta)
    return read, dict(name='native_repository_le90_float32', cv2_version=cv2.__version__,
                     numpy_version=np.__version__, function_file_sha256=packet.sha(path))


def checked_packet(path):
    path = Path(path).resolve()
    if path.parent != packet.PACKET_ROOT or not path.name.startswith(packet.VERSION+'_'):
        raise ValueError('Use the owned Git-synced annotation packet')
    if packet.sha(path/'packet.json') != PACKET_SHA: raise ValueError('Frozen packet differs')
    receipt = json.loads((path/'packet.json').read_text())
    if receipt['split_assigned'] is not False or receipt['ready_for_training'] is not False:
        raise ValueError('Cannot use promoted/reassigned packet')
    if len(receipt['frames']) != 20 or len({r['image'] for r in receipt['frames']}) != 20:
        raise ValueError('Expected fixed 20-frame plan')
    rows = []; sources = {str(path/'packet.json'): PACKET_SHA}
    for frame in receipt['frames']:
        ann = packet.owned(path/frame['annotation'], path/'annotations')
        image = packet.owned(path/frame['image'], path/'images')
        if packet.sha(image) != frame['image_sha256']: raise ValueError('Original PNG differs')
        d = json.loads(ann.read_text())
        points = enclosing_points(d, frame)
        rows.append(dict(frame=frame, annotation=d, points=points))
        sources[str(ann)] = packet.sha(ann); sources[str(image)] = frame['image_sha256']
    expected = {str((path/r['frame']['annotation']).resolve()) for r in rows}
    if {str(f.resolve()) for f in (path/'annotations').glob('*.json')} != expected:
        raise ValueError('Unexpected annotation inventory')
    return path, rows, sources


def enclosing_points(document, frame):
    flags = document.get('flags', {})
    if flags.get('independent_reviewed') is not True or flags.get('unmeasurable') is not False:
        raise ValueError('Enclosing annotation needs completed human review')
    name = Path(frame['image']).name
    if document.get('imagePath') not in (name, '../images/'+name, '../../images/'+name):
        raise ValueError('Enclosing annotation image identity differs')
    d = copy.deepcopy(document); d['imagePath'] = name
    d['flags']['independent_reviewed'] = False
    # Reuse finite/in-bounds/convex/order/type checks, without falsely declaring
    # visible physical corners. TARGET is the explicit human-declared contract.
    packet.validate_annotation(d, frame)
    if len(d.get('shapes', [])) != 1: raise ValueError('Expected one enclosing polygon')
    return d['shapes'][0]['points']


def prepare(path):
    path, rows, sources = checked_packet(path)
    out = path/'size_readout_inputs_v1'
    if out.exists(): raise FileExistsError('Keep existing manual inputs: '+str(out))
    out.mkdir(); (out/'repeat_annotations').mkdir()
    records = {}
    for r in rows:
        f = r['frame']; name = Path(f['image']).name
        records[name] = dict(image_sha256=f['image_sha256'], axis_points=[], axis_reviewed=False,
            axis_source='PENDING_MANUAL_SAME_FRAME_AXIS', reference_k=None, k_basis='')
        if name in REPEAT_IMAGES:
            packet.write_json(out/'repeat_annotations'/(Path(name).stem+'.json'),
                dict(version='3.3.10', flags=dict(independent_reviewed=False, unmeasurable=False),
                     shapes=[], imagePath='../../images/'+name, imageData=None,
                     imageWidth=f['size'][0], imageHeight=f['size'][1],
                     object_definition=TARGET, measurement_basis='manual_enclosing_boundary_repeat'))
    packet.write_json(out/'measurements.json', dict(protocol=VERSION, target=TARGET, records=records,
        instructions='Axis endpoints must be measured independently on the same original frame; '
        'never copy polygon-derived direction, use detector predictions, or infer width from k. '
        'Only supply reference_k with explicit object/sequence basis. Repeats should be blind to first polygon.'))
    for f, digest in sources.items():
        if packet.sha(f) != digest: raise ValueError('Source changed during preparation')
    print('READOUT_INPUTS_PREPARED axes=0/20 independent_repeats=0/3 '+str(out))


def axis_measurement(record, frame, points):
    if record['image_sha256'] != frame['image_sha256']: raise ValueError('Axis PNG binding differs')
    if type(record.get('axis_reviewed')) is not bool: raise ValueError('Axis reviewed must be Boolean')
    if record.get('axis_reviewed') is not True:
        if record.get('axis_points') or record.get('reference_k') is not None:
            raise ValueError('Axis/k entered but independent axis not reviewed')
        return dict(status='PENDING_INDEPENDENT_SAME_FRAME_AXIS', fixed_k_status='NOT_EVALUATED')
    p = record.get('axis_points')
    if record.get('axis_source') != 'manual_same_frame_long_axis': raise ValueError('Axis source not independently declared')
    if not isinstance(p, list) or len(p) != 2 or not all(isinstance(x, list) and len(x)==2 and
        all(type(v) in (float, int) and math.isfinite(v) for v in x) for x in p):
        raise ValueError('Expected two finite independent axis endpoints')
    if not all(0 <= x[0] < frame['size'][0] and 0 <= x[1] < frame['size'][1] for x in p):
        raise ValueError('Axis outside original image')
    length = math.dist(p[0], p[1])
    if length < 2: raise ValueError('Axis too short')
    theta = math.atan2(p[1][1]-p[0][1], p[1][0]-p[0][0]); box = project(points, theta)
    result = dict(status='HUMAN_DECLARED_INDEPENDENT_AXIS', axis_length_px=length,
                  fixed_axis_enclosing=box, fixed_k_status='NOT_EVALUATED')
    k = record.get('reference_k')
    if k is not None:
        basis = record.get('k_basis')
        if type(k) not in (float, int) or not math.isfinite(k) or k <= 1 or not isinstance(basis, str) or not basis.strip():
            raise ValueError('k requires a positive justified same-object reference')
        generated = length/k
        result.update(fixed_k_status='PAIRED_GEOMETRIC_DISCREPANCY_NOT_TRUTH_ERROR',
            generated_short_px=generated,
            generated_minus_measured_relative=generated/box['transverse_px']-1)
    return result


def descriptive(values):
    """Frame-level descriptions, not independent-event significance tests."""
    if not values:
        return dict(count=0, mean=None, median=None, abs_p95=None,
                    positive=0, negative=0, zero=0)
    ordered = sorted(abs(v) for v in values); pos = .95*(len(values)-1)
    lo = math.floor(pos); hi = math.ceil(pos)
    return dict(count=len(values), mean=statistics.mean(values),
                median=statistics.median(values),
                abs_p95=ordered[lo]+(ordered[hi]-ordered[lo])*(pos-lo),
                positive=sum(v > 1e-12 for v in values),
                negative=sum(v < -1e-12 for v in values),
                zero=sum(abs(v) <= 1e-12 for v in values))


def check_axis_annotation_source(record, frame, input_dir, sources):
    """Bind imported GUI coordinates to their retained manual source file."""
    if 'axis_annotation_file' not in record and 'axis_annotation_sha256' not in record:
        return  # Existing manually entered measurements retain their contract.
    if not isinstance(record.get('axis_annotation_file'), str) or not isinstance(record.get('axis_annotation_sha256'), str):
        raise ValueError('Imported axis source needs both file and SHA')
    path = packet.owned(input_dir/record['axis_annotation_file'], input_dir/'axis_annotations')
    name = Path(frame['image']).name
    if path.name != Path(name).stem+'.json' or packet.sha(path) != record.get('axis_annotation_sha256'):
        raise ValueError('Imported axis annotation filename/SHA differs')
    document = json.loads(path.read_text()); shapes = document.get('shapes', [])
    if (document.get('imagePath') not in (name, '../../images/'+name) or
        (document.get('imageWidth'), document.get('imageHeight')) != tuple(frame['size']) or
        not isinstance(shapes, list) or len(shapes) != 1 or shapes[0].get('label') != 'axis' or
        shapes[0].get('shape_type') != 'line' or shapes[0].get('points') != record['axis_points']):
        raise ValueError('Imported axis annotation coordinates/image contract differs')
    sources[str(path)] = packet.sha(path)


def same_frame_pair(axis, points, obb, record, repeat_points=None):
    """Compare label measurements; four-point annotation is not assumed truth.

    Primary comparison fixes independent axis direction. Free native OBB is a
    separate comparison because center/direction/extent can change together.
    """
    if axis['status'] != 'HUMAN_DECLARED_INDEPENDENT_AXIS':
        return dict(status='MISSING_INDEPENDENT_SAME_FRAME_AXIS', eligible=False)
    fixed = axis['fixed_axis_enclosing']; length = axis['axis_length_px']
    if fixed['along_axis_px'] < fixed['transverse_px']:
        return dict(status='AXIS_LONG_IDENTITY_CONFLICT_REVIEW_REQUIRED', eligible=False)
    along, width = fixed['along_axis_px'], fixed['transverse_px']
    k = record.get('reference_k')
    endpoints = record['axis_points']
    result = dict(status='LONG_DIRECTION_COMPARISON_ONLY_K_UNCONFIRMED' if k is None else
                  'PAIRED_AXIS_K_AND_ENCLOSING_ANNOTATIONS', eligible=True,
        primary=dict(axis_long_px=length, four_point_along_axis_px=along,
            four_point_transverse_px=width,
            long_delta_px=length-along, long_relative=length/along-1,
            long_log_ratio=math.log(length/along)),
        free_obb=dict(axis_vs_obb_angle_deg=angle_delta(fixed['angle_rad'], obb['angle_rad']),
            long_relative=length/obb['long_px']-1,
            center_delta_px=math.dist([(endpoints[0][i]+endpoints[1][i])/2 for i in (0,1)],obb['center_px'])),
        scope='annotation_discrepancy_not_GT_error_or_training_gate')
    if k is not None:
        generated = length/k; scale_log = math.log(length/along)
        ratio_log = math.log((along/width)/k)
        result['primary'].update(reference_k=k, generated_short_px=generated,
            short_delta_px=generated-width, short_relative=generated/width-1,
            short_log_ratio=math.log(generated/width),
            short_log_decomposition=dict(long_extent_component=scale_log,
                projected_ratio_component=ratio_log, total=scale_log+ratio_log,
                scope='exact_algebra_not_causal_attribution'))
        result['free_obb'].update(short_relative=generated/obb['short_px']-1,
            axis_k_vs_obb_ratio_log=math.log(k/(obb['long_px']/obb['short_px'])))
    if repeat_points is not None:
        again = project(repeat_points, fixed['angle_rad'])
        repeat = dict(long_relative=again['along_axis_px']/along-1,
            short_relative=again['transverse_px']/width-1,
            short_delta_px=again['transverse_px']-width,
            points_identical_to_first=repeat_points==points,
            scope='one_repeat_difference_not_independent_truth_or_error_bound')
        if k is not None:
            gap1, gap2 = generated-width, generated-again['transverse_px']
            spread = abs(again['transverse_px']-width)
            repeat.update(generated_vs_repeat_short_relative=generated/again['transverse_px']-1,
                short_gap_same_sign=gap1*gap2 > 0,
                both_short_gaps_exceed_observed_repeat_difference=min(abs(gap1),abs(gap2)) > spread,
                no_systematic_bias_inference=True)
        result['fixed_axis_repeat'] = repeat
    return result


def pair_summary(rows, group_by_image):
    eligible = [r for r in rows if r['same_frame_pair']['eligible']]
    paired = [r for r in eligible if 'short_relative' in r['same_frame_pair']['primary']]
    def group_stats(items):
        has_k = [r for r in items if 'short_relative' in r['same_frame_pair']['primary']]
        return dict(axis_count=len(items), paired_k_count=len(has_k),
            long_relative=descriptive([r['same_frame_pair']['primary']['long_relative'] for r in items]),
            short_relative=descriptive([r['same_frame_pair']['primary']['short_relative'] for r in has_k]),
            native_angle_delta_deg=descriptive([r['same_frame_pair']['free_obb']['axis_vs_obb_angle_deg'] for r in items]),
            paired_repeats=sum('fixed_axis_repeat' in r['same_frame_pair'] for r in has_k),
            reference_k_values=sorted({r['same_frame_pair']['primary']['reference_k'] for r in has_k}))
    groups = {name: group_stats([r for r in eligible if group_by_image[r['image']]==name])
              for name in sorted(set(group_by_image.values()))}
    return dict(eligible_axis_count=len(eligible), paired_k_count=len(paired),
        missing_axis_count=sum(r['same_frame_pair']['status']=='MISSING_INDEPENDENT_SAME_FRAME_AXIS' for r in rows),
        long_identity_conflict_count=sum(r['same_frame_pair']['status']=='AXIS_LONG_IDENTITY_CONFLICT_REVIEW_REQUIRED' for r in rows),
        pooled_frame_descriptions=group_stats(eligible), by_source_video=groups,
        systematic_bias_status='NOT_ESTABLISHED_REQUIRES_REPEATABILITY_OBJECT_AND_EVENT_EVIDENCE',
        supervision_only_training_contrast_ready=False,
        limitations='Direction-fixed differences isolate readout orientation, not annotation truth. '
        'Adjacent frames are correlated. Mixed k within a video needs object/sequence review. '
        'No automatic significance, causal conclusion, GT replacement or training approval.')


def diagnose(path, out, backend, require_axis=False):
    path, rows, sources = checked_packet(path)
    out = Path(out).resolve()
    if out.parent != ROOT/'work_dirs' or not out.name.startswith(VERSION+'_') or out.exists():
        raise ValueError('Use a new work_dirs/'+VERSION+'_... result directory')
    for f in (Path(__file__), ROOT/'tools/data/prepare_port_independent_size_labels_v1.py',
              ROOT/'tools/data/prepare_port_size_reference_review_v1.py',
              ROOT/'crane_project/utils/port_geometry_size_support_v1.py',
              ROOT/'crane_project/utils/port_midpoint_depth_v1.py',
              ROOT/'tests/test_port_size_annotation_readout_v1.py', ROOT/'mmrotate/core/bbox/transforms.py'):
        sources[str(f)] = packet.sha(f)
    input_dir = path/'size_readout_inputs_v1'; measure_file = input_dir/'measurements.json'
    inputs = json.loads(measure_file.read_text()); sources[str(measure_file)] = packet.sha(measure_file)
    names = {Path(r['frame']['image']).name for r in rows}
    if inputs['protocol'] != VERSION or inputs['target'] != TARGET or set(inputs['records']) != names:
        raise ValueError('Measurement contract/frame identity differs')
    if {p.name for p in (input_dir/'repeat_annotations').glob('*.json')} != {Path(n).stem+'.json' for n in REPEAT_IMAGES}:
        raise ValueError('Repeat inventory differs from fixed three-frame plan')
    for r in rows:
        check_axis_annotation_source(inputs['records'][Path(r['frame']['image']).name],
                                     r['frame'], input_dir, sources)
    axes = {r['frame']['image']: axis_measurement(inputs['records'][Path(r['frame']['image']).name],
                                                r['frame'], r['points']) for r in rows}
    if require_axis and not any(a['status']=='HUMAN_DECLARED_INDEPENDENT_AXIS' for a in axes.values()):
        raise ValueError('SAME_FRAME_PAIRS_MISSING independent_axis=0/20. Fill independent same-frame '
                         'axis endpoints in measurements.json first; no report or training is generated.')
    if backend == 'native':
        reader, info = native_loader()
    else:
        reader = reference_obb; info = dict(name='reference_float64_edge_candidates_not_native')
    results = []
    for r in rows:
        points, frame = r['points'], r['frame']; name = Path(frame['image']).name
        obb = reader(points); theta = proxy_axis(points); fixed = project(points, theta)
        axis = axes[frame['image']]; again = None
        deltas = [project(points, theta+x*math.pi/180)['transverse_px']-fixed['transverse_px'] for x in (-1, 1)]
        lengths = [math.dist(points[i],points[(i+1)%4]) for i in range(4)]
        mean_short = min((lengths[0]+lengths[2])/2,(lengths[1]+lengths[3])/2)
        result = dict(image=frame['image'], native_obb=obb if backend=='native' else None,
            reference_obb=obb if backend=='reference' else None,
            quad_derived_direction_description=fixed,
            independent_axis=axis,
            obb_vs_quad_proxy_angle_deg=angle_delta(obb['angle_rad'], theta),
            obb_vs_quad_proxy_short_relative=obb['short_px']/fixed['transverse_px']-1,
            quad_opposite_edge_mean_short_px=mean_short,
            obb_vs_quad_edge_mean_short_relative=obb['short_px']/mean_short-1,
            proxy_direction_perturbation_1deg_short_delta_px=deltas,
            near_square_direction_ambiguous=obb['long_px']/obb['short_px'] < 1.05,
            repeat_status='NOT_IN_FIXED_REPEAT_PLAN')
        if axis['status']=='HUMAN_DECLARED_INDEPENDENT_AXIS':
            ref = axis['fixed_axis_enclosing']
            result['obb_vs_independent_axis'] = dict(angle_delta_deg=angle_delta(obb['angle_rad'],ref['angle_rad']),
                short_relative=obb['short_px']/ref['transverse_px']-1,
                along_axis_relative=obb['long_px']/ref['along_axis_px']-1)
        if name in REPEAT_IMAGES:
            result['repeat_status']='PENDING_INDEPENDENT_REPEAT'
            repeat_file = input_dir/'repeat_annotations'/(Path(name).stem+'.json')
            repeat = json.loads(repeat_file.read_text()); sources[str(repeat_file)] = packet.sha(repeat_file)
            if any(type(repeat.get('flags',{}).get(k)) is not bool for k in ('independent_reviewed','unmeasurable')):
                raise ValueError('Repeat review flags must be Boolean')
            if repeat.get('flags', {}).get('independent_reviewed') is True:
                if repeat.get('flags', {}).get('unmeasurable') is True:
                    if repeat.get('shapes') or not repeat.get('review_note', '').strip():raise ValueError('Unmeasurable repeat needs no guessed points and reason')
                    result['repeat_status']='UNMEASURABLE_REPEAT'
                else:
                    again = enclosing_points(repeat, frame); rb = reader(again)
                    result.update(repeat_status='REVIEWED_REPEAT_NOT_GROUND_TRUTH',
                        repeat_readout_delta=dict(center_px=math.dist(obb['center_px'],rb['center_px']),
                            short_relative=rb['short_px']/obb['short_px']-1,
                            long_relative=rb['long_px']/obb['long_px']-1,
                            angle_deg=angle_delta(rb['angle_rad'],obb['angle_rad'])),
                        repeat_points_identical_to_first=again==points)
                    if axis['status']=='HUMAN_DECLARED_INDEPENDENT_AXIS':
                        width = project(again, axis['fixed_axis_enclosing']['angle_rad'])['transverse_px']
                        result['repeat_fixed_axis_short_relative']=width/axis['fixed_axis_enclosing']['transverse_px']-1
            elif repeat.get('shapes'):
                result['repeat_status']='PENDING_REPEAT_REVIEW'
        result['same_frame_pair'] = same_frame_pair(axis, points, obb, inputs['records'][name], again)
        results.append(result)
    summary = dict(frames=len(results), independent_axis_count=sum(r['independent_axis']['status']=='HUMAN_DECLARED_INDEPENDENT_AXIS' for r in results),
        paired_k_count=sum(r['independent_axis']['fixed_k_status']!='NOT_EVALUATED' for r in results),
        completed_repeats=sum(r['repeat_status']=='REVIEWED_REPEAT_NOT_GROUND_TRUTH' for r in results),
        proxy_angle_delta_deg_median=statistics.median(r['obb_vs_quad_proxy_angle_deg'] for r in results),
        proxy_angle_delta_deg_max=max(r['obb_vs_quad_proxy_angle_deg'] for r in results),
        proxy_short_relative_abs_max=max(abs(r['obb_vs_quad_proxy_short_relative']) for r in results),
        edge_mean_vs_obb_short_relative_abs_max=max(abs(r['obb_vs_quad_edge_mean_short_relative']) for r in results),
        perturbation_1deg_short_abs_px_max=max(abs(x) for r in results for x in r['proxy_direction_perturbation_1deg_short_delta_px']))
    report = dict(protocol=VERSION, target=TARGET, backend=info, summary=summary, rows=results,
        same_frame_comparison_version='port_size_same_frame_comparison_v1',
        same_frame_comparison=pair_summary(results,{r['frame']['image']:r['frame']['source_video'] for r in rows}),
        ready_for_training=False, split_assigned=False, physical_reference_match='UNCONFIRMED',
        limitations='Readout differences/1deg sensitivity/repeatability are not truth errors or detector/depth gains. '
        'Quad-derived direction is not an independent axis. No label export or automatic promotion.', sources_sha256=sources)
    for f, digest in sources.items():
        if packet.sha(f) != digest: raise ValueError('Inputs changed during diagnostic')
    out.mkdir(); packet.write_json(out/'readout_report.json',report)
    packet.write_json(out/'artifacts.json', {'readout_report.json':packet.sha(out/'readout_report.json')})
    print('SIZE_ANNOTATION_READOUT_COMPLETE', json.dumps(summary), 'backend='+info['name'], 'ready_for_training=false')
    if require_axis:
        print('SAME_FRAME_SIZE_COMPARISON_COMPLETE', json.dumps(report['same_frame_comparison']))


def main():
    parser=argparse.ArgumentParser(description=__doc__); subs=parser.add_subparsers(dest='command',required=True)
    p=subs.add_parser('prepare');p.add_argument('--packet-dir',type=Path,required=True)
    for command in ('diagnose','paired'):
        p=subs.add_parser(command);p.add_argument('--packet-dir',type=Path,required=True)
        p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--backend',choices=('native','reference'),default='native')
    args=parser.parse_args()
    try:
        if args.command=='prepare': prepare(args.packet_dir)
        else: diagnose(args.packet_dir,args.out_dir,args.backend,require_axis=args.command=='paired')
    except (ValueError,FileNotFoundError,FileExistsError,ImportError) as error:
        parser.exit(2,'ERROR: '+str(error)+'\n')


if __name__=='__main__':main()
