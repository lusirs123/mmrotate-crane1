#!/usr/bin/env python3
"""Zero-update TRAIN/VAL size diagnosis; no model selection or TEST stage.

check needs only the standard library. diagnose additionally needs the existing
torch environment, the sealed sigma1.5/ep03 checkpoint and three CPU ROI shards.
Never builds a detector, extracts features, repairs boxes or trains a new head.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.utils import port_geometry_midpoint_edge_feasibility_v1 as geometry

VERSION = 'port_geometry_midpoint_edge_feasibility_v1'
TOOL_DIR = 'crane_project/tools/'
PROTOCOL_NAME = TOOL_DIR+VERSION+'_protocol.json'
SOURCES_NAME = TOOL_DIR+VERSION+'_sources.json'
SAMPLE_NAME = TOOL_DIR+'port_geometry_g_v1_train_samples.json'
FORMAL_PROTOCOL = TOOL_DIR+'port_geometry_midpoint_formal_v1_protocol.json'
FORMAL_SOURCES = TOOL_DIR+'port_geometry_midpoint_formal_v1_sources.json'
SIGMA_PROTOCOL = TOOL_DIR+'port_geometry_midpoint_sigma_v1_protocol.json'
SIGMA_SOURCES = TOOL_DIR+'port_geometry_midpoint_sigma_v1_sources.json'
PARENT_HASHES = {
    FORMAL_PROTOCOL: '359bc1e3128ff472c568197860d0c911de204699f1b3cf14a4e3e9ff40568ff3',
    FORMAL_SOURCES: '2d489e06b06ce390cd6dc3dfeee8a849134e37b103d11951e4830dda91c37804',
    SIGMA_PROTOCOL: '48e7f1144ea442ae45218a754013521b0de50ba27b399204c23b77f74776c65d',
    SIGMA_SOURCES: 'b3e2ab222a8c16930cf483d4870373b522abb04b3a5f793ff0ea92072aae18e3',
    SAMPLE_NAME: '616ac890b9a60689c7fb52c08bf58b4fe85e2542090b158d79ce0d429902209c'}
NEW_SOURCES = {
    PROTOCOL_NAME,
    TOOL_DIR+'diagnose_port_geometry_midpoint_edge_feasibility_v1.py',
    'crane_project/utils/port_geometry_midpoint_edge_feasibility_v1.py',
    'tests/test_port_geometry_midpoint_edge_feasibility_v1.py'}
CACHE_SHA = '046c5998dee0ba3703f1ae4e08fc6a02e9804d216ea12241357f69f2a4afd1e3'
FIXED = dict(path='head_epoch_03.pth', epoch=3, updates=2706, sigma_cells=1.5,
    sha256='16c2fb448ac4e1c53530b8086d547f6f6ccb8d6b0763a42391c34f9b337982d7',
    head_digest=dict(parameters='f84d99dac7cb0f2dcb86b5eaf33c21c1d360bdca5d0193b7b3da6c98f92bdc63',
        buffers='44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a',
        parameter_count=17696, buffer_count=0), save_reload_exact=True)
SHARDS = {'train_s1': (2558, 'train', 1.), 'train_s05': (2558, 'train', .5),
          'val_s1': (887, 'val', 1.)}
TRAIN_COUNTS = dict(real_seq01=339, real_seq05=560, real_seq06=466,
                    real_seq12=141, real_seq13=304, sim_seq08=748)
VAL_COUNTS = dict(real_seq07=226, real_seq14=149, sim_seq10=512)
TENSOR_KEYS = ('roi', 'support', 'boxes_original', 'boxes_model', 'scale_xy', 'gt_original')
DEFAULT_HEAD = 'work_dirs/crane_symeood_k1_port_day2night_midpoint_sigma_v1/sigma_1p5'
CACHE_BASENAME = 'port_geometry_midpoint_formal_v1_roi_cache'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=invalid)


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                indent=2, allow_nan=False)+'\n')


def protocol_document():
    return dict(protocol=VERSION, stages=['check', 'diagnose'], settings=geometry.SETTINGS,
        fixed_checkpoint=FIXED, parent_hashes=PARENT_HASHES, cache_manifest_sha256=CACHE_SHA,
        record_counts={k: v[0] for k, v in SHARDS.items()},
        objective='OFFLINE existence of both canonical edge errors <=10%, with fixed M center/raw angle/long-axis identity/score/output frames.',
        constraints='Intersect nominal original-B raw edge ratios [0.8,1.25], 10% GT intervals and four-midpoint ROI upper bounds; keep M raw order (strict when w<h). Actual sx/sy, original/model swaps and minimum16px context remain separate.',
        numeric='Float64 analytic inequalities do not widen the 10% target. Validate native M against original float32/device B guards, record differences from float64 guard arithmetic. Up to3 fixed certificate attempts are reconstructed through log/exp in original float32/device, checked with the original torch ROI map and existing B guards. Boundaries without a verified witness remain numeric_unresolved.',
        bypass='Exact-square M is frozen. Zero-residual/full-M fallback retains an already-correct M even if reconstructed M midpoints fail a new ROI constraint. No-output frames remain absent.',
        targets='Metric target sorted GT edges follow M raw order; fixed-B cyclic matched point-pair distances follow original-B raw order then model-B ROI order. Report disagreements, never relabel the training target.',
        inputs='Only sealed checkpoint/selected epoch03 indexed VAL rows and three existing CPU TRAIN/VAL shards. No historical selection, sigma1 replay, image/dataset reads, detector construction or cache rebuild. Verify source, byte SHA, checkpoint/context and head digest.',
        reporting='All TRAIN1/TRAIN0.5 and full VAL1; real/sim/sequences and existing64-image fit/probe roles with eligible subsets separately. Output coverage, conditional center correctness, full-frame center correctness and joint-size coverage use explicit denominators. GT certificates are not candidate predictions or measured improvements.',
        scope=dict(detector_updates=0, head_updates=0, feature_extractions=0,
            cache_rebuilt=False, test_access=False, selection_on_test=False,
            automatic_promotion=False, independent_size_head_implemented=False),
        limitations=['Feasible size does not prove ROI information or learning sufficiency.',
            'No RIoU/continuity or depth-precision guarantee from size certificates.',
            'Previously used TRAIN probe is diagnostic, not independent generalization.',
            'No smoke/probe/formal training, paper edit or reliability-front-end migration.'])


def checked_sources(root=ROOT):
    """Reconstruct sealed identities without importing historical training tools."""
    for name, digest in PARENT_HASHES.items():
        if sha(root/name) != digest:
            raise ValueError('Sealed parent source SHA differs: '+name)
    formal, sigma = read(root/FORMAL_SOURCES), read(root/SIGMA_SOURCES)
    manifest = read(root/SOURCES_NAME)
    required = set(sigma['sources']) | {SIGMA_SOURCES} | NEW_SOURCES
    if (manifest['protocol'] != VERSION or manifest['parent_sources_sha256'] != PARENT_HASHES[SIGMA_SOURCES]
            or set(manifest['sources']) != required or read(root/PROTOCOL_NAME) != protocol_document()):
        raise ValueError('Diagnostic source/protocol scope differs')
    for name, digest in manifest['sources'].items():
        if sha(root/name) != digest:
            raise ValueError('Diagnostic source SHA differs: '+name)
    if any(manifest['sources'][k] != v for k, v in sigma['sources'].items()):
        raise ValueError('Diagnostic parent source closure differs')
    fp = read(root/FORMAL_PROTOCOL)
    parent = dict(protocol_sha256=PARENT_HASHES[FORMAL_PROTOCOL],
        sources_sha256=PARENT_HASHES[FORMAL_SOURCES], sources=formal['sources'], frozen_b=fp['frozen_b'])
    training = dict(protocol_sha256=PARENT_HASHES[SIGMA_PROTOCOL],
        sources_sha256=PARENT_HASHES[SIGMA_SOURCES], sources=sigma['sources'], parent_identity=parent)
    return dict(protocol_sha256=sha(root/PROTOCOL_NAME), sources_sha256=sha(root/SOURCES_NAME),
        sources=manifest['sources'], training_identity=training)


def resolve_cache(explicit=None, root=ROOT):
    if explicit:
        path = Path(explicit).resolve()
        if not (path/'cache_manifest.json').is_file():
            raise ValueError('Explicit cache manifest missing: '+str(path))
        return path
    work = root/'work_dirs'
    candidates = [work/CACHE_BASENAME, work/'port_results/geometry'/CACHE_BASENAME]
    index = work/'port_results/INDEX.md'
    if index.is_file():
        for line in index.read_text(encoding='utf-8').splitlines():
            if line.startswith('- port_results/geometry/'):
                name = line[2:].strip()
                relative = Path(name)
                if '..' not in relative.parts and relative.name == CACHE_BASENAME:
                    candidates.append(work/relative)
    # Old compatibility symlink and organized path can point to the same cache.
    found = {p.resolve() for p in candidates if (p/'cache_manifest.json').is_file()}
    if len(found) != 1:
        raise ValueError('Expected one existing cache; consult '+str(index)+
                         ' and provide --cache-dir. Found: '+str(sorted(str(p) for p in found)))
    return found.pop()


def checked_cache(directory, identity):
    path = directory/'cache_manifest.json'
    if sha(path) != CACHE_SHA:
        raise ValueError('Fixed ROI cache manifest SHA differs')
    cache = read(path)
    if (cache['protocol'] != 'port_geometry_midpoint_formal_v1' or
            cache['identity'] != identity['training_identity']['parent_identity'] or
            cache['status'] != 'COMPLETE_FROZEN_B_TRAIN_VAL_CACHE' or
            cache['record_counts'] != {k: v[0] for k, v in SHARDS.items()} or
            set(cache['files']) != {k+'.pt' for k in SHARDS} or
            cache['feature_extractions'] != 6003 or cache['native_head_calls'] != 18009 or
            cache['detector_updates'] != 0 or cache['test_access'] is not False or
            not 0 < cache['tensor_bytes'] <= 1024**3):
        raise ValueError('Sealed TRAIN/VAL cache contract differs')
    return dict(cache, manifest_sha256=CACHE_SHA)


def checked_selection(head_dir, identity):
    index = read(head_dir/'artifacts.json')
    names = ('selection.json', FIXED['path'], 'val_epoch_03.rows.jsonl')
    if index['protocol'] != 'port_geometry_midpoint_sigma_v1':
        raise ValueError('Sigma arm artifact protocol differs')
    for name in names:
        if sha(head_dir/name) != index['files'].get(name):
            raise ValueError('Indexed selected artifact SHA differs: '+name)
    selection = read(head_dir/'selection.json')
    if (selection['protocol'] != 'port_geometry_midpoint_sigma_v1' or
            selection['sigma_cells'] != 1.5 or selection['split'] != 'val' or
            selection['selected_checkpoint'] != FIXED or sha(head_dir/FIXED['path']) != FIXED['sha256'] or
            selection['cache_manifest_sha256'] != CACHE_SHA or
            selection['proof']['identity'] != identity['training_identity'] or
            selection['proof']['cache_manifest_sha256'] != CACHE_SHA or
            any(selection[k] is not False for k in ('test_access', 'selection_on_test', 'automatic_promotion'))):
        raise ValueError('Fixed sigma1.5/epoch03 selection identity differs')
    references = {}
    with (head_dir/'val_epoch_03.rows.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
            image = row['image']
            if image in references or row['sequence'] not in VAL_COUNTS or row['scale'] != 1.:
                raise ValueError('Selected VAL rows contain duplicate/wrong view or split')
            references[image] = row
    if Counter(r['sequence'] for r in references.values()) != Counter(VAL_COUNTS):
        raise ValueError('Selected epoch03 full-VAL counts differ')
    return selection, references, {name: index['files'][name] for name in names}


def torch_load(torch, path):
    # Inputs have already passed sealed byte SHA checks. Old torch1.8 has no
    # weights_only argument; recent torch defaults otherwise reject this cache.
    kwargs = dict(map_location='cpu')
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = False
    return torch.load(str(path), **kwargs)


def state_digest(head):
    def tensor_sha(value):
        value = value.detach().cpu().contiguous()
        h = hashlib.sha256(str((str(value.dtype), list(value.shape))).encode())
        h.update(value.numpy().tobytes())
        return h.hexdigest()
    def merged(values):
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
    params = {n: tensor_sha(v) for n, v in head.named_parameters()}
    buffers = {n: tensor_sha(v) for n, v in head.named_buffers()}
    return dict(parameters=merged(params), buffers=merged(buffers),
        parameter_count=sum(v.numel() for v in head.parameters()), buffer_count=len(buffers))


def load_head(torch, model, directory, selection, cache, device):
    payload = torch_load(torch, directory/FIXED['path'])
    context = {k: cache[k] for k in ('manifest_sha256', 'detector_state', 'runtime', 'data_identity')}
    if (payload['protocol'] != 'port_geometry_midpoint_sigma_v1' or
            payload['proof'] != selection['proof'] or payload['context'] != context or
            payload['frozen_b'] != cache['identity']['frozen_b'] or
            any(payload[k] != FIXED[k] for k in ('sigma_cells', 'epoch', 'updates', 'head_digest'))):
        raise ValueError('Checkpoint source/context or fixed head identity differs')
    head = model.SigmaMidpointHead(1.5).to(device)
    head.load_state_dict(payload['head_state'], strict=True)
    head.eval()
    for param in head.parameters():
        param.requires_grad_(False)
    if state_digest(head) != FIXED['head_digest']:
        raise ValueError('Reloaded fixed head digest differs')
    return head


def checked_record(row, label, torch, original):
    _, role, scale = SHARDS[label]
    counts = TRAIN_COUNTS if role == 'train' else VAL_COUNTS
    seq = row['sequence']
    if (seq not in counts or row['role'] != role or row['scale'] != scale or
            row['domain'] != seq.split('_')[0] or
            row['split'] != ('val' if role == 'val' else 'train_sim' if row['domain'] == 'sim' else 'train') or
            row['image'] != seq+'_'+str(row['frame_id']).zfill(5)):
        raise ValueError('Cached TRAIN/VAL identity or view differs')
    tensors = [row[k] for k in TENSOR_KEYS]
    if any(not isinstance(t, torch.Tensor) or t.device.type != 'cpu' or
           t.dtype != torch.float32 or t.requires_grad or t.grad_fn is not None or
           not bool(torch.isfinite(t).all()) for t in tensors):
        raise ValueError('Expected finite detached float32 CPU cache tensors')
    n = len(row['boxes_original'])
    shapes = ((n, 256, 9, 9), (n, 1, 9, 9), (n, 6), (n, 5), (n, 2), (1, 5))
    if n not in (0, 1) or any(tuple(t.shape) != shape for t, shape in zip(tensors, shapes)):
        raise ValueError('Cache tensor shapes differ')
    original.frame(row['boxes_original'], row['boxes_model'], row['scale_xy'])
    center = float((row['boxes_original'][0, :2]-row['gt_original'][0, :2]).norm()) if n else None
    if row['eligible'] is not (role == 'train' and center is not None and center < 15.):
        raise ValueError('Inherited TRAIN eligibility differs')
    return sum(v.numel()*v.element_size() for v in row.values() if isinstance(v, torch.Tensor))


def tensor_witness(torch, original, b, bm, xy, m, pair, goal, frame):
    """Recheck an OFFLINE witness using the unchanged native geometry functions."""
    requested = m.new_tensor(pair).reshape(1, 2)
    delta = (requested/m[:, 2:4]).log()
    c = m.clone()
    c[:, 2:4] = m[:, 2:4]*delta.exp()
    realized = c[0, 2:4].cpu().tolist()
    finite = bool(torch.isfinite(c).all()) and bool(torch.isfinite(delta).all())
    positive = bool((c[:, 2:4] > 0).all())
    if not finite or not positive:
        return dict(passed=False, checks=dict(finite=finite, positive=positive))
    nominal = geometry.check_sizes(realized, b[0].cpu().tolist(), m[0].cpu().tolist(), goal, frame)
    points = original.box_midpoints(c[:, :5])
    local = original.original_to_roi(points, b, bm, xy)
    ratios = c[:, 2:4]/b[:, 2:4]
    checks = dict(finite=finite, positive=bool((c[:, 2:4] > original.SETTINGS['min_pair_length_px']).all()),
        nominal_b_edge_bound=nominal['checks']['nominal_b_edge_bound'],
        joint10=nominal['checks']['joint10'], raw_order=nominal['checks']['raw_order'],
        canonical_direction=bool(torch.equal(c[:, 2] < c[:, 3], m[:, 2] < m[:, 3])),
        center_angle_score_exact=bool(torch.equal(c[:, [0, 1, 4, 5]], m[:, [0, 1, 4, 5]])),
        roi_midpoints=bool((local.abs() <= .5+original.SETTINGS['roi_bound_tolerance']).all()),
        inherited_b_edge_guard=bool(((ratios >= .8-1e-6) & (ratios <= 1.25+1e-6)).all()),
        inherited_center_guard=bool(((c[:, :2]-b[:, :2]).norm(dim=1) <=
            .30*torch.minimum(b[:, 2], b[:, 3])+1e-6).all()),
        inherited_angle_guard=bool((original.wrap_pi(c[:, 4]-b[:, 4]).abs() <= math.pi/18+1e-6).all()))
    return dict(passed=all(checks.values()), checks=checks, delta_raw=delta[0].cpu().tolist(),
        realized_raw_width_height=realized, size_errors=nominal['size_errors'],
        points_roi=local[0].cpu().tolist(), roi_min_margin=.5+1e-5-float(local.abs().max()),
        dtype=str(m.dtype), device=str(m.device))


def native_source_guards(torch, original, b, m):
    ratios = m[:, 2:4]/b[:, 2:4]
    settings = original.SETTINGS
    return dict(center_bound=bool(((m[:, :2]-b[:, :2]).norm(dim=1) <=
        settings['max_center_over_b_short']*torch.minimum(b[:, 2], b[:, 3])+1e-6).all()),
        angle_bound=bool((original.wrap_pi(m[:, 4]-b[:, 4]).abs() <= settings['max_angle_residual_rad']+1e-6).all()),
        edge_bound=bool(((ratios >= math.exp(-settings['max_log_edge_residual'])-1e-6) &
                        (ratios <= math.exp(settings['max_log_edge_residual'])+1e-6)).all()))


def diagnose_record(row, head, torch, original, device):
    roi, support, b, bm, xy = [row[k].to(device) for k in TENSOR_KEYS[:-1]]
    out = head(roi, support, b, bm, xy)
    m = out['boxes_original']
    if (tuple(m.shape) != tuple(b.shape) or m.dtype != b.dtype or m.device != b.device or
            not bool(torch.isfinite(m).all()) or m.requires_grad or
            not torch.equal(m[:, 5], b[:, 5])):
        raise ValueError('Frozen midpoint output identity/score contract differs')
    gt = row['gt_original'][0].tolist()
    base = b[0].cpu().tolist() if len(b) else None
    pred = m[0].cpu().tolist() if len(b) else None
    diag = geometry.diagnose_frame(None, None, gt) if not len(b) else None
    if len(b):
        _, _, _, sides, rotation = original.frame(b, bm, xy)
        runtime_frame = dict(context_sides_model=sides[0].cpu().tolist(), rotation_model=rotation[0].cpu().tolist())
        frame = geometry.make_frame(base, bm[0].cpu().tolist(), xy[0].cpu().tolist(), runtime_frame)
        target = original.target_points(row['gt_original'].to(device), b, bm, xy)
        p = target['original'][0]
        u, v = (p[0]-p[2]).norm(), (p[1]-p[3]).norm()
        matched = torch.stack((v, u) if base[2] < base[3] else (u, v)).cpu().tolist()
        diag = geometry.diagnose_frame(base, pred, gt, bm[0].cpu().tolist(), xy[0].cpu().tolist(),
            runtime_frame=runtime_frame, matched_gt_raw=matched,
            source_guard_checks=native_source_guards(torch, original, b, m),
            verify_witness=lambda pair, goal: tensor_witness(torch, original, b, bm, xy, m, pair, goal, frame))
        diag['matched_training_target'].update(permutation=int(target['permutation'][0]),
            roi_axis_width_height=list(reversed(matched)) if frame['model_b_swap'] else matched,
            original_model_b_swap_differ=frame['original_b_swap'] != frame['model_b_swap'])
    return dict(image=row['image'], sequence=row['sequence'], domain=row['domain'],
        frame_id=row['frame_id'], split=row['split'], role=row['role'], scale=row['scale'],
        eligible_train=row['eligible'], gt=gt, b=base, midpoint=pred, diagnosis=diag)


def compare_reference(row, ref):
    for key in ('image', 'sequence', 'domain', 'frame_id', 'scale'):
        if row[key] != ref[key]:
            raise ValueError('Selected VAL frame identity differs: '+row['image'])
    for key in ('gt', 'b', 'midpoint'):
        a, b = row[key], ref[key]
        if (a is None) != (b is None) or (a is not None and (len(a) != len(b) or
                any(not math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-4) for x, y in zip(a, b)))):
            raise ValueError('Fixed cached VAL replay differs: '+row['image']+' '+key)
    if row['midpoint'] is not None:
        a, b = row['midpoint'], ref['midpoint']
        if a[5] != b[5] or (a[2] < a[3]) != (b[2] < b[3]):
            raise ValueError('Fixed VAL score or canonical axis differs')
        if geometry.joint_correct(b[2:4], geometry.goal_edges(ref['gt'], b)) != row['diagnosis']['baseline_joint10']:
            raise ValueError('Fixed VAL joint10 count changed at numeric boundary')
        if (math.hypot(b[0]-ref['gt'][0], b[1]-ref['gt'][1]) < 15.) != row['diagnosis']['baseline_center_hit']:
            raise ValueError('Fixed VAL center count changed at numeric boundary')


def sample_roles(root=ROOT):
    items = read(root/SAMPLE_NAME)['samples']
    result = {r['image']: r for r in items}
    if len(items) != 64 or len(result) != 64 or Counter((r['domain'], r['role']) for r in items) != Counter(
            {('real', 'fit'): 24, ('real', 'probe'): 8, ('sim', 'fit'): 24, ('sim', 'probe'): 8}):
        raise ValueError('Fixed64 TRAIN fit/probe roles differ')
    return result


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return None
    pos = (len(ordered)-1)*q
    low, high = math.floor(pos), math.ceil(pos)
    return ordered[low]+(ordered[high]-ordered[low])*(pos-low)


def summarize(rows):
    ds = [r['diagnosis'] for r in rows]
    outputs = [d for d in ds if d['output']]
    n, out = len(ds), len(outputs)
    count = lambda key: sum(bool(d.get(key)) for d in outputs)
    def rate(num, den):
        return dict(numerator=num, denominator=den, fraction=num/den if den else None)
    center, joint = count('baseline_center_hit'), count('baseline_joint10')
    statuses = Counter(d['constraint_status'] for d in ds)
    known = sum(d['delivered_size_reachable'] is True for d in ds)
    unknown = sum(d['delivered_size_reachable'] is None for d in ds)
    result = dict(frames=n, outputs=out, no_output=n-out,
        output_coverage=rate(out, n), center_correct_conditional=rate(center, out),
        center_correct_full_frame=rate(center, n), joint_size10_conditional=rate(joint, out),
        joint_size10_full_frame=rate(joint, n),
        center_and_joint_size10_full_frame=rate(count('baseline_center_and_joint10'), n),
        constraint_status_counts=dict(statuses),
        nominal_feasible_counts={k: sum(d['stages'][k] for d in outputs)
            for k in ('b_bounds_only', 'b_bounds_and_order', 'b_roi_and_order')},
        baseline_bypass_reachable=count('baseline_bypass_reachable'),
        exact_square_bypass=count('exact_square_bypass'),
        exact_gt_inside_b=count('exact_sorted_gt_size_inside_b_bound'),
        exact_gt_passes_all_nominal_constraints=count('exact_sorted_gt_size_passes_all_constraints'),
        exact_gt_outside_b_but_nominal_joint10_feasible=sum(not d['exact_sorted_gt_size_inside_b_bound']
            and d['stages']['b_roi_and_order'] for d in outputs),
        baseline_rebuilt_roi_outside=sum(not d['baseline_rebuilt_midpoints_pass_nominal_constraints']['checks']['roi_midpoints'] for d in outputs),
        verified_additional_joint10_frames=sum(not d['baseline_joint10'] and
            d['constraint_status'] == 'runtime_witness_verified' for d in outputs),
        delivered_reachable_known=rate(known, n), delivered_reachable_unresolved=unknown,
        delivered_reachable_upper_including_unresolved=rate(known+unknown, n),
        numeric_warning_frames=sum(d['full_intersection']['numeric_boundary'] for d in outputs),
        source_guard_numeric_disagreement_frames=sum(bool(d['source_guards']['numeric_disagreements']) for d in outputs),
        failed_nominal_constraint_counts=dict(Counter(k for d in outputs for k in d['full_intersection']['failed_constraints'])),
        matched_training_targets=sum('matched_training_target' in d for d in outputs),
        matched_target_exact_outside_b=sum(not d['matched_training_target']['inside_b_edge_bound'] for d in outputs if 'matched_training_target' in d),
        matched_target_m_order_conflicts=sum(not d['matched_training_target']['keeps_m_raw_order'] for d in outputs if 'matched_training_target' in d),
        matched_target_axis_disagreements=sum(not d['matched_training_target']['agrees_with_metric_axis_target'] for d in outputs if 'matched_training_target' in d),
        original_model_swap_disagreements=sum(d['matched_training_target']['original_model_b_swap_differ'] for d in outputs if 'matched_training_target' in d))
    result['baseline_size'] = {}
    for index, name in ((0, 'long'), (1, 'short')):
        # raw errors are rearranged using frozen M long-axis identity only.
        values = [d['baseline_size_errors'][1-index if d['m_raw_swap'] else index] for d in outputs]
        signed = [math.log(max(r['midpoint'][2:4]) / max(r['gt'][2:4]) if index == 0 else
                           min(r['midpoint'][2:4]) / min(r['gt'][2:4])) for r in rows if r['diagnosis']['output']]
        result['baseline_size'][name] = dict(n=out, mae=sum(values)/out if out else None,
            p95=percentile(values, .95), maximum=max(values) if values else None,
            signed_log_mean=sum(signed)/out if out else None,
            oversized=sum(v > 0 for v in signed), undersized=sum(v < 0 for v in signed))
    return result


def grouped(rows):
    groups = dict(overall=rows)
    groups.update({d: [r for r in rows if r['domain'] == d] for d in ('real', 'sim')})
    groups.update({s: [r for r in rows if r['sequence'] == s] for s in sorted({r['sequence'] for r in rows})})
    return {k: summarize(v) for k, v in groups.items()}


def shard_summary(rows):
    subsets = dict(all=rows)
    if rows[0]['role'] == 'train':
        subsets['eligible_train'] = [r for r in rows if r['eligible_train']]
        for role in ('fit', 'probe'):
            subsets['fixed_'+role] = [r for r in rows if r['sample_role'] == role]
            subsets['fixed_'+role+'_eligible'] = [r for r in subsets['fixed_'+role] if r['eligible_train']]
    return {k: grouped(v) for k, v in subsets.items()}


def publish(directory, completion):
    write(directory/'completion.json', completion)
    write(directory/'artifacts.json', dict(protocol=VERSION,
        files={p.name: sha(p) for p in directory.iterdir() if p.is_file() and p.name != 'artifacts.json'}))


def run(args):
    if args.stage == 'check':
        identity = checked_sources()
        sample_roles()
        print(json.dumps(dict(status='EDGE_FEASIBILITY_SOURCE_CHECK_PASS',
            protocol_sha256=identity['protocol_sha256'], sources_sha256=identity['sources_sha256'],
            checked_source_files=len(identity['sources']), fixed_train_sample_images=64,
            scope=protocol_document()['scope']), sort_keys=True))
        return
    if not args.out_dir:
        raise ValueError('--out-dir is required for diagnose (fresh directory only)')
    directory = Path(args.out_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    completion = dict(protocol=VERSION, status='EDGE_FEASIBILITY_FAILED',
        started_utc=datetime.now(timezone.utc).isoformat(), **protocol_document()['scope'])
    try:
        identity = checked_sources()
        write(directory/'protocol.json', protocol_document())
        write(directory/'source_identity.json', identity)
        cache_dir = resolve_cache(args.cache_dir)
        cache = checked_cache(cache_dir, identity)
        head_dir = Path(args.head_dir).resolve()
        selection, references, indexed = checked_selection(head_dir, identity)
        completion['inputs'] = dict(cache_dir=str(cache_dir), cache_manifest_sha256=CACHE_SHA,
            head_dir=str(head_dir), selected_checkpoint=FIXED, indexed_selected_artifacts=indexed)
        import torch
        from crane_project.utils import port_geometry_midpoint_sigma_v1 as model
        original = model.original
        device = torch.device(args.device)
        if device.type not in ('cpu', 'cuda') or (device.type == 'cuda' and not torch.cuda.is_available()):
            raise ValueError('Requested CPU/CUDA device is unavailable')
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        head = load_head(torch, model, head_dir, selection, cache, device)
        before = state_digest(head)
        roles, found, summaries, total_bytes = sample_roles(), {}, {}, 0
        with (directory/'progress.jsonl').open('x', encoding='utf-8') as progress, \
                (directory/'rows.jsonl').open('x', encoding='utf-8') as output, torch.no_grad():
            for label, (expected, role, scale) in SHARDS.items():
                path = cache_dir/(label+'.pt')
                if sha(path) != cache['files'][path.name]:
                    raise ValueError('Sealed ROI shard SHA differs: '+path.name)
                payload = torch_load(torch, path)
                if payload['identity'] != cache['identity'] or payload['detector_state'] != cache['detector_state']:
                    raise ValueError('Shard source or detector identity differs')
                records = payload['records']
                if len(records) != expected or Counter(r['sequence'] for r in records) != Counter(TRAIN_COUNTS if role == 'train' else VAL_COUNTS):
                    raise ValueError('Full TRAIN/VAL sequence counts differ: '+label)
                if len({r['image'] for r in records}) != expected:
                    raise ValueError('Duplicate cache frame: '+label)
                rows, found[label] = [], set()
                for index, record in enumerate(records):
                    total_bytes += checked_record(record, label, torch, original)
                    row = diagnose_record(record, head, torch, original, device)
                    row['shard'] = label
                    row['sample_role'] = 'val' if role == 'val' else 'other_train'
                    if role == 'train' and row['image'] in roles:
                        sample = roles[row['image']]
                        for key in ('image', 'domain', 'sequence', 'split', 'image_sha256', 'annotation_sha256'):
                            if record[key] != sample[key]:
                                raise ValueError('Fixed TRAIN sample source differs: '+row['image'])
                        row['sample_role'] = sample['role']
                        found[label].add(row['image'])
                    if role == 'val':
                        compare_reference(row, references[row['image']])
                    output.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                    rows.append(row)
                    if index % 100 == 0 or index+1 == expected:
                        event = dict(shard=label, done=index+1, total=expected)
                        progress.write(json.dumps(event)+'\n'); progress.flush(); output.flush()
                        print(json.dumps(event), flush=True)
                if role == 'train' and found[label] != set(roles):
                    raise ValueError('Missing fixed64 TRAIN sample identity/view: '+label)
                support = dict(total=expected, eligible_for_training=sum(r['eligible_train'] for r in rows),
                    no_output=sum(not r['diagnosis']['output'] for r in rows),
                    center_mismatch=sum(role == 'train' and r['diagnosis']['output'] and not r['eligible_train'] for r in rows))
                if support != cache['support'][label]:
                    raise ValueError('Sealed shard support counts differ: '+label)
                summaries[label] = shard_summary(rows)
                del payload, records, rows, record
        if total_bytes != cache['tensor_bytes']:
            raise ValueError('Total cache tensor bytes differ')
        if state_digest(head) != before or any(p.requires_grad or p.grad is not None for p in head.parameters()):
            raise ValueError('Frozen head changed or acquired gradients')
        write(directory/'summary.json', dict(protocol=VERSION, shards=summaries,
            note='OFFLINE GT feasibility, not corrected predictions/learning gains; constraints do not certify RIoU or temporal protection.'))
        completion.update(status='EDGE_FEASIBILITY_DIAGNOSIS_COMPLETE_REVIEW_REQUIRED', frames=6003,
            shard_frames={k: v[0] for k, v in SHARDS.items()}, selected_val_replay_frames=887,
            head_before=before, head_after=state_digest(head), frozen_head_no_grad=True,
            dtype='torch.float32', device=str(device), torch_version=torch.__version__,
            cuda_version=torch.version.cuda, tensor_bytes=total_bytes,
            elapsed_seconds=time.monotonic()-start,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None)
    except Exception as error:
        completion.update(error_type=type(error).__name__, error=str(error), elapsed_seconds=time.monotonic()-start)
        publish(directory, completion)
        raise
    publish(directory, completion)
    print(completion['status']+' '+str(directory), flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--stage', required=True, choices=('check', 'diagnose'))
    result.add_argument('--head-dir', default=str(ROOT/DEFAULT_HEAD))
    result.add_argument('--cache-dir', help='Existing sealed cache; default resolves old/organized path and INDEX.md')
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--out-dir', help='New result directory; overwrite/resume is forbidden')
    return result


if __name__ == '__main__':
    run(parser().parse_args())
