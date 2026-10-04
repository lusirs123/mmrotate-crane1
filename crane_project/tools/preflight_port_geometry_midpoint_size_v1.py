#!/usr/bin/env python3
"""Finite TRAIN point-only vs point+decoded-size comparison, no B inference.

Reuse only formal TRAIN CPU cache shards; do not load VAL/TEST tensors or any
checkpoint. 128 TRAIN identities/256 views, 200 updates per arm, identical
initialization/batches, no fitted-weight export or automatic formal approval.
Static mode uses the standard library only, without tensor/cache/GPU loads.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
VERSION = 'port_geometry_midpoint_size_v1_train_preflight'
PROTOCOL = ROOT/'crane_project/tools/port_geometry_midpoint_size_v1_protocol.json'
SOURCES = ROOT/'crane_project/tools/port_geometry_midpoint_size_v1_sources.json'
PARENT = ROOT/'crane_project/tools/port_geometry_midpoint_formal_v1_sources.json'
SETTINGS = dict(seed=1703, schedule_seed=1705, steps_per_arm=200, batch_size=8,
    lr=.001, weight_decay=0., clip_norm=10., size_weight=.025, size_beta=.1,
    block_frames=8, fit_blocks_per_domain=6, probe_blocks_per_domain=2,
    min_interblock_gap_frames=16, scales=[1., .5])
TOL = dict(center_px=1e-6, edge_fraction=1e-6, angle_deg=1e-5,
           riou=1e-5, motion=1e-8, aci=1e-8)
ARMS = ('point_only', 'point_size')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError('Nonfinite JSON: '+value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def write_json(path, value):
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    with Path(path).open('x') as stream:
        stream.write(encoded)


def protocol_document():
    return dict(protocol=VERSION, settings=deepcopy(SETTINGS), tolerances=deepcopy(TOL),
        conditions=list(ARMS), parameter_count=17696,
        objective='Unchanged point SmoothL1 beta.1 + fixed .025 times mean two matched-axis SmoothL1(log(projected_edge/GT_edge), beta.1). Control uses point loss only.',
        projection='Continuous original-coordinate v1 rectangle projection BEFORE zero-identity replacement/fallback. GT axes from existing fixed cyclic midpoint matching; no pred/GT edge sorting, clamping or dropping invalid edges.',
        online='Use original SpatialMidpointHead, zero-output initialization, sampler, expectation calibration, raw w/h-angle restoration and full-B fallback unchanged. No temporal module, smoothing, teacher, extra outputs or classification objective.',
        cache='Exact completed formal CPU cache identity/SHA. Load train_s1.pt and train_s05.pt only. Do not instantiate B, extract features, load weights, VAL or TEST tensors.',
        sampling='Metadata only: sorted sequence round-robin, earliest consecutive8-frame block with>=16 intervening frame IDs from every selected block of that sequence. First6 blocks/domain fit, next2 probe. Same identity role at1.0/.5. No geometry/VAL/TEST-driven resampling.',
        eligibility='Only inherited TRAIN B-present center<15px views enter fit updates. All selected probe views evaluated, including missing/wrong-center outputs. Failed eligibility never causes block reselection.',
        schedule='Seed1705 per-domain shuffled cyclic fit-view queues,4real+4sim,200 steps/arm. Identical seed1703 initial head and minibatches. Both arms are newly initialized, not formal/short-fit checkpoint continuations.',
        review='Probe size/static/GT-relative motion/center/angle/RIoU/coverage jointly checked versus point-only; center/RIoU/sim angle also checked versus B. DFR reported, not a substitute for GT-relative motion. Strict short-MAE gain in each domain across scales required. No step/coefficient/seed sweep.',
        scope=dict(train_identities=128, train_views=256, fit_identities=96,
            probe_identities=32, head_updates_total=400, detector_updates=0,
            detector_forward_calls=0, feature_extractions=0, val_access=False,
            test_access=False, weight_bytes_read=False, checkpoint_exported=False,
            formal_training_approved=False, automatic_promotion=False),
        limitations=['TRAIN probe is excluded from these head updates, but B and earlier full-TRAIN head have seen TRAIN. Not independent VAL/generalization evidence.',
            'Short disjoint segments give limited temporal support, not full-video performance or significance.',
            'Size constraint does not directly supervise center, angle or time; shared head gradients can change them.',
            'TEST repeatedly exposed. Do not tune/reselect using TEST. Unchanged depth interface does not certify depth accuracy.'])


def checked_contract(training_dir, cache_dir):
    protocol = protocol_document()
    if read_json(PROTOCOL) != protocol:
        raise ValueError('Fixed size preflight protocol differs')
    manifest = read_json(SOURCES)
    parent = read_json(PARENT)
    required = {str(p.relative_to(ROOT)) for p in (Path(__file__), PROTOCOL,
        ROOT/'crane_project/utils/port_geometry_midpoint_size_v1.py',
        ROOT/'tests/test_port_geometry_midpoint_size_v1.py', PARENT,
        ROOT/'crane_project/tools/analyze_port_geometry_midpoint_size_temporal_v1.py')}
    if (manifest['protocol'] != VERSION or manifest['parent_sources_sha256'] != sha(PARENT) or
            set(manifest['sources']) != set(parent['sources']) | required):
        raise ValueError('Size preflight source scope differs')
    for name, digest in manifest['sources'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Source SHA differs: '+name)
    training_dir, cache_dir = Path(training_dir), Path(cache_dir)
    complete = read_json(training_dir/'completion.json')
    artifacts = read_json(training_dir/'artifacts.json')['files']
    cache = read_json(cache_dir/'cache_manifest.json')
    identity = complete['identity']
    frozen = read_json(ROOT/'crane_project/tools/port_geometry_midpoint_formal_v1_protocol.json')['frozen_b']
    if (artifacts.get('completion.json') != sha(training_dir/'completion.json') or
            complete['status'] != 'FORMAL_MIDPOINT_TRAIN_VAL_COMPLETE_REVIEW_REQUIRED' or
            complete['epochs_completed'] != 24 or complete['detector_updates'] != 0 or
            complete['test_access'] is not False or identity['sources'] != parent['sources'] or
            identity['sources_sha256'] != sha(PARENT) or identity['frozen_b'] != frozen or
            identity['protocol_sha256'] != sha(ROOT/'crane_project/tools/port_geometry_midpoint_formal_v1_protocol.json') or
            complete['cache_manifest_sha256'] != sha(cache_dir/'cache_manifest.json') or
            cache['identity'] != identity or cache['status'] != 'COMPLETE_FROZEN_B_TRAIN_VAL_CACHE' or
            cache['record_counts'] != dict(train_s1=2558, train_s05=2558, val_s1=887) or
            set(cache['files']) != {'train_s1.pt', 'train_s05.pt', 'val_s1.pt'} or
            cache['feature_extractions'] != 6003 or cache['native_head_calls'] != 18009 or
            cache['detector_updates'] != 0 or cache['test_access'] is not False):
        raise ValueError('Completed formal/cache provenance differs')
    proof = dict(parent_sources_sha256=sha(PARENT), sources_sha256=sha(SOURCES),
        protocol_sha256=sha(PROTOCOL), training_completion_sha256=sha(training_dir/'completion.json'),
        cache_manifest_sha256=sha(cache_dir/'cache_manifest.json'), identity=identity,
        input_tensor_files=['train_s1.pt', 'train_s05.pt'], val_tensor_loaded=False,
        test_access=False, weight_bytes_read=False)
    return protocol, cache, proof


def runtime_modules():
    import torch
    from crane_project.tools import train_port_geometry_midpoint_formal_v1 as formal
    from crane_project.utils import port_geometry_midpoint_size_v1 as size
    return torch, formal, size


def execution_device(gpu, formal, torch):
    return torch.device('cuda', gpu), formal.runtime(gpu)


def load_train_cache(cache_dir, cache, formal, torch, on_load=None):
    records = []
    for name, scale in (('train_s1.pt', 1.), ('train_s05.pt', .5)):
        path = Path(cache_dir)/name
        if sha(path) != cache['files'][name]:
            raise ValueError('TRAIN cache shard SHA differs: '+name)
        payload = torch.load(str(path), map_location='cpu')
        if on_load is not None:
            on_load(name)
        if (payload['identity'] != cache['identity'] or
                payload['detector_state'] != cache['detector_state'] or len(payload['records']) != 2558):
            raise ValueError('TRAIN shard identity/state/count differs')
        rows = payload['records']
        if Counter(r['sequence'] for r in rows) != Counter(formal.g.ready.TRAIN_COUNTS):
            raise ValueError('Full TRAIN sequence counts differ')
        for r in rows:
            if (r['role'] != 'train' or r['scale'] != scale or
                    r['split'] != formal.g.ready.TRAIN_SPLITS[r['sequence']] or
                    r['domain'] != r['sequence'].split('_')[0] or
                    r['image'] != r['sequence']+'_'+str(r['frame_id']).zfill(5)):
                raise ValueError('TRAIN role/scale/identity differs')
            values = [r[k] for k in ('roi', 'support', 'boxes_original', 'boxes_model', 'scale_xy', 'gt_original')]
            if any(v.device.type != 'cpu' or v.requires_grad or v.grad_fn is not None or
                   v.dtype != torch.float32 or not bool(torch.isfinite(v).all()) for v in values):
                raise ValueError('Expected finite detached float32 CPU tensors')
            n = len(r['boxes_original'])
            if (n > 1 or r['roi'].shape != (n,256,9,9) or r['support'].shape != (n,1,9,9) or
                    r['gt_original'].shape != (1,5) or min(r['gt_original'][0,2:4]) <= 0 or
                    bool(((r['support'] < -1e-6) | (r['support'] > 1+1e-6)).any())):
                raise ValueError('TRAIN tensor shape/value differs')
            formal.m.frame(r['boxes_original'], r['boxes_model'], r['scale_xy'])
            center = float((r['boxes_original'][0,:2]-r['gt_original'][0,:2]).norm()) if n else None
            if r['eligible'] != (center is not None and center < 15.):
                raise ValueError('Inherited TRAIN eligibility differs')
        records.extend(rows)
    if len({(r['image'], r['scale']) for r in records}) != 5116:
        raise ValueError('Duplicate TRAIN identity/view')
    if formal.tensor_bytes(records) > formal.SETTINGS['max_cpu_tensor_bytes']:
        raise ValueError('TRAIN CPU tensor budget exceeded')
    return records


def select_blocks(records):
    """Only metadata determines blocks; both scales share roles."""
    indexed = {(r['image'], r['scale']): r for r in records}
    names = {r['image'] for r in records}
    if len(indexed) != len(records) or any((n, s) not in indexed for n in names for s in SETTINGS['scales']):
        raise ValueError('Missing/duplicate paired TRAIN view')
    metadata = [indexed[(n, 1.)] for n in sorted(names)]
    for r in metadata:
        if (r['role'] != 'train' or r['domain'] not in ('real','sim') or
                not isinstance(r['frame_id'], int) or isinstance(r['frame_id'], bool) or
                r['sequence'].split('_')[0] != r['domain'] or
                r['sequence'] not in ('real_seq01','real_seq05','real_seq06','real_seq12','real_seq13','sim_seq08') or
                r['image'] != r['sequence']+'_'+str(r['frame_id']).zfill(5)):
            raise ValueError('Fixed TRAIN metadata only; VAL/TEST prohibited')
        for scale in SETTINGS['scales']:
            other = indexed[(r['image'], scale)]
            if any(other[k] != r[k] for k in ('role','domain','sequence','frame_id','image')):
                raise ValueError('Cross-scale TRAIN identity differs')
    assignments, selected = [], {}
    length, gap = SETTINGS['block_frames'], SETTINGS['min_interblock_gap_frames']
    for domain in ('real', 'sim'):
        sequences = sorted({r['sequence'] for r in metadata if r['domain'] == domain})
        if not sequences:
            raise ValueError('No TRAIN sequence in domain')
        previous = {s: [] for s in sequences}
        for number in range(SETTINGS['fit_blocks_per_domain']+SETTINGS['probe_blocks_per_domain']):
            sequence = sequences[number % len(sequences)]
            rows = sorted((r for r in metadata if r['sequence'] == sequence), key=lambda r:r['frame_id'])
            block = None
            for start in range(len(rows)-length+1):
                candidate = rows[start:start+length]
                lo, hi = candidate[0]['frame_id'], candidate[-1]['frame_id']
                if hi-lo != length-1 or any(not (hi+gap < a or lo > b+gap) for a,b in previous[sequence]):
                    continue
                block = candidate
                break
            if block is None:
                raise ValueError('Insufficient fixed consecutive TRAIN blocks: '+sequence)
            previous[sequence].append((block[0]['frame_id'], block[-1]['frame_id']))
            role = 'fit' if number < SETTINGS['fit_blocks_per_domain'] else 'probe'
            assignments.append(dict(domain=domain, sequence=sequence, role=role,
                frame_ids=[r['frame_id'] for r in block], images=[r['image'] for r in block]))
            for r in block:
                if r['image'] in selected:
                    raise ValueError('Fit/probe overlap')
                selected[r['image']] = role
    prepared = [dict(indexed[(n,s)], role=selected[n]) for s in SETTINGS['scales'] for n in sorted(selected)]
    return prepared, assignments


def schedule(records):
    pools = {d:[i for i,r in enumerate(records) if r['role'] == 'fit' and r['eligible'] and r['domain'] == d]
             for d in ('real','sim')}
    if any(len(p) < 4 for p in pools.values()):
        raise ValueError('Insufficient inherited eligible fit views; do not resample')
    rng = random.Random(SETTINGS['schedule_seed'])
    orders = {d: rng.sample(p, len(p)) for d,p in pools.items()}
    offsets = {d:0 for d in pools}; batches = []
    for _ in range(SETTINGS['steps_per_arm']):
        batch = []
        for d in ('real','sim'):
            for _ in range(4):
                if offsets[d] == len(orders[d]):
                    orders[d] = rng.sample(pools[d], len(pools[d])); offsets[d] = 0
                batch.append(orders[d][offsets[d]]); offsets[d] += 1
        batches.append(batch)
    return batches


def batch(records, indices, device, torch):
    if any(records[i]['role'] != 'fit' or not records[i]['eligible'] for i in indices):
        raise ValueError('Only inherited eligible fit TRAIN enters gradients')
    return [torch.cat([records[i][k] for i in indices]).to(device) for k in
            ('roi','support','boxes_original','boxes_model','scale_xy','gt_original')]


def gradient_report(point, size, params, torch):
    gradients = [torch.autograd.grad(loss, params, retain_graph=True) for loss in (point, size)]
    vectors = [torch.cat([g.detach().double().reshape(-1) for g in gs]) for gs in gradients]
    pn, sn = (float(v.norm()) for v in vectors)
    cosine = float((vectors[0]*vectors[1]).sum())/(pn*sn) if pn and sn else None
    if not all(math.isfinite(v) for v in (pn,sn)) or (cosine is not None and not math.isfinite(cosine)):
        raise ValueError('Nonfinite component gradient')
    return dict(point_norm=pn, size_unit_norm=sn, size_weighted_norm=sn*SETTINGS['size_weight'],
        weighted_size_over_point=sn*SETTINGS['size_weight']/pn if pn else None,
        cosine=cosine)


def evaluate(head, records, device, formal):
    # formal.evaluate performs GT-free forward/count/score/state checks. GT is
    # used only after the forward for metrics, never to accept the candidate.
    source = [dict(r, role='train') for r in records]
    rows = formal.evaluate(head, source, device)
    for row,r in zip(rows,records):
        row['role'] = r['role']
    from crane_project.tools import analyze_port_geometry_midpoint_size_temporal_v1 as residual
    groups = {}
    for role in ('fit','probe'):
        for domain in ('real','sim'):
            for scale in SETTINGS['scales']:
                subset = [r for r in rows if r['role'] == role and r['domain'] == domain and r['scale'] == scale]
                frames = [residual.static_frame(r) for r in subset]
                pairs, support = residual.temporal_pairs(frames)
                groups[role+'/'+domain+'/'+str(scale)] = dict(residual.group_summary(frames,pairs), temporal_support=support)
    return dict(rows=rows, groups=groups)


def fit_arm(records, batches, initial, arm, device, formal, size, torch, progress):
    formal.seed_all()
    head = formal.m.SpatialMidpointHead().to(device); head.load_state_dict(initial, strict=True)
    before = formal.g.state_digest(head); start = evaluate(head,records,device,formal)
    if any(r['b'] != r['midpoint'] for r in start['rows']):
        raise ValueError('Zero initialization is not exact B')
    optimizer = torch.optim.Adam(head.parameters(), lr=SETTINGS['lr'], weight_decay=SETTINGS['weight_decay'])
    logs = []
    for step,indices in enumerate(batches,1):
        head.train(); optimizer.zero_grad()
        roi,support,b,bm,xy,gt = batch(records,indices,device,torch)
        out = head(roi,support,b,bm,xy)
        targets = formal.m.target_points(gt,b,bm,xy)
        point, _ = formal.m.point_loss(out['points_roi'],targets)
        edge, detail = size.size_loss(out['points_original'],targets)
        loss = point + SETTINGS['size_weight']*edge if arm == 'point_size' else point
        params = list(head.parameters())
        diagnostics = gradient_report(point,edge,params,torch) if step in (1,2,25,100,200) else None
        if diagnostics is not None and diagnostics['size_unit_norm'] <= 0:
            raise ValueError('Decoded size gradient ineffective on fixed TRAIN batch')
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite combined loss')
        loss.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params):
            raise ValueError('Missing/nonfinite head gradient')
        stem = sum(float(p.grad.double().square().sum()) for p in head.stem.parameters())**.5
        if step == 2 and stem <= 0:
            raise ValueError('Stem gradient ineffective after zero-output update')
        norm = float(torch.nn.utils.clip_grad_norm_(params,SETTINGS['clip_norm']))
        after = sum(float(p.grad.double().square().sum()) for p in params)**.5
        if not math.isfinite(norm) or after > SETTINGS['clip_norm']+1e-4:
            raise ValueError('Invalid gradient clipping')
        optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in params):
            raise ValueError('Nonfinite updated parameters')
        log = dict(stage='update', arm=arm, step=step, point_loss=float(point.detach()),
            size_unit_loss=float(edge.detach()), size_weighted_loss=float(edge.detach())*SETTINGS['size_weight'],
            size_applied=arm=='point_size', total_loss=float(loss.detach()),
            matched_axis_size_parts=detail['parts'].detach().cpu().tolist(),
            grad_norm_before_clip=norm, grad_norm_after_clip=after,
            clip_multiplier=min(1.,SETTINGS['clip_norm']/(norm+1e-6)), stem_grad_norm=stem,
            component_gradients=diagnostics)
        logs.append(log); progress(log)
    final = evaluate(head,records,device,formal)
    result = dict(initial_state=before, final_state=formal.g.state_digest(head),
        initial=start, final=final, logs=logs, completed_updates=len(logs),
        checkpoint_exported=False)
    del head,optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def review(point, sized):
    checks = {}; gains = {'real':[], 'sim':[]}
    for key,c in point['groups'].items():
        if not key.startswith('probe/'):
            continue
        n = sized['groups'][key]; domain = key.split('/')[1]
        b, control, added = c['b'], c['midpoint'], n['midpoint']
        conditions = dict(same_output_support=n['output_coverage']==c['output_coverage'],
                          same_pair_support=n['temporal_support']==c['temporal_support'])
        for name,ref in (('point',control),('b',b)):
            conditions[name+'/correct_center_count'] = added['all_frame_center_correct']['numerator'] >= ref['all_frame_center_correct']['numerator']
            conditions[name+'/center_mean'] = added['center_error_px']['mean'] <= ref['center_error_px']['mean']+TOL['center_px']
            conditions[name+'/riou'] = added['all_frame_mean_riou'] >= ref['all_frame_mean_riou']-TOL['riou']
            if domain == 'sim':
                conditions[name+'/pure_angle_rmse'] = added['pure_angle_error_deg']['rmse'] <= ref['pure_angle_error_deg']['rmse']+TOL['angle_deg']
        conditions['point/center_rmse'] = added['center_error_px']['rmse'] <= control['center_error_px']['rmse']+TOL['center_px']
        conditions['point/aci'] = added['aci'] >= control['aci']-TOL['aci']
        for field in ('long','short'):
            conditions[field+'/static_mae'] = added['static'][field]['relative_error']['mae'] <= control['static'][field]['relative_error']['mae']+TOL['edge_fraction']
        for field in ('short','diagonal'):
            a = added['temporal'][field]['log_increment_error']['rmse']
            z = control['temporal'][field]['log_increment_error']['rmse']
            conditions[field+'/gt_relative_motion_rmse'] = a is not None and z is not None and a <= z+TOL['motion']
        gains[domain].append(added['static']['short']['relative_error']['mae'] < control['static']['short']['relative_error']['mae']-TOL['edge_fraction'])
        checks[key] = conditions
    old = {(r['image'],r['scale']):r for r in point['rows'] if r['role']=='probe'}
    new = {(r['image'],r['scale']):r for r in sized['rows'] if r['role']=='probe'}
    if old.keys() != new.keys():
        raise ValueError('Probe identities differ')
    failures = []
    for k,r in new.items():
        previous = old[k]
        if any(r[x] != previous[x] for x in ('gt','b','domain','sequence','frame_id')):
            raise ValueError('Probe B/GT identity differs')
        for method in ('b','midpoint'):
            a,z = previous['metrics'][method], r['metrics']['midpoint']
            if ((a['output'] and not z['output']) or (a['center_hit'] and not z['center_hit']) or
                    (a['riou'] >= .5 and z['riou'] < .5)):
                failures.append(dict(image=r['image'],scale=r['scale'],reference=method))
    checks['no_new_probe_output_center_riou_failures'] = not failures
    for domain in gains:
        checks[domain+'/strict_short_mae_gain'] = any(gains[domain])
    passed = all(all(v.values()) if isinstance(v,dict) else v for v in checks.values())
    return dict(checks=checks, new_failures=failures, diagnostic_conditions_met=passed,
        formal_training_approved=False, automatic_promotion=False,
        note='Numerical gates on limited TRAIN probe only. No final VAL/TEST, significance or formal approval; DFR is reported alongside GT-relative motion.')


def publish(out):
    write_json(out/'artifacts.json',dict(protocol=VERSION,
        files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='artifacts.json'}))


def run(args):
    out = Path(args.out_dir); out.mkdir(parents=True,exist_ok=False)
    report = dict(protocol=VERSION, status='STARTED', settings=deepcopy(SETTINGS),
        detector_updates=0, detector_forward_calls=0, feature_extractions=0,
        head_updates_total=0, val_access=False, test_access=False, weight_bytes_read=False,
        cache_tensor_loads=0, checkpoint_exported=False, formal_training_approved=False,
        automatic_promotion=False, gpu_devices_used=[])
    started = time.perf_counter()
    with (out/'progress.jsonl').open('x') as stream:
        def progress(log):
            if log['stage']=='update':
                report['head_updates_total'] += 1
            stream.write(json.dumps(log,ensure_ascii=False,allow_nan=False)+'\n'); stream.flush()
            if log['stage']=='update' and (log['step'] in (1,2) or log['step']%25==0):
                print('TRAIN',log['arm'],log['step'],'point',log['point_loss'],
                    'size_unit',log['size_unit_loss'],'total',log['total_loss'],flush=True)
        try:
            protocol,cache,proof = checked_contract(args.training_dir,args.cache_dir)
            report['proof'] = proof
            write_json(out/'protocol.json',protocol); write_json(out/'sources.json',read_json(SOURCES))
            print('Finite TRAIN size check: 128 identities/256 views; 200 updates per arm; B updates/inference=0; no weight export. Not formal training.',flush=True)
            if args.check_only:
                report['status']='STATIC_SIZE_CONTRACT_PASS_NO_TENSOR_LOAD_NO_GPU_NO_UPDATES'
            else:
                torch,formal,size = runtime_modules()
                if size.SIZE_WEIGHT != SETTINGS['size_weight'] or size.SIZE_BETA != SETTINGS['size_beta']:
                    raise ValueError('Fixed loss constants differ')
                device,runtime = execution_device(args.gpu,formal,torch)
                report.update(runtime=runtime,gpu_devices_used=[args.gpu])
                if runtime != cache['runtime']:
                    raise ValueError('Original cache/runtime versions or GPU differ')
                def loaded(name):
                    report['cache_tensor_loads'] += 1
                    progress(dict(stage='train_tensor_loaded',file=name))
                full = load_train_cache(args.cache_dir,cache,formal,torch,loaded)
                records,assignments = select_blocks(full); del full
                batches = schedule(records)
                counts = {role:dict(Counter(r['domain'] for r in records if r['role']==role and r['scale']==1.)) for role in ('fit','probe')}
                if counts != {'fit':{'real':48,'sim':48},'probe':{'real':16,'sim':16}} or len(records)!=256:
                    raise ValueError('Fixed TRAIN sampling budget differs')
                report.update(sample_counts=counts,selected_cpu_tensor_bytes=formal.tensor_bytes(records),
                    minibatches=batches, inherited_fit_ineligible=sum(r['role']=='fit' and not r['eligible'] for r in records))
                write_json(out/'sample_selection.json',dict(blocks=assignments,
                    views=[{k:r[k] for k in ('image','sequence','domain','frame_id','role','scale','eligible')} for r in records]))
                formal.seed_all()
                initial_head = formal.m.SpatialMidpointHead()
                initial = deepcopy(initial_head.state_dict()); del initial_head
                results = {}
                for arm in ARMS:
                    if device.type=='cuda':
                        torch.cuda.reset_peak_memory_stats(args.gpu)
                    result = fit_arm(records,batches,initial,arm,device,formal,size,torch,progress)
                    if device.type=='cuda':
                        result['cuda_peak_bytes'] = dict(allocated=torch.cuda.max_memory_allocated(args.gpu),
                            reserved=torch.cuda.max_memory_reserved(args.gpu))
                    write_json(out/(arm+'.json'),dict(protocol=VERSION,proof=proof,arm=arm,result=result))
                    results[arm] = result
                if (results[ARMS[0]]['initial'] != results[ARMS[1]]['initial'] or
                        results[ARMS[0]]['initial_state'] != results[ARMS[1]]['initial_state'] or
                        report['head_updates_total'] != len(ARMS)*SETTINGS['steps_per_arm']):
                    raise ValueError('Paired initialization/update budget differs')
                verdict = review(results['point_only']['final'],results['point_size']['final'])
                write_json(out/'review.json',verdict)
                report.update(status='TRAIN_SIZE_SHORT_FIT_COMPLETE_REVIEW_REQUIRED',review=verdict,
                    arm_states={k:dict(initial=v['initial_state'],final=v['final_state'],updates=v['completed_updates']) for k,v in results.items()})
            report['elapsed_seconds']=time.perf_counter()-started
            progress(dict(stage='complete',status=report['status']))
        except Exception as error:
            report.update(status='FAILED_SIZE_PREFLIGHT_REVIEW_REQUIRED',error=type(error).__name__+': '+str(error),
                          elapsed_seconds=time.perf_counter()-started)
            progress(dict(stage='failed',error=report['error']))
            write_json(out/'completion.json',report); stream.flush(); publish(out)
            raise
        write_json(out/'completion.json',report)
    publish(out)
    print('Saved',out/'completion.json','status',report['status'],flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--training-dir',default='work_dirs/crane_symeood_k1_port_day2night_midpoint_formal_v1')
    p.add_argument('--cache-dir',default='work_dirs/port_geometry_midpoint_formal_v1_roi_cache')
    p.add_argument('--out-dir',required=True)
    return p


if __name__ == '__main__':
    run(parser().parse_args())
