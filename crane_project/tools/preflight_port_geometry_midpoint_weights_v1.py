#!/usr/bin/env python3
"""Paired TRAIN weight sensitivity: point-only, size/motion x three weights.

Reuse the reviewed paired point-only and motion=.025 JSON results. Fit only
five missing200-step heads. No detector, weights, VAL/TEST or checkpoint IO.
Static mode uses stdlib and existing text evidence, never torch/CUDA/tensors.
"""
import argparse
from collections import Counter
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import preflight_port_geometry_midpoint_motion_v1 as paired

base = paired.base
sha, read_json, write_json = base.sha, base.read_json, base.write_json
VERSION = 'port_geometry_midpoint_weights_v1_train_preflight'
PROTOCOL = ROOT/'crane_project/tools/port_geometry_midpoint_weights_v1_protocol.json'
SOURCES = ROOT/'crane_project/tools/port_geometry_midpoint_weights_v1_sources.json'
SETTINGS = {k: deepcopy(v) for k,v in paired.SETTINGS.items() if k != 'motion_weight'}
WEIGHTS = (.0125, .025, .05)
CONDITIONS = [dict(name='point_only', family='point', weight=0., reused='point_only')]
for family in ('size', 'motion'):
    for weight in WEIGHTS:
        CONDITIONS.append(dict(name=family+'_w'+str(weight).replace('.', 'p'),
            family=family, weight=weight,
            reused='point_motion' if family == 'motion' and weight == .025 else None))
BASELINE_FILES = {'axis_correspondence.json', 'completion.json', 'pair_catalog.json',
    'point_motion.json', 'point_only.json', 'progress.jsonl', 'protocol.json',
    'review.json', 'sample_selection.json', 'sources.json'}


def protocol_document():
    return dict(protocol=VERSION, settings=deepcopy(SETTINGS), weights=list(WEIGHTS),
        conditions=deepcopy(CONDITIONS), parameter_count=17696, tolerances=deepcopy(base.TOL),
        objective='Unchanged point loss + ONE size OR motion unit loss times .0125/.025/.05. Never combine the two auxiliaries. Unit formulas/beta/GT matching/transport remain v1.',
        fairness='Same128 TRAIN identities/256 views,96fit/32probe; original paired4pairs/8frames,2real+2sim, seed1703 initialization and seed1705 schedule,200 updates per condition. No independent-frame size-v1 result reuse.',
        reuse='Require SHA-indexed completed motion-v1 text evidence, exact proof/runtime/protocol/sources/initialization/sample/pair/schedule/log consistency. Reuse point-only and motion=.025; no silent retraining on mismatch. Current TRAIN tensors must reproduce original selection/catalog/initial results exactly.',
        diagnostics='Report point/size/motion unit losses for every new update; component gradients at1/2/25/100/200. Valid zero auxiliary gradient is logged. Active auxiliary needs a measured positive signal for numerical gate. Finite gradients/parameters, step2 stem and clip10 checks retained.',
        review='Same joint static geometry/GT-relative motion/center/sim angle/RIoU/ACI/coverage safeguards for all six variants, including long motion. Size goal: strict short static-MAE gain at>=one scale in each domain. Motion goal: strict short AND diagonal GT-relative motion gain at>=one same scale in each domain. Record every configuration; do not rank/select a winner or relax a failed v1 gate.',
        online='Original sampler,9x9 ROI,context1.5,prior sigma1,point beta.1,raw w/h-angle association,zero identity,full-B fallback,score/count/missing and original sx/sy restoration unchanged. No inference history/smoothing.',
        output='Five new full text results, reused source pointers, all-seven summary JSON/CSV, reviews, sampling/pair/axis audit, progress/completion/artifacts. No PTH, cache copy or fabricated paper curves.',
        scope=dict(train_identities=128, train_views=256, fit_identities=96, probe_identities=32,
            conditions_total=7, new_conditions=5, reused_conditions=2,
            new_head_updates_total=5*SETTINGS['steps_per_arm'],
            reused_historical_updates=2*SETTINGS['steps_per_arm'],
            represented_updates=7*SETTINGS['steps_per_arm'], detector_updates=0,
            detector_forward_calls=0, feature_extractions=0, val_access=False,
            test_access=False, weight_bytes_read=False, checkpoint_exported=False,
            formal_training_approved=False, automatic_promotion=False),
        limitations=['Fixed single-seed short TRAIN screen, not formal full-TRAIN/full-VAL parameter sensitivity or independent generalization/significance.',
            'Same numeric weight does not imply equal size/motion gradient strength. Log gradients; lower DFR/higher ACI alone do not prove GT tracking.',
            'Preserve old failed v1 results. New user-authorized fixed grid, not retrospective reinterpretation or unlimited coefficient/step/seed search.',
            'Auxiliaries without confirmed joint gain are candidates, not final-method contributions. Negative ablations may be reported truthfully; paper main sensitivity needs formal budget/full VAL.',
            'TEST repeatedly exposed; no TEST tuning/reselection. Unchanged depth interface does not certify depth accuracy.'])


def audit_baseline(directory, parent_proof, cache):
    """Validate text reuse with no torch imports or tensor deserialization."""
    directory = Path(directory)
    index = read_json(directory/'artifacts.json')
    if index['protocol'] != paired.VERSION or set(index['files']) != BASELINE_FILES:
        raise ValueError('Only complete paired motion-v1 text evidence can be reused')
    for name,digest in index['files'].items():
        if sha(directory/name) != digest:
            raise ValueError('Baseline artifact SHA differs: '+name)
    if (read_json(directory/'protocol.json') != paired.protocol_document() or
            read_json(directory/'sources.json') != read_json(paired.SOURCES)):
        raise ValueError('Baseline paired protocol/sources differ')
    complete = read_json(directory/'completion.json')
    forbidden = ('detector_updates','detector_forward_calls','feature_extractions')
    flags = ('val_access','test_access','weight_bytes_read','checkpoint_exported',
             'formal_training_approved','automatic_promotion')
    if (complete['protocol'] != paired.VERSION or
            complete['status'] != 'TRAIN_MOTION_SHORT_FIT_COMPLETE_REVIEW_REQUIRED' or
            complete['settings'] != paired.SETTINGS or complete['proof'] != parent_proof or
            complete['runtime'] != cache['runtime'] or
            complete['head_updates_total'] != 2*SETTINGS['steps_per_arm'] or
            complete['cache_tensor_loads'] != 2 or
            any(complete[k] != 0 for k in forbidden) or any(complete[k] is not False for k in flags)):
        raise ValueError('Baseline completion/cache/runtime/scope differs')
    catalog = read_json(directory/'pair_catalog.json')
    batches = paired.schedule(catalog)
    if complete['pair_minibatches'] != batches or complete['pair_support'] != catalog['support']:
        raise ValueError('Baseline pair schedule/support differs')
    progress = [json.loads(s, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
                for s in (directory/'progress.jsonl').read_text().splitlines()]
    counts = Counter(z['stage'] for z in progress)
    expected = Counter(train_tensor_loaded=2,optimizer_step=complete['head_updates_total'],
        update=complete['head_updates_total'],complete=1)
    if counts != expected or any(z.get('arm') not in paired.ARMS for z in progress
            if z['stage'] in ('optimizer_step','update')):
        raise ValueError('Baseline total progress budget differs')
    reused = {}
    for arm in paired.ARMS:
        document = read_json(directory/(arm+'.json'))
        result = document['result']
        if (document['protocol'] != paired.VERSION or document['proof'] != parent_proof or
                document['arm'] != arm or result['completed_updates'] != SETTINGS['steps_per_arm'] or
                result['checkpoint_exported'] is not False or
                len(result['logs']) != SETTINGS['steps_per_arm'] or
                complete['arm_states'][arm] != dict(initial=result['initial_state'],
                    final=result['final_state'],updates=result['completed_updates'])):
            raise ValueError('Baseline arm completion/identity differs: '+arm)
        logs = result['logs']
        if ([z for z in progress if z.get('stage') == 'update' and z.get('arm') == arm] != logs or
                [z['step'] for z in progress if z.get('stage') == 'optimizer_step' and z.get('arm') == arm] != list(range(1,len(logs)+1))):
            raise ValueError('Baseline progress/update accounting differs: '+arm)
        for step,z in enumerate(logs,1):
            expected = z['point_loss']+(.025*z['motion_unit_loss'] if arm == 'point_motion' else 0.)
            if (z['stage'] != 'update' or z['arm'] != arm or z['step'] != step or
                    z['pair_indices'] != batches[step-1] or z['motion_applied'] != (arm == 'point_motion') or
                    not all(math.isfinite(z[k]) for k in ('point_loss','motion_unit_loss','total_loss','grad_norm_before_clip','grad_norm_after_clip','stem_grad_norm')) or
                    not math.isclose(z['total_loss'],expected,abs_tol=1e-7,rel_tol=1e-6)):
                raise ValueError('Baseline fixed batch/loss formula differs: '+arm)
        measured = any(z.get('component_gradients') is not None and
            z['component_gradients']['motion_unit_norm'] > 0 for z in logs)
        if measured != result['motion_gradient_signal_seen']:
            raise ValueError('Baseline recorded motion gradient signal differs: '+arm)
        reused[arm] = result
    a,b = (reused[k] for k in paired.ARMS)
    if a['initial'] != b['initial'] or a['initial_state'] != b['initial_state'] or any(r['b'] != r['midpoint'] for r in a['initial']['rows']):
        raise ValueError('Baseline zero initialization differs')
    verdict = paired.review(a['final'],b['final'])
    verdict['checks']['measured_motion_gradient_signal_both_arms'] = all(v['motion_gradient_signal_seen'] for v in reused.values())
    verdict['diagnostic_conditions_met'] = verdict['diagnostic_conditions_met'] and verdict['checks']['measured_motion_gradient_signal_both_arms']
    if verdict != read_json(directory/'review.json') or complete['review'] != verdict:
        raise ValueError('Baseline review differs')
    return dict(completion=complete, results=reused, catalog=catalog,
        selection=read_json(directory/'sample_selection.json'),
        axis_audit=read_json(directory/'axis_correspondence.json'),
        files=dict(index['files'], **{'artifacts.json':sha(directory/'artifacts.json')}))


def checked_contract(training_dir, cache_dir, baseline_dir):
    if read_json(PROTOCOL) != protocol_document():
        raise ValueError('Fixed weight grid protocol differs')
    manifest,parent = read_json(SOURCES),read_json(paired.SOURCES)
    required = {str(p.relative_to(ROOT)) for p in (Path(__file__).resolve(),PROTOCOL,
        ROOT/'tests/test_port_geometry_midpoint_weights_v1.py',paired.SOURCES)}
    if (manifest['protocol'] != VERSION or manifest['parent_sources_sha256'] != sha(paired.SOURCES) or
            set(manifest['sources']) != set(parent['sources']) | required):
        raise ValueError('Weight grid source scope differs')
    for name,digest in manifest['sources'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Source SHA differs: '+name)
    if SETTINGS != {k:deepcopy(v) for k,v in paired.SETTINGS.items() if k != 'motion_weight'}:
        raise ValueError('Paired fixed training settings differ')
    _,cache,parent_proof = paired.checked_contract(training_dir,cache_dir)
    reused = audit_baseline(baseline_dir,parent_proof,cache)
    proof = dict(parent_proof=parent_proof, sources_sha256=sha(SOURCES),
        protocol_sha256=sha(PROTOCOL), baseline_files=reused['files'])
    return protocol_document(),cache,proof,reused


def runtime_modules():
    torch,formal,motion = paired.runtime_modules()
    from crane_project.utils import port_geometry_midpoint_size_v1 as size
    return torch,formal,size,motion


def component_gradients(point, edge, change, params, weight, family, torch):
    vectors = [torch.cat([g.detach().double().reshape(-1) for g in
        torch.autograd.grad(loss,params,retain_graph=True)]) for loss in (point,edge,change)]
    norms = [float(v.norm()) for v in vectors]
    result = dict(point_norm=norms[0])
    for i,name in enumerate(('size','motion'),1):
        cosine = float((vectors[0]*vectors[i]).sum())/(norms[0]*norms[i]) if norms[0] and norms[i] else None
        result[name] = dict(unit_norm=norms[i],cosine_with_point=cosine,
            weighted_norm=norms[i]*weight if name == family else 0.,
            active_weighted_over_point=norms[i]*weight/norms[0] if name == family and norms[0] else None)
    if not all(math.isfinite(v) for v in norms) or any(
            d['cosine_with_point'] is not None and not math.isfinite(d['cosine_with_point']) for d in (result['size'],result['motion'])):
        raise ValueError('Nonfinite component gradient')
    return result


def fit_condition(records,catalog,batches,initial,condition,device,formal,size,motion,torch,progress,reference=None):
    formal.seed_all()
    head = formal.m.SpatialMidpointHead().to(device);head.load_state_dict(initial,strict=True)
    before = formal.g.state_digest(head);start = base.evaluate(head,records,device,formal)
    if any(r['b'] != r['midpoint'] for r in start['rows']):
        raise ValueError('New condition zero initialization is not exact B')
    if reference is not None and (before != reference['initial_state'] or start != reference['initial']):
        raise ValueError('Current initial outputs differ from reused control before updates')
    optimizer = torch.optim.Adam(head.parameters(),lr=SETTINGS['lr'],weight_decay=SETTINGS['weight_decay'])
    family,weight = condition['family'],condition['weight'];logs = []
    for step,indices in enumerate(batches,1):
        head.train();optimizer.zero_grad()
        roi,support,b,bm,xy,gt = paired.batch(records,catalog,indices,device,torch)
        out = head(roi,support,b,bm,xy);targets = formal.m.target_points(gt,b,bm,xy)
        point,_ = formal.m.point_loss(out['points_roi'],targets)
        edge,ed = size.size_loss(out['points_original'],targets)
        change,cd = motion.motion_loss(out['points_original'],targets)
        active = edge if family == 'size' else change
        total = point+weight*active
        params = list(head.parameters())
        diagnostic = component_gradients(point,edge,change,params,weight,family,torch) if step in (1,2,25,100,200) else None
        if not all(bool(torch.isfinite(z)) for z in (point,edge,change,total)):
            raise ValueError('Nonfinite parameter condition loss')
        total.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params):
            raise ValueError('Missing/nonfinite condition gradient')
        stem = sum(float(p.grad.double().square().sum()) for p in head.stem.parameters())**.5
        if step == 2 and stem <= 0:
            raise ValueError('Condition step2 stem gradient ineffective')
        norm = float(torch.nn.utils.clip_grad_norm_(params,SETTINGS['clip_norm']))
        after = sum(float(p.grad.double().square().sum()) for p in params)**.5
        if not math.isfinite(norm) or after > SETTINGS['clip_norm']+1e-4:
            raise ValueError('Invalid condition clipping')
        optimizer.step();progress(dict(stage='optimizer_step',condition=condition['name'],step=step))
        if any(not bool(torch.isfinite(p).all()) for p in params):
            raise ValueError('Nonfinite updated condition parameters')
        log = dict(stage='update',condition=condition['name'],family=family,weight=weight,
            step=step,pair_indices=indices,point_loss=float(point.detach()),
            size_unit_loss=float(edge.detach()),motion_unit_loss=float(change.detach()),
            auxiliary_weighted_loss=weight*float(active.detach()),total_loss=float(total.detach()),
            matched_axis_size_parts=ed['parts'].detach().cpu().tolist(),
            matched_axis_motion_parts=cd['parts'].detach().cpu().tolist(),
            predicted_log_increment=cd['predicted_log_increment'].detach().cpu().tolist(),
            gt_log_increment=cd['gt_log_increment'].cpu().tolist(),
            next_axes_exchanged=cd['swap'].cpu().tolist(),axis_ambiguous=cd['ambiguous'].cpu().tolist(),
            grad_norm_before_clip=norm,grad_norm_after_clip=after,
            clip_multiplier=min(1.,SETTINGS['clip_norm']/(norm+1e-6)),stem_grad_norm=stem,
            component_gradients=diagnostic)
        logs.append(log);progress(log)
    final = base.evaluate(head,records,device,formal)
    result = dict(initial_state=before,final_state=formal.g.state_digest(head),initial=start,
        final=final,logs=logs,completed_updates=len(logs),checkpoint_exported=False,
        active_auxiliary_gradient_signal_seen=any(z['component_gradients'] is not None and
            z['component_gradients'][family]['unit_norm'] > 0 for z in logs))
    del head,optimizer
    if device.type == 'cuda':torch.cuda.empty_cache()
    return result


def condition_review(control,result,condition):
    verdict = paired.review(control['final'],result['final'])
    if condition['family'] == 'size':
        size_verdict = base.review(control['final'],result['final'])
        for domain in ('real','sim'):
            del verdict['checks'][domain+'/strict_short_and_diagonal_motion_gain']
            key = domain+'/strict_short_mae_gain'
            verdict['checks'][key] = size_verdict['checks'][key]
    signal = result.get('active_auxiliary_gradient_signal_seen',result.get('motion_gradient_signal_seen',False))
    verdict['checks']['active_auxiliary_gradient_signal_seen'] = signal
    verdict['diagnostic_conditions_met'] = all(all(v.values()) if isinstance(v,dict) else v for v in verdict['checks'].values())
    verdict['note'] = 'Fixed grid numerical TRAIN screen only; all conditions retained, no automatic winner/formal approval or paper efficacy claim.'
    return verdict


def summary_rows(results):
    rows = []
    for condition in CONDITIONS:
        result = results[condition['name']]
        for key,group in result['final']['groups'].items():
            role,domain,scale = key.split('/');value = group['midpoint']
            row = dict(condition=condition['name'],family=condition['family'],weight=condition['weight'],
                origin='reused' if condition['reused'] else 'new',role=role,domain=domain,scale=float(scale),
                frames=group['frames'],output_frames=group['output_frames'],pairs=group['temporal_pairs'],
                output_coverage_pct=group['output_coverage']['pct'],
                conditional_center_correct_count=value['conditional_center_correct']['numerator'],
                conditional_center_correct_pct=value['conditional_center_correct']['pct'],
                all_frame_center_correct_count=value['all_frame_center_correct']['numerator'],
                all_frame_center_correct_pct=value['all_frame_center_correct']['pct'],
                center_mean_px=value['center_error_px']['mean'],center_rmse_px=value['center_error_px']['rmse'],
                all_frame_mean_riou=value['all_frame_mean_riou'],
                output_mean_riou=value['all_frame_mean_riou']*group['frames']/group['output_frames'] if group['output_frames'] else None,
                pure_angle_rmse_deg=value['pure_angle_error_deg']['rmse'],
                dfr_pct_per_frame=value['dfr_pct_per_frame'],aci=value['aci'],
                angle_gt_increment_rmse_deg=value['angle_increment_error_deg']['rmse'],
                fallbacks=group['fallbacks'],acceptance_switch_pairs=group['acceptance_switch_pairs'])
            for field in ('long','short','diagonal'):
                row[field+'_mae_pct'] = 100*value['static'][field]['relative_error']['mae']
                row[field+'_signed_relative_bias_pct'] = 100*value['static'][field]['relative_error']['mean']
                row[field+'_gt_log_increment_rmse'] = value['temporal'][field]['log_increment_error']['rmse']
            rows.append(row)
    return rows


def publish(out):
    write_json(out/'artifacts.json',dict(protocol=VERSION,
        files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'artifacts.json'}))


def run(args):
    out = Path(args.out_dir)
    if any(Path(p).resolve() == out.resolve() or Path(p).resolve() in out.resolve().parents
           for p in (args.training_dir,args.cache_dir,args.baseline_dir)):
        raise ValueError('Output must be separate from immutable input directories')
    try:out.mkdir(parents=True,exist_ok=False)
    except FileExistsError:
        raise FileExistsError('Output exists; existing results preserved. Use a new --out-dir suffix.') from None
    report = dict(protocol=VERSION,status='STARTED',settings=deepcopy(SETTINGS),conditions=deepcopy(CONDITIONS),
        head_updates_total=0,reused_historical_updates=0,represented_updates=0,cache_tensor_loads=0,
        detector_updates=0,detector_forward_calls=0,feature_extractions=0,gpu_devices_used=[],
        val_access=False,test_access=False,weight_bytes_read=False,checkpoint_exported=False,
        formal_training_approved=False,automatic_promotion=False,selected_condition=None)
    started = time.perf_counter()
    with (out/'progress.jsonl').open('x') as stream:
        def progress(log):
            if log['stage'] == 'optimizer_step':
                report['head_updates_total'] += 1
                report['represented_updates'] = report['head_updates_total']+report['reused_historical_updates']
            stream.write(json.dumps(log,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
            if log['stage'] == 'update' and (log['step'] in (1,2) or log['step'] % 25 == 0):
                print('TRAIN',log['condition'],log['step'],'point',log['point_loss'],
                    'size',log['size_unit_loss'],'motion',log['motion_unit_loss'],'total',log['total_loss'],flush=True)
        try:
            protocol,cache,proof,reused = checked_contract(args.training_dir,args.cache_dir,args.baseline_dir)
            report['proof'] = proof
            write_json(out/'protocol.json',protocol);write_json(out/'sources.json',read_json(SOURCES))
            write_json(out/'reuse_sources.json',dict(baseline_dir=str(Path(args.baseline_dir)),
                files=reused['files'],conditions={c['name']:c['reused']+'.json' for c in CONDITIONS if c['reused']}))
            print('Fixed TRAIN weight grid:7 conditions;2 text results reused;5x200=1000 NEW updates;B inference/updates=0. Not formal training.',flush=True)
            if args.check_only:
                report['status'] = 'STATIC_WEIGHTS_CONTRACT_PASS_NO_TENSOR_LOAD_NO_GPU_NO_UPDATES'
            else:
                torch,formal,size,motion = runtime_modules()
                if size.SIZE_BETA != base.SETTINGS['size_beta'] or motion.MOTION_BETA != SETTINGS['motion_beta'] or motion.AXIS_TIE_TOLERANCE != SETTINGS['axis_tie_tolerance']:
                    raise ValueError('Inherited auxiliary loss constants differ')
                device,runtime = base.execution_device(args.gpu,formal,torch)
                if runtime != cache['runtime']:raise ValueError('Original cache/runtime/GPU differ')
                report.update(runtime=runtime,gpu_devices_used=[args.gpu])
                def loaded(name):
                    report['cache_tensor_loads'] += 1;progress(dict(stage='train_tensor_loaded',file=name))
                full = base.load_train_cache(args.cache_dir,cache,formal,torch,loaded)
                records,blocks = base.select_blocks(full);del full
                selection = dict(blocks=blocks,views=[{k:r[k] for k in ('image','sequence','domain','frame_id','role','scale','eligible')} for r in records])
                catalog = paired.pair_catalog(records);batches = paired.schedule(catalog)
                if selection != reused['selection'] or catalog != reused['catalog'] or batches != reused['completion']['pair_minibatches']:
                    raise ValueError('Current TRAIN selection/pairs differ from reused results')
                formal.seed_all();initial_head = formal.m.SpatialMidpointHead()
                initial = deepcopy(initial_head.state_dict());initial_state = formal.g.state_digest(initial_head)
                del initial_head
                if initial_state != reused['results']['point_only']['initial_state']:
                    raise ValueError('Current initialized head differs from reused results')
                write_json(out/'sample_selection.json',selection);write_json(out/'pair_catalog.json',catalog)
                # CPU targets use the same implementation as the reviewed audit.
                audit = paired.correspondence_audit(records,catalog,formal,motion,torch)
                if audit != reused['axis_audit']:raise ValueError('Current GT axis audit differs')
                write_json(out/'axis_correspondence.json',audit)
                report.update(selected_cpu_tensor_bytes=formal.tensor_bytes(records),pair_count=len(catalog['pairs']),
                    eligible_fit_pair_count=sum(p['eligible_for_gradient'] for p in catalog['pairs']),
                    reused_historical_updates=2*SETTINGS['steps_per_arm'],
                    represented_updates=2*SETTINGS['steps_per_arm'])
                results = {c['name']:reused['results'][c['reused']] for c in CONDITIONS if c['reused']}
                for c in CONDITIONS:
                    if c['reused']:continue
                    if device.type == 'cuda':torch.cuda.reset_peak_memory_stats(args.gpu)
                    result = fit_condition(records,catalog,batches,initial,c,device,formal,size,motion,torch,progress,
                        reference=results['point_only'])
                    if result['initial_state'] != initial_state or result['initial'] != results['point_only']['initial']:
                        raise ValueError('Current initial outputs differ from reused control')
                    if device.type == 'cuda':
                        result['cuda_peak_bytes'] = dict(allocated=torch.cuda.max_memory_allocated(args.gpu),reserved=torch.cuda.max_memory_reserved(args.gpu))
                    write_json(out/(c['name']+'.json'),dict(protocol=VERSION,proof=proof,condition=c,result=result))
                    results[c['name']] = result
                if report['head_updates_total'] != 5*SETTINGS['steps_per_arm']:
                    raise ValueError('New update budget differs')
                reviews = {c['name']:condition_review(results['point_only'],results[c['name']],c)
                    for c in CONDITIONS if c['family'] != 'point'}
                write_json(out/'reviews.json',reviews)
                rows = summary_rows(results)
                write_json(out/'summary.json',dict(protocol=VERSION,scope=protocol['scope'],rows=rows,
                    groups={c['name']:results[c['name']]['final']['groups'] for c in CONDITIONS},
                    original_reused_review=reused['completion']['review'],reviews=reviews,
                    selected_condition=None,formal_training_approved=False,
                    paper_evidence='Limited TRAIN screening only; final paper sensitivity requires fixed formal budget/full VAL.'))
                with (out/'summary.csv').open('x',newline='') as csv_stream:
                    writer = csv.DictWriter(csv_stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                report.update(status='TRAIN_WEIGHTS_GRID_COMPLETE_REVIEW_REQUIRED',
                    represented_updates=report['head_updates_total']+report['reused_historical_updates'],
                    numerical_gate_flags={name:v['diagnostic_conditions_met'] for name,v in reviews.items()})
                for c in CONDITIONS:
                    print('CONDITION',c['name'],'origin', 'reused' if c['reused'] else 'new',
                        'numerical_gate',report['numerical_gate_flags'].get(c['name'],'reference'),flush=True)
            report['elapsed_seconds'] = time.perf_counter()-started
            progress(dict(stage='complete',status=report['status']))
        except Exception as error:
            report.update(status='FAILED_WEIGHTS_PREFLIGHT_REVIEW_REQUIRED',
                error=type(error).__name__+': '+str(error),elapsed_seconds=time.perf_counter()-started)
            progress(dict(stage='failed',error=report['error']))
            write_json(out/'completion.json',report);stream.flush();publish(out);raise
        write_json(out/'completion.json',report)
    publish(out)
    print('Saved',out/'completion.json','status',report['status'],flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-only',action='store_true');p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--training-dir',default='work_dirs/crane_symeood_k1_port_day2night_midpoint_formal_v1')
    p.add_argument('--cache-dir',default='work_dirs/port_geometry_midpoint_formal_v1_roi_cache')
    p.add_argument('--baseline-dir',default='work_dirs/port_geometry_midpoint_motion_v1_train')
    p.add_argument('--out-dir',required=True)
    return p


if __name__ == '__main__':run(parser().parse_args())
