#!/usr/bin/env python3
"""Finite TRAIN point-only vs GT-relative size-change supervision.

Reuse two formal TRAIN CPU shards; no B/weight/VAL/TEST loads or checkpoint
export. Same newly initialized head, paired8-frame batches and200 updates
per arm. Static mode: stdlib/metadata only, no torch/cache tensors/GPU.
"""
import argparse
from collections import Counter
from copy import deepcopy
import math
from pathlib import Path
import random
import sys
import time
import json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import preflight_port_geometry_midpoint_size_v1 as base

VERSION = 'port_geometry_midpoint_motion_v1_train_preflight'
PROTOCOL = ROOT/'crane_project/tools/port_geometry_midpoint_motion_v1_protocol.json'
SOURCES = ROOT/'crane_project/tools/port_geometry_midpoint_motion_v1_sources.json'
SETTINGS = dict(seed=1703, schedule_seed=1705, steps_per_arm=200, batch_size=8,
    pairs_per_batch=4, real_pairs_per_batch=2, sim_pairs_per_batch=2,
    lr=.001, weight_decay=0., clip_norm=10., motion_weight=.025, motion_beta=.1,
    axis_tie_tolerance=1e-6, block_frames=8, fit_blocks_per_domain=6,
    probe_blocks_per_domain=2, min_interblock_gap_frames=16, scales=[1., .5])
ARMS = ('point_only', 'point_motion')
sha, read_json, write_json = base.sha, base.read_json, base.write_json


def protocol_document():
    return dict(protocol=VERSION, settings=deepcopy(SETTINGS),
        tolerances=deepcopy(base.TOL), conditions=list(ARMS), parameter_count=17696,
        objective='Original mean per-frame four-point SmoothL1 beta.1; experimental arm adds fixed .025 times mean pair/two-axis SmoothL1(delta log projected edge - delta log GT edge,beta.1). No single-frame size auxiliary term.',
        axis_transport='GT-only unsigned normalized matched-axis dot scores choose same/exchanged next-frame axes. Apply identical mapping to GT and predicted projected edges. Absolute score tie<=1e-6 keeps same order and is reported, never dropped. Existing per-frame cyclic point labels remain unchanged. No predicted-side sorting; not physical-side identity.',
        projection='Continuous original-coordinate v1 projection BEFORE detached-B zero identity/full fallback, finite positive edges required. GT detached; no clamping/dropping invalid prediction or GT geometry.',
        gradients='Finite point/motion/combined gradients; report component norms/ratio/cosine at1/2/25/100/200. Zero motion gradient at a valid minimum is logged, not an exception; diagnostic continuation also requires positive measured motion gradient signal in each arm. Stem step1 zero is expected; step2 combined gradient must be effective.',
        sampling='Reuse fixed size-v1 metadata-only 128 TRAIN identities/256 views and96fit/32probe roles,6fit+2probe consecutive8-frame blocks per domain,>=16 intervening frames. Same two scales. No reselection by eligibility, GT motion, angle, aspect, VAL/TEST or observed effect.',
        pairing='Same TRAIN fit role/sequence/scale and adjacent frame IDs only, both inherited B-present center<15px for gradient eligibility. Retain excluded pair counts; gaps/missing frames/role boundaries break support. Probe never enters gradients.',
        schedule='Seed1705 shuffled cyclic eligible pair queues,2real+2sim pairs =>8 frames/update; duplicate shared endpoints retained and disclosed. Same newly initialized seed1703 head and200 pair minibatches per arm. Point-only uses identical pair batches; previous independent-frame point-only results are not this control.',
        cache='Unchanged completed formal cache/source/runtime identity. Deserialize train_s1.pt and train_s05.pt only; no B, feature extraction, weight, VAL/TEST tensor or image/annotation-byte access.',
        online='Original stateless SpatialMidpointHead, sampler, prior calibration, isotropic transforms/original sx/sy restoration, raw w/h-angle association, zero identity, full-B fallback and score/count/missing outputs unchanged.',
        review='Keep original probe joint nonregression conditions versus paired point-only (static edges,short/diagonal GT-relative motion,center,sim pure angle,RIoU,ACI) and B center/RIoU/sim angle/coverage; also protect long-edge GT-relative motion. Replace old strict static-short gain requirement with strict short AND diagonal GT-relative motion gain at>=one same scale per domain. Numerical tolerances only; no coefficient/step/seed sweep or automatic promotion.',
        scope=dict(train_identities=128, train_views=256, fit_identities=96, probe_identities=32,
            head_updates_total=400, detector_updates=0, detector_forward_calls=0,
            feature_extractions=0, val_access=False, test_access=False, weight_bytes_read=False,
            checkpoint_exported=False, formal_training_approved=False, automatic_promotion=False),
        limitations=['Only paired TRAIN probe; B and prior formal head have seen TRAIN. Not independent VAL/generalization or full-video/significance evidence.',
            'Relative-change loss cannot correct constant multiplicative size bias; original point loss supplies static supervision. Shared head can still change center/angle/static dimensions.',
            'GT axis transport is unoriented geometric correspondence, ambiguous at45deg and not guaranteed physical-side identity. GT motion is annotation-derived, not depth/physical truth.',
            'DFR/ACI are proxies; lower DFR does not certify GT tracking. TEST repeatedly exposed; no TEST tuning/reselection. Unchanged depth interface does not certify depth accuracy.'])


def checked_contract(training_dir, cache_dir):
    if read_json(PROTOCOL) != protocol_document():
        raise ValueError('Fixed motion protocol differs')
    manifest, parent = read_json(SOURCES), read_json(base.SOURCES)
    required = {str(p.relative_to(ROOT)) for p in (Path(__file__).resolve(), PROTOCOL,
        ROOT/'crane_project/utils/port_geometry_midpoint_motion_v1.py',
        ROOT/'tests/test_port_geometry_midpoint_motion_v1.py', base.SOURCES)}
    if (manifest['protocol'] != VERSION or manifest['parent_sources_sha256'] != sha(base.SOURCES) or
            set(manifest['sources']) != set(parent['sources']) | required):
        raise ValueError('Motion source scope differs')
    for name,digest in manifest['sources'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Source SHA differs: '+name)
    _,cache,proof = base.checked_contract(training_dir,cache_dir)
    proof = dict(proof, inherited_size_sources_sha256=sha(base.SOURCES),
        inherited_size_protocol_sha256=sha(base.PROTOCOL),
        sources_sha256=sha(SOURCES), protocol_sha256=sha(PROTOCOL))
    return protocol_document(),cache,proof


def pair_catalog(records):
    keys = [(r['image'],r['scale']) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate selected TRAIN view')
    grouped = {}
    for i,r in enumerate(records):
        if r['role'] not in ('fit','probe') or r['scale'] not in SETTINGS['scales']:
            raise ValueError('Only selected TRAIN roles/scales allowed')
        grouped.setdefault((r['role'],r['sequence'],r['scale']),[]).append(i)
    pairs=[]; support={}
    for (role,sequence,scale),indices in sorted(grouped.items()):
        domain=sequence.split('_')[0]
        if domain not in ('real','sim') or any(records[i]['domain']!=domain for i in indices):
            raise ValueError('Selected pair domain differs')
        previous=None
        key=role+'/'+domain+'/'+str(scale)
        existing_group=key in support
        count=support.setdefault(key,dict(sequence_or_gap_boundaries=0,adjacent_identity_pairs=0,
            missing_output_pairs=0,ineligible_gradient_pairs=0,eligible_fit_pairs=0))
        if existing_group:count['sequence_or_gap_boundaries']+=1
        for current in sorted(indices,key=lambda i:records[i]['frame_id']):
            if previous is not None:
                a,b=records[previous],records[current]
                if b['frame_id']!=a['frame_id']+1:
                    count['sequence_or_gap_boundaries']+=1
                else:
                    count['adjacent_identity_pairs']+=1
                    present=bool(a['boxes_original'].shape[0] and b['boxes_original'].shape[0])
                    eligible=present and a['eligible'] and b['eligible']
                    count['missing_output_pairs']+=not present
                    count['ineligible_gradient_pairs']+=role=='fit' and not eligible
                    count['eligible_fit_pairs']+=role=='fit' and eligible
                    pairs.append(dict(indices=[previous,current],role=role,domain=domain,
                        sequence=sequence,scale=scale,previous=a['image'],image=b['image'],
                        present=present,eligible_for_gradient=role=='fit' and eligible))
            previous=current
    return dict(pairs=pairs,support=support)


def schedule(catalog):
    pairs=catalog['pairs']
    pools={d:[i for i,p in enumerate(pairs) if p['domain']==d and p['eligible_for_gradient']]
           for d in ('real','sim')}
    if any(len(v)<2 for v in pools.values()):
        raise ValueError('Insufficient inherited eligible fit pairs; do not resample')
    rng=random.Random(SETTINGS['schedule_seed'])
    orders={d:rng.sample(p,len(p)) for d,p in pools.items()}; offsets={d:0 for d in pools}
    batches=[]
    for _ in range(SETTINGS['steps_per_arm']):
        batch=[]
        for d in ('real','sim'):
            for _ in range(2):
                if offsets[d]==len(orders[d]):
                    orders[d]=rng.sample(pools[d],len(pools[d]));offsets[d]=0
                batch.append(orders[d][offsets[d]]);offsets[d]+=1
        batches.append(batch)
    return batches


def batch(records,catalog,indices,device,torch):
    pairs=[catalog['pairs'][i] for i in indices]
    if any(not p['eligible_for_gradient'] for p in pairs):
        raise ValueError('Only eligible consecutive fit pairs enter gradients')
    frames=[i for p in pairs for i in p['indices']]
    return base.batch(records,frames,device,torch)


def runtime_modules():
    import torch
    from crane_project.tools import train_port_geometry_midpoint_formal_v1 as formal
    from crane_project.utils import port_geometry_midpoint_motion_v1 as motion
    return torch,formal,motion


def correspondence_audit(records,catalog,formal,motion,torch):
    rows=[]
    for p in catalog['pairs']:
        if not p['present']:
            rows.append(dict(p,status='MISSING_OUTPUT_NO_GT_TRANSPORT'));continue
        tensors=[torch.cat([records[i][k] for i in p['indices']]) for k in
                 ('gt_original','boxes_original','boxes_model','scale_xy')]
        targets=formal.m.target_points(*tensors)
        detail=motion.axis_correspondence(targets['original'])
        rows.append(dict(p,status='GT_ONLY_AXIS_TRANSPORT',
            next_axes_exchanged=bool(detail['swap'][0]),ambiguous=bool(detail['ambiguous'][0]),
            score_margin=float(detail['score_margin'][0]),
            transported_angle_deg=float(detail['transported_angle_deg'][0]),
            gt_log_increment=(detail['gt_current'].log()-detail['gt_previous'].log())[0].tolist()))
    return rows


def gradient_report(point,motion,params,torch):
    gradients=[torch.autograd.grad(loss,params,retain_graph=True) for loss in (point,motion)]
    vectors=[torch.cat([g.detach().double().reshape(-1) for g in gs]) for gs in gradients]
    pn,mn=(float(v.norm()) for v in vectors)
    cosine=float((vectors[0]*vectors[1]).sum())/(pn*mn) if pn and mn else None
    if not all(math.isfinite(v) for v in (pn,mn)) or (cosine is not None and not math.isfinite(cosine)):
        raise ValueError('Nonfinite motion component gradient')
    return dict(point_norm=pn,motion_unit_norm=mn,motion_weighted_norm=mn*SETTINGS['motion_weight'],
        weighted_motion_over_point=mn*SETTINGS['motion_weight']/pn if pn else None,cosine=cosine)


def fit_arm(records,catalog,batches,initial,arm,device,formal,motion,torch,progress):
    formal.seed_all()
    head=formal.m.SpatialMidpointHead().to(device);head.load_state_dict(initial,strict=True)
    before=formal.g.state_digest(head);start=base.evaluate(head,records,device,formal)
    if any(r['b']!=r['midpoint'] for r in start['rows']):
        raise ValueError('Zero initialization is not exact B')
    optimizer=torch.optim.Adam(head.parameters(),lr=SETTINGS['lr'],weight_decay=SETTINGS['weight_decay'])
    logs=[]
    for step,indices in enumerate(batches,1):
        head.train();optimizer.zero_grad()
        roi,support,b,bm,xy,gt=batch(records,catalog,indices,device,torch)
        out=head(roi,support,b,bm,xy);targets=formal.m.target_points(gt,b,bm,xy)
        point,_=formal.m.point_loss(out['points_roi'],targets)
        change,detail=motion.motion_loss(out['points_original'],targets)
        loss=point+SETTINGS['motion_weight']*change if arm=='point_motion' else point
        params=list(head.parameters())
        diagnostics=gradient_report(point,change,params,torch) if step in (1,2,25,100,200) else None
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite paired motion loss')
        loss.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params):
            raise ValueError('Missing/nonfinite paired head gradient')
        stem=sum(float(p.grad.double().square().sum()) for p in head.stem.parameters())**.5
        if step==2 and stem<=0:
            raise ValueError('Stem gradient ineffective after zero-output update')
        norm=float(torch.nn.utils.clip_grad_norm_(params,SETTINGS['clip_norm']))
        after=sum(float(p.grad.double().square().sum()) for p in params)**.5
        if not math.isfinite(norm) or after>SETTINGS['clip_norm']+1e-4:
            raise ValueError('Invalid paired gradient clipping')
        optimizer.step()
        # Count the executed update before validating the new state. If it is
        # nonfinite, preserve the actual budget even without a completed log.
        progress(dict(stage='optimizer_step',arm=arm,step=step))
        if any(not bool(torch.isfinite(p).all()) for p in params):
            raise ValueError('Nonfinite updated paired parameters')
        log=dict(stage='update',arm=arm,step=step,pair_indices=indices,
            point_loss=float(point.detach()),motion_unit_loss=float(change.detach()),
            motion_weighted_loss=float(change.detach())*SETTINGS['motion_weight'],
            motion_applied=arm=='point_motion',total_loss=float(loss.detach()),
            matched_axis_motion_parts=detail['parts'].detach().cpu().tolist(),
            predicted_log_increment=detail['predicted_log_increment'].detach().cpu().tolist(),
            gt_log_increment=detail['gt_log_increment'].cpu().tolist(),
            next_axes_exchanged=detail['swap'].cpu().tolist(),axis_ambiguous=detail['ambiguous'].cpu().tolist(),
            grad_norm_before_clip=norm,grad_norm_after_clip=after,
            clip_multiplier=min(1.,SETTINGS['clip_norm']/(norm+1e-6)),stem_grad_norm=stem,
            component_gradients=diagnostics)
        logs.append(log);progress(log)
    final=base.evaluate(head,records,device,formal)
    result=dict(initial_state=before,final_state=formal.g.state_digest(head),initial=start,
        final=final,logs=logs,completed_updates=len(logs),checkpoint_exported=False,
        motion_gradient_signal_seen=any(z['component_gradients'] is not None and
            z['component_gradients']['motion_unit_norm']>0 for z in logs))
    del head,optimizer
    if device.type=='cuda':torch.cuda.empty_cache()
    return result


def review(point,changed):
    verdict=base.review(point,changed)
    for domain in ('real','sim'):
        del verdict['checks'][domain+'/strict_short_mae_gain']
        gains=[]
        for scale in SETTINGS['scales']:
            key='probe/'+domain+'/'+str(scale)
            old=point['groups'][key]['midpoint'];new=changed['groups'][key]['midpoint']
            a,z=new['temporal']['long']['log_increment_error']['rmse'],old['temporal']['long']['log_increment_error']['rmse']
            verdict['checks'][key]['long/gt_relative_motion_rmse']=a is not None and z is not None and a<=z+base.TOL['motion']
            gains.append(all(new['temporal'][f]['log_increment_error']['rmse'] is not None and
                old['temporal'][f]['log_increment_error']['rmse'] is not None and
                new['temporal'][f]['log_increment_error']['rmse'] < old['temporal'][f]['log_increment_error']['rmse']-base.TOL['motion']
                for f in ('short','diagonal')))
        verdict['checks'][domain+'/strict_short_and_diagonal_motion_gain']=any(gains)
    verdict['diagnostic_conditions_met']=all(all(v.values()) if isinstance(v,dict) else v for v in verdict['checks'].values())
    verdict['note']='Limited paired TRAIN probe only. Numerical joint geometry/motion conditions, not VAL/generalization/significance/formal approval; DFR alone cannot pass.'
    return verdict


def publish(out):
    write_json(out/'artifacts.json',dict(protocol=VERSION,
        files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='artifacts.json'}))


def run(args):
    out=Path(args.out_dir)
    try:out.mkdir(parents=True,exist_ok=False)
    except FileExistsError:
        raise FileExistsError('Output directory already exists: %s. Existing results are preserved; use a new --out-dir suffix, including after a failed run.' % out) from None
    report=dict(protocol=VERSION,status='STARTED',settings=deepcopy(SETTINGS),
        detector_updates=0,detector_forward_calls=0,feature_extractions=0,head_updates_total=0,
        val_access=False,test_access=False,weight_bytes_read=False,cache_tensor_loads=0,
        checkpoint_exported=False,formal_training_approved=False,automatic_promotion=False,gpu_devices_used=[])
    started=time.perf_counter()
    with (out/'progress.jsonl').open('x') as stream:
        def progress(log):
            if log['stage']=='optimizer_step':report['head_updates_total']+=1
            stream.write(json.dumps(log,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
            if log['stage']=='update' and (log['step'] in (1,2) or log['step']%25==0):
                print('TRAIN',log['arm'],log['step'],'point',log['point_loss'],'motion_unit',log['motion_unit_loss'],'total',log['total_loss'],flush=True)
        try:
            protocol,cache,proof=checked_contract(args.training_dir,args.cache_dir)
            report['proof']=proof
            write_json(out/'protocol.json',protocol);write_json(out/'sources.json',read_json(SOURCES))
            print('Finite TRAIN motion check:128 identities/256 views;4pairs=8frames/batch;200updates/arm;B inference/updates=0;no weight export. Not formal training.',flush=True)
            if args.check_only:
                report['status']='STATIC_MOTION_CONTRACT_PASS_NO_TENSOR_LOAD_NO_GPU_NO_UPDATES'
            else:
                torch,formal,motion=runtime_modules()
                if (motion.MOTION_WEIGHT!=SETTINGS['motion_weight'] or motion.MOTION_BETA!=SETTINGS['motion_beta'] or motion.AXIS_TIE_TOLERANCE!=SETTINGS['axis_tie_tolerance']):
                    raise ValueError('Fixed motion constants differ')
                device,runtime=base.execution_device(args.gpu,formal,torch)
                report.update(runtime=runtime,gpu_devices_used=[args.gpu])
                if runtime!=cache['runtime']:raise ValueError('Original cache/runtime versions or GPU differ')
                def loaded(name):
                    report['cache_tensor_loads']+=1;progress(dict(stage='train_tensor_loaded',file=name))
                full=base.load_train_cache(args.cache_dir,cache,formal,torch,loaded)
                records,blocks=base.select_blocks(full);del full
                counts={role:dict(Counter(r['domain'] for r in records if r['role']==role and r['scale']==1.)) for role in ('fit','probe')}
                if counts!={'fit':{'real':48,'sim':48},'probe':{'real':16,'sim':16}} or len(records)!=256:
                    raise ValueError('Fixed TRAIN sampling budget differs')
                catalog=pair_catalog(records);batches=schedule(catalog)
                audit=correspondence_audit(records,catalog,formal,motion,torch)
                report.update(sample_counts=counts,selected_cpu_tensor_bytes=formal.tensor_bytes(records),
                    inherited_fit_ineligible=sum(r['role']=='fit' and not r['eligible'] for r in records),
                    pair_support=catalog['support'],pair_minibatches=batches,
                    pair_count=len(catalog['pairs']),eligible_fit_pair_count=sum(p['eligible_for_gradient'] for p in catalog['pairs']),
                    gt_axis_transport=dict(exchanged=sum(r.get('next_axes_exchanged',False) for r in audit),
                        ambiguous=sum(r.get('ambiguous',False) for r in audit)),
                    batches_with_repeated_frame_endpoints=sum(len({i for j in bs for i in catalog['pairs'][j]['indices']})<8 for bs in batches))
                write_json(out/'sample_selection.json',dict(blocks=blocks,views=[{k:r[k] for k in ('image','sequence','domain','frame_id','role','scale','eligible')} for r in records]))
                write_json(out/'pair_catalog.json',catalog);write_json(out/'axis_correspondence.json',audit)
                formal.seed_all();initial_head=formal.m.SpatialMidpointHead()
                initial=deepcopy(initial_head.state_dict());del initial_head
                results={}
                for arm in ARMS:
                    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(args.gpu)
                    result=fit_arm(records,catalog,batches,initial,arm,device,formal,motion,torch,progress)
                    if device.type=='cuda':
                        result['cuda_peak_bytes']=dict(allocated=torch.cuda.max_memory_allocated(args.gpu),reserved=torch.cuda.max_memory_reserved(args.gpu))
                    write_json(out/(arm+'.json'),dict(protocol=VERSION,proof=proof,arm=arm,result=result));results[arm]=result
                if (results[ARMS[0]]['initial']!=results[ARMS[1]]['initial'] or
                        results[ARMS[0]]['initial_state']!=results[ARMS[1]]['initial_state'] or
                        report['head_updates_total']!=2*SETTINGS['steps_per_arm']):
                    raise ValueError('Paired initialization/update budget differs')
                verdict=review(results['point_only']['final'],results['point_motion']['final'])
                verdict['checks']['measured_motion_gradient_signal_both_arms']=all(v['motion_gradient_signal_seen'] for v in results.values())
                verdict['diagnostic_conditions_met']=verdict['diagnostic_conditions_met'] and verdict['checks']['measured_motion_gradient_signal_both_arms']
                write_json(out/'review.json',verdict)
                report.update(status='TRAIN_MOTION_SHORT_FIT_COMPLETE_REVIEW_REQUIRED',review=verdict,
                    arm_states={k:dict(initial=v['initial_state'],final=v['final_state'],updates=v['completed_updates']) for k,v in results.items()})
            report['elapsed_seconds']=time.perf_counter()-started;progress(dict(stage='complete',status=report['status']))
        except Exception as error:
            report.update(status='FAILED_MOTION_PREFLIGHT_REVIEW_REQUIRED',error=type(error).__name__+': '+str(error),elapsed_seconds=time.perf_counter()-started)
            progress(dict(stage='failed',error=report['error']));write_json(out/'completion.json',report);stream.flush();publish(out);raise
        write_json(out/'completion.json',report)
    publish(out)
    print('Saved',out/'completion.json','status',report['status'],flush=True)
    return report


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-only',action='store_true');p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--training-dir',default='work_dirs/crane_symeood_k1_port_day2night_midpoint_formal_v1')
    p.add_argument('--cache-dir',default='work_dirs/port_geometry_midpoint_formal_v1_roi_cache')
    p.add_argument('--out-dir',required=True)
    return p


if __name__=='__main__':run(parser().parse_args())
