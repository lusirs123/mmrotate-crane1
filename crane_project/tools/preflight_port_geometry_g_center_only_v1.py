#!/usr/bin/env python3
"""Fixed independent-center TRAIN check, not formal detector training.

Read saved B/joint results and reviewed CPU ROIs; only center-only head updates
200 times. No joint replay, fitted checkpoint, VAL/TEST or parameter search.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import preflight_port_geometry_g_center_v1 as c
from crane_project.utils.port_geometry_refine_g_center_only_v1 import (
    CENTER_SETTINGS, PARAMETER_COUNT, TRAINABLE_PARAMETER_COUNT, PART_NAMES,
    CenterOnlyLocalGeometryRefiner, regression_loss)

g = c.g
VERSION = 'port_geometry_g_center_only_v1_train_preflight'
STEPS = 200
PROTOCOL = ROOT/'crane_project/tools/port_geometry_g_center_only_v1_protocol.json'
MANIFEST = ROOT/'crane_project/tools/port_geometry_g_center_only_v1_sources.json'
C_MANIFEST_SHA = '5ff6be9e37e0b3fb51dca2bab55f169853532270326929fd2b4a3c424e6b9ff3'
JOINT_REPORT_SHA = '7adb734eae56d703584f2a1a0d025465c07bf80b6ddb18612df0cdb07bf10452'
JOINT_REPORT = ROOT/'work_dirs/port_geometry_g_center_v1_train/completion.json'
# Numerical identity tolerance, not a post-result performance allowance.
TOLERANCE = dict(center_px=1e-6, riou=1e-5)


def protocol_document(samples):
    return dict(protocol=VERSION, evidence_role='finite_TRAIN_center_task_intervention',
        reference_g_settings=deepcopy(g.SETTINGS), center_settings=deepcopy(CENTER_SETTINGS),
        conditions=['b_identity', 'saved_joint_center_b_shape', 'independent_center_b_shape'],
        parameter_counts=dict(total=PARAMETER_COUNT, trainable=TRAINABLE_PARAMETER_COUNT,
                              frozen_bypassed_shape=PARAMETER_COUNT-TRAINABLE_PARAMETER_COUNT),
        baseline_report_sha256=c.a.BASELINE_SHA, joint_report_sha256=JOINT_REPORT_SHA,
        cache_manifest_sha256=c.a.CACHE_MANIFEST_SHA, prior_center_manifest_sha256=C_MANIFEST_SHA,
        fixed_sample_counts={role:dict(Counter(s['domain'] for s in samples if s['role']==role)) for role in ('fit','probe')},
        intervention='Remove shape supervision from shared stem/FC64 updates. Keep original shape output tensors frozen and bypassed; copy B raw w/h/angle/score exactly.',
        initialization='Copy exact reviewed untrained joint state, including zero xy and frozen zero shape tensors. Same shared initialization and two center units; require full digest equal saved joint initial.',
        center_definition='Same .30*S_B radial xy squash, original coordinates and detached ORIGINAL B short edge. No clipping/rejection/GT-selected bound.',
        loss='Only(p_x+p_y)/3; each raw SmoothL1(beta.1) part uses errors=(c_new-c_GT)/(.30*S_B). No shape loss, constant loss gain or mean-of-two reweighting.',
        optimizer='Same seed1703 Adam lr.001 wd0 clip10, identical200 fit-only balanced batches8(4real+4sim). Initial/final evaluation; all200 updates logged, no intermediate selection.',
        scope=dict(detector_forward_calls=0, detector_updates=0, additional_joint_updates=0,
            new_head_updates=STEPS, formal_training=False, checkpoint_export=False,
            val_access=False, test_access=False),
        evaluation='All128 views by fit/probe/domain/scale. Center mean/RMSE/p90, signed xy residual and desired/actual movement, RIoU mean/p10/min and paired tails. Explicit output/conditional-correct/all-frame-correct fractions. Shapes/angles exact B are constraints, not gains.',
        tolerance=deepcopy(TOLERANCE),
        review='All4 probe center mean/RMSE/p90 and RIoU mean/p10/min non-regressing vs B; no new center or overlap tail failures. Both real fit center mean/RMSE non-regressing. At least one real probe jointly improves center mean and RIoU mean vs B AND saved joint-center hybrid beyond numerical tolerance. Diagnostic continuation only, never formal approval.',
        runtime_policy='Same python/torch/cuda/cudnn/numpy/opencv strings as reviewed baseline and joint; physical GPU number may differ.',
        limitations=['Repeated TRAIN probe16 identities/32 related views, all seen by B; not independent VAL.',
            'Task interaction is a hypothesis; no per-task shared gradient cosine evidence or PCGrad intervention.',
            'Removing shape gradients changes optimization, not proof of a unique root or final five-parameter method.',
            'Single-frame sparse checks cannot guarantee real video continuity or depth accuracy.',
            'TEST repeatedly exposed; no TEST tuning, checkpoint selection or access in this run.',
            'Failure ends incremental same-cache P3+FC64 trials; no weight/capacity/bound/step sweep.'])


def summarize_rows(rows):
    groups = {}
    for role in ('fit','probe'):
        for domain in ('real','sim'):
            for scale in g.SETTINGS['scales']:
                key = role+'/'+domain+'/'+str(scale)
                subset = [r for r in rows if r['role']==role and r['domain']==domain and r['scale']==scale]
                summary = g.summarize(subset, continuous=False)
                values = [r['metrics']['riou'] for r in subset]
                hits = sum(r['metrics']['center_hit'] for r in subset)
                summary.update(riou_p10=float(np.percentile(values,10)) if values else None,
                    riou_min=min(values) if values else None, center_correct_frames=hits,
                    output_coverage_fraction=dict(numerator=summary['output_frames'],denominator=len(subset)),
                    conditional_center_correct_fraction=dict(numerator=hits,denominator=summary['output_frames']),
                    all_frame_center_correct_fraction=dict(numerator=hits,denominator=len(subset)))
                groups[key] = summary
    return groups


def saved_joint_center_b_shape(baseline, joint):
    b_rows = baseline['arms']['ordinary']['initial']['rows']
    j_rows = joint['joint_center']['final']['rows']
    # Validate paired metadata/outputs first, before component substitution.
    c.paired_changes(dict(rows=b_rows), dict(rows=j_rows))
    rows = deepcopy(b_rows)
    for row, learned in zip(rows,j_rows):
        if row['pred'] is not None:
            row['pred'][:2] = learned['pred'][:2]
        row['metrics'] = g.decompose(row['gt_original'],row['pred'][:5] if row['pred'] is not None else None)
    return dict(rows=rows, groups=summarize_rows(rows),
        evidence_role='saved_prediction_hybrid_not_separately_trained',
        source_sha256=JOINT_REPORT_SHA, additional_updates=0)


def checked_contract(baseline_path, joint_path):
    manifest, protocol = [json.loads(p.read_text()) for p in (MANIFEST,PROTOCOL)]
    if (manifest.get('protocol')!=VERSION or manifest.get('reference_g_settings')!=g.SETTINGS
            or manifest.get('center_settings')!=CENTER_SETTINGS):
        raise ValueError('Center-only source/settings contract differs')
    actual = {p:g.ready.sha(ROOT/p) for p in manifest['sources']}
    if actual!=manifest['sources']:
        raise ValueError('Reviewed center-only source SHA differs')
    if g.ready.sha(c.MANIFEST)!=C_MANIFEST_SHA:
        raise ValueError('Requires unchanged reviewed joint provenance')
    _, samples, center_identity, baseline = c.checked_contract(baseline_path)
    if protocol!=protocol_document(samples):
        raise ValueError('Predeclared center-only protocol differs')
    if g.ready.sha(joint_path)!=JOINT_REPORT_SHA:
        raise ValueError('Requires exact reviewed joint completion report')
    joint = json.loads(joint_path.read_text())
    if (joint['protocol']!=c.VERSION or joint['identity']!=center_identity
            or joint['status']!='TRAIN_CENTER_CHECK_COMPLETE_REVIEW_REQUIRED'
            or joint['head_updates_total']!=STEPS or joint['additional_reference_updates']!=0
            or joint['detector_forward_calls']!=0 or joint['detector_updates']!=0
            or joint['formal_training'] or joint['checkpoint_exported']
            or not joint['detector_state_unchanged'] or joint['minibatches']!=baseline['minibatches']
            or joint['joint_center']['initial_state']!=joint['initialization']['joint_initial_state']):
        raise ValueError('Reviewed joint identity/scope/schedule differs')
    if any(joint['runtime'][k]!=baseline['runtime'][k]
           for k in ('python','torch','cuda','cudnn','numpy','opencv')):
        raise ValueError('Joint/baseline runtime differs')
    c.verify_initial(joint['joint_center']['initial'],baseline['arms']['ordinary']['initial'])
    hybrid = saved_joint_center_b_shape(baseline,joint)
    identity = dict(manifest_sha256=g.ready.sha(MANIFEST),protocol_sha256=g.ready.sha(PROTOCOL),
        sources=actual, prior_identity=center_identity, joint_report_sha256=JOINT_REPORT_SHA)
    return protocol,samples,identity,baseline,joint,hybrid


def checked_initialization(baseline, joint):
    initial, evidence = c.checked_initialization(baseline['arms']['ordinary'])
    head = CenterOnlyLocalGeometryRefiner(); head.load_state_dict(initial,strict=True)
    digest = g.state_digest(head)
    if (digest!=joint['joint_center']['initial_state'] or
            sum(p.numel() for p in head.parameters() if p.requires_grad)!=TRAINABLE_PARAMETER_COUNT
            or any(p.requires_grad for p in head.output.parameters())):
        raise ValueError('Center-only initialization/trainable partition differs from joint')
    return initial, dict(shared_joint_state_exact=True, initial_state=digest,
        shape_tensors_frozen_and_bypassed=True, trainable_parameter_count=TRAINABLE_PARAMETER_COUNT,
        reference_initialization=evidence)


def validate_prediction(output, b):
    c.validate_prediction(output,b)
    if not torch.equal(output['boxes_original'][:,2:],b[:,2:]):
        raise ValueError('Center-only changed raw B size/angle/score')


def evaluate(head,records,device):
    rows, losses, components = [], {}, {}
    saturated, radial = [0,0], 0
    head.eval()
    with torch.no_grad():
        for i,r in enumerate(records):
            roi,support,b,bm,gt = g.batch_tensors(records,[i],'ordinary',device)
            output = head(roi,support,b,bm); validate_prediction(output,b)
            pred = output['boxes_original'].cpu().numpy()
            row = {k:r[k] for k in ('image','domain','sequence','frame_id','role','scale','eligible')}
            row.update(pred=pred[0].tolist() if len(pred) else None,gt_original=gt[0].cpu().tolist(),
                parsed_gt_original=r.get('parsed_gt_original'),
                reference_gt_absolute_delta=r.get('reference_gt_absolute_delta'),
                metrics=g.decompose(gt[0].cpu().numpy(),pred[0,:5] if len(pred) else None))
            rows.append(row)
            if r['eligible']:
                loss,parts = regression_loss(output['boxes_original'],gt,b)
                key = r['role']+'/'+r['domain']+'/'+str(r['scale'])
                losses.setdefault(key,[]).append(float(loss)); components.setdefault(key,[]).append(parts.cpu().tolist())
            near = output['center_residual'].abs()>=.95*CENTER_SETTINGS['max_center_over_b_short']
            saturated = [v+int(n) for v,n in zip(saturated,near.sum(dim=0))]
            radial += int((output['center_residual'].norm(dim=1)>=.95*CENTER_SETTINGS['max_center_over_b_short']).sum())
    return dict(rows=rows,groups=summarize_rows(rows),
        eligible_center_loss={k:float(np.mean(v)) for k,v in losses.items()},
        eligible_raw_center_parts={k:np.mean(v,axis=0).tolist() for k,v in components.items()},
        center_axis_saturation_counts=saturated,center_radial_saturation_views=radial,
        raw_b_shape_score_exact=True, shape_loss_optimized=False)


def gradient_check(records,indices,initial,device):
    head = CenterOnlyLocalGeometryRefiner().to(device); head.load_state_dict(initial)
    joint = c.CenterLocalGeometryRefiner().to(device); joint.load_state_dict(initial)
    roi,support,b,bm,gt = g.batch_tensors(records,indices,'ordinary',device)
    output = head(roi,support,b,bm); validate_prediction(output,b)
    other = joint(roi,support,b,bm)
    loss,parts = regression_loss(output['boxes_original'],gt,b)
    _, jp = c.regression_loss(other['boxes_original'],gt,b)
    names,params = zip(*[(k,p) for k,p in head.named_parameters() if p.requires_grad])
    oldparams = dict(joint.named_parameters())
    hg = torch.autograd.grad(loss,params,retain_graph=True)
    jg = torch.autograd.grad(jp[3:].sum()/3.,tuple(oldparams[k] for k in names))
    if (not torch.equal(output['boxes_original'],b) or not torch.equal(parts,jp[3:])
            or any(not torch.equal(x,y) for x,y in zip(hg,jg))):
        raise ValueError('Initial center task or common gradients differ from joint center task')
    tasks = [float(torch.autograd.grad(p/3.,head.center_output.weight,retain_graph=True)[0].double().norm()) for p in parts]
    loss.backward()
    if (not all(v>0 for v in tasks) or
            any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params)
            or any(p.grad is not None for p in head.output.parameters())):
        raise ValueError('Missing/nonfinite center gradient or shape gradient leakage')
    return dict(no_optimizer_updates=True,initial_center_parts_exact=True,
        initial_shared_center_gradients_exact=True,weighted_center_task_grad_norms=tasks,
        both_center_tasks_effective=True,frozen_shape_gradients_absent=True,
        initial_center_loss=float(loss))


def fit_center_only(records,batches,initial,device,progress):
    head = CenterOnlyLocalGeometryRefiner().to(device); head.load_state_dict(initial,strict=True)
    frozen_before = {k:v.detach().clone() for k,v in head.output.state_dict().items()}
    before = g.state_digest(head); start = evaluate(head,records,device)
    if any(row['pred']!=(r['boxes_original'][0].tolist() if len(r['boxes_original']) else None)
           for row,r in zip(start['rows'],records)):
        raise ValueError('Initial center-only outputs are not exact B')
    params = [p for p in head.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params,lr=g.SETTINGS['lr'],weight_decay=0.)
    logs = []; head.train()
    for step,indices in enumerate(batches,1):
        roi,support,b,bm,gt = g.batch_tensors(records,indices,'ordinary',device)
        optimizer.zero_grad(); output = head(roi,support,b,bm); validate_prediction(output,b)
        loss,parts = regression_loss(output['boxes_original'],gt,b)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite center-only loss')
        tasks = ([float(torch.autograd.grad(p/3.,head.center_output.weight,retain_graph=True)[0].double().norm()) for p in parts]
                 if step==1 else None)
        loss.backward()
        if (any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params)
                or any(p.grad is not None for p in head.output.parameters())):
            raise ValueError('Missing/nonfinite center gradient or shape gradient leakage')
        stem_norm = sum(float(p.grad.double().square().sum()) for p in head.stem.parameters())**.5
        norm = float(torch.nn.utils.clip_grad_norm_(params,g.SETTINGS['clip_norm']))
        after = sum(float(p.grad.double().square().sum()) for p in params)**.5
        if not math.isfinite(norm) or not math.isfinite(after) or after>g.SETTINGS['clip_norm']+1e-4:
            raise ValueError('Invalid center-only gradient clipping')
        optimizer.step(); progress(dict(stage='update',step=step))
        if any(not bool(torch.isfinite(p).all()) for p in head.parameters()):
            raise ValueError('Nonfinite updated head')
        log = dict(stage='fit',arm='center_only',step=step,center_loss=float(loss.detach()),
            raw_center_parts=parts.detach().cpu().tolist(),weighted_part_coefficients=[1./3.]*2,
            grad_norm_before_clip=norm,grad_norm_after_clip=after,
            clip_multiplier=min(1.,g.SETTINGS['clip_norm']/(norm+1e-6)),
            stem_grad_norm=stem_norm,weighted_final_layer_task_grad_norms=tasks,
            frozen_shape_gradients_absent=True)
        logs.append(log); progress(log)
    if any(not torch.equal(frozen_before[k],v) for k,v in head.output.state_dict().items()):
        raise ValueError('Frozen shape output tensors changed')
    final = evaluate(head,records,device)
    result = dict(initial_state=before,final_state=g.state_digest(head),initial=start,final=final,
        logs=logs,finite_gradients=True,complete_update_trajectory=True,loss_part_names=PART_NAMES,
        frozen_shape_state_unchanged=True,
        initial_both_center_tasks_effective=all(v>0 for v in logs[0]['weighted_final_layer_task_grad_norms']),
        stem_gradient_after_zero_output_step_effective=len(logs)>1 and logs[1]['stem_grad_norm']>0,
        fit_center_loss_decreased_by_group={k:final['eligible_center_loss'][k]<v-1e-6
            for k,v in start['eligible_center_loss'].items() if k.startswith('fit/')},
        clipping=dict(records=len(logs),clipped_updates=sum(v['clip_multiplier']<1 for v in logs),
            before_norm=g.describe([v['grad_norm_before_clip'] for v in logs]),
            max_before_norm=max(v['grad_norm_before_clip'] for v in logs)))
    del head,optimizer
    if device.type=='cuda': torch.cuda.empty_cache()
    return result


def verify_initial(result,baseline,joint):
    old = baseline['arms']['ordinary']['initial']
    if len(result['rows'])!=len(old['rows']):
        raise ValueError('Initial B view count differs')
    for row,ref in zip(result['rows'],old['rows']):
        if any(row[k]!=ref[k] for k in ('image','role','domain','sequence','frame_id','scale','eligible','pred','gt_original')):
            raise ValueError('Initial B/GT view identity differs')
    if result['eligible_center_loss']!=joint['joint_center']['initial']['eligible_center_loss']:
        raise ValueError('Initial center loss differs from saved joint')


def motion_report(reference,current):
    # Pair validation also prevents silently mixing a scale/image or absent output.
    comparison = c.paired_changes(reference,current)
    groups = {}
    for old,new in zip(reference['rows'],current['rows']):
        key = old['role']+'/'+old['domain']+'/'+str(old['scale'])
        rows = groups.setdefault(key,[])
        if old['pred'] is None: continue
        b,p,gt = [np.asarray(v,dtype=float) for v in (old['pred'],new['pred'],old['gt_original'])]
        short = min(b[2:4]); desired,actual = gt[:2]-b[:2],p[:2]-b[:2]
        rows.append(dict(image=old['image'],desired_shift_px=desired.tolist(),actual_shift_px=actual.tolist(),
            desired_shift_over_b_short=(desired/short).tolist(),actual_shift_over_b_short=(actual/short).tolist(),
            residual_before_px=(b[:2]-gt[:2]).tolist(),residual_after_px=(p[:2]-gt[:2]).tolist(),
            center_better=new['metrics']['center_error_px']<old['metrics']['center_error_px']-TOLERANCE['center_px'],
            center_worse=new['metrics']['center_error_px']>old['metrics']['center_error_px']+TOLERANCE['center_px'],
            riou_better=new['metrics']['riou']>old['metrics']['riou']+TOLERANCE['riou'],
            riou_worse=new['metrics']['riou']<old['metrics']['riou']-TOLERANCE['riou']))
    return dict(reference='original frozen B',comparison=comparison,groups={key:dict(
        output_views=len(rows),rows=rows,
        summaries={field:{axis:g.describe([r[field][i] for r in rows]) for i,axis in enumerate(('x','y'))}
            for field in ('desired_shift_px','actual_shift_px','desired_shift_over_b_short',
                          'actual_shift_over_b_short','residual_before_px','residual_after_px')},
        sign_counts={field:{axis:dict(negative=sum(r[field][i]<0 for r in rows),
            positive=sum(r[field][i]>0 for r in rows),zero=sum(r[field][i]==0 for r in rows))
            for i,axis in enumerate(('x','y'))} for field in ('desired_shift_px','actual_shift_px')},
        paired_counts={name:sum(r[name] for r in rows) for name in ('center_better','center_worse','riou_better','riou_worse')}
        ) for key,rows in groups.items()})


def review(comparison,against_joint):
    expected = {role+'/'+domain+'/'+str(scale) for role in ('fit','probe')
                for domain in ('real','sim') for scale in g.SETTINGS['scales']}
    if set(comparison['groups'])!=expected or set(against_joint['groups'])!=expected:
        raise ValueError('Expected all8 fixed evaluation groups')
    groups, real_fit, gains = {}, {}, []
    for key,value in comparison['groups'].items():
        d,e = value['delta'],value['events']
        if key.startswith('fit/real/'):
            real_fit[key] = {stat:d['center_error_px/'+stat] is not None and d['center_error_px/'+stat]<=TOLERANCE['center_px']
                            for stat in ('mean','rmse')}
        if not key.startswith('probe/'): continue
        checks = {'center_'+stat+'_not_worse':d['center_error_px/'+stat] is not None and d['center_error_px/'+stat]<=TOLERANCE['center_px']
                  for stat in ('mean','rmse','p90')}
        checks.update({name+'_not_worse':d[field] is not None and d[field]>=-TOLERANCE['riou']
            for name,field in [('riou_mean','all_frame_mean_riou'),('riou_p10','riou_p10'),('riou_min','riou_min')]})
        checks.update(output_coverage_exact=d['output_coverage_pct']==0,
            conditional_center_correct_not_worse=d['output_center_hit_pct'] is not None and d['output_center_hit_pct']>=0,
            all_frame_center_correct_not_worse=d['all_frame_center_hit_pct'] is not None and d['all_frame_center_hit_pct']>=0,
            no_new_center_failures=not e['new_center_failures'],no_new_riou_below_0_5=not e['new_riou_below_0_5'],
            no_new_zero_riou_outputs=not e['new_zero_riou_outputs'])
        jd = against_joint['groups'][key]['delta']
        gain = (key.startswith('probe/real/') and
            d['center_error_px/mean'] is not None and jd['center_error_px/mean'] is not None and
            d['center_error_px/mean']<-TOLERANCE['center_px'] and d['all_frame_mean_riou']>TOLERANCE['riou'] and
            jd['center_error_px/mean']<-TOLERANCE['center_px'] and jd['all_frame_mean_riou']>TOLERANCE['riou'])
        if gain: gains.append(key)
        groups[key] = dict(checks=checks,all_non_regression_checks_met=all(checks.values()),
                           real_center_and_overlap_gain_over_b_and_joint=gain)
    nonregression = all(v['all_non_regression_checks_met'] for v in groups.values())
    fit_ok = all(all(v.values()) for v in real_fit.values())
    return dict(groups=groups,real_fit_center_checks=real_fit,
        all_probe_non_regression_checks_met=nonregression,real_fit_center_non_regression_met=fit_ok,
        real_gain_groups=gains,diagnostic_continuation_conditions_met=nonregression and fit_ok and bool(gains),
        formal_training_approved=False,
        note='Fixed finite TRAIN criterion. Exact B shape is not a gain; no statistical/causal root claim or automatic formal/VAL/TEST approval.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check-only',action='store_true',help='Source/TRAIN/saved report checks only; no ROI cache load, GPU, head or updates.')
    ap.add_argument('--gpu',type=int,default=0)
    ap.add_argument('--baseline-report',default=str(c.a.BASELINE))
    ap.add_argument('--joint-report',default=str(JOINT_REPORT))
    ap.add_argument('--cache-dir',default='work_dirs/port_geometry_g_v1_roi_cache')
    ap.add_argument('--out-dir',required=True)
    args = ap.parse_args(); os.chdir(ROOT)
    out = Path(args.out_dir).resolve(); out.mkdir(parents=True,exist_ok=False)
    report = dict(protocol=VERSION,evidence_role='finite_TRAIN_center_task_intervention',status='STARTED',
        formal_training=False,detector_forward_calls=0,detector_updates=0,additional_joint_updates=0,
        head_updates_total=0,checkpoint_exported=False,gpu_devices_used=0)
    with (out/'progress.jsonl').open('x') as stream:
        def progress(value):
            if value['stage']=='update': report['head_updates_total']+=1; return
            stream.write(json.dumps(g.json_native(value),ensure_ascii=False,allow_nan=False)+'\n'); stream.flush()
            if value['step'] in (1,2) or value['step']%25==0:
                print('TRAIN center_only step',value['step'],'center',value['center_loss'],flush=True)
        try:
            protocol,samples,identity,baseline,joint,hybrid = checked_contract(
                Path(args.baseline_report).resolve(),Path(args.joint_report).resolve())
            report.update(identity=identity,reference_g_settings=g.SETTINGS,center_settings=CENTER_SETTINGS,
                parameter_counts=protocol['parameter_counts'],fixed_sample_counts=protocol['fixed_sample_counts'],
                limitations=protocol['limitations'],tolerance=TOLERANCE,
                saved_joint_center_b_shape=hybrid,
                fit_center_support=c.fit_center_support(baseline['arms']['ordinary']['initial']['rows']))
            g.write_new(out/'protocol.json',protocol); shutil.copyfile(str(MANIFEST),str(out/'sources.json'))
            print('G center-only check: new head updates='+str(STEPS)+'; saved joint/B updates=0; B inference=0. Not formal training.',flush=True)
            if args.check_only:
                report['status']='STATIC_CENTER_ONLY_CONTRACT_COMPLETE_NO_CACHE_LOAD_NO_GPU_NO_UPDATES'
            else:
                if int(os.environ.get('WORLD_SIZE','1'))!=1:
                    raise ValueError('Single-process finite check; do not use torchrun/DDP')
                report['runtime'] = c.a.checked_runtime(baseline)
                cache_dir = Path(args.cache_dir).resolve()
                if g.ready.sha(cache_dir/'cache_manifest.json')!=c.a.CACHE_MANIFEST_SHA:
                    raise ValueError('Requires exact reviewed complete ROI cache')
                payload,cache = g.checked_cache(cache_dir,identity['prior_identity']['prior_identity']['g_identity'],samples)
                records = payload['records']; g.validate_records(records,samples)
                c.a.check_cache_reference(records,cache,baseline)
                if any(len(r['boxes_original'])!=1 or not r['eligible'] for r in records):
                    raise ValueError('Expected all128 reviewed eligible B outputs')
                g.write_new(out/'cache_manifest.json',cache)
                report.update(cache=cache,cache_reused=True,detector_state_unchanged=True,
                    detector_evidence='Reviewed cached B digests verified; no detector constructed.',
                    original_cpu_roi_bytes=g.cpu_roi_bytes(records))
                if not torch.cuda.is_available() or not 0<=args.gpu<torch.cuda.device_count():
                    raise ValueError('A valid logical CUDA device is required')
                torch.cuda.set_device(args.gpu); device = torch.device('cuda',args.gpu)
                report['gpu_devices_used']=1; report['runtime']['gpu']=torch.cuda.get_device_name(args.gpu)
                initial,initialization = checked_initialization(baseline,joint)
                batches = g.schedule(records)
                if len(batches)!=STEPS or batches!=baseline['minibatches'] or batches!=joint['minibatches']:
                    raise ValueError('Fixed200 reference batches differ')
                report.update(initialization=initialization,minibatches=batches)
                print('Reusing reviewed CPU ordinary ROIs. One logical GPU:',args.gpu,'; only independent center head fitted.',flush=True)
                def operation():
                    report['gradient_contract']=gradient_check(records,batches[0],initial,device)
                    return fit_center_only(records,batches,initial,device,progress)
                result,cost = g.measured(operation,args.gpu)
                report.update(center_only=result,refiner_cost=cost)
                # Publish completed head result before review can fail.
                g.write_new(out/'center_only.json',dict(protocol=VERSION,identity=identity,result=result,cost=cost,
                    completed_head_updates=STEPS,formal_training=False,checkpoint_exported=False))
                verify_initial(result['initial'],baseline,joint)
                if report['head_updates_total']!=STEPS or result['initial_state']!=initialization['initial_state']:
                    raise ValueError('Update budget or initial state differs')
                b = baseline['arms']['ordinary']['initial']
                report.update(center_only_minus_b=c.paired_changes(b,result['final']),
                    center_only_minus_joint_center=c.paired_changes(hybrid,result['final']),
                    center_only_motion=motion_report(b,result['final']),saved_joint_motion=motion_report(b,hybrid))
                report['probe_review']=review(report['center_only_minus_b'],report['center_only_minus_joint_center'])
                report['status']='TRAIN_CENTER_ONLY_CHECK_COMPLETE_REVIEW_REQUIRED'
            g.write_new(out/'completion.json',report)
        except Exception as error:
            report['status']='FAILED_REVIEW_REQUIRED'; report['error']=type(error).__name__+': '+str(error)
            destination=out/('failure.json' if (out/'completion.json').exists() else 'completion.json')
            try: g.write_new(destination,report)
            except (TypeError,ValueError) as serialization_error:
                g.write_new(destination,dict(protocol=VERSION,status=report['status'],formal_training=False,
                    detector_updates=0,detector_forward_calls=0,additional_joint_updates=0,
                    head_updates_total=report['head_updates_total'],error=report['error'],
                    serialization_error=str(serialization_error),full_report_available=False))
            stream.flush(); publish_artifacts(out,report); raise
    publish_artifacts(out,report)
    print('Saved',out/'completion.json','status',report['status'],flush=True)


def publish_artifacts(out,report):
    g.write_new(out/'artifacts.json',dict(protocol=VERSION,status=report['status'],
        files={str(p.relative_to(out)):g.ready.sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifacts.json'}))


if __name__=='__main__': main()
