#!/usr/bin/env python3
"""Finite TRAIN center/size/angle check against saved fixed-center FC64.

Reuse the reviewed CPU ordinary ROI cache. Fit only the new head for200 steps;
detector/reference updates0. No checkpoint, VAL/TEST, bound/step/seed search.
Static mode validates provenance and fit-only center targets without cache/GPU.
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
from crane_project.tools import preflight_port_geometry_g_roi_ablation_v1 as a
from crane_project.utils.port_geometry_refine_g_center_v1 import (
    CENTER_SETTINGS, PART_NAMES, PARAMETER_COUNT, CenterLocalGeometryRefiner,
    initial_state_from_reference, regression_loss)

g = a.g
VERSION = 'port_geometry_g_center_v1_train_preflight'
STEPS = 200
PROTOCOL = ROOT/'crane_project/tools/port_geometry_g_center_v1_protocol.json'
MANIFEST = ROOT/'crane_project/tools/port_geometry_g_center_v1_sources.json'
A_MANIFEST_SHA = '0250eafb9feef5fff88a4072dca02b1a0cb71f345f6fa1e0b866c2d8a2ada170'


def protocol_document(samples):
    return dict(protocol=VERSION, evidence_role='finite_TRAIN_bounded_center_intervention',
        reference_g_settings=deepcopy(g.SETTINGS), center_settings=deepcopy(CENTER_SETTINGS),
        conditions=['b_identity', 'fixed_center_fc64_saved', 'joint_center_fc64'],
        parameter_counts=dict(fixed_center=183907, joint_center=PARAMETER_COUNT),
        baseline_report_sha256=a.BASELINE_SHA, cache_manifest_sha256=a.CACHE_MANIFEST_SHA,
        prior_manifest_sha256=A_MANIFEST_SHA,
        fixed_sample_counts={role:dict(Counter(s['domain'] for s in samples if s['role']==role)) for role in ('fit','probe')},
        intervention='Same ordinary ROI/stem/FC64 and descriptor. Add two zero xy units; no center input descriptor, learned ROI movement, new sampling, classification or assignment.',
        center_definition='c_new=c_B+.3*S_B*z_xy/sqrt(1+sum(z_xy**2)); radial movement<.3*S_B in original image xy. S_B=min(original B raw w,h), detached. No GT target clipping or out-of-bound sample removal.',
        bound_evidence='Only96 saved fit TRAIN views inspected before fixing. Largest required radial shift/S_B=.27409; .30 is a conservative finite-check design, not a selected optimum. Probe/VAL/TEST not used to choose bounds.',
        loss='Keep original mean of3 normalized SmoothL1(beta.1) parts EXACTLY. Add(p_x+p_y)/3, errors=(c_new-c_GT)/(.3*S_B), beta.1. Each of5 raw component means has coefficient1/3; do not average5 or use corrected edges for center normalization.',
        initialization='Seed1703 untrained FC64 digest must equal reviewed reference; copy every shared tensor, add zero xy output tensors. Initial six-column boxes exact B.',
        optimizer='Original Adam lr.001 weight_decay0 clip10, identical200 balanced batches8 (4real+4sim), fit only. All200 clipping/loss snapshots recorded; initial/final only evaluated.',
        runtime_policy='Same python/torch/cuda/cudnn/numpy/opencv version strings as saved reference; physical GPU number may differ.',
        scope=dict(detector_forward_calls=0, detector_updates=0, additional_reference_updates=0,
                   new_head_updates=STEPS, formal_training=False, checkpoint_export=False,
                   val_access=False, test_access=False),
        evaluation='All128 views retained. Separate fit/probe/domain/scale, original center mean/RMSE/p90, relative center error, edges mean/p90, pure angle RMSE/p90, strict RIoU mean/p10/min, output coverage, conditional center hits<15px, all-frame center correctness. Protocol angle10px penalties reported separately. New center failures, zero-RIoU outputs and RIoU<.5 paired tails.',
        review='Fixed probe non-regression checks against B, plus center and other-geometry strict gains, are descriptive finite-check gates. No automatic formal-training approval; preserve future original17 VAL conditions. Comparison with fixed-center includes two units and center supervision together, not proof of a unique root cause.',
        limitations=['TRAIN probe16 image identities/32 correlated views repeatedly inspected in development; B trained on all, not independent VAL.',
                     'Output frame decisions/scores remain B; moving centers no longer guarantees center-correct coverage.',
                     'A bounded single-frame head cannot recover absent outputs or guarantee continuity/depth.',
                     'Frozen P3 support, component-loss/RIoU mismatch and generalization may still limit the method.',
                     'TEST repeatedly exposed; no TEST tuning, checkpoint selection or access here.'])


def fit_center_support(rows):
    """Only FIT targets determine this support report; never inspect probe GT."""
    selected = [r for r in rows if r['role']=='fit']
    groups = {}
    for domain in ('real','sim'):
        for scale in g.SETTINGS['scales']:
            out = [r for r in selected if r['domain']==domain and r['scale']==scale and r['pred'] is not None]
            targets = []
            for r in out:
                b, gt = np.asarray(r['pred']), np.asarray(r['gt_original'])
                xy = (gt[:2]-b[:2])/min(b[2:4]); norm = float(np.linalg.norm(xy))
                targets.append(dict(image=r['image'], target_xy_over_b_short=xy.tolist(),
                    radial_target_over_b_short=norm,
                    inside_fixed_radial_bound=norm<CENTER_SETTINGS['max_center_over_b_short']))
            norms = [t['radial_target_over_b_short'] for t in targets]
            groups[domain+'/'+str(scale)] = dict(outputs=len(out), radial_target=g.describe(norms),
                max_radial_target=max(norms) if norms else None,
                outside_or_on_bound=sum(not t['inside_fixed_radial_bound'] for t in targets), targets=targets)
    return dict(evidence_role='FIT_TRAIN_targets_only_no_probe_bound_selection',
                views=len(selected), bound=CENTER_SETTINGS['max_center_over_b_short'], groups=groups)


def checked_contract(baseline_path):
    manifest, protocol = [json.loads(p.read_text()) for p in (MANIFEST, PROTOCOL)]
    if (manifest.get('protocol')!=VERSION or manifest.get('reference_g_settings')!=g.SETTINGS
            or manifest.get('center_settings')!=CENTER_SETTINGS):
        raise ValueError('Center source/settings contract differs')
    actual = {p:g.ready.sha(ROOT/p) for p in manifest['sources']}
    if actual!=manifest['sources']:
        raise ValueError('Reviewed center source SHA differs')
    if g.ready.sha(a.MANIFEST)!=A_MANIFEST_SHA:
        raise ValueError('Requires unchanged reviewed provenance')
    _, samples, prior_identity, baseline = a.checked_contract(baseline_path)
    if protocol!=protocol_document(samples):
        raise ValueError('Predeclared center protocol differs')
    identity = dict(manifest_sha256=g.ready.sha(MANIFEST), protocol_sha256=g.ready.sha(PROTOCOL),
        sources=actual, prior_identity=prior_identity, baseline_report_sha256=a.BASELINE_SHA,
        cache_manifest_sha256=a.CACHE_MANIFEST_SHA)
    return protocol, samples, identity, baseline


def checked_initialization(reference):
    g.seed_all(); wide = g.LocalGeometryRefiner()
    digest = g.state_digest(wide)
    if digest!=reference['initial_state']:
        raise ValueError('Seed1703 untrained FC64 initialization differs from reference')
    state = initial_state_from_reference(wide.state_dict())
    head = CenterLocalGeometryRefiner(); head.load_state_dict(state, strict=True)
    common = all(torch.equal(head.state_dict()[k],v) for k,v in wide.state_dict().items())
    joint_digest = g.state_digest(head)
    if not common or joint_digest['parameter_count']!=PARAMETER_COUNT or joint_digest['buffer_count']!=0:
        raise ValueError('Shared initialization or joint architecture differs')
    return state, dict(fixed_center_initial_state=digest, joint_initial_state=joint_digest,
                       all_shared_tensors_exact=True, added_center_units_zero=True)


def validate_prediction(output, b):
    refined = output['boxes_original']
    if refined.shape!=b.shape or not torch.equal(refined[:,5], b[:,5]):
        raise ValueError('Refinement changed B output counts/scores')
    shift = (refined[:,:2]-b[:,:2]).norm(dim=1)
    cap = CENTER_SETTINGS['max_center_over_b_short']*torch.minimum(b[:,2],b[:,3])
    # Original-coordinate float32 addition/subtraction can round near large xy.
    tolerance = 4*torch.finfo(b.dtype).eps*(1+b[:,:2].abs().max(dim=1)[0]) if len(b) else cap
    if bool((shift>cap+tolerance).any()):
        raise ValueError('Actual original-coordinate center movement exceeds bound')


def evaluate(head, records, device):
    rows, losses, shapes, centers, components = [], {}, {}, {}, {}
    projections, saturated, radial_saturated = 0, [0]*5, 0
    head.eval()
    with torch.no_grad():
        for i,r in enumerate(records):
            roi, support, b, bm, gt = g.batch_tensors(records,[i],'ordinary',device)
            output = head(roi,support,b,bm); validate_prediction(output,b)
            pred = output['boxes_original'].cpu().numpy()
            row = {k:r[k] for k in ('image','domain','sequence','frame_id','role','scale','eligible')}
            row.update(pred=pred[0].tolist() if len(pred) else None, gt_original=gt[0].cpu().tolist(),
                parsed_gt_original=r.get('parsed_gt_original'),
                reference_gt_absolute_delta=r.get('reference_gt_absolute_delta'),
                metrics=g.decompose(gt[0].cpu().numpy(),pred[0,:5] if len(pred) else None))
            rows.append(row)
            if r['eligible']:
                loss, parts = regression_loss(output['boxes_original'],gt,b)
                key = r['role']+'/'+r['domain']+'/'+str(r['scale'])
                losses.setdefault(key,[]).append(float(loss))
                shapes.setdefault(key,[]).append(float(parts[:3].mean()))
                centers.setdefault(key,[]).append(float(parts[3:].sum()/3.))
                components.setdefault(key,[]).append(parts.cpu().tolist())
            projections += int(output['edge_projection'].sum())
            cap = b.new_tensor([g.SETTINGS['max_log_edge_residual']]*2+[g.SETTINGS['max_angle_residual_rad']])
            near = torch.cat((output['residual'].abs()>=.95*cap,
                output['center_residual'].abs()>=.95*CENTER_SETTINGS['max_center_over_b_short']),dim=1)
            saturated = [v+int(n) for v,n in zip(saturated,near.sum(dim=0))]
            radial_saturated += int((output['center_residual'].norm(dim=1)>=.95*CENTER_SETTINGS['max_center_over_b_short']).sum())
    groups = {}
    for role in ('fit','probe'):
        for domain in ('real','sim'):
            for scale in g.SETTINGS['scales']:
                key = role+'/'+domain+'/'+str(scale)
                group_rows = [r for r in rows if r['role']==role and r['domain']==domain and r['scale']==scale]
                groups[key] = g.summarize(group_rows,continuous=False)
                values = [r['metrics']['riou'] for r in group_rows]
                groups[key]['riou_p10'] = float(np.percentile(values,10)) if values else None
                groups[key]['riou_min'] = min(values) if values else None
                groups[key]['center_correct_frames'] = sum(r['metrics']['center_hit'] for r in group_rows)
                groups[key]['zero_riou_output_frames'] = sum(r['metrics']['output'] and r['metrics']['riou']==0 for r in group_rows)
    return dict(rows=rows,groups=groups,
        eligible_loss={k:float(np.mean(v)) for k,v in losses.items()},
        eligible_shape_loss={k:float(np.mean(v)) for k,v in shapes.items()},
        eligible_center_loss={k:float(np.mean(v)) for k,v in centers.items()},
        eligible_raw_parts={k:np.mean(v,axis=0).tolist() for k,v in components.items()},
        edge_projection_views=projections,residual_saturation_counts=saturated,
        center_radial_saturation_views=radial_saturated)


def gradient_check(records,indices,initial,device):
    """No optimizer step: original shape path exact; added center path effective."""
    joint = CenterLocalGeometryRefiner().to(device); joint.load_state_dict(initial)
    wide = g.LocalGeometryRefiner().to(device)
    wide.load_state_dict({k:v for k,v in initial.items() if not k.startswith('center_output.')})
    roi,support,b,bm,gt = g.batch_tensors(records,indices,'ordinary',device)
    old = wide(roi,support,b,bm)['boxes_original']
    new = joint(roi,support,b,bm)['boxes_original']
    if not torch.equal(old,new) or not torch.equal(new,b):
        raise ValueError('Initial shared outputs are not exact B')
    old_loss, _ = g.regression_loss(old,gt)
    new_loss, parts = regression_loss(new,gt,b)
    shape_loss, _ = g.regression_loss(new,gt)
    og = torch.autograd.grad(old_loss,tuple(wide.parameters()))
    jg = torch.autograd.grad(shape_loss,tuple(joint.parameters()),retain_graph=True,allow_unused=True)
    named = dict(zip(dict(joint.named_parameters()),jg))
    if not torch.equal(old_loss,shape_loss) or any(not torch.equal(x,named[k]) for (k,_),x in zip(wide.named_parameters(),og)):
        raise ValueError('Original shape loss/shared gradients changed at initialization')
    tasks = [float(torch.autograd.grad(p/3.,joint.center_output.weight,retain_graph=True)[0].double().norm()) for p in parts[3:]]
    new_loss.backward()
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in joint.parameters()):
        raise ValueError('Missing/nonfinite initialization gradient')
    result = dict(no_optimizer_updates=True,original_shape_loss_exact=True,
        original_shape_gradients_exact=True,weighted_center_task_grad_norms=tasks,
        both_center_tasks_effective=all(v>0 for v in tasks),
        initial_shape_loss=float(old_loss), initial_added_center_loss=float(parts[3:].sum()/3.),
        initial_total_loss=float(new_loss))
    del joint,wide
    return result


def fit_joint(records,batches,initial,device,progress):
    head = CenterLocalGeometryRefiner().to(device); head.load_state_dict(initial,strict=True)
    before = g.state_digest(head); start = evaluate(head,records,device)
    if any(row['pred']!=(r['boxes_original'][0].tolist() if len(r['boxes_original']) else None)
           for row,r in zip(start['rows'],records)):
        raise ValueError('Zero-initialized joint head is not exact B')
    optimizer = torch.optim.Adam(head.parameters(),lr=g.SETTINGS['lr'],weight_decay=0.)
    logs = []; head.train()
    for step,indices in enumerate(batches,1):
        roi,support,b,bm,gt = g.batch_tensors(records,indices,'ordinary',device)
        optimizer.zero_grad(); output = head(roi,support,b,bm); validate_prediction(output,b)
        loss,parts = regression_loss(output['boxes_original'],gt,b)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite joint loss')
        tasks = None
        if step==1:
            weights = [head.output.weight]*3+[head.center_output.weight]*2
            tasks = [float(torch.autograd.grad(p/3.,w,retain_graph=True)[0].double().norm()) for p,w in zip(parts,weights)]
        loss.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in head.parameters()):
            raise ValueError('Missing/nonfinite head gradient')
        stem_norm = sum(float(p.grad.double().square().sum()) for p in head.stem.parameters())**.5
        norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(),g.SETTINGS['clip_norm']))
        if not math.isfinite(norm):
            raise ValueError('Nonfinite joint clipping norm')
        after_norm = sum(float(p.grad.double().square().sum()) for p in head.parameters())**.5
        if not math.isfinite(after_norm) or after_norm>g.SETTINGS['clip_norm']+1e-4:
            raise ValueError('Gradient clipping did not respect fixed norm')
        optimizer.step(); progress(dict(stage='update',step=step))
        if any(not bool(torch.isfinite(p).all()) for p in head.parameters()):
            raise ValueError('Nonfinite updated head')
        log = dict(stage='fit',arm='joint_center',step=step,loss=float(loss.detach()),
            shape_loss=float(parts[:3].detach().mean()), added_center_loss=float(parts[3:].detach().sum()/3.),
            raw_parts=parts.detach().cpu().tolist(), weighted_part_coefficients=[1./3.]*5,
            grad_norm_before_clip=norm,clip_multiplier=min(1.,g.SETTINGS['clip_norm']/(norm+1e-6)),
            grad_norm_after_clip=after_norm,
            stem_grad_norm=stem_norm,weighted_final_layer_task_grad_norms=tasks)
        logs.append(log); progress(log)
    final = evaluate(head,records,device)
    result = dict(initial_state=before,final_state=g.state_digest(head),initial=start,final=final,
        logs=logs,finite_gradients=True,complete_update_trajectory=True,
        loss_part_names=PART_NAMES,
        initial_all_task_gradients_effective=all(v>0 for v in logs[0]['weighted_final_layer_task_grad_norms']),
        stem_gradient_after_zero_output_step_effective=len(logs)>1 and logs[1]['stem_grad_norm']>0,
        fit_total_loss_decreased_by_group={k:final['eligible_loss'][k]<v-1e-6 for k,v in start['eligible_loss'].items() if k.startswith('fit/')},
        clipping=dict(records=len(logs),clipped_updates=sum(v['clip_multiplier']<1 for v in logs),
                      before_norm=g.describe([v['grad_norm_before_clip'] for v in logs]),
                      max_before_norm=max(v['grad_norm_before_clip'] for v in logs)))
    del head,optimizer; torch.cuda.empty_cache()
    return result


def verify_initial(result,reference):
    """Compare unchanged initial boxes/labels; center loss adds to old loss."""
    if len(result['rows'])!=len(reference['rows']):
        raise ValueError('Initial B view count differs')
    for new,old in zip(result['rows'],reference['rows']):
        if any(new[k]!=old[k] for k in ('image','role','domain','sequence','frame_id','scale','eligible','pred','gt_original')):
            raise ValueError('Initial B/GT view identity differs')
    if result['eligible_shape_loss']!=reference['eligible_loss']:
        raise ValueError('Original initial shape loss differs')


def paired_changes(reference,current):
    """Permit/report center correctness changes; output identities stay exact."""
    old_rows,new_rows = reference['rows'],current['rows']
    if len(old_rows)!=len(new_rows):
        raise ValueError('Paired view counts differ')
    pairs = {}
    for old,new in zip(old_rows,new_rows):
        if any(old[k]!=new[k] for k in ('image','role','domain','sequence','frame_id','scale','eligible','gt_original')):
            raise ValueError('Paired view/GT identity differs')
        if ((old['pred'] is None)!=(new['pred'] is None)
                or (old['pred'] is not None and old['pred'][5]!=new['pred'][5])):
            raise ValueError('Paired output presence/score changed')
        pairs.setdefault(old['role']+'/'+old['domain']+'/'+str(old['scale']),[]).append((old,new))
    groups = {}
    for key,pp in pairs.items():
        old,new = [r for r,_ in pp],[r for _,r in pp]
        osum,nsum = g.summarize(old,False),g.summarize(new,False)
        delta = {k:nsum[k]-osum[k] if osum[k] is not None and nsum[k] is not None else None
                 for k in ('output_coverage_pct','output_center_hit_pct','all_frame_center_hit_pct','all_frame_mean_riou')}
        for field,stats in [('center_error_px',('mean','rmse','p90')),('center_error_over_gt_short',('mean','p90')),
                            ('long_edge_relative_error',('mean','p90')),('short_edge_relative_error',('mean','p90')),
                            ('angle_error_deg',('rmse','p90')),('protocol_angle',('rmse',))]:
            for stat in stats:
                bv,cv = osum[field][stat],nsum[field][stat]
                delta[field+'/'+stat] = cv-bv if bv is not None and cv is not None else None
        riou_old,riou_new = [r['metrics']['riou'] for r in old],[r['metrics']['riou'] for r in new]
        delta.update(riou_p10=float(np.percentile(riou_new,10)-np.percentile(riou_old,10)),riou_min=min(riou_new)-min(riou_old))
        events = {k:[] for k in ('new_center_failures','recovered_center_failures','new_riou_below_0_5','recovered_riou_below_0_5','new_zero_riou_outputs','recovered_zero_riou_outputs')}
        tails,hybrids = [],[]
        for br,cr in pp:
            bm,cm = br['metrics'],cr['metrics']; image = br['image']
            for name,bad_old,bad_new in [
                ('center_failures',not bm['center_hit'],not cm['center_hit']),
                ('riou_below_0_5',bm['riou']<.5,cm['riou']<.5),
                ('zero_riou_outputs',bm['output'] and bm['riou']==0,cm['output'] and cm['riou']==0)]:
                if bad_new and not bad_old: events['new_'+name].append(image)
                if bad_old and not bad_new: events['recovered_'+name].append(image)
            if not bm['output']:
                continue
            bp,cp = np.asarray(br['pred']),np.asarray(cr['pred'])
            center_only,shape_only = bp[:5].copy(),cp[:5].copy()
            center_only[:2],shape_only[:2] = cp[:2],bp[:2]
            hybrids.append(dict(image=image,
                current_center_reference_shape_riou=g.decompose(br['gt_original'],center_only)['riou'],
                reference_center_current_shape_riou=g.decompose(br['gt_original'],shape_only)['riou']))
            tails.append(dict(image=image,center_error_before=bm['center_error_px'],center_error_after=cm['center_error_px'],
                center_error_delta=cm['center_error_px']-bm['center_error_px'],riou_before=bm['riou'],riou_after=cm['riou'],
                riou_delta=cm['riou']-bm['riou'],center_shift_px=float(np.linalg.norm(cp[:2]-bp[:2]))))
        groups[key] = dict(delta=delta,events=events,
            worst_riou_drops=sorted(tails,key=lambda r:r['riou_delta'])[:8],
            worst_center_increases=sorted(tails,key=lambda r:r['center_error_delta'],reverse=True)[:8],
            predicted_component_hybrids=hybrids)
    return dict(direction='current minus reference',groups=groups,
        hybrid_note='Saved predicted component substitutions only; not separately trained ablations, additive causes or deployment outputs.')


def probe_review(comparison):
    """Predeclared finite-check criteria, separate from future17 VAL gates."""
    groups = {}
    for key,value in comparison['groups'].items():
        if not key.startswith('probe/'):
            continue
        d,e = value['delta'],value['events']
        errors = ['center_error_px/'+s for s in ('mean','rmse','p90')]
        errors += ['center_error_over_gt_short/p90']
        errors += [f+'/'+s for f in ('long_edge_relative_error','short_edge_relative_error') for s in ('mean','p90')]
        errors += ['angle_error_deg/rmse','angle_error_deg/p90']
        checks = {f+'_not_worse':d[f] is not None and d[f]<=0 for f in errors}
        checks.update(output_coverage_unchanged=d['output_coverage_pct']==0,
            conditional_center_correct_not_worse=d['output_center_hit_pct'] is not None and d['output_center_hit_pct']>=0,
            all_frame_center_correct_not_worse=d['all_frame_center_hit_pct'] is not None and d['all_frame_center_hit_pct']>=0,
            riou_mean_not_worse=d['all_frame_mean_riou']>=0,riou_p10_not_worse=d['riou_p10']>=0,
            no_new_center_failures=not e['new_center_failures'],no_new_zero_riou_outputs=not e['new_zero_riou_outputs'],
            no_new_riou_below_0_5=not e['new_riou_below_0_5'])
        other = ['long_edge_relative_error/mean','short_edge_relative_error/mean','angle_error_deg/rmse']
        groups[key] = dict(checks=checks,all_non_regression_checks_met=all(checks.values()),
            center_mean_strict_improved=d['center_error_px/mean'] is not None and d['center_error_px/mean']<0,
            other_geometry_strict_improved=any(d[f] is not None and d[f]<0 for f in other) or d['all_frame_mean_riou']>0)
    return dict(groups=groups,
        all_probe_non_regression_checks_met=bool(groups) and all(v['all_non_regression_checks_met'] for v in groups.values()),
        all_groups_have_center_and_other_gain=bool(groups) and all(v['center_mean_strict_improved'] and v['other_geometry_strict_improved'] for v in groups.values()),
        formal_training_approved=False,
        note='No post-result relaxation or significance claim. Sparse repeatedly used TRAIN probe cannot establish video/VAL/depth improvement.')


def publish_artifacts(out,report):
    g.write_new(out/'artifacts.json',dict(protocol=VERSION,status=report['status'],
        files={str(p.relative_to(out)):g.ready.sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifacts.json'}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check-only',action='store_true',help='Provenance/fit targets only; no cache load, head, GPU or updates.')
    ap.add_argument('--gpu',type=int,default=0)
    ap.add_argument('--baseline-report',default=str(a.BASELINE))
    ap.add_argument('--cache-dir',default='work_dirs/port_geometry_g_v1_roi_cache')
    ap.add_argument('--out-dir',required=True)
    args = ap.parse_args(); os.chdir(ROOT)
    out = Path(args.out_dir).resolve(); out.mkdir(parents=True,exist_ok=False)
    report = dict(protocol=VERSION,evidence_role='finite_TRAIN_bounded_center_intervention',status='STARTED',
        formal_training=False,detector_forward_calls=0,detector_updates=0,additional_reference_updates=0,
        head_updates_total=0,checkpoint_exported=False,gpu_devices_used=0)
    with (out/'progress.jsonl').open('x') as stream:
        def progress(value):
            if value['stage']=='update':
                report['head_updates_total']+=1; return
            stream.write(json.dumps(g.json_native(value),ensure_ascii=False,allow_nan=False)+'\n'); stream.flush()
            if value['step'] in (1,2) or value['step']%25==0:
                print('TRAIN joint_center step',value['step'],'total',value['loss'],
                      'shape',value['shape_loss'],'center',value['added_center_loss'],flush=True)
        try:
            protocol,samples,identity,baseline = checked_contract(Path(args.baseline_report).resolve())
            report.update(identity=identity,reference_g_settings=g.SETTINGS,center_settings=CENTER_SETTINGS,
                parameter_counts=protocol['parameter_counts'],fixed_sample_counts=protocol['fixed_sample_counts'],
                limitations=protocol['limitations'],fit_center_support=fit_center_support(baseline['arms']['ordinary']['initial']['rows']))
            g.write_new(out/'protocol.json',protocol); shutil.copyfile(str(MANIFEST),str(out/'sources.json'))
            print('G center check: saved fixed-center FC64 updates=0; joint FC64 updates='+str(STEPS)+'; B inference/updates=0. Not formal training.',flush=True)
            if args.check_only:
                report['status']='STATIC_CENTER_CONTRACT_COMPLETE_NO_CACHE_LOAD_NO_GPU_NO_UPDATES'
            else:
                if int(os.environ.get('WORLD_SIZE','1'))!=1:
                    raise ValueError('Single-process finite check; do not use torchrun/DDP')
                report['runtime']=a.checked_runtime(baseline)
                cache_dir=Path(args.cache_dir).resolve()
                if g.ready.sha(cache_dir/'cache_manifest.json')!=a.CACHE_MANIFEST_SHA:
                    raise ValueError('Requires the exact reviewed complete ROI cache')
                payload,cache=g.checked_cache(cache_dir,identity['prior_identity']['g_identity'],samples)
                records=payload['records']; g.validate_records(records,samples)
                a.check_cache_reference(records,cache,baseline)
                if any(len(r['boxes_original'])!=1 or not r['eligible'] for r in records):
                    raise ValueError('Expected all128 reviewed eligible B outputs')
                g.write_new(out/'cache_manifest.json',cache)
                report.update(cache=cache,cache_reused=True,detector_state_unchanged=True,
                    detector_evidence='Reviewed cached B before/after digests verified; no detector constructed in this run.',
                    original_cpu_roi_bytes=g.cpu_roi_bytes(records),
                    train_support=g.support_report(records,json.loads(g.PROTOCOL.read_text())))
                if not torch.cuda.is_available() or not 0<=args.gpu<torch.cuda.device_count():
                    raise ValueError('A valid logical CUDA device is required')
                torch.cuda.set_device(args.gpu); device=torch.device('cuda',args.gpu)
                report['gpu_devices_used']=1; report['runtime']['gpu']=torch.cuda.get_device_name(args.gpu)
                initial,initialization=checked_initialization(baseline['arms']['ordinary'])
                batches=g.schedule(records)
                if len(batches)!=STEPS or batches!=baseline['minibatches']:
                    raise ValueError('Fixed200 reference batches differ')
                report.update(initialization=initialization,minibatches=batches,
                    fixed_center_fc64=baseline['arms']['ordinary'],reference_origin='reviewed saved ordinary; no new fixed-center updates')
                print('Reusing reviewed CPU ordinary ROIs. One logical GPU:',args.gpu,'; only joint head fitted.',flush=True)
                def operation():
                    report['gradient_contract']=gradient_check(records,batches[0],initial,device)
                    return fit_joint(records,batches,initial,device,progress)
                result,cost=g.measured(operation,args.gpu)
                report.update(joint_center=result,refiner_cost=cost)
                g.write_new(out/'joint_center.json',dict(protocol=VERSION,identity=identity,result=result,cost=cost,
                    completed_head_updates=STEPS,formal_training=False,checkpoint_exported=False))
                verify_initial(result['initial'],baseline['arms']['ordinary']['initial'])
                if report['head_updates_total']!=STEPS or result['initial_state']!=initialization['joint_initial_state']:
                    raise ValueError('Update budget or initial joint digest differs')
                report.update(joint_minus_b=paired_changes(baseline['arms']['ordinary']['initial'],result['final']),
                    joint_minus_fixed_center=paired_changes(baseline['arms']['ordinary']['final'],result['final']))
                report['probe_review']=probe_review(report['joint_minus_b'])
                report['status']='TRAIN_CENTER_CHECK_COMPLETE_REVIEW_REQUIRED'
            g.write_new(out/'completion.json',report)
        except Exception as error:
            report['status']='FAILED_REVIEW_REQUIRED'; report['error']=type(error).__name__+': '+str(error)
            destination=out/('failure.json' if (out/'completion.json').exists() else 'completion.json')
            try:
                g.write_new(destination,report)
            except (TypeError,ValueError) as serialization_error:
                g.write_new(destination,dict(protocol=VERSION,status=report['status'],formal_training=False,
                    detector_updates=0,detector_forward_calls=0,additional_reference_updates=0,
                    head_updates_total=report['head_updates_total'],error=report['error'],
                    serialization_error=str(serialization_error),full_report_available=False))
            stream.flush(); publish_artifacts(out,report); raise
    publish_artifacts(out,report)
    print('Saved',out/'completion.json','status',report['status'],flush=True)


if __name__=='__main__':
    main()
