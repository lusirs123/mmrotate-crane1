#!/usr/bin/env python3
"""Fixed TRAIN spatial midpoint check: matched/shuffled aligned cached ROIs.

64 images/128 related views, 200 head updates per arm, B updates/inference=0.
No fitted checkpoint, formal training, new VAL/TEST, step or coefficient search.
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
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import preflight_port_geometry_g_roi_ablation_v1 as a
from crane_project.utils import port_geometry_midpoint_v1 as m
from crane_project.utils.port_train_geometry_evidence_v1 import view_geometry

g = a.g
VERSION = 'port_geometry_midpoint_v1_train_preflight'
CONDITIONS = ('matched', 'shuffled')
PROTOCOL = ROOT/'crane_project/tools/port_geometry_midpoint_v1_protocol.json'
MANIFEST = ROOT/'crane_project/tools/port_geometry_midpoint_v1_sources.json'
PARENT_MANIFEST_SHA = '0250eafb9feef5fff88a4072dca02b1a0cb71f345f6fa1e0b866c2d8a2ada170'
TOLERANCE = dict(center_px=1e-6, point_px=1e-6, point_roi=1e-8,
                 edge_fraction=1e-6, angle_deg=1e-5, riou=1e-5)


def protocol_document(samples):
    return dict(protocol=VERSION, settings=deepcopy(m.SETTINGS),
        reference_g_settings=deepcopy(g.SETTINGS), parameter_count=m.PARAMETER_COUNT,
        evidence_role='finite_TRAIN_spatial_midpoint_intervention',
        conditions=['frozen_b_identity']+list(CONDITIONS),
        donor_mapping=a.donor_map(samples),
        prior_manifest_sha256=PARENT_MANIFEST_SHA,
        baseline_report_sha256=a.BASELINE_SHA, cache_manifest_sha256=a.CACHE_MANIFEST_SHA,
        fixed_sample_counts={role:dict(Counter(s['domain'] for s in samples if s['role']==role)) for role in ('fit','probe')},
        input='Reuse existing detached aligned CPU ROI and support. Move both for shuffled; retain recipient B/GT/scale metadata. No B forward or new cache extraction.',
        architecture='259(256ROI+support+fixedxy)->Conv1x1/32/ReLU->Conv3x3/32/ReLU->Conv1x1/4(no bias). No flatten FC, BN, dropout, upsample or trainable sampling.',
        readout='Four actual-location spatial softmax distributions at9x9 with fixed B-only Gaussian log prior sigma1ROIcell centered on each B anchor. Point=anchor+E(prior+logits)-E(prior), calibrating truncated-grid expectation for exact zero initialization. Continuous expectation, not hard argmax. No new native resolution.',
        targets='Four midpoints from original fixed TRAIN OBB; select among4 cyclic permutations by minimum squared normalized distance to frozen B anchors, first argmin tie. Not visible keypoint annotations. No GT-dependent online acceptance.',
        coordinates='Exact aligned sampler frame: canonical MODEL B axes,1.5 side context/min16modelpx. Original xy points map by sx/sy and inverse separately; raw B w/h-angle association checked BEFORE canonicalization. Keep existing detector box restoration and annotation sqrt(sx*sy) pipeline unchanged.',
        objective='Mean4x2 SmoothL1(beta.1) between predicted and target normalized ROI coordinates only. Equal point/xy weights; no additional center/shape/IoU/classification loss or heatmap pseudo-label.',
        decode='Original-coordinate points: center=mean4,u=p0-p2,v=p1-p3,theta=atan2(u_y-v_x,u_x+v_y),positive projected width/height. Fixed orthogonal rectangle projection; preserve raw B width/height association and closest periodic angle. Zero delta returns exact B.',
        fallback='Full B fallback if nonfinite/nonpositive/degenerate direction/any point outside ROI or center>.30Bshort,raw edge ratio outside[.8,1.25],periodic angle change>10deg. No GT, score adjustment, component clipping or outcome-selected acceptance. Report raw candidate AND delivered box, point error, projection discrepancy and all fallback reasons.',
        optimizer='Same seed1703 initial tensors and original200 balanced fit-only batches8(4real+4sim) for both arms. Adam lr.001 wd0 clip10. Log every update; evaluate only initial and fixed final200. No donor/step/seed/bound/weight search.',
        scope=dict(detector_forward_calls=0,detector_updates=0,feature_extractions=0,
            cache_loads_full_run=1,updates_per_arm=m.SETTINGS['steps_per_arm'],
            head_updates_total=2*m.SETTINGS['steps_per_arm'],formal_training=False,
            checkpoint_export=False,val_access=False,test_access=False),
        evaluation='All128 views by fit/probe,real/sim,1.0/.5. Point originalpx/context units,entropy/distributions,raw and delivered center/edges/pure periodic angle/strictRIoU. Output coverage,conditional center correctness only outputs,all-frame correct coverage. No sparse temporal metric.',
        tolerance=deepcopy(TOLERANCE),
        continuation='Point mean/RMSE in BOTH originalpx and ROI units non-regressing matched vs B and shuffled in EACH4 probe groups, with aggregate matched vs shuffled improvement. Delivered matched all4 probe center(mean/RMSE/p90),edges(mean),pureangle(RMSE/p90),RIoU(mean/p10/min) non-regressing vs B; no new center/RIoU<.5/zero-overlap failures. At least one real probe group improves centermean AND meanRIoU; at least one sim group improves pureangleRMSE AND meanRIoU. Numerical tolerance only, never significance or formal approval.',
        limitations=['Repeated TRAIN probe16 identities/32 related views, all seen by B; not independent VAL.',
            'Four OBB midpoints can lie on blank/occluded areas; they are geometry, not necessarily physical boundaries.',
            '9x9 interpolation and continuous readout do not increase native P3 resolution.',
            'Aligned sampling/spatial conv/point supervision change together; results do not isolate a unique root or supervision effect.',
            'Expectation can average multimodal maps; inspect distributions and rectangle projection, not loss alone.',
            'Fixed within-stratum shuffle can preserve video appearance or learn donor relations; failure is not proof of absent information.',
            'Point improvement and preserved sparse output counts do not guarantee final OBB, real video continuity or depth accuracy.',
            'TEST repeatedly exposed previously; no new TEST/VAL or method/checkpoint selection in this check.'])


def checked_sources():
    manifest = json.loads(MANIFEST.read_text())
    expected_paths = set(json.loads(a.MANIFEST.read_text())['sources']) | {
        str(a.MANIFEST.relative_to(ROOT)), str(PROTOCOL.relative_to(ROOT)),
        'crane_project/utils/port_train_geometry_evidence_v1.py',
        'crane_project/utils/port_geometry_midpoint_v1.py',
        'crane_project/tools/preflight_port_geometry_midpoint_v1.py',
        'tests/test_port_geometry_midpoint_v1.py'}
    if (manifest.get('protocol')!=VERSION or manifest.get('settings')!=m.SETTINGS or
            set(manifest.get('sources',{}))!=expected_paths or
            g.ready.sha(a.MANIFEST)!=PARENT_MANIFEST_SHA):
        raise ValueError('Midpoint source/settings/upstream contract differs')
    actual = {p:g.ready.sha(ROOT/p) for p in manifest['sources']}
    if actual!=manifest['sources']:
        raise ValueError('Midpoint source SHA differs: '+', '.join(p for p in actual if actual[p]!=manifest['sources'][p]))
    return actual


def checked_contract(baseline_path):
    sources = checked_sources()
    _, samples, parent_identity, baseline = a.checked_contract(baseline_path)
    protocol = json.loads(PROTOCOL.read_text())
    if protocol!=protocol_document(samples):
        raise ValueError('Fixed midpoint protocol differs')
    if any(m.SETTINGS[k]!=g.SETTINGS[k] for k in ('seed','steps_per_arm','batch_size','lr','weight_decay','clip_norm','roi_size','stride','context_multiplier','min_context_cells')):
        raise ValueError('Midpoint shared budget/sampler differs from reviewed cache')
    return protocol,samples,dict(manifest_sha256=g.ready.sha(MANIFEST),
        protocol_sha256=g.ready.sha(PROTOCOL),sources=sources,
        prior_identity=parent_identity),baseline


def prepare_records(records,samples,mapping):
    if mapping!=a.donor_map(samples):
        raise ValueError('Fixed midpoint donor mapping differs')
    lookup={(r['image'],r['scale']):r for r in records}
    expected=[(s['image'],scale,s['role']) for scale in g.SETTINGS['scales'] for s in samples]
    if len(lookup)!=len(records) or [(r['image'],r['scale'],r['role']) for r in records]!=expected:
        raise ValueError('Midpoint view count/order differs')
    by_image={s['image']:s for s in samples}
    prepared,assignment=[],[]
    for r in records:
        donor=lookup[(mapping[r['image']],r['scale'])]
        if donor['image']==r['image'] or any(donor[k]!=r[k] for k in ('role','domain','scale')):
            raise ValueError('Donor crossed role/domain/scale or used self')
        if len(r['boxes_original'])!=1 or not r['eligible']:
            raise ValueError('Requires all128 reviewed genuine eligible B outputs')
        sample=by_image[r['image']]
        geom=view_geometry(sample['image_size'],r['scale'])
        w,h=sample['image_size']; iw,ih=geom['image_wh']
        meta=dict(ori_shape=(h,w,3),img_shape=(ih,iw,3),pad_shape=(1024,1024,3),
                  scale_factor=geom['scale_factor'],flip=False)
        g.checked_meta(meta)
        scales=r['boxes_original'].new_tensor([geom['scale_factor'][:2]])
        m.frame(r['boxes_original'],r['boxes_model'],scales)
        target=m.target_points(r['gt_original'],r['boxes_original'],r['boxes_model'],scales)
        if bool((target['roi'].abs()>.5+m.SETTINGS['roi_bound_tolerance']).any()):
            raise ValueError('Supervision point outside reviewed aligned ROI; no dropping/clamping: '+r['image'])
        _,_,neutral=m.point_prior(target['reference_roi'])
        required=target['roi']-target['reference_roi']+neutral
        if bool((required.abs()>.5+m.SETTINGS['roi_bound_tolerance']).any()):
            raise ValueError('Supervision point expectation unreachable after B-prior calibration: '+r['image'])
        current=dict(r,local=dict(r['local']),scale_xy=scales,geometry=geom,
                     midpoint_target=target)
        current['local']['matched']=r['local']['aligned']
        current['local']['shuffled']=donor['local']['aligned']
        prepared.append(current)
        assignment.append(dict(recipient=r['image'],donor=donor['image'],
            role=r['role'],domain=r['domain'],scale=r['scale']))
    return prepared,assignment


def batch(records,indices,condition,device):
    roi,support,b,bm,gt=g.batch_tensors(records,indices,condition,device)
    scales=torch.cat([records[i]['scale_xy'] for i in indices]).to(device)
    return roi,support,b,bm,scales,gt


def summarize_rows(rows,field='metrics'):
    groups={}
    for role in ('fit','probe'):
        for domain in ('real','sim'):
            for scale in g.SETTINGS['scales']:
                key=role+'/'+domain+'/'+str(scale)
                subset=[r for r in rows if r['role']==role and r['domain']==domain and r['scale']==scale]
                effective=[dict(r,metrics=r[field]) for r in subset]
                summary=g.summarize(effective,continuous=False)
                riou=[r['metrics']['riou'] for r in effective]
                hits=sum(r['metrics']['center_hit'] for r in effective)
                summary.update(riou_p10=float(np.percentile(riou,10)) if riou else None,
                    riou_min=min(riou) if riou else None,center_correct_frames=hits,
                    output_coverage_fraction=dict(numerator=summary['output_frames'],denominator=len(subset)),
                    conditional_center_correct_fraction=dict(numerator=hits,denominator=summary['output_frames']),
                    all_frame_center_correct_fraction=dict(numerator=hits,denominator=len(subset)))
                if field=='metrics':
                    for units in ('px','roi'):
                        summary['point_error_'+units]=g.describe([v for r in subset for v in r['point_error_'+units]])
                    summary['fallback_views']=sum(not r['accepted'] for r in subset)
                    summary['candidate_invalid_views']=sum(not r['candidate_valid'] for r in subset)
                    summary['entropy_mean']=float(np.mean([v for r in subset for v in r['point_entropy']])) if subset else None
                groups[key]=summary
    return groups


def evaluate(head,records,condition,device,keep_maps=False):
    rows,maps,losses=[],[],{}
    head.eval()
    with torch.no_grad():
        for i,r in enumerate(records):
            roi,support,b,bm,scales,gt=batch(records,[i],condition,device)
            output=head(roi,support,b,bm,scales)
            target=m.target_points(gt,b,bm,scales)
            loss,parts=m.point_loss(output['points_roi'],target)
            delivered=output['boxes_original'][0].cpu().tolist()
            candidate=(output['candidate_original'][0].cpu().tolist() if bool(output['candidate_valid'][0]) else None)
            if not torch.equal(output['boxes_original'][:,5],b[:,5]):
                raise ValueError('Midpoint changed B output score')
            errors_px=(output['points_original']-target['original']).norm(dim=-1)[0].cpu().tolist()
            errors_roi=(output['points_roi']-target['roi']).norm(dim=-1)[0].cpu().tolist()
            probability=output['probabilities'][0]
            entropy=(-(probability*probability.clamp(min=1e-30).log()).sum(dim=(1,2))).cpu().tolist()
            row={k:r[k] for k in ('image','domain','sequence','frame_id','role','scale','eligible')}
            row.update(pred=delivered,candidate=candidate,gt_original=gt[0].cpu().tolist(),
                metrics=g.decompose(gt[0].cpu().numpy(),np.array(delivered[:5])),
                candidate_metrics=g.decompose(gt[0].cpu().numpy(),np.array(candidate[:5]) if candidate is not None else None),
                candidate_valid=bool(output['candidate_valid'][0]),accepted=bool(output['accepted'][0]),
                failed_checks=[k for k,v in output['checks'].items() if not bool(v[0])],
                point_error_px=errors_px,point_error_roi=errors_roi,point_entropy=entropy,
                predicted_points_original=output['points_original'][0].cpu().tolist(),
                predicted_points_roi=output['points_roi'][0].cpu().tolist(),
                target_points_original=target['original'][0].cpu().tolist(),
                target_points_roi=target['roi'][0].cpu().tolist(),
                target_permutation=int(target['permutation'][0]),
                rectangle_projection_rms_px=float(output['rectangle_projection_rms_px'][0]),
                actual_center_shift_px=(output['boxes_original'][0,:2]-b[0,:2]).cpu().tolist(),
                desired_center_shift_px=(gt[0,:2]-b[0,:2]).cpu().tolist())
            rows.append(row)
            key=r['role']+'/'+r['domain']+'/'+str(r['scale'])
            losses.setdefault(key,[]).append(float(loss))
            if keep_maps:
                maps.append(dict(image=r['image'],scale=r['scale'],role=r['role'],domain=r['domain'],
                    geometry=r['geometry'],anchor_roi=output['anchor_roi'][0].cpu().tolist(),
                    neutral_roi=output['neutral_roi'][0].cpu().tolist(),
                    predicted_roi=row['predicted_points_roi'],target_roi=row['target_points_roi'],
                    probability=probability.cpu().tolist(),entropy=entropy,
                    note='Calibrated actual-location map; point=anchor+E(probability)-E(B prior). B-only sigma1cell prior, no GT prior. Not an object heatmap.'))
    return dict(rows=rows,groups=summarize_rows(rows),
        raw_candidate_groups=summarize_rows(rows,'candidate_metrics'),
        point_loss={k:float(np.mean(v)) for k,v in losses.items()}),maps


def fit_arm(records,batches,initial,condition,device,progress):
    head=m.SpatialMidpointHead().to(device); head.load_state_dict(initial,strict=True)
    before=g.state_digest(head)
    start,_=evaluate(head,records,condition,device)
    if any(row['pred']!=r['boxes_original'][0].tolist() or not row['accepted']
           for row,r in zip(start['rows'],records)):
        raise ValueError('Zero midpoint correction is not exact delivered B')
    optimizer=torch.optim.Adam(head.parameters(),lr=m.SETTINGS['lr'],weight_decay=m.SETTINGS['weight_decay'])
    logs=[]; head.train()
    for step,indices in enumerate(batches,1):
        roi,support,b,bm,scales,gt=batch(records,indices,condition,device)
        optimizer.zero_grad()
        output=head(roi,support,b,bm,scales)
        target=m.target_points(gt,b,bm,scales)
        loss,parts=m.point_loss(output['points_roi'],target)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite point loss')
        task_norms=([float(torch.autograd.grad(p/4.,head.output.weight,retain_graph=True)[0].double().norm()) for p in parts] if step==1 else None)
        if task_norms is not None and not all(x>0 and math.isfinite(x) for x in task_norms):
            raise ValueError('Initial four-point task gradient ineffective')
        loss.backward()
        params=list(head.parameters())
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params):
            raise ValueError('Missing/nonfinite midpoint gradient')
        stem_norm=sum(float(p.grad.double().square().sum()) for p in head.stem.parameters())**.5
        if step==2 and stem_norm<=0:
            raise ValueError('Stem gradient ineffective after zero-output step')
        norm=float(torch.nn.utils.clip_grad_norm_(params,m.SETTINGS['clip_norm']))
        after=sum(float(p.grad.double().square().sum()) for p in params)**.5
        if not math.isfinite(norm) or not math.isfinite(after) or after>m.SETTINGS['clip_norm']+1e-4:
            raise ValueError('Invalid midpoint gradient clipping')
        optimizer.step()
        progress(dict(stage='update',arm=condition,step=step))
        if any(not bool(torch.isfinite(p).all()) for p in params):
            raise ValueError('Nonfinite updated midpoint head')
        log=dict(stage='fit',arm=condition,step=step,loss=float(loss.detach()),
            point_parts=parts.detach().cpu().tolist(),grad_norm_before_clip=norm,
            grad_norm_after_clip=after,clip_multiplier=min(1.,m.SETTINGS['clip_norm']/(norm+1e-6)),
            stem_grad_norm=stem_norm,initial_point_task_grad_norms=task_norms)
        logs.append(log); progress(log)
    final,maps=evaluate(head,records,condition,device,keep_maps=True)
    result=dict(initial_state=before,final_state=g.state_digest(head),initial=start,final=final,logs=logs,
        completed_updates=len(logs),finite_gradients=True,
        initial_four_point_gradients_effective=all(x>0 for x in logs[0]['initial_point_task_grad_norms']),
        stem_step2_effective=logs[1]['stem_grad_norm']>0)
    del head,optimizer; torch.cuda.empty_cache()
    return result,maps


def paired_changes(reference,current):
    if len(reference['rows'])!=len(current['rows']):
        raise ValueError('Paired midpoint row count differs')
    changes=[]
    for old,new in zip(reference['rows'],current['rows']):
        if any(old[k]!=new[k] for k in ('image','role','domain','sequence','frame_id','scale','gt_original')):
            raise ValueError('Paired midpoint identity/GT differs')
        om,nm=old['metrics'],new['metrics']
        changes.append(dict(image=old['image'],role=old['role'],domain=old['domain'],scale=old['scale'],
            center_delta_px=nm.get('center_error_px',0)-om.get('center_error_px',0),
            riou_delta=nm['riou']-om['riou'],
            new_center_failure=om['output'] and om['center_hit'] and not nm['center_hit'],
            new_no_output=om['output'] and not nm['output'],
            new_riou_below_0_5=om['riou']>=.5 and nm['riou']<.5,
            new_zero_riou=om['riou']>0 and nm['riou']<=0))
    return changes


def review_conditions(baseline,matched,shuffled):
    checks={}
    paired=paired_changes(baseline,matched)
    probe=[r for r in paired if r['role']=='probe']
    checks['no_new_probe_tail_failures']=not any(r[k] for r in probe for k in ('new_center_failure','new_no_output','new_riou_below_0_5','new_zero_riou'))
    keys=[k for k in baseline['groups'] if k.startswith('probe/')]
    real_gain,sim_gain=False,False
    for key in keys:
        old,new,control=(value['groups'][key] for value in (baseline,matched,shuffled))
        conditions={}
        for field,stats,tol in (
            ('center_error_px',('mean','rmse','p90'),TOLERANCE['center_px']),
            ('long_edge_relative_error',('mean',),TOLERANCE['edge_fraction']),
            ('short_edge_relative_error',('mean',),TOLERANCE['edge_fraction']),
            ('angle_error_deg',('rmse','p90'),TOLERANCE['angle_deg'])):
            for stat in stats:
                conditions[field+'/'+stat]=new[field][stat]<=old[field][stat]+tol
        for field in ('all_frame_mean_riou','riou_p10','riou_min'):
            conditions[field]=new[field]>=old[field]-TOLERANCE['riou']
        for field in ('output_coverage_fraction','all_frame_center_correct_fraction'):
            conditions[field]=new[field]['numerator']>=old[field]['numerator'] and new[field]['denominator']==old[field]['denominator']
        for units,tol in (('px',TOLERANCE['point_px']),('roi',TOLERANCE['point_roi'])):
            for stat in ('mean','rmse'):
                field='point_error_'+units
                conditions[field+'/'+stat+'/vs_b']=new[field][stat]<=old[field][stat]+tol
                conditions[field+'/'+stat+'/vs_shuffle']=new[field][stat]<=control[field][stat]+tol
        checks[key]=conditions
        if key.startswith('probe/real/'):
            real_gain|=(new['center_error_px']['mean']<old['center_error_px']['mean']-TOLERANCE['center_px'] and
                        new['all_frame_mean_riou']>old['all_frame_mean_riou']+TOLERANCE['riou'])
        if key.startswith('probe/sim/'):
            sim_gain|=(new['angle_error_deg']['rmse']<old['angle_error_deg']['rmse']-TOLERANCE['angle_deg'] and
                       new['all_frame_mean_riou']>old['all_frame_mean_riou']+TOLERANCE['riou'])
    def mean_points(value,units):
        return float(np.mean([v for r in value['rows'] if r['role']=='probe' for v in r['point_error_'+units]]))
    checks['matched_point_probe_gain_vs_shuffle']=all(mean_points(matched,u)<mean_points(shuffled,u)-TOLERANCE['point_'+u] for u in ('px','roi'))
    checks['real_center_and_riou_gain']=real_gain
    checks['sim_angle_and_riou_gain']=sim_gain
    passed=all(all(v.values()) if isinstance(v,dict) else v for v in checks.values())
    return dict(checks=checks,diagnostic_continuation_conditions_met=passed,
                formal_training_approved=False,
                note='Fixed numerical gate on repeated TRAIN probe; no stable/significant/generalization claim.')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-only',action='store_true',help='Sources/TRAIN/baseline/protocol only; no feature cache/GPU/updates.')
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--baseline-report',default=str(a.BASELINE))
    p.add_argument('--cache-dir',default='work_dirs/port_geometry_g_v1_roi_cache')
    p.add_argument('--out-dir',required=True)
    return p


def publish_artifacts(out,report):
    files=sorted(p for p in out.rglob('*') if p.is_file() and p.name!='artifacts.json')
    g.write_new(out/'artifacts.json',dict(protocol=VERSION,status=report['status'],
        files={str(p.relative_to(out)):g.ready.sha(p) for p in files}))


def run(args):
    os.chdir(ROOT)
    out=Path(args.out_dir).resolve(); out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    report=dict(protocol=VERSION,status='STARTED',formal_training=False,
        formal_training_approved=False,checkpoint_exported=False,
        detector_updates=0,detector_forward_calls=0,feature_extractions=0,
        head_updates_total=0,cache_loads=0,gpu_devices_used=[],detector_not_instantiated=True)
    with (out/'progress.jsonl').open('x') as stream:
        def progress(value):
            if value['stage']=='update':
                report['head_updates_total']+=1
            stream.write(json.dumps(g.json_native(value),ensure_ascii=False,allow_nan=False)+'\n'); stream.flush()
            if value['stage']=='fit' and (value['step'] in (1,2) or value['step']%25==0):
                print('TRAIN midpoint',value['arm'],'step',value['step'],'loss',value['loss'],flush=True)
        progress(dict(stage='begin',protocol=VERSION))
        try:
            protocol,samples,identity,baseline=checked_contract(Path(args.baseline_report).resolve())
            report.update(identity=identity,settings=m.SETTINGS,limitations=protocol['limitations'],
                fixed_sample_counts=protocol['fixed_sample_counts'])
            g.write_new(out/'protocol.json',protocol)
            shutil.copyfile(str(MANIFEST),str(out/'sources.json'))
            progress(dict(stage='verified',samples=len(samples),cache_loads=0))
            print('Spatial midpoint check: 64 TRAIN images/128 views; matched/shuffled aligned ROIs; 200 updates each. B inference/updates=0. Not formal training.',flush=True)
            if args.check_only:
                report['status']='STATIC_MIDPOINT_CONTRACT_COMPLETE_NO_CACHE_LOAD_NO_GPU_NO_UPDATES'
            else:
                if int(os.environ.get('WORLD_SIZE','1'))!=1:
                    raise ValueError('Single-process finite check; do not launch torchrun/DDP')
                report['runtime']=a.checked_runtime(baseline)
                cache_dir=Path(args.cache_dir).resolve()
                if g.ready.sha(cache_dir/'cache_manifest.json')!=a.CACHE_MANIFEST_SHA:
                    raise ValueError('Requires exact reviewed aligned ROI cache')
                report['cache_loads']+=1
                payload,cache=g.checked_cache(cache_dir,identity['prior_identity']['g_identity'],samples)
                records=payload['records']; g.validate_records(records,samples)
                a.check_cache_reference(records,cache,baseline)
                prepared,assignments=prepare_records(records,samples,protocol['donor_mapping'])
                report.update(cache_reused=True,original_cpu_roi_bytes=g.cpu_roi_bytes(records),
                    cached_detector_state_sha256=payload['detector_state_before'],
                    all_supervision_points_inside_aligned_roi=True,
                    all_supervision_points_expectation_reachable=True)
                g.write_new(out/'cache_manifest.json',cache)
                g.write_new(out/'donor_assignment.json',dict(protocol=VERSION,assignments=assignments))
                progress(dict(stage='cache_verified',views=len(prepared),cache_loads=1))
                if not torch.cuda.is_available() or not 0<=args.gpu<torch.cuda.device_count():
                    raise ValueError('A valid logical CUDA device is required')
                torch.cuda.set_device(args.gpu); device=torch.device('cuda',args.gpu)
                report['gpu_devices_used']=[args.gpu]
                report['runtime']['gpu']=torch.cuda.get_device_name(args.gpu)
                print('Reusing CPU aligned ROIs; no B extraction. One logical GPU:',args.gpu,flush=True)
                g.seed_all(); head=m.SpatialMidpointHead()
                if sum(p.numel() for p in head.parameters())!=m.PARAMETER_COUNT:
                    raise ValueError('Declared spatial head parameter count differs')
                initial=deepcopy(head.state_dict()); del head
                batches=g.schedule(prepared)
                if len(batches)!=m.SETTINGS['steps_per_arm'] or batches!=baseline['minibatches']:
                    raise ValueError('Fit-only balanced schedule differs from reviewed baseline')
                report.update(minibatches=batches,arms={},head_costs={})
                for condition in CONDITIONS:
                    (result,maps),cost=g.measured(lambda:fit_arm(prepared,batches,initial,condition,device,progress),args.gpu)
                    report['arms'][condition]=result; report['head_costs'][condition]=cost
                    g.write_new(out/(condition+'.json'),dict(protocol=VERSION,identity=identity,
                        condition=condition,result=result,cost=cost,checkpoint_exported=False))
                    g.write_new(out/(condition+'.spatial.json'),dict(protocol=VERSION,
                        condition=condition,point_names=list(m.POINT_NAMES),views=maps))
                matched,shuffled=(report['arms'][c] for c in CONDITIONS)
                if matched['initial']!=shuffled['initial'] or matched['initial_state']!=shuffled['initial_state']:
                    raise ValueError('Matched/shuffled zero initial results or tensors differ')
                old_rows=baseline['arms']['ordinary']['initial']['rows']
                if any(r['pred']!=old['pred'] for r,old in zip(matched['initial']['rows'],old_rows)):
                    raise ValueError('Initial spatial head differs from saved frozen B')
                if report['head_updates_total']!=len(CONDITIONS)*m.SETTINGS['steps_per_arm']:
                    raise ValueError('Midpoint update budget differs')
                report['paired_vs_b']={c:paired_changes(report['arms'][c]['initial'],report['arms'][c]['final']) for c in CONDITIONS}
                report['paired_matched_vs_shuffled']=paired_changes(shuffled['final'],matched['final'])
                report['review']=review_conditions(matched['initial'],matched['final'],shuffled['final'])
                report['status']='TRAIN_MIDPOINT_CHECK_COMPLETE_REVIEW_REQUIRED'
            report['elapsed_seconds']=time.perf_counter()-started
            progress(dict(stage='complete',status=report['status'],head_updates_total=report['head_updates_total']))
            g.write_new(out/'completion.json',report)
        except Exception as error:
            report.update(status='FAILED_REVIEW_REQUIRED',error=type(error).__name__+': '+str(error),
                          elapsed_seconds=time.perf_counter()-started)
            progress(dict(stage='failed',error=report['error'],head_updates_total=report['head_updates_total']))
            destination=out/('failure.json' if (out/'completion.json').exists() else 'completion.json')
            try:
                g.write_new(destination,report)
            except (TypeError,ValueError) as serialization_error:
                g.write_new(destination,dict(protocol=VERSION,status=report['status'],error=report['error'],
                    serialization_error=str(serialization_error),head_updates_total=report['head_updates_total'],
                    formal_training=False,checkpoint_exported=False))
            publish_artifacts(out,report)
            raise
    publish_artifacts(out,report)
    print('Saved',out/'completion.json','status',report['status'],flush=True)
    return report


def main():
    run(parser().parse_args())


if __name__=='__main__':
    main()
