#!/usr/bin/env python3
"""Fixed cached single-scale quality evidence: TRAIN gate -> conditional VAL."""
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.tools import run_port_reliability_scale_asymmetry_v1 as parent
from crane_project.utils import port_reliability_opposite_border_v1 as method
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils import port_reliability_feature_ablation_v1 as ab
from crane_project.utils import port_reliability_state_continuity_v1 as states

PROTOCOL=ROOT/'crane_project/tools/port_reliability_opposite_border_v1_protocol.json'
SOURCES=ROOT/'crane_project/tools/port_reliability_opposite_border_v1_sources.json'
CACHE='work_dirs/port_geometry_midpoint_formal_v1_roi_cache'
CACHE_PINS={'cache_manifest.json':'046c5998dee0ba3703f1ae4e08fc6a02e9804d216ea12241357f69f2a4afd1e3',
 'train_s1.pt':'903ae2a2146665e6caa557fee6bed4cd6cadd51f842e6ee9e13bf82482976ec0',
 'val_s1.pt':'88a189cd707e2406acafbfabb198cbb71d72e118617a2cb1ed93702fff1c70db'}
BASE_KEYS=('image','sequence','domain','split','frame_id','image_size','gt','pred',
           'train_angle_eligible','angle_axis_well_defined','b_original','reliability_role',
           'size_risks','original_simple_decision','sample_role')
SOURCE_FILES=parent.SOURCE_FILES+[
 'crane_project/tools/port_reliability_scale_asymmetry_v1_sources.json',
 'crane_project/utils/port_reliability_opposite_border_v1.py',
 'crane_project/tools/run_port_reliability_opposite_border_v1.py',
 'crane_project/tools/port_reliability_opposite_border_v1_protocol.json',
 'crane_project/utils/port_structure_reliability_v1.py',
 'crane_project/utils/port_geometry_refine_g_v1.py',
 'crane_project/utils/port_geometry_midpoint_v1.py',
 'tests/test_port_reliability_opposite_border_v1.py',
 'tools/run_port_reliability_opposite_border_v1.sh']
sha=parent.sha
write=parent.write


def checked_sources():
    p=json.loads(PROTOCOL.read_text()); manifest=json.loads(SOURCES.read_text())
    actual={name:sha(ROOT/name) for name in SOURCE_FILES}
    if (manifest!=dict(protocol=method.VERSION,sources=actual)
        or p['protocol']!=method.VERSION or p['sampling']!=method.SAMPLING
        or p['schemas']!={k:list(v) for k,v in method.SCHEMAS.items()}
        or p['fitting']!=method.FIT or p['cache_pins']!=CACHE_PINS
        or p['control_pins']!=parent.CONTROL_PINS or p['controls']!=list(method.CONTROLS)):
        raise ValueError('Source/contract changed before fitting')
    parent.checked_sources()
    return p,dict(manifest_sha256=sha(SOURCES),sources=actual)


def load_train_rows():
    pins=parent.checked_control_artifacts()
    report=json.loads((ROOT/parent.CONTROL/'report.json').read_text())
    if report['sources']!=parent.prior.checked_sources()[1] or report['input_sha256']!=parent.prior.PINS:
        raise ValueError('Frozen standard-M parent source identity differs')
    with gzip.open(ROOT/parent.CONTROL/'scored_TRAIN.jsonl.gz','rt') as f:
        saved=[json.loads(x) for x in f]
    rows=[{k:deepcopy(r[k]) for k in BASE_KEYS} for r in saved]
    states.checked_order(rows)
    expected={('fit','real'):1377,('fit','sim'):565,('probe','real'):351,('probe','sim'):141,
              ('legacy','real'):32,('legacy','sim'):32,('purged','real'):50,('purged','sim'):10}
    if (len(rows)!=2558 or Counter((r['sample_role'],r['domain']) for r in rows)!=expected
        or any(r['reliability_role']!='train' for r in rows)):
        raise ValueError('Original TRAIN roles/counts differ')
    old_models=json.loads((ROOT/parent.CONTROL/'models.json').read_text())
    model=old_models['refit_simple']; old_scores={r['image']:r['experiment_risks']['refit_simple'] for r in saved}
    max_delta=0.
    for r in rows:
        d=r['original_simple_decision']; pred=r['pred']
        if d['final_box_original']!=pred or d['center_accepted']!=(pred is not None):
            raise ValueError('Original output/center identity differs')
        x=None if pred is None else simple.descriptor(pred,r['image_size'])
        value=None if x is None else float(method.risk(model,x,'refit_simple'))
        old=old_scores[r['image']]
        if (old is None)!=(value is None) or (old is not None and abs(old-value)>1e-12):
            raise ValueError('Same-fit frozen three-feature control differs')
        if value is not None:max_delta=max(max_delta,abs(value-old))
    fit=[r for r in rows if r['sample_role']=='fit' and r['pred'] is not None]
    x=np.array([simple.descriptor(r['pred'],r['image_size']) for r in fit])
    y=np.array([ab.size_bad(r) for r in fit],dtype=float)
    mean=x.mean(0); scale=np.where(x.std(0)<1e-8,1.,x.std(0))
    if (not np.allclose(mean,model['mean'],rtol=0,atol=1e-12)
        or not np.allclose(scale,model['scale'],rtol=0,atol=1e-12)
        or model['bad_rows']!=int(y.sum()) or model['train_rows']!=len(fit)):
        raise ValueError('Same-fit control training identity differs')
    design=np.column_stack((np.ones(len(x)),(x-model['mean'])/model['scale']))
    sw=np.where(y==1,.5/y.sum(),.5/(len(y)-y.sum()))
    loss,grad,_=simple.logistic_objective(np.asarray(model['weights']),design,y,sw,.1)
    if abs(loss-model['final_objective'])>1e-12 or np.max(np.abs(grad))>1e-8:
        raise ValueError('Frozen same-fit optimum does not reproduce')
    return rows,model,old_scores,dict(artifact_sha256=pins,maximum_risk_replay_delta=max_delta,
        frozen_control_refits=0,exact_saved_control_scores=True,main_source='sealed standard M TRAIN only; auxiliary fields discarded')


def load_val_rows():
    pin=parent.prior.PRIMARY+'/scored_TRAIN_VAL.jsonl.gz'
    if sha(ROOT/pin)!=parent.prior.PINS[pin]: raise ValueError('Frozen original VAL bytes changed')
    with gzip.open(ROOT/pin,'rt') as f:
        rows=[r for r in map(json.loads,f) if r['reliability_role']=='val']
    rows=[{k:deepcopy(r[k]) for k in BASE_KEYS if k!='sample_role'} for r in rows]
    for r in rows:r['sample_role']='val'
    if len(rows)!=887 or Counter(r['domain'] for r in rows)!={'real':375,'sim':512}:
        raise ValueError('Full fixed VAL identities differ')
    states.checked_order(rows)
    return rows,{pin:parent.prior.PINS[pin]}


def checked_cache_path(path):
    path=Path(path).resolve()
    if sha(path/'cache_manifest.json')!=CACHE_PINS['cache_manifest.json']:
        raise ValueError('Original formal cache manifest changed')
    manifest=json.loads((path/'cache_manifest.json').read_text())
    if (manifest['status']!='COMPLETE_FROZEN_B_TRAIN_VAL_CACHE'
        or any(manifest['files'][k]!=v for k,v in CACHE_PINS.items() if k!='cache_manifest.json')):
        raise ValueError('Original formal cache contract differs')
    return path,manifest


def checked_gt_pair(cached,original):
    """Keep original labels; tolerate only float32 periodic-angle rounding.

    Centers and both sizes must match exactly. Cache annotation theta was
    rounded before a different le90 wrap; it is never used in this descriptor.
    """
    a=np.asarray(cached,dtype=float);b=np.asarray(original,dtype=float)
    if a.shape!=(5,) or b.shape!=(5,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid paired GT')
    delta=abs((a[4]-b[4]+np.pi/2)%np.pi-np.pi/2)
    if not np.array_equal(a[:4],b[:4]) or delta>2e-6:
        raise ValueError('GT center/size or periodic angle pairing differs')
    return float(delta)


def collect(rows,cache_dir,stage):
    import torch
    torch.set_num_threads(1)
    cache_dir,manifest=checked_cache_path(cache_dir)
    shard='train_s1.pt' if stage=='train' else 'val_s1.pt'
    path=cache_dir/shard
    if sha(path)!=CACHE_PINS[shard]:raise ValueError('Sealed standard-scale shard changed')
    # Trusted byte-pinned repository cache only; never arbitrary uploaded pickle.
    try:payload=torch.load(str(path),map_location='cpu',weights_only=False)
    except TypeError:payload=torch.load(str(path),map_location='cpu')
    if payload['identity']!=manifest['identity'] or payload['detector_state']!=manifest['detector_state']:
        raise ValueError('Frozen cache/model identity differs')
    cached=payload['records']; index={r['image']:r for r in cached}
    if len(index)!=len(cached) or set(index)!={r['image'] for r in rows}:
        raise ValueError('Complete single-scale cached frame identities differ')
    result=[]; fields_all=[]; supports_all=[]; sampling=[];gt_angle_delta=0.
    for n,r in enumerate(sorted(rows,key=lambda r:r['image'])):
        c=index[r['image']]
        if (c['scale']!=1. or c['role']!=stage
            or any(c[k]!=r[k] for k in ('image','sequence','domain','split','frame_id','image_size'))
            or c['gt_original'].shape!=(1,5)):
            raise ValueError('Standard view/metadata/GT cache pairing differs')
        gt_angle_delta=max(gt_angle_delta,checked_gt_pair(c['gt_original'][0].tolist(),r['gt']))
        for k in ('roi','support','boxes_original','boxes_model','scale_xy','gt_original'):
            t=c[k]
            if t.device.type!='cpu' or t.requires_grad or not bool(torch.isfinite(t).all()):
                raise ValueError('Cache must be finite detached CPU only')
        count=len(c['boxes_original'])
        if count not in (0,1) or (count==0)!=(r['pred'] is None):
            raise ValueError('Main/base output identity differs')
        if (c['boxes_original'].shape!=(count,6)
            or c['roi'].shape!=(count,256,9,9) or c['support'].shape!=(count,1,9,9)
            or c['boxes_model'].shape!=(count,5) or c['scale_xy'].shape!=(count,2)):
            raise ValueError('Native cache tensor shape differs')
        if count:
            b=c['boxes_original'][0].tolist()
            if b!=r['b_original']:raise ValueError('Exact original B cache identity differs')
            bm=c['boxes_model'][0].tolist(); xy=c['scale_xy'][0].tolist()
            fields=method.summarize_channels(c['roi'][0].numpy())
            support=c['support'][0,0].numpy().astype(float)
            evidence=method.sample_evidence(r['pred'],b,bm,xy,fields,support)
            proof={k:np.asarray(evidence[k]).tolist() for k in ('ordinary_support','border_support',
                'offsets_original','axis_resolution_qualified','cache_band_separation_Linf',
                'native_P3_band_separation')}
            proof['M_short_native_P3_cells']=float(evidence['M_short_native_P3_cells'])
            if (min(proof['cache_band_separation_Linf'])<1-1e-10
                or min(proof['native_P3_band_separation'])<1-1e-10):
                raise ValueError('Three bands cannot claim sub-grid evidence')
        else:
            b=bm=xy=evidence=proof=None; fields=np.zeros((2,9,9)); support=np.zeros((9,9))
        x=method.descriptors(r['pred'],r['image_size'],evidence)
        result.append(dict(r,base_original=b,base_model=bm,scale_xy=xy,
            experiment_features={k:None if v is None else v.tolist() for k,v in x.items()},sampling=proof))
        fields_all.append(fields); supports_all.append(support)
        if proof is not None:sampling.append((r['sample_role'],proof))
        if n%500==0:print('fixed cached fields',stage,n+1,'/',len(rows),flush=True)
    source={str(path.relative_to(ROOT)):sha(path),'cache_manifest.json':sha(cache_dir/'cache_manifest.json')}
    if source[str(path.relative_to(ROOT))]!=CACHE_PINS[shard]:raise ValueError('Cache mutated during extraction')
    fit_proofs=[p for role,p in sampling if role=='fit']
    if stage=='train' and not any(all(p['axis_resolution_qualified']) and min(p['border_support'])>0 for p in fit_proofs):
        raise ValueError('No fit frame has separated supported inner/edge/outer bands on both axes; stop before fitting')
    arrays=dict(image_ids=np.array([r['image'] for r in result]),fields=np.array(fields_all),support=np.array(supports_all))
    return result,arrays,dict(source_sha256=source,full_P3_available=False,stored_grid=[9,9],
        GT_centers_and_sizes_exact=True,maximum_GT_periodic_angle_delta_rad=gt_angle_delta,
        original_GT_labels_preserved=True,
        field_summary='fixed channel mean/RMS, no learned channel projection',
        interpolation_does_not_add_information=True,no_GT_or_center_eligibility_filter=True,
        source_valid_for='spatial organization test only; 10% error discrimination unproven',
        both_axes_resolved_frames=sum(all(p['axis_resolution_qualified']) for _,p in sampling),
        all_twelve_bands_supported_frames=sum(min(p['border_support'])>0 for _,p in sampling),
        short_axis_unresolved_frames=sum(not p['axis_resolution_qualified'][1] for _,p in sampling),
        role_support={role:dict(frames=len(v),both_axes_resolved=sum(all(p['axis_resolution_qualified']) for p in v),
            all_bands_supported=sum(min(p['border_support'])>0 for p in v))
            for role in sorted({role for role,_ in sampling}) for v in [[p for a,p in sampling if a==role]]},
        runtime=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,GPU_used=False))


def engineering_check(x,y):
    x=np.asarray(x);y=np.asarray(y,dtype=float); scale=np.where(x.std(0)<1e-8,1.,x.std(0))
    design=np.column_stack((np.ones(len(x)),(x-x.mean(0))/scale)); n=design.shape[1]
    sw=np.where(y==1,.5/y.sum(),.5/(len(y)-y.sum()))
    objective=lambda w:simple.logistic_objective(w,design,y,sw,.1)
    w=np.linspace(-.1,.1,n); loss,g,h=objective(w); ng=np.zeros(n); nh=np.zeros((n,n)); eps=1e-5
    for j in range(n):
        d=np.zeros(n);d[j]=eps;a,b=objective(w+d),objective(w-d)
        ng[j]=(a[0]-b[0])/(2*eps);nh[:,j]=(a[1]-b[1])/(2*eps)
    gd=float(np.max(abs(g-ng)));hd=float(np.max(abs(h-nh)))
    initial,g0,h0=objective(np.zeros(n));after=objective(-np.linalg.solve(h0,g0))[0]
    if gd>1e-7 or hd>1e-7 or not after<initial:raise ValueError('Actual fixed objective/derivatives fail before fitting')
    return dict(passed=True,parameters=n,gradient_FD_max=gd,hessian_FD_max=hd,
        zero_init_loss=initial,one_Newton_update_loss=after,zero_init_gradient=g0.tolist())


def score_rows(rows,models,old_scores=None):
    scores={m:{} for m in (method.ARM,)+method.CONTROLS}
    for r in rows:
        i=r['image']
        for arm in method.SCHEMAS:
            x=r['experiment_features'][arm]
            if x is None:value=None
            elif arm=='refit_simple' and old_scores is not None:value=old_scores[i]
            else:value=float(method.risk(models[arm],x,arm))
            scores[arm][i]=value
        scores['full_simple'][i]=r['size_risks']['full_simple']
        scores['score_only'][i]=None if r['pred'] is None else 1-r['pred'][5]
    return scores


def checked_out(path,stage):
    p=Path(path).resolve()
    if p.parent.parent!=(ROOT/'work_dirs'/method.VERSION).resolve() or p.name!=stage+'_check':
        raise ValueError('Use one work_dirs/version/RUN_ID/'+stage+'_check')
    if p.exists():raise FileExistsError('Refuse overwrite')
    return p


def frozen_train(path):
    path=Path(path).resolve()
    if path.name!='train_check' or path.parent.parent!=(ROOT/'work_dirs'/method.VERSION).resolve():
        raise ValueError('Use current version frozen TRAIN result')
    receipt=json.loads((path/'completion.json').read_text());report=json.loads((path/'report.json').read_text())
    if any(sha(path/name)!=digest for name,digest in receipt['artifacts'].items()):
        raise ValueError('Frozen TRAIN result changed')
    if (receipt['protocol']!=method.VERSION or report['protocol']!=method.VERSION
        or not report['probe_gate']['passed'] or report['sources']!=checked_sources()[1]
        or report['models']!=json.loads((path/'models.json').read_text())):
        raise ValueError('VAL forbidden unless fixed TRAIN passed and artifacts/sources unchanged')
    return report


def run(args):
    out=checked_out(args.out,args.stage);protocol,sources=checked_sources()
    if args.stage=='train':
        base_rows,base_model,old_scores,control_proof=load_train_rows()
        rows,arrays,cache=collect(base_rows,args.cache_dir,'train')
        before=simple.fingerprint(rows)
        fit=[r for r in rows if r['sample_role']=='fit' and r['pred'] is not None]
        labels=[ab.size_bad(r) for r in fit]
        if len(fit)!=1942 or sum(labels)!=54:raise ValueError('Exact finite fit labels changed')
        engineering={arm:engineering_check([r['experiment_features'][arm] for r in fit],labels)
                     for arm in ('ordinary_roi',method.ARM)}
        # Exactly two new fixed fits. No network, checkpoint selection or control refit.
        models=dict(refit_simple=base_model)
        for arm in ('ordinary_roi',method.ARM):
            models[arm]=method.fit([r['experiment_features'][arm] for r in fit],labels,arm)
        scores=score_rows(rows,models,old_scores)
        points={m:method.fit_cutoff(fit,scores[m]) for m in scores}
        roles=('fit','probe','legacy','purged')
        input_proof=control_proof; fit_calls=2
    else:
        train=frozen_train(args.train_result)
        if Path(args.train_result).resolve().parent!=out.parent:raise ValueError('Keep one experiment parent')
        base_rows,input_proof=load_val_rows()
        rows,arrays,cache=collect(base_rows,args.cache_dir,'val');before=simple.fingerprint(rows)
        models=train['models'];points=train['fit_points'];engineering=train['engineering']
        scores=score_rows(rows,models);roles=('val',);fit_calls=0
    diagnostics={};scored=[];components={};ordinary_diagnostics={};fixed_points={}
    for role in roles:
        subset=[r for r in rows if r['sample_role']==role]
        diagnostics[role]=method.describe(subset,scores,points[method.ARM]['risk_le'])
        ordinary_diagnostics[role]=method.describe(subset,scores,points['ordinary_roi']['risk_le'],'ordinary_roi')
        fixed_points[role]={m:states.summarize(subset,{r['image'] for r in subset if r['pred'] is not None and scores[m][r['image']]<=points[m]['risk_le']}) for m in scores}
        decisions={}
        for r in subset:
            d=method.decide(models[method.ARM],points[method.ARM]['risk_le'],r['original_simple_decision'],r['experiment_features'][method.ARM])
            expected=deepcopy(r['original_simple_decision'])
            if r['pred'] is not None:
                expected['risks']['size']=scores[method.ARM][r['image']];expected['size_accepted']=scores[method.ARM][r['image']]<=points[method.ARM]['risk_le']
            if d!=expected:raise ValueError('Only size risk/flag may change')
            decisions[r['image']]=d
            scored.append(dict(r,experiment_risks={m:scores[m][r['image']] for m in scores},candidate_diagnostic_decision=d))
        components[role]=parent.prior.components(subset,decisions)
    gate=method.gate(diagnostics['probe' if args.stage=='train' else 'val'])
    if args.stage=='val':gate['gate_scope']='complete_VAL_frozen_fit_models_and_cutoffs'
    status=('TRAIN_CAPABILITY_PASS_FROZEN_VAL_ALLOWED' if gate['passed'] else 'TRAIN_CAPABILITY_FAILED_STOP') if args.stage=='train' else ('VAL_FINITE_PASS_NOT_DEPLOYED' if gate['passed'] else 'VAL_FINITE_FAILED_STOP')
    if simple.fingerprint(rows)!=before or checked_sources()[1]!=sources:raise ValueError('Input/source mutation')
    report=dict(protocol=method.VERSION,stage=args.stage,contract=protocol,sources=sources,
        input_proof=input_proof,cache=cache,engineering=engineering,models=models,fit_points=points,
        diagnostics=diagnostics,ordinary_roi_diagnostics=ordinary_diagnostics,fixed_fit_points=fixed_points,
        three_components=components,probe_gate=gate if args.stage=='train' else None,
        VAL_gate=gate if args.stage=='val' else None,conclusion=status,selected_arm=None,
        frontend='B24+sigma1.5/epoch03',original_policy_sha256=protocol['original_policy_sha256'],
        rows=len(rows),role_counts=dict(Counter(r['sample_role'] for r in rows)),
        fit_calls=fit_calls,control_refits=0,threshold_refits=5 if args.stage=='train' else 0,
        threshold_role='fit_only',probe_threshold_fits=0,VAL_threshold_fits=0,
        detector_updates=0,midpoint_updates=0,feature_network_updates=0,new_inference_calls=0,
        auxiliary_used=False,TEST_read=False,VAL_evaluated=args.stage=='val',
        policy_modified=False,geometry_modified=False,depth_formula_modified=False,
        exposed_probe=True,independent_confirmation=False)
    out.mkdir(parents=True);write(out/'report.json',report);write(out/'models.json',models)
    np.savez_compressed(str(out/'fixed_fields.npz'),**arrays)
    with gzip.open(out/'scored_rows.jsonl.gz','xt') as f:
        for r in sorted(scored,key=lambda r:r['image']):f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    write(out/'completion.json',dict(protocol=method.VERSION,stage=args.stage,status=status,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT),text=True).strip(),
        artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file()},TEST_read=False))
    print(json.dumps(dict(status=status,gate=gate,out=str(out)),ensure_ascii=False),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('train','val'),required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--cache-dir',type=Path,default=ROOT/CACHE)
    p.add_argument('--train-result',type=Path)
    run(p.parse_args())
