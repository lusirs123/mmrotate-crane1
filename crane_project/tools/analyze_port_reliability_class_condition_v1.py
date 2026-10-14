#!/usr/bin/env python3
"""Frozen TRAIN-only class-conditional support diagnostic. No fitting/reweighting."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.tools import run_port_reliability_video_transfer_v1 as loader
from crane_project.tools import review_port_reliability_video_transfer_v1 as independent
VERSION='port_reliability_class_condition_v1'
SETTINGS=dict(quantiles=[.25,.5,.75],quantile_method='linear',cutoff_ties='right',minimum_errors_for_descriptive_comparison=10,feature_dimensions=3,fit_calls=0,weight_updates=0,VAL_opened=False,TEST_opened=False)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n')
def auc(good,bad):
    return float(((bad[:,None]>good[None])+.5*(bad[:,None]==good[None])).mean()) if len(good) and len(bad) else None

def segments(rows,mask):
    count=0;previous=None;active=False
    for r,selected in zip(rows,mask):
        adjacent=previous is not None and r['sequence']==previous['sequence'] and r['frame_id']==previous['frame_id']+1 and not r['temporal'].get('acquisition_boundary',False)
        if selected and (not active or not adjacent):count+=1
        active=bool(selected);previous=r
    return count

def analyze(rows,model):
    before=deepcopy(rows)
    if any(r['pred'] is None for r in rows):raise ValueError('This bound complete TRAIN has no missing outputs')
    if len({r['image'] for r in rows})!=len(rows):raise ValueError('Duplicate frame')
    x=np.array([independent.descriptor(r,3) for r in rows]);y=np.array([independent.bad(r) for r in rows],dtype=bool)
    if not np.isfinite(x).all():raise ValueError('Nonfinite feature')
    edges=np.quantile(x,SETTINGS['quantiles'],axis=0).T
    bins=np.column_stack([np.searchsorted(edges[j],x[:,j],side='right') for j in range(3)])
    cells=bins[:,0]*16+bins[:,1]*4+bins[:,2]
    contribution=((x-np.array(model['mean']))/np.array(model['scale']))*np.array(model['weights'][1:])
    videos={};masks={};hist={}
    for seq in sorted({r['sequence'] for r in rows}):
        mask=np.array([r['sequence']==seq for r in rows]);masks[seq]=mask;parts={};features=[]
        for label,value in [('correct',False),('wrong',True)]:
            sel=mask&(y==value);n=int(sel.sum());h=np.bincount(cells[sel],minlength=64);hist[(seq,label)]=h
            parts[label]=dict(frames=n,continuous_segments=segments(rows,sel),joint_histogram=h.tolist(),features=[])
            for j in range(3):
                values=x[sel,j];parts[label]['features'].append(dict(mean=float(values.mean()) if n else None,quantiles=np.quantile(values,[0,.25,.5,.75,1]).tolist() if n else None,histogram=np.bincount(bins[sel,j],minlength=4).tolist()))
        for j in range(3):
            good=contribution[mask&~y,j];bad=contribution[mask&y,j]
            features.append(dict(feature=loader.core.fusion.simple.FEATURES[j],coefficient=model['weights'][j+1],contribution_AUC=auc(good,bad),wrong_minus_correct_mean=float(bad.mean()-good.mean()) if len(good) and len(bad) else None))
        videos[seq]=dict(domain=rows[np.flatnonzero(mask)[0]]['domain'],classes=parts,contributions=features,comparison_supported=int((mask&y).sum())>=10,segment_counts_are_not_independent_events=True)
    pairs=[]
    for i,a in enumerate(videos):
        for b in list(videos)[i+1:]:
            same=videos[a]['domain']==videos[b]['domain'];details={}
            for label in ('correct','wrong'):
                ha=hist[a,label];hb=hist[b,label];na=int(ha.sum());nb=int(hb.sum());shared=(ha>0)&(hb>0)
                marginal=[]
                for j in range(3):
                    aa=np.array(videos[a]['classes'][label]['features'][j]['histogram']);bb=np.array(videos[b]['classes'][label]['features'][j]['histogram'])
                    marginal.append(float(.5*np.abs(aa/na-bb/nb).sum()) if na and nb else None)
                details[label]=dict(joint_total_variation=float(.5*np.abs(ha/na-hb/nb).sum()) if na and nb else None,marginal_total_variation=marginal,shared_joint_cells=int(shared.sum()),a_shared_frames=int(ha[shared].sum()),b_shared_frames=int(hb[shared].sum()))
            # Both labels available in both videos in precisely the same joint cell.
            four=(hist[a,'correct']>0)&(hist[a,'wrong']>0)&(hist[b,'correct']>0)&(hist[b,'wrong']>0)
            support={v:{label:int(hist[v,label][four].sum()) for label in ('correct','wrong')} for v in (a,b)}
            flips=[j for j in range(3) if videos[a]['comparison_supported'] and videos[b]['comparison_supported'] and (videos[a]['contributions'][j]['contribution_AUC']-.5)*(videos[b]['contributions'][j]['contribution_AUC']-.5)<0]
            pairs.append(dict(a=a,b=b,within_domain=same,primary_real_comparison=same and videos[a]['domain']=='real',both_supported=videos[a]['comparison_supported'] and videos[b]['comparison_supported'],classes=details,both_labels_both_videos_joint_cells=int(four.sum()),four_way_support=support,descriptive_direction_flip_features=flips))
    if rows!=before:raise ValueError('Source mutated')
    return dict(frames=len(rows),correct=int((~y).sum()),wrong=int(y.sum()),frozen_model=model,quartile_edges=edges.tolist(),videos=videos,pairs=pairs,primary_supported_real_pairs=[p for p in pairs if p['primary_real_comparison'] and p['both_supported']],new_training_authorized_by_report=False,limitations=['Coarse joint-bin overlap is not proof of image/causal equivalence.','AUC sign differences are descriptive, not significance tests.','No weights learned; support cannot guarantee feasible bounded balancing or generalization.','Sparse-video AUC is descriptive only; continuous segments are not independent events.','Do not force geometry features independent of quality labels.'])

def review(rows,model,result):
    # Different scalar descriptor implementation and Python bucket counts.
    edges=result['quartile_edges'];buckets={};good=bad=0
    for r in rows:
        p=r['pred'];l,s=sorted(p[2:4],reverse=True);q=max(1e-6,min(1-1e-6,p[5]));x=[math.log(q/(1-q)),.5*math.log(l*s/(r['image_size'][0]*r['image_size'][1])),math.log(l/s)]
        err=any(abs(v-t)/t>.1 for v,t in zip(sorted(p[2:4]),sorted(r['gt'][2:4])))
        bad+=err;good+=not err;cell=[sum(v>=e for e in cut) for v,cut in zip(x,edges)];k=cell[0]*16+cell[1]*4+cell[2];key=(r['sequence'],'wrong' if err else 'correct');buckets.setdefault(key,[0]*64)[k]+=1
    if good!=result['correct'] or bad!=result['wrong']:raise ValueError('Independent labels differ')
    for seq,v in result['videos'].items():
        for label in ('correct','wrong'):
            if buckets.get((seq,label),[0]*64)!=v['classes'][label]['joint_histogram']:raise ValueError('Independent joint bins differ')
    # Deterministic complete replay catches mutated summaries/pairs; scalar check above
    # independently checks labels/descriptors/bin identities, not statistical guarantees.
    if analyze(rows,model)!=result:raise ValueError('Replayed summaries/pairs differ')
    return dict(passed=True,rows=len(rows),independent_scalar_label_bin_checks=len(rows),complete_summary_replay=True,no_statistical_guarantee=True)

def run(args):
    _,pins,absent,rows,policy=loader.load(args.allow_absent_frontend)
    manifest=json.loads((ROOT/'crane_project/tools/port_reliability_class_condition_v1_sources.json').read_text())['sources']
    for n,h in manifest.items():
        if sha(ROOT/n)!=h:raise ValueError('Source changed: '+n)
    pins=dict(pins,**manifest)
    if args.check_only:print('Frozen TRAIN and source checked');return
    out=args.out
    if out is None or out.is_symlink() or out.name!='result' or out.resolve().parent.parent!=(ROOT/'work_dirs'/VERSION).resolve():raise ValueError('Use unified version/run/result')
    if args.review_only:
        report=json.loads((out/'report.json').read_text())
        if report['input_pins']!=pins or report['settings']!=SETTINGS:raise ValueError('Report provenance changed')
        receipt=review(rows,policy['models']['size'],report['result'])
    else:
        if absent:raise ValueError('Server run requires frontend SHA check')
        out.mkdir(parents=True,exist_ok=False)
        write(out/'frozen_settings.json',SETTINGS)
        result=analyze(rows,policy['models']['size']);receipt=review(rows,policy['models']['size'],result)
        write(out/'report.json',dict(protocol=VERSION,settings=SETTINGS,input_pins=pins,result=result,fits=0,weight_updates=0,VAL_opened=False,TEST_opened=False,GPU_used=False,formal_policy_changed=False))
    for n,h in pins.items():
        if n not in absent and sha(ROOT/n)!=h:raise ValueError('Source changed during run')
    receipt.update(report_sha256=sha(out/'report.json'),absent_frontend=absent)
    write(out/args.receipt,receipt);print(receipt)
    print(json.dumps(json.loads((out/'report.json').read_text())['result']['primary_supported_real_pairs'],ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);p.add_argument('--allow-absent-frontend',action='store_true');p.add_argument('--check-only',action='store_true');p.add_argument('--review-only',action='store_true');p.add_argument('--receipt',default='server_review.json');run(p.parse_args())
