#!/usr/bin/env python3
"""Verify S025 artifacts; export preselected natural-VAL and TRAIN diagnostics.

No re-selection, checkpoint loading, model evaluation, TEST or depth access.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from datetime import datetime,timezone


def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()
def rows(p):
    with gzip.open(p,'rt') as f:return [json.loads(s) for s in f]
def mean(xs):return math.fsum(xs)/len(xs) if xs else None
def csv_write(p,rs):
    fields=list(dict.fromkeys(k for r in rs for k in r))
    with p.open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rs)


def temporal(v):
    def avg(key,den):
        cells=[r for r in v.values() if r[key] is not None and r[den]];n=sum(r[den] for r in cells)
        return sum(r[key]*r[den] for r in cells)/n if n else None
    mx=[r['mcml_max'] for r in v.values() if r['mcml_max'] is not None]
    return dict(DFR=avg('dfr_percent_per_frame','dfr_adjacent_pairs'),ACI=avg('aci','aci_adjacent_pairs'),
        MCML_max=max(mx) if mx else None,MCML_mean=avg('mcml_mean','mcml_segments'),
        TDR_correct=sum(r['tdr_w10']['numerator'] for r in v.values()),TDR_denominator=sum(r['tdr_w10']['denominator'] for r in v.values()))


def single_edges(rs):
    result=[]
    for domain in ('real','sim'):
        for scale in sorted({r['scale'] for r in rs}):
            cell=[r for r in rs if r['domain']==domain and r['scale']==scale and r['midpoint'] is not None]
            for j,edge in enumerate(('long','short')):
                groups={g:[] for g in ('undersized','oversized','near3')}
                for r in cell:
                    gt=sorted(r['gt'][2:4],reverse=True)[j];m=sorted(r['midpoint'][2:4],reverse=True)[j];p=sorted(r['edge_residual'][2:4],reverse=True)[j]
                    a,b=m/gt-1,p/gt-1;group='near3' if abs(a)<=.03 else 'undersized' if a<0 else 'oversized'
                    groups[group].append((a,b,p/m-1))
                for group,values in groups.items():
                    a,b=[v[0] for v in values],[v[1] for v in values]
                    result.append(dict(domain=domain,scale=scale,edge=edge,group=group,count=len(values),
                        M_mae_pct=100*mean(list(map(abs,a))) if a else None,candidate_mae_pct=100*mean(list(map(abs,b))) if b else None,
                        M_bias_pct=100*mean(a) if a else None,candidate_bias_pct=100*mean(b) if b else None,
                        improved=sum(abs(y)<abs(x)-1e-12 for x,y,u in values),harmed=sum(abs(y)>abs(x)+1e-12 for x,y,u in values),
                        enlarge=sum(u>1e-12 for x,y,u in values),enlarge_harmed=sum(u>1e-12 and abs(y)>abs(x)+1e-12 for x,y,u in values),
                        shrink=sum(u < -1e-12 for x,y,u in values),shrink_harmed=sum(u < -1e-12 and abs(y)>abs(x)+1e-12 for x,y,u in values)))
    return result


def common_probe(directory,output,cache_dir,i_run):
    # Preselected heads only; same seven TRAIN inputs for both frozen heads.
    import torch
    from argparse import Namespace
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from crane_project.tools import run_port_geometry_size_proposal_v1 as t
    if output.exists():raise ValueError('Refuse existing probe')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='3':raise ValueError('Fourth physical GPU only')
    done=read(directory/'completion.json')
    if not done['status'].startswith('COMPLETE_'):raise ValueError('Training must finish before common probe')
    torch.set_num_threads(1);torch.cuda.set_device(0);t.old.seed_all(torch)
    torch.cuda.set_per_process_memory_fraction(1100/(torch.cuda.get_device_properties(0).total_memory/2**20),0)
    sources=t.check_sources();used={};train,val,cache,_=t.load_cache(Namespace(cache_dir=cache_dir,i_run=i_run),torch,used)
    packed=t.old.pack(train,torch.device('cuda:0'),torch);inventory={};picks={};samples=[t.memory()]
    index=read(directory/'artifacts.json')['files']
    for arm in ('repeat7','proposal7'):
        chosen=done['selections'][arm]['selected'] or done['selections'][arm]['representative']
        name=arm+'/head_epoch_%02d.pth'%chosen['epoch'];path=directory/name
        if sha(path)!=index[name] or sha(path)!=chosen['checkpoint']['sha256']:raise ValueError('Selected checkpoint SHA differs')
        payload=t.sealed.torch_load(torch,path)
        if payload['protocol']!=t.VERSION or payload['arm']!=arm or payload['epoch']!=chosen['epoch']:raise ValueError('Frozen probe identity')
        h=t.model.GeometryConditionSizeHead().cuda();h.load_state_dict(payload['head_state']);h.eval().requires_grad_(False)
        if t.sealed.state_digest(h)!=payload['head_digest']:raise ValueError('Selected head digest')
        inventory[arm]=t.inventory(h,packed,train,torch,'proposal7',True)
        if t.sealed.state_digest(h)!=payload['head_digest']:raise ValueError('Probe mutated head')
        picks[arm]=dict(epoch=chosen['epoch'],sha256=sha(path));samples.append(t.memory());del h
    value=dict(status='FROZEN_COMMON_TRAIN_PROPOSAL_PROBE_COMPLETE',selections=picks,query='Same proposal7 on complete TRAIN paired views; no new epoch choice',inventories=inventory,
        script_sha256=sha(Path(__file__)),training_source_manifest=sources['manifest_sha256'],training_updates=0,val_evaluations=0,test_access=False,depth_access=False,memory_samples=samples)
    with output.open('x') as f:json.dump(value,f,indent=2)
    print(json.dumps(dict(status=value['status'],selections=picks,memory_samples=samples),indent=2))


def run(directory,output):
    index=read(directory/'artifacts.json')['files']
    for f,h in index.items():
        if sha(directory/f)!=h:raise ValueError('Artifact changed '+f)
    done=read(directory/'completion.json')
    if not done['status'].startswith('COMPLETE_') or done['complete_val_epochs_per_arm']!=24 or done['formal_updates_total']!=30672:raise ValueError('Incomplete two-arm run')
    if output.exists():raise ValueError('Refuse existing analysis')
    allval=[];train=[];valedges=[];recovery=[];chosen=[]
    for arm in ('repeat7','proposal7'):
        pick=done['selections'][arm]['selected'] or done['selections'][arm]['representative']
        chosen.append(dict(arm=arm,epoch=pick['epoch'],intrinsic_adoption=pick['adoption']['pass_all'],failed=pick['adoption']['failed'],checkpoint_sha256=pick['checkpoint']['sha256']))
        for epoch in range(1,25):
            summary=read(directory/arm/('val_epoch_%02d.summary.json'%epoch));comps=read(directory/arm/('val_epoch_%02d.components.json'%epoch));nr=read(directory/arm/('val_epoch_%02d.near.json'%epoch))
            for group,methods in summary.items():
                c=comps[group]['edge_residual'];m=methods['edge_residual'];s=m['sizes']
                if c['frames']!=c['outputs']+c['missing']:raise ValueError('Denominator mismatch')
                r=dict(arm=arm,epoch=epoch,group=group,**c,mean_RIoU=m['mean_riou_full_frame'],**temporal(m['temporal']),
                    center_rate=c['center_correct']/c['outputs'] if c['outputs'] else None,size_rate=c['size_correct']/c['outputs'] if c['outputs'] else None,
                    angle_rate=c['angle_correct']/c['angle_eligible_outputs'] if c['angle_eligible_outputs'] else None,
                    output_coverage=c['outputs']/c['frames'],center_full=c['center_correct']/c['frames'],size_full=c['size_correct']/c['frames'],
                    angle_full_eligible=c['angle_correct']/c['angle_eligible_frames'] if c['angle_eligible_frames'] else None,
                    a_meanabs=s['a']['abs_mae'],r_meanabs=s['r']['abs_mae'],
                    wrong_to_right=methods['paired']['wrong_to_right'],right_to_wrong=methods['paired']['right_to_wrong'],new_output=methods['paired']['new_output'],lost_output=methods['paired']['lost_output'])
                for edge in ('long','short'):
                    for k in ('mae','p95','bias'):r[edge+'_'+k+'_pct']=100*s['edges'][edge][k]
                    if group in nr:
                        r[edge+'_fixed_M_near_n']=nr[group][edge]['count'];r[edge+'_near_mae_pct']=100*nr[group][edge]['methods']['edge_residual']['mae']
                r.update({('AP'+('50' if th=='0.5' else '75')):val for th,val in m['cache_gt_ap11'].items()})
                allval.append(r)
            for split,dest in [('train',train),('val',valedges)]:
                saved=rows(directory/arm/(split+'_epoch_%02d.rows.jsonl.gz'%epoch))
                if len(saved)!=(5116 if split=='train' else 887):raise ValueError('Incomplete '+split)
                dest.extend(dict(arm=arm,epoch=epoch,**r) for r in single_edges(saved))
        for cell,r in read(directory/arm/'chosen_proposal_recovery.json').items():
            recovery.append(dict(arm=arm,chosen_epoch=pick['epoch'],cell=cell,count=r['count'],before_long_pct=100*r['before_abs_sum'][0]/r['count'],before_short_pct=100*r['before_abs_sum'][1]/r['count'],after_long_pct=100*r['after_abs_sum'][0]/r['count'],after_short_pct=100*r['after_abs_sum'][1]/r['count'],long_improved=r['improved'][0],long_harmed=r['harmed'][0],short_improved=r['improved'][1],short_harmed=r['harmed'][1]))
    output.mkdir(parents=True);csv_write(output/'all_epochs_natural_val.csv',allval);csv_write(output/'all_epochs_train_single_edges.csv',train);csv_write(output/'all_epochs_val_single_edges.csv',valedges);csv_write(output/'chosen_proposal_recovery.csv',recovery)
    result=dict(status='TWO_ARM_S025_ARTIFACTS_VERIFIED',utc=datetime.now(timezone.utc).isoformat(),analysis_script_sha256=sha(Path(__file__)),source_artifacts_sha256=sha(directory/'artifacts.json'),files_verified=len(index),chosen=chosen,run_status=done['status'],mechanism=done['mechanism'],training_updates=0,inference_calls=0,test_access=False,depth_access=False)
    (output/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');(output/'artifacts.json').write_text(json.dumps(dict(files={p.name:sha(p) for p in output.iterdir() if p.is_file()}),indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--common-probe',action='store_true');p.add_argument('--cache-dir',type=Path);p.add_argument('--i-run',type=Path);args=p.parse_args()
    if args.common_probe:common_probe(args.directory,args.out,args.cache_dir,args.i_run)
    else:run(args.directory,args.out)
