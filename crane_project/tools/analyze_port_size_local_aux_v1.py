#!/usr/bin/env python3
"""S035 final local auxiliary contrast with verified reused V control."""
import argparse
import gzip
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.tools.run_port_size_native_train_v1 import sha,read,write,eligibility,ranking,near_joint
from crane_project.tools.analyze_port_size_local_vectors_v1 import flat,csv_write,numerical_equal
from crane_project.tools.analyze_port_size_correction_behavior_v1 import classify,summary as behavior_summary,KINDS


def load(path):
    with gzip.open(path,'rt') as f:return [__import__('json').loads(x) for x in f]


def run(source,control_source,out):
    if out.exists() or out.parent!=source.parent:raise ValueError('Fresh result-only analysis')
    for directory in (source,control_source):
        for n,s in read(directory/'artifacts.json')['files'].items():
            if sha(directory/n)!=s:raise ValueError('Result SHA '+n)
    t=read(source/'training.json');done=read(source/'finished.json')
    if t['status']!=done['status'] or done['status']!='COMPLETE' or t['formal_updates']!=15336:raise ValueError('Incomplete finite candidate')
    if t['control_artifacts_sha256']!=sha(control_source/'artifacts.json'):raise ValueError('Control result identity')
    from crane_project.utils import port_geometry_size_distribution_metrics_v1 as metrics
    from crane_project.tools.run_port_geometry_size_condition_v1 import components
    prior_verification=read(control_source.parent/'analysis_20261018_v1/analysis.json')
    if prior_verification['source_artifacts_sha256']!=sha(control_source/'artifacts.json') or prior_verification['verification']['epochs_recomputed']!=48:raise ValueError('Prior verified control identity')
    records=[];selected={};behavior=[];paired=[];verification=dict(val_epochs_recomputed=0,reused_already_verified_control_epochs=24,prior_control_verification_sha256=sha(control_source.parent/'analysis_20261018_v1/analysis.json'),maximum_cross_platform_riou_difference=0.)
    all_models={'size_only':t['control'],'size_plus_local':t};cache={}
    for arm,report in all_models.items():
        epochs=report['epochs'];good=[]
        if [e['epoch'] for e in epochs]!=list(range(1,25)):raise ValueError('Missing full VAL epoch')
        for ep in epochs:
            if ranking(ep['metrics'],ep['epoch'])!=ep['selection_key'] or eligibility(ep['metrics'],ep['diagnostics'])!=ep['eligibility']:raise ValueError('Rule drift')
            if ep['eligibility']['pass_all']:good.append(ep)
            rows=load((control_source/'size_only' if arm=='size_only' else source)/('epoch_%02d_val.rows.jsonl.gz'%ep['epoch']))
            recompute=metrics.summary(rows,cache) if arm=='size_plus_local' else None
            if recompute is not None and (components(rows)!=ep['components'] or not numerical_equal(near_joint(rows),ep['diagnostics'])):raise ValueError('Geometric recomputation')
            for group in ep['metrics']:
                for method in (metrics.METHODS if recompute is not None else ()):
                    a,b=recompute[group][method],ep['metrics'][group][method]
                    delta=abs(a['mean_riou_full_frame']-b['mean_riou_full_frame']);verification['maximum_cross_platform_riou_difference']=max(delta,verification['maximum_cross_platform_riou_difference'])
                    if delta>5e-6 or not numerical_equal(a['sizes'],b['sizes']) or a['cache_gt_ap11']!=b['cache_gt_ap11'] or not numerical_equal(a['temporal'],b['temporal'],1e-4):raise ValueError('Numeric protocol drift')
                r=flat(group,'edge_residual',ep['metrics'][group]['edge_residual'],ep['components'][group]['edge_residual'],arm=arm,epoch=ep['epoch'],eligible=ep['eligibility']['pass_all'])
                r.update(wrong_to_right=ep['metrics'][group]['paired']['wrong_to_right'],right_to_wrong=ep['metrics'][group]['paired']['right_to_wrong']);records.append(r)
            verification['val_epochs_recomputed']+=int(recompute is not None)
        chosen=min(good or epochs,key=lambda e:e['selection_key']);selected[arm]=dict(epoch=chosen['epoch'],status='eligible_selected' if good else 'descriptive_only',eligible_epochs=[e['epoch'] for e in good],**{k:chosen[k] for k in ('metrics','components','diagnostics','eligibility','checkpoint_sha256')})
    selections={arm:[r for r in records if r['arm']==arm and r['epoch']==selected[arm]['epoch']] for arm in all_models}
    for split in ('train','val'):
        maps={}
        for arm in all_models:
            path=(control_source/'size_only' if arm=='size_only' else source)/('epoch_%02d_%s.rows.jsonl.gz'%(selected[arm]['epoch'],split))
            maps[arm]={(r['scale'],r['image']):r for r in load(path)}
        a,b=maps.values()
        if a.keys()!=b.keys():raise ValueError('Paired identities')
        for scale in sorted({r[0] for r in a}):
            cells=[]
            for key,r in a.items():
                if key[0]!=scale:continue
                s=b[key]
                if any(r[k]!=s[k] for k in ('gt','midpoint','b','domain','sequence','image')):raise ValueError('Paired metadata')
                if r['midpoint'] is None:continue
                for j,edge in enumerate(('long','short')):
                    g=sorted(r['gt'][2:4],reverse=True)[j];m=sorted(r['midpoint'][2:4],reverse=True)[j]
                    ee=[classify(m,sorted(z['edge_residual'][2:4],reverse=True)[j],g) for z in (r,s)]
                    cells.append(dict(domain=r['domain'],sequence=r['sequence'],edge=edge,events=ee))
            for group in ['overall','real','sim']+sorted({r['sequence'] for r in cells}):
                for edge in ('long','short','both'):
                    rows=[r for r in cells if group in ('overall',r['domain'],r['sequence']) and edge in ('both',r['edge'])]
                    for stratum in ('all','near','nonnear'):
                        rr=[r for r in rows if stratum=='all' or r['events'][0]['near']==(stratum=='near')]
                        prefix=dict(split=split,scale=scale,group=group,edge=edge,stratum=stratum)
                        for j,arm in enumerate(all_models):
                            events=[r['events'][j] for r in rr]
                            for kind in ('all',)+KINDS:behavior.append(dict(prefix,arm=arm,kind=kind,denominator_edges=len(events),**behavior_summary(events if kind=='all' else [e for e in events if e['kind']==kind])))
                        good=[r for r in rr if r['events'][0]['effect']=='improved']
                        paired.append(dict(prefix,control_beneficial=len(good),candidate_retained=sum(r['events'][1]['effect']=='improved' for r in good),candidate_better_than_control=sum(r['events'][1]['abs_error_delta']<r['events'][0]['abs_error_delta']-1e-12 for r in good)))
    a,b=selected.values();checks={}
    for edge in ('long','short'):
        checks['sim/'+edge]=b['metrics']['sim']['edge_residual']['sizes']['edges'][edge]['mae']<a['metrics']['sim']['edge_residual']['sizes']['edges'][edge]['mae']-1e-6
        checks['real/'+edge]=b['metrics']['real']['edge_residual']['sizes']['edges'][edge]['mae']<=a['metrics']['real']['edge_residual']['sizes']['edges'][edge]['mae']+.001
    checks['overall/riou']=b['metrics']['overall']['edge_residual']['mean_riou_full_frame']>=a['metrics']['overall']['edge_residual']['mean_riou_full_frame']-1e-5
    result=dict(protocol='port_size_local_aux_analysis_v1',selected=selected,verification=verification,mechanism=dict(checks=checks,pass_all=all(checks.values())),finished=done,
        boundaries='24new candidate epochs vs24verified reused control epochs; no new TEST/depth, no rule/epoch retuning. Mechanism vs control separate from M adoption.',code_sha256=sha(Path(__file__).resolve()))
    out.mkdir();csv_write(out/'all_epochs_val.csv',records);csv_write(out/'selected_val.csv',sum(selections.values(),[]));csv_write(out/'correction_behavior.csv',behavior);csv_write(out/'beneficial_retention.csv',paired)
    write(out/'analysis.json',result);write(out/'artifacts.json',dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print({a:(v['epoch'],v['status'],v['eligibility']['failed']) for a,v in selected.items()});print(checks)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--control',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.run.resolve(),a.control.resolve(),a.out.resolve())
