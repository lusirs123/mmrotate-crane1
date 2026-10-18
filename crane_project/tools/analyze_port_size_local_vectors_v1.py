#!/usr/bin/env python3
"""Read complete SHA-bound S034 outputs; no inference, fitting or TEST access."""
import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.tools.run_port_size_native_train_v1 import read,write,sha,ranking,eligibility
from crane_project.tools.run_port_size_native_train_v1 import near_joint
from crane_project.tools.analyze_port_size_correction_behavior_v1 import classify,summary as behavior_summary,KINDS
from crane_project.tools.analyze_port_size_proposal_downstream_v1 import aggregate_temporal
ARMS=('size_only','size_plus_geometry')


def csv_write(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def load_rows(path):
    with gzip.open(path,'rt') as f:return [json.loads(x) for x in f]


def numerical_equal(a,b,tolerance=1e-12):
    if isinstance(a,dict):return set(a)==set(b) and all(numerical_equal(a[k],b[k],tolerance) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(numerical_equal(x,y,tolerance) for x,y in zip(a,b))
    if isinstance(a,float) and isinstance(b,(float,int)):return abs(a-b)<=tolerance
    return a==b


def verify_saved_val(source,training):
    """Recompute all48 VAL tables from rows, independent of training summaries.

    Preserve server OpenCV RIoU as formal values; allow5e-6 cross-platform
    arithmetic only. Counts, AP and temporal definitions may not change.
    """
    from crane_project.utils import port_geometry_size_distribution_metrics_v1 as metrics
    from crane_project.tools.run_port_geometry_size_condition_v1 import components
    baseline=load_rows(source/'baseline_val.rows.jsonl.gz')
    index={r['image']:r for r in baseline};cache={};max_riou=0.;max_dfr=0.;checked=0
    if len(baseline)!=887 or len(index)!=887:raise ValueError('VAL identities')
    a,b=[training['arms'][k] for k in ARMS]
    if a['initial_state_digest']!=b['initial_state_digest'] or [e['schedule_sha256'] for e in a['epochs']]!=[e['schedule_sha256'] for e in b['epochs']]:raise ValueError('Paired init/order')
    for arm in ARMS:
        for ep in training['arms'][arm]['epochs']:
            rows=load_rows(source/arm/('epoch_%02d_val.rows.jsonl.gz'%ep['epoch']))
            if len(rows)!=887 or {r['image'] for r in rows}!=set(index):raise ValueError('Epoch identities')
            for r in rows:
                for k in ('gt','b','midpoint','scale','split','domain','sequence','frame_id'):
                    if r[k]!=index[r['image']][k]:raise ValueError('Epoch metadata drift '+k)
            actual=metrics.summary(rows,cache)
            if components(rows)!=ep['components'] or not numerical_equal(near_joint(rows),ep['diagnostics']):raise ValueError('Geometry/near counts drift')
            for group,cell in actual.items():
                ref=ep['metrics'][group]
                if cell['paired']!=ref['paired']:raise ValueError('Paired identity/transition counts')
                for method in metrics.METHODS:
                    x,y=cell[method],ref[method]
                    difference=abs(x['mean_riou_full_frame']-y['mean_riou_full_frame']);max_riou=max(max_riou,difference)
                    if difference>5e-6:raise ValueError('RIoU drift exceeds tolerance')
                    if not numerical_equal(x['sizes'],y['sizes']) or x['cache_gt_ap11']!=y['cache_gt_ap11']:raise ValueError('Sizes/AP drift')
                    for domain,t in x['temporal'].items():
                        reference=y['temporal'][domain]
                        for k,v in t.items():
                            if k=='dfr_percent_per_frame':
                                difference=abs(v-reference[k]);max_dfr=max(max_dfr,difference)
                                if difference>1e-4:raise ValueError('DFR arithmetic drift')
                            elif not numerical_equal(v,reference[k]):raise ValueError('Temporal count/definition drift '+k)
            checked+=1
    return dict(epochs_recomputed=checked,val_frames_per_epoch=887,output_frames_per_epoch=886,
                counts_components_transitions_ap_exact=True,sizes_near_temporal_float_tolerance=1e-12,paired_initialization_and_schedules_exact=True,
                maximum_cross_platform_mean_riou_difference=max_riou,riou_tolerance=5e-6,
                maximum_cross_platform_dfr_difference=max_dfr,dfr_tolerance=1e-4,
                formal_values='Server saved summaries; local OpenCV recomputation verifies arithmetic only.')


def flat(group,method,v,comp,**prefix):
    s=v['sizes'];t=v['temporal'].get(group,v['temporal'].get('overall',{}))
    if not t and len(v['temporal'])==1:t=next(iter(v['temporal'].values()))
    r=dict(prefix,group=group,method=method,frames=comp['frames'],outputs=comp['outputs'],missing=comp['missing'],mean_RIoU=v['mean_riou_full_frame'],
           center_correct=comp['center_correct'],center_rate=comp['center_correct']/comp['outputs'],center_full=comp['center_correct']/comp['frames'],
           size_correct=comp['size_correct'],size_rate=comp['size_correct']/comp['outputs'],size_full=comp['size_correct']/comp['frames'],
           angle_correct=comp['angle_correct'],angle_eligible_outputs=comp['angle_eligible_outputs'],angle_eligible_frames=comp['angle_eligible_frames'],
           angle_rate=comp['angle_correct']/comp['angle_eligible_outputs'] if comp['angle_eligible_outputs'] else None,angle_full=comp['angle_correct']/comp['frames'],
           coverage=comp['outputs']/comp['frames'],a_abs=s['a']['abs_mae'],a_bias=s['a']['bias'],r_abs=s['r']['abs_mae'],r_bias=s['r']['bias'],AP50=v['cache_gt_ap11']['0.5'],AP75=v['cache_gt_ap11']['0.75'])
    for edge in ('long','short'):
        for k in ('mae','p95','bias'):r[edge+'_'+k+'_pct']=100*s['edges'][edge][k]
    for k in ('dfr_percent_per_frame','aci','aci_adjacent_pairs','mcml_max','mcml_mean'):
        r[k]=t.get(k)
    r.update(TDR_num=t.get('tdr_w10',{}).get('numerator'),TDR_den=t.get('tdr_w10',{}).get('denominator'))
    # Reuse the historical pair/segment/window-weighted aggregation, never
    # average Real/Sim percentages with equal domain weights.
    r.update(aggregate_temporal(v['temporal']))
    return r


def run(source,out):
    if out.exists() or out.parent!=source.parent:raise ValueError('Fresh sibling result-only analysis directory')
    index=read(source/'artifacts.json')['files']
    for n,s in index.items():
        if sha(source/n)!=s:raise ValueError('Artifact SHA '+n)
    finished=read(source/'finished.json');training=read(source/'training.json')
    if finished['status']!='COMPLETE' or training['status']!='COMPLETE' or training['formal_head_updates']!=30672:raise ValueError('Incomplete budget')
    if read(source/'protocol.json')['protocol']!='port_size_local_vectors_v1':raise ValueError('Protocol')
    verification=verify_saved_val(source,training)
    out.mkdir();all_epochs=[];selected={};selected_rows=[];records={}
    base=read(source/'baseline_val.summary.json')
    for group in base['metrics']:
        for method in ('b','midpoint'):
            selected_rows.append(flat(group,method,base['metrics'][group][method],base['components'][group][method],arm='baseline',epoch=0,status='fixed'))
    for arm in ARMS:
        info=training['arms'][arm];epochs=info['epochs']
        if [e['epoch'] for e in epochs]!=list(range(1,25)):raise ValueError('Missing epoch')
        eligible=[]
        for ep in epochs:
            if ranking(ep['metrics'],ep['epoch'])!=ep['selection_key'] or eligibility(ep['metrics'],ep['diagnostics'])!=ep['eligibility']:raise ValueError('Selection drift')
            if ep['eligibility']['pass_all']:eligible.append(ep)
            for group in ep['metrics']:
                row=flat(group,'edge_residual',ep['metrics'][group]['edge_residual'],ep['components'][group]['edge_residual'],arm=arm,epoch=ep['epoch'],eligible=ep['eligibility']['pass_all'])
                row.update(wrong_to_right=ep['metrics'][group]['paired']['wrong_to_right'],right_to_wrong=ep['metrics'][group]['paired']['right_to_wrong'],failed=';'.join(ep['eligibility']['failed']))
                all_epochs.append(row)
        chosen=min(eligible or epochs,key=lambda e:e['selection_key']);expected=info['selected'] if eligible else info['representative']
        if chosen['epoch']!=expected:raise ValueError('Selected identity')
        status='eligible_selected' if eligible else 'descriptive_only'
        selected[arm]=dict(epoch=chosen['epoch'],status=status,eligible_epochs=info['eligible_epochs'],checkpoint_sha256=chosen['checkpoint_sha256'],failed=chosen['eligibility']['failed'],
            metrics=chosen['metrics'],components=chosen['components'],diagnostics=chosen['diagnostics'],eligibility_fail_counts=dict(Counter(k for e in epochs for k in e['eligibility']['failed'])))
        for r in all_epochs:
            if r['arm']==arm and r['epoch']==chosen['epoch']:selected_rows.append(dict(r,status=status))
        records[arm]={}
        for split in ('train','val'):
            rr=load_rows(source/arm/('epoch_%02d_%s.rows.jsonl.gz'%(chosen['epoch'],split)))
            records[arm][split]={(r['scale'],r['image']):r for r in rr}
    behavior=[];retention=[]
    for split in ('train','val'):
        a,b=[records[arm][split] for arm in ARMS]
        if a.keys()!=b.keys():raise ValueError('Paired sample identity')
        for scale in sorted({k[0] for k in a}):
            pairs=[]
            for key,r in a.items():
                if key[0]!=scale:continue
                s=b[key]
                for k in ('image','sequence','domain','frame_id','scale','gt','b','midpoint'):
                    if r[k]!=s[k]:raise ValueError('Paired metadata '+k)
                if r['midpoint'] is None:
                    if s['edge_residual'] is not None:raise ValueError('New output')
                    continue
                for j,edge in enumerate(('long','short')):
                    gt=sorted(r['gt'][2:4],reverse=True)[j];m=sorted(r['midpoint'][2:4],reverse=True)[j]
                    events=[classify(m,sorted(x['edge_residual'][2:4],reverse=True)[j],gt) for x in (r,s)]
                    pairs.append(dict(domain=r['domain'],sequence=r['sequence'],edge=edge,events=events))
            for group in ['overall','real','sim']+sorted({r['sequence'] for r in pairs}):
                cells=[r for r in pairs if group in ('overall',r['domain'],r['sequence'])]
                for edge in ('long','short','both_edges'):
                    cc=[r for r in cells if edge=='both_edges' or r['edge']==edge]
                    for stratum in ('all','near','nonnear'):
                        ss=[r for r in cc if stratum=='all' or r['events'][0]['near']==(stratum=='near')]
                        prefix=dict(split=split,scale=scale,group=group,edge=edge,stratum=stratum)
                        for i,arm in enumerate(ARMS):
                            ee=[r['events'][i] for r in ss]
                            for kind in ('all',)+KINDS:
                                sub=ee if kind=='all' else [e for e in ee if e['kind']==kind]
                                behavior.append(dict(prefix,arm=arm,kind=kind,denominator_edges=len(ee),**behavior_summary(sub)))
                        good=[r for r in ss if r['events'][0]['effect']=='improved']
                        retention.append(dict(prefix,control_beneficial=len(good),candidate_also_beneficial=sum(r['events'][1]['effect']=='improved' for r in good),
                            candidate_better_than_control=sum(r['events'][1]['abs_error_delta']<r['events'][0]['abs_error_delta']-1e-12 for r in good),
                            both_harmed=sum(all(e['effect']=='harmed' for e in r['events']) for r in ss)))
    control,candidate=[selected[a] for a in ARMS];checks={}
    for edge in ('long','short'):
        checks['sim/'+edge]=candidate['metrics']['sim']['edge_residual']['sizes']['edges'][edge]['mae']<control['metrics']['sim']['edge_residual']['sizes']['edges'][edge]['mae']-1e-6
        checks['real/'+edge]=candidate['metrics']['real']['edge_residual']['sizes']['edges'][edge]['mae']<=control['metrics']['real']['edge_residual']['sizes']['edges'][edge]['mae']+.001
    checks['overall/riou']=candidate['metrics']['overall']['edge_residual']['mean_riou_full_frame']>=control['metrics']['overall']['edge_residual']['mean_riou_full_frame']-1e-5
    result=dict(protocol='port_size_local_vectors_analysis_v1',source_artifacts_sha256=sha(source/'artifacts.json'),finished=finished,selected=selected,verification=verification,
        extra_supervision_vs_common_control=dict(checks=checks,pass_all=all(checks.values())),
        boundaries='No TEST/depth, no changing selection/thresholds; selected geometry supervision value distinct from shared structure benefit. Direction classifications offline only.',
        code_sha256={str(Path(__file__).resolve().relative_to(ROOT)):sha(Path(__file__)),
            'crane_project/tools/analyze_port_size_proposal_downstream_v1.py':sha(ROOT/'crane_project/tools/analyze_port_size_proposal_downstream_v1.py'),
            'crane_project/tools/analyze_port_size_correction_behavior_v1.py':sha(ROOT/'crane_project/tools/analyze_port_size_correction_behavior_v1.py')})
    csv_write(out/'all_epochs_val.csv',all_epochs);csv_write(out/'selected_val.csv',selected_rows)
    csv_write(out/'correction_behavior.csv',behavior);csv_write(out/'beneficial_retention.csv',retention)
    write(out/'analysis.json',result);write(out/'artifacts.json',dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(selected={a:{k:v[k] for k in ('epoch','status','eligible_epochs','failed')} for a,v in selected.items()},extra_supervision=checks),ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();run(args.run.resolve(),args.out.resolve())
