#!/usr/bin/env python3
"""Verify all48 S045 VAL and report same-epoch V-D deltas; no new inference."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.tools.run_port_size_structure_v1 import read,write,sha,ranking,eligibility,near_report,structure_value
from crane_project.tools.analyze_port_size_qalign_v1 import load_rows,numerical_equal,flat,csv_write
from crane_project.utils.port_size_structure_v1 import ARMS,cross_edge


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
    if a['initial_stem_digest']!=b['initial_stem_digest'] or [e['schedule_sha256'] for e in a['epochs']]!=[e['schedule_sha256'] for e in b['epochs']]:raise ValueError('Paired init/order')
    for arm in ARMS:
        for ep in training['arms'][arm]['epochs']:
            rows=load_rows(source/arm/('epoch_%02d_val.rows.jsonl.gz'%ep['epoch']))
            if len(rows)!=887 or {r['image'] for r in rows}!=set(index):raise ValueError('Epoch identities')
            for r in rows:
                for k in ('gt','b','midpoint','scale','split','domain','sequence','frame_id'):
                    if r[k]!=index[r['image']][k]:raise ValueError('Epoch metadata drift '+k)
            actual=metrics.summary(rows,cache)
            if components(rows)!=ep['components'] or not numerical_equal(near_report(rows),ep['diagnostics']):raise ValueError('Geometry/near counts drift')
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



def analyze(source,out):
    if out.exists() or out.parent!=source.parent:raise ValueError('Fresh sibling analysis required')
    idx=read(source/'artifacts.json')['files']
    if any(sha(source/n)!=h for n,h in idx.items()):raise ValueError('Artifact drift')
    t=read(source/'training.json');finished=read(source/'finished.json')
    if t['status']!=finished['status'] or t['status']!='COMPLETE' or t['formal_updates']!=30672 or t['protocol']!='port_size_structure_v1':raise ValueError('Incomplete frozen fit')
    if any(t[k]!=0 for k in ('test_reads','depth_reads','feature_extractions')) or t['automatic_adoption']:raise ValueError('Scope changed')
    proof=verify_saved_val(source,t);allrows=[];selected=[];reps={};diagnostics=[];paired=[]
    base=read(source/'baseline_val.summary.json')
    for g,methods in base['metrics'].items():
        for method in ('b','midpoint'):selected.append(flat(g,method,methods[method],base['components'][g][method],arm='baseline',epoch=0))
    for arm in ARMS:
        info=t['arms'][arm];epochs=info['epochs']
        if [e['epoch'] for e in epochs]!=list(range(1,25)):raise ValueError('Missing full VAL')
        for e in epochs:
            if eligibility(e['metrics'],e['diagnostics'])!=e['eligibility'] or ranking(e['metrics'],e['epoch'])!=e['selection_key']:raise ValueError('Gate/order drift')
            rr=load_rows(source/arm/('epoch_%02d_val.rows.jsonl.gz'%e['epoch']))
            if not numerical_equal(cross_edge(rr),e['cross_edge']):raise ValueError('Cross harm arithmetic')
            for g,mm in e['metrics'].items():
                value=flat(g,'edge_residual',mm['edge_residual'],e['components'][g]['edge_residual'],arm=arm,epoch=e['epoch'],eligible=e['eligibility']['pass_all'])
                value.update(mm['paired']);allrows.append(value)
                diagnostics.append(dict(arm=arm,epoch=e['epoch'],group=g,near=e['diagnostics'][g],cross_edge=e['cross_edge'].get(g)))
        ok=[e for e in epochs if e['eligibility']['pass_all']];chosen=min(ok or epochs,key=lambda e:e['selection_key'])
        if info['representative']!=chosen['epoch'] or info['selected']!=(chosen['epoch'] if ok else None):raise ValueError('Frozen selection drift')
        reps[arm]=chosen;selected.extend(dict(row,status='eligible_selected' if ok else 'descriptive_only') for row in allrows if row['arm']==arm and row['epoch']==chosen['epoch'])
        for split in ('train','val'):
            rr=load_rows(source/arm/('epoch_%02d_%s.rows.jsonl.gz'%(chosen['epoch'],split)))
            if len(rr)!=(5116 if split=='train' else 887):raise ValueError('Representative complete identity')
            if split=='train' and not numerical_equal(near_report(rr),read(source/arm/('epoch_%02d_train.json'%chosen['epoch']))['diagnostics']):raise ValueError('TRAIN near arithmetic')
    mechanism=structure_value(*[reps[a] for a in ARMS])
    if mechanism!=t['structure_value']:raise ValueError('Structure interpretation drift')
    for epoch in range(1,25):
        for group in reps[ARMS[0]]['metrics']:
            a,b=[next(row for row in allrows if row['arm']==arm and row['epoch']==epoch and row['group']==group) for arm in ARMS]
            fields=['long_mae_pct','short_mae_pct','a_abs','r_abs','mean_RIoU','AP75','DFR_percent_per_frame','size_correct']
            row=dict(epoch=epoch,group=group)
            for k in fields:row[k+'_V_minus_D']=a[k]-b[k]
            for edge in ('long','short'):
                for arm in ARMS:
                    cell=t['arms'][arm]['epochs'][epoch-1]['diagnostics'][group]['near'][edge]
                    for k in ('after_mae','harmed','positive_harm_sum'):row[arm+'/'+edge+'/'+k]=cell[k]
            paired.append(row)
    out.mkdir();csv_write(out/'all_epochs_val.csv',allrows);csv_write(out/'selected_val.csv',selected);csv_write(out/'same_epoch_V_minus_D.csv',paired)
    write(out/'near_cross_all_epochs.json',diagnostics)
    write(out/'analysis.json',dict(protocol=t['protocol'],verification=proof,structure_value=mechanism,
        representatives={a:dict(epoch=reps[a]['epoch'],eligible=reps[a]['eligibility']['pass_all'],failed=reps[a]['eligibility']['failed'],checkpoint_sha256=reps[a]['checkpoint_sha256']) for a in ARMS},
        boundaries='Fixed single seed; no significance from correlated frames, no re-selection/TEST/depth/adoption. Negative V-D size deltas favor V; 54parameter difference retained.',
        source_artifacts_sha256=sha(source/'artifacts.json'),source_sha256=sha(Path(__file__))))
    write(out/'artifacts.json',dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();analyze(a.run.resolve(),a.out.resolve())
