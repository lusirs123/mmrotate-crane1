#!/usr/bin/env python3
"""Verify returned axis-balance evidence; no model inference or epoch re-selection."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.tools import run_port_geometry_size_axis_balance_v1 as run
from crane_project.tools import analyze_port_geometry_size_sim_support_v1 as support


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def analyze(base,uniform):
    raw=base/'axis_20261017_v2_run'
    if (base/'local_review.json').exists():raise ValueError('Refuse to overwrite review')
    index=read(raw/'artifacts.json')['files'];omitted=[]
    for name,digest in index.items():
        p=raw/name
        if p.exists():
            if sha(p)!=digest:raise ValueError('Returned artifact SHA differs: '+name)
        elif name.endswith('.pth'):omitted.append(dict(path=name,sha256=digest,retained_on_server=True))
        else:raise ValueError('Missing nonweight artifact: '+name)
    done=read(raw/'completion.json');proof=read(raw/'input_runtime_proof.json')
    if not done['status'].startswith('COMPLETE_') or done['formal_head_updates']!=15336 or done['control_training_updates']!=0:raise ValueError('Incomplete run')
    if proof['source_identity']!=run.checked_sources():raise ValueError('Fitted source identity drift')
    reference=read(raw/'fixed_I07_summary.json');near_ref=read(raw/'fixed_I07_near.json')
    run.condition.val_replay.compare_numeric(reference,read(uniform/'candidate/val_epoch_07.summary.json'))
    frames=[];values={};choices={}
    for arm in ('control','candidate'):
        records=[]
        for epoch in range(1,25):
            cell=raw/arm
            v=read(cell/('val_epoch_%02d.summary.json'%epoch));n=read(cell/('val_epoch_%02d.near.json'%epoch))
            comp=read(cell/('val_epoch_%02d.components.json'%epoch));meta=read(cell/('epoch_%02d.json'%epoch))
            if meta['selection_key']!=run.selection_key(v,epoch) or meta['adoption']!=run.adoption(v,n,reference,near_ref):raise ValueError('Selection or adoption drift')
            records.append(meta);values[(arm,epoch)]=v
            for group,cellvalue in v.items():
                x=cellvalue['edge_residual'];s=x['sizes'];c=comp[group]['edge_residual']
                if c['size_correct']!=s['joint10'] or c['outputs']!=s['outputs']:raise ValueError('Component denominator mismatch')
                row=dict(arm=arm,epoch=epoch,group=group,mean_RIoU=x['mean_riou_full_frame'],**c,
                    long_MAE_percent=s['edges']['long']['mae']*100,short_MAE_percent=s['edges']['short']['mae']*100,
                    long_P95_percent=s['edges']['long']['p95']*100,short_P95_percent=s['edges']['short']['p95']*100,
                    long_bias_percent=s['edges']['long']['bias']*100,short_bias_percent=s['edges']['short']['bias']*100,
                    a_meanabs=s['a']['abs_mae'],r_meanabs=s['r']['abs_mae'],
                    wrong_to_right=cellvalue['paired']['wrong_to_right'],right_to_wrong=cellvalue['paired']['right_to_wrong'],
                    AP50=x['cache_gt_ap11']['0.5'],AP75=x['cache_gt_ap11']['0.75'])
                for label,num,den in [('center_rate','center_correct','outputs'),('size_rate','size_correct','outputs'),
                    ('angle_rate','angle_correct','angle_eligible_outputs'),('output_coverage','outputs','frames'),
                    ('center_full','center_correct','frames'),('size_full','size_correct','frames'),('angle_full','angle_correct','angle_eligible_frames')]:
                    row[label]=c[num]/c[den] if c[den] else None
                for domain,t in x['temporal'].items():
                    for k in ('dfr_percent_per_frame','dfr_adjacent_pairs','aci','aci_adjacent_pairs','mcml_max','mcml_mean','mcml_segments','mrf_mean','mrf_events'):
                        row[domain+'_'+k]=t[k]
                    row[domain+'_tdr_hits']=t['tdr_w10']['numerator'];row[domain+'_tdr_denominator']=t['tdr_w10']['denominator']
                frames.append(row)
            if arm=='candidate':
                losses=read(cell/('train_epoch_%02d.loss_groups.json'%epoch))
                if sum(c['count'] for c in losses['groups'].values())!=10224:raise ValueError('Training loss-group count mismatch')
                if abs(losses['weighted_sum']/10224-meta['mean_weighted_loss'])>1e-7:raise ValueError('Weighted loss accounting mismatch')
        choices[arm]=run.choose(records)
        if choices[arm]!=done[arm]:raise ValueError('Stored all24 choice mismatch')
    selected={arm:(choices[arm]['selected'] or choices[arm]['representative']) for arm in choices}
    ce,re=selected['candidate']['epoch'],selected['control']['epoch']
    candidate=run.old.read_rows(raw/'candidate'/('val_epoch_%02d.rows.jsonl.gz'%ce))
    controls=run.old.read_rows(uniform/'candidate'/('val_epoch_%02d.rows.jsonl.gz'%re))
    i07=run.old.read_rows(uniform/'candidate/val_epoch_07.rows.jsonl.gz')
    byid={r['image']:r for r in i07};controlbyid={r['image']:r for r in controls}
    for r in candidate:
        for other in (byid[r['image']],controlbyid[r['image']]):
            for k in ('gt','b','midpoint','domain','sequence','scale'):
                if r[k]!=other[k]:raise ValueError('Paired frame baseline identity drift')
        r['i07']=byid[r['image']]['edge_residual'];r['uniform_selected']=controlbyid[r['image']]['edge_residual']
    paired={};groups={'overall':candidate}
    groups.update({d:[r for r in candidate if r['domain']==d] for d in ('real','sim')})
    groups.update({s:[r for r in candidate if r['sequence']==s] for s in sorted({r['sequence'] for r in candidate})})
    for group,rs in groups.items():
        paired[group]={m:run.metrics.paired(rs,before=m) for m in ('b','midpoint','i07','uniform_selected')}
    fixedgroups={domain:{g:{m:run.metrics.size_summary([r for r in candidate if r['domain']==domain and run.metrics.group(r['gt'],r['midpoint'])==g],m)
                           for m in ('midpoint','i07','edge_residual')}
                         for g in support.GROUPS} for domain in ('real','sim')}
    selected_near=read(raw/'candidate'/('val_epoch_%02d.near.json'%ce))
    training=read(raw/'candidate'/('train_epoch_%02d.summary.json'%ce))
    loss_trajectories=[]
    for epoch in range(1,25):
        l=read(raw/'candidate'/('train_epoch_%02d.loss_groups.json'%epoch))
        for k,c in l['groups'].items():
            loss_trajectories.append(dict(epoch=epoch,cell_edge_group=k,**c,unweighted_mean=c['unweighted_sum']/c['count'],weighted_mean=c['weighted_sum']/c['count']))
    result=dict(status=done['status'],source_commit=proof['git_head'],selected=selected,
        passing_epochs={a:choices[a]['passing_epochs'] for a in choices},selected_values={a:values[(a,m['epoch'])] for a,m in selected.items()},
        fixed_I07=reference,fixed_I07_near=near_ref,selected_near=selected_near,
        paired=paired,fixed_M_groups=fixedgroups,selected_train=training,train_weights=read(raw/'train_weights.json'),
        effects_vs_control=done['effects_vs_control'],candidate_failed_checks=selected['candidate']['adoption']['failed'],
        adoption_check_pass_counts=dict(Counter(k for e in choices['candidate']['epochs'] for k,v in e['adoption']['checks'].items() if v)),
        omitted_heads=omitted,frames_report=frames,completion=done,
        frozen_cached_replay=read(raw/'frozen_cached_val_replay.json'),
        limits='Single seed; Sim TRAIN/VAL one video each. Actual sequential group loss logged for candidate only; uniform control historical per-group online losses unavailable. No native-image inference/TEST/depth/promotion.',
        source=dict(artifact_index_sha256=sha(raw/'artifacts.json'),script_sha256=sha(Path(__file__))))
    (base/'local_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    support.write_csv(base/'all48_val_components_custom.csv',frames)
    support.write_csv(base/'candidate24_training_loss_groups.csv',loss_trajectories)
    (base/'local_review_artifacts.json').write_text(json.dumps(dict(files={p.name:sha(p) for p in
        [base/'local_review.json',base/'all48_val_components_custom.csv',base/'candidate24_training_loss_groups.csv']}),indent=2)+'\n')
    print(json.dumps(dict(status=done['status'],candidate_epoch=ce,control_epoch=re,passing=result['passing_epochs'],
                         failed=result['candidate_failed_checks'],effects=result['effects_vs_control']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,default=ROOT/'work_dirs/port_geometry_size_axis_balance_v1')
    p.add_argument('--uniform-run',type=Path,default=ROOT/'work_dirs/port_geometry_size_condition_v1/condition_20261017_v1_run')
    args=p.parse_args();analyze(args.base,args.uniform_run)
