#!/usr/bin/env python3
"""S043 complete18 VAL arithmetic, selection and protection replay. CPU only."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.tools import run_port_size_a20_protection_v1 as run
from crane_project.tools.analyze_port_size_qalign_v1 import numerical_equal,flat,csv_write
from crane_project.utils import port_size_a20_protection_train_v1 as obj


def analyze(source,out):
    if out.exists() or out.parent!=source.parent:raise ValueError('Fresh sibling analysis required')
    idx=run.read(source/'artifacts.json')['files']
    if any(run.sha(source/n)!=h for n,h in idx.items()):raise ValueError('Artifact SHA drift')
    training=run.read(source/'training.json');finished=run.read(source/'finished.json')
    if training['status']!=finished['status'] or training['status']!='COMPLETE' or training['formal_updates']!=11502:raise ValueError('Incomplete18-epoch run')
    tables={s:{(r['image'],r['scale']):r for r in run.read(source/(s+'_fixed_references.json'))} for s in ('train','val')}
    a20=run.read(source/'a20_val.json');cache={};allrows=[];selected=[];protection=[];representatives={};maxdiff=0.
    from crane_project.utils import port_geometry_size_distribution_metrics_v1 as metrics
    from crane_project.tools.run_port_geometry_size_condition_v1 import components
    baseline=run.load_rows(source/'a20_val.rows.jsonl.gz');identity={obj.key(r):r for r in baseline}
    for group,values in a20['metrics'].items():
        for method,label in [('b','B'),('midpoint','M'),('edge_residual','A20')]:
            selected.append(flat(group,method,values[method],a20['components'][group][method],arm=label,epoch=0,status='fixed'))
    for arm in obj.ARMS:
        info=training['arms'][arm]
        if [e['epoch'] for e in info['epochs']]!=list(range(1,7)):raise ValueError('Not all6 VAL epochs')
        for ep in info['epochs']:
            rr=run.load_rows(source/arm/('epoch_%02d_val.rows.jsonl.gz'%ep['epoch']))
            if len(rr)!=887 or {obj.key(r) for r in rr}!=set(identity):raise ValueError('Full VAL identity')
            for r in rr:
                old=identity[obj.key(r)]
                if any(r[k]!=old[k] for k in ('gt','b','midpoint','domain','sequence','frame_id','split')):raise ValueError('Baseline frame drift')
            fresh=metrics.summary(rr,cache)
            if components(rr)!=ep['components'] or not numerical_equal(run.near_joint(rr),ep['diagnostics']):raise ValueError('Geometry/near drift')
            if not numerical_equal(obj.protection_report(rr,tables['val']),ep['protection']):raise ValueError('Protection/retention drift')
            for group,methods in fresh.items():
                reference=ep['metrics'][group]
                if methods['paired']!=reference['paired']:raise ValueError('Transition/identity drift')
                for method in metrics.METHODS:
                    a,b=methods[method],reference[method]
                    diff=abs(a['mean_riou_full_frame']-b['mean_riou_full_frame']);maxdiff=max(maxdiff,diff)
                    if diff>5e-6 or not numerical_equal(a['sizes'],b['sizes']) or a['cache_gt_ap11']!=b['cache_gt_ap11']:raise ValueError('VAL geometry/AP drift')
                    for domain,t in a['temporal'].items():
                        for k,v in t.items():
                            if k=='dfr_percent_per_frame':
                                if abs(v-b['temporal'][domain][k])>1e-4:raise ValueError('DFR drift')
                            elif not numerical_equal(v,b['temporal'][domain][k]):raise ValueError('Temporal drift')
            if obj.eligibility(ep['metrics'],ep['diagnostics'],a20['metrics'])!=ep['eligibility'] or run.ranking(ep['metrics'],ep['epoch'])!=ep['selection_key']:raise ValueError('Gate/ranking drift')
            for group,values in ep['metrics'].items():
                r=flat(group,'edge_residual',values['edge_residual'],ep['components'][group]['edge_residual'],arm=arm,epoch=ep['epoch'],eligible=ep['eligibility']['pass_all'])
                r.update({k:values['paired'][k] for k in ('wrong_to_right','right_to_wrong','new_output','lost_output')})
                allrows.append(r)
                protection.append(dict(arm=arm,epoch=ep['epoch'],group=group,statistics=ep['protection'][group]))
        eligible=[e for e in info['epochs'] if e['eligibility']['pass_all']]
        chosen=min(eligible or info['epochs'],key=lambda e:e['selection_key'])
        if info['representative']!=chosen['epoch'] or info['selected']!=(chosen['epoch'] if eligible else None):raise ValueError('Representative drift')
        representatives[arm]=chosen
        selected.extend(dict(r,status='eligible_selected' if eligible else 'descriptive_only') for r in allrows if r['arm']==arm and r['epoch']==chosen['epoch'])
        for split,n in [('train',5116),('val',887)]:
            rr=run.load_rows(source/arm/('epoch_%02d_%s.rows.jsonl.gz'%(chosen['epoch'],split)))
            if len(rr)!=n:raise ValueError('Selected full rows')
            if split=='train':
                saved=run.read(source/arm/('epoch_%02d_train.json'%chosen['epoch']))
                if not numerical_equal(obj.protection_report(rr,tables[split]),saved['protection']):raise ValueError('TRAIN protection drift')
    if obj.mechanism(representatives)!=training['mechanism']:raise ValueError('Mechanism judgment drift')
    schedules=[[e['schedule_sha256'] for e in training['arms'][a]['epochs']] for a in obj.ARMS]
    if schedules[0]!=schedules[1] or schedules[1]!=schedules[2]:raise ValueError('Paired schedule mismatch')
    out.mkdir();csv_write(out/'all_epochs_val.csv',allrows);csv_write(out/'selected_val.csv',selected)
    run.write(out/'protection_all_epochs.json',protection)
    run.write(out/'verification.json',dict(epochs_recomputed=18,full_val_frames=887,formal_updates=11502,
        max_cross_platform_mean_RIoU_difference=maxdiff,all_gates_selection_protection_verified=True,mechanism=training['mechanism'],
        selected={a:dict(epoch=e['epoch'],eligible=e['eligibility']['pass_all'],failed=e['eligibility']['failed'],checkpoint_sha256=e['checkpoint_sha256']) for a,e in representatives.items()},
        eligibility_failure_counts={a:dict(Counter(k for e in training['arms'][a]['epochs'] for k in e['eligibility']['failed'])) for a in obj.ARMS},
        boundary='Server formal values retained; local CPU replay verifies arithmetic, no re-selection, TEST, depth or automatic adoption.'))
    run.write(out/'artifacts.json',dict(files={p.name:run.sha(p) for p in out.iterdir() if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();analyze(a.run.resolve(),a.out.resolve())
