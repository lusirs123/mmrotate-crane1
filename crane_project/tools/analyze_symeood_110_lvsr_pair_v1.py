#!/usr/bin/env python3
"""Completed pair-specific LVSR24: saved TRAIN/VAL statistics, no inference."""
from collections import Counter
import csv
import gzip
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import train_symeood_110_lvsr_pair_v1 as frozen
from crane_project.tools.run_port_size_native_train_v1 import read, write, sha, near_joint
from crane_project.tools.analyze_port_size_qalign_v1 import flat, numerical_equal
from crane_project.tools.analyze_port_size_proposal_downstream_v1 import aggregate_temporal
from crane_project.tools.run_port_geometry_size_condition_v1 import components
from crane_project.tools.diagnose_symeood_110_lvsr_edges import groups, edge_diagnostic
from crane_project.utils.port_geometry_size_distribution_metrics_v1 import size_summary
from crane_project.utils.port_geometry_midpoint_edge_residual_metrics_v1 import frozen_identity
from crane_project.utils.port_size_a20_protection_review_v1 import relative_errors

OUT = frozen.OUT
COMMON = frozen.COMMON


def csv_write(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def correct(row, method):
    return row[method] is not None and max(relative_errors(row[method], row['gt'])['absolute']) <= .1


def frame_delta(new, old):
    rr = []
    for a,b in zip(new,old):
        if a['midpoint'] is None:
            continue
        value = {k:a[k] for k in ('image','domain','sequence','frame_id','scale')}
        value.update(pair_joint10=correct(a,'edge_residual'),original_joint10=correct(b,'edge_residual'))
        for stage, row, method in [('M',a,'midpoint'),('pair',a,'edge_residual'),('original',b,'edge_residual')]:
            error = relative_errors(row[method], row['gt'])
            for j,e in enumerate(('long','short')):
                value[stage+'_'+e+'_abs_pct'] = 100*error['absolute'][j]
        rr.append(value)
    return rr


def main():
    manifest = read(OUT/'formal_return_manifest.json')
    for name,digest in manifest['files'].items():
        if sha(OUT/name) != digest:
            raise ValueError('Transferred artifact differs: '+name)
    proof = read(frozen.CONTRACT)
    if sha(frozen.CONTRACT) != read(OUT/'formal_completion.json')['contract_sha256']:
        raise ValueError('Frozen formal contract differs')
    for name,digest in {**proof['sources'], **proof['control_files']}.items():
        if sha(ROOT/name) != digest:
            raise ValueError('Source/control differs: '+name)
    selected = read(OUT/'train/selection.json')
    events = selected['epochs']
    if len(events)!=24 or selected['formal_updates']!=15336:
        raise ValueError('Incomplete budget')
    allval, paired, subset, train_table, frame_tables = [], [], [], [], {}
    controls, predictions = {}, {}
    failures = Counter()
    preserved = None
    for ep in events:
        n = ep['epoch']
        rr = frozen.rows(OUT/'train'/('epoch_%02d_val.rows.jsonl.gz'%n))
        old = read(COMMON/'fixed_b'/('epoch_%02d_val.json'%n))
        cc = frozen.rows(COMMON/'fixed_b'/('epoch_%02d_val.rows.jsonl.gz'%n))
        if len(rr)!=887 or len({r['image'] for r in rr})!=887 or len(cc)!=887 or ep['updates']!=639*n:
            raise ValueError('Epoch/sample count differs')
        if ep['schedule_sha256']!=old['schedule_sha256'] or sha(OUT/'train'/('head_epoch_%02d.pth'%n))!=ep['checkpoint_sha256']:
            raise ValueError('Schedule/checkpoint identity differs')
        for a,b in zip(rr,cc):
            if any(a[k]!=b[k] for k in ('image','scale','gt','b','midpoint','domain','split','sequence','frame_id')):
                raise ValueError('Same-epoch input differs')
            if not frozen_identity(a):
                raise ValueError('Frozen M identity changed')
        if preserved is None:
            preserved = [{k:r[k] for k in ('image','gt','b','midpoint','domain','sequence','frame_id')} for r in rr]
        elif preserved != [{k:r[k] for k in ('image','gt','b','midpoint','domain','sequence','frame_id')} for r in rr]:
            raise ValueError('Across-epoch input identity differs')
        actual_near, damage, old_damage = near_joint(rr), frozen.measures(rr), frozen.measures(cc)
        gate = frozen.eligibility(ep['metrics'], actual_near, damage, old['metrics'], old_damage)
        if (not numerical_equal(actual_near,ep['diagnostics']) or not numerical_equal(damage,ep['damages']) or
                gate!=ep['eligibility'] or frozen.selection_key(ep['metrics'],n)!=ep['selection_key'] or components(rr)!=ep['components']):
            raise ValueError('Frozen arithmetic/selection reconstruction differs')
        failures.update(gate['failed'])
        controls[n], predictions[n] = old, rr
        nd, od = groups(rr), groups(cc)
        for group, cell in ep['metrics'].items():
            v = cell['edge_residual']
            if not numerical_equal(size_summary(nd[group],'edge_residual'),v['sizes']):
                raise ValueError('Saved size arithmetic differs')
            if abs(sum(r['metrics']['edge_residual']['riou'] for r in nd[group])/len(nd[group])-v['mean_riou_full_frame'])>1e-12:
                raise ValueError('Saved native RIoU aggregation differs')
            a = flat(group,'edge_residual',v,ep['components'][group]['edge_residual'],epoch=n,arm='pair',qualified=gate['pass_all'])
            b = flat(group,'edge_residual',old['metrics'][group]['edge_residual'],old['components'][group]['edge_residual'],epoch=n,arm='original')
            allval.append(a)
            delta = dict(epoch=n,group=group,pair_qualified=gate['pass_all'])
            names = ['size_correct','long_mae_pct','short_mae_pct','long_p95_pct','short_p95_pct','long_bias_pct','short_bias_pct','mean_RIoU','a_abs','r_abs','DFR_percent_per_frame']
            for key in names:
                delta.update({ 'pair_'+key:a[key], 'original_'+key:b[key], 'delta_'+key:a[key]-b[key] })
            for key in damage[group]:
                if key not in ('short_gain_with_long_harm_fraction',):
                    delta['pair_'+key], delta['original_'+key] = damage[group][key], old_damage[group][key]
            for edge in ('long','short'):
                subset.append(dict(epoch=n,group=group,edge=edge,
                    **{arm+'_'+suffix:values[group][edge+'_'+suffix] for arm,values in [('pair',damage),('original',old_damage)] for suffix in
                       ('near_count','near_positive_harm_mean_pct','near_material_harmed_0p1pp','near_crossed_3pct','near_crossed_10pct','other_count','other_mae','large10_count','large10_mae')}))
            paired.append(delta)
        td = read(OUT/'train'/('epoch_%02d_train.json'%n))
        old_td = read(COMMON/'fixed_b'/('epoch_%02d_train.json'%n))
        for group,cell in td['size_groups']['cells'].items():
            for arm,data in [('pair',cell),('original',old_td['size_groups']['cells'][group])]:
                v,m = data['all']['edge_residual'],data['all']['midpoint']
                train_table.append(dict(epoch=n,group=group,arm=arm,frames=v['frames'],outputs=v['outputs'],joint10=v['joint10'],
                    M_joint10=m['joint10'],long_mae_pct=100*v['edges']['long']['mae'],short_mae_pct=100*v['edges']['short']['mae'],
                    M_long_mae_pct=100*m['edges']['long']['mae'],M_short_mae_pct=100*m['edges']['short']['mae'],
                    a_abs=v['a']['abs_mae'],r_abs=v['r']['abs_mae']))
        if n in (selected['representative'],24):
            frame_tables[n] = frame_delta(rr,cc)
    if frozen.choose(events) != {k:selected[k] for k in ('selected','representative','eligible_epochs','fixed_primary')}:
        raise ValueError('Final selection differs')
    progress = [json.loads(x) for x in (OUT/'train/progress.jsonl').read_text().splitlines()]
    if len(progress)!=15336:
        raise ValueError('Update log count differs')
    for i,row in enumerate(progress,1):
        if row['updates']!=i or row['epoch']!=(i-1)//639+1 or row['step']!=(i-1)%639+1 or row['batch_size']!=8:
            raise ValueError('Update schedule/counter differs')
        if any(not math.isfinite(v) for v in row.values() if isinstance(v,(float,int))):
            raise ValueError('Nonfinite training log')
    chain=[]
    parent=read(COMMON/'fixed_b/epoch_07_val.json')
    versions=[('110',parent,'b'),('M21',parent,'midpoint'),('original_ep7',parent,'edge_residual'),
              ('original_ep24',controls[24],'edge_residual'),('pair_descriptive',events[selected['representative']-1],'edge_residual'),
              ('pair_ep24',events[-1],'edge_residual')]
    for name,ep,method in versions:
        for group,cell in ep['metrics'].items():
            chain.append(flat(group,method,cell[method],ep['components'][group][method],version=name,epoch=ep.get('epoch',0)))
    trajectory={}
    for group in ('overall','real','sim'):
        rr=[r for r in paired if r['group']==group]
        trajectory[group]=dict(joint_higher=sum(r['delta_size_correct']>0 for r in rr),joint_equal=sum(r['delta_size_correct']==0 for r in rr),
            joint_lower=sum(r['delta_size_correct']<0 for r in rr),long_MAE_better=sum(r['delta_long_mae_pct']<-1e-4 for r in rr),
            short_MAE_better=sum(r['delta_short_mae_pct']<-1e-4 for r in rr),
            both_MAE_better=sum(max(r['delta_long_mae_pct'],r['delta_short_mae_pct'])<-1e-4 for r in rr),
            accurate_long_positive_harm_lower=sum(r['pair_long_near_positive_harm_mean_pct']<r['original_long_near_positive_harm_mean_pct']-1e-4 for r in rr))
    result=dict(state='complete_and_analyzed',selected=selected['selected'],representative=selected['representative'],fixed_primary=24,
        formal_updates=15336,VAL_rows=24*887,checkpoint_files_verified=24,backup_live_verification=manifest['weight_and_backup_SHA_live_checked'],
        source_control_unchanged=manifest['original_source_and_control_files_unchanged'],
        max_grad_norm=max(r['grad_norm'] for r in progress),clip_events=sum(r['grad_norm']>10 for r in progress),
        exposure_views=sum(r['batch_size'] for r in progress),
        gpu_process_memory_peak_MiB=max(e['resource']['process_polled_mib'] for e in events),
        allocator_peak_MiB=max(e['resource']['allocator_peak_mib'] for e in events),
        first_training_loss=progress[0]['loss'],last_training_loss=progress[-1]['loss'],
        epoch_average_losses=[e['training_losses']['loss'] for e in events],
        trajectory=trajectory,qualification_failure_epoch_counts=dict(failures),
        primary=[r for r in paired if r['epoch']==24],descriptive=[r for r in paired if r['epoch']==selected['representative']],
        pair_role_counts=[dict(epoch=e['epoch'],**e['pair_role_counts']) for e in events],
        raw_native_RIoU_reused=True,polygon_intersections_recomputed=False,
        TRAIN_raw_prediction_recomputed=False,TEST=False,depth=False,new_training=False,
        analysis_source_sha256=sha(Path(__file__)))
    csv_write(OUT/'all24_VAL.csv',allval)
    csv_write(OUT/'pair_vs_original_same_epoch.csv',paired)
    csv_write(OUT/'fixed_chain_VAL.csv',chain)
    csv_write(OUT/'near_and_other_VAL.csv',subset)
    csv_write(OUT/'saved_TRAIN_groups.csv',train_table)
    for n,rr in frame_tables.items():
        csv_write(OUT/('epoch_%02d_frame_changes.csv'%n),rr)
    write(OUT/'analysis.json',result)
    print(json.dumps(dict(selected=result['selected'],representative=result['representative'],fixed_primary=24,
        trajectory=trajectory,clip_events=result['clip_events']),ensure_ascii=False))


if __name__=='__main__':
    main()
