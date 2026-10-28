#!/usr/bin/env python3
"""Saved TRAIN/VAL only: edge damage, paired tradeoffs and historical P audit."""
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.tools.run_port_size_native_train_v1 import sha,read,write
from crane_project.utils.port_size_a20_protection_review_v1 import relative_errors,fixed_references,capped_retention
from crane_project.utils.port_geometry_size_distribution_metrics_v1 import size_summary
from crane_project.utils.port_geometry_midpoint_edge_residual_metrics_v1 import frozen_identity

COMMON=ROOT/'work_dirs/symeood_110_retrain_seed0_20261026_v1/downstream/lvsr'
HISTORY=ROOT/'work_dirs/port_size_a20_protection_train_v1/train_20261020_v1_gpu2'
OUT=COMMON/'edge_diagnostic'
TOL=1e-12


def checked_rows(folder,name,receipts):
    expected=read(folder/'artifacts.json')['files'][name]
    if sha(folder/name)!=expected:raise ValueError('Saved artifact SHA differs '+name)
    receipts[str((folder/name).relative_to(ROOT))]=expected
    if name.endswith('.gz'):
        with gzip.open(folder/name,'rt') as f:return [json.loads(x) for x in f]
    return read(folder/name)


def edge_diagnostic(rows,method):
    present=[r for r in rows if r['midpoint'] is not None]
    if any((r['midpoint'] is None)!=(r[method] is None) for r in rows):raise ValueError('Presence differs')
    entries=[]
    for r in present:
        m,c=[relative_errors(r[k],r['gt']) for k in ('midpoint',method)]
        d=[c['absolute'][j]-m['absolute'][j] for j in (0,1)]
        entries.append((r,m,c,d))
    result=dict(views=len(rows),unique_images=len({r['image'] for r in rows}),outputs=len(entries),missing=len(rows)-len(entries),
        short_improved=sum(d[1]<-TOL for r,m,c,d in entries),long_worsened=sum(d[0]>TOL for r,m,c,d in entries),
        short_improved_long_worsened=sum(d[1]<-TOL and d[0]>TOL for r,m,c,d in entries),
        both_improved=sum(max(d)<-TOL for r,m,c,d in entries),both_worsened=sum(min(d)>TOL for r,m,c,d in entries))
    result['short_gain_with_long_harm_fraction']=result['short_improved_long_worsened']/result['short_improved'] if result['short_improved'] else None
    for j,e in enumerate(('long','short')):
        near=[x for x in entries if x[1]['absolute'][j]<=.03]
        n=len(near)
        def avg(k):return sum(x[k]['absolute'][j] for x in near)/n if n else None
        result.update({e+'_near_count':n,e+'_near_before_mae_pct':100*avg(1) if n else None,e+'_near_after_mae_pct':100*avg(2) if n else None,
            e+'_near_harmed':sum(d[j]>TOL for r,m,c,d in near),e+'_near_material_harmed_0p1pp':sum(d[j]>.001 for r,m,c,d in near),
            e+'_near_crossed_3pct':sum(c['absolute'][j]>.03 for r,m,c,d in near),
            e+'_near_crossed_10pct':sum(c['absolute'][j]>.1 for r,m,c,d in near),
            e+'_near_positive_harm_mean_pct':100*sum(max(d[j],0.) for r,m,c,d in near)/n if n else None})
    result['near_long_short_gain_long_harm']=sum(m['absolute'][0]<=.03 and d[1]<-TOL and d[0]>TOL for r,m,c,d in entries)
    return result,entries


def groups(rows):
    result={'overall':rows}
    result.update({d:[r for r in rows if r['domain']==d] for d in ('real','sim')})
    if rows[0]['split']!='val':
        result.update({'scale_'+str(sc):[r for r in rows if r['scale']==sc] for sc in (.5,1.)})
        result.update({d+'/'+str(sc):[r for r in rows if r['domain']==d and r['scale']==sc] for d in ('real','sim') for sc in (.5,1.)})
    else:result.update({s:[r for r in rows if r['sequence']==s] for s in sorted({r['sequence'] for r in rows})})
    return result


def protected(rows,method,refs):
    result={}
    pp=[r for r in rows if r['midpoint'] is not None]
    for j,e in enumerate(('long','short')):
        cells=[(refs[r['image'],r['scale']],relative_errors(r[method],r['gt'])['absolute'][j]) for r in pp]
        active=[(ref,x) for ref,x in cells if ref['active'][j]]
        useful=[(ref,x) for ref,x in cells if ref['gain'][j]]
        excess=sum(max(x-ref['reference'][j],0.) for ref,x in active)
        retention=capped_retention([ref['m_errors'][j] for ref,x in useful],[ref['a_errors'][j] for ref,x in useful],[x for ref,x in useful])
        result.update({e+'_protected_count':len(active),e+'_protected_excess_sum':excess,
            e+'_protected_excess_per_all_edges':excess/(2*len(pp)) if pp else None,
            e+'_useful_retention':retention['fraction']})
    return result


def reference_map(rows):
    table={}
    for r in rows:
        if r['midpoint'] is None:continue
        m,a=[relative_errors(r[k],r['gt'])['absolute'] for k in ('midpoint','edge_residual')]
        table[r['image'],r['scale']]=dict(fixed_references(m,a),m_errors=m,a_errors=a)
    return table


def csv_write(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def main():
    if OUT.exists():raise ValueError('Existing diagnosis preserved')
    receipts={};table=[];damage=[];new={};old={};refs={};evidence={}
    for split,expected in (('train',5116),('val',887)):
        parent=checked_rows(COMMON,'fixed_b/epoch_07_'+split+'.rows.jsonl.gz',receipts)
        reference={(r['image'],r['scale']):r for r in parent}
        refs[split]=reference_map(parent)
        candidates=[('110',parent,'b'),('M21',parent,'midpoint'),('ep7',parent,'edge_residual')]
        for label,epoch in (('C3',3),('C6',6)):
            data=checked_rows(COMMON/'continuation_ep7','C/epoch_%02d_%s.rows.jsonl.gz'%(epoch,split),receipts)
            candidates.append((label,data,'edge_residual'))
        for label,data,method in candidates:
            if len(data)!=expected or len({(r['image'],r['scale']) for r in data})!=expected:raise ValueError('Incomplete saved split')
            for r in data:
                ref=reference[r['image'],r['scale']]
                if any(r[k]!=ref[k] for k in ('gt','b','midpoint','domain','split','sequence','frame_id')):raise ValueError('New saved identities differ')
                if not frozen_identity(r):raise ValueError('Frozen midpoint identity differs')
            for group,rr in groups(data).items():
                d,entries=edge_diagnostic(rr,method);sizes=size_summary(rr,method)
                cell=dict(chain='new',split=split,stage=label,group=group,**d,
                    long_mae_pct=100*sizes['edges']['long']['mae'],short_mae_pct=100*sizes['edges']['short']['mae'],
                    joint10=sizes['joint10'],a_abs=sizes['a']['abs_mae'],r_abs=sizes['r']['abs_mae'])
                if label in ('ep7','C3','C6'):cell.update(protected(rr,method,refs[split]))
                table.append(cell);new[split,label,group]=cell
            if label not in ('ep7','C3','C6'):continue
            _,entries=edge_diagnostic(data,method)
            for r,m,c,d in entries:
                if d[0]<=TOL and d[1]<=TOL:continue
                before_parent=relative_errors(reference[r['image'],r['scale']]['edge_residual'],r['gt'])
                damage.append(dict(split=split,stage=label,image=r['image'],sequence=r['sequence'],domain=r['domain'],scale=r['scale'],
                    M_long_abs_pct=100*m['absolute'][0],long_abs_pct=100*c['absolute'][0],long_change_pp=100*d[0],
                    M_short_abs_pct=100*m['absolute'][1],short_abs_pct=100*c['absolute'][1],short_change_pp=100*d[1],
                    long_near_M=m['absolute'][0]<=.03,long_harmed=d[0]>TOL,long_material_harmed_0p1pp=d[0]>.001,
                    long_crossed_3pct=m['absolute'][0]<=.03<c['absolute'][0],long_crossed_10pct=m['absolute'][0]<=.03 and c['absolute'][0]>.1,
                    short_improved_long_worsened=d[1]<-TOL and d[0]>TOL,
                    long_change_vs_ep7_pp=100*(c['absolute'][0]-before_parent['absolute'][0]),short_change_vs_ep7_pp=100*(c['absolute'][1]-before_parent['absolute'][1]),
                    M_long_signed_pct=100*m['signed'][0],long_signed_pct=100*c['signed'][0],M_short_signed_pct=100*m['signed'][1],short_signed_pct=100*c['signed'][1]))
    old_training=checked_rows(HISTORY,'training.json',receipts)
    for arm in ('C','R','P'):
        info=old_training['arms'][arm];evidence[arm]=dict(selected=info['selected'],representative=info['representative'],failed_representative=info['epochs'][info['representative']-1]['eligibility']['failed'])
        for ep in info['epochs']:
            split='val';epoch=ep['epoch'];data=checked_rows(HISTORY,arm+'/epoch_%02d_val.rows.jsonl.gz'%epoch,receipts)
            # TRAIN only at P5 representative and common fixed6; no unnecessary full historical replay.
            for split in (('val','train') if epoch in (5,6) else ('val',)):
                if split=='train':data=checked_rows(HISTORY,arm+'/epoch_%02d_train.rows.jsonl.gz'%epoch,receipts)
                reference=checked_rows(HISTORY,'a20_'+split+'.rows.jsonl.gz',receipts)
                fixed=reference_map(reference)
                if len(data)!=(887 if split=='val' else 5116):raise ValueError('Historical count differs')
                for group,rr in groups(data).items():
                    if split=='val' and group.startswith(('real_seq','sim_seq')):continue
                    d,entries=edge_diagnostic(rr,'edge_residual');s=size_summary(rr,'edge_residual')
                    cell=dict(chain='old',split=split,stage=arm+str(epoch),group=group,**d,
                        long_mae_pct=100*s['edges']['long']['mae'],short_mae_pct=100*s['edges']['short']['mae'],joint10=s['joint10'],a_abs=s['a']['abs_mae'],r_abs=s['r']['abs_mae'],**protected(rr,'edge_residual',fixed))
                    if split=='val':
                        v=ep['metrics'][group]['edge_residual'];cell.update(mean_RIoU=v['mean_riou_full_frame'],DFR=v['temporal'].get(group,{}).get('dfr_percent_per_frame'),eligible=ep['eligibility']['pass_all'])
                    table.append(cell);old[split,arm,epoch,group]=cell
    same_epoch={}
    for group in ('real','sim'):
        cells={}
        for edge in ('long','short'):
            for field in ('near_after_mae_pct','near_harmed','protected_excess_sum','useful_retention'):
                k=edge+'_'+field
                diffs=[old['val','P',n,group][k]-old['val','R',n,group][k] for n in range(1,7)]
                cells[k]=dict(P_minus_R_by_epoch=diffs,lower_count=sum(x<-TOL for x in diffs),higher_count=sum(x>TOL for x in diffs))
        same_epoch[group]=cells
    summary=dict(status='SAVED_TRAIN_VAL_DIAGNOSIS_COMPLETE',no_model_or_GPU=True,training_updates=0,TEST_or_depth_inputs_read=False,
        source_receipts=receipts,old_representatives=evidence,old_P_vs_R_same_epoch=same_epoch,
        new_key_groups={split:{label:{group:new[split,label,group] for group in ('overall','real','sim')} for label in ('110','M21','ep7','C3','C6')} for split in ('train','val')},
        counting='TRAIN combined=5116 views of2558 images at1.0/0.5, outputs5112; scale/domain cells are overlapping descriptive groups, not independent repetitions. near masks fixed by M error<=.03; numerical harm>1e-12, material harm>.001 (0.1pp).',
        group_rows=len(table),damage_rows=len(damage),damage_list='Every new ep7/C3/C6 view with either edge worse than its own M; tiny float changes and material0.1pp flags reported separately.')
    OUT.mkdir();csv_write(OUT/'grouped_TRAIN_VAL.csv',table);csv_write(OUT/'per_frame_damage.csv',damage);write(OUT/'analysis.json',summary)
    print(json.dumps(dict(group_rows=len(table),damage_rows=len(damage),source_files=len(receipts),old_representatives=evidence,old_P_vs_R_same_epoch=same_epoch),indent=2))


if __name__=='__main__':main()
