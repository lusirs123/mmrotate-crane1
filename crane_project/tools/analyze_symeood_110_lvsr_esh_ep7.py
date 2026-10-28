#!/usr/bin/env python3
"""Completed ESH6 VAL analysis, saved native metrics only; no model/TEST."""
from collections import Counter
import csv
import gzip
import json
import math
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.tools.run_port_size_native_train_v1 import sha,read,ranking,near_joint
from crane_project.tools.analyze_port_size_qalign_v1 import flat,numerical_equal
from crane_project.tools.analyze_port_size_proposal_downstream_v1 import aggregate_temporal
from crane_project.utils import port_size_a20_protection_train_v1 as original
from crane_project.utils.port_size_a20_protection_review_v1 import relative_errors
from crane_project.utils.port_geometry_size_distribution_metrics_v1 import size_summary
from crane_project.tools.diagnose_symeood_110_lvsr_edges import groups,edge_diagnostic

COMMON=ROOT/'work_dirs/symeood_110_retrain_seed0_20261026_v1/downstream/lvsr'
OUT=COMMON/'continuation_ESH_ep7'
CONTROL=COMMON/'continuation_ep7'


def rows(path):
    with gzip.open(path,'rt') as f:return [json.loads(x) for x in f]


def csv_write(path,data):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in data for k in r)))
        w.writeheader();w.writerows(data)


def subgroups(rr,method):
    out={}
    pp=[r for r in rr if r['midpoint'] is not None]
    for j,e in enumerate(('long','short')):
        vals=[(relative_errors(r['midpoint'],r['gt'])['absolute'][j],relative_errors(r[method],r['gt'])['absolute'][j]) for r in pp]
        for name,condition in [('near',lambda x:x<=.03),('other',lambda x:x>.03),('large10',lambda x:x>.1)]:
            vv=[(m,a) for m,a in vals if condition(m)]
            out[e+'_'+name+'_count']=len(vv)
            out[e+'_'+name+'_mae']=sum(a for m,a in vv)/len(vv) if vv else None
            out[e+'_'+name+'_gain_vs_M']=sum(m-a for m,a in vv)/len(vv) if vv else None
            out[e+'_'+name+'_improved_vs_M']=sum(a<m-1e-6 for m,a in vv)
            out[e+'_'+name+'_worsened_vs_M']=sum(a>m+1e-6 for m,a in vv)
    return out


def hits(v):return v['frames']-sum(x['length'] for x in v['failure_intervals']['riou'])


def main():
    done=read(OUT/'completion.json');proof=read(ROOT/'crane_project/data_contracts/symeood_110_lvsr_esh_ep7.json')
    if done['state']!='complete' or done['formal_updates']!=3834:raise ValueError('Requires all six complete VAL epochs')
    receipts={}
    def checked(p,h=None):
        actual=sha(p)
        if h is not None and actual!=h:raise ValueError('Artifact changed: '+str(p))
        receipts[str(p.relative_to(ROOT))]=actual
    for n,h in done['files'].items():
        if n=='status.json':
            # Writer captures epoch_complete, then publishes final complete.
            # Status is intentionally mutable; scientific artifacts are not.
            if read(OUT/n)['state']!='complete':raise ValueError('Final status not complete')
            checked(OUT/n)
        else:checked(OUT/n,h)
    for n,h in proof['control_files'].items():checked(ROOT/n,h)
    selection=read(OUT/'ESH/selection.json');events=selection['epochs']
    parent=read(COMMON/'fixed_b/epoch_07_val.json')
    parent_rows=rows(COMMON/'fixed_b/epoch_07_val.rows.jsonl.gz')
    parent_index={r['image']:r for r in parent_rows}
    fixed_refs=original.reference_table(parent_rows)
    if len(events)!=6 or done['versions']['fixed_final']!=6:raise ValueError('Complete six-epoch selection required')
    full=[];paired=[];subset=[];old={};new={};ep_rows={};epoch_stats=[]
    for ep in events:
        n=ep['epoch'];path=OUT/('ESH/epoch_%02d_val.rows.jsonl.gz'%n);rr=rows(path)
        c=read(CONTROL/('C/epoch_%02d_val.json'%n));cc=rows(CONTROL/('C/epoch_%02d_val.rows.jsonl.gz'%n))
        old[n]=c;new[n]=ep;ep_rows[n]=rr
        if len(rr)!=887 or len(cc)!=887 or len({r['image'] for r in rr})!=887:raise ValueError('VAL count differs')
        if ep['schedule_sha256']!=c['schedule_sha256'] or ep['updates']!=639*n:raise ValueError('Original schedule/budget differs')
        for a,b in zip(rr,cc):
            if any(a[k]!=b[k] for k in ('image','scale','gt','b','midpoint','domain','sequence','frame_id','split')):
                raise ValueError('Paired input identity differs')
            base=parent_index[a['image']]
            if any(a[k]!=base[k] for k in ('gt','b','midpoint')):raise ValueError('Fixed ep7/M21 identity differs')
        actual_near=near_joint(rr)
        if not numerical_equal(actual_near,ep['diagnostics']):raise ValueError('Near report differs')
        inherited=original.eligibility(ep['metrics'],actual_near,parent['metrics'])
        if inherited!=ep['eligibility'] or ranking(ep['metrics'],n)!=ep['selection_key']:raise ValueError('Original eligibility/ranking differs')
        checkpoint=OUT/('ESH/head_epoch_%02d.pth'%n)
        # Server checkpoint remains SHA-bound; local transfer need not include weights.
        if checkpoint.exists():checked(checkpoint,ep['checkpoint_sha256'])
        for group,rgroup in groups(rr).items():
            cg=groups(cc)[group]
            nv=ep['metrics'][group]['edge_residual'];cv=c['metrics'][group]['edge_residual']
            if not numerical_equal(size_summary(rgroup,'edge_residual'),nv['sizes']):raise ValueError('Size arithmetic differs')
            recomputed=sum(r['metrics']['edge_residual']['riou'] for r in rgroup)/len(rgroup)
            if abs(recomputed-nv['mean_riou_full_frame'])>1e-12:raise ValueError('Native RIoU arithmetic differs')
            full.append(flat(group,'edge_residual',nv,ep['components'][group]['edge_residual'],epoch=n,qualified=inherited['pass_all']))
            nd=edge_diagnostic(rgroup,'edge_residual')[0];cd=edge_diagnostic(cg,'edge_residual')[0]
            ns=subgroups(rgroup,'edge_residual');cs=subgroups(cg,'edge_residual')
            cell=dict(epoch=n,group=group,frames=len(rgroup),outputs=nd['outputs'],ESH_qualified=inherited['pass_all'],
                ESH_joint=nv['sizes']['joint10'],C_joint=cv['sizes']['joint10'],joint_delta=nv['sizes']['joint10']-cv['sizes']['joint10'],
                ESH_RIoU=nv['mean_riou_full_frame'],C_RIoU=cv['mean_riou_full_frame'],
                ESH_riou_hits=hits(nv),C_riou_hits=hits(cv))
            for name,values in [('ESH',nd),('C',cd),('ESH',ns),('C',cs)]:
                cell.update({name+'_'+k:v for k,v in values.items()})
            for name,value in [('ESH',nv),('C',cv)]:
                for e in ('long','short'):
                    cell.update({name+'_'+e+'_'+k:v for k,v in value['sizes']['edges'][e].items() if k in ('mae','p95','bias')})
                cell[name+'_a']=value['sizes']['a']['abs_mae'];cell[name+'_r']=value['sizes']['r']['abs_mae']
                tm=aggregate_temporal(value['temporal'])
                cell[name+'_DFR']=tm['DFR_percent_per_frame'];cell[name+'_ACI']=tm['ACI']
                cell[name+'_MCML_max']=tm['MCML_max'];cell[name+'_MCML_mean']=tm['MCML_mean']
                cell[name+'_TDR_hits']=tm['TDR_hits'];cell[name+'_TDR_denominator']=tm['TDR_denominator']
            paired.append(cell)
            for name,vals in [('ESH',ns),('C',cs)]:subset.append(dict(epoch=n,arm=name,group=group,**vals))
        epoch_stats.append(dict(epoch=n,eligible=inherited['pass_all'],failed=inherited['failed'],
            ESH_joint=ep['metrics']['overall']['edge_residual']['sizes']['joint10'],C_joint=c['metrics']['overall']['edge_residual']['sizes']['joint10'],
            training_losses=ep['training_losses'],weights=ep['weight_distribution'],epoch_seconds=ep['epoch_seconds']))
    eligible=[e for e in events if e['eligibility']['pass_all']]
    selected=min(eligible,key=lambda e:e['selection_key'])['epoch'] if eligible else None
    representative=min(events,key=lambda e:e['selection_key'])['epoch']
    if selected!=done['versions']['selected'] or representative!=done['versions']['representative']:raise ValueError('Original selection differs')
    comparison=[]
    for label,ep,method in [('110',parent,'b'),('M21',parent,'midpoint'),('ep7',parent,'edge_residual'),
        ('C3',old[3],'edge_residual'),('C6',old[6],'edge_residual'),('ESH_rep',new[representative],'edge_residual'),('ESH6',new[6],'edge_residual')]:
        for group,cell in ep['metrics'].items():
            v=cell[method]
            comparison.append(flat(group,method,v,ep['components'][group][method],stage=label,
                epoch=20 if label=='110' else 21 if label=='M21' else ep['epoch'],source_VAL_epoch=ep['epoch'],
                status='descriptive_only' if label=='ESH_rep' and selected is None else 'fixed'))
    frame_changes=[]
    cc=rows(CONTROL/'C/epoch_06_val.rows.jsonl.gz')
    for r,c in zip(ep_rows[6],cc):
        if r['midpoint'] is None:continue
        m,a,b=[relative_errors(box,r['gt']) for box in (r['midpoint'],c['edge_residual'],r['edge_residual'])]
        frame_changes.append(dict(image=r['image'],domain=r['domain'],sequence=r['sequence'],
            M_long=m['absolute'][0],M_short=m['absolute'][1],C_long=a['absolute'][0],C_short=a['absolute'][1],
            ESH_long=b['absolute'][0],ESH_short=b['absolute'][1],
            C_joint=max(a['absolute'])<=.1,ESH_joint=max(b['absolute'])<=.1))
    progress=[json.loads(x) for x in (OUT/'ESH/progress.jsonl').read_text().splitlines()]
    if len(progress)!=3834 or [r['updates'] for r in progress]!=list(range(1,3835)):raise ValueError('Actual update count differs')
    if sum(r['batch_size'] for r in progress)!=30672 or max(r['grad_norm'] for r in progress)>10:raise ValueError('Exposure or clipping differs')
    if any(not math.isfinite(v) for r in progress for v in r.values() if isinstance(v,(int,float))):raise ValueError('Nonfinite progress')
    fixed={r['group']:r for r in paired if r['epoch']==6}
    report=dict(status='COMPLETE_ESH6_VAL_ONLY',formal_updates=3834,versions=done['versions'],
        short32_failure_preserved=True,independent_budget_authorization=True,
        original_gate_failed_ESH6=new[6]['eligibility']['failed'],fixed_ESH6_vs_C6=fixed,epochs=epoch_stats,
        joint_transitions=dict(recovered=sum(not r['C_joint'] and r['ESH_joint'] for r in frame_changes),
            newly_failed=sum(r['C_joint'] and not r['ESH_joint'] for r in frame_changes)),
        related_epochs={group:dict(joint_higher=sum(r['joint_delta']>0 for r in paired if r['group']==group),
            joint_equal=sum(r['joint_delta']==0 for r in paired if r['group']==group),
            joint_lower=sum(r['joint_delta']<0 for r in paired if r['group']==group)) for group in ('overall','real','sim')},
        training=dict(weight_mean=statistics.mean(r['weight_mean'] for r in progress),weight_min=min(r['weight_min'] for r in progress),
            weight_max=max(r['weight_max'] for r in progress),max_grad_norm=max(r['grad_norm'] for r in progress),
            guard_counts=dict(Counter(r['solver_guard'] for r in progress)),identity_fallbacks=sum(r['identity_fallback'] for r in progress)),
        source_receipts=receipts,VAL_frames_per_epoch=887,saved_rows=6*887,
        RIoU_source='Saved native metrics, no re-inference or polygon recalculation',TEST=False,depth=False,
        independent_seeds=1,automatic_extension=False)
    for b in done['backups'].values():
        if len({x['device'] for x in b['copies']})!=2 or any(x['sha256']!=b['sha256'] for x in b['copies']):raise ValueError('Independent backup receipt differs')
    csv_write(OUT/'all6_VAL.csv',full);csv_write(OUT/'ESH_vs_C_same_epoch.csv',paired)
    csv_write(OUT/'fixed_chain_VAL.csv',comparison);csv_write(OUT/'VAL_subsets.csv',subset)
    csv_write(OUT/'ESH6_vs_C6_frame_changes.csv',frame_changes)
    with (OUT/'analysis.json').open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(versions=done['versions'],fixed={g:{k:v for k,v in cell.items() if k in ('ESH_joint','C_joint','joint_delta','ESH_long_mae','C_long_mae','ESH_short_mae','C_short_mae','ESH_RIoU','C_RIoU','ESH_DFR','C_DFR')} for g,cell in fixed.items()},failed=report['original_gate_failed_ESH6']),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
