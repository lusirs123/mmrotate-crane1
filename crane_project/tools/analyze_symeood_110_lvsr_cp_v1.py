#!/usr/bin/env python3
"""Read-only CP/C paired TRAIN/VAL analysis; never loads a model or TEST."""
from collections import Counter
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.tools.run_port_size_native_train_v1 import read,write,sha,near_joint
from crane_project.tools.analyze_port_size_qalign_v1 import load_rows,flat,csv_write,numerical_equal
from crane_project.utils import symeood_110_lvsr_cp_v1 as cp
from crane_project.utils.port_geometry_size_distribution_metrics_v1 import size_summary
from crane_project.utils.port_size_a20_protection_review_v1 import relative_errors
from crane_project.utils.port_geometry_midpoint_edge_residual_metrics_v1 import frozen_identity

COMMON=ROOT/'work_dirs/symeood_110_retrain_seed0_20261026_v1/downstream/lvsr'
OUT=COMMON/'continuation_CP_ep7'
CONTROL=COMMON/'continuation_ep7'


def run():
    files=read(OUT/'artifacts.json')['files']
    for name,h in files.items():
        if sha(OUT/name)!=h:raise ValueError('CP artifact differs: '+name)
    receipts={}
    def existing(folder,name):
        expected=read(folder/'artifacts.json')['files'][name]
        if sha(folder/name)!=expected:raise ValueError('Control artifact differs: '+name)
        receipts[str((folder/name).relative_to(ROOT))]=expected
        return read(folder/name) if name.endswith('.json') else load_rows(folder/name)
    done=read(OUT/'completion.json');selection=read(OUT/'CP/selection.json');events=selection['epochs']
    if done['state']!='complete' or done['formal_updates']!=3834 or [e['epoch'] for e in events]!=list(range(1,7)):raise ValueError('Incomplete CP budget')
    versions=cp.select(events)
    if versions!=done['versions']:raise ValueError('Frozen CP selection differs')
    parent=existing(COMMON,'fixed_b/epoch_07_val.json')
    identity=read(OUT/'identity.json');cidentity=existing(CONTROL,'identity.json')
    if identity['runtime']!=dict(read(COMMON.parent/'midpoint/formal/roi_cache/cache_manifest.json')['runtime']):raise ValueError('Runtime identity differs')
    for k,v in cidentity['runtime'].items():
        if identity['runtime'][k]!=v:raise ValueError('C/CP runtime differs')
    if identity['initial_digest']!=existing(CONTROL,'C/selection.json')['initial_digest']:raise ValueError('C/CP parent differs')
    progress=[json.loads(x) for x in (OUT/'CP/progress.jsonl').read_text().splitlines()]
    if len(progress)!=3834:raise ValueError('CP update count differs')
    for i,r in enumerate(progress,1):
        if r['updates']!=i or not math.isclose(r['total'],r['size']+.1*r['protect'],rel_tol=1e-12,abs_tol=1e-12):raise ValueError('CP objective/counter differs')
        if any(not math.isfinite(v) for v in r.values() if isinstance(v,(int,float))):raise ValueError('Nonfinite training statistic')
    if sum(r['batch_size'] for r in progress)!=30672 or sum(r['edges'] for r in progress)!=61344:raise ValueError('CP exposure differs')
    valrows={};paired=[];allval=[];verification=0
    for ep in events:
        n=ep['epoch'];c=existing(CONTROL,'C/epoch_%02d_val.json'%n)
        if ep['updates']!=639*n or ep['schedule']['index_sha256']!=c['schedule_sha256']:raise ValueError('C/CP schedule differs')
        rr=load_rows(OUT/('CP/epoch_%02d_val.rows.jsonl.gz'%n));cc=existing(CONTROL,'C/epoch_%02d_val.rows.jsonl.gz'%n)
        valrows[n]=rr
        if len(rr)!=887 or len({r['image'] for r in rr})!=887:raise ValueError('CP VAL count differs')
        for a,b in zip(rr,cc):
            if any(a[k]!=b[k] for k in ('image','scale','gt','b','midpoint','domain','sequence','frame_id','split')) or not frozen_identity(a):raise ValueError('C/CP VAL frame identity differs')
        damage=cp.damage(rr);cdamage=cp.damage(cc)
        if not numerical_equal(damage,ep['damage']) or not numerical_equal(near_joint(rr),ep['diagnostics']):raise ValueError('CP damage/near arithmetic differs')
        if cp.eligibility(ep['metrics'],ep['diagnostics'],damage,c['metrics'],cdamage,parent['metrics'])!=ep['eligibility']:raise ValueError('Frozen CP gate differs')
        if cp.selection_key(ep['metrics'],damage,n)!=ep['selection_key']:raise ValueError('Frozen CP key differs')
        for split,data,control_data in [('val',rr,cc),('train',load_rows(OUT/('CP/epoch_%02d_train.rows.jsonl.gz'%n)),existing(CONTROL,'C/epoch_%02d_train.rows.jsonl.gz'%n))]:
            if len(data)!=(887 if split=='val' else 5116):raise ValueError('Saved split count differs')
            for a,b in zip(data,control_data):
                if any(a[k]!=b[k] for k in ('image','scale','gt','b','midpoint','domain','sequence','frame_id','split')) or not frozen_identity(a):raise ValueError('Saved split identity differs')
            dd=cp.damage(data);dc=cp.damage(control_data)
            if split=='train' and not numerical_equal(dd,read(OUT/('CP/epoch_%02d_train.json'%n))['damage']):raise ValueError('CP TRAIN damage differs')
            from crane_project.tools.diagnose_symeood_110_lvsr_edges import groups
            gg=groups(data);cg=groups(control_data)
            for g in gg:
                s=size_summary(gg[g],'edge_residual');cs=size_summary(cg[g],'edge_residual')
                cell=dict(split=split,epoch=n,group=g,frames=len(gg[g]),outputs=s['outputs'],missing=s['missing'],CP_joint=s['joint10'],C_joint=cs['joint10'],joint_delta=s['joint10']-cs['joint10'],CP_a=s['a']['abs_mae'],C_a=cs['a']['abs_mae'],CP_r=s['r']['abs_mae'],C_r=cs['r']['abs_mae'])
                for edge in ('long','short'):
                    for stat in ('mae','p95','bias'):
                        cell.update({f'CP_{edge}_{stat}_pct':100*s['edges'][edge][stat],f'C_{edge}_{stat}_pct':100*cs['edges'][edge][stat]})
                    for field in ('near_count','near_after_mae_pct','near_harmed','near_material_harmed_0p1pp','near_crossed_3pct','near_positive_harm_mean_pct'):
                        cell.update({f'CP_{edge}_{field}':dd[g][edge+'_'+field],f'C_{edge}_{field}':dc[g][edge+'_'+field]})
                for field in ('short_improved','short_improved_long_worsened','short_gain_with_long_harm_fraction'):
                    cell.update({'CP_'+field:dd[g][field],'C_'+field:dc[g][field]})
                paired.append(cell)
                if split=='val':
                    for method in ('b','midpoint','edge_residual'):
                        v=ep['metrics'][g][method]
                        if not numerical_equal(size_summary(gg[g],method),v['sizes']):raise ValueError('CP VAL size summary differs')
                        if abs(sum(r['metrics'][method]['riou'] for r in gg[g])/len(gg[g])-v['mean_riou_full_frame'])>1e-12:raise ValueError('CP native RIoU arithmetic differs')
                        verification+=1
                    row=flat(g,'edge_residual',ep['metrics'][g]['edge_residual'],ep['components'][g]['edge_residual'],epoch=n,qualified=ep['eligibility']['pass_all'])
                    row.update(riou_hits=cp.hits(ep['metrics'][g]['edge_residual']),failed=';'.join(ep['eligibility']['failed']))
                    allval.append(row)
    comparison=[]
    sources=[('110',parent,'b'),('M21',parent,'midpoint'),('ep7',parent,'edge_residual'),('C3',existing(CONTROL,'C/epoch_03_val.json'),'edge_residual'),('C6',existing(CONTROL,'C/epoch_06_val.json'),'edge_residual'),('CP_rep',events[versions['representative']-1],'edge_residual'),('CP6',events[5],'edge_residual')]
    for label,ep,method in sources:
        for group in ep['metrics']:
            v=ep['metrics'][group][method];row=flat(group,method,v,ep['components'][group][method],stage=label,
                epoch=20 if label=='110' else 21 if label=='M21' else ep['epoch'],source_VAL_epoch=ep['epoch'],
                status='descriptive_only' if label=='CP_rep' and versions['selected'] is None else 'fixed')
            row['riou_hits']=cp.hits(v);comparison.append(row)
    frame_changes=[]
    cc=existing(CONTROL,'C/epoch_06_val.rows.jsonl.gz')
    for r,c in zip(valrows[6],cc):
        if r['midpoint'] is None:continue
        m,ce,pe=[relative_errors(r[k] if k!='C' else c['edge_residual'],r['gt']) for k in ('midpoint','C','edge_residual')]
        frame_changes.append(dict(image=r['image'],domain=r['domain'],sequence=r['sequence'],M_long_error_pct=100*m['absolute'][0],C_long_error_pct=100*ce['absolute'][0],CP_long_error_pct=100*pe['absolute'][0],M_short_error_pct=100*m['absolute'][1],C_short_error_pct=100*ce['absolute'][1],CP_short_error_pct=100*pe['absolute'][1],near_long_M=m['absolute'][0]<=.03,C_joint=max(ce['absolute'])<=.1,CP_joint=max(pe['absolute'])<=.1))
    for b in done['backups'].values():
        if len({x['device'] for x in b['copies']})!=2 or any(x['sha256']!=b['sha256'] for x in b['copies']):raise ValueError('CP backup evidence differs')
    fixed={r['group']:r for r in paired if r['split']=='val' and r['epoch']==6}
    trend={}
    for g in ('overall','real','sim'):
        pp=[r for r in paired if r['split']=='val' and r['group']==g]
        trend[g]=dict(positive_long_harm_lower_epochs=sum(r['CP_long_near_positive_harm_mean_pct']<r['C_long_near_positive_harm_mean_pct'] for r in pp),material_long_harm_fewer_epochs=sum(r['CP_long_near_material_harmed_0p1pp']<r['C_long_near_material_harmed_0p1pp'] for r in pp),joint_higher_epochs=sum(r['joint_delta']>0 for r in pp),joint_equal_epochs=sum(r['joint_delta']==0 for r in pp),joint_lower_epochs=sum(r['joint_delta']<0 for r in pp))
    report=dict(status='COMPLETE',versions=versions,formal_updates=3834,CP6_mechanism_contract_pass=events[5]['eligibility']['pass_all'],
        CP6_failed=events[5]['eligibility']['failed'],representative_failed=events[versions['representative']-1]['eligibility']['failed'],full_chain_CP6=events[5]['eligibility']['full_chain_size_claim'],
        failure_counts=dict(Counter(k for e in events for k in e['eligibility']['failed'])),CP6_vs_C6=fixed,related_epoch_trends=trend,
        integrity=dict(CP_files_sha_checked=len(files),control_receipts=receipts,val_group_method_arithmetic_checks=verification,TRAIN_VAL_frame_identities_matched=True,init_runtime_schedules_matched=True,all_objective_counters_finite=True,view_updates=30672,edge_updates=61344,RIoU_source='Saved native intersection values; no polygon/model replay'),
        checkpoint_sha256={str(e['epoch']):e['checkpoint_sha256'] for e in events if e['epoch'] in (versions['representative'],6)},max_grad_norm=max(r['grad_norm'] for r in progress),TEST=False,depth=False,
        boundary='Single fixed-seed matched CP6/C6 and related same-epoch observations. New joint-first selection cannot be attributed solely to protect vs old C3 selection. No adoption, no extra training or automatic evaluation.')
    write(OUT/'analysis.json',report);csv_write(OUT/'VAL_comparison.csv',comparison);csv_write(OUT/'CP_vs_C_same_epoch.csv',paired);csv_write(OUT/'all_CP_epochs_VAL.csv',allval);csv_write(OUT/'CP6_vs_C6_frame_changes.csv',frame_changes)
    print(json.dumps({k:v for k,v in report.items() if k!='integrity'},ensure_ascii=False,indent=2))


if __name__=='__main__':run()
