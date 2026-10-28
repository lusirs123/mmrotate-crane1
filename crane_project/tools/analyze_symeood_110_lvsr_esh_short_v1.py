#!/usr/bin/env python3
"""Analyze only sealed 32-step TRAIN predictions/receipts, never runs a model."""
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.utils.port_geometry_size_distribution_metrics_v1 import size_summary
from crane_project.utils.port_size_a20_protection_review_v1 import relative_errors
from crane_project.utils.port_geometry_midpoint_edge_residual_metrics_v1 import percentile

OUT=ROOT/'work_dirs/symeood_110_retrain_seed0_20261026_v1/downstream/lvsr/edge_diagnostic/sample_harm_review/run32'


def read(p): return json.loads(p.read_text())


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(2**20),b''):h.update(c)
    return h.hexdigest()


def load(p):
    with gzip.open(p,'rt') as f:return [json.loads(x) for x in f]


def groups(rows):
    result={'overall':rows}
    for d in ('real','sim'):
        result[d]=[r for r in rows if r['domain']==d]
        for scale in (1.,.5):result[d+'/'+str(scale)]=[r for r in rows if r['domain']==d and r['scale']==scale]
    return result


def metric(rows,method,initial):
    s=size_summary(rows,method)
    result=dict(frames=len(rows),outputs=len(rows),joint10=s['joint10'],fallback=sum(not r['accepted'] for r in rows),
        vector_invalid=sum(not r['vector_valid'] for r in rows),candidate_invalid=sum(not r['candidate_valid'] for r in rows),
        a_abs=s['a']['abs_mae'],r_abs=s['r']['abs_mae'])
    for j,e in enumerate(('long','short')):
        v=s['edges'][e]
        result.update({e+'_'+k:v[k] for k in ('mae','p95','bias')})
        es=[relative_errors(r[method],r['gt'])['absolute'][j] for r in rows]
        before=[relative_errors(initial[r['image'],r['scale']][method],r['gt'])['absolute'][j] for r in rows]
        ms=[relative_errors(r['midpoint'],r['gt'])['absolute'][j] for r in rows]
        for field,label in [('near_M21','near'),('other_M21','other'),('large10_M21','large10')]:
            ids=[i for i,r in enumerate(rows) if r[field][j]];n=len(ids)
            result[e+'_'+label+'_count']=n
            result[e+'_'+label+'_mae']=sum(es[i] for i in ids)/n if n else None
            result[e+'_'+label+'_p95']=percentile([es[i] for i in ids])
            result[e+'_'+label+'_worsened_vs_start']=sum(es[i]>before[i]+1e-7 for i in ids)
            if label=='near':
                result[e+'_near_positive_harm']=sum(max(es[i]-ms[i],0.) for i in ids)/n if n else None
                result[e+'_near_material_harmed']=sum(es[i]-ms[i]>.001 for i in ids)
                result[e+'_near_crossed3']=sum(es[i]>.03 for i in ids)
                result[e+'_near_crossed10']=sum(es[i]>.1 for i in ids)
    return result


def csv_write(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)


def dist(xs):
    return dict(min=min(xs),median=statistics.median(xs),mean=statistics.mean(xs),max=max(xs)) if xs else None


def run():
    done=read(OUT/'completion.json')
    assert done['status']=='complete' and done['actual_updates']==64 and done['new_adoptable_checkpoints']==0
    for name,h in done['files'].items():assert sha(OUT/name)==h,name
    summaries=[];damages=[];metrics={};initials={};steps={}
    for arm in ('C','ESH'):
        steps[arm]=[json.loads(x) for x in (OUT/arm/'steps.jsonl').read_text().splitlines()]
        assert len(steps[arm])==32 and [r['updates'] for r in steps[arm]]==list(range(1,33))
        for role,count in [('update',256),('probe',64)]:
            start=load(OUT/arm/('step00_'+role+'.jsonl.gz'))
            initials[arm,role]=start;initial={(r['image'],r['scale']):r for r in start}
            for node in (0,8,32):
                rows=load(OUT/arm/('step%02d_'%node+role+'.jsonl.gz'))
                assert len(rows)==count and len({(r['image'],r['scale']) for r in rows})==count
                assert [(r['image'],r['scale']) for r in rows]==[(r['image'],r['scale']) for r in start]
                for r in rows:
                    assert r['delivered'][:2]+r['delivered'][4:]==r['midpoint'][:2]+r['midpoint'][4:]
                    assert r['raw'][:2]+r['raw'][4:]==r['midpoint'][:2]+r['midpoint'][4:]
                for group,rr in groups(rows).items():
                    for method in ('raw','delivered'):
                        m=metric(rr,method,initial);metrics[arm,role,node,group,method]=m
                        summaries.append(dict(arm=arm,role=role,step=node,group=group,method=method,**m))
                for r in rows:
                    for method in ('raw','delivered'):
                        errors=relative_errors(r[method],r['gt']);m=relative_errors(r['midpoint'],r['gt']);
                        before=relative_errors(initial[r['image'],r['scale']][method],r['gt'])
                        for j,e in enumerate(('long','short')):
                            damages.append(dict(arm=arm,role=role,step=node,image=r['image'],scale=r['scale'],domain=r['domain'],
                                method=method,edge=e,near=r['near_M21'][j],other=r['other_M21'][j],large10=r['large10_M21'][j],
                                error=errors['absolute'][j],signed_error=errors['signed'][j],M_error=m['absolute'][j],
                                positive_harm=max(errors['absolute'][j]-m['absolute'][j],0.),
                                start_error=before['absolute'][j],change_vs_start=errors['absolute'][j]-before['absolute'][j],
                                fallback=not r['accepted']))
    assert initials['C','update']==initials['ESH','update'] and initials['C','probe']==initials['ESH','probe']
    assert not {r['image'] for r in initials['C','update']} & {r['image'] for r in initials['C','probe']}
    def m(arm,group='overall'):return metrics[arm,'probe',32,group,'delivered']
    checks={}
    checks['probe_long_near_harm_reduced']=m('ESH')['long_near_positive_harm']<m('C')['long_near_positive_harm']-1e-6
    checks['probe_short_near_harm_kept']=m('ESH')['short_near_positive_harm']<=m('C')['short_near_positive_harm']+1e-6
    for dom in ('real','sim'):
        for e in ('long','short'):
            checks[dom+'/'+e+'/near_harm']=m('ESH',dom)[e+'_near_positive_harm']<=m('C',dom)[e+'_near_positive_harm']+.001
            checks[dom+'/'+e+'/other_error']=m('ESH',dom)[e+'_other_mae']<=m('C',dom)[e+'_other_mae']+.001
    for e in ('long','short'):
        checks['overall/'+e+'/MAE']=m('ESH')[e+'_mae']<=m('C')[e+'_mae']+1e-6
        for dom in ('real','sim'):
            for scale in ('1.0','0.5'):
                group=dom+'/'+scale
                checks[group+'/'+e+'/MAE']=m('ESH',group)[e+'_mae']<=m('C',group)[e+'_mae']+.001
    checks['one_edge_MAE_improved']=any(m('ESH')[e+'_mae']<m('C')[e+'_mae']-1e-6 for e in ('long','short'))
    stats={};step_table=[]
    for arm,rr in steps.items():
        stats[arm]=dict(elapsed_step_seconds=sum(r['elapsed_seconds'] for r in rr),
            gradient_norm=dist([r['gradient_norm'] for r in rr]),
            peak_allocator_bytes=max(r['peak_allocated_bytes'] for r in rr),
            actual_harm_vs_own_unweighted_better=sum(r['actual_update_proxy']['full']['harm_sum']<r['original_update_proxy']['full']['harm_sum']-1e-12 for r in rr) if arm=='ESH' else None,
            actual_harm_vs_own_unweighted_worse=sum(r['actual_update_proxy']['full']['harm_sum']>r['original_update_proxy']['full']['harm_sum']+1e-12 for r in rr) if arm=='ESH' else None,
            control_identity_note='Control actual delta vs analytic delta differs only by fp32 parameter-update rounding; sign counts are not treatment effects.' if arm=='C' else None,
            predicted_harm=dist([r['predicted_update_proxy']['full']['harm_sum'] for r in rr]),
            actual_harm=dist([r['actual_update_proxy']['full']['harm_sum'] for r in rr]),
            actual_current_batch_harm=dist([r['finite_current_batch_harm_sum'] for r in rr]),
            actual_predicted_residual_relative=dist([r['actual_predicted_relative_residual'] for r in rr]),
            actual_predicted_cosine=dist([r['actual_predicted_cosine'] for r in rr]),
            second_moment_change=dist([r['second_moment_relative_change'] for r in rr]),
            actual_radius=dist([r['actual_vs_original_radius'] for r in rr]),
            actual_radius_above_0p2=sum(r['actual_vs_original_radius']>.2+1e-6 for r in rr),
            weights=dist([v for r in rr for v in r['weights']]),
            weights_mean_by_step=dist([statistics.mean(r['weights']) for r in rr]),
            solver_guards=dict(Counter(str(r['solver'].get('guard',0)) for r in rr)),
            identity_weight_fallback=sum(r['solver'].get('fallback_identity',False) for r in rr))
        for r in rr:
            cell=dict(arm=arm,step=r['step'],weights_mean=statistics.mean(r['weights']),weights_max=max(r['weights']),
                gradient_norm=r['gradient_norm'],actual_update_harm=r['actual_update_proxy']['full']['harm_sum'],
                own_unweighted_update_harm=r['original_update_proxy']['full']['harm_sum'],predicted_update_harm=r['predicted_update_proxy']['full']['harm_sum'],
                actual_batch_finite_harm=r['finite_current_batch_harm_sum'],actual_radius=r['actual_vs_original_radius'],proxy_radius=r['proxy_vs_original_radius'],
                actual_predicted_relative_residual=r['actual_predicted_relative_residual'],second_moment_relative_change=r['second_moment_relative_change'],
                **{k:v for k,v in r['solver'].items() if isinstance(v,(int,float,bool))})
            step_table.append(cell)
    weight_groups=[]
    for group in ['overall','near','other']:
        for e,j in [('long',0),('short',1)]:
            values=[v for r in steps['ESH'] for i,row in enumerate(r['batch_before']) for v in [r['weights'][2*i+j]]
                    if group=='overall' or row['near_M21' if group=='near' else 'other_M21'][j]]
            weight_groups.append(dict(group=group,edge=e,**dist(values)))
    report=dict(status='COMPLETE_FROZEN32',actual_model_updates=64,contract_sha256=done['contract_sha256'],
        short_gate_pass=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],
        main_probe32={arm:m(arm) for arm in ('C','ESH')},steps=stats,weights_by_membership=weight_groups,
        node_comparison={role:{str(step):{arm:metrics[arm,role,step,'overall','delivered'] for arm in ('C','ESH')}
            for step in (0,8,32)} for role in ('update','probe')},
        identical_initial_predictions=True,update_probe_image_overlap=0,
        interpretation='Only fixed32 exploratory paired updates; no candidate selection, VAL, TEST, depth or generalization proof',
        limits=['Two arms confound differential weighting and total strength/Adam state','Probe large10 long0/short1; scales correlated; previously exposed TRAIN'],
        automatic_full_training=False,additional_updates=0,TEST=False,depth=False,
        input_files_verified=len(done['files']))
    csv_write(OUT/'TRAIN_node_summary.csv',summaries)
    csv_write(OUT/'TRAIN_edge_errors.csv',damages)
    csv_write(OUT/'step_mechanism.csv',step_table)
    with (OUT/'analysis.json').open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(short_gate_pass=report['short_gate_pass'],failed=report['failed'],probe32=report['main_probe32'],steps=stats),ensure_ascii=False,indent=2))


if __name__=='__main__':run()
