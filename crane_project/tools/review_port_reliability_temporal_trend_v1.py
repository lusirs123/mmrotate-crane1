#!/usr/bin/env python3
"""Independent endpoint-formula replay and state/ranking checks of both veto arms."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import sys

ROOT=Path(__file__).resolve().parents[2]


def read(path):
    with gzip.open(path,'rt') as f:return [json.loads(line) for line in f]


def bad(r):
    if r['pred'] is None:return None
    p,g=sorted(r['pred'][2:4]),sorted(r['gt'][2:4])
    return any(abs(x-y)/y>.1 for x,y in zip(p,g))


def state(r, ids):
    wrong=bad(r)
    return 'MISSING' if wrong is None else ('FA' if wrong else 'CR') if r['image'] in ids else ('ED' if wrong else 'FR')


def check_summary(rows, ids, summary):
    counts={s:sum(state(r,ids)==s for r in rows) for s in ('FA','FR','ED','CR','MISSING')}
    if counts!=summary['states']:raise ValueError('State counts differ')
    if len(ids)!=summary['accepted_outputs']:raise ValueError('Acceptance count differs')
    kinds=dict(incorrect_acceptance={'FA'},correct_rejection={'FR'},error_detection={'ED'},
               correct_retention={'CR'},missing_output={'MISSING'},unavailable={'FR','ED','MISSING'})
    for kind,members in kinds.items():
        runs=[];length=0;prev=None
        for r in rows:
            boundary=(prev is None or r['sequence']!=prev['sequence'] or r['frame_id']!=prev['frame_id']+1
                      or r['temporal']['acquisition_boundary'])
            if boundary or state(r,ids) not in members:
                if length:runs.append(length)
                length=0
            if state(r,ids) in members:length+=1
            prev=r
        if length:runs.append(length)
        stored=summary['runs'][kind]
        if max(runs,default=0)!=stored['longest'] or len(runs)!=stored['segment_count'] or sum(runs)!=stored['total_frames']:
            raise ValueError('Independent run lengths differ')


def veto(rows, cutoff, level=False):
    key='level_temporal' if level else 'temporal'
    return {r['image'] for r in rows if r['original_simple_decision']['size_accepted']
            and (not r[key]['available'] or r[key]['primary']<=cutoff)}


def rank_ids(rows, method, count=None, correct=None):
    out=sorted((r for r in rows if r['pred'] is not None),key=lambda r:(r['risks'][method],r['image']))
    if count is not None:return {r['image'] for r in out[:count]}
    seen=0;threshold=None
    if correct:
        for r in out:
            seen+=int(not bad(r))
            if seen>=correct:threshold=r['risks'][method];break
    return {r['image'] for r in out if threshold is not None and r['risks'][method]<=threshold}


def matched_veto_ids(rows, target, kind, level=False):
    key='level_temporal' if level else 'temporal'
    fallback={r['image'] for r in rows if r['original_simple_decision']['size_accepted'] and not r[key]['available']}
    out=sorted((r for r in rows if r['original_simple_decision']['size_accepted'] and r[key]['available']),
               key=lambda r:(r[key]['primary'],r['image']))
    if kind=='count':
        n=target-len(fallback)
        return None if n<0 or n>len(out) else fallback|{r['image'] for r in out[:n]}
    n=target-sum(not bad(r) for r in rows if r['image'] in fallback)
    if n<0 or n>sum(not bad(r) for r in out):return None
    if n==0:return fallback
    seen=0;threshold=None
    for r in out:
        seen+=int(not bad(r))
        if seen>=n:threshold=r[key]['primary'];break
    return fallback|{r['image'] for r in out if r[key]['primary']<=threshold}


def FA_budget_ids(rows, target):
    fixed={r['image'] for r in rows if r['original_simple_decision']['size_accepted'] and not r['temporal']['available']}
    if sum(bool(bad(r)) for r in rows if r['image'] in fixed)>target:return None
    out=[r for r in rows if r['original_simple_decision']['size_accepted'] and r['temporal']['available']]
    values=sorted({r['temporal']['primary'] for r in out})
    result=fixed
    for t in values:
        ids=fixed|{r['image'] for r in out if r['temporal']['primary']<=t}
        if sum(bool(bad(r)) for r in rows if r['image'] in ids)>target:break
        result=ids
    return result


def part_for(rows, group):
    return rows if group=='all' else [r for r in rows if r[group.split(':')[0]]==group.split(':')[1]]


def main():
    p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--allow-absent-frontend',action='store_true');a=p.parse_args()
    report=json.loads((a.result/'report.json').read_text())
    protocol=json.loads((ROOT/'crane_project/tools/port_reliability_temporal_trend_v1_protocol.json').read_text())
    absent=[]
    for name,h in dict(report['input_pins'],**report['sources']['sources']).items():
        if a.allow_absent_frontend and not (ROOT/name).exists() and name in protocol['frozen_frontend_identity']:
            if h!=protocol['frozen_frontend_identity'][name]:raise ValueError('Frontend identity differs')
            absent.append(name);continue
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h:raise ValueError('Source changed '+name)
    raw={r['image']:r for r in read(ROOT/protocol['rows'])};checks=0;total=0;diff=0.;seen=set()
    for role in ('TRAIN','VAL'):
        rows=read(a.result/('scored_'+role+'.jsonl.gz'));past=[];prev=None
        for r in rows:
            if r['image'] in seen:raise ValueError('Duplicate role identity')
            seen.add(r['image']);original=raw[r['image']]
            for field in ('pred','gt','image_size','original_simple_decision','size_risks','domain','split','reliability_role','frame_id','sequence'):
                if r[field]!=original[field]:raise ValueError('Original evidence mutated')
            boundary=(prev is None or r['sequence']!=prev['sequence'] or r['frame_id']!=prev['frame_id']+1
                      or r['image_size']!=prev['image_size'] or r['frame_id'] in protocol['boundaries'].get(r['sequence'],[]))
            if boundary or r['pred'] is None:past=[]
            eligible=r['pred'] is not None and len(past)>=3
            if r['temporal']['available']!=eligible or r['level_temporal']['available']!=eligible:raise ValueError('History availability differs')
            if eligible:
                current=[math.log(v) for v in sorted(r['pred'][2:4],reverse=True)]
                # Independent algebra: specified median slope/median extrapolation
                # equals endpoint secant and 1.5*last - 0.5*first (middle cancels).
                expected=[1.5*past[-1][1][j]-.5*past[-3][1][j] for j in (0,1)]
                slope=[(past[-1][1][j]-past[-3][1][j])/2 for j in (0,1)]
                residual=[current[j]-expected[j] for j in (0,1)]
                level=max(abs(current[j]-median(h[1][j] for h in past[-3:])) for j in (0,1))
                values=[(max(map(abs,residual)),r['temporal']['primary']),(level,r['level_temporal']['primary'])]
                for name,values_expected in [('slope',slope),('predicted_log_edges',expected),('signed_log_change',residual)]:
                    values.extend(zip(values_expected,r['temporal'][name]))
                delta=max(abs(x-y) for x,y in values);diff=max(diff,delta)
                if delta>1e-12:raise ValueError('Past-only replay differs')
                if r['temporal']['history_frames']!=[h[0] for h in past[-3:]]:raise ValueError('History identity differs')
            elif r['temporal']['primary'] is not None:raise ValueError('Unsupported scored')
            if r['pred'] is not None:past.append((r['frame_id'],[math.log(v) for v in sorted(r['pred'][2:4],reverse=True)]))
            prev=r;total+=1
        for level,cut in ((False,report['cutoff']),(True,report['level_cutoff'])):
            key='level_temporal' if level else 'temporal'
            if role=='TRAIN':
                eligible=[r for r in rows if r[key]['available'] and r['original_simple_decision']['size_accepted'] and bad(r) is False]
                thresholds=[]
                for group,record in cut['group_protection'].items():
                    v=sorted(r[key]['primary'] for r in part_for(eligible,group))
                    value=v[math.ceil(.95*len(v))-1]
                    if value!=record['cutoff'] or len(v)!=record['correct_outputs']:raise ValueError('Group95 fitting differs')
                    thresholds.append(value)
                if max(thresholds)!=cut['risk_le']:raise ValueError('Global TRAIN cutoff differs')
            evals=report['level_roles' if level else 'roles'][role]
            for group,group_report in evals['full_groups'].items():
                part=part_for(rows,group);base={r['image'] for r in part if r['original_simple_decision']['size_accepted']}
                chosen=veto(part,cut['risk_le'],level)
                check_summary(part,base,group_report['formal']);check_summary(part,chosen,group_report['temporal_veto']);checks+=2
                for method,controls in group_report['controls'].items():
                    for name,target in [('same_count',len(chosen)),('same_CR',sum(not bad(r) for r in part if r['image'] in chosen))]:
                        ids=rank_ids(part,method,count=target) if name=='same_count' else rank_ids(part,method,correct=target)
                        check_summary(part,ids,controls[name]);checks+=1
                supported=[r for r in part if r[key]['available']]
                for method,c in evals['supported_evidence'][group]['comparisons'].items():
                    good=[r for r in supported if bad(r) is False];wrong=[r for r in supported if bad(r) is True]
                    risk=lambda r: r[key]['primary'] if method=='temporal' else r['risks'][method]
                    auc=sum((risk(x)>risk(y))+.5*(risk(x)==risk(y)) for x in wrong for y in good)/(len(good)*len(wrong)) if good and wrong else None
                    if auc is not None and abs(auc-c['AUROC'])>1e-12:raise ValueError('Pairwise AUROC differs')
                    checks+=1
        for group,c in report['trend_vs_level'][role].items():
            part=part_for(rows,group);old=veto(part,report['level_cutoff']['risk_le'],True);new=veto(part,report['cutoff']['risk_le'])
            check_summary(part,old,c['level']);check_summary(part,new,c['trend']);checks+=2
            changes={name:[] for name in ('restored_CR','new_FR','rescued_FA','new_FA')}
            for r in part:
                o,n=r['image'] in old,r['image'] in new
                if o==n or bad(r) is None:continue
                name=('new_FA' if bad(r) else 'restored_CR') if n else ('rescued_FA' if bad(r) else 'new_FR')
                changes[name].append(r['image'])
            if changes!=c['transitions']['images'] or {k:len(v) for k,v in changes.items()}!=c['transitions']['counts']:raise ValueError('Transition decomposition differs')
            targets=[('level_same_trend_count',len(new),'count',True),
                     ('level_same_trend_CR',c['trend']['states']['CR'],'correct',True),
                     ('trend_same_level_CR',c['level']['states']['CR'],'correct',False)]
            for name,target,kind,level in targets:
                ids=matched_veto_ids(part,target,kind,level)
                if (ids is not None)!=c[name]['feasible']:raise ValueError('Matched feasibility differs')
                if ids is not None:check_summary(part,ids,c[name]);checks+=1
            ids=FA_budget_ids(part,c['level']['states']['FA'])
            if (ids is not None)!=c['trend_at_level_FA_budget']['feasible']:raise ValueError('FA budget feasibility differs')
            if ids is not None:check_summary(part,ids,c['trend_at_level_FA_budget']);checks+=1
    if total!=3445 or seen!=set(raw) or report['TEST_opened'] or report['training_updates'] or report['detector_inferences']:raise ValueError('Scope differs')
    sys.path.insert(0,str(ROOT))
    from crane_project.utils import port_reliability_temporal_trend_v1 as core
    val=read(a.result/'scored_VAL.jsonl.gz')
    if core.gate(core.evaluate(val,report['cutoff']['risk_le']),core.comparison(val,report['level_cutoff']['risk_le'],report['cutoff']['risk_le']))!=report['gate']:raise ValueError('Gate differs')
    result=dict(passed=True,rows=total,independent_summary_AUROC_checks=checks,
                max_temporal_replay_difference=diff,source_hashes_verified=True,TEST_opened=False,
                independent_run_lengths=True,absent_local_frontend_files=absent,
                local_input_hashes_complete=not absent,report_sha256=hashlib.sha256((a.result/'report.json').read_bytes()).hexdigest())
    with a.out.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
