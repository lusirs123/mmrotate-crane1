#!/usr/bin/env python3
"""Read completed S028 slots, classify final delivered paired responses only."""
import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.tools import replay_port_size_seven_v1 as r

EPS=1e-6  # Relative/log diagnostic tolerance; never an adoption threshold.
NATURAL_EPS=1e-12  # Retain the original deterministic fixed-M harm definition.
SLOTS=('long_plus','long_minus','short_plus','short_minus')


def mean(x):return math.fsum(x)/len(x) if x else None
def p95(x):
    if not x:return None
    x=sorted(x);k=(len(x)-1)*.95;i=int(k);return x[i]+(x[min(i+1,len(x)-1)]-x[i])*(k-i)
def pct(x):return 100*mean(x) if x else None
def csv_write(p,rs):
    import csv
    keys=list(dict.fromkeys(k for x in rs for k in x))
    with Path(p).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rs)
def read_rows(p):
    with gzip.open(p,'rt') as f:return [json.loads(s) for s in f]


def natural_stats(rs,edge):
    j=0 if edge=='long' else 1
    before=[x['M_errors']['relative'][j] for x in rs];after=[x['slot']['final_errors']['relative'][j] for x in rs]
    return dict(n=len(rs),M_MAE_pct=pct(list(map(abs,before))),natural_MAE_pct=pct(list(map(abs,after))),
        harmed=sum(abs(b)>abs(a)+NATURAL_EPS for a,b in zip(before,after)),
        improved=sum(abs(b)<abs(a)-NATURAL_EPS for a,b in zip(before,after)),
        changed=sum(abs(math.log1p(b)-math.log1p(a))>EPS for a,b in zip(before,after)),
        natural_damage_positive_mean_pp=pct([max(abs(b)-abs(a),0.) for a,b in zip(before,after)]))


def response_detail(view,name):
    slots={s['slot']:s for s in view['slots']};natural=slots['natural'];s=slots[name]
    changed=0 if name.startswith('long') else 1;u=1-changed
    m=r.errors(sorted(view['midpoint'][2:4],reverse=True),view['gt'])
    # Finals always preserve natural-M raw association. Canonical long/short
    # can therefore be compared; Q/pre-candidate crossings are not responses.
    a,b=natural['final_errors']['relative'],s['final_errors']['relative']
    response=math.log(s['final_canonical'][u+2]/natural['final_canonical'][u+2])
    dm=abs(a[u])-abs(m['relative'][u]);additional=abs(b[u])-abs(a[u]);total=abs(b[u])-abs(m['relative'][u])
    return dict(image=view['image'],scale=view['scale'],domain=view['domain'],sequence=view['sequence'],slot=name,
        untouched_edge=('long','short')[u],M_near=abs(m['relative'][u])<=.03,both_M_near=max(map(abs,m['relative']))<=.03,
        M_error=m['relative'][u],natural_error=a[u],perturbed_error=b[u],log_response=response,
        relative_response=s['final_canonical'][u+2]/natural['final_canonical'][u+2]-1,
        error_change=additional,natural_damage=dm,total_damage=total,
        natural_harmed=dm>NATURAL_EPS,extra_harmed=additional>EPS,extra_improved=additional < -EPS,
        newly_harmed_M=dm<=NATURAL_EPS and total>EPS,rescued_to_M=dm>NATURAL_EPS and total<=EPS,
        natural_to_perturbed_status=natural['status']+'->'+s['status'],status_changed=natural['status']!=s['status'],
        either_fallback=natural['whole_pair_fallback'] or s['whole_pair_fallback'],
        both_accepted=natural['accepted'] and s['accepted'],input_axis_cross=s['input_axis_cross'],pre_axis_cross=s['pre_axis_cross'],
        target_recovered_vs_input=abs(b[changed])<abs(s['input_errors']['relative'][changed])-EPS,
        target_improved_vs_natural=abs(b[changed])<abs(a[changed])-EPS,
        ratio_before=natural['final_errors']['r'],ratio_after=s['final_errors']['r'],
        ratio_change=abs(s['final_errors']['r'])-abs(natural['final_errors']['r']),
        a_before=natural['final_errors']['a'],a_after=s['final_errors']['a'])


def response_stats(xs):
    logs=[abs(x['log_response']) for x in xs];changes=[x['error_change'] for x in xs]
    return dict(n=len(xs),output_log_meanabs=mean(logs),output_log_P95=p95(logs),
        output_relative_meanabs_pct=pct([abs(x['relative_response']) for x in xs]),
        response_above_roundoff=sum(v>EPS for v in logs),natural_MAE_pct=pct([abs(x['natural_error']) for x in xs]),
        perturbed_MAE_pct=pct([abs(x['perturbed_error']) for x in xs]),mean_error_change_pp=pct(changes),
        natural_harmed=sum(x['natural_harmed'] for x in xs),extra_harmed=sum(x['extra_harmed'] for x in xs),
        extra_improved=sum(x['extra_improved'] for x in xs),natural_harmed_and_extra_harmed=sum(x['natural_harmed'] and x['extra_harmed'] for x in xs),
        newly_harmed_M=sum(x['newly_harmed_M'] for x in xs),rescued_to_M=sum(x['rescued_to_M'] for x in xs),
        natural_damage_positive_mean_pp=pct([max(x['natural_damage'],0.) for x in xs]),
        extra_damage_positive_mean_pp=pct([max(x['error_change'],0.) for x in xs]),
        target_recovered_vs_input=sum(x['target_recovered_vs_input'] for x in xs),
        target_recovered_and_untouched_harmed=sum(x['target_recovered_vs_input'] and x['extra_harmed'] for x in xs),
        target_improved_vs_natural=sum(x['target_improved_vs_natural'] for x in xs),
        target_improved_and_untouched_harmed=sum(x['target_improved_vs_natural'] and x['extra_harmed'] for x in xs),
        ratio_meanabs_before=mean([abs(x['ratio_before']) for x in xs]),ratio_meanabs_after=mean([abs(x['ratio_after']) for x in xs]),
        ratio_harmed=sum(x['ratio_change']>EPS for x in xs),ratio_improved=sum(x['ratio_change'] < -EPS for x in xs),
        status_changed=sum(x['status_changed'] for x in xs),either_fallback=sum(x['either_fallback'] for x in xs),
        both_accepted=sum(x['both_accepted'] for x in xs))


def group_views(vs):
    for scale in (.5,1.):
        base=[v for v in vs if v['scale']==scale]
        groups=dict(overall=base)
        groups.update({d:[v for v in base if v['domain']==d] for d in ('real','sim')})
        groups.update({s:[v for v in base if v['sequence']==s] for s in sorted({v['sequence'] for v in base})})
        for group,sub in groups.items():yield scale,group,sub


def selected(xs,member):
    return xs if member=='all' else [x for x in xs if x['both_M_near']] if member=='both_near' else [x for x in xs if x['M_near']==(member=='near')]


def run(directory,out):
    index=r.read(directory/'artifacts.json')['files']
    for n,h in index.items():
        if r.sha(directory/n)!=h:raise ValueError('Replay artifact SHA')
    done=r.read(directory/'completion.json')
    if done['status']!='FROZEN_TRAIN_SEVEN_REPLAY_COMPLETE':raise ValueError('Engineering gate not passed')
    out.mkdir(parents=True,exist_ok=False)
    models={a:read_rows(directory/(a+'.rows.jsonl.gz')) for a in ('c20','p07')}
    for arm,vs in models.items():
        if len(vs)!=5116 or sum(v['midpoint'] is not None for v in vs)!=5112:raise ValueError('Replay view coverage')
        for v in vs:
            if [s['slot'] for s in v['slots']]!=['natural',*SLOTS,'long_exact','short_exact']:raise ValueError('Slot order')
            if v['midpoint'] is None:continue
            for s in v['slots']:
                if s['final_axis_cross']:raise ValueError('Protected output axis changed')
                for j in (0,1):
                    e=s['final_canonical'][j+2]/sorted(v['gt'][2:4],reverse=True)[j]-1
                    if abs(e-s['final_errors']['relative'][j])>1e-12:raise ValueError('Final geometry error mismatch')
    natural=[];responses=[];exact=[];details={}
    for arm,views in models.items():
        present=[v for v in views if v['midpoint'] is not None]
        details[arm]=[response_detail(v,s) for v in present for s in SLOTS]
        for scale,group,vs in group_views(present):
            stem=dict(arm=arm,scale=scale,group=group)
            for j,edge in enumerate(('long','short')):
                ns=[]
                for v in vs:
                    m=r.errors(sorted(v['midpoint'][2:4],reverse=True),v['gt'])
                    ns.append(dict(M_errors=m,slot=v['slots'][0],M_near=abs(m['relative'][j])<=.03,both_M_near=max(map(abs,m['relative']))<=.03))
                for member in ('all','near','error','both_near'):natural.append(dict(stem,edge=edge,member=member,**natural_stats(selected(ns,member),edge)))
            subset=[d for d in details[arm] if d['scale']==scale and (group=='overall' or d['domain']==group or d['sequence']==group)]
            for slot in SLOTS:
                for member in ('all','near','error','both_near'):
                    xs=selected([x for x in subset if x['slot']==slot],member)
                    responses.append(dict(stem,slot=slot,untouched_edge='short' if slot.startswith('long') else 'long',member=member,**response_stats(xs)))
            for slot,j in [('long_exact',0),('short_exact',1)]:
                cells=[]
                for v in vs:
                    s=next(x for x in v['slots'] if x['slot']==slot)
                    # FP32 GT proposal is exactly zero in target; reconstructing
                    # raw decimal error may have tiny rounding, handled by EPS.
                    if abs(s['input_errors']['relative'][j])>EPS:raise ValueError('GT exact input not exact')
                    cells.append((v,s))
                exact.append(dict(stem,slot=slot,n=len(cells),
                    pre_fixed_MAE_pct=pct([abs(s['pre_fixed_errors']['relative'][j]) for v,s in cells]),
                    final_MAE_pct=pct([abs(s['final_errors']['relative'][j]) for v,s in cells]),
                    pre_fixed_nonzero=sum(abs(s['pre_fixed_errors']['relative'][j])>EPS for v,s in cells),
                    final_deviates_from_GT=sum(abs(s['final_errors']['relative'][j])>EPS for v,s in cells),
                    fallback_count=sum(s['whole_pair_fallback'] for v,s in cells),
                    final_deviates_accepted=sum(s['accepted'] and abs(s['final_errors']['relative'][j])>EPS for v,s in cells),
                    final_deviates_fallback=sum(s['whole_pair_fallback'] and abs(s['final_errors']['relative'][j])>EPS for v,s in cells)))
    intersections=[]
    a=details['c20'];b=details['p07']
    if len(a)!=len(b):raise ValueError('Unpaired responses')
    pairs=[]
    for x,y in zip(a,b):
        if any(x[k]!=y[k] for k in ('image','scale','domain','sequence','slot','M_near','both_M_near')):raise ValueError('Response identity mismatch')
        pairs.append((x,y))
    for scale,group,vs in group_views([v for v in models['p07'] if v['midpoint'] is not None]):
        for slot in SLOTS:
            for member in ('all','near','error','both_near'):
                sub=[(x,y) for x,y in pairs if x['scale']==scale and x['slot']==slot and (group=='overall' or x['domain']==group or x['sequence']==group)]
                keep={id(x) for x in selected([x for x,y in sub],member)};sub=[(x,y) for x,y in sub if id(x) in keep]
                intersections.append(dict(scale=scale,group=group,slot=slot,member=member,n=len(sub),
                    both_extra_harmed=sum(x['extra_harmed'] and y['extra_harmed'] for x,y in sub),
                    C_only_extra_harmed=sum(x['extra_harmed'] and not y['extra_harmed'] for x,y in sub),
                    P_only_extra_harmed=sum(not x['extra_harmed'] and y['extra_harmed'] for x,y in sub),
                    neither_extra_harmed=sum(not x['extra_harmed'] and not y['extra_harmed'] for x,y in sub),
                    both_extra_improved=sum(x['extra_improved'] and y['extra_improved'] for x,y in sub),
                    C_only_extra_improved=sum(x['extra_improved'] and not y['extra_improved'] for x,y in sub),
                    P_only_extra_improved=sum(not x['extra_improved'] and y['extra_improved'] for x,y in sub)))
    for name,xs in [('natural_damage.csv',natural),('paired_responses.csv',responses),('GT_exact_damage.csv',exact),('C_P_intersections.csv',intersections)]:csv_write(out/name,xs)
    with gzip.open(out/'paired_response_rows.jsonl.gz','xt') as f:
        for arm,xs in details.items():
            for x in xs:f.write(json.dumps(dict(arm=arm,**x),allow_nan=False)+'\n')
    r.write(out/'verification.json',dict(status='FINAL_DELIVERED_PAIRED_RESPONSES_ANALYZED',replay_artifact_index_sha256=r.sha(directory/'artifacts.json'),
        slots_per_arm=5116*7,present_slots_per_arm=5112*7,missing_slots_per_arm=28,relative_log_response_tolerance=EPS,
        natural_harm_tolerance=NATURAL_EPS,head_weights=done['weights'],source_script_sha256=r.sha(__file__),
        note='All response utility uses final boxes vs same-view natural slot. Positive-part damage means are NOT an additive causal decomposition. GT-near is diagnostic only, not an inference input.',
        model_inference_calls=0,training_updates=0,val_test_depth_read=False))
    r.write(out/'artifacts.json',dict(files={p.name:r.sha(p) for p in out.iterdir() if p.is_file()}))
    print('ANALYZED',len(natural),len(responses),len(exact),len(intersections))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--out',required=True,type=Path);x=p.parse_args();run(x.directory,x.out)
