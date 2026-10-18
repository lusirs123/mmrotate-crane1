#!/usr/bin/env python3
"""Aggregate S035 independent same-batch responses; no model/VAL access."""
import argparse
from collections import defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.tools.run_port_size_native_train_v1 import sha,read,write
EPS=1e-7


def average(x):return math.fsum(x)/len(x) if x else None
def percentile(x):
    if not x:return None
    x=sorted(x);a=(len(x)-1)*.95;i=int(a)
    return x[i]+(x[min(i+1,len(x)-1)]-x[i])*(a-i)
def csv_write(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def cells(records):
    for scale in (.5,1.):
        view=[r for r in records if r['scale']==scale]
        for group in ['overall','real','sim']+sorted({r['sequence'] for r in view}):
            yield scale,group,[r for r in view if group in ('overall',r['domain'],r['sequence'])]


def summarize(records,parent):
    rows=[];paired=[]
    for scale,group,view in cells(records):
        for edge,j in (('long',0),('short',1),('both',None)):
            events=[]
            for r in view:
                g=sorted(r['gt'][2:4],reverse=True);before=sorted(r['before'][2:4],reverse=True);m=sorted(r['midpoint'][2:4],reverse=True)
                for k in ((0,1) if j is None else (j,)):
                    events.append(dict(before=before[k]/g[k]-1,m=m[k]/g[k]-1,parent_joint10=max(abs(before[q]/g[q]-1) for q in (0,1))<=.1,
                        after={name:dict(error=sorted(v['box'][2:4],reverse=True)[k]/g[k]-1,pre=sorted(v['pre_guard'][2:4],reverse=True)[k]/g[k]-1,accepted=v['accepted'],valid=v['valid']) for name,v in r['after'].items()},
                        pre_before=sorted(r['pre_guard_before'][2:4],reverse=True)[k]/g[k]-1))
            for stratum in ('all','parent_near','fixed_M_near','parent_joint10'):
                ee=[e for e in events if stratum=='all' or (stratum=='parent_near' and abs(e['before'])<=.03) or (stratum=='fixed_M_near' and abs(e['m'])<=.03) or (stratum=='parent_joint10' and e['parent_joint10'])]
                prefix=dict(parent=parent,scale=scale,group=group,edge=edge,stratum=stratum,edges=len(ee))
                modes=records[0]['after']
                for mode in modes:
                    changes=[abs(e['after'][mode]['error'])-abs(e['before']) for e in ee]
                    pre_changes=[abs(e['after'][mode]['pre'])-abs(e['pre_before']) for e in ee]
                    row=dict(prefix,mode=mode,before_mae_pct=100*average([abs(e['before']) for e in ee]) if ee else None,
                        after_mae_pct=100*average([abs(e['after'][mode]['error']) for e in ee]) if ee else None,
                        delta_mae_pp=100*average(changes) if ee else None,delta_preguard_mae_pp=100*average(pre_changes) if ee else None,
                        harmed=sum(d>EPS for d in changes),improved=sum(d<-EPS for d in changes),tied=sum(abs(d)<=EPS for d in changes),
                        harm_mass_pp_edges=100*math.fsum(max(d,0) for d in changes),gain_mass_pp_edges=100*math.fsum(max(-d,0) for d in changes),
                        accepted=sum(e['after'][mode]['accepted'] for e in ee),invalid=sum(not e['after'][mode]['valid'] for e in ee),
                        after_bias_pct=100*average([e['after'][mode]['error'] for e in ee]) if ee else None,
                        after_p95_pct=100*percentile([abs(e['after'][mode]['error']) for e in ee]) if ee else None)
                    rows.append(row)
                for optimizer in ('sgd','fresh_adam'):
                    s=optimizer+'/size';c=optimizer+'/combined';g=optimizer+'/geometry'
                    good=[e for e in ee if abs(e['after'][s]['error'])<abs(e['before'])-EPS]
                    weakened=[e for e in good if abs(e['after'][c]['error'])>abs(e['after'][s]['error'])+EPS]
                    paired.append(dict(prefix,optimizer=optimizer,size_beneficial=len(good),combined_retained=sum(abs(e['after'][c]['error'])<abs(e['before'])-EPS for e in good),
                        combined_harmed=sum(abs(e['after'][c]['error'])>abs(e['before'])+EPS for e in good),combined_weakened=len(weakened),
                        geometry_harmed_on_size_good=sum(abs(e['after'][g]['error'])>abs(e['before'])+EPS for e in good),
                        combination_minus_size_error_pp=100*average([abs(e['after'][c]['error'])-abs(e['after'][s]['error']) for e in ee]) if ee else None,
                        suppressed_gain_mass_pp_edges=100*math.fsum(abs(e['after'][c]['error'])-abs(e['after'][s]['error']) for e in weakened)))
    return rows,paired


def run(source,out):
    if out.exists() or out.parent!=source.parent:raise ValueError('Fresh sibling output')
    index=read(source/'artifacts.json')['files']
    if any(sha(source/n)!=s for n,s in index.items()):raise ValueError('Artifact SHA')
    diagnosis=read(source/'diagnosis.json');finished=read(source/'finished.json')
    if diagnosis['status']!='COMPLETE' or finished['status']!='COMPLETE':raise ValueError('Incomplete diagnosis')
    outcomes=[];pairs=[];batch_stats={}
    for parent,v in diagnosis['parents'].items():
        with gzip.open(source/(parent+'_responses.jsonl.gz'),'rt') as f:records=[json.loads(x) for x in f]
        if len(records)!=5112 or len({(r['scale'],r['image']) for r in records})!=5112:raise ValueError('Full TRAIN identities')
        if {r['scale'] for r in records}!={.5,1.} or any(r['sequence'] not in ('real_seq01','real_seq05','real_seq06','real_seq12','real_seq13','sim_seq08') for r in records):raise ValueError('TRAIN scope')
        rows,pp=summarize(records,parent);outcomes.extend(rows);pairs.extend(pp)
        steps=read(source/(parent+'_steps.json'))
        batch_stats[parent]=dict(batches=len(steps),negative_cosine=sum(s['gradient_cosine'] is not None and s['gradient_cosine']<0 for s in steps),
            decomposition_vector_max_abs=max(s['vector_sum_max_abs'] for s in steps),decomposition_parameter_max_abs=max(s['parameter_sum_max_abs'] for s in steps),
            changes={name:dict(size_loss_decreased=sum(s[name]['size_loss']<s['size_loss'] for s in steps),geometry_loss_decreased=sum(s[name]['geometry_loss']<s['geometry_loss'] for s in steps)) for name in steps[0] if '/' in name})
    # This is an evidence inventory, never an automatic training-admission gate.
    highlights=[r for r in outcomes if r['group'] in ('real','sim') and r['stratum']=='parent_near' and r['edge'] in ('long','short') and r['mode'] in ('sgd/geometry','sgd/size','sgd/combined')]
    result=dict(protocol='port_size_auxiliary_conflict_analysis_v1',source_artifacts_sha256=sha(source/'artifacts.json'),diagnosis=diagnosis,finished=finished,batch_statistics=batch_stats,
        eps_relative_for_counts=EPS,primary_near_inventory=highlights,
        boundary='Independent same-batch local responses at fixed lr; not new trained models, VAL evidence, causal reconstruction of24epochs, or a test of global uniqueness. Projection effects can interact through shared parameters.',
        code_sha256=sha(Path(__file__).resolve()))
    out.mkdir();csv_write(out/'actual_size_changes.csv',outcomes);csv_write(out/'size_gain_retention.csv',pairs);write(out/'analysis.json',result)
    write(out/'artifacts.json',dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(status='COMPLETE',rows=len(outcomes),paired_rows=len(pairs),batches=batch_stats),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.run.resolve(),a.out.resolve())
