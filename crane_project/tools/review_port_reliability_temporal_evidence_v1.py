#!/usr/bin/env python3
"""Independent raw-input temporal replay and confusion/rank review, CPU only."""
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
    with gzip.open(path,'rt') as f:return [json.loads(l) for l in f]


def bad(r):
    if r['pred'] is None:return None
    p,g=sorted(r['pred'][2:4]),sorted(r['gt'][2:4])
    return any(abs(x-y)/y>.1 for x,y in zip(p,g))


def counts(rows, ids):
    c={s:0 for s in ('FA','FR','ED','CR','MISSING')}
    for r in rows:
        wrong=bad(r);keep=r['image'] in ids
        if wrong is None:s='MISSING'
        elif keep:s='FA' if wrong else 'CR'
        else:s='ED' if wrong else 'FR'
        c[s]+=1
    return c


def main():
    p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    report=json.loads((a.result/'report.json').read_text());protocol=json.loads((ROOT/'crane_project/tools/port_reliability_temporal_evidence_v1_protocol.json').read_text())
    for name,h in dict(report['input_pins'],**report['sources']['sources']).items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h:raise ValueError('Source changed '+name)
    raw={r['image']:r for r in read(ROOT/protocol['rows'])};rows_count=0;checks=0;maxdiff=0.
    for role in ('TRAIN','VAL'):
        scored=read(a.result/('scored_'+role+'.jsonl.gz'));past=[];prev=None;resolution=None
        for r in scored:
            original=raw[r['image']]
            for key in ('pred','gt','image_size','original_simple_decision','split','frame_id','sequence'):
                if r[key]!=original[key]:raise ValueError('Original evidence mutated')
            boundary=(prev is None or r['sequence']!=prev['sequence'] or r['frame_id']!=prev['frame_id']+1
                or resolution!=r['image_size'] or r['frame_id'] in protocol['boundaries'].get(r['sequence'],[]))
            if boundary or r['pred'] is None:past=[]
            eligible=r['pred'] is not None and len(past)>=3
            if r['temporal']['available']!=eligible:raise ValueError('Causal availability differs')
            if eligible:
                e=[math.log(x) for x in sorted(r['pred'][2:4],reverse=True)]
                delta=[e[i]-median([h[1][i] for h in past[-3:]]) for i in (0,1)]
                difference=abs(max(map(abs,delta))-r['temporal']['primary']);maxdiff=max(maxdiff,difference)
                if difference>1e-12 or r['temporal']['history_frames']!=[h[0] for h in past[-3:]]:
                    raise ValueError('Temporal replay differs')
            if r['pred'] is not None:past.append((r['frame_id'],[math.log(x) for x in sorted(r['pred'][2:4],reverse=True)]))
            prev=r;resolution=r['image_size'];rows_count+=1
        cutoff=report['cutoff']['risk_le']
        if role=='TRAIN':
            eligible=[r for r in scored if r['temporal']['available'] and bad(r) is False and r['original_simple_decision']['size_accepted']]
            thresholds=[]
            for group in report['cutoff']['group_protection']:
                part=eligible if group=='all' else [r for r in eligible if r[group.split(':')[0]]==group.split(':')[1]]
                v=sorted(r['temporal']['primary'] for r in part)
                thresholds.append(v[math.ceil(.95*len(v))-1])
            if max(thresholds)!=cutoff:raise ValueError('TRAIN threshold differs')
        for group,v in report['roles'][role]['full_groups'].items():
            part=scored if group=='all' else [r for r in scored if r[group.split(':')[0]]==group.split(':')[1]]
            base={r['image'] for r in part if r['original_simple_decision']['size_accepted']}
            chosen={r['image'] for r in part if r['image'] in base and (not r['temporal']['available'] or r['temporal']['primary']<=cutoff)}
            if counts(part,base)!=v['formal']['states'] or counts(part,chosen)!=v['temporal_veto']['states']:raise ValueError('Full states differ')
            checks+=2
            for method,controls in v['controls'].items():
                out=sorted([r for r in part if r['pred'] is not None],key=lambda r:(r['risks'][method],r['image']))
                k=len(chosen);cr=v['temporal_veto']['states']['CR'];seen=0;t=None
                if counts(part,{r['image'] for r in out[:k]})!=controls['same_count']['states']:raise ValueError('Same count differs')
                if cr:
                    for r in out:
                        seen+=int(not bad(r))
                        if seen==cr:t=r['risks'][method];break
                selected={r['image'] for r in out if t is not None and r['risks'][method]<=t}
                if counts(part,selected)!=controls['same_CR']['states']:raise ValueError('Same CR differs')
                checks+=2
            supported=[r for r in part if r['temporal']['available']]
            ev=report['roles'][role]['supported_evidence'][group]
            for method,c in ev['comparisons'].items():
                good=[r for r in supported if bad(r) is False];wrong=[r for r in supported if bad(r) is True]
                auc=sum((x['risks'][method]>y['risks'][method])+.5*(x['risks'][method]==y['risks'][method]) for x in wrong for y in good)/(len(good)*len(wrong)) if good and wrong else None
                if auc is not None and abs(auc-c['AUROC'])>1e-12:raise ValueError('Independent pairwise AUROC differs')
                checks+=1
    if rows_count!=3445 or report['TEST_opened'] or report['training_updates']:raise ValueError('Scope differs')
    # Rebuild the gate from independently checked counts and frozen implementation.
    sys.path.insert(0,str(ROOT))
    from crane_project.utils import port_reliability_temporal_evidence_v1 as core
    rereview=core.evaluate(read(a.result/'scored_VAL.jsonl.gz'),report['cutoff']['risk_le'])
    if rereview['gate']!=report['roles']['VAL']['gate']:raise ValueError('Gate differs')
    result=dict(passed=True,rows=rows_count,independent_summary_AUROC_checks=checks,
        max_temporal_replay_difference=maxdiff,source_hashes_verified=True,TEST_opened=False,
        report_sha256=hashlib.sha256((a.result/'report.json').read_bytes()).hexdigest())
    with a.out.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
