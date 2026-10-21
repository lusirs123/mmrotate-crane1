#!/usr/bin/env python3
"""Post-fit same-epoch/selected response analysis; no model or re-selection."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.tools.analyze_port_size_qalign_v1 import csv_write

TOL=1e-6


def read(path):return json.loads(path.read_text())
def rows(path):
    with gzip.open(path,'rt') as f:return [json.loads(x) for x in f]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def response(row,field):
    m=sorted(row['midpoint'][2:4],reverse=True);g=sorted(row['gt'][2:4],reverse=True)
    p=row[field][2:4]
    if row['midpoint'][2]<row['midpoint'][3]:p=p[::-1]
    errors=[abs(x/y-1) for x,y in zip(p,g)];logs=[math.log(x/y) for x,y in zip(p,g)]
    updates=[math.log(x/y) for x,y in zip(p,m)]
    return errors,updates,sum(logs)/2,logs[0]-logs[1]


def summarize_pair(left,right,refs):
    if len(left)!=len(right):raise ValueError('Full paired rows required')
    cells=defaultdict(lambda:defaultdict(float))
    for l,r in zip(left,right):
        if any(l[k]!=r[k] for k in ('image','scale','domain','sequence','gt','midpoint')):raise ValueError('Pair identity mismatch')
        if l['midpoint'] is None:
            if r['edge_residual'] is not None:raise ValueError('New output')
            continue
        ref=refs[l['image'],l['scale']]
        for mode,field in [('delivered','edge_residual'),('pre_guard','pre_guard')]:
            el,ul,al,rl=response(l,field);er,ur,ar,rr=response(r,field)
            for group in ('overall',l['domain'],l['sequence']):
                joint=cells[l['scale'],group,'joint',mode];joint['n']+=1
                joint['ratio_left_sum']+=abs(rl);joint['ratio_right_sum']+=abs(rr)
                joint['scale_left_sum']+=abs(al);joint['scale_right_sum']+=abs(ar)
                better_ratio=abs(rr)<abs(rl)-TOL
                joint['ratio_improved']+=better_ratio
                joint['ratio_improved_with_edge_harm']+=better_ratio and any(y>x+TOL for x,y in zip(el,er))
                joint['ratio_improved_without_edge_harm']+=better_ratio and all(y<=x+TOL for x,y in zip(el,er))
                for j,edge in enumerate(('long','short')):
                    c=cells[l['scale'],group,edge,mode];c['n']+=1
                    c['mae_left_sum']+=el[j];c['mae_right_sum']+=er[j]
                    c['update_abs_left_sum']+=abs(ul[j]);c['update_abs_right_sum']+=abs(ur[j])
                    c['all_weaker']+=abs(ur[j])<abs(ul[j])-TOL
                    c['same_direction_weaker']+=ul[j]*ur[j]>1e-12 and abs(ur[j])<abs(ul[j])-TOL
                    c['reversed_vs_left']+=ul[j]*ur[j]<-1e-12
                    c['returned_to_neutral']+=abs(ul[j])>TOL and abs(ur[j])<=TOL
                    if ref['near'][j]:
                        c['near_count']+=1
                        c['near_harm_left_sum']+=max(0.,el[j]-ref['m_errors'][j])
                        c['near_harm_right_sum']+=max(0.,er[j]-ref['m_errors'][j])
                    if ref['gain'][j]:
                        c['gain_count']+=1
                        b=ref['m_errors'][j]-ref['a_errors'][j];c['a20_gain_sum']+=b
                        c['retained_left_sum']+=min(max(ref['m_errors'][j]-el[j],0.),b)
                        c['retained_right_sum']+=min(max(ref['m_errors'][j]-er[j],0.),b)
                        useful_left=el[j]<ref['m_errors'][j]-TOL
                        c['left_useful_preserved']+=useful_left and er[j]<ref['m_errors'][j]-TOL
                        c['left_useful_lost']+=useful_left and er[j]>=ref['m_errors'][j]-TOL
    result=[]
    for (scale,group,edge,mode),values in cells.items():
        v=dict(values)
        for k,x in list(v.items()):
            if k.endswith('_sum') and k not in ('a20_gain_sum','retained_left_sum','retained_right_sum'):
                v[k[:-4]+'_mean']=x/v['n']
        if v.get('a20_gain_sum',0)>0:
            v['retention_left']=v['retained_left_sum']/v['a20_gain_sum'];v['retention_right']=v['retained_right_sum']/v['a20_gain_sum']
        result.append(dict(scale=scale,group=group,edge=edge,mode=mode,**v))
    return result


def analyze(source,out):
    if out.exists() or out.parent!=source.parent:raise ValueError('Fresh sibling output required')
    idx=read(source/'artifacts.json')['files']
    if any(sha(source/n)!=h for n,h in idx.items()):raise ValueError('Frozen run SHA differs')
    training=read(source/'training.json')
    if training['status']!='COMPLETE' or training['formal_updates']!=11502:raise ValueError('Incomplete fit')
    selections={a:training['arms'][a]['representative'] for a in ('C','R','P')}
    if selections!={'C':6,'R':6,'P':5}:raise ValueError('Frozen original representative identity differs')
    results=[]
    for split in ('train','val'):
        refs={(r['image'],r['scale']):r for r in read(source/(split+'_fixed_references.json'))}
        cache={}
        for a in ('C','R','P'):
            for e in range(1,7):cache[a,e]=rows(source/a/('epoch_%02d_%s.rows.jsonl.gz'%(e,split)))
        comparisons=[('same_epoch',e,e,l,r) for e in range(1,7) for l,r in [('C','R'),('R','P'),('C','P')]]
        comparisons += [('original_representatives',selections[l],selections[r],l,r) for l,r in [('C','R'),('R','P'),('C','P')]]
        for kind,le,re,l,r in comparisons:
            results.extend(dict(split=split,comparison=kind,left=l,right=r,left_epoch=le,right_epoch=re,**v)
                for v in summarize_pair(cache[l,le],cache[r,re],refs))
    out.mkdir();csv_write(out/'paired_response.csv',results)
    (out/'verification.json').write_text(json.dumps(dict(source_artifacts_sha256=sha(source/'artifacts.json'),
        source_sha256=sha(Path(__file__).resolve()),same_epochs=list(range(1,7)),original_representatives=selections,
        pairwise_groups='R vs C, P vs R, P vs C; all TRAIN scales and natural VAL; overall/domain/video; pre-guard and final',
        tolerance=TOL,no_training_or_inference=True,no_reselection=True,
        limitation='Response magnitude and direction are descriptive; lower amplitude does not establish causal origin of protection. Two TRAIN scales are related views.'),indent=2)+'\n')
    (out/'artifacts.json').write_text(json.dumps(dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();analyze(a.run.resolve(),a.out.resolve())
