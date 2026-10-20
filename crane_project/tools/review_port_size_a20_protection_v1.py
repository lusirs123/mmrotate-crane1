#!/usr/bin/env python3
"""Review ONLY saved A20 TRAIN/VAL outputs. No model/inference/fit/network."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.utils import port_size_a20_protection_review_v1 as core
from crane_project.utils.port_geometry_size_distribution_metrics_v1 import group

A20_SHA='545e18459d3ead4a54c441d64de27ca44a8394bfdc20cc2caf4c2f7534326479'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def load(path):return json.loads(path.read_text())


def source_identity():
    paths=[Path(__file__).resolve(),ROOT/'crane_project/utils/port_size_a20_protection_review_v1.py',
           ROOT/'tests/test_port_size_a20_protection_review_v1.py',
           ROOT/'crane_project/tools/port_size_a20_protection_review_v1_protocol.json']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def summary(rows):
    valid=[r for r in rows if r['midpoint'] is not None]
    result=dict(views=len(rows),outputs=len(valid),missing=len(rows)-len(valid),
                delivery=dict(Counter(r['size_delivery'] for r in rows)),
                edges={},patterns=Counter(),frame_groups=Counter())
    values={a:[] for a in ('m','a20')}
    for r in valid:
        m,a=[core.relative_errors(r[k],r['gt']) for k in ('midpoint','edge_residual')]
        refs=core.fixed_references(m['absolute'],a['absolute'])
        values['m'].append(m);values['a20'].append(a)
        improve=[y < x-core.GAIN_TOL for x,y in zip(m['absolute'],a['absolute'])]
        worse=[y > x+core.GAIN_TOL for x,y in zip(m['absolute'],a['absolute'])]
        result['patterns']['both_improved']+=int(all(improve))
        result['patterns']['both_harmed']+=int(all(worse))
        result['patterns']['long_improved_short_not_improved']+=int(improve[0] and not improve[1])
        result['patterns']['long_improved_short_not_improved_ratio_harmed']+=int(improve[0] and not improve[1] and abs(a['r'])>abs(m['r'])+core.GAIN_TOL)
        result['patterns']['scale_improved_ratio_harmed']+=int(abs(a['a'])<abs(m['a'])-core.GAIN_TOL and abs(a['r'])>abs(m['r'])+core.GAIN_TOL)
        result['frame_groups'][group(r['gt'],r['midpoint'])]+=1
        for j,edge in enumerate(('long','short')):
            cell=result['edges'].setdefault(edge,dict(near=0,gain=0,overlap=0,union=0,near_harmed=0,
                near_error_m_sum=0.,near_error_a20_sum=0.,benefit_sum=0.,initial_hinge_sum=0.,
                axis_groups={g:dict(count=0,near=0,gain=0,union=0) for g in ('undersized','oversized','near3')}))
            n,g=refs['near'][j],refs['gain'][j];em,ea=m['absolute'][j],a['absolute'][j]
            cell['near']+=int(n);cell['gain']+=int(g);cell['overlap']+=int(n and g);cell['union']+=int(n or g)
            cell['near_harmed']+=int(n and ea>em+core.GAIN_TOL)
            if n:cell['near_error_m_sum']+=em;cell['near_error_a20_sum']+=ea
            if g:cell['benefit_sum']+=em-ea
            if n or g:cell['initial_hinge_sum']+=max(0.,ea-refs['reference'][j])
            label='near3' if n else 'undersized' if m['signed'][j]<0 else 'oversized'
            sub=cell['axis_groups'][label];sub['count']+=1;sub['near']+=int(n);sub['gain']+=int(g);sub['union']+=int(n or g)
    for arm,vs in values.items():
        result[arm]=dict(long_mae=sum(v['absolute'][0] for v in vs)/len(vs) if vs else None,
            short_mae=sum(v['absolute'][1] for v in vs)/len(vs) if vs else None,
            a_meanabs=sum(abs(v['a']) for v in vs)/len(vs) if vs else None,
            r_meanabs=sum(abs(v['r']) for v in vs)/len(vs) if vs else None)
    for e in result['edges'].values():
        assert e['union']==e['near']+e['gain']-e['overlap']
        for arm in ('m','a20'):
            e['near_mae_'+arm]=e['near_error_'+arm+'_sum']/e['near'] if e['near'] else None
    return result


def review(folder,out):
    if out.exists() or out.parent!=(ROOT/'work_dirs'/core.VERSION).resolve():
        raise ValueError('Fresh result-only local directory required')
    index=load(folder/'artifacts.json')['files'];names=['training.json','finished.json','protocol.json',
        'fixed_b/head_epoch_20.pth','fixed_b/epoch_20_train.rows.jsonl.gz','fixed_b/epoch_20_val.rows.jsonl.gz']
    for n in names:
        if sha(folder/n)!=index[n]:raise ValueError('Sealed A20 artifact drift '+n)
    if index['fixed_b/head_epoch_20.pth']!=A20_SHA:raise ValueError('Only fixed A20 ep20 allowed')
    training,finished=load(folder/'training.json'),load(folder/'finished.json')
    if training['status']!=finished['status'] or training['status']!='COMPLETE' or training['formal_head_updates']!=30672:
        raise ValueError('Requires complete S039, not a partial run')
    datasets={}
    for split,n in [('train',5116),('val',887)]:
        with gzip.open(folder/('fixed_b/epoch_20_'+split+'.rows.jsonl.gz'),'rt') as f:rr=[json.loads(line) for line in f]
        if len(rr)!=n or len({(r['image'],r['scale']) for r in rr})!=n:raise ValueError('Full unique rows required')
        for r in rr:
            allowed=('train','train_sim') if split=='train' else ('val',)
            if r['split'] not in allowed:raise ValueError('Wrong source split')
            m,a=r['midpoint'],r['edge_residual']
            if (m is None)!=(a is None):raise ValueError('Output identity drift')
            if m is not None and any(m[i]!=a[i] for i in (0,1,4,5)):raise ValueError('Frozen geometry/score changed')
        if sum(r['midpoint'] is not None for r in rr)!=(5112 if split=='train' else 886):raise ValueError('Output count drift')
        datasets[split]=rr
    if Counter(r['scale'] for r in datasets['train'])!=Counter({.5:2558,1.:2558}):raise ValueError('Two full TRAIN views required')
    contract=load(ROOT/'crane_project/tools/port_size_a20_protection_review_v1_protocol.json')
    if (contract['fixed_start']['sha256']!=A20_SHA or contract['losses']['ratio_weight']!=core.RATIO_WEIGHT
            or contract['losses']['protect_weight']!=core.PROTECT_WEIGHT
            or contract['fixed_masks']['epsilon']!=core.EPSILON):
        raise ValueError('Design/prototype constants drift')
    proof=source_identity()
    out.mkdir(parents=True)
    groups={};edge_rows=[]
    for split,rr in datasets.items():
        for scale in sorted({r['scale'] for r in rr}):
            ss=[r for r in rr if r['scale']==scale]
            keys=['overall','real','sim']+sorted({r['sequence'] for r in ss})
            for key in keys:
                sub=[r for r in ss if key=='overall' or r['domain']==key or r['sequence']==key]
                groups[split+'/'+str(scale)+'/'+key]=summary(sub)
            for label in ('need_enlarge','need_shrink','near_correct','mixed'):
                groups[split+'/'+str(scale)+'/M_group/'+label]=summary([r for r in ss if r['midpoint'] is not None and group(r['gt'],r['midpoint'])==label])
        for r in rr:
            if r['midpoint'] is None:continue
            m,a=[core.relative_errors(r[k],r['gt']) for k in ('midpoint','edge_residual')];refs=core.fixed_references(m['absolute'],a['absolute'])
            for j,edge in enumerate(('long','short')):
                edge_rows.append(dict(split=split,image=r['image'],sequence=r['sequence'],domain=r['domain'],scale=r['scale'],edge=edge,
                    frame_group=group(r['gt'],r['midpoint']),m_signed=m['signed'][j],m_error=m['absolute'][j],a20_error=a['absolute'][j],
                    near=refs['near'][j],gain=refs['gain'][j],active=refs['active'][j],reference=refs['reference'][j],
                    initial_hinge=max(0.,a['absolute'][j]-refs['reference'][j]) if refs['active'][j] else 0.,delivery=r['size_delivery']))
    result=dict(protocol=core.VERSION,status='SAVED_TRAIN_VAL_REVIEW_COMPLETE_NO_FIT',
        sources={n:index[n] for n in names},source_index_sha256=sha(folder/'artifacts.json'),a20_sha256=A20_SHA,
        image_identities=2558,train_views=5116,train_outputs=5112,val_frames=887,val_outputs=886,
        settings=dict(near=core.NEAR,gain_tolerance=core.GAIN_TOL,ratio_weight=core.RATIO_WEIGHT,protect_weight=core.PROTECT_WEIGHT,epsilon=core.EPSILON),
        groups=groups,train_missing=[dict(image=r['image'],scale=r['scale']) for r in datasets['train'] if r['midpoint'] is None],
        train_fallback=[dict(image=r['image'],scale=r['scale'],checks=r['failed_checks']) for r in datasets['train'] if r['size_delivery']=='fallback'],
        boundary='Saved final deliveries only. No cache/model load, parameter updates, new inference, TEST or depth access. Two TRAIN scales are paired views, not independent images. Prototype losses are not trained effects. Two TRAIN fallback pre-guard sizes are unavailable here; no inference from delivered identity about hidden candidate.')
    (out/'review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (out/'fixed_edge_memberships.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(edge_rows[0]));w.writeheader();w.writerows(edge_rows)
    if source_identity()!=proof:raise ValueError('Review source changed during execution')
    (out/'source_identity.json').write_text(json.dumps(proof,indent=2)+'\n')
    (out/'protocol.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2)+'\n')
    (out/'artifacts.json').write_text(json.dumps(dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],train_views=5116,val_frames=887,output=str(out))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();review(args.run.resolve(),args.out.resolve())
