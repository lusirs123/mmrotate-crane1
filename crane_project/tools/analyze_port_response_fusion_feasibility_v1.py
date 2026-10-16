"""Read-only five frozen scores, TRAIN/VAL only; no fitting, TEST or policy API."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import numpy as np
from crane_project.utils import port_reliability_paired_ranking_audit_v1 as rank
from crane_project.tools import analyze_port_midpoint_simple_disagreement_v1 as historical

ROOT=Path(__file__).resolve().parents[2]
VERSION='port_response_fusion_feasibility_v1'
MANIFEST=ROOT/'crane_project/tools'/ (VERSION+'_manifest.json')
METHODS=('formal','bce_1701','soft_1701','bce0_1701','soft0_1701')
SOURCE=ROOT/'work_dirs/port_reliability_response_l2_v1/20261015_l2zero_gpu_v1/result'
WARM=ROOT/'work_dirs/port_reliability_response_warm_v1/20261015_response_warm_gpu_v1/result'
CELLS=('all_accept','disagreement','all_reject','missing')
sha=historical.sha
read=historical.read
write=historical.write
rows=historical.rows

def cell(row,cuts):
    if row['pred'] is None:return 'missing'
    mask=[row['audit_risks'][n]<=cuts[n] for n in METHODS]
    return 'all_accept' if all(mask) else ('all_reject' if not any(mask) else 'disagreement')

def counts(part):
    return dict(correct=sum(rank.bad(r) is False for r in part),wrong=sum(rank.bad(r) is True for r in part),missing=sum(rank.bad(r) is None for r in part),frames=len(part))

def evidence(part):
    good=[r for r in part if rank.bad(r) is False];wrong=[r for r in part if rank.bad(r) is True]
    pairs=len(good)*len(wrong)
    auc={n:rank.Ranking(part,n).auc() for n in METHODS}
    dominance=dict(error_higher_all=0,error_lower_all=0,conflicting=0,identical=0)
    if pairs:
        g=np.array([[r['audit_risks'][n] for n in METHODS] for r in good])
        for r in wrong:
            d=np.array([r['audit_risks'][n] for n in METHODS])-g
            same=(d==0).all(1);hi=(d>=0).all(1)&~same;lo=(d<=0).all(1)&~same
            dominance['identical']+=int(same.sum());dominance['error_higher_all']+=int(hi.sum());dominance['error_lower_all']+=int(lo.sum());dominance['conflicting']+=int((~(same|hi|lo)).sum())
    return dict(**counts(part),pairs=pairs,AUROC=auc,pair_dominance=dominance,
        pair_fractions={k:v/pairs if pairs else None for k,v in dominance.items()},single_class=not bool(pairs),
        fewer_than10_in_either_class=min(len(good),len(wrong))<10,
        limitations='Fixed risk direction; conditional AUC and dominance do not establish learnable fusion; correlated pairs are not independent observations. Lower-all only obstructs coordinatewise nondecreasing fusion, not arbitrary signed models.')

def complement(part,first,other):
    out=dict(new_ED=0,new_FR=0,rescued_CR=0,released_FA=0)
    for r in part:
        b=rank.bad(r)
        if b is None:continue
        a=r['image'] in first;c=r['image'] in other
        if a and not c:out['new_ED' if b else 'new_FR']+=1
        if not a and c:out['released_FA' if b else 'rescued_CR']+=1
    return out

def accepted_at(part,name,cut):
    return {r['image'] for r in part if r['pred'] is not None and r['audit_risks'][name]<=cut}

def analyze(data,cuts):
    result={}
    for group,part in rank.groups(data).items():
        cells={c:[r for r in part if cell(r,cuts)==c] for c in CELLS}
        patterns={}
        for r in part:
            key='MISSING' if r['pred'] is None else ''.join('1' if r['audit_risks'][n]<=cuts[n] else '0' for n in METHODS)
            patterns.setdefault(key,[]).append(r)
        simple=accepted_at(part,'formal',cuts['formal'])
        each={};matched={}
        for name in METHODS:
            keep=accepted_at(part,name,cuts[name]);each[name]=dict(delivery=rank.summarize(part,keep),changes_vs_simple=complement(part,simple,keep))
            if name!='formal':
                sr=rank.Ranking(part,'formal');nr=rank.Ranking(part,name)
                cr=each['formal']['delivery']['states']['CR'];number=each['formal']['delivery']['accepted']
                a=nr.same_cr(cr);b=nr.same_count(number)
                matched[name]={}
                for mode,point in [('same_CR',a),('same_count',b)]:
                    ids=historical.accepted(nr,point,mode)
                    matched[name][mode]=dict(point=point,changes_vs_simple=complement(part,simple,ids))
        allids={r['image'] for r in cells['all_accept']};anyids={r['image'] for r in part if r['pred'] is not None and cell(r,cuts)!='all_reject'}
        result[group]=dict(cells={c:evidence(p) for c,p in cells.items()},patterns={k:counts(v) for k,v in patterns.items()},
            individual=each,matched=matched,intersection=rank.summarize(part,allids),union=rank.summarize(part,anyids),
            oracle_fixed_decision=dict(FA_lower_bound=sum(rank.bad(r) is True for r in cells['all_accept']),CR_upper_bound=sum(rank.bad(r) is False for r in part if r['image'] in anyids),
                GT_only=True,not_jointly_achievable_without_GT=True,scope='Only selecting among five original binary decisions; not a bound on continuous-score fusion'))
    return result

def inventory():
    # Only filename inventory, not file contents; never open/hash TEST artifacts.
    scopes=('port_reliability_response_stats_v1','port_reliability_response_tail_v1','port_reliability_response_warm_v1','port_reliability_response_l2_v1','port_reliability_image_temporal_crossfit_v1')
    found={}
    for name in scopes:
        base=ROOT/'work_dirs'/name
        found[name]=[str(p.relative_to(ROOT)) for p in sorted(base.rglob('*')) if p.is_file() and ('crossfit' in p.name.lower() or 'oof' in p.name.lower() or 'leave' in p.name.lower() or p.name in ('models.json','protocol.json','TRAIN_image_scores.json','quality_heads.json'))]
    return dict(scope=list(scopes),matching_filename_inventory=found,
        exact_five_score_video_holdout_material_found=False,
        reason='Current response parents and continuations fit full TRAIN, including normalization; historical image/temporal crossfit uses different input/head/objective and cannot substitute for current five-score held-out predictions.',
        limitation='Scoped saved artifact inventory plus pinned current protocols; not a claim that every server directory has been searched. No new crossfit fits, no inferred OOF identity from a filename.')

def run(out):
    if Path.cwd().resolve()!=ROOT or out.exists() or out.parent.parent!=ROOT/'work_dirs'/VERSION:raise ValueError('New result path required from project root')
    protocol=read(MANIFEST)
    for p,h in protocol['pins'].items():
        if sha(ROOT/p)!=h:raise ValueError('Changed source '+p)
    out.mkdir(parents=True);write(out/'protocol.json',protocol)
    write(out/'runtime.json',dict(cwd=str(ROOT),environment=os.environ.get('CONDA_DEFAULT_ENV'),HEAD=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),device='CPU',training=False))
    freeze=read(SOURCE/'freeze.json');cuts={k:freeze['cutoffs'][k] for k in METHODS}
    result={}
    for role,expected in [('TRAIN',2558),('VAL',887)]:
        data=rows(SOURCE/('scored_'+role+'.jsonl.gz'));reference=rank.validate_rows(rows(WARM/('scored_'+role+'.jsonl.gz')),role,expected)
        paired=rank.validate_rows(data,role,expected);rank.pair_identity(reference,paired)
        if len(data)!=expected:raise ValueError('Incomplete role')
        for r in data:
            r['audit_risks']={k:r['audit_risks'][k] for k in METHODS}
            for n in METHODS:
                risk=r['audit_risks'][n]
                if r['pred'] is not None and not (math.isfinite(risk) and 0<=risk<=1):raise ValueError('Invalid risk')
            for n in METHODS[:3]:
                if r['audit_risks'][n]!=reference[r['image']]['audit_risks'][n]:raise ValueError('Reused scoring changed')
        result[role]=analyze(data,cuts)
        with gzip.open(out/('paired_'+role+'.jsonl.gz'),'xt') as f:
            for r in data:f.write(json.dumps(r,allow_nan=False)+'\n')
        print('Analyzed',role,flush=True)
    inv=inventory();write(out/'material_inventory.json',inv)
    report=dict(status='FUSION_FEASIBILITY_AUDIT_ONLY',methods=METHODS,cutoffs=cuts,results=result,material=inv,
        conclusion=dict(fusion_fit_performed=False,ready_to_train_now=False,reason='Exact video-held-out meta-training scores missing; conditional signals and support must be assessed, no causal or joint-separability proof from marginal AUROC.'),
        TEST_opened=False,TEST_hashes_read=False,scorer_fits=0,policy_export=False,GT_online=False)
    write(out/'report.json',report)
    with (out/'counts.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['role','group','cell','correct','wrong','missing'])
        for role,gg in result.items():
            for group,v in gg.items():
                for c,d in v['cells'].items():w.writerow([role,group,c,d['correct'],d['wrong'],d['missing']])
    with (out/'conditional_auc.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['role','group','cell','method','correct','wrong','AUROC'])
        for role,gg in result.items():
            for group,v in gg.items():
                for c,d in v['cells'].items():
                    for n,a in d['AUROC'].items():w.writerow([role,group,c,n,d['correct'],d['wrong'],a])
    lines=['# 五个冻结评分融合可行性核对','','无训练、无TEST读取、无新policy。','','|角色|组|共同接受 对/错|分歧 对/错|共同拒绝 对/错|交集FA/CR|并集FA/CR|','|---|---|---|---|---|---|---|']
    for role,gg in result.items():
        for group,v in gg.items():
            celltext=['%d/%d'%(v['cells'][c]['correct'],v['cells'][c]['wrong']) for c in CELLS[:3]]
            state=lambda key:'%d/%d'%(v[key]['states']['FA'],v[key]['states']['CR'])
            lines.append('|%s|%s|%s|%s|%s|'%(role,group,'|'.join(celltext),state('intersection'),state('union')))
    lines+=['',json.dumps(report['conclusion'],ensure_ascii=False)]
    (out/'analysis.md').write_text('\n'.join(lines)+'\n')
    write(out/'completion.json',dict(artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file()}))

def review(out,receipt):
    report=read(out/'report.json');protocol=read(out/'protocol.json')
    for p,h in protocol['pins'].items():
        if sha(ROOT/p)!=h:raise ValueError('Pin mismatch')
    for n,h in read(out/'completion.json')['artifacts'].items():
        if sha(out/n)!=h:raise ValueError('Artifact changed')
    cuts=report['cutoffs'];pairs=0;checks=0
    for role,gg in report['results'].items():
        data=rows(out/('paired_'+role+'.jsonl.gz'));original=rank.validate_rows(rows(SOURCE/('scored_'+role+'.jsonl.gz')),role,len(data))
        rank.pair_identity(rank.validate_rows(data,role,len(data)),original)
        for r in data:
            for n in METHODS:
                if r['audit_risks'][n]!=original[r['image']]['audit_risks'][n]:raise ValueError('Risk changed')
        if analyze(data,cuts)!=gg:raise ValueError('Analysis replay differs')
        for group,part in rank.groups(data).items():
            v=gg[group]
            for c,d in v['cells'].items():
                chosen=[]
                for r in part:
                    if r['pred'] is None:key='missing'
                    else:
                        accepts=sum(r['audit_risks'][n]<=cuts[n] for n in METHODS)
                        key='all_accept' if accepts==5 else ('all_reject' if accepts==0 else 'disagreement')
                    if key==c:chosen.append(r)
                if counts(chosen)!={k:d[k] for k in counts(chosen)}:raise ValueError('Cell count differs')
                good=[r for r in chosen if rank.bad(r) is False];wrong=[r for r in chosen if rank.bad(r) is True]
                totals={n:0. for n in METHODS};dominance=dict(error_higher_all=0,error_lower_all=0,conflicting=0,identical=0)
                for e in wrong:
                    for g in good:
                        delta=[e['audit_risks'][n]-g['audit_risks'][n] for n in METHODS]
                        for n,z in zip(METHODS,delta):totals[n]+=(z>0)+.5*(z==0)
                        key='identical' if all(z==0 for z in delta) else ('error_higher_all' if all(z>=0 for z in delta) else ('error_lower_all' if all(z<=0 for z in delta) else 'conflicting'))
                        dominance[key]+=1
                denom=len(good)*len(wrong);pairs+=denom
                for n in METHODS:
                    expected=totals[n]/denom if denom else None
                    if expected!=d['AUROC'][n]:raise ValueError('Independent AUC differs')
                if dominance!=d['pair_dominance']:raise ValueError('Independent dominance differs')
                checks+=1
            simple={r['image'] for r in part if r['pred'] is not None and r['audit_risks']['formal']<=cuts['formal']}
            for n in METHODS:
                accepted={r['image'] for r in part if r['pred'] is not None and r['audit_risks'][n]<=cuts[n]}
                rank.independent_check(rank.Ranking(part,n),v['individual'][n]['delivery'],cuts[n])
                base=rank.summarize(part,simple)['states'];candidate=rank.summarize(part,accepted)['states'];change=v['individual'][n]['changes_vs_simple']
                if candidate['FA']-base['FA']!=change['released_FA']-change['new_ED'] or candidate['CR']-base['CR']!=change['rescued_CR']-change['new_FR']:raise ValueError('Change accounting differs')
    if report['TEST_opened'] or report['scorer_fits'] or report['policy_export']:raise ValueError('Unexpected task expansion')
    write(out/receipt,dict(status='INDEPENDENT_REVIEW_PASS',conditional_cells=checks,conditional_error_good_pairs=pairs,report_sha256=sha(out/'report.json'),source_sha256=sha(Path(__file__))))
    print('INDEPENDENT_REVIEW_PASS',checks,pairs,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--review',action='store_true');p.add_argument('--receipt',default='independent_review.json');a=p.parse_args()
    if a.review:review(a.out.resolve(),a.receipt)
    else:run(a.out.resolve())
