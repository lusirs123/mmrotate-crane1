"""Frozen-score low-FA audit; no model fitting, inference or policy export."""
import argparse,csv,gzip,hashlib,json,math,platform,subprocess
from pathlib import Path
from crane_project.utils.port_reliability_paired_ranking_audit_v1 import bad,groups,summarize,identity
ROOT=Path(__file__).resolve().parents[2]
MANIFEST=ROOT/'crane_project/tools/port_low_fa_v1_manifest.json'
METHODS=('simple','midpoint','score','temporal_fixed_gate')
TARGETS=[('CR',95,100),('CR',90,100),('CR',85,100),('ED',50,100),('ED',60,100),('ED',243,342)]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    with gzip.open(p,'rt') as f:return [json.loads(l) for l in f]
def allowed(r,m):return r['pred'] is not None and (m!='temporal_fixed_gate' or r['gate'])
def score(r,m):return r['scores']['simple' if m=='temporal_fixed_gate' else m]
def curve(rows,m):
    ordered=sorted((r for r in rows if allowed(r,m)),key=lambda r:score(r,m))
    good=sum(bad(r) is False for r in rows);wrong=sum(bad(r) is True for r in rows)
    pts=[dict(cut=None,FA=0,CR=0,FR=good,ED=wrong)];fa=cr=0
    for i,r in enumerate(ordered):
        fa+=int(bad(r));cr+=int(not bad(r))
        if i+1==len(ordered) or score(ordered[i+1],m)!=score(r,m):
            pts.append(dict(cut=score(r,m),FA=fa,CR=cr,FR=good-cr,ED=wrong-fa))
    return pts

def select(pts,kind,num,den,good,wrong):
    # Exact rational comparisons; entire risk ties are accepted together.
    if kind=='CR':
        valid=[p for p in pts if p['CR']*den>=good*num]
        return valid[0] if valid else None
    valid=[p for p in pts if p['ED']*den>=wrong*num]
    return valid[-1] if valid else None

def metrics(rows,m,cut):
    ids={r['image'] for r in rows if cut is not None and allowed(r,m) and score(r,m)<=cut}
    out=summarize(rows,ids)
    out['error_detection_rate']=out['states']['ED']/(out['states']['ED']+out['states']['FA']) if out['states']['ED']+out['states']['FA'] else None
    # Observation-index runs are separate from strict frame-id adjacency.
    run=longest=gap=longgap=0;prev=None
    for r in sorted(rows,key=lambda r:(r['sequence'],r['frame_id'],r['image'])):
        if r['sequence']!=prev:run=gap=0
        run=run+1 if bad(r) is False and r['image'] not in ids else 0
        gap=0 if bad(r) is False and r['image'] in ids else gap+1
        longest=max(longest,run);longgap=max(longgap,gap);prev=r['sequence']
    out.update(longest_FR_observations=longest,longest_without_CR_observations=longgap)
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
    manifest=json.loads(MANIFEST.read_text());out=ROOT/a.out
    if out.exists():raise ValueError('Refuse overwrite')
    for p,h in manifest['pins'].items():
        if sha(ROOT/p)!=h:raise ValueError('SHA mismatch '+p)
    allrows={};frozen={};records=[];curves={}
    for role,n in [('TRAIN',2558),('VAL',887),('TEST',1440)]:
        rows=read(ROOT/manifest['roles'][role]['source']);other=read(ROOT/manifest['roles'][role]['temporal'])
        assert len(rows)==len(other)==n
        ref={r['image']:r for r in other};assert len(ref)==n and len({r['image'] for r in rows})==n
        for r in rows:
            t=ref[r['image']]
            if role=='TEST':
                assert r['split']==t['split']=='test'
                r.setdefault('reliability_role','test')
            assert identity(r)==identity(t)
            assert r['reliability_role']==role.lower()
            risks=r['experiment_risks'];r['scores']=dict(simple=risks['full_simple'],midpoint=risks['midpoint'],score=risks['score_only'])
            point=t['temporal'];r['gate']=not point['available'] or point['primary']<=manifest['temporal_cutoff']
            if r['pred'] is not None:
                assert all(math.isfinite(v) and 0<=v<=1 for v in r['scores'].values())
                assert r['scores']['simple']==r['original_simple_decision']['risks']['size']
                assert r['scores']['score']==1-r['pred'][5]
            else:assert all(v is None for v in r['scores'].values())
            if 'flags' in t:assert t['flags']['temporal_veto']==(r['original_simple_decision']['size_accepted'] and r['gate'])
        allrows[role]=rows;curves[role]={};frozen[role]={}
        good=sum(bad(r) is False for r in rows);wrong=sum(bad(r) is True for r in rows)
        for m in METHODS:
            pts=curve(rows,m);curves[role][m]=pts
            cut=manifest['midpoint_cutoff'] if m=='midpoint' else manifest['simple_cutoff']
            if m!='score':frozen[role][m]=metrics(rows,m,cut)
            for kind,num,den in TARGETS:
                point=select(pts,kind,num,den,good,wrong);target='%s_%s_%s'%(kind,num,den)
                if point is None:
                    records.append(dict(role=role,method=m,target=target,group='all',status='UNREACHABLE',max_CR=pts[-1]['CR']));continue
                # Select on pooled split only; apply this SAME cutoff to every group.
                for group,part in groups(rows).items():
                    records.append(dict(role=role,method=m,target=target,group=group,status='OK',cut=point['cut'],metrics=metrics(part,m,point['cut'])))
    # Frozen original outputs must reproduce prior reports, not only input hashes.
    expected={'TRAIN':{'simple':(13,1037,58,1450),'midpoint':(1,0,70,2487)},'VAL':{'simple':(190,10,33,653),'midpoint':(206,3,17,660),'temporal_fixed_gate':(155,33,68,630)},'TEST':{'simple':(275,34,67,1055),'midpoint':(274,28,68,1061),'temporal_fixed_gate':(244,83,98,1006)}}
    for role,ms in expected.items():
        for m,e in ms.items():assert tuple(frozen[role][m]['states'][s] for s in ('FA','FR','ED','CR'))==e,(role,m)
    report=dict(protocol=manifest['protocol'],input_pins=manifest['pins'],frozen=frozen,records=records,curves=curves,scope=dict(training=0,inference=0,new_policy=False,TEST='existing_scores_descriptive_only'),runtime=dict(python=platform.python_version(),cwd=str(ROOT),HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT)).decode().strip()))
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    flat=[]
    for r in records:
        d={k:v for k,v in r.items() if k!='metrics'}
        if 'metrics' in r:
            d.update({k:v for k,v in r['metrics'].items() if k!='states'});d.update(r['metrics']['states'])
        flat.append(d)
    keys=sorted(set().union(*(d.keys() for d in flat)))
    with (out/'target_tables.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(flat)
    lines=['# 已有评分低FA工作区分析','',manifest['protocol'],'','所有目标点均为各数据集内GT辅助诊断，TEST门限不得用于部署。统一总体门限后分组；不按视频选门限。时序曲线固定旧硬拒绝，仅扫描simple。','', '|数据|方法|目标|FA|CR|正确保留|Real CR|最长连续FR观测|','|---|---|---|---:|---:|---:|---:|---:|']
    for r in records:
        if r['group']!='all':continue
        if r['status']!='OK':lines.append('|%s|%s|%s|不可达|||||'%(r['role'],r['method'],r['target']));continue
        v=r['metrics'];real=next(x['metrics'] for x in records if x['role']==r['role'] and x['method']==r['method'] and x['target']==r['target'] and x['group']=='domain:real')
        lines.append('|%s|%s|%s|%d|%d|%.2f%%|%d|%d|'%(r['role'],r['method'],r['target'],v['states']['FA'],v['states']['CR'],100*v['correct_retention'],real['states']['CR'],v['longest_FR_observations']))
    lines+=['','最长FR观测按同视频相邻保存观测计算；严格frame_id连续FR和无正确供给最长观测段另列CSV。均不换算秒。没有训练、容量选择、门限policy或新TEST推理。']
    (out/'analysis.md').write_text('\n'.join(lines)+'\n')
    (out/'completion.json').write_text(json.dumps(dict(artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file()},records=len(records),pins=len(manifest['pins'])),indent=2)+'\n')
    print(json.dumps(dict(output=str(out),records=len(records),report_sha=sha(out/'report.json'))))
if __name__=='__main__':main()
