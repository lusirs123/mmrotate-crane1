"""Train six cached-feature reliability heads; frozen detector never loaded."""
import argparse,gzip,hashlib,json,os,platform,subprocess
from pathlib import Path
import numpy as np
import torch
from crane_project.utils import port_reliability_capacity_v1 as core
from crane_project.utils import port_reliability_redc_size_v1_torch as training
from crane_project.utils.port_reliability_paired_ranking_audit_v1 import identity,bad
ROOT=Path(__file__).resolve().parents[2]
MANIFEST=ROOT/'crane_project/tools/port_reliability_capacity_v1_manifest.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def readrows(p):
    with gzip.open(p,'rt') as f:return [json.loads(l) for l in f]
def checkpins(pins):
    for p,h in pins.items():
        if sha(ROOT/p)!=h:raise ValueError('Input/source SHA differs '+p)
def load(role,manifest):
    cfg=manifest['roles'][role];rows=readrows(ROOT/cfg['rows'])
    if len(rows)!={'TRAIN':2558,'VAL':887,'TEST':1440}[role]:raise ValueError('Incomplete role')
    ids=[r['image'] for r in rows]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate rows')
    for r in rows:
        if role=='TEST':
            if r['split']!='test':raise ValueError('Wrong TEST identity')
            r.setdefault('reliability_role','test')
        if r['reliability_role']!=role.lower():raise ValueError('Wrong role')
        if r['original_simple_decision']['final_box_original']!=r['pred']:raise ValueError('Box identity')
        r['audit_risks']={'simple':r['experiment_risks']['full_simple'],'historical_midpoint':r['experiment_risks']['midpoint'],'score':r['experiment_risks']['score_only']}
    with np.load(ROOT/cfg['features'],allow_pickle=False) as f:matrix,images=f['features'].copy(),f['images'].copy()
    return rows,matrix,images

def run(out):
    if out.exists():raise ValueError('Refuse overwrite')
    if Path.cwd().resolve()!=ROOT:raise ValueError('Wrong working directory')
    manifest=json.loads(MANIFEST.read_text());checkpins(manifest['development_pins'])
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    original=json.loads((ROOT/manifest['original_models']).read_text());normalizer=original['normalizers']['midpoint']
    train,tm,ti=load('TRAIN',manifest);val,vm,vi=load('VAL',manifest)
    tx,tz,ty=core.prepare(train,tm,ti,normalizer)
    # Recompute only TRAIN normalizer to verify provenance, never refit PCA.
    from crane_project.utils import port_reliability_feature_source_v1 as feature
    if feature.normalization(tm)!=normalizer:raise ValueError('Frozen TRAIN normalizer mismatch')
    core.prepare(val,vm,vi,normalizer)
    assert int(ty.sum())==71 and sum(bad(r) is True for r in val)==223
    out.mkdir(parents=True)
    runtime=dict(cwd=str(ROOT),environment=os.environ.get('CONDA_DEFAULT_ENV'),python=platform.python_version(),torch=torch.__version__,device='cpu',HEAD=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip())
    write(out/'runtime.json',runtime);write(out/'protocol.json',manifest)
    baseline=core.at_cut(val,'simple',manifest['simple_cutoff'])
    assert tuple(baseline['all']['states'][k] for k in ('FA','FR','ED','CR'))==(190,10,33,653)
    results={};exported={};cuts={};batchhash={};replay={}
    with (out/'train_log.jsonl').open('x') as logs:
        for seed in core.SEEDS:
            models,records,order=core.fit(tx,tz,ty,seed)
            batchhash[str(seed)]=sha_bytes=hashlib.sha256(order).hexdigest();print('batch order',seed,sha_bytes,flush=True)
            for r in records:logs.write(json.dumps(r,allow_nan=False)+'\n')
            results[str(seed)]={};exported[str(seed)]={};cuts[str(seed)]={}
            for arm,model in models.items():
                name=arm+'_'+str(seed);model.eval();weights=training.exported(model);exported[str(seed)][arm]=weights
                core.score(train,tm,ti,normalizer,weights,name);core.score(val,vm,vi,normalizer,weights,name)
                with torch.no_grad():t=torch.sigmoid(model(torch.tensor(tx,dtype=torch.float32),torch.tensor(tz,dtype=torch.float32))).numpy()
                train_by_id={r['image']:r for r in train}
                independent=np.asarray([train_by_id[str(i)]['audit_risks'][name] for i in ti])
                delta=float(np.max(np.abs(t-independent)));assert delta<1e-4
                replay[name]=delta
                stats=core.points(val,name,653);cut=stats['primary_point']['risk_le'];cuts[str(seed)][arm]=cut
                results[str(seed)][arm]=dict(VAL=stats,TRAIN=core.points(train,name,1450),TRAIN_at_VAL_cut=core.at_cut(train,name,cut),parameter_count=sum(p.numel() for p in model.parameters()))
                expected=4290 if arm=='small' else 8834;assert results[str(seed)][arm]['parameter_count']==expected
            torch.save({a:m.state_dict() for a,m in models.items()},out/('seed_'+str(seed)+'.pth'))
            loaded=torch.load(out/('seed_'+str(seed)+'.pth'),map_location='cpu')
            assert all(torch.equal(v,loaded[a][k]) for a,m in models.items() for k,v in m.state_dict().items())
    decision=core.gate(results,baseline)
    # Freeze decision and deployment cutoff BEFORE any TEST file is read.
    write(out/'freeze.json',dict(decision=decision,cutoffs=cuts,models_sha256={str(seed):sha(out/('seed_'+str(seed)+'.pth')) for seed in core.SEEDS},protocol_sha256=sha(MANIFEST)))
    test=None
    if decision['passed']:
        checkpins(manifest['test_pins'])
        rows,matrix,images=load('TEST',manifest);core.prepare(rows,matrix,images,normalizer)
        test={'simple':core.at_cut(rows,'simple',manifest['simple_cutoff'])}
        for arm in ('small','large'):
            name=arm+'_1701';core.score(rows,matrix,images,normalizer,exported['1701'][arm],name)
            test[arm]=dict(frozen=core.at_cut(rows,name,cuts['1701'][arm]),offline_same_CR1055=core.points(rows,name,1055))
        with gzip.open(out/'scored_TEST.jsonl.gz','xt') as f:
            for r in rows:f.write(json.dumps(r)+'\n')
    for role,rows in [('TRAIN',train),('VAL',val)]:
        with gzip.open(out/('scored_'+role+'.jsonl.gz'),'xt') as f:
            for r in rows:f.write(json.dumps(r,allow_nan=False)+'\n')
    # Compare the original1701 saved score, disclose CPU/GPU numerical changes.
    replay['historical_small1701_max_abs_TRAIN']=max(abs(r['audit_risks']['small_1701']-r['audit_risks']['historical_midpoint']) for r in train)
    replay['historical_small1701_max_abs_VAL']=max(abs(r['audit_risks']['small_1701']-r['audit_risks']['historical_midpoint']) for r in val if r['pred'] is not None)
    write(out/'models.json',dict(normalizer=normalizer,weights=exported))
    checkpins(manifest['development_pins'])
    report=dict(status='VAL_PASS_TEST_REPORTED' if decision['passed'] else 'VAL_FAILED_STOP',baseline=baseline,results=results,decision=decision,TEST=test,TEST_read=decision['passed'],baseline_ranking=core.points(val,'simple',653),score_ranking=core.points(val,'score',653),training_updates=6000,detector_inference=0,feature_extraction=0,batch_order_sha256=batchhash,replay=replay,runtime=runtime,protocol_sha256=sha(MANIFEST))
    write(out/'report.json',report)
    lines=['# 同输入小幅容量对照','',report['status'],'','|seed|模型|VAL FA|FR|ED|CR|Real CR|Sim CR|最长FR观测|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for seed in core.SEEDS:
        for arm in ('small','large'):
            v=results[str(seed)][arm]['VAL']['primary'];s=v['all']['states']
            lines.append('|%d|%s|%d|%d|%d|%d|%d|%d|%d|'%(seed,arm,s['FA'],s['FR'],s['ED'],s['CR'],v['domain:real']['states']['CR'],v['domain:sim']['states']['CR'],v['all']['longest_FR_observations']))
    lines+=['','正式simple：FA190/FR10/ED33/CR653。3种子均须同CR改善小模型和simple，且Real CR不低于正式simple。固定代表种子1701；未选择最好种子。','',json.dumps(decision,ensure_ascii=False),'','TEST读取：'+str(report['TEST_read'])+'。失败即停止，不自动扩容/换损失/加种子。所有曲线和分域逐视频指标见report.json。']
    (out/'analysis.md').write_text('\n'.join(lines)+'\n')
    write(out/'completion.json',dict(status=report['status'],artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(report['status'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();run(a.out.resolve())
