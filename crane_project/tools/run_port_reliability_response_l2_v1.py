"""Two L2=0 GPU fits; reuse paired L2=.1, freeze VAL cuts, report exposed TEST."""
import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
import torch
from crane_project.tools import eval_port_reliability_response_warm_test_v1 as saved
from crane_project.tools import run_port_reliability_response_warm_v1 as warm
from crane_project.tools import review_port_reliability_response_warm_v1 as audit
from crane_project.utils import port_reliability_response_l2_v1 as core
from crane_project.utils import port_reliability_response_stats_v1 as response
from crane_project.utils.port_reliability_paired_ranking_audit_v1 import Ranking, groups, bad, independent_check
ROOT=warm.ROOT
SOURCE=saved.SOURCE
TEST_SOURCE=ROOT/'work_dirs/port_reliability_response_warm_test_v1/20261015_frozen1701_test_v1_fix1/result'
MANIFEST=ROOT/'crane_project/tools/port_reliability_response_l2_v1_manifest.json'
METHODS=('formal','score','parent_1701','bce_1701','soft_1701','bce0_1701','soft0_1701')
NEW=('bce0_1701','soft0_1701')

def read_rows(path):
    with gzip.open(path,'rt') as f:return [json.loads(s) for s in f]

def role_data(role,out,protocol):
    source=TEST_SOURCE if role=='TEST' else SOURCE
    hashes=protocol['saved_artifacts'][role]
    warm.pins({str(source/n):h for n,h in hashes.items()})
    rows=read_rows(source/('scored_'+role+'.jsonl.gz'))
    file=role.lower()+'_responses.npz';shutil.copyfile(source/file,out/file)
    with np.load(out/file,allow_pickle=False) as f:raw=f['features'];ids=f['images'].tolist()
    outputs={r['image']:r for r in rows if r['pred'] is not None}
    if set(outputs)!=set(ids) or len(outputs)!=len(ids):raise ValueError('Unpaired saved features')
    return rows,raw,ids,np.array([bad(outputs[i]) for i in ids],dtype=float)

def decision(results):
    value={}
    for name in NEW:
        m=results[name];d=m['at_frozen_VAL_cut'];s=d['all']['states']
        paired=results[name.replace('0_','_')]['at_frozen_VAL_cut']['all']['states']
        checks=dict(protect_CR=s['CR']>=653,protect_Real=d['domain:real']['states']['CR']>=189,
            gain_at_least3_vs_formal=s['FA']<=187,gain_vs_L2_control=s['FA']<paired['FA'],
            same_count_better_than_formal=m['same_count_as_formal']['tie_bounds']['FA_max']<190)
        value[name]=dict(passed=all(checks.values()),checks=checks)
    return dict(VAL_only=value,automatic_promotion=False,TEST_for_reporting_regardless_of_VAL=True)

def run(out):
    if out.exists() or out.parent.parent!=ROOT/'work_dirs'/core.VERSION or Path.cwd().resolve()!=ROOT:raise ValueError('New correct result path required')
    protocol=json.loads(MANIFEST.read_text());warm.pins(protocol['source_pins'])
    saved.source_check()
    if protocol['settings']!=core.SETTINGS:raise ValueError('Settings mismatch')
    out.mkdir(parents=True);warm.write(out/'protocol.json',protocol)
    warm.write(out/'runtime.json',dict(cwd=str(ROOT),environment=os.environ.get('CONDA_DEFAULT_ENV'),
        torch=torch.__version__,GPU=torch.cuda.get_device_name(0),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),HEAD=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    oldmodels=json.loads((SOURCE/'models.json').read_text())
    models={k:oldmodels[k] for k in METHODS[2:5]};parent=models['parent_1701']
    tau=json.loads((SOURCE/'temperature_frozen.json').read_text())['temperature']
    train,raw,ids,y=role_data('TRAIN',out,protocol)
    if raw.shape!=(2558,19) or int(y.sum())!=71:raise ValueError('TRAIN material changed')
    histories={}
    for arm,name in zip(core.ARMS,NEW):
        obj,model,records=core.fit(raw,y,parent,arm,tau,'cuda:0')
        models[name]=model;histories[name]=records
        print('Final fit',name,model['objective_final'],records[-1]['train_points'],flush=True)
    with gzip.open(out/'train_history.json.gz','xt') as f:json.dump(histories,f,allow_nan=False)
    warm.write(out/'models.json',models)
    warm.write(out/'TRAIN_frozen.json',dict(models_sha256=warm.sha(out/'models.json'),all_fits_complete_before_VAL=True,
        steps=5000,seed=1701,scorer_fits=2,TEST_read=False))
    val,vraw,vids,_=role_data('VAL',out,protocol)
    policy=json.loads((ROOT/warm.previous.history.POLICY).read_text())
    for rows,x,images in [(train,raw,ids),(val,vraw,vids)]:warm.previous.add_scores(rows,x,images,models,policy)
    oldcuts=json.loads((SOURCE/'freeze.json').read_text())['cutoffs']
    cuts={name:oldcuts[name] for name in METHODS if name not in NEW}
    for name in NEW:cuts[name]=warm.core.global_cut(val,name)
    results={role:warm.metrics(rows,METHODS,cuts) for role,rows in [('TRAIN',train),('VAL',val)]}
    choice=decision(results['VAL'])
    warm.write(out/'freeze.json',dict(cutoffs=cuts,decision=choice,models_sha256=warm.sha(out/'models.json'),TEST_read=False,seed=1701))
    test,traw,tids,_=role_data('TEST',out,protocol)
    warm.previous.add_scores(test,traw,tids,models,policy)
    results['TEST']=warm.metrics(test,METHODS,cuts)
    for role,rows in [('TRAIN',train),('VAL',val),('TEST',test)]:
        with gzip.open(out/('scored_'+role+'.jsonl.gz'),'xt') as f:
            for row in rows:
                row['response_decisions']={name:response.size_decision(row['original_simple_decision'],row['audit_risks'][name],cuts[name]) for name in models}
                f.write(json.dumps(row,allow_nan=False)+'\n')
    report=dict(status='L2_CONTROL_COMPLETE_FROZEN_TEST_REPORTED',results=results,cutoffs=cuts,decision=choice,
        scorer_fits=2,seed=1701,TEST_used_for_selection=False,TEST_repeatedly_exposed=True,
        detector_updates=0,midpoint_updates=0,GT_online=False,original_policy_changed=False)
    warm.dump_report(out,report)
    # Replace inherited experiment prose with this experiment's actual scope.
    p=out/'analysis.md';s=p.read_text().split('\nparent=')[0]
    p.write_text(s+'\n\nbce0/soft0为本轮L2=0两次GPU训练；bce/soft为已有L2=.1同设置对照。固定1701，19→8→1，共同BCE父模型与原温度，5000步最终模型。\nTRAIN每250步权重/评分保留；VAL统一门限保护CR653、RealCR189；TEST只报告不选择。无新检测推理，无policy替换。\n'+json.dumps(choice,ensure_ascii=False)+'\n')
    warm.pins(protocol['source_pins'])
    warm.write(out/'completion.json',dict(artifacts={p.name:warm.sha(p) for p in out.iterdir() if p.is_file()}))
    print(report['status'],flush=True)

def review(out,receipt,offline=False):
    manifest,_=saved.source_check(offline)
    protocol=json.loads((out/'protocol.json').read_text());warm.pins(protocol['source_pins'])
    warm.pins({str(out/n):h for n,h in json.loads((out/'completion.json').read_text())['artifacts'].items()})
    report=json.loads((out/'report.json').read_text());models=json.loads((out/'models.json').read_text());cuts=report['cutoffs']
    frozen=json.loads((out/'freeze.json').read_text());trainf=json.loads((out/'TRAIN_frozen.json').read_text())
    if frozen['models_sha256']!=warm.sha(out/'models.json') or trainf['models_sha256']!=frozen['models_sha256'] or cuts!=frozen['cutoffs'] or report['decision']!=decision(report['results']['VAL']):raise ValueError('Freeze mismatch')
    old=json.loads((SOURCE/'models.json').read_text());oldcuts=json.loads((SOURCE/'freeze.json').read_text())['cutoffs']
    for name in METHODS[2:5]:
        if models[name]!=old[name] or cuts[name]!=oldcuts[name]:raise ValueError('Reused control changed')
    checks=[];err=0.;staterr=0.;objectives={};trainraw=None
    for role,values in report['results'].items():
        rows=read_rows(out/('scored_'+role+'.jsonl.gz'));original={r['image']:r for r in warm.load_rows(role,manifest)}
        if len(rows)!=len(original) or set(r['image'] for r in rows)!=set(original):raise ValueError('Frame membership mismatch')
        for row in rows:
            if saved.stable_identity(row,role)!=saved.stable_identity(original[row['image']],role):raise ValueError('Identity changed')
            for name,d in row['response_decisions'].items():
                if d!=response.size_decision(row['original_simple_decision'],row['audit_risks'][name],cuts[name]):raise ValueError('Flags changed')
        file=role.lower()+'_responses.npz'
        if warm.sha(out/file)!=protocol['saved_artifacts'][role][file]:raise ValueError('Raw cache changed')
        with np.load(out/file) as f:raw=f['features'];ids=f['images'].tolist();p=f['probability'];q=f['prior']
        lookup={r['image']:r for r in rows if r['pred'] is not None}
        if len(ids)!=len(lookup) or set(ids)!=set(lookup):raise ValueError('Feature membership mismatch')
        y=np.array([bad(lookup[i]) for i in ids],dtype=float)
        if role=='TRAIN':trainraw=raw;trainy=y
        for i,key in enumerate(ids):
            expected=np.r_[warm.simple.descriptor(lookup[key]['pred'],lookup[key]['image_size']),[a-b for pm,qm in zip(p[i],q[i]) for a,b in zip(audit.scalar_statistics(pm),audit.scalar_statistics(qm))]]
            staterr=max(staterr,float(np.max(np.abs(expected-raw[i]))))
        for name,model in models.items():
            old_l2=warm.core.SETTINGS['l2']
            try:
                if name in NEW:warm.core.SETTINGS['l2']=0.
                risk,proof=audit.independent_model(model,raw,y if role=='TRAIN' and name in NEW else None)
            finally:warm.core.SETTINGS['l2']=old_l2
            if proof:objectives[name]=proof
            err=max(err,max(abs(v-lookup[k]['audit_risks'][name]) for k,v in zip(ids,risk)))
        if warm.metrics(rows,METHODS,cuts)!=values:raise ValueError('Metrics differ')
        for name,v in values.items():
            for g,part in groups(rows).items():checks.append(independent_check(Ranking(part,name),v['at_frozen_VAL_cut'][g],cuts[name]))
        if role=='VAL':
            for name in NEW:
                good=sorted(r['audit_risks'][name] for r in rows if bad(r) is False)
                real=sorted(r['audit_risks'][name] for r in rows if bad(r) is False and r['domain']=='real')
                if cuts[name]!=max(good[652],real[188]):raise ValueError('VAL quantile differs')
    with gzip.open(out/'train_history.json.gz','rt') as f:histories=json.load(f)
    history_checks=0
    for name,records in histories.items():
        if [r['step'] for r in records]!=list(range(0,5001,250)):raise ValueError('Snapshot steps differ')
        if records[0]['weights']!=models['parent_1701']['weights'] or records[-1]['weights']!=models[name]['weights']:raise ValueError('Snapshot endpoints changed')
        if models[name]['initial_state_sha256']!=models['parent_1701']['final_state_sha256'] or models[name]['l2']!=0:raise ValueError('Start/l2 changed')
        for r in records:
            m=dict(models[name],weights=r['weights']);z=response.numpy_logits(m,trainraw)
            np.testing.assert_allclose(z,r['train_logits'],rtol=0,atol=1e-10)
            for rate,point in r['train_points'].items():
                k=int(np.ceil(float(rate)*sum(trainy==0)));boundary=sorted(z[trainy==0])[k-1]
                if (point['FA'],point['CR'])!=(int(sum(z[trainy==1]<=boundary)),int(sum(z[trainy==0]<=boundary))):raise ValueError('Snapshot ordering differs')
                history_checks+=1
    if err>1e-11 or staterr>1e-12:raise ValueError('Independent feature/model failure')
    omitted=[p for p in {**manifest['development_pins'],**manifest['val_pins']} if not (ROOT/p).exists()] if offline else []
    if omitted:
        server=json.loads((out/'independent_review_server.json').read_text())
        if server['report_sha256']!=warm.sha(out/'report.json') or server['status']!='INDEPENDENT_REVIEW_PASS':raise ValueError('Full server review required')
    warm.write(out/receipt,dict(status='INDEPENDENT_REVIEW_PASS',groups=len(checks),pairs=sum(c['pair_comparisons'] for c in checks),curve_points=sum(c['curve_points'] for c in checks),
        snapshot_rank_checks=history_checks,model_risk_max_abs=err,scalar_statistics_max_abs=staterr,objectives=objectives,
        omitted_local_runtime_inputs_verified_on_server=omitted,report_sha256=warm.sha(out/'report.json')))
    print('INDEPENDENT_REVIEW_PASS',len(checks),history_checks,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--review',action='store_true');p.add_argument('--offline-review',action='store_true');p.add_argument('--receipt',default='independent_review.json');a=p.parse_args()
    if a.review:review(a.out.resolve(),a.receipt,a.offline_review)
    else:run(a.out.resolve())
