#!/usr/bin/env python3
"""Frozen native/midpoint source TEST supplement. No fitting or thresholds from TEST."""
import argparse
import gzip
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.utils import port_reliability_feature_source_test_v1 as core
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils import port_midpoint_sigma15_reliability_v1 as binding
from crane_project.tools import run_port_reliability_feature_source_v1 as parent
from crane_project.tools import eval_port_reliability_readout_test_v1 as history
FIT='work_dirs/port_reliability_feature_source_v1/20261009_source_v1/result'
STEM='work_dirs/port_reliability_readout_test_v1/20261009_frozen_test_v1/result'
BASE,COLLECT,ROI,HEAD,POLICY=history.BASE,history.COLLECT,history.ROI,history.HEAD,history.POLICY
PROTOCOL=ROOT/'crane_project/tools/port_reliability_feature_source_test_v1_protocol.json'
SOURCES=ROOT/'crane_project/tools/port_reliability_feature_source_test_v1_sources.json'
# Filled from immutable local artifacts before committing; never computed from runtime input.
PINS = {'work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1/frozen_test_20261007_163625_1260566/completion.json': 'bc4f9ea6af189987a1a8b37a72ba3cb54763a039b916811e1cd0206729e1ee48', 'work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1/frozen_test_20261007_163625_1260566/summary.json': '152cf0b32eff0ed0117bb1a97ba23a35f4cbbd355d33758c109a6bfff97bbe94', 'work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1/frozen_test_20261007_163625_1260566/test_decisions.jsonl': '8d6755cc73f5e849b0dadc288bc0588878c9b6b835b7361406fe80f1610c6520', 'work_dirs/port_geometry_size_boundary_v1/size_candidate_downstream_20261007_191454_1852169/collect/completion.json': '9a308449444fd35a60e9595a4b73b1c7c1502ba2d6a39f1e8090389a9f45e661', 'work_dirs/port_geometry_size_boundary_v1/size_candidate_downstream_20261007_191454_1852169/collect/port_test/predictions.jsonl': 'f39d60b35f6628cd1142299cb55d81a33b53ece3fc8607bd6142379aba3efb93', 'work_dirs/port_geometry_size_boundary_v1/size_candidate_downstream_20261007_191454_1852169/collect/port_test/roi.pt': 'a0a1cb1f4245238b90f409f5e8a6ed5b306f8ae07dbfb06f1c8377891371003d', 'work_dirs/crane_symeood_k1_port_day2night_midpoint_sigma_v1/sigma_1p5/head_epoch_03.pth': '16c2fb448ac4e1c53530b8086d547f6f6ccb8d6b0763a42391c34f9b337982d7', 'work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1/fit/policy.json': '38d914f114fcdb257f33f1cd6390ebcb4e103adfa1d9022628875e137c916e1f', 'work_dirs/port_reliability_feature_source_v1/20261009_source_v1/result/models.json': '338b7d514135a1e66dc371829ac5b2c8c060fd7ae0347c35174266d2de5b2851', 'work_dirs/port_reliability_feature_source_v1/20261009_source_v1/result/pca.json': '2c7c690e29828b3599acf468f54e6da231e9a88d2297e7ffd0083bb67daa539d', 'work_dirs/port_reliability_feature_source_v1/20261009_source_v1/result/cutoffs.json': '8a500768076d50b29282fcbf5a6012708365d46273307117a74048daf2e4acab', 'work_dirs/port_reliability_feature_source_v1/20261009_source_v1/result/report.json': '9231b9344c3aa7665479f2ae2e0fcbd4df3a3678815c139fbae6d42e966c4203', 'work_dirs/port_reliability_feature_source_v1/20261009_source_v1/result/completion.json': 'd0e8117b1845b2a626a73758be81fffe1fec8f589de0856ab9ad4e7864648f5c', 'work_dirs/port_reliability_readout_test_v1/20261009_frozen_test_v1/result/report.json': '1a432a20042d94f4439e19338d10519d62128a8b310647750aa311fb5072b2bd', 'work_dirs/port_reliability_readout_test_v1/20261009_frozen_test_v1/result/completion.json': '8c3f123896579837f2cab57773074e5be1791aa23d8b119a229d0d1d356178de', 'work_dirs/port_reliability_readout_test_v1/20261009_frozen_test_v1/result/test_features.npz': 'e2d09791d97939b962f5740d8200dfd168fb7da0946396f529b07b03f8c4667f', 'work_dirs/crane_symeood_k1_port_day2night_aug_b_v1/epoch_24.pth': '8f8008c4944807a65ed0f2ee0cc348ea78690d54a4176944b2c9b0ebc83cec23'}
SOURCE_FILES=list(dict.fromkeys(parent.SOURCE_FILES+[str(parent.SOURCES.relative_to(ROOT)),
    'crane_project/utils/port_reliability_readout_test_v1.py',
    'crane_project/tools/eval_port_reliability_readout_test_v1.py',
    'crane_project/utils/port_reliability_feature_source_test_v1.py',
    'crane_project/utils/port_reliability_feature_source_test_v1_torch.py',
    'crane_project/tools/eval_port_reliability_feature_source_test_v1.py',
    'crane_project/tools/review_port_reliability_feature_source_test_v1.py',
    'crane_project/tools/port_reliability_feature_source_test_v1_protocol.json',
    'tests/test_port_reliability_feature_source_test_v1.py',
    'tools/run_port_reliability_feature_source_test_v1.sh']))
sha,write,read_rows=parent.sha,parent.write,history.read_rows

def checked_sources():
    parent.checked_sources()
    protocol=json.loads(PROTOCOL.read_text());actual={n:sha(ROOT/n) for n in SOURCE_FILES}
    if (json.loads(SOURCES.read_text())!=dict(protocol=core.VERSION,sources=actual)
            or protocol['protocol']!=core.VERSION or protocol['input_pins']!=PINS
            or protocol['counts']!=core.COUNTS or protocol['automatic_promotion']
            or protocol['training'] or protocol['threshold_updates']!=0):
        raise ValueError('Frozen TEST source/contract differs')
    return protocol,dict(manifest_sha256=sha(SOURCES),sources=actual)

def checked_inputs(require_tensors):
    missing=[];actual={}
    for name,pin in PINS.items():
        path=ROOT/name
        if not path.exists():
            if not require_tensors and name in (ROI,HEAD,parent.B_PATH):missing.append(name);continue
            raise FileNotFoundError('Required frozen artifact unavailable: '+name)
        actual[name]=sha(path)
        if actual[name]!=pin:raise ValueError('Frozen input SHA changed: '+name)
    model=json.loads((ROOT/FIT/'models.json').read_text())
    cutoffs=json.loads((ROOT/FIT/'cutoffs.json').read_text())
    report=json.loads((ROOT/FIT/'report.json').read_text())
    model,cutoffs=core.frozen_bundle(model,cutoffs,report)
    complete=json.loads((ROOT/FIT/'completion.json').read_text())
    if complete['status']!=report['status'] or complete['TEST_read']:
        raise ValueError('Original failed VAL history changed')
    pcas=json.loads((ROOT/FIT/'pca.json').read_text())
    if set(pcas)!=set(core.scoring.ARMS) or any(p['train_count']!=2558 or p['dimensions']!=256 or p['whiten'] for p in pcas.values()):
        raise ValueError('Only frozen full TRAIN PCA permitted')
    policy=json.loads((ROOT/POLICY).read_text());binding.checked_frontend(policy['front_end'])
    baseline=json.loads((ROOT/BASE/'summary.json').read_text())
    if baseline['policy_sha256']!=PINS[POLICY] or baseline['test_used_for_selection']:
        raise ValueError('Formal baseline differs')
    old=json.loads((ROOT/STEM/'report.json').read_text())
    receipt=json.loads((ROOT/STEM/'completion.json').read_text())
    if (old['input_sha256'][HEAD]!=PINS[HEAD] or old['input_sha256'][ROI]!=PINS[ROI]
            or not old['boxes_scores_center_angle_output_unchanged'] or old['TEST_used_for_selection']
            or receipt['artifacts']['test_features.npz']!=PINS[STEM+'/test_features.npz']):
        raise ValueError('Frozen stem feature provenance differs')
    return model,cutoffs,pcas,policy,dict(available_sha256=actual,missing_required_tensors=missing,
        ready_for_evaluation=not missing,parent_VAL_status=report['status'],TEST_evaluated=False)

def table(report):
    lines=['# 原生／midpoint特征的冻结TEST补评','',
        '完整TRAIN最终权重、PCA与VAL门限固定；原VAL失败保留。TEST已暴露，不用于调整或采用。','',
        '| 分组／方法 | 接受数 | FA | FR | ED | CR | 接受覆盖 | 正确保留 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for group,methods in report['fixed_VAL_cutoff_summary'].items():
        for name,v in dict(methods,original_simple=report['original_policy_summary'][group]).items():
            s=v['states'];lines.append('| %s／%s | %d | %d | %d | %d | %d | %.2f%% | %.2f%% |'%(group,name,
                s['FA']+s['CR'],s['FA'],s['FR'],s['ED'],s['CR'],100*v['acceptance_coverage_all_frames'],100*v['good_retention']))
    lines+=['','| 分组／方法（同候选接受数） | FA | FR | 同候选正确保留的FA |','|---|---:|---:|---:|']
    for group,p in report['statistics']['native'].items():
        lines.append('| %s／native | %d | %d | %d |'%(group,p['actual']['states']['FA'],p['actual']['states']['FR'],p['actual']['states']['FA']))
        for m,v in p['same_count_controls'].items():
            lines.append('| %s／%s | %d | %d | %d |'%(group,m,v['states']['FA'],v['states']['FR'],p['matched_CR_controls'][m]['states']['FA']))
    return '\n'.join(lines)+'\n'

def run(args):
    args._created_output=False
    protocol,source=checked_sources()
    if args.out.resolve().parent.parent!=(ROOT/'work_dirs'/core.VERSION).resolve():raise ValueError('Use unified experiment/RUN_ID/result')
    if args.out.exists():raise FileExistsError('Refuse overwrite')
    model,cutoffs,pcas,policy,proof=checked_inputs(args.mode=='evaluate')
    args.out.mkdir(parents=True);args._created_output=True
    write(args.out/'protocol.json',protocol);write(args.out/'sources.json',source);write(args.out/'input_check.json',proof)
    if args.mode=='check':
        write(args.out/'completion.json',dict(status='STATIC_READY' if proof['ready_for_evaluation'] else 'STATIC_OK_TENSORS_REQUIRED',TEST_evaluated=False,training=False))
        print('STATIC_CHECK_OK',proof['missing_required_tensors'],flush=True);return
    import torch
    from crane_project.utils.port_reliability_feature_source_test_v1_torch import collect
    torch.set_num_threads(1);torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    rows=core.bind_rows(read_rows(ROOT/BASE/'test_decisions.jsonl'),read_rows(ROOT/COLLECT/'port_test/predictions.jsonl'))
    before=simple.fingerprint([rows,model,cutoffs,pcas,policy])
    api=binding.Sigma15Reliability(policy,policy['front_end'])
    for r in rows:
        if api.decide(r['pred'],r['image_size'])!=r['original_simple_decision']:raise ValueError('Formal flags no longer replay')
    stem=np.load(ROOT/STEM/'test_features.npz',allow_pickle=False);ids=stem['images'].tolist();x=stem['features']
    if ids!=[r['image'] for r in rows if r['pred'] is not None] or x.shape!=(1431,291):raise ValueError('Original stem pairing differs')
    descriptors=np.asarray([simple.descriptor(r['pred'],r['image_size']) for r in rows if r['pred'] is not None])
    np.testing.assert_allclose(x[:,:3],descriptors,atol=1e-12,rtol=0)
    raw,nids,feature_proof=collect(rows,args.gpu,args.out)
    if nids!=ids:raise ValueError('Native/stem frame pairing differs')
    matrices={a:core.scoring.features(descriptors,v,pcas[a]) for a,v in dict(midpoint=x[:,3:],native=raw).items()}
    for a,v in matrices.items():np.savez_compressed(args.out/('test_'+a+'_projected.npz'),features=v,images=np.asarray(ids))
    risks,decisions=core.score_rows(rows,matrices,ids,model,cutoffs)
    with gzip.open(args.out/'scored_TEST.jsonl.gz','xt') as f:
        for r,d in zip(rows,decisions):f.write(json.dumps(dict(r,experiment_risks={m:v[r['image']] for m,v in risks.items()},candidate_decisions=d['methods']),allow_nan=False)+'\n')
    result=core.summarize(rows,risks,cutoffs)
    if (before!=simple.fingerprint([rows,model,cutoffs,pcas,policy]) or checked_sources()[1]!=source
            or any(sha(ROOT/n)!=pin for n,pin in PINS.items())):raise ValueError('Frozen inputs/sources changed')
    report=dict(result,protocol=core.VERSION,parent_VAL_status='VAL_FAILED_STOP',VAL_failure_retracted=False,
        frozen_cutoffs=cutoffs,input_sha256=PINS,sources=source,feature_proof=feature_proof,
        inference_model=FIT+'/models.json',TEST_repeatedly_exposed=True,TEST_used_for_selection=False,
        detector_updates=0,head_updates=0,quality_updates=0,threshold_updates=0,GT_online=False,
        boxes_scores_center_angle_output_unchanged=True,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT),text=True).strip())
    write(args.out/'report.json',report);(args.out/'summary.md').write_text(table(report))
    write(args.out/'completion.json',dict(protocol=core.VERSION,status=report['status'],TEST_evaluated=True,parent_VAL_status='VAL_FAILED_STOP',automatic_promotion=False,
        artifacts={p.name:sha(p) for p in args.out.iterdir() if p.is_file()}))
    print(report['status'],flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('check','evaluate'),required=True)
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    try:run(args)
    except Exception as e:
        if getattr(args,'_created_output',False):write(args.out/'failure.json',dict(error=repr(e),status='ENGINEERING_FAILED'))
        raise
if __name__=='__main__':main()
