#!/usr/bin/env python3
"""Frozen diagnostic after failed VAL; never trains or calibrates on TEST."""
import argparse
import gzip
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.utils import port_reliability_readout_test_v1 as core
from crane_project.utils import port_reliability_readout_compare_v1 as scoring
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils import port_midpoint_sigma15_reliability_v1 as binding
from crane_project.tools import run_port_reliability_readout_compare_v1 as parent

FIT='work_dirs/port_reliability_readout_compare_v1/20261008_readout_v1/result'
BASE='work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1/frozen_test_20261007_163625_1260566'
COLLECT='work_dirs/port_geometry_size_boundary_v1/size_candidate_downstream_20261007_191454_1852169/collect'
ROI=COLLECT+'/port_test/roi.pt'
HEAD=parent.prior.HEAD
POLICY=parent.prior.POLICY
PROTOCOL=ROOT/'crane_project/tools/port_reliability_readout_test_v1_protocol.json'
SOURCES=ROOT/'crane_project/tools/port_reliability_readout_test_v1_sources.json'
PINS={FIT+'/'+k:v for k,v in {
    'completion.json':'fcdd32e756a7d9292b28c5cc4f2c3e0756f69c71fb75ab42541a2605026cac58',
    'report.json':'382f5c94ea636c1ca1a4ea6c0f63a57122f950f6be03de10f3e123e6a510a673',
    'models.json':'b49cc7c69ba8285cfc6f137c56b992bdc157410c070bbdbe0e7eb736922f41aa',
    'cutoffs.json':'208131d7afc1565e752a9e54c043accebf732d98338b2af9cb90f2401cdd8ef0'}.items()}
PINS.update({BASE+'/completion.json':'bc4f9ea6af189987a1a8b37a72ba3cb54763a039b916811e1cd0206729e1ee48',
    BASE+'/summary.json':'152cf0b32eff0ed0117bb1a97ba23a35f4cbbd355d33758c109a6bfff97bbe94',
    BASE+'/test_decisions.jsonl':'8d6755cc73f5e849b0dadc288bc0588878c9b6b835b7361406fe80f1610c6520',
    COLLECT+'/completion.json':'9a308449444fd35a60e9595a4b73b1c7c1502ba2d6a39f1e8090389a9f45e661',
    COLLECT+'/port_test/predictions.jsonl':'f39d60b35f6628cd1142299cb55d81a33b53ece3fc8607bd6142379aba3efb93',
    ROI:'a0a1cb1f4245238b90f409f5e8a6ed5b306f8ae07dbfb06f1c8377891371003d',
    HEAD:binding.HEAD_SHA,POLICY:'38d914f114fcdb257f33f1cd6390ebcb4e103adfa1d9022628875e137c916e1f'})
SOURCE_FILES=list(dict.fromkeys(parent.SOURCE_FILES+[
    str(parent.SOURCES.relative_to(ROOT)),
    'crane_project/utils/port_reliability_readout_test_v1.py',
    'crane_project/utils/port_reliability_readout_test_v1_torch.py',
    'crane_project/tools/eval_port_reliability_readout_test_v1.py',
    'crane_project/tools/review_port_reliability_readout_test_v1.py',
    'crane_project/tools/port_reliability_readout_test_v1_protocol.json',
    'tests/test_port_reliability_readout_test_v1.py',
    'tools/run_port_reliability_readout_test_v1.sh']))
sha=parent.sha
write=parent.write


def checked_sources():
    protocol=json.loads(PROTOCOL.read_text());actual={n:sha(ROOT/n) for n in SOURCE_FILES}
    if (json.loads(SOURCES.read_text())!=dict(protocol=core.VERSION,sources=actual)
            or protocol['protocol']!=core.VERSION or protocol['input_pins']!=PINS
            or protocol['counts']!=core.COUNTS or protocol['automatic_promotion']
            or protocol['training'] or protocol['threshold_updates']!=0):
        raise ValueError('Fixed TEST diagnostic source/contract changed')
    parent.checked_sources()
    return protocol,dict(manifest_sha256=sha(SOURCES),sources=actual)


def checked_inputs(require_tensors):
    missing=[];actual={}
    for name,pin in PINS.items():
        path=ROOT/name
        if not path.exists():
            if not require_tensors and name in (ROI,HEAD):missing.append(name);continue
            raise FileNotFoundError('Required frozen artifact unavailable: '+name)
        actual[name]=sha(path)
        if actual[name]!=pin:raise ValueError('Frozen input SHA changed: '+name)
    # Resolve the final TRAIN weights and VAL cutoffs before any TEST tensors.
    model=json.loads((ROOT/FIT/'models.json').read_text())
    cutoffs=json.loads((ROOT/FIT/'cutoffs.json').read_text())
    report=json.loads((ROOT/FIT/'report.json').read_text())
    model,cutoffs=core.frozen_bundle(model,cutoffs,report)
    complete=json.loads((ROOT/FIT/'completion.json').read_text())
    if complete['status']!=report['status'] or complete['TEST_read'] or complete['selected_arm'] is not None:
        raise ValueError('Parent failed-VAL history changed')
    collection=json.loads((ROOT/COLLECT/'completion.json').read_text())
    if (collection['status']!='GT_FREE_ROI_COLLECTION_COMPLETE' or collection['GT_online']
            or collection['updates'] or collection['state_before']!=collection['state_after']
            or collection['artifacts']['port_test']['count']!=1440
            or collection['artifacts']['port_test']['roi_sha256']!=PINS[ROI]
            or collection['artifacts']['port_test']['predictions_sha256']!=PINS[COLLECT+'/port_test/predictions.jsonl']):
        raise ValueError('Historical GT-free ROI collection identity differs')
    policy=json.loads((ROOT/POLICY).read_text())
    binding.checked_frontend(policy['front_end'])
    baseline=json.loads((ROOT/BASE/'summary.json').read_text())
    if baseline['policy_sha256']!=PINS[POLICY] or baseline['test_used_for_selection']:
        raise ValueError('Formal TEST baseline policy differs')
    return model,cutoffs,policy,dict(available_sha256=actual,missing_required_tensors=missing,
        ready_for_evaluation=not missing,parent_VAL_status=report['status'],TEST_evaluated=False)


def read_rows(path):
    with Path(path).open() as f:return [json.loads(s) for s in f if s.strip()]


def table(report):
    lines=['# 冻结TEST补评：保留VAL失败身份','',
        '温度／修正权重与四种VAL门限固定；原simple使用点另报。TEST已多次暴露，不选模型或门限。','',
        '| 分组／方法 | 接受数 | FA | FR | ED | CR | 接受覆盖 | 正确保留 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for group,methods in report['fixed_VAL_cutoff_summary'].items():
        points=dict(methods,original_simple=report['original_policy_summary'][group])
        for name,v in points.items():
            s=v['states'];good=s['CR']+s['FR']
            lines.append('| %s／%s（固定门限） | %d | %d | %d | %d | %d | %.2f%% | %.2f%% |'%(group,name,
                s['FA']+s['CR'],s['FA'],s['FR'],s['ED'],s['CR'],100*v['acceptance_coverage_all_frames'],100*s['CR']/good if good else 0.))
    lines+=['','| 分组／方法 | 接受数 | FA | FR | ED | CR | 接受覆盖 | 正确保留 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for group,point in report['statistics']['residual'].items():
        controls={m:v for m,v in point['same_count_controls'].items()}
        controls['residual']=point['actual']
        for name in ('residual','temperature','full_simple','score_only'):
            v=controls[name];s=v['states'];good=s['CR']+s['FR']
            lines.append('| %s／%s（同数离线） | %d | %d | %d | %d | %d | %.2f%% | %.2f%% |'%(group,name,
                s['FA']+s['CR'],s['FA'],s['FR'],s['ED'],s['CR'],100*v['acceptance_coverage_all_frames'],100*s['CR']/good if good else 0.))
    lines+=['','所有同数/同CR参照为离线诊断，不生成TEST门限，不自动晋级候选。固定门限实际结果、原policy、中心三种分母及连续状态见report.json。']
    return '\n'.join(lines)+'\n'


def run(args):
    args._created_output=False
    protocol,source=checked_sources()
    if args.out.resolve().parent.parent!=(ROOT/'work_dirs'/core.VERSION).resolve():
        raise ValueError('Use work_dirs/'+core.VERSION+'/RUN_ID/result')
    if args.out.exists():raise FileExistsError('Refuse overwrite')
    model,cutoffs,policy,proof=checked_inputs(args.mode=='evaluate')
    args.out.mkdir(parents=True)
    args._created_output=True
    write(args.out/'protocol.json',protocol);write(args.out/'sources.json',source)
    write(args.out/'input_check.json',proof)
    if args.mode=='check':
        write(args.out/'completion.json',dict(status='STATIC_READY' if proof['ready_for_evaluation'] else 'STATIC_OK_TENSORS_REQUIRED',
            TEST_evaluated=False,training=False,missing_required_tensors=proof['missing_required_tensors']))
        print('STATIC_CHECK_OK',proof['missing_required_tensors'],flush=True);return
    import torch
    from crane_project.utils import port_reliability_readout_test_v1_torch as native
    from crane_project.utils.port_geometry_midpoint_sigma_v1 import SigmaMidpointHead
    records=read_rows(ROOT/BASE/'test_decisions.jsonl')
    predictions=read_rows(ROOT/COLLECT/'port_test/predictions.jsonl')
    rows=core.bind_rows(records,predictions);before=simple.fingerprint([rows,model,cutoffs,policy])
    api=binding.Sigma15Reliability(policy,policy['front_end'])
    for r in rows:
        if api.decide(r['pred'],r['image_size'])!=r['original_simple_decision']:
            raise ValueError('Formal baseline three flags no longer replay')
    inventory=[dict(image=r['image'],b=p['b'],midpoint=r['pred'],image_size=r['image_size'],scale_xy=p['scale_xy'])
               for r,p in zip(rows,predictions)]
    # Only byte-pinned original tensor/checkpoint files are unpickled.
    torch.set_num_threads(1);device='cpu' if args.gpu<0 else 'cuda:'+str(args.gpu)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    payload=torch.load(str(ROOT/HEAD),map_location='cpu')
    if (payload['sigma_cells']!=1.5 or payload['epoch']!=3 or payload['updates']!=2706
            or payload['frozen_b']['checkpoint_sha256']!=binding.B_SHA):raise ValueError('Wrong frozen head')
    head=SigmaMidpointHead(1.5).to(device);head.load_state_dict(payload['head_state'],strict=True)
    head.eval().requires_grad_(False)
    cache=torch.load(str(ROOT/ROI),map_location='cpu')
    if (cache['protocol']!='port_size_candidate_downstream_v1_fixed_diagnostic'
            or cache['GT_online'] or cache['input_names']!=list(core.INPUT_NAMES)):
        raise ValueError('Only pinned GT-free port ROI cache is admissible')
    x,ids,feature_proof=native.extract(head,cache['rows'],inventory,device)
    np.savez_compressed(args.out/'test_features.npz',features=x,images=np.asarray(ids))
    risks,decisions=core.score_rows(rows,x,ids,model,cutoffs)
    with gzip.open(args.out/'scored_TEST.jsonl.gz','xt') as f:
        for r,d in zip(rows,decisions):
            f.write(json.dumps(dict(r,experiment_risks={m:v[r['image']] for m,v in risks.items()},candidate_decisions=d['methods']),allow_nan=False)+'\n')
    result=core.summarize(rows,risks,cutoffs)
    if (before!=simple.fingerprint([rows,model,cutoffs,policy]) or checked_sources()[1]!=source
            or any(sha(ROOT/n)!=pin for n,pin in PINS.items())):raise ValueError('Frozen inputs/sources changed')
    report=dict(result,protocol=core.VERSION,parent_VAL_status='VAL_FAILED_STOP',VAL_failure_retracted=False,
        frozen_cutoffs=cutoffs,input_sha256=PINS,sources=source,feature_proof=feature_proof,
        inference_model=FIT+'/models.json',TEST_repeatedly_exposed=True,TEST_used_for_selection=False,
        detector_updates=0,head_updates=0,quality_updates=0,threshold_updates=0,GT_online=False,
        boxes_scores_center_angle_output_unchanged=True,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT),text=True).strip())
    write(args.out/'report.json',report);(args.out/'summary.md').write_text(table(report))
    write(args.out/'completion.json',dict(protocol=core.VERSION,status=report['status'],TEST_evaluated=True,
        parent_VAL_status='VAL_FAILED_STOP',automatic_promotion=False,
        artifacts={p.name:sha(p) for p in args.out.iterdir() if p.is_file()}))
    print(report['status'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('check','evaluate'),required=True)
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    try:run(args)
    except Exception as e:
        if getattr(args,'_created_output',False) and not (args.out/'failure.json').exists():
            write(args.out/'failure.json',dict(error=repr(e),status='ENGINEERING_FAILED'))
        raise


if __name__=='__main__':main()
