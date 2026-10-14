#!/usr/bin/env python3
"""CPU TRAIN-only material audit; no VAL/TEST, fitting or threshold changes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.utils import port_reliability_train_support_v1 as core
from crane_project.tools import run_port_reliability_conditional_evidence_v1 as io
from crane_project.tools.review_port_reliability_conditional_evidence_v1 import equivalent_payload


def load(allow_absent=False):
    p=json.loads((ROOT/('crane_project/tools/'+core.VERSION+'_protocol.json')).read_text());s=json.loads((ROOT/('crane_project/tools/'+core.VERSION+'_sources.json')).read_text())
    if p['settings']!=core.SETTINGS or p['protocol']!=core.VERSION:raise ValueError('Fixed diagnostic contract differs')
    pins=dict(p['input_pins'],**p['frontend_pins'],**s['sources']);absent=[]
    for n,h in pins.items():
        if allow_absent and n in p['frontend_pins'] and not (ROOT/n).exists():absent.append(n);continue
        if io.sha(ROOT/n)!=h:raise ValueError('Source/input changed: '+n)
    for key,review_file in [('previous','server_review.json'),('native','mac_review.json')]:
        path=ROOT/p[key];review=json.loads((path/review_file).read_text())
        if not review['passed'] or review['report_sha256']!=io.sha(path/'report.json'):raise ValueError('Unreviewed cache identity')
    wrapped=json.loads((ROOT/p['policy']).read_text());policy=wrapped['simple_policy']
    from crane_project.utils import port_midpoint_sigma15_reliability_v1 as binding
    binding.checked_frontend(wrapped['front_end'])
    rows=io.read(ROOT/p['previous']/'scored_TRAIN.jsonl.gz');core.validate(rows)
    if len(rows)!=2558:raise ValueError('Complete TRAIN2558 required')
    api=core.simple.SimpleComponentReliability(policy)
    tracker=None;previous=None
    for r in rows:
        v=api.decide(r['pred'],r['image_size']);v['final_box_original']=v.pop('raw_b_output');saved=r['original_simple_decision']
        if any(v[k]!=saved[k] for k in v if k!='risks') or any((v['risks'][k] is None)!=(saved['risks'][k] is None) or (v['risks'][k] is not None and abs(v['risks'][k]-saved['risks'][k])>1e-12) for k in v['risks']):raise ValueError('Formal simple replay differs')
        reset=previous!=r['sequence'] or r['frame_id'] in p['boundaries'].get(r['sequence'],[])
        if previous!=r['sequence']:tracker=core.temporal.CausalSizeEvidence()
        point=tracker.observe(r['pred'],r['frame_id'],r['image_size'],reset);point['acquisition_boundary']=r['frame_id'] in p['boundaries'].get(r['sequence'],[])
        if not equivalent_payload(point,r['temporal']):raise ValueError('Frozen causal history identity differs')
        previous=r['sequence']
    with np.load(ROOT/p['native']/'train_native_projected.npz',allow_pickle=False) as f:features=np.array(f['features'],dtype=float);ids=f['images'].tolist()
    return p,pins,absent,rows,features,ids


def run(args):
    p,pins,absent,rows,x,ids=load(args.allow_absent_frontend)
    if args.check_only:print('Reviewed full TRAIN/cache/policy/history verified; no analysis or fit');return
    if absent:raise ValueError('Server analysis requires frontend SHA verification')
    out=args.out
    if out is None or out.is_symlink() or out.name!='result' or out.resolve().parent.parent!=(ROOT/'work_dirs'/core.VERSION).resolve():raise ValueError('Use unified work_dirs/version/RUN_ID/result')
    out.mkdir(parents=True,exist_ok=False)
    result=core.audit(rows,x,ids)
    report=dict(protocol=core.VERSION,status=result['status'],settings=core.SETTINGS,result=result,input_pins=pins,frames=len(rows),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),fits=0,threshold_updates=0,
        detector_inferences=0,neural_updates=0,policy_changed=False,formal_flags_changed=False,VAL_opened=False,TEST_opened=False,
        historical_VAL_TEST_exposure_disclosed=True,GT_offline_only=True,candidate_adopted=False)
    for n,h in pins.items():
        if io.sha(ROOT/n)!=h:raise ValueError('Frozen input changed')
    io.write(out/'report.json',report);io.write(out/'error_runs.json',result['error_runs']);io.write(out/'remaining_FA.json',result['remaining_FA'])
    io.write(out/'completion.json',dict(status=report['status'],report_sha256=io.sha(out/'report.json'),frames=len(rows)))
    print(result['groups']['all']['states']);print(result['support']);print(result['proximity'])


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',type=Path);a.add_argument('--check-only',action='store_true');a.add_argument('--allow-absent-frontend',action='store_true');run(a.parse_args())
