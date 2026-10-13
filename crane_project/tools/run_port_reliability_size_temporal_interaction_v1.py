#!/usr/bin/env python3
"""CPU-only scalar TRAIN fit, one full exposed VAL calibration, frozen controls."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.utils import port_reliability_size_temporal_interaction_v1 as core
from crane_project.tools import run_port_reliability_conditional_evidence_v1 as io


def load(allow_absent=False):
    p=json.loads((ROOT/('crane_project/tools/'+core.VERSION+'_protocol.json')).read_text())
    s=json.loads((ROOT/('crane_project/tools/'+core.VERSION+'_sources.json')).read_text())
    if p['settings']!=core.SETTINGS:raise ValueError('Fixed scalar fitting contract changed')
    pins=dict(p['input_pins'],**p['frontend_pins'],**s['sources']);absent=[]
    for name,digest in pins.items():
        if allow_absent and name in p['frontend_pins'] and not (ROOT/name).exists():absent.append(name);continue
        if io.sha(ROOT/name)!=digest:raise ValueError('Input/source changed: '+name)
    previous=ROOT/p['previous_result'];report=json.loads((previous/'report.json').read_text())
    review=json.loads((previous/'server_review.json').read_text())
    if not review['passed'] or review['report_sha256']!=io.sha(previous/'report.json'):raise ValueError('Base report unreviewed')
    models=json.loads((previous/'models.json').read_text());cutoffs=json.loads((previous/'calibration.json').read_text())
    if models!=report['models'] or cutoffs!=report['cutoffs']:raise ValueError('Frozen model/cutoff identity differs')
    if report['TEST_opened'] or report['formal_policy_changed'] or report['candidate_adopted']:raise ValueError('Base role changed')
    return p,pins,absent,models,cutoffs


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path)
    parser.add_argument('--check-only',action='store_true');parser.add_argument('--allow-absent-frontend',action='store_true')
    a=parser.parse_args();p,pins,absent,models,previous_cutoffs=load(a.allow_absent_frontend)
    if a.check_only:print('Contract/source/input pins verified; no fit',flush=True);return
    if a.out is None or a.out.is_symlink() or a.out.resolve().parent.parent!=(ROOT/'work_dirs'/core.VERSION).resolve():
        raise ValueError('Use work_dirs/version/RUN_ID/result')
    a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    previous=ROOT/p['previous_result'];train=io.read(previous/'scored_TRAIN.jsonl.gz')
    if len(train)!=2558:raise ValueError('Complete TRAIN required')
    model=core.fit(train,models)
    io.write(a.out/'scalar_model.json',model)  # Freeze gamma BEFORE reading/scoring VAL.
    scored={'TRAIN':core.score(train,models,model['gamma'])}
    val=io.read(previous/'scored_VAL.jsonl.gz')
    if len(val)!=887 or any(r['split']!='val' or r['reliability_role']!='val' for r in val):raise ValueError('Complete VAL required')
    if set(r['image'] for r in train)&set(r['image'] for r in val):raise ValueError('TRAIN/VAL overlap')
    scored['VAL']=core.score(val,models,model['gamma'])
    cutoff=core.base.select_cutoff(scored['VAL'],core.METHOD)
    io.write(a.out/'calibration.json',cutoff)
    evaluations={role:core.evaluate(rows,cutoff['risk_le'],previous_cutoffs['fusion4']['risk_le']) for role,rows in scored.items()}
    import gzip
    for role,rows in scored.items():
        with gzip.open(a.out/('scored_'+role+'.jsonl.gz'),'xt') as f:
            for row in rows:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
        print(role,evaluations[role]['groups']['all']['candidate']['states'],flush=True)
    gate=core.gate(evaluations['VAL'])
    for name,digest in pins.items():
        if name not in absent and io.sha(ROOT/name)!=digest:raise ValueError('Frozen inputs modified')
    report=dict(protocol=core.VERSION,status='EVIDENCE_PASS_NO_ADOPTION' if gate['passed'] else 'EVIDENCE_FAILED_STOP',
        model=model,calibration=cutoff,base_models=models,base_cutoff=previous_cutoffs['fusion4']['risk_le'],
        roles=evaluations,gate=gate,input_pins=pins,absent_frontend=absent,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        scalar_saved_before_VAL=True,fit_calls=1,trainable_parameters=1,
        neural_training_updates=0,detector_inferences=0,TEST_opened=False,
        VAL_previously_exposed=True,formal_policy_changed=False,candidate_adopted=False,
        elapsed_seconds=time.monotonic()-started)
    io.write(a.out/'report.json',report)
    io.write(a.out/'completion.json',dict(status=report['status'],frames=3445,TEST_opened=False,fit_calls=1))
    print(report['status'],'gamma',model['gamma'],'VAL cutoff',cutoff['risk_le'],flush=True)


if __name__=='__main__':main()
