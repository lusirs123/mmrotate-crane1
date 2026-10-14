#!/usr/bin/env python3
"""Full TRAIN once; freeze and evaluate complete VAL for every valid incumbent."""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from crane_project.tools import run_port_reliability_actual_fr_v1 as old
from crane_project.tools import eval_port_reliability_actual_fr_val_v1 as val
from crane_project.utils import port_reliability_actual_fr_bound_v1 as core

OLD = 'work_dirs/port_reliability_actual_fr_v1/20261014_actual_fr_fix1/result/'
PINS = dict(val.PINS, **{OLD+'control_feasible_witnesses.json':
    'c77db876ec1ea8992cdcd53232facd7fafe4623569e16d7eafb8080bc389f5f0'})


def load(allow_absent=False):
    _, inherited, absent, rows, policy, _ = old.load(allow_absent)
    contract = val.read(ROOT/'crane_project/tools/port_reliability_actual_fr_bound_v1_protocol.json')
    if contract['settings'] != core.SETTINGS:
        raise ValueError('Fixed contract changed')
    manifest = val.read(ROOT/'crane_project/tools/port_reliability_actual_fr_bound_v1_sources.json')['sources']
    pins = dict(inherited, **PINS, **manifest)
    for name, digest in pins.items():
        if name in absent:
            continue  # Only missing frontend weights already disclosed by old.load.
        if val.sha(ROOT/name) != digest:
            raise ValueError('Changed source: '+name)
    review = val.read(ROOT/(OLD+'server_review.json'))
    if not review['passed'] or review['report_sha256'] != PINS[OLD+'report.json']:
        raise ValueError('Prior report lacks matching independent review')
    previous = val.read(ROOT/(OLD+'report.json'))['result']['reference']
    witness = val.read(ROOT/(OLD+'control_feasible_witnesses.json'))['full_TRAIN']
    control = previous['models']['control3']
    data = core.prior.build(rows)
    for key in ('mean', 'scale'):
        if not np.allclose(data[key], control[key], atol=1e-12, rtol=0):
            raise ValueError('TRAIN membership/normalizer differs')
    cert = core.witness_certificate(data, witness['weights'], 59)
    if cert != witness['certificate'] or cert['FA'] != 59:
        raise ValueError('Known FA59 witness differs')
    formal = val.read(ROOT/val.POLICY)['simple_policy']
    base = dict(candidate_model=deepcopy(previous['models']['actual_count3']),
        control_model=deepcopy(control), candidate_logit_lt=0.,
        formal_risk_le=formal['cutoffs']['simple']['size']['risk_le'],
        protected_risk_le=previous['thresholds']['full_simple']['risk_le'],
        score_risk_le=previous['thresholds']['score_only']['risk_le'])
    oldkeep = {r['image'] for r in rows if r['pred'] is not None and
        core.prior.proxy.logit(control,r) <= previous['thresholds']['control3']['risk_le']}
    witnesskeep = {r['image'] for r,z in zip(data['rows'], data['design']@witness['weights']) if z < 0}
    if oldkeep != witnesskeep:
        raise ValueError('Witness does not preserve old control membership')
    return contract, pins, absent, rows, witness, base


def run(args):
    contract,pins,absent,train,witness,base = load(args.allow_absent_frontend)
    if args.check_only:
        print('TRAIN/source/FA59 witness checked; no fitting'); return
    if absent:
        raise ValueError('Server fitting requires original frontend weights')
    out=args.out
    if out is None or out.is_symlink() or out.name!='result' or out.resolve().parent.parent!=(ROOT/'work_dirs'/core.VERSION).resolve():
        raise ValueError('Use version/run_id/result')
    out.mkdir(parents=True,exist_ok=False)
    val.write(out/'frozen_protocol.json',contract)
    solved=core.solve(train,witness['weights'])
    frozen=deepcopy(base)
    if solved['model'] is not None:
        frozen['candidate_model']=deepcopy(solved['model'])
        frozen['original_solver_status']=deepcopy(solved['solver'])
    else:
        frozen=None
    val.write(out/'frozen_model_before_VAL.json',frozen)
    evaluations={};curves={}
    if frozen is not None:
        evaluations['TRAIN'],curves['TRAIN']=val.evaluate(train,frozen)
        # Only now open complete VAL rows. No threshold fitting or model reselection.
        with gzip.open(ROOT/(val.VAL+'scored_VAL.jsonl.gz'),'rt') as stream:
            rows=[json.loads(line) for line in stream]
        if len(rows)!=887 or len({r['image'] for r in rows})!=887 or any(
                r['split']!='val' or r['reliability_role']!='val' for r in rows):
            raise ValueError('Wrong complete VAL identity')
        if {r['image'] for r in train}&{r['image'] for r in rows}:
            raise ValueError('TRAIN/VAL overlap')
        evaluations['VAL'],curves['VAL']=val.evaluate(rows,frozen)
    for name,digest in pins.items():
        if val.sha(ROOT/name)!=digest:raise ValueError('Source changed during run')
    report=dict(protocol=core.VERSION,settings=core.SETTINGS,input_pins=pins,
        solving=solved,frozen_workpoints=frozen,evaluations=evaluations,
        solver_calls=1,control_refits=0,threshold_updates=0,
        VAL_opened=frozen is not None,TEST_opened=False,GPU_used=False,detector_inferences=0,
        candidate_adopted=False,formal_policy_changed=False,
        training_objective_improved=bool(solved['model'] is not None and solved['model']['certificate']['FA']<59),
        status='VALID_CANDIDATE_COMPLETE_VAL_REPORT' if frozen is not None else 'NO_NEW_CANDIDATE_STOP')
    val.write(out/'report.json',report)
    val.write(out/'risk_correct_retention_curves.json',curves)
    val.write(out/'completion.json',dict(status=report['status'],report_sha256=val.sha(out/'report.json')))
    print('solver',solved['solver'],flush=True)
    for role,result in evaluations.items():
        print(role,{m:v['delivery']['states'] for m,v in result['groups']['all']['methods'].items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);p.add_argument('--check-only',action='store_true')
    p.add_argument('--allow-absent-frontend',action='store_true');run(p.parse_args())
