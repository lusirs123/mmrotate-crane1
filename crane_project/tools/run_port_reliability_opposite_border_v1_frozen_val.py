#!/usr/bin/env python3
"""Explicitly authorized frozen VAL supplement, including a failed historical probe.

No fitting, cutoff search, TRAIN data extraction, TEST branch or policy update.
The original source/contract/failed result remains untouched.
"""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.tools import run_port_reliability_opposite_border_v1 as old
from crane_project.utils import port_reliability_opposite_border_v1 as method
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils import port_reliability_state_continuity_v1 as states

VERSION='port_reliability_opposite_border_v1_frozen_VAL_supplement'
TRAIN=ROOT/'work_dirs/port_reliability_opposite_border_v1/20261008_border_evidence_fix1/train_check'
OUT=TRAIN.parent/'val_supplement'
PINS={'report.json':'bdf70c2ca312a02d3abf93bf9790b1075e4f6e52a0fe9e72ce592de5c9228795',
      'models.json':'b533fd0efb2849ac22d449c3302aa398f419813d8d2c04330c003e9156c084ef',
      'completion.json':'2c94a84c96d6efea580aed34383a875cca3782d5bbf4727ad36f9e918f5e8732'}
SOURCES=ROOT/'crane_project/tools/port_reliability_opposite_border_v1_frozen_val_sources.json'
SOURCE_FILES=old.SOURCE_FILES+[
 'crane_project/tools/port_reliability_opposite_border_v1_sources.json',
 'crane_project/tools/run_port_reliability_opposite_border_v1_frozen_val.py',
 'tests/test_port_reliability_opposite_border_v1_frozen_val.py',
 'tools/run_port_reliability_opposite_border_v1_frozen_val.sh']


def checked_sources():
    original_contract,original=old.checked_sources()
    actual={name:old.sha(ROOT/name) for name in SOURCE_FILES}
    if json.loads(SOURCES.read_text())!=dict(protocol=VERSION,sources=actual):
        raise ValueError('Frozen VAL supplement source changed')
    return original_contract,dict(manifest_sha256=old.sha(SOURCES),sources=actual,original_sources=original)


def frozen_model_source(path=TRAIN):
    """Accept the exact authorized failed result, never an arbitrary new fit."""
    path=Path(path).resolve()
    if path!=TRAIN.resolve():raise ValueError('Only the authorized historical result is permitted')
    if {name:old.sha(path/name) for name in PINS}!=PINS:
        raise ValueError('Frozen historical model/report/receipt changed')
    receipt=json.loads((path/'completion.json').read_text())
    if any(old.sha(path/name)!=digest for name,digest in receipt['artifacts'].items()):
        raise ValueError('Historical result artifact changed')
    report=json.loads((path/'report.json').read_text());models=json.loads((path/'models.json').read_text())
    contract,source=old.checked_sources()
    if (receipt['protocol']!=method.VERSION or receipt['stage']!='train'
        or receipt['status']!='TRAIN_CAPABILITY_FAILED_STOP'
        or report['conclusion']!='TRAIN_CAPABILITY_FAILED_STOP' or report['stage']!='train'
        or report['probe_gate']['passed'] or report['selected_arm'] is not None
        or report['VAL_evaluated'] or report['TEST_read'] or report['sources']!=source
        or report['contract']!=contract or report['models']!=models
        or set(models)!=set(method.SCHEMAS)
        or set(report['fit_points'])!={method.ARM,*method.CONTROLS}
        or any(v['cutoff_role']!='fit' for v in report['fit_points'].values())):
        raise ValueError('Authorized historical identity differs; no contract bypass for arbitrary model')
    return report


def checked_out(path):
    path=Path(path).resolve()
    if path!=OUT.resolve():raise ValueError('Keep frozen VAL supplement in original experiment parent')
    if path.exists():raise FileExistsError('Refuse overwrite')
    return path


def evaluate(rows,models,points):
    """Score first, then use GT only for offline state/rank/continuity reports."""
    if not rows or any(r['reliability_role']!='val' or r['split']!='val' for r in rows):
        raise ValueError('Only existing VAL data allowed; no added data roles')
    states.checked_order(rows)
    before=simple.fingerprint(dict(rows=rows,models=models,points=points))
    scores=old.score_rows(rows,models)
    groups=old.ab.groups(rows)
    diagnostics=method.describe(rows,scores,points[method.ARM]['risk_le'])
    ordinary_diagnostics=method.describe(rows,scores,points['ordinary_roi']['risk_le'],'ordinary_roi')
    fixed_points={name:{m:states.summarize(group,{r['image'] for r in group if r['pred'] is not None
        and scores[m][r['image']]<=points[m]['risk_le']}) for m in scores} for name,group in groups.items()}
    formal={name:states.summarize(group,{r['image'] for r in group if r['original_simple_decision']['size_accepted']})
            for name,group in groups.items()}
    decisions={};scored=[]
    for r in rows:
        d=method.decide(models[method.ARM],points[method.ARM]['risk_le'],r['original_simple_decision'],r['experiment_features'][method.ARM])
        expected=deepcopy(r['original_simple_decision'])
        if r['pred'] is not None:
            expected['risks']['size']=scores[method.ARM][r['image']]
            expected['size_accepted']=scores[method.ARM][r['image']]<=points[method.ARM]['risk_le']
        if d!=expected:raise ValueError('Only size risk/flag may change')
        decisions[r['image']]=d
        scored.append(dict(r,experiment_risks={m:scores[m][r['image']] for m in scores},candidate_diagnostic_decision=d))
    if simple.fingerprint(dict(rows=rows,models=models,points=points))!=before:
        raise ValueError('Frozen data/model/cutoff mutation')
    checks=method.gate(diagnostics)
    checks['gate_scope']='descriptive_frozen_VAL_supplement_not_original_contract_pass'
    return dict(diagnostics=diagnostics,ordinary_roi_diagnostics=ordinary_diagnostics,
        fixed_fit_points=fixed_points,formal_policy_summary=formal,
        three_components=old.parent.prior.components(rows,decisions),descriptive_checks=checks),scored


def run(args):
    out=checked_out(args.out);contract,sources=checked_sources();train=frozen_model_source()
    rows,input_proof=old.load_val_rows()
    rows,arrays,cache=old.collect(rows,args.cache_dir,'val')
    values,scored=evaluate(rows,train['models'],train['fit_points'])
    if checked_sources()[1]!=sources or frozen_model_source()!=train:
        raise ValueError('Source or historical result changed during supplement')
    report=dict(protocol=VERSION,stage='VAL',evaluation_identity='user_authorized_frozen_VAL_supplement',
        status='FROZEN_VAL_SUPPLEMENT_COMPLETE_REVIEW_REQUIRED',
        frontend='B24+sigma1.5/head_epoch03',sources=sources,input_proof=input_proof,cache=cache,
        original_policy_sha256=contract['original_policy_sha256'],original_sampling_contract=contract,
        frozen_TRAIN_source=dict(path=str(TRAIN.relative_to(ROOT)),pins=PINS,
            source_commit=json.loads((TRAIN/'completion.json').read_text())['git_commit'],
            original_conclusion=train['conclusion'],original_probe_gate=deepcopy(train['probe_gate']),
            original_training_roles_preserved=True),
        models=deepcopy(train['models']),fit_points=deepcopy(train['fit_points']),
        data_roles=['VAL'],rows=len(rows),outputs=sum(r['pred'] is not None for r in rows),
        fit_calls=0,control_refits=0,cutoff_fits=0,threshold_role='historical_fit_frozen',
        original_contract_passed=False,selected_arm=None,policy_modified=False,TEST_read=False,
        geometry_modified=False,detector_updates=0,midpoint_updates=0,feature_network_updates=0,
        new_inference_calls=0,VAL_cutoff_changed=False,
        future_workflow='existing TRAIN fit; complete VAL evaluation/working point; frozen TEST report; no new subroles',
        **values)
    out.mkdir(parents=True)
    old.write(out/'report.json',report)
    (out/'frozen_models.json').write_bytes((TRAIN/'models.json').read_bytes())
    old.write(out/'frozen_fit_points.json',train['fit_points'])
    np.savez_compressed(str(out/'fixed_fields.npz'),**arrays)
    with gzip.open(out/'scored_VAL.jsonl.gz','xt') as f:
        for r in sorted(scored,key=lambda r:r['image']):f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    old.write(out/'completion.json',dict(protocol=VERSION,status=report['status'],stage='VAL',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT),text=True).strip(),
        artifacts={p.name:old.sha(p) for p in out.iterdir() if p.is_file()},fit_calls=0,cutoff_fits=0,TEST_read=False))
    all_stats=report['diagnostics']['all']
    print(json.dumps(dict(status=report['status'],states=all_stats['actual']['states'],
        accepted=all_stats['actual']['accepted_outputs'],same_count_controls={k:v['states'] for k,v in all_stats['same_count_controls'].items()},
        descriptive_check_failures=len(report['descriptive_checks']['failures']),original_probe_passed=False,
        fit_calls=0,cutoff_fits=0,out=str(out)),ensure_ascii=False),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--cache-dir',type=Path,default=ROOT/old.CACHE)
    run(p.parse_args())
