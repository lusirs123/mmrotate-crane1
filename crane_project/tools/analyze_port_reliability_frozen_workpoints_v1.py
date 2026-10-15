#!/usr/bin/env python3
"""Frozen working-point supplement on existing cached TRAIN/VAL/exposed TEST.

No fitting, threshold selection, detector inference or online GT access.
"""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.utils import port_reliability_video_transfer_v1 as metrics
from crane_project.utils import port_reliability_fr_constraint_v1 as model_core
from crane_project.tools import review_port_reliability_video_transfer_v1 as independent
VERSION='port_reliability_frozen_workpoints_v1'
FR='work_dirs/port_reliability_fr_constraint_v1/20261014_fr_constraint_v1/result/'
STACK='work_dirs/port_reliability_temporal_stack_test_v1/20261014_frozen_stack_test_v1/result/'
PINS={FR+'report.json':'11222fe2f7be88bd14363f512050cd7ad737ff743d03ef30780e3e0d12261029',
 FR+'reference_TRAIN.jsonl.gz':'8cf3f79cbbc7cbffbaaa2d7f606a7adc49dc6a2cf2055ed1f587e907895c1a61',
 STACK+'report.json':'a5c29f7aa3cad07900f414d425ad6babe304bd58f834292b45874673c0a1e7c9',
 STACK+'scored_VAL.jsonl.gz':'7bcd34372fe4bb3095b513dd01f8d92c5d8ea200398627d9d6ee9ea3c0b7341b',
 STACK+'scored_TEST.jsonl.gz':'03ddf68d16d2178ebbd9c2e288b3f9266a428f5552642eaa1aa35461a0710660',
 'work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1/fit/policy.json':'38d914f114fcdb257f33f1cd6390ebcb4e103adfa1d9022628875e137c916e1f'}
METHODS=('raw','formal_simple','protected_simple','constrained_zero','score_train95')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(l) for l in gzip.open(p,'rt')]
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')


def freeze():
    for n,h in PINS.items():
        if sha(ROOT/n)!=h:raise ValueError('Changed frozen source: '+n)
    for directory in (FR,STACK):
        rev=read(ROOT/(directory+'mac_review.json'))
        if not rev['passed'] or rev['report_sha256']!=sha(ROOT/(directory+'report.json')):
            raise ValueError('Prior independent review differs')
    r=read(ROOT/(FR+'report.json'))['result']['reference']
    policy=read(ROOT/'work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1/fit/policy.json')['simple_policy']
    for k in ('mean','scale','weights'):
        if not np.allclose(r['models']['control3'][k],policy['models']['size'][k],atol=1e-12,rtol=0):
            raise ValueError('Control not formal simple')
    return dict(control_model=r['models']['control3'],candidate_model=r['models']['constrained3'],
        protected_risk_le=r['thresholds']['full_simple']['risk_le'],
        protected_logit_le=r['thresholds']['control3']['risk_le'],
        protected_fit_videos=r['thresholds']['full_simple']['fit_videos'],
        candidate_logit_lt=0.,score_risk_le=r['thresholds']['score_only']['risk_le'],
        formal_risk_le=policy['cutoffs']['simple']['size']['risk_le'])


def apply(part,frozen):
    """Decisions only read current prediction/features and frozen constants."""
    result=deepcopy(part)
    for r in result:
        if r['pred'] is None:
            r['supplement_flags']={m:False for m in METHODS}
            r['risks']['frozen_constraint_logit']=None
            continue
        control=model_core.logit(frozen['control_model'],r)
        candidate=model_core.logit(frozen['candidate_model'],r)
        replay=float(metrics.fusion.simple.sigmoid(control))
        if abs(replay-r['risks']['full_simple'])>1e-12:
            raise ValueError('Current formal scoring replay differs')
        formal=r['risks']['full_simple']<=frozen['formal_risk_le']
        if formal!=r['original_simple_decision']['size_accepted']:
            raise ValueError('Formal flag replay differs')
        r['risks']['frozen_constraint_logit']=candidate
        r['supplement_flags']=dict(raw=True,formal_simple=formal,
            protected_simple=r['risks']['full_simple']<=frozen['protected_risk_le'],
            constrained_zero=candidate<frozen['candidate_logit_lt'],
            score_train95=r['risks']['score_only']<=frozen['score_risk_le'])
    return result


def audit(part,frozen):
    scored=apply(part,frozen); groups={}; checks=0
    for a,r in zip(scored,part):
        if any(a[k]!=v for k,v in r.items() if k!='risks'):
            raise ValueError('Prediction/GT/output/original flags changed')
        if any(a['risks'][k]!=v for k,v in r['risks'].items()):
            raise ValueError('Historical score changed')
        if r['pred'] is not None:
            x=independent.descriptor(r,3); m=frozen['candidate_model']
            z=m['weights'][0]+sum((a-mu)/s*w for a,mu,s,w in zip(x,m['mean'],m['scale'],m['weights'][1:]))
            if abs(z-a['risks']['frozen_constraint_logit'])>1e-12:
                raise ValueError('Independent candidate logit differs')
        checks+=1
    for name,g in metrics.fusion.level.metrics.grouped(scored).items():
        methods={}
        for method in METHODS:
            kept={r['image'] for r in g if r['supplement_flags'][method]}
            delivery=metrics.counts(g,kept);independent.check_summary(delivery,g,kept)
            controls={}
            for c in ('full_simple','score_only'):
                sc=metrics.point(g,c,count=delivery['accepted']);cr=metrics.point(g,c,correct=delivery['states']['CR'])
                independent.check_point(sc,g,c,count=delivery['accepted'])
                independent.check_point(cr,g,c,correct=delivery['states']['CR'])
                controls[c]=dict(same_count=sc,same_CR=cr)
            methods[method]=dict(delivery=delivery,controls=controls); checks+=5
        aucs={m:metrics.fusion.level.metrics.auc(g,m) for m in ('full_simple','score_only','frozen_constraint_logit')}
        for m,auc in aucs.items():
            good=np.array([r['risks'][m] for r in g if independent.bad(r) is False]);err=np.array([r['risks'][m] for r in g if independent.bad(r) is True])
            expected=float(((err[:,None]>good[None])+.5*(err[:,None]==good[None])).mean()) if len(good) and len(err) else None
            if (expected is None)!=(auc is None) or (auc is not None and abs(expected-auc)>1e-12):raise ValueError('Independent AUC differs')
        out=[r for r in g if r['pred'] is not None]
        hit=sum(math.hypot(r['pred'][0]-r['gt'][0],r['pred'][1]-r['gt'][1])<15 for r in out)
        groups[name]=dict(methods=methods,AUROC=aucs,center=dict(output_frames=len(out),
            correct_output_frames=hit,hit_rate_outputs=hit/len(out) if out else None,
            output_coverage=len(out)/len(g),full_frame_correct_coverage=hit/len(g),
            center_and_angle_flags_unchanged=True))
    return dict(frames=len(part),groups=groups,independent_checks=checks)


def validate_role(role,part,expected):
    allowed={'train','train_sim'} if role=='TRAIN' else {role.lower()}
    if len(part)!=expected or any(r['split'] not in allowed or r['reliability_role']!=role.lower() for r in part):
        raise ValueError('Wrong complete split')
    if role=='TRAIN' and any(r['split']=='train_sim' and r['domain']!='sim' for r in part):
        raise ValueError('Wrong native sim TRAIN identity')


def run(out):
    if out.is_symlink() or out.name!='result' or out.resolve().parent.parent!=(ROOT/'work_dirs'/VERSION).resolve():
        raise ValueError('Use work_dirs/version/RUN_ID/result')
    frozen=freeze();out.mkdir(parents=True,exist_ok=False)
    write(out/'frozen_workpoints.json',frozen) # persisted before VAL/TEST scoring
    datasets={'TRAIN':rows(ROOT/(FR+'reference_TRAIN.jsonl.gz')),
              'VAL':rows(ROOT/(STACK+'scored_VAL.jsonl.gz')),
              'TEST':rows(ROOT/(STACK+'scored_TEST.jsonl.gz'))}
    expected={'TRAIN':2558,'VAL':887,'TEST':1440}; ids=set()
    for role,part in datasets.items():
        validate_role(role,part,expected[role])
        new={r['image'] for r in part}
        if len(new)!=len(part) or ids&new:raise ValueError('Duplicate/overlapping split identity')
        ids|=new
    # Recompute training-only quantiles, never calibrate on VAL/TEST.
    train=datasets['TRAIN']; fitting=metrics.cutoff(train,'full_simple')
    if fitting['risk_le']!=frozen['protected_risk_le'] or fitting['fit_videos']!=frozen['protected_fit_videos']:
        raise ValueError('Training-only threshold provenance differs')
    roles={role:audit(part,frozen) for role,part in datasets.items()}
    for n,h in PINS.items():
        if sha(ROOT/n)!=h:raise ValueError('Frozen source changed while scoring')
    report=dict(protocol=VERSION,roles=roles,frozen_workpoints=frozen,input_pins=PINS,
        fits=0,threshold_updates=0,detector_inferences=0,GPU_used=False,
        formal_policy_changed=False,candidate_adopted=False,TEST_used_for_selection=False,
        TEST_repeatedly_exposed=True,TRAIN_reference_in_sample=True,
        prior_TRAIN_gate_failure_retained=True,supplement_authorized_after_gate_failure=True,
        cached_box_evaluation_only=True,actual_frontend_weight_check_this_run=False,
        limitations=['TRAIN-retention constraint does not guarantee VAL/TEST retention.',
         'The zero cutoff is a threshold, not threshold-free confidence.',
         'Offline same-count/same-CR comparisons use evaluation GT only.',
         'No new frames/features or duplicate per-frame cache written.'])
    write(out/'report.json',report)
    write(out/'review.json',dict(passed=True,rows=sum(expected.values()),
        independent_checks=sum(v['independent_checks'] for v in roles.values()),
        report_sha256=sha(out/'report.json'),frozen_before_VAL_TEST=True,
        source_code_sha256=sha(__file__),input_pins=PINS))
    for role,value in roles.items():print(role,{m:d['delivery']['states'] for m,d in value['groups']['all']['methods'].items()})

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);run(a.parse_args().out)
