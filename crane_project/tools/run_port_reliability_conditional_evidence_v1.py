#!/usr/bin/env python3
"""Frozen-cache CPU diagnostic. No fitting, threshold selection or TEST access."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.utils import port_reliability_conditional_evidence_v1 as core


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')


def read(path):
    with gzip.open(path, 'rt') as f: return [json.loads(line) for line in f]


def load(allow_absent=False):
    p = json.loads((ROOT/('crane_project/tools/'+core.VERSION+'_protocol.json')).read_text())
    s = json.loads((ROOT/('crane_project/tools/'+core.VERSION+'_sources.json')).read_text())
    if p['settings'] != core.SETTINGS: raise ValueError('Prespecified settings differ')
    absent = []
    pins = dict(p['input_pins'], **s['sources'], **p['frontend_pins'])
    for name, digest in pins.items():
        if allow_absent and name in p['frontend_pins'] and not (ROOT/name).exists():
            absent.append(name); continue
        if sha(ROOT/name) != digest: raise ValueError('Source/input changed: '+name)
    previous = ROOT/p['previous_result']
    r = json.loads((previous/'report.json').read_text())
    reviewed = json.loads((previous/'server_review.json').read_text())
    if not reviewed['passed'] or reviewed['report_sha256'] != sha(previous/'report.json'):
        raise ValueError('Previous independent review not bound to report')
    if r['TEST_opened'] or r['candidate_adopted'] or r['formal_policy_changed']:
        raise ValueError('Previous run role changed')
    if r['cutoffs'] != json.loads((previous/'calibration.json').read_text()):
        raise ValueError('Previous threshold identity changed')
    return p, s, pins, absent


def main():
    a = argparse.ArgumentParser(); a.add_argument('--out',type=Path,required=True)
    a.add_argument('--allow-absent-frontend',action='store_true'); args=a.parse_args()
    if args.out.is_symlink() or args.out.resolve().parent.parent != (ROOT/'work_dirs'/core.VERSION).resolve():
        raise ValueError('Use work_dirs/version/RUN_ID/result')
    p,s,pins,absent = load(args.allow_absent_frontend)
    args.out.mkdir(parents=True,exist_ok=False)
    previous=ROOT/p['previous_result']; train=read(previous/'scored_TRAIN.jsonl.gz')
    core.validate(train,'TRAIN')
    if len(train)!=2558: raise ValueError('Incomplete TRAIN')
    frozen=core.freeze_train(train)
    write(args.out/'TRAIN_rules.json',frozen)  # Before reading VAL rows.
    cutoff=json.loads((previous/'calibration.json').read_text())['fusion4']['risk_le']
    roles={'TRAIN':core.analyze(train,frozen,cutoff)}
    val=read(previous/'scored_VAL.jsonl.gz');core.validate(val,'VAL')
    if len(val)!=887 or set(r['image'] for r in train)&set(r['image'] for r in val):
        raise ValueError('Incomplete/overlapping role identities')
    roles['VAL']=core.analyze(val,frozen,cutoff)
    report=dict(protocol=core.VERSION,settings=core.SETTINGS,roles=roles,
        screen=core.prerequisite_screen(roles),input_pins=pins,absent_frontend=absent,
        frozen_cutoff=cutoff,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        TRAIN_rules_saved_before_VAL=True,fit_calls=0,training_updates=0,detector_inferences=0,
        TEST_opened=False,VAL_previously_exposed=True,policy_changed=False,
        formal_flags_changed=False,frames=3445,screen_is_not_model_performance=True)
    for name,digest in pins.items():
        if name not in absent and sha(ROOT/name)!=digest: raise ValueError('Inputs changed during diagnostic')
    write(args.out/'report.json',report)
    write(args.out/'completion.json',dict(status=report['screen']['status'],frames=3445,fit_calls=0,TEST_opened=False))
    print(report['screen']['status'],flush=True)
    for role in ('TRAIN','VAL'):
        g=roles[role]['groups']['all'];print(role,g['total'],'FA regimes',g['FA_regimes'],flush=True)


if __name__=='__main__':main()
