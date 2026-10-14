#!/usr/bin/env python3
"""Fixed-budget TRAIN count experiment; reviewed old control is reused."""
import argparse
import gzip
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.tools import run_port_reliability_fr_constraint_v1 as previous
from crane_project.utils import port_reliability_actual_fr_v1 as core
parent=previous.parent
OLD='work_dirs/port_reliability_fr_constraint_v1/20261014_fr_constraint_v1/result/'
OLD_SHA='11222fe2f7be88bd14363f512050cd7ad737ff743d03ef30780e3e0d12261029'


def load(allow_absent=False):
    p=json.loads((ROOT/('crane_project/tools/'+core.VERSION+'_protocol.json')).read_text())
    s=json.loads((ROOT/('crane_project/tools/'+core.VERSION+'_sources.json')).read_text())
    if p['settings']!=core.SETTINGS or p['protocol']!=core.VERSION:raise ValueError('Changed registered contract')
    for n,h in s['sources'].items():
        if parent.io.sha(ROOT/n)!=h:raise ValueError('Changed source: '+n)
    _,pins,absent,rows,policy=previous.load(allow_absent)
    if parent.io.sha(ROOT/(OLD+'report.json'))!=OLD_SHA:raise ValueError('Reviewed old control source changed')
    rev=json.loads((ROOT/(OLD+'server_review.json')).read_text())
    if not rev['passed'] or rev['report_sha256']!=OLD_SHA:raise ValueError('Old control not independently reviewed')
    old=json.loads((ROOT/(OLD+'report.json')).read_text())['result']
    return p,dict(pins,**s['sources'],**{OLD+'report.json':OLD_SHA}),absent,rows,policy,old


def run(args):
    p,pins,absent,rows,policy,old=load(args.allow_absent_frontend)
    if args.check_only:print('Sources, original TRAIN and reviewed controls checked; no fitting');return
    if absent:raise ValueError('Server fitting needs formal frontend weight checks')
    out=args.out
    if out is None or out.is_symlink() or out.name!='result' or out.resolve().parent.parent!=(ROOT/'work_dirs'/core.VERSION).resolve():raise ValueError('Use version/run_id/result')
    out.mkdir(parents=True,exist_ok=False)
    parent.io.write(out/'frozen_protocol.json',p)
    result,held,reference=core.audit(rows,old)
    report=dict(protocol=core.VERSION,settings=core.SETTINGS,result=result,frames=len(rows),input_pins=pins,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,universal_newlines=True).strip(),
        formal_policy_changed=False,geometry_changed=False,output_identity_changed=False,
        GT_offline_only=True,historical_VAL_TEST_exposure_disclosed=True,limits=p['limitations'])
    for n,h in pins.items():
        if parent.io.sha(ROOT/n)!=h:raise ValueError('Changed inputs during fitting')
    parent.io.write(out/'report.json',report)
    for n,data in [('held_TRAIN.jsonl.gz',held),('reference_TRAIN.jsonl.gz',reference)]:
        with gzip.open(out/n,'wt') as f:
            for r in data:f.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')
    parent.io.write(out/'completion.json',dict(status=result['status'],report_sha256=parent.io.sha(out/'report.json')))
    print('TRAIN gate',result['gate'],flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',type=Path);a.add_argument('--check-only',action='store_true')
    a.add_argument('--allow-absent-frontend',action='store_true');run(a.parse_args())
