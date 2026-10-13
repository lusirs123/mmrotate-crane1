#!/usr/bin/env python3
"""Independent labels/vectors/counts/size matches plus deterministic report audit."""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.tools import run_port_reliability_conditional_evidence_v1 as runner
from crane_project.utils import port_reliability_conditional_evidence_v1 as core


def vector(r):
    a,b=sorted(r['pred'][2:4],reverse=True);w,h=r['image_size']
    s=min(max(r['pred'][5],1e-6),1-1e-6)
    return [math.log(s)-math.log1p(-s),.5*(math.log(a)+math.log(b)-math.log(w)-math.log(h)),math.log(a)-math.log(b)]


def label(r):
    if r['pred'] is None:return None
    return any(abs(a-b)/b>.1 for a,b in zip(sorted(r['pred'][2:4]),sorted(r['gt'][2:4])))


def review(result,allow_absent=False):
    p,s,pins,absent=runner.load(allow_absent)
    report=json.loads((result/'report.json').read_text()); frozen=json.loads((result/'TRAIN_rules.json').read_text())
    if (report['input_pins']!=pins or not set(report['absent_frontend'])<=set(p['frontend_pins'])
            or (not allow_absent and report['absent_frontend'])):raise ValueError('Manifest differs')
    previous=ROOT/p['previous_result'];parts={role:runner.read(previous/('scored_'+role+'.jsonl.gz')) for role in ('TRAIN','VAL')}
    if frozen!=core.freeze_train(parts['TRAIN']):raise ValueError('TRAIN-only rules differ')
    checks=0
    for role,rows in parts.items():
        core.validate(rows,role)
        details={d['image']:d for d in report['roles'][role]['details']}
        if len(details)!=len(rows):raise ValueError('Diagnostic identities differ')
        for r in rows:
            y=label(r);d=details[r['image']]
            expected='MISSING' if y is None else ('FA' if y else 'CR') if r['risks']['fusion4']<=report['frozen_cutoff'] else ('ED' if y else 'FR')
            if expected!=d['state']:raise ValueError('Independent state differs')
            if y is not None and not np.allclose(vector(r),d['features'],rtol=0,atol=1e-12):raise ValueError('Independent vector differs')
            checks+=1
        for name,g in report['roles'][role]['groups'].items():
            selected=rows if name=='all' else [r for r in rows if r[name.split(':')[0]]==name.split(':')[1]]
            bs=sum(label(r) is True for r in selected);gs=sum(label(r) is False for r in selected)
            if (g['total']['bad'],g['total']['good'],g['total']['frames'])!=(bs,gs,len(selected)):raise ValueError('Independent counts differ')
            lookup={r['image']:r for r in selected}
            for pair in g['size_matched']['pairs']:
                a,b=lookup[pair['bad_image']],lookup[pair['good_image']]
                if label(a) is not True or label(b) is not False or a['sequence']!=b['sequence']:raise ValueError('Matched role/video invalid')
                distance=abs(vector(a)[1]-vector(b)[1])
                if distance>core.SETTINGS['size_match_log_tolerance'] or abs(distance-pair['size_distance'])>1e-12:raise ValueError('Size tolerance invalid')
            checks+=1
        # Deterministic full audit complements the independent contracts above.
        if report['roles'][role]!=core.analyze(rows,frozen,report['frozen_cutoff']):raise ValueError('Full conditional analysis differs')
    if report['screen']!=core.prerequisite_screen(report['roles']):raise ValueError('Screen differs')
    cutoff=json.loads((previous/'calibration.json').read_text())['fusion4']['risk_le']
    if report['frozen_cutoff']!=cutoff or any(report[k]!=0 for k in ('fit_calls','training_updates','detector_inferences')) or report['TEST_opened'] or report['policy_changed'] or report['formal_flags_changed']:
        raise ValueError('Diagnostic scope changed')
    return dict(passed=True,rows=sum(map(len,parts.values())),independent_checks=checks,
                report_sha256=runner.sha(result/'report.json'),absent_frontend=absent,
                deterministic_full_replay=True,TEST_opened=False,fit_calls=0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--allow-absent-frontend',action='store_true');a=p.parse_args()
    runner.write(a.out,review(a.result,a.allow_absent_frontend));print('Independent review passed',flush=True)


if __name__=='__main__':main()
