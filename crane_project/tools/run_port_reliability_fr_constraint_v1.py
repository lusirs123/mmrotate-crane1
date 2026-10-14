#!/usr/bin/env python3
"""Complete frozen TRAIN; internal video holdout only, CPU."""
import argparse
import gzip
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import run_port_reliability_video_transfer_v1 as previous_runner
from crane_project.utils import port_reliability_fr_constraint_v1 as core


parent = previous_runner.parent

def load(allow_absent=False):
    p = json.loads((ROOT/('crane_project/tools/'+core.VERSION+'_protocol.json')).read_text())
    sources = json.loads((ROOT/('crane_project/tools/'+core.VERSION+'_sources.json')).read_text())
    if p['settings'] != core.SETTINGS or p['protocol'] != core.VERSION:
        raise ValueError('Diagnostic contract changed')
    for n, h in sources['sources'].items():
        if parent.io.sha(ROOT/n) != h:
            raise ValueError('Diagnostic source changed: '+n)
    previous, pins, absent, rows, policy = previous_runner.load(allow_absent)
    return p, dict(pins, **sources['sources']), absent, rows, policy


def run(args):
    p, pins, absent, rows, policy = load(args.allow_absent_frontend)
    if args.check_only:
        print('Complete frozen TRAIN/history/simple/source verified; no fitting')
        return
    if absent:
        raise ValueError('Actual run requires both formal frontend SHA checks')
    out = args.out
    if (out is None or out.is_symlink() or out.name != 'result' or
            out.resolve().parent.parent != (ROOT/'work_dirs'/core.VERSION).resolve()):
        raise ValueError('Use unified work_dirs/version/RUN_ID/result')
    out.mkdir(parents=True, exist_ok=False)
    result, held, reference = core.audit(rows, policy['models']['size'])
    report = dict(protocol=core.VERSION, settings=core.SETTINGS, result=result,
                  input_pins=pins, frames=len(rows), source_commit=subprocess.check_output(
                      ['git', 'rev-parse', 'HEAD'], cwd=ROOT, universal_newlines=True).strip(),
                  formal_policy_changed=False, geometry_changed=False, output_identity_changed=False,
                  historical_VAL_TEST_exposure_disclosed=True, GT_offline_only=True,
                  limits=p['limitations'])
    for name, h in pins.items():
        if parent.io.sha(ROOT/name) != h:
            raise ValueError('Input/source changed during run')
    parent.io.write(out/'report.json', report)
    for name, values in [('held_TRAIN.jsonl.gz', held), ('reference_TRAIN.jsonl.gz', reference)]:
        with gzip.open(out/name, 'wt') as f:
            for r in values:
                f.write(json.dumps(r, sort_keys=True, allow_nan=False)+'\n')
    parent.io.write(out/'completion.json', dict(status=result['status'],
        report_sha256=parent.io.sha(out/'report.json'), frames=len(rows)))
    for seq, fold in result['folds'].items():
        data = fold['held_evaluation']['all']['methods']
        print(seq, {m: dict(states=data[m]['delivery']['states'], AUROC=data[m]['AUROC'],
                           FA_correct95=data[m]['offline_correct95']['states']['FA']) for m in core.SETTINGS['arms']})


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--out', type=Path)
    a.add_argument('--check-only', action='store_true')
    a.add_argument('--allow-absent-frontend', action='store_true')
    run(a.parse_args())
