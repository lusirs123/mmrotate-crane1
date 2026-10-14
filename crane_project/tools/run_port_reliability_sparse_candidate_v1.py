#!/usr/bin/env python3
"""Read only pinned TRAIN caches; write a descriptive report in one run folder."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.utils import port_reliability_sparse_candidate_v1 as core


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')


def read(path):
    with gzip.open(path, 'rt') as f:
        return [json.loads(line) for line in f]


def load():
    p = json.loads((ROOT / ('crane_project/tools/' + core.VERSION + '_protocol.json')).read_text())
    sources = json.loads((ROOT / ('crane_project/tools/' + core.VERSION + '_sources.json')).read_text())
    if p['protocol'] != core.VERSION or p['features'] != list(core.FEATURES) or p['scope'] != 'TRAIN_CACHE_ONLY_NO_FIT':
        raise ValueError('Contract changed')
    parent = json.loads((ROOT / 'crane_project/tools/port_reliability_fallback_decision_v1_protocol.json').read_text())
    if p['cutoffs'] != parent['frozen_cutoffs']:
        raise ValueError('Frozen parent cutoffs changed')
    pins = dict(p['input_pins'], **sources['sources'])
    for name, digest in pins.items():
        if sha(ROOT / name) != digest:
            raise ValueError('Pinned input/source differs: ' + name)
    for name in ('candidate', 'reference'):
        base = ROOT / p[name + '_result']
        report = json.loads((base / 'report.json').read_text())
        review = json.loads((base / 'server_review.json').read_text())
        if not review['passed'] or review['report_sha256'] != sha(base / 'report.json'):
            raise ValueError('Parent result is not reviewed')
        if report['TEST_opened'] or report['formal_policy_changed'] or report['candidate_adopted']:
            raise ValueError('Parent evidence identity changed')
    return p, pins


def prepare(p):
    candidates = read(ROOT / p['candidate_result'] / 'candidates_TRAIN.jsonl.gz')
    reference = read(ROOT / p['reference_result'] / 'scored_TRAIN.jsonl.gz')
    if len(candidates) != 2558 or len(reference) != 2558:
        raise ValueError('Complete original TRAIN required')
    return core.join(candidates, reference, p['cutoffs'])


def run(args):
    p, pins = load()
    if args.check_only:
        prepare(p)
        print('TRAIN-only source and frozen-decision identities verified')
        return
    out = args.out
    if out is None or out.is_symlink() or out.name != 'result' or out.resolve().parent.parent != (ROOT / 'work_dirs' / core.VERSION).resolve():
        raise ValueError('Use work_dirs/version/RUN_ID/result')
    out.mkdir(parents=True, exist_ok=False)
    rows = prepare(p)
    report = dict(protocol=core.VERSION, input_pins=pins, cutoffs=p['cutoffs'],
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        rows=len(rows), result=core.analyze(rows), fits=0, calibration_calls=0,
        detector_inferences=0, detector_updates=0, quality_updates=0,
        VAL_opened=False, TEST_opened=False, formal_policy_changed=False, candidate_adopted=False)
    write(out / 'report.json', report)
    write(out / 'diagnostic_rows.json', rows)
    for path, digest in pins.items():
        if sha(ROOT / path) != digest:
            raise ValueError('Input/source changed during run')
    write(out / 'completion.json', dict(status=report['result']['status'], rows=len(rows)))
    print(report['result']['status'], 'rows', len(rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check-only', action='store_true')
    run(parser.parse_args())
