#!/usr/bin/env python3
"""One frozen TRAIN/VAL delivery correction; zero fits/calibration/TEST reads."""
import argparse
import gzip
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.utils import port_reliability_fallback_decision_v1 as core
from crane_project.tools import run_port_reliability_image_temporal_crossfit_v1 as old
io = old.io


def load(allow_absent=False):
    protocol = json.loads((ROOT / ('crane_project/tools/' + core.VERSION + '_protocol.json')).read_text())
    _, original_pins, absent, partial_report = old.load(allow_absent)
    cross = ROOT / protocol['crossfit_result']
    report = json.loads((cross / 'report.json').read_text())
    if report['calibration'] != json.loads((cross / 'calibration.json').read_text()):
        raise ValueError('Parent saved calibration differs')
    if report['TEST_opened'] or report['candidate_adopted'] or not report['model_saved_before_VAL']:
        raise ValueError('Unexpected frozen experiment identity')
    pins = dict(original_pins, **report['input_pins'], **protocol['input_pins'])
    sources = json.loads((ROOT / ('crane_project/tools/' + core.VERSION + '_sources.json')).read_text())['sources']
    pins.update(sources)
    for name, digest in pins.items():
        if name not in absent and io.sha(ROOT / name) != digest:
            raise ValueError('Pinned source/input changed: ' + name)
    for review_name in ('server_review.json', 'mac_review.json'):
        review = json.loads((cross / review_name).read_text())
        if not review['passed'] or review['report_sha256'] != io.sha(cross / 'report.json'):
            raise ValueError('Frozen parent has no valid independent review')
    cuts = dict(supported=report['calibration']['crossfit_fusion']['risk_le'],
                fallback=partial_report['calibration']['risk_le'])
    if cuts != protocol['frozen_cutoffs']:
        raise ValueError('Frozen cutoff identity differs')
    return protocol, pins, absent, cuts


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path)
    p.add_argument('--check-only', action='store_true'); p.add_argument('--allow-absent-frontend', action='store_true')
    a = p.parse_args(); protocol, pins, absent, cuts = load(a.allow_absent_frontend)
    if a.check_only:
        print('Frozen inputs/source/cutoffs verified; zero fitting'); return
    if absent:
        raise ValueError('Server evaluation requires actual frontend identities')
    if a.out is None or a.out.is_symlink() or a.out.resolve().parent.parent != (ROOT / 'work_dirs' / core.VERSION).resolve():
        raise ValueError('Use work_dirs/version/RUN_ID/result')
    a.out.mkdir(parents=True, exist_ok=False)
    evaluations = {}
    for role, count in (('TRAIN', 2558), ('VAL', 887)):
        rows = io.read(ROOT / protocol['crossfit_result'] / ('scored_' + role + '.jsonl.gz'))
        if len(rows) != count or len({r['image'] for r in rows}) != count:
            raise ValueError('Complete unique role required')
        if any(r['reliability_role'] != role.lower() or r['split'] not in
               (('train', 'train_sim') if role == 'TRAIN' else ('val',)) for r in rows):
            raise ValueError('Unexpected dataset role')
        scored = core.score(rows, cuts['supported'], cuts['fallback'])
        evaluations[role] = core.evaluate(scored, cuts['supported'], cuts['fallback'])
        with gzip.open(a.out / ('scored_' + role + '.jsonl.gz'), 'xt') as f:
            for row in scored:
                f.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        print(role, evaluations[role]['groups']['all']['candidate']['states'], flush=True)
    gate = core.gate(evaluations['VAL'])
    report = dict(protocol=core.VERSION, roles=evaluations, gate=gate, frozen_cutoffs=cuts,
        input_pins=pins, source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        status='EVIDENCE_PASS_NO_ADOPTION' if gate['passed'] else 'EVIDENCE_FAILED_STOP',
        fits=0, quality_updates=0, calibration_calls=0, detector_inferences=0, detector_updates=0,
        TEST_opened=False, VAL_previously_exposed=True, design_informed_by_exposed_VAL=True,
        formal_policy_changed=False, candidate_adopted=False,
        scope='Only unsupported-history cutoff restored; supported scores and decisions unchanged. No unified calibrated probability or new AUROC claim.')
    for name, digest in pins.items():
        if io.sha(ROOT / name) != digest:
            raise ValueError('Frozen input/source modified during evaluation')
    io.write(a.out / 'report.json', report)
    print(report['status'], flush=True)


if __name__ == '__main__':
    main()
