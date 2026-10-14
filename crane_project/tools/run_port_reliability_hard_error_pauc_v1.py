#!/usr/bin/env python3
"""Fixed paired TRAIN objective fit, then one full exposed VAL calibration."""
import argparse
import gzip
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.utils import port_reliability_hard_error_pauc_v1 as core
from crane_project.tools import run_port_reliability_conditional_evidence_v1 as io


def load(allow_absent=False):
    p = json.loads((ROOT / ('crane_project/tools/' + core.VERSION + '_protocol.json')).read_text())
    sources = json.loads((ROOT / ('crane_project/tools/' + core.VERSION + '_sources.json')).read_text())
    if p['settings'] != core.SETTINGS or p['fallback_cutoff'] != core.FALLBACK_CUTOFF or p['fusion_cutoff'] != core.FUSION_CUTOFF:
        raise ValueError('Prespecified contract changed')
    pins = dict(p['input_pins'], **p['frontend_pins'], **sources['sources']); absent = []
    for name, digest in pins.items():
        if allow_absent and name in p['frontend_pins'] and not (ROOT / name).exists():
            absent.append(name); continue
        if io.sha(ROOT / name) != digest: raise ValueError('Source/input differs: ' + name)
    reports = {}
    for key in ('pauc_result', 'reference_result'):
        base = ROOT / p[key]; report = json.loads((base / 'report.json').read_text())
        review = json.loads((base / 'server_review.json').read_text())
        if not review['passed'] or review['report_sha256'] != io.sha(base / 'report.json'):
            raise ValueError('Unreviewed parent')
        if report['TEST_opened'] or report['candidate_adopted'] or report['formal_policy_changed']:
            raise ValueError('Parent identity changed')
        reports[key] = report
    return p, pins, absent, reports['pauc_result']


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path)
    parser.add_argument('--check-only', action='store_true'); parser.add_argument('--allow-absent-frontend', action='store_true')
    args = parser.parse_args(); p, pins, absent, previous = load(args.allow_absent_frontend)
    if args.check_only:
        print('Sources, parent identities and fixed contract verified; no fit'); return
    if absent: raise ValueError('Actual fit requires server frontend SHA verification')
    out = args.out
    if out is None or out.is_symlink() or out.name != 'result' or out.resolve().parent.parent != (ROOT / 'work_dirs' / core.VERSION).resolve():
        raise ValueError('Use work_dirs/version/RUN_ID/result')
    out.mkdir(parents=True, exist_ok=False)
    train = io.read(ROOT / p['reference_result'] / 'scored_TRAIN.jsonl.gz')
    if len(train) != 2558 or len({r['image'] for r in train}) != 2558: raise ValueError('Full unique TRAIN required')
    base_models = previous['base_models']; hard = core.hard_set(train)
    io.write(out / 'hard_TRAIN.json', hard)  # Before fitting/reading VAL.
    models = {core.CONTROL: core.fit(train, base_models, hard, False), core.METHOD: core.fit(train, base_models, hard, True)}
    import numpy as np
    if not np.allclose(models[core.CONTROL]['weights'], previous['model']['weights'], atol=1e-11, rtol=0):
        raise ValueError('Uniform control did not reproduce original partial-AUC')
    io.write(out / 'models.json', models)  # Frozen before parsing VAL rows.
    scored = {'TRAIN': core.score(train, models)}
    val = io.read(ROOT / p['reference_result'] / 'scored_VAL.jsonl.gz')
    if len(val) != 887 or len({r['image'] for r in val}) != 887 or any(r['split'] != 'val' or r['reliability_role'] != 'val' for r in val):
        raise ValueError('Full unique VAL required')
    if set(r['image'] for r in train) & set(r['image'] for r in val): raise ValueError('Role overlap')
    scored['VAL'] = core.score(val, models)
    cutoffs = {m: core.calibrate(scored['VAL'], m) for m in models}
    io.write(out / 'calibration.json', cutoffs)
    evaluations = {role: core.evaluate(rows, cutoffs) for role, rows in scored.items()}
    for role, rows in scored.items():
        with gzip.open(out / ('scored_' + role + '.jsonl.gz'), 'xt') as f:
            for row in rows: f.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        print(role, {m: evaluations[role]['groups']['all'][m]['candidate']['states'] for m in models}, flush=True)
    gate = core.gate(evaluations['VAL'])
    for name, digest in pins.items():
        if io.sha(ROOT / name) != digest: raise ValueError('Frozen source/input changed')
    report = dict(protocol=core.VERSION, settings=core.SETTINGS, input_pins=pins, absent_frontend=[],
        base_models=base_models, models=models, hard_TRAIN=hard, calibration=cutoffs, roles=evaluations, gate=gate,
        status='EVIDENCE_PASS_NO_ADOPTION' if gate['passed'] else 'EVIDENCE_FAILED_STOP',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        model_saved_before_VAL=True, hard_set_saved_before_fit_and_VAL=True, fit_calls=2,
        detector_inferences=0, detector_updates=0, neural_training_updates=0,
        VAL_previously_exposed=True, TEST_opened=False, formal_policy_changed=False, candidate_adopted=False,
        limitations=['Four inputs add no independent image evidence.', 'TRAIN hard mining uses in-training predictions, not held-out errors.',
            '2x weighting and 25% fraction are prespecified one-shot engineering constants.',
            'No claim that scalar gradient weight or biased smooth pAUC guarantees lower FA.',
            'Matched branch-locked comparisons are offline GT diagnostics, not online threshold selection.'])
    io.write(out / 'report.json', report)
    io.write(out / 'completion.json', dict(status=report['status'], rows=3445, TEST_opened=False))
    print(report['status'], 'hard errors', len(hard['images']), 'failed entries', len(gate['failures']))


if __name__ == '__main__': main()
