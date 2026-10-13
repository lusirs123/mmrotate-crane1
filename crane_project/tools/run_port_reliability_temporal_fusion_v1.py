#!/usr/bin/env python3
"""Fixed CPU-only full TRAIN fit -> exposed VAL calibration/evaluation."""
import argparse
import gzip
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import run_port_reliability_temporal_history_gate_v1 as loader
from crane_project.utils import port_reliability_temporal_fusion_v1 as core

PROTOCOL = ROOT/('crane_project/tools/'+core.VERSION+'_protocol.json')
SOURCES = ROOT/('crane_project/tools/'+core.VERSION+'_sources.json')


def load(verify_frontend=True):
    p = json.loads(PROTOCOL.read_text()); s = json.loads(SOURCES.read_text())
    if p['fitting'] != core.FITTING or p['history'] != 3 or p['target_good_retention'] != .95:
        raise ValueError('Fixed fitting contract changed')
    pins = dict(p['input_pins'], **s['sources'])
    if verify_frontend: pins.update(p['frozen_frontend_identity'])
    for name, digest in pins.items():
        if loader.sha(ROOT/name) != digest: raise ValueError('Input/source changed '+name)
    old, _, parts = loader.load(verify_frontend)
    for key in ('input_pins', 'frozen_frontend_identity', 'boundaries', 'rows', 'policy'):
        if p[key] != old[key]: raise ValueError('Reviewed data/front-end identity changed')
    return p, s, parts


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.is_symlink() or args.out.resolve().parent.parent != (ROOT/'work_dirs'/core.VERSION).resolve():
        raise ValueError('Use work_dirs/version/RUN_ID/result')
    protocol, sources, parts = load(); args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    pins = dict(protocol['input_pins'], **protocol['frozen_frontend_identity'])
    before = {p:loader.sha(ROOT/p) for p in pins}
    train = core.level.enrich(parts['TRAIN'], protocol['boundaries'])
    models = core.fit(train)
    loader.write(args.out/'models.json', models)  # Frozen before VAL features/calibration.
    old = core.level.fit_cutoff(train)
    if abs(old['risk_le']-core.LEVEL_CUTOFF) > 1e-15: raise ValueError('Old rule changed')
    scored = {'TRAIN':core.score(train, models)}
    scored['VAL'] = core.score(core.level.enrich(parts['VAL'], protocol['boundaries']), models)
    cutoffs = {a:core.select_cutoff(scored['VAL'], a) for a in core.ARMS}
    loader.write(args.out/'calibration.json', cutoffs)
    roles = {role:core.evaluate(rows, cutoffs) for role, rows in scored.items()}
    gate = core.gate(roles['VAL'])
    for role, rows in scored.items():
        with gzip.open(args.out/('scored_'+role+'.jsonl.gz'), 'xt') as file:
            for row in rows: file.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
        c = roles[role]['groups']['all']['candidate']
        print(role, 'frames', len(rows), 'candidate', c['states'], flush=True)
    if before != {p:loader.sha(ROOT/p) for p in pins}: raise ValueError('Frozen inputs modified')
    status = 'EVIDENCE_PASS_NO_ADOPTION' if gate['passed'] else 'EVIDENCE_FAILED_STOP'
    report = dict(protocol=core.VERSION, status=status, roles=roles, models=models, cutoffs=cutoffs,
        gate=gate, old_level_cutoff=old, input_pins=before, sources=sources,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        fit_calls=3, detector_inferences=0, neural_training_updates=0, TEST_opened=False,
        formal_policy_changed=False, candidate_adopted=False,
        models_saved_before_VAL_calibration=True, elapsed_seconds=time.monotonic()-started,
        time_units='sampled observations, not elapsed seconds',
        fallback='shared full-TRAIN static model; arm cutoffs separately VAL-calibrated',
        whole_VAL_used=True, VAL_previously_exposed=True)
    loader.write(args.out/'report.json', report)
    loader.write(args.out/'completion.json', dict(status=status, rows=3445, TEST_opened=False))
    print(status, flush=True)


if __name__ == '__main__': main()
