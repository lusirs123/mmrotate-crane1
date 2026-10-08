#!/usr/bin/env python3
"""One fixed eight-feature fit on cached TRAIN; no inference/VAL/TEST stage."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import run_port_reliability_scale_consistency_v1 as prior
from crane_project.utils import port_reliability_scale_asymmetry_v1 as method
from crane_project.utils import port_reliability_scale_consistency_v1 as old
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils import port_reliability_feature_ablation_v1 as ab

PROTOCOL = ROOT/'crane_project/tools/port_reliability_scale_asymmetry_v1_protocol.json'
SOURCES = ROOT/'crane_project/tools/port_reliability_scale_asymmetry_v1_sources.json'
CONTROL = 'work_dirs/port_reliability_scale_consistency_v1/20261008_paired_train_v1/train_check'
CONTROL_PINS = {'work_dirs/port_reliability_scale_consistency_v1/20261008_paired_train_v1/train_check/scored_TRAIN.jsonl.gz': 'f363f502f831a5396bb1782d8481d6e6e684fb67f616b14737f18eea68e6080f', 'work_dirs/port_reliability_scale_consistency_v1/20261008_paired_train_v1/train_check/models.json': '4e757fcad73a630cf625ed22d4551d2f4c729046e2eed9f3021fa800fa2489d2', 'work_dirs/port_reliability_scale_consistency_v1/20261008_paired_train_v1/train_check/report.json': '40e6b7c2090425b2a70ad7d19959531f6437f894bcca0ab2ff929fa075df1775', 'work_dirs/port_reliability_scale_consistency_v1/20261008_paired_train_v1/train_check/completion.json': 'd45af55f0b64e9a2a514e08724964e8174a8b9a73ef240555bd7a916f2c25f6f'}
SOURCE_FILES = prior.SOURCE_FILES+[
    'crane_project/tools/port_reliability_scale_consistency_v1_sources.json',
    'crane_project/utils/port_reliability_scale_asymmetry_v1.py',
    'crane_project/tools/run_port_reliability_scale_asymmetry_v1.py',
    'crane_project/tools/port_reliability_scale_asymmetry_v1_protocol.json',
    'tests/test_port_reliability_scale_asymmetry_v1.py',
    'tools/run_port_reliability_scale_asymmetry_v1.sh',
]
sha = prior.sha
write = prior.write


def checked_sources():
    protocol = json.loads(PROTOCOL.read_text()); manifest = json.loads(SOURCES.read_text())
    actual = {name: sha(ROOT/name) for name in SOURCE_FILES}
    if (manifest != dict(protocol=method.VERSION, sources=actual)
            or protocol['protocol'] != method.VERSION or protocol['features'] != list(method.FEATURES)
            or protocol['fitting'] != method.FIT or protocol['input_pins'] != prior.PINS
            or protocol['control_pins'] != CONTROL_PINS or protocol['controls'] != list(method.CONTROLS)):
        raise ValueError('Sealed source/protocol contract changed')
    return protocol, dict(manifest_sha256=sha(SOURCES), sources=actual)


def checked_control_artifacts():
    actual = {name: sha(ROOT/name) for name in CONTROL_PINS}
    if actual != CONTROL_PINS:
        raise ValueError('Frozen control artifacts changed')
    receipt = json.loads((ROOT/CONTROL/'completion.json').read_text())
    expected = {Path(name).name: digest for name, digest in CONTROL_PINS.items()
                if Path(name).name != 'completion.json'}
    if (receipt['protocol'] != old.VERSION or receipt['status'] != 'TRAIN_CAPABILITY_FAILED_STOP'
            or receipt['artifacts'] != expected or receipt['VAL_evaluated'] or receipt['TEST_read']
            or receipt['selected_arm'] is not None):
        raise ValueError('Frozen control receipt identity differs')
    return actual


def checked_controls(rows):
    pins = checked_control_artifacts()
    protocol, sources = prior.checked_sources()
    report = json.loads((ROOT/CONTROL/'report.json').read_text())
    models = json.loads((ROOT/CONTROL/'models.json').read_text())
    if (report['sources'] != sources or report['contract'] != protocol or report['models'] != models
            or report['input_sha256'] != prior.PINS or report['probe_gate']['passed']
            or report['VAL_evaluated'] or report['TEST_read']):
        raise ValueError('Frozen model/report source identity differs')
    with gzip.open(ROOT/CONTROL/'scored_TRAIN.jsonl.gz', 'rt') as f:
        saved = [json.loads(line) for line in f]
    by_id = {r['image']: r for r in saved}
    if len(by_id) != len(saved) or set(by_id) != {r['image'] for r in rows}:
        raise ValueError('Frozen control frame identities differ')
    feature_delta = 0.
    scores = {m: {} for m in method.CONTROLS}; max_delta = {m: 0. for m in method.CONTROLS}
    for r in rows:
        s = by_id[r['image']]
        if any(s.get(k) != v for k, v in r.items()):
            raise ValueError('Frozen control metadata/GT/main/aux/role identity differs')
        x = old.descriptor(r['pred'], r['auxiliary_pred_original'], r['image_size'])
        if (s['features'] is None) != (x is None):
            raise ValueError('Frozen five-feature missing readout differs')
        if x is not None:
            saved_x = np.asarray(s['features'], dtype=np.float64)
            if saved_x.shape != (5,) or not np.isfinite(saved_x).all():
                raise ValueError('Invalid frozen feature schema')
            delta = float(np.max(np.abs(saved_x-x)))
            feature_delta = max(feature_delta, delta)
            if delta > 1e-12:
                raise ValueError('Frozen five-feature readout differs')
        computed = dict(scale_consistency=None if x is None else float(old.risk(models['scale_consistency'], x)),
            refit_simple=None if x is None else float(old.risk(models['refit_simple'], x[:3])),
            full_simple=r['size_risks']['full_simple'], score_only=None if x is None else 1-r['pred'][5])
        for m in scores:
            stored = s['experiment_risks'][m]
            if (stored is None) != (computed[m] is None):
                raise ValueError('Frozen control missing risk differs')
            if stored is not None:
                delta = abs(stored-computed[m]); max_delta[m] = max(max_delta[m], delta)
                if delta > 1e-12:
                    raise ValueError('Frozen control risk replay differs')
            # Use exact original scores for cutoffs/ties. BLAS replay is audited,
            # never allowed to move a historical threshold member.
            scores[m][r['image']] = stored
    return models, scores, dict(artifact_sha256=pins, rows=len(saved),
        maximum_feature_replay_delta=feature_delta,
        maximum_risk_replay_delta=max_delta, thresholds_use_original_saved_scores=True,
        control_refits=0)


def engineering_check(features, labels):
    """Check the actual fixed objective and both derivative orders BEFORE fit."""
    x = np.asarray(features, dtype=np.float64); y = np.asarray(labels, dtype=np.float64)
    scale = x.std(axis=0); scale = np.where(scale < 1e-8, 1., scale)
    design = np.column_stack((np.ones(len(x)), (x-x.mean(axis=0))/scale))
    bad = int(y.sum()); good = len(y)-bad
    if (x.shape != (1942, 8) or bad != 54 or good != 1888 or not np.isfinite(design).all()):
        raise ValueError('Actual fit identity/class/feature contract changed')
    sw = np.where(y == 1., .5/bad, .5/good)
    objective = lambda w: simple.logistic_objective(w, design, y, sw, method.FIT['l2'])
    w = np.linspace(-.2, .2, 9); loss, grad, hessian = objective(w)
    numeric_g = np.zeros(9); numeric_h = np.zeros((9, 9)); eps = 1e-5
    for j in range(9):
        step = np.zeros(9); step[j] = eps
        a, b = objective(w+step), objective(w-step)
        numeric_g[j] = (a[0]-b[0])/(2*eps)
        numeric_h[:, j] = (a[1]-b[1])/(2*eps)
    gradient_delta = float(np.max(np.abs(grad-numeric_g)))
    hessian_delta = float(np.max(np.abs(hessian-numeric_h)))
    initial, zero_grad, zero_h = objective(np.zeros(9))
    direction = np.linalg.solve(zero_h, zero_grad)
    after = objective(-direction)[0]
    if (gradient_delta > 1e-7 or hessian_delta > 1e-7 or not after < initial
            or not np.isfinite(np.r_[zero_grad, hessian.ravel()]).all()
            or not np.any(np.abs(zero_grad[6:]) > 1e-10)):
        raise ValueError('Actual objective/gradient/Hessian/update check failed')
    return dict(passed=True, max_gradient_finite_difference_delta=gradient_delta,
        max_hessian_finite_difference_delta=hessian_delta,
        initial_objective=initial, one_zero_init_Newton_step_objective=after,
        zero_init_gradient=zero_grad.tolist(), GT_and_features_are_fixed=True,
        only_nine_candidate_coefficients_updated=True)


def preflight():
    protocol, sources = checked_sources()
    # Check the old receipt/bytes before loading its models or evaluating scores.
    checked_control_artifacts()
    rows, inputs = prior.load_inputs()
    models, controls, replay = checked_controls(rows)
    fit_rows = [r for r in rows if r['sample_role'] == 'fit' and r['pred'] is not None]
    features = [method.descriptor(r['pred'], r['auxiliary_pred_original'], r['image_size']) for r in fit_rows]
    labels = [ab.size_bad(r) for r in fit_rows]
    engineering = engineering_check(features, labels)
    for r in rows:
        x = method.descriptor(r['pred'], r['auxiliary_pred_original'], r['image_size'])
        a = old.descriptor(r['pred'], r['auxiliary_pred_original'], r['image_size'])
        if x is not None and not np.array_equal(x[:5], a):
            raise ValueError('Original five features were altered')
    support = {role: dict(Counter(method.combination(r) for r in rows if r['sample_role'] == role))
               for role in ('fit', 'probe', 'legacy', 'purged')}
    expected = dict(fit=[1718, 166, 36, 18, 4, 0], probe=[433, 44, 7, 8, 0, 0])
    for role, counts in expected.items():
        if [support[role].get(k, 0) for k in method.COMBINATIONS] != counts:
            raise ValueError('Predeclared main/auxiliary support changed')
    return rows, inputs, models, controls, features, labels, dict(
        contract=protocol, sources=sources, controls=replay,
        engineering=engineering, combination_support=support)


def checked_out(path):
    path = path.resolve()
    if path.parent.parent != (ROOT/'work_dirs'/method.VERSION).resolve() or path.name != 'train_check':
        raise ValueError('Use work_dirs/'+method.VERSION+'/RUN_ID/train_check')
    if path.exists():
        raise FileExistsError('Refuse to overwrite reliability evidence')
    return path


def run(out):
    out = checked_out(out)
    rows, inputs, controls_models, controls, features, labels, checks = preflight()
    before = simple.fingerprint(rows)
    # The only fit in this version. Controls are reused byte-pinned, never refit.
    candidate = method.fit(features, labels)
    models = dict(scale_asymmetry=candidate, frozen_controls=controls_models)
    scores = dict(controls); scores[method.ARM] = {}
    for r in rows:
        x = method.descriptor(r['pred'], r['auxiliary_pred_original'], r['image_size'])
        scores[method.ARM][r['image']] = None if x is None else float(method.risk(candidate, x))
    diagnostics = {}; points = {}; protections = {}; scored = []
    for role in ('fit', 'probe', 'legacy', 'purged'):
        subset = [r for r in rows if r['sample_role'] == role]
        point = method.cutoff(subset, scores[method.ARM]); points[role] = point
        diagnostics[role] = method.describe(subset, scores, point['risk_le'])
        decisions = {}
        for r in subset:
            d = method.decide(candidate, point['risk_le'], r['original_simple_decision'],
                              r['pred'], r['auxiliary_pred_original'], r['image_size'])
            expected = json.loads(json.dumps(r['original_simple_decision']))
            if r['pred'] is not None:
                expected['risks']['size'] = scores[method.ARM][r['image']]
                expected['size_accepted'] = scores[method.ARM][r['image']] <= point['risk_le']
            if d != expected:
                raise ValueError('Only size risk/flag may change; online/offline mismatch')
            decisions[r['image']] = d
            x = method.descriptor(r['pred'], r['auxiliary_pred_original'], r['image_size'])
            scored.append(dict(r, features=None if x is None else x.tolist(),
                experiment_risks={m: scores[m][r['image']] for m in scores},
                candidate_diagnostic_decision=d))
        protections[role] = prior.components(subset, decisions)
    gate = method.gate(diagnostics['probe'])
    if (simple.fingerprint(rows) != before or {name: sha(ROOT/name) for name in prior.PINS} != inputs
            or checked_control_artifacts() != checks['controls']['artifact_sha256']
            or checked_sources()[1] != checks['sources']):
        raise ValueError('Sealed input/source/control mutated')
    status = 'TRAIN_CAPABILITY_PASS_REQUIRES_SEPARATE_VAL_AUTHORIZATION' if gate['passed'] else 'TRAIN_CAPABILITY_FAILED_STOP'
    report = dict(protocol=method.VERSION, preflight=checks, input_sha256=inputs, models=models,
        TRAIN_roles=dict(Counter(r['sample_role'] for r in rows)), diagnostic_points=points,
        diagnostics=diagnostics, three_components=protections, probe_gate=gate,
        frontend='B24+sigma1.5/epoch03', original_policy_sha256=checks['contract']['original_policy_sha256'],
        exact_standard_replay_frames=len(rows), fit_calls=1, control_refits=0,
        detector_updates=0, midpoint_updates=0, new_inference_calls=0, GPU_used=False,
        VAL_evaluated=False, mixed_container_VAL_bytes_read=True, TEST_read=False,
        policy_modified=False, geometry_modified=False, depth_formula_modified=False,
        selected_arm=None, conclusion=status,
        future_stage='Stop if failed; even if passed, freeze artifacts then request a separately defined VAL stage. No automatic VAL/TEST/deployment.')
    out.mkdir(parents=True)
    write(out/'report.json', report); write(out/'models.json', models)
    with gzip.open(out/'scored_TRAIN.jsonl.gz', 'xt') as f:
        for r in sorted(scored, key=lambda x: x['image']):
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n')
    write(out/'completion.json', dict(protocol=method.VERSION, status=status,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True).strip(),
        selected_arm=None, VAL_evaluated=False, TEST_read=False,
        artifacts={p.name: sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(status=status, probe_gate=gate, out=str(out)), ensure_ascii=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--out', type=Path); group.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        checks = preflight()[-1]
        print(json.dumps(dict(status='ASYMMETRY_STATIC_AND_INPUT_CONTRACT_PASS', checks=checks), ensure_ascii=False))
    else:
        run(args.out)
