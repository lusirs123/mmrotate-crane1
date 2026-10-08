#!/usr/bin/env python3
"""Paired frozen TRAIN predictions -> bounded size reliability capability check."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.utils import port_reliability_scale_consistency_v1 as method
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils import port_reliability_feature_ablation_v1 as ab
from crane_project.utils import port_reliability_separability_v1 as sep
from crane_project.utils import port_reliability_state_continuity_v1 as states

PROTOCOL = ROOT/'crane_project/tools/port_reliability_scale_consistency_v1_protocol.json'
SOURCES = ROOT/'crane_project/tools/port_reliability_scale_consistency_v1_sources.json'
PAIRED = 'work_dirs/port_geometry_size_boundary_v1/port_geometry_size_boundary_continuous_v1_20261007_154944_1078779'
PRIMARY = 'work_dirs/port_reliability_feature_ablation_v1/20261008_size_features_v1/fit'
PINS = {
    PAIRED+'/finite/boundary_continuous/final_rows.jsonl': '34c0df37f49a36027377a1a950886d8628e0aca0c34d1082882fcef3bb10a8fc',
    PAIRED+'/prepare/plan.json': '74e67df8810205a24fae8daeeaa2de75d644de2e9b53e0f9c5aca927e75b5ba7',
    PRIMARY+'/scored_TRAIN_VAL.jsonl.gz': 'e804241c2b7d529ed816f9d2f33dedad1e2961a5b132426bc816777b324c03b4',
}
SOURCE_FILES = [
    'crane_project/utils/port_reliability_scale_consistency_v1.py',
    'crane_project/tools/run_port_reliability_scale_consistency_v1.py',
    'crane_project/tools/port_reliability_scale_consistency_v1_protocol.json',
    'crane_project/utils/port_simple_component_reliability_v1.py',
    'crane_project/utils/port_reliability_feature_ablation_v1.py',
    'crane_project/utils/port_reliability_tradeoff_v1.py',
    'crane_project/utils/port_reliability_separability_v1.py',
    'crane_project/utils/port_reliability_state_continuity_v1.py',
    'crane_project/utils/port_midpoint_sigma15_reliability_v1.py',
    'crane_project/tools/preflight_port_geometry_g_v1.py',
    'crane_project/tools/train_port_geometry_midpoint_formal_v1.py',
    'mmrotate/datasets/pipelines/port_train_augment.py',
    'tests/test_port_reliability_scale_consistency_v1.py',
    'tools/run_port_reliability_scale_consistency_v1.sh',
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def checked_sources():
    protocol = json.loads(PROTOCOL.read_text())
    manifest = json.loads(SOURCES.read_text())
    actual = {name: sha(ROOT/name) for name in SOURCE_FILES}
    if (manifest != dict(protocol=method.VERSION, sources=actual)
            or protocol['protocol'] != method.VERSION
            or protocol['features'] != list(method.FEATURES)
            or protocol['fitting'] != method.FIT or protocol['input_pins'] != PINS):
        raise ValueError('Source/protocol contract changed')
    return protocol, dict(manifest_sha256=sha(SOURCES), sources=actual)


def pair_train(primary, paired, plan):
    """Verify standard replay and original-coordinate correspondence first."""
    states.checked_order(primary)
    if any(r['reliability_role'] != 'train' for r in primary):
        raise ValueError('Only original TRAIN identities may enter capability check')
    ids = {r['image'] for r in primary}
    role_map = {r['image']: r['sample_role'] for r in plan['records']}
    if len(role_map) != len(plan['records']) or set(role_map) != ids:
        raise ValueError('Diagnostic role plan identities differ')
    views = {}
    for r in paired:
        key = r['image'], r['scale']
        if (key in views or r['image'] not in ids or r['scale'] not in (1., .5)
                or r['split'] not in ('train', 'train_sim')
                or r['sample_role'] != role_map[r['image']]):
            raise ValueError('Duplicate/unexpected paired TRAIN view or role')
        views[key] = r
    if len(views) != 2*len(ids):
        raise ValueError('One original and one auxiliary view required for every TRAIN frame')
    result = []
    for original in sorted(primary, key=lambda r: r['image']):
        a, b = views[original['image'], 1.], views[original['image'], .5]
        for view in (a, b):
            if (any(view[k] != original[k] for k in ('image', 'sequence', 'domain', 'frame_id', 'split'))
                    or not all(abs(x-y) <= 2e-6 for x, y in zip(view['gt'], original['gt']))
                    or len(view['gt']) != len(original['gt'])):
                raise ValueError('Paired metadata/GT identity differs')
            simple.prediction(view['midpoint'])
        if a['midpoint'] != original['pred']:
            raise ValueError('Standard M does not exactly replay frozen reliability frontend')
        decision = original['original_simple_decision']
        if (decision['final_box_original'] != original['pred']
                or decision['center_accepted'] != (original['pred'] is not None)
                or decision['risks']['size'] != original['size_risks']['full_simple']):
            raise ValueError('Original geometry/decision differs')
        # Never read candidate/edge_residual/boundary fields of the failed head.
        result.append(dict(original, sample_role=role_map[original['image']],
                           auxiliary_pred_original=b['midpoint']))
    return result


def load_inputs():
    actual = {name: sha(ROOT/name) for name in PINS}
    if actual != PINS:
        raise ValueError('Reviewed input bytes changed')
    # The gzip is a sealed mixed TRAIN/VAL container. Its bytes and row role
    # are read; VAL records are discarded and never scored/fitted/evaluated.
    with gzip.open(ROOT/(PRIMARY+'/scored_TRAIN_VAL.jsonl.gz'), 'rt') as f:
        primary = [r for r in map(json.loads, f) if r['reliability_role'] == 'train']
    with (ROOT/(PAIRED+'/finite/boundary_continuous/final_rows.jsonl')).open() as f:
        paired = [json.loads(line) for line in f]
    plan = json.loads((ROOT/(PAIRED+'/prepare/plan.json')).read_text())
    rows = pair_train(primary, paired, plan)
    expected = Counter({('fit', 'real'): 1377, ('fit', 'sim'): 565,
                        ('probe', 'real'): 351, ('probe', 'sim'): 141,
                        ('legacy', 'real'): 32, ('legacy', 'sim'): 32,
                        ('purged', 'real'): 50, ('purged', 'sim'): 10})
    if len(rows) != 2558 or Counter((r['sample_role'], r['domain']) for r in rows) != expected:
        raise ValueError('Original fixed diagnostic roles/counts changed')
    return rows, actual


def score_rows(rows, models):
    scores = {m: {} for m in ('scale_consistency',)+method.CONTROLS}
    for r in rows:
        image = r['image']
        if r['pred'] is None:
            for v in scores.values(): v[image] = None
            continue
        features = method.descriptor(r['pred'], r['auxiliary_pred_original'], r['image_size'])
        scores['scale_consistency'][image] = float(method.risk(models['scale_consistency'], features))
        scores['refit_simple'][image] = float(method.risk(models['refit_simple'], features[:3]))
        scores['full_simple'][image] = r['size_risks']['full_simple']
        scores['score_only'][image] = 1-r['pred'][5]
    return scores


def components(rows, decisions):
    result = {}
    for name, group in ab.groups(rows).items():
        out = [r for r in group if r['pred'] is not None]
        hits = sum(simple.geometry_errors(r['gt'], r['pred'])['center_px'] < 15 for r in out)
        stats = {}
        for component in simple.COMPONENTS:
            stats[component] = simple._component_stats(group, component,
                {r['image'] for r in group if decisions[r['image']][component+'_accepted']})
        result[name] = dict(frames=len(group), outputs=len(out), center_hits_on_outputs=hits,
            center_hit_rate_on_outputs=sep.ratio(hits, len(out)),
            output_coverage=sep.ratio(len(out), len(group)),
            all_frame_center_correct_coverage=sep.ratio(hits, len(group)), components=stats)
    return result


def checked_out(path):
    path = path.resolve()
    if path.parent.parent != (ROOT/'work_dirs'/method.VERSION).resolve() or path.name != 'train_check':
        raise ValueError('Use work_dirs/'+method.VERSION+'/RUN_ID/train_check')
    if path.exists():
        raise FileExistsError('Refuse to overwrite reliability evidence')
    return path


def run(out):
    out = checked_out(out)
    protocol, sources = checked_sources()
    rows, inputs = load_inputs()
    before = simple.fingerprint(rows)
    fit_rows = [r for r in rows if r['sample_role'] == 'fit' and r['pred'] is not None]
    features = [method.descriptor(r['pred'], r['auxiliary_pred_original'], r['image_size']) for r in fit_rows]
    labels = [ab.size_bad(r) for r in fit_rows]
    models = dict(scale_consistency=method.fit(features, labels),
                  refit_simple=method.fit([x[:3] for x in features], labels))
    diagnostics = {}; points = {}; scored = []; protections = {}
    for role in ('fit', 'probe', 'legacy', 'purged'):
        subset = [r for r in rows if r['sample_role'] == role]
        scores = score_rows(subset, models)
        point = method.cutoff(subset, scores['scale_consistency'])
        points[role] = point
        diagnostics[role] = method.describe(subset, scores, point['risk_le'])
        decisions = {}
        for r in subset:
            d = method.decide(models['scale_consistency'], point['risk_le'],
                r['original_simple_decision'], r['pred'], r['auxiliary_pred_original'], r['image_size'])
            base = r['original_simple_decision']
            if (any(d[k] != base[k] for k in ('final_box_original', 'center_accepted', 'angle_accepted'))
                    or d['risks']['angle'] != base['risks']['angle']
                    or d['risks']['size'] != scores['scale_consistency'][r['image']]):
                raise ValueError('Online/offline mismatch or frozen component changed')
            decisions[r['image']] = d
            feature = method.descriptor(r['pred'], r['auxiliary_pred_original'], r['image_size'])
            scored.append(dict(r, features=None if feature is None else feature.tolist(),
                experiment_risks={m: v[r['image']] for m, v in scores.items()},
                candidate_diagnostic_decision=d))
        protections[role] = components(subset, decisions)
    gate = method.gate(diagnostics['probe'])
    if simple.fingerprint(rows) != before or {name: sha(ROOT/name) for name in PINS} != inputs:
        raise ValueError('Input evidence mutated')
    report = dict(protocol=method.VERSION, contract=protocol, sources=sources, input_sha256=inputs,
        frontend=dict(name='B24+sigma1.5/epoch03', original_policy_sha256=protocol['original_policy_sha256'],
                      exact_standard_replay_frames=len(rows)), models=models,
        TRAIN_roles=Counter(r['sample_role'] for r in rows),
        paired_missing={role: dict(main=sum(r['pred'] is None for r in rows if r['sample_role']==role),
                                  auxiliary=sum(r['auxiliary_pred_original'] is None for r in rows if r['sample_role']==role))
                        for role in points}, diagnostic_points=points, diagnostics=diagnostics,
        three_components=protections, probe_gate=gate, selected_arm=None,
        detector_updates=0, midpoint_updates=0, new_inference_calls=0, GPU_used=False,
        VAL_evaluated=False, mixed_container_VAL_bytes_read=True, TEST_read=False,
        policy_modified=False, geometry_modified=False, depth_formula_modified=False,
        conclusion='TRAIN_CAPABILITY_PASS_REQUIRES_VAL' if gate['passed'] else 'TRAIN_CAPABILITY_FAILED_STOP',
        future_stage='Only if TRAIN passes: separately collect one fixed half-scale VAL auxiliary, fit no further models; predeclared VAL adoption gates in contract. No automatic TEST or deployment.')
    out.mkdir(parents=True)
    write(out/'report.json', report)
    write(out/'models.json', models)
    with gzip.open(out/'scored_TRAIN.jsonl.gz', 'xt') as f:
        for r in sorted(scored, key=lambda x: x['image']):
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n')
    write(out/'completion.json', dict(protocol=method.VERSION, status=report['conclusion'],
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True).strip(),
        selected_arm=None, VAL_evaluated=False, TEST_read=False,
        artifacts={p.name: sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(status=report['conclusion'], probe_gate=gate, out=str(out)), ensure_ascii=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
