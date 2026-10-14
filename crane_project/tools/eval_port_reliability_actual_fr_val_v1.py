#!/usr/bin/env python3
"""Frozen full-TRAIN count-constrained candidate, descriptive VAL supplement."""
import argparse
from copy import deepcopy
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.utils import port_reliability_video_transfer_v1 as metrics
from crane_project.utils import port_reliability_fr_constraint_v1 as scoring
from crane_project.tools import review_port_reliability_video_transfer_v1 as independent

VERSION = 'port_reliability_actual_fr_val_v1'
TRAIN = 'work_dirs/port_reliability_actual_fr_v1/20261014_actual_fr_fix1/result/'
VAL = 'work_dirs/port_reliability_temporal_stack_test_v1/20261014_frozen_stack_test_v1/result/'
POLICY = 'work_dirs/port_midpoint_sigma15_reliability_v1_cachefix1/fit/policy.json'
PINS = {
    TRAIN+'report.json': 'aa1fe6341cde0dc928bc009b4d02ba975613d133a3496ab34ee770882a3a6c08',
    VAL+'scored_VAL.jsonl.gz': '7bcd34372fe4bb3095b513dd01f8d92c5d8ea200398627d9d6ee9ea3c0b7341b',
    POLICY: '38d914f114fcdb257f33f1cd6390ebcb4e103adfa1d9022628875e137c916e1f',
}
METHODS = ('raw', 'formal_simple', 'protected_simple', 'actual_count_zero', 'score_train95')
RISKS = ('full_simple', 'score_only', 'actual_count_logit')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   indent=2, allow_nan=False)+'\n')


def freeze():
    for name, digest in PINS.items():
        if sha(ROOT/name) != digest:
            raise ValueError('Changed frozen source: '+name)
    review = read(ROOT/(TRAIN+'mac_review.json'))
    if not review['passed'] or review['report_sha256'] != PINS[TRAIN+'report.json']:
        raise ValueError('Candidate lacks matching passed review')
    previous = read(ROOT/(TRAIN+'report.json'))['result']
    ref = previous['reference']
    candidate = ref['models']['actual_count3']
    if not candidate['solver']['incumbent']:
        raise ValueError('Full TRAIN model is a fallback reference')
    policy = read(ROOT/POLICY)['simple_policy']
    control = ref['models']['control3']
    for key in ('mean', 'scale', 'weights'):
        if not np.allclose(control[key], policy['models']['size'][key], atol=1e-12, rtol=0):
            raise ValueError('Control differs from formal simple')
    return dict(candidate_model=deepcopy(candidate), control_model=deepcopy(control),
                candidate_logit_lt=0., formal_risk_le=policy['cutoffs']['simple']['size']['risk_le'],
                protected_risk_le=ref['thresholds']['full_simple']['risk_le'],
                score_risk_le=ref['thresholds']['score_only']['risk_le'],
                prior_TRAIN_gate=deepcopy(previous['gate']),
                original_solver_status=deepcopy(candidate['solver']))


def apply(rows, frozen):
    """No GT/sequence identity enters scoring or actual acceptance decisions."""
    scored = deepcopy(rows)
    for row in scored:
        if row['pred'] is None:
            row['risks']['actual_count_logit'] = None
            row['supplement_flags'] = {method: False for method in METHODS}
            continue
        control = scoring.logit(frozen['control_model'], row)
        if abs(float(metrics.fusion.simple.sigmoid(control))-row['risks']['full_simple']) > 1e-12:
            raise ValueError('Formal scoring replay differs')
        risk = scoring.logit(frozen['candidate_model'], row)
        row['risks']['actual_count_logit'] = risk
        formal = row['risks']['full_simple'] <= frozen['formal_risk_le']
        if formal != row['original_simple_decision']['size_accepted']:
            raise ValueError('Formal flag replay differs')
        row['supplement_flags'] = dict(raw=True, formal_simple=formal,
            protected_simple=row['risks']['full_simple'] <= frozen['protected_risk_le'],
            actual_count_zero=risk < frozen['candidate_logit_lt'],
            score_train95=row['risks']['score_only'] <= frozen['score_risk_le'])
    return scored


def curve(rows, method):
    """All whole-tie offline thresholds; never exports a chosen working point."""
    buckets = {}
    for row in rows:
        if row['pred'] is not None:
            buckets.setdefault(row['risks'][method], []).append(row)
    points = [dict(threshold=None, **metrics.counts(rows, set()))]
    chosen = set()
    for threshold, bucket in sorted(buckets.items()):
        chosen.update(row['image'] for row in bucket)
        summary = metrics.counts(rows, chosen)
        independent.check_summary(summary, rows, chosen)
        points.append(dict(threshold=threshold, **summary))
    return points


def evaluate(rows, frozen):
    scored = apply(rows, frozen)
    checks = 0
    for source, row in zip(rows, scored):
        if any(row[k] != value for k, value in source.items() if k != 'risks'):
            raise ValueError('Boxes/labels/original flags/identity changed')
        if any(row['risks'][k] != value for k, value in source['risks'].items()):
            raise ValueError('Old risk changed')
        if row['pred'] is not None:
            model = frozen['candidate_model']; x = independent.descriptor(row, 3)
            expected = model['weights'][0]+sum((a-m)/s*w for a,m,s,w in
                zip(x, model['mean'], model['scale'], model['weights'][1:]))
            if abs(expected-row['risks']['actual_count_logit']) > 1e-12:
                raise ValueError('Independent score differs')
        checks += 1
    groups = {}; curves = {}
    for name, part in metrics.fusion.level.metrics.grouped(scored).items():
        methods = {}
        for method in METHODS:
            chosen = {r['image'] for r in part if r['supplement_flags'][method]}
            delivery = metrics.counts(part, chosen)
            independent.check_summary(delivery, part, chosen)
            controls = {}
            for risk in RISKS:
                count = metrics.point(part, risk, count=delivery['accepted'])
                correct = metrics.point(part, risk, correct=delivery['states']['CR'])
                independent.check_point(count, part, risk, count=delivery['accepted'])
                independent.check_point(correct, part, risk, correct=delivery['states']['CR'])
                controls[risk] = dict(same_count=count, same_CR=correct)
                checks += 3
            methods[method] = dict(delivery=delivery, controls=controls)
        aucs = {}; target = {}; group_curves = {}
        required = math.ceil(.95*sum(independent.bad(r) is False for r in part))
        for risk in RISKS:
            auc = metrics.fusion.level.metrics.auc(part, risk)
            good = np.array([r['risks'][risk] for r in part if independent.bad(r) is False])
            bad = np.array([r['risks'][risk] for r in part if independent.bad(r) is True])
            expected = float(((bad[:,None] > good[None])+.5*(bad[:,None] == good[None])).mean()) if len(good) and len(bad) else None
            if (expected is None) != (auc is None) or (auc is not None and abs(auc-expected) > 1e-12):
                raise ValueError('Independent AUC differs')
            aucs[risk] = auc
            point = metrics.point(part, risk, correct=required)
            independent.check_point(point, part, risk, correct=required)
            target[risk] = point
            group_curves[risk] = curve(part, risk)
            checks += 2+len(group_curves[risk])
        outputs = [r for r in part if r['pred'] is not None]
        hits = sum(math.hypot(r['pred'][0]-r['gt'][0], r['pred'][1]-r['gt'][1]) < 15 for r in outputs)
        groups[name] = dict(methods=methods, AUROC=aucs, offline_correct95=target,
            center=dict(output_frames=len(outputs), correct_output_frames=hits,
                hit_rate_outputs=hits/len(outputs) if outputs else None,
                output_coverage=len(outputs)/len(part), full_frame_correct_coverage=hits/len(part)),
            center_angle_flags_boxes_scores_output_identity_unchanged=True)
        curves[name] = group_curves
    return dict(frames=len(rows), groups=groups, independent_checks=checks), curves


def run(out):
    if out.is_symlink() or out.name != 'result' or out.resolve().parent.parent != (ROOT/'work_dirs'/VERSION).resolve():
        raise ValueError('Use work_dirs/version/RUN_ID/result')
    frozen = freeze()
    out.mkdir(parents=True, exist_ok=False)
    write(out/'frozen_model_workpoints.json', frozen)  # Before opening VAL rows.
    with gzip.open(ROOT/(VAL+'scored_VAL.jsonl.gz'), 'rt') as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != 887 or len({r['image'] for r in rows}) != 887 or any(
            r['split'] != 'val' or r['reliability_role'] != 'val' for r in rows):
        raise ValueError('Not complete original VAL')
    result, curves = evaluate(rows, frozen)
    for name, digest in PINS.items():
        if sha(ROOT/name) != digest:
            raise ValueError('Frozen source changed during scoring')
    write(out/'report.json', dict(protocol=VERSION, result=result, input_pins=PINS,
        frozen_workpoints=frozen, fits=0, solver_calls=0, threshold_updates=0,
        detector_inferences=0, GPU_used=False, TEST_opened=False,
        formal_policy_changed=False, candidate_adopted=False,
        prior_TRAIN_gate_failure_retained=True, supplement_authorized_after_gate_failure=True,
        frontend_checked_from_reviewed_cache_only=True,
        offline_matching_and_curves_use_evaluation_GT=True))
    write(out/'risk_correct_retention_curves.json', curves)
    with (out/'metrics.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['group','method','FA','FR','ED','CR','MISSING','accepted','correct_retention','accepted_error_rate','acceptance_coverage','full_frame_correct_coverage','longest_correct_FR'])
        for group, data in result['groups'].items():
            for method, value in data['methods'].items():
                d = value['delivery']
                writer.writerow([group,method]+[d['states'][s] for s in metrics.STATES]+[d[k] for k in
                    ('accepted','correct_retention','accepted_error_rate','acceptance_coverage','full_frame_correct_coverage','longest_correct_FR')])
    write(out/'review.json', dict(passed=True, rows=len(rows), independent_checks=result['independent_checks'],
        report_sha256=sha(out/'report.json'), curves_sha256=sha(out/'risk_correct_retention_curves.json'),
        source_code_sha256=sha(__file__), frozen_before_VAL_scoring=True, TEST_opened=False))
    print(json.dumps(result['groups']['all'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
