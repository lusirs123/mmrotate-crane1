"""Offline sequence-excluded prediction comparison; deployed detector stays fixed."""
from copy import deepcopy
import hashlib
import json
import math
import numpy as np
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils import port_reliability_complementarity_v1 as metrics
from crane_project.utils import port_reliability_state_continuity_v1 as states
from crane_project.utils import port_reliability_within_video_rank_v1 as ranking

VERSION = 'port_reliability_oof_v1'
COUNTS = dict(real_seq01=339, real_seq05=560, real_seq06=466,
              real_seq12=141, real_seq13=304, sim_seq08=748)
GROUPS = dict(A=('real_seq01', 'real_seq05'),
              B=('real_seq06', 'real_seq12', 'real_seq13'))
FITTING = dict(l2=.1, max_iterations=100, gradient_tolerance=1e-8)
METHODS = ('in_sample', 'oof', 'common_in_sample', 'common_oof',
           'full_simple', 'score_only')
CONTROLS = ('in_sample', 'full_simple', 'score_only')
SETTINGS = dict(seed=1701, detector_epochs=24, detector_batch=2,
                midpoint_epochs=3, midpoint_seed=1703, sigma_cells=1.5,
                preflight_steps=8, minimum_output_coverage=.95,
                minimum_center_matching_fraction=.90, correct_retention=.95,
                auxiliary_weight_initialization='ImageNet_only_no_port_task_checkpoint',
                source_scope='sequence_excluded_Real_only_not_independent_acquisition_events')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def plan(rows):
    train = [r for r in rows if r['split'] in ('train', 'train_sim')]
    counts = {s: sum(r['sequence'] == s for r in train) for s in COUNTS}
    if counts != COUNTS or len(train) != sum(COUNTS.values()):
        raise ValueError('Complete original TRAIN membership required')
    if len({r['image'] for r in train}) != len(train):
        raise ValueError('Duplicate training identity')
    groups = {g: {r['image'] for r in train if r['sequence'] in seqs}
              for g, seqs in GROUPS.items()}
    sim = {r['image'] for r in train if r['domain'] == 'sim'}
    folds = {}
    for name, excluded in groups.items():
        allowed = {r['image'] for r in train} - excluded
        if allowed & excluded or not sim <= allowed:
            raise ValueError('Fold exclusion leaked')
        folds[name] = dict(train_images=sorted(allowed), predict_images=sorted(excluded),
                           excluded_sequences=list(GROUPS[name]),
                           real_train=len(allowed)-len(sim), sim_train=len(sim))
    if groups['A'] & groups['B'] or groups['A'] | groups['B'] | sim != {r['image'] for r in train}:
        raise ValueError('Folds do not partition original TRAIN')
    return folds


def materialize_training(original, auxiliary):
    """GT label follows that row's prediction. Missing outputs stay missing."""
    lookup = {r['image']: r for r in auxiliary}
    real = {r['image'] for r in original if r['domain'] == 'real'}
    if set(lookup) != real or len(lookup) != len(auxiliary):
        raise ValueError('Every Real TRAIN identity must have exactly one OOF record')
    result = []
    for old in original:
        if old['split'] not in ('train', 'train_sim'):
            raise ValueError('Quality fitting may only consume TRAIN')
        r = deepcopy(old)
        if old['domain'] == 'real':
            new = lookup[old['image']]
            for k in ('gt', 'image_size', 'sequence', 'split'):
                if new[k] != old[k]: raise ValueError('Offline source identity changed: '+k)
            expected = next(g for g, seqs in GROUPS.items() if old['sequence'] in seqs)
            if new['excluded_fold'] != expected:
                raise ValueError('Wrong auxiliary fold')
            r['pred'] = deepcopy(new['pred'])
            r['prediction_source'] = 'sequence_excluded_'+expected
        else:
            r['prediction_source'] = 'formal_M_in_sample_sim'
        result.append(r)
    return result


def fit(rows):
    if any(r['split'] not in ('train', 'train_sim') for r in rows):
        raise ValueError('VAL/TEST cannot fit a judgement model')
    out = [r for r in rows if r['pred'] is not None]
    return simple.fit_linear_risk(
        np.asarray([simple.descriptor(r['pred'], r['image_size']) for r in out]),
        np.asarray([metrics.bad(r) for r in out], dtype=float), FITTING)


def support(rows):
    result = {}
    for group, part in metrics.grouped(rows).items():
        out = [r for r in part if r['pred'] is not None]
        errors = np.asarray([np.sort(r['pred'][2:4])[::-1]/
                             np.sort(r['gt'][2:4])[::-1]-1 for r in out])
        result[group] = dict(frames=len(part), outputs=len(out), missing=len(part)-len(out),
                            good=sum(metrics.bad(r) is False for r in out),
                            bad=sum(metrics.bad(r) is True for r in out),
                            center=metrics.center(part))
        if len(out):
            result[group].update(signed_edge_bias=errors.mean(0).tolist(),
                edge_MAE=np.abs(errors).mean(0).tolist(),
                edge_P95=np.quantile(np.abs(errors), .95, axis=0).tolist(),
                both_small3=int((errors.max(1) <= -.03).sum()),
                both_large3=int((errors.min(1) >= .03).sum()),
                error_images=[r['image'] for r in out if metrics.bad(r)])
    return result


def source_gate(rows):
    """Engineering sanity, not evidence of transferable error coverage."""
    failures = []
    for name, v in support(rows).items():
        if name.startswith('sequence:real_'):
            if v['outputs']/v['frames'] < SETTINGS['minimum_output_coverage']:
                failures.append(dict(group=name, check='aux_output_coverage95'))
            if (v['center']['hit_rate_on_outputs'] or 0) < SETTINGS['minimum_center_matching_fraction']:
                failures.append(dict(group=name, check='aux_center_matching90'))
    all_counts = support(rows)['all']
    if not all_counts['good'] or not all_counts['bad']:
        failures.append(dict(group='all', check='two_quality_classes'))
    return dict(passed=not failures, failures=failures,
                guarantees_representative_errors=False)


def score(rows, models):
    result = deepcopy(rows)
    for r in result:
        r['risks'] = dict(full_simple=r['size_risks']['full_simple'],
                          score_only=1-r['pred'][5] if r['pred'] is not None else None)
        for name, model in models.items():
            r['risks'][name] = float(simple.linear_risk(model, simple.descriptor(r['pred'], r['image_size']))) if r['pred'] is not None else None
    return result


def statistics(rows, cutoffs):
    result = {}
    for method in METHODS:
        result[method] = {}
        for name, group in metrics.grouped(rows).items():
            actual = states.summarize(group, metrics.accepted(group, method, cutoffs[method]['risk_le']))
            result[method][name] = dict(actual=actual, error_AUROC=metrics.auc(group, method),
                controls={m: metrics.matched(group, m, actual) for m in METHODS if m != method})
    return result


def gate(stats):
    failures = []
    for name, value in stats['oof'].items():
        c = value['actual']['states']
        if c['CR'] < math.ceil(.95*(c['CR']+c['FR'])):
            failures.append(dict(group=name, check='correct_retention95'))
        for control in CONTROLS:
            v = value['controls'][control]
            checks = dict(same_count_FA_nonincrease=c['FA'] <= v['same_count_tie_bounds']['bad_min'],
                same_CR_exact=v['same_CR_exact'], same_CR_FA_nonincrease=c['FA'] <= v['same_CR']['states']['FA'],
                longest_correct_FR_nonincrease=value['actual']['runs']['correct_rejection']['longest'] <= v['same_CR']['runs']['correct_rejection']['longest'])
            if name == 'all':
                checks.update(overall_strict_same_count_FA_gain=c['FA'] < v['same_count_tie_bounds']['bad_min'],
                              overall_strict_same_CR_FA_gain=c['FA'] < v['same_CR']['states']['FA'])
            failures += [dict(group=name, control=control, check=k) for k, passed in checks.items() if not passed]
    return dict(passed=not failures, failures=failures, adopted=False,
                frozen_TEST_allowed=not failures, no_common_arm_reselection=True)
