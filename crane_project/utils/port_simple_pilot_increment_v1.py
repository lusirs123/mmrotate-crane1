"""Fixed three-feature size-only data increment; no images or detector imports."""
import math
import numpy as np
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils.port_reliability_paired_ranking_audit_v1 import (
    Ranking, bad, groups, summarize, compare_high_region)

VERSION = 'port_simple_pilot_increment_v1'
METHODS = ('formal', 'refit', 'augmented')


def checked_rows(rows, role, expected):
    if len(rows) != expected or len({r['image'] for r in rows}) != len(rows):
        raise ValueError('Incomplete or duplicate role')
    for r in rows:
        if role == 'PILOT':
            if (r['split'] != 'extra_source_real_seq11_pilot_k1p9_v2_train'
                    or r['reliability_role'] != 'train' or r['sequence'] != 'real_seq11'
                    or r['domain'] != 'real' or r['image_size'] != [1920, 1080]):
                raise ValueError('Wrong pilot historical identity')
            d = r['decision']
        else:
            allowed = {'TRAIN': ('train', 'train_sim'), 'VAL': ('val',), 'TEST': ('test',)}[role]
            if r['split'] not in allowed or r.get('reliability_role', role.lower()) != role.lower():
                raise ValueError('Wrong saved role')
            d = r['original_simple_decision']
        if d['final_box_original'] != r['pred'] or d['center_accepted'] != (r['pred'] is not None):
            raise ValueError('Box or center identity changed')
        if not all(math.isfinite(x) for x in r['gt']) or min(r['gt'][2:4]) <= 0:
            raise ValueError('Invalid GT')
        if r['pred'] is not None:
            simple.descriptor(r['pred'], r['image_size'])
        bad(r)
        r['audit_risks'] = {}
    return rows


def fit(rows, settings):
    outputs = [r for r in rows if r['pred'] is not None]
    if len({r['image'] for r in rows}) != len(rows):
        raise ValueError('Duplicate material across sources')
    if any(r['reliability_role'] != 'train' for r in rows):
        raise ValueError('Fitting accepts TRAIN only')
    return simple.fit_linear_risk([simple.descriptor(r['pred'], r['image_size']) for r in outputs],
                                  [bad(r) for r in outputs], settings)


def add_scores(rows, models):
    for r in rows:
        for method, model in models.items():
            r['audit_risks'][method] = (float(simple.linear_risk(model, simple.descriptor(
                r['pred'], r['image_size']))) if r['pred'] is not None else None)


def at_cut(rows, method, cut):
    accepted = {r['image'] for r in rows if r['pred'] is not None and cut is not None
                and r['audit_risks'][method] <= cut}
    result = {}
    for group, part in groups(rows).items():
        v = summarize(part, accepted)
        run = gap = longest = longest_gap = 0
        previous = None
        for r in sorted(part, key=lambda r: (r['sequence'], r['frame_id'], r['image'])):
            if r['sequence'] != previous:
                run = gap = 0
            correct = bad(r) is False
            run = run+1 if correct and r['image'] not in accepted else 0
            gap = 0 if correct and r['image'] in accepted else gap+1
            longest = max(longest, run); longest_gap = max(longest_gap, gap)
            previous = r['sequence']
        v.update(longest_FR_observations=longest, longest_without_CR_observations=longest_gap)
        result[group] = v
    return result


def diagnostics(rows, method, cuts):
    rank = Ranking(rows, method)
    formal = Ranking(rows, 'formal')
    original = at_cut(rows, 'formal', cuts['formal'])['all']
    candidate = at_cut(rows, method, cuts[method])['all']
    return dict(at_frozen_VAL_cut=at_cut(rows, method, cuts[method]),
        same_CR_as_formal=rank.same_cr(original['states']['CR']),
        same_count_as_formal=rank.same_count(original['accepted']),
        formal_at_candidate_count=formal.same_count(candidate['accepted']),
        group_AUROC={g: Ranking(p, method).auc() for g, p in groups(rows).items()},
        high_retention_vs_formal=compare_high_region(rank, formal),
        auxiliary={str(p): rank.same_cr(math.ceil(rank.good*p)) for p in (.95, .97)},
        curve=rank.curve())


def decision(val_metrics):
    a = val_metrics['augmented']['at_frozen_VAL_cut']
    f = val_metrics['formal']['at_frozen_VAL_cut']
    checks = dict(protects_CR=a['all']['states']['CR'] >= 653,
                  reduces_FA=a['all']['states']['FA'] < f['all']['states']['FA'],
                  protects_real=a['domain:real']['states']['CR'] >= 189)
    return dict(passed=all(checks.values()), checks=checks,
                selected_method='augmented' if all(checks.values()) else None,
                continuity='Report all saved-observation and strict-frame FR runs; no posthoc threshold veto/search',
                test_repeatedly_exposed=True, automatic_promotion=False)
