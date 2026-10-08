"""Size-only readout ablation. Reuse pinned features; no new box or TEST."""
from copy import deepcopy

import numpy as np

from crane_project.utils import port_reliability_redc_size_v1 as old
from crane_project.utils import port_reliability_feature_ablation_v1 as ab
from crane_project.utils.port_reliability_tradeoff_v1 import required_good

VERSION = 'port_reliability_readout_compare_v1'
ARMS = ('temperature', 'residual')
CONTROLS = ('full_simple', 'score_only', 'temperature')
SETTINGS = deepcopy(old.SETTINGS)
SETTINGS['readouts'] = dict(temperature='-(z/(.25+softplus(a(h)))+beta)',
                           residual='-z+a(h)+beta')
normalize = old.normalize
normalization = old.normalization
class_weights = old.class_weights
decide = old.decide
describe = old.describe


def hidden_and_output(model, features, normalizer):
    h = normalize(features, normalizer)[..., 1:]
    for n in (0, 2):
        h = np.maximum(h @ np.asarray(model['network.%d.weight'%n]).T
                       + model['network.%d.bias'%n], 0.)
    a = (h @ np.asarray(model['network.4.weight']).T
         + model['network.4.bias'])[..., 0]
    return h, a


def numpy_logits(model, features, normalizer, arm):
    if arm == 'temperature':
        return old.numpy_logits(model, features, normalizer, 'redc')
    if arm != 'residual':
        raise ValueError('Unknown fixed readout arm')
    _, a = hidden_and_output(model, features, normalizer)
    return -np.asarray(features)[..., 0] + a + float(model['beta'][0])


def gate(statistics):
    """VAL adoption is stricter than a mechanism signal. No post hoc winner."""
    failures = []
    for group, value in statistics.items():
        a = value['actual']; c = a['states']
        if c['CR'] < required_good(c['CR']+c['FR'], .95):
            failures.append(dict(group=group, check='correct_retention'))
        for name in CONTROLS:
            bounds = value['control_tie_bounds'][name]
            matched = value['matched_CR_controls'][name]
            if c['FA'] > bounds['bad_min_over_tie']:
                failures.append(dict(group=group,control=name,check='same_count_FA_nonincrease'))
            if group == 'all' and c['FA'] >= bounds['bad_min_over_tie']:
                failures.append(dict(group=group,control=name,check='strict_overall_same_count_FA_gain'))
            if c['FA'] > matched['states']['FA']:
                failures.append(dict(group=group,control=name,check='same_CR_FA_nonincrease'))
            if group == 'all' and c['FA'] >= matched['states']['FA']:
                failures.append(dict(group=group,control=name,check='strict_overall_same_CR_FA_gain'))
            if not matched['exact_CR']:
                failures.append(dict(group=group,control=name,check='same_CR_tie_inconclusive'))
            if a['runs']['correct_rejection']['longest'] > matched['runs']['correct_rejection']['longest']:
                failures.append(dict(group=group,control=name,check='longest_FR_same_CR_nonincrease'))
    return dict(passed=not failures,failures=failures,
                gate_role='VAL_development_not_independent_test',
                selected_arm='residual' if not failures else None,TEST_used=False)


def cross_sign_diagnostic(rows, scores, arm):
    """Offline secondary mechanism check, never a model/threshold selector."""
    wrong_high = [scores[arm][r['image']] for r in rows
                  if r['pred'] is not None and r['pred'][5] > .5 and ab.size_bad(r)]
    good_low = [scores[arm][r['image']] for r in rows
                if r['pred'] is not None and r['pred'][5] < .5 and not ab.size_bad(r)]
    equal_score = sum(r['pred'] is not None and r['pred'][5] == .5 for r in rows)
    pairs = np.asarray(wrong_high)[:, None]-np.asarray(good_low)[None, :]
    return dict(bad_score_above_half=len(wrong_high),good_score_below_half=len(good_low),
                score_exact_half=equal_score,pairs=int(pairs.size),
                correctly_ordered=int((pairs > 0).sum()),ties=int((pairs == 0).sum()),
                pair_AUROC=float(np.mean((pairs > 0)+.5*(pairs == 0))) if pairs.size else None,
                diagnostic_only=True)


def hidden_diagnostic(rows, ids, features, model, normalizer):
    h, a = hidden_and_output(model, features, normalizer)
    zero = np.all(h == 0, axis=1)
    lookup = {r['image']:r for r in rows}
    result = {}
    for name, group in ab.groups(rows).items():
        members = {r['image'] for r in group if r['pred'] is not None}
        indices = [i for i,image in enumerate(ids) if image in members]
        result[name] = dict(outputs=len(indices),bad=sum(ab.size_bad(lookup[ids[i]]) for i in indices),
            second_hidden_all_zero=int(zero[indices].sum()),
            zero_bad=sum(bool(zero[i]) and ab.size_bad(lookup[ids[i]]) for i in indices),
            raw_output_range=[float(a[indices].min()),float(a[indices].max())] if indices else None)
    return result
