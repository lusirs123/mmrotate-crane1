"""Offline audit of existing scores; no learned fusion or online GT API."""
import math
from collections import Counter
from crane_project.utils import port_reliability_state_continuity_v1 as continuity

VERSION = 'port_reliability_complementarity_v1'
REFERENCES = ('formal_simple', 'simple_retention95')
CONTROLS = ('full_simple', 'score_only')
STATES = ('FA', 'FR', 'ED', 'CR', 'MISSING')


def bad(row):
    if row['pred'] is None:
        return None
    predicted, truth = sorted(row['pred'][2:4]), sorted(row['gt'][2:4])
    if any(not math.isfinite(v) or v <= 0 for v in predicted + truth):
        raise ValueError('Nonfinite/nonpositive size')
    return max(abs(p-g)/g for p, g in zip(predicted, truth)) > .1


def state(row, keep):
    wrong = bad(row)
    if wrong is None:
        if keep:
            raise ValueError('Missing output cannot be accepted')
        return 'MISSING'
    return ('FA' if wrong else 'CR') if keep else ('ED' if wrong else 'FR')


def grouped(rows):
    return dict(all=rows, **{field+':'+name: [r for r in rows if r[field] == name]
        for field in ('domain', 'sequence') for name in sorted({r[field] for r in rows})})


def accepted(rows, method, cutoff):
    if not math.isfinite(cutoff) or not 0 <= cutoff <= 1:
        raise ValueError('Invalid frozen risk cutoff')
    return {r['image'] for r in rows if r['pred'] is not None and r['risks'][method] <= cutoff}


def auc(rows, method):
    """High risk denotes error. Exact equal scores receive half pair credit."""
    out = [r for r in rows if r['pred'] is not None]
    good = sum(not bad(r) for r in out)
    wrong = len(out)-good
    if not good or not wrong:
        return None
    ordered = sorted(out, key=lambda r: r['risks'][method])
    credit = 0.; preceding_good = 0; index = 0
    while index < len(ordered):
        end = index+1
        while end < len(ordered) and ordered[end]['risks'][method] == ordered[index]['risks'][method]:
            end += 1
        tied_good = sum(not bad(r) for r in ordered[index:end])
        tied_wrong = end-index-tied_good
        credit += tied_wrong*(preceding_good+.5*tied_good)
        preceding_good += tied_good
        index = end
    return credit/(good*wrong)


def matched(rows, method, candidate_summary):
    """Offline comparisons only. Never write these GT-derived cutoffs to policy."""
    out = sorted((r for r in rows if r['pred'] is not None),
                 key=lambda r: (r['risks'][method], r['image']))
    count = candidate_summary['accepted_outputs']
    target = candidate_summary['states']['CR']
    selected = {r['image'] for r in out[:count]}
    same_count = continuity.summarize(rows, selected)
    if count:
        boundary = out[count-1]['risks'][method]
        before = [r for r in out if r['risks'][method] < boundary]
        tied = [r for r in out if r['risks'][method] == boundary]
        slots = count-len(before)
        wrong_before = sum(bad(r) for r in before)
        tie_bounds = dict(bad_min=wrong_before+max(0, slots-sum(not bad(r) for r in tied)),
                          bad_max=wrong_before+min(slots, sum(bad(r) for r in tied)),
                          partial_boundary_tie=slots < len(tied), tie_count=len(tied),
                          selection='image_order_not_GT')
    else:
        tie_bounds = dict(bad_min=0, bad_max=0, partial_boundary_tie=False,
                          tie_count=0, selection='image_order_not_GT')
    good = 0; threshold = None
    if target:
        for r in out:
            good += int(not bad(r))
            if good >= target:
                threshold = r['risks'][method]
                break
    same_cr = continuity.summarize(rows, accepted(rows, method, threshold) if threshold is not None else set())
    return dict(same_count=same_count, same_count_tie_bounds=tie_bounds,
                same_CR=same_cr, same_CR_exact=same_cr['states']['CR'] == target,
                same_CR_offline_risk_le=threshold, offline_GT_comparison_only=True)


def complement(rows, baseline, candidate, method):
    group_ids = {r['image'] for r in rows}
    baseline, candidate = baseline & group_ids, candidate & group_ids
    base = continuity.summarize(rows, baseline)
    proposed = continuity.summarize(rows, candidate)
    reject_only = continuity.summarize(rows, baseline & candidate)
    ids = dict(rescued_FA=[], new_FR=[], restored_CR=[], new_FA=[])
    table = {a: {b: 0 for b in STATES} for a in STATES}
    for r in rows:
        a, b = state(r, r['image'] in baseline), state(r, r['image'] in candidate)
        table[a][b] += 1
        key = {('FA', 'ED'): 'rescued_FA', ('CR', 'FR'): 'new_FR',
               ('FR', 'CR'): 'restored_CR', ('ED', 'FA'): 'new_FA'}.get((a, b))
        if key:
            ids[key].append(r['image'])
    counts = {k: len(v) for k, v in ids.items()}
    if proposed['states']['FA']-base['states']['FA'] != counts['new_FA']-counts['rescued_FA']:
        raise ValueError('FA decomposition failed')
    if proposed['states']['FR']-base['states']['FR'] != counts['new_FR']-counts['restored_CR']:
        raise ValueError('FR decomposition failed')
    conditional = [r for r in rows if r['image'] in baseline]
    return dict(baseline=base, candidate=proposed, state_transition=table,
        changes=counts, change_images={k: sorted(v) for k, v in ids.items()},
        reject_only_AND=reject_only,
        AND_comparison={name: matched(rows, name, reject_only) for name in CONTROLS},
        candidate_comparison={name: matched(rows, name, proposed) for name in CONTROLS},
        conditional_on_simple_accept=dict(frames=len(conditional),
            good=sum(bad(r) is False for r in conditional), bad=sum(bad(r) is True for r in conditional),
            error_AUROC={name: auc(conditional, name) for name in (method, *CONTROLS)}),
        GT_oracle_upper_bound=dict(rescued_FA=counts['rescued_FA'], restored_CR=counts['restored_CR'],
            remaining_FA=base['states']['FA']-counts['rescued_FA'],
            remaining_FR=base['states']['FR']-counts['restored_CR'],
            executable_policy=False, GT_selects_changes=True))


def gate(groups):
    """Only a necessary evidence check for a later design; not model adoption."""
    failures = []
    for name, point in groups.items():
        overlay = point['reject_only_AND']; counts = overlay['states']
        good = counts['CR']+counts['FR']
        if good and counts['CR'] < math.ceil(.95*good):
            failures.append(dict(group=name, check='correct_retention95'))
        for control, comparison in point['AND_comparison'].items():
            if counts['FA'] > comparison['same_count_tie_bounds']['bad_min']:
                failures.append(dict(group=name, control=control, check='same_count_FA_nonincrease'))
            if name == 'all' and counts['FA'] >= comparison['same_count_tie_bounds']['bad_min']:
                failures.append(dict(group=name, control=control, check='overall_strict_same_count_FA_gain'))
            if not comparison['same_CR_exact']:
                failures.append(dict(group=name, control=control, check='same_CR_tie_inconclusive'))
            if counts['FA'] > comparison['same_CR']['states']['FA']:
                failures.append(dict(group=name, control=control, check='same_CR_FA_nonincrease'))
            if name == 'all' and counts['FA'] >= comparison['same_CR']['states']['FA']:
                failures.append(dict(group=name, control=control, check='overall_strict_same_CR_FA_gain'))
            if overlay['runs']['correct_rejection']['longest'] > comparison['same_CR']['runs']['correct_rejection']['longest']:
                failures.append(dict(group=name, control=control, check='longest_correct_FR_nonincrease'))
    return dict(passed=not failures, failures=failures, candidate_adopted=False,
                necessary_evidence_only=True, GT_oracle_cannot_pass_gate=True)


def center(rows):
    out = [r for r in rows if r['pred'] is not None]
    hits = sum(math.hypot(r['pred'][0]-r['gt'][0], r['pred'][1]-r['gt'][1]) < 15 for r in out)
    return dict(frames=len(rows), outputs=len(out), hits=hits,
        hit_rate_on_outputs=hits/len(out) if out else None,
        output_coverage=len(out)/len(rows) if rows else None,
        correct_coverage_all_frames=hits/len(rows) if rows else None)
