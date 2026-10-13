"""Past-only log-size trend residual; saved-box reliability, never box correction."""
from copy import deepcopy
from statistics import median
from crane_project.utils import port_reliability_temporal_evidence_v1 as previous

VERSION = 'port_reliability_temporal_trend_v1'
SETTINGS = dict(previous.SETTINGS, primary='past3_log_edge_trend_residual')


class CausalTrendEvidence(previous.CausalSizeEvidence):
    def observe(self, pred, frame_id, image_size, reset=False):
        past = list(self.history)
        point = super().observe(pred, frame_id, image_size, reset=reset)
        point['level_primary'] = point['primary']
        point['slope'] = point['predicted_log_edges'] = None
        if point['available']:
            slopes, predicted, residual = [], [], []
            current = self.history[-1][1]
            for axis in (0, 1):
                a, b, c = [h[1][axis] for h in past]
                # With three equally spaced observations this median is the secant
                # (c-a)/2, and expected equals c+v: no outlier robustness claim.
                v = median((b-a, c-b, (c-a)/2))
                expected = median((a+3*v, b+2*v, c+v))
                slopes.append(v); predicted.append(expected)
                residual.append(current[axis]-expected)
            point.update(primary=max(map(abs, residual)), signed_log_change=residual,
                         slope=slopes, predicted_log_edges=predicted)
        return point


def enrich(rows, boundaries):
    result = previous.enrich(rows, boundaries)
    tracker = None; sequence = None
    for row in result:
        new = row['sequence'] != sequence
        if new: tracker = CausalTrendEvidence()
        row['level_temporal'] = deepcopy(row['temporal'])
        row['temporal'] = tracker.observe(row['pred'], row['frame_id'], row['image_size'],
            reset=new or row['frame_id'] in boundaries.get(row['sequence'], []))
        row['temporal']['acquisition_boundary'] = row['level_temporal']['acquisition_boundary']
        if row['temporal']['available'] != row['level_temporal']['available']:
            raise ValueError('History support changed')
        if row['temporal']['level_primary'] != row['level_temporal']['primary']:
            raise ValueError('Old evidence changed')
        row['risks']['temporal'] = row['temporal']['primary']
        sequence = row['sequence']
    return result


def level_view(rows):
    result = deepcopy(rows)
    for r in result:
        r['temporal'] = r['level_temporal']
        r['risks']['temporal'] = r['temporal']['primary']
    return result


def matched_veto(rows, count=None, correct=None):
    """Offline old-veto curve, keeping original rejects and unsupported fallback fixed."""
    fallback = {r['image'] for r in rows if r['original_simple_decision']['size_accepted']
                and not r['temporal']['available']}
    supported = [r for r in rows if r['original_simple_decision']['size_accepted']
                 and r['temporal']['available']]
    extra = previous.summarize(rows, fallback)
    if count is not None:
        required = count-len(fallback)
        if not 0 <= required <= len(supported): return dict(feasible=False)
        point = previous.point(supported, 'temporal', count=required)
        chosen = sorted(supported, key=lambda r: (r['risks']['temporal'], r['image']))[:required]
        summary = previous.summarize(rows, fallback | {r['image'] for r in chosen})
        point['tie_bounds']['FA_min'] += extra['states']['FA']
        point['tie_bounds']['FA_max'] += extra['states']['FA']
        summary.update(tie_bounds=point['tie_bounds'], feasible=True)
    else:
        required = correct-extra['states']['CR']
        if not 0 <= required <= sum(previous.metrics.bad(r) is False for r in supported):
            return dict(feasible=False)
        point = previous.point(supported, 'temporal', correct=required)
        t = point['offline_GT_threshold']
        chosen = {r['image'] for r in supported if t is not None and r['risks']['temporal'] <= t}
        summary = previous.summarize(rows, fallback | chosen)
        summary.update(exact_CR=summary['states']['CR']==correct, feasible=True,
                       offline_GT_threshold=t)
    summary['offline_comparison_only'] = True
    return summary


def same_FA_veto(rows, wrong):
    """Largest whole-tie prefix meeting old FA budget; diagnostic, never online fitting."""
    fallback = {r['image'] for r in rows if r['original_simple_decision']['size_accepted']
                and not r['temporal']['available']}
    chosen = set(fallback)
    baseline = previous.summarize(rows, chosen)
    if baseline['states']['FA'] > wrong: return dict(feasible=False)
    supported = sorted((r for r in rows if r['original_simple_decision']['size_accepted']
                        and r['temporal']['available']),
                       key=lambda r: (r['risks']['temporal'], r['image']))
    index = 0; fa = baseline['states']['FA']; threshold = None
    while index < len(supported):
        end = index+1; value = supported[index]['risks']['temporal']
        while end < len(supported) and supported[end]['risks']['temporal']==value: end += 1
        added = sum(previous.metrics.bad(r) for r in supported[index:end])
        if fa+added > wrong: break
        chosen.update(r['image'] for r in supported[index:end]); fa += added
        threshold = value; index = end
    result = previous.summarize(rows, chosen)
    result.update(feasible=True, exact_FA=result['states']['FA']==wrong,
                  offline_GT_threshold=threshold, offline_comparison_only=True)
    return result


def comparison(rows, old_cutoff, new_cutoff):
    old = level_view(rows)
    old_ids = previous.veto_accept(old, old_cutoff)
    new_ids = previous.veto_accept(rows, new_cutoff)
    groups = {}
    for group, part in previous.metrics.grouped(rows).items():
        ids = {r['image'] for r in part}; oldpart = [r for r in old if r['image'] in ids]
        o, n = previous.summarize(part, old_ids & ids), previous.summarize(part, new_ids & ids)
        transitions = {name: [] for name in ('restored_CR', 'new_FR', 'rescued_FA', 'new_FA')}
        for r in part:
            was, now = r['image'] in old_ids, r['image'] in new_ids
            wrong = previous.metrics.bad(r)
            if was == now or wrong is None: continue
            name = ('new_FA' if wrong else 'restored_CR') if now else ('rescued_FA' if wrong else 'new_FR')
            transitions[name].append(r['image'])
        transitions = dict(counts={k: len(v) for k, v in transitions.items()}, images=transitions)
        groups[group] = dict(level=o, trend=n, transitions=transitions,
            level_same_trend_count=matched_veto(oldpart, count=n['accepted_outputs']),
            level_same_trend_CR=matched_veto(oldpart, correct=n['states']['CR']),
            trend_same_level_CR=matched_veto(part, correct=o['states']['CR']),
            trend_at_level_FA_budget=same_FA_veto(part, o['states']['FA']))
    return groups


def evaluate(rows, cutoff):
    return previous.evaluate(rows, cutoff)


def fit_cutoff(rows):
    return previous.fit_cutoff(rows)


def gate(new_evaluation, comparisons):
    result = deepcopy(new_evaluation['gate'])
    all_groups = comparisons['all']; new = all_groups['trend']
    at_cr = all_groups['level_same_trend_CR']
    at_fa = all_groups['trend_at_level_FA_budget']
    improved_cr = (at_cr['feasible'] and at_cr['exact_CR']
                   and new['states']['FA'] < at_cr['states']['FA'])
    improved_fa = (at_fa['feasible'] and at_fa['exact_FA']
                   and at_fa['states']['CR'] > all_groups['level']['states']['CR'])
    if not (improved_cr or improved_fa):
        result['failures'].append(dict(group='all', check='improve_over_level_at_matched_CR_or_FA'))
    # Also protect correct rejection continuity against the previous veto at matched CR.
    for group, c in comparisons.items():
        control = c['level_same_trend_CR']
        if not control['feasible'] or not control['exact_CR']:
            result['failures'].append(dict(group=group, check='level_same_CR_inconclusive'))
        elif c['trend']['runs']['correct_rejection']['longest'] > control['runs']['correct_rejection']['longest']:
            result['failures'].append(dict(group=group, check='level_longest_correct_FR_nonincrease'))
    result.update(passed=not result['failures'], improved_over_level_at_same_CR=improved_cr,
                  improved_over_level_at_same_FA=improved_fa)
    return result
