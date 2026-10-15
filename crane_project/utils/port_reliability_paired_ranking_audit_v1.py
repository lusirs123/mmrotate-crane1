"""Offline saved-score diagnostics. Standard library only; no fit/inference API."""
from bisect import bisect_left, bisect_right
from collections import Counter
import math

STATES = ('FA', 'FR', 'ED', 'CR', 'MISSING')
IDENTITY = ('image', 'sequence', 'domain', 'split', 'frame_id', 'image_size',
            'pred', 'gt', 'reliability_role', 'original_simple_decision')


def bad(row):
    if row['pred'] is None:
        return None
    p = sorted(row['pred'][2:4])
    g = sorted(row['gt'][2:4])
    if any(not math.isfinite(x) or x <= 0 for x in p + g):
        raise ValueError('Invalid dimensions: ' + row['image'])
    return any(abs(a-b)/b > .1 for a, b in zip(p, g))


def identity(row):
    return {k: row[k] for k in IDENTITY}


def validate_rows(rows, role, expected):
    if role not in ('TRAIN', 'VAL') or len(rows) != expected:
        raise ValueError('Only complete saved TRAIN/VAL are supported')
    ids = [r['image'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate identities')
    for r in rows:
        allowed = ('train', 'train_sim') if role == 'TRAIN' else ('val',)
        if r['split'] not in allowed or r['reliability_role'] != role.lower():
            raise ValueError('Wrong historical role')
        if r['domain'] not in ('real', 'sim') or not r['sequence'].startswith(r['domain']+'_'):
            raise ValueError('Wrong domain/video identity')
        if r['split'] == 'train_sim' and r['domain'] != 'sim':
            raise ValueError('Wrong native Sim role')
        for k, n in (('gt', 5), ('pred', 6)):
            if r[k] is None and k == 'pred':
                continue
            if len(r[k]) != n or not all(math.isfinite(v) for v in r[k]):
                raise ValueError('Invalid ' + k)
        if r['pred'] is not None and not 0 <= r['pred'][5] <= 1:
            raise ValueError('Invalid score')
        b = bad(r)
        d = r['original_simple_decision']
        if d['final_box_original'] != r['pred']:
            raise ValueError('Original box changed')
        if b is None and any(d[k] for k in ('center_accepted', 'size_accepted', 'angle_accepted')):
            raise ValueError('Missing cannot be accepted')
    return {r['image']: r for r in rows}


def pair_identity(reference, other):
    if set(reference) != set(other):
        raise ValueError('Different frame material')
    for image in reference:
        if identity(reference[image]) != identity(other[image]):
            raise ValueError('Prediction/label/flags/identity mismatch: ' + image)


def groups(rows):
    result = {'all': rows}
    for field in ('domain', 'sequence'):
        for name in sorted({r[field] for r in rows}):
            result[field+':'+name] = [r for r in rows if r[field] == name]
    return result


def summarize(rows, accepted):
    states = Counter({s: 0 for s in STATES})
    longest = run = 0
    previous = None
    for r in sorted(rows, key=lambda x: (x['sequence'], x['frame_id'], x['image'])):
        b = bad(r)
        keep = r['image'] in accepted
        if b is None and keep:
            raise ValueError('Missing cannot be accepted')
        state = 'MISSING' if b is None else (('FA' if b else 'CR') if keep else ('ED' if b else 'FR'))
        states[state] += 1
        adjacent = previous == (r['sequence'], r['frame_id']-1)
        run = (run+1 if adjacent else 1) if state == 'FR' else 0
        longest = max(longest, run)
        previous = r['sequence'], r['frame_id']
    g = states['CR']+states['FR']
    a = states['FA']+states['CR']
    n = len(rows)
    return dict(states=dict(states), frames=n, outputs=n-states['MISSING'], accepted=a,
                correct_retention=states['CR']/g if g else None,
                accepted_error_rate=states['FA']/a if a else None,
                acceptance_coverage=a/n if n else None,
                output_coverage=(n-states['MISSING'])/n if n else None,
                full_frame_correct_coverage=states['CR']/n if n else None,
                longest_correct_FR=longest)


class Ranking:
    """Risk ascending means accepted first. CR matching never splits a tie."""
    def __init__(self, rows, method):
        self.rows = rows
        self.method = method
        self.ordered = sorted((r for r in rows if r['pred'] is not None),
                              key=lambda r: (r['audit_risks'][method], r['image']))
        self.scores = [r['audit_risks'][method] for r in self.ordered]
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in self.scores):
            raise ValueError('Invalid saved risk')
        self.prefix_bad = [0]
        for r in self.ordered:
            self.prefix_bad.append(self.prefix_bad[-1]+int(bad(r)))
        self.good_scores = [r['audit_risks'][method] for r in self.ordered if not bad(r)]
        self.good = len(self.good_scores)
        self.wrong = self.prefix_bad[-1]

    def point(self, n):
        fa = self.prefix_bad[n]
        cr = n-fa
        return dict(accepted=n, FA=fa, CR=cr, FR=self.good-cr, ED=self.wrong-fa,
                    correct_retention=cr/self.good if self.good else None,
                    accepted_error_rate=fa/n if n else None,
                    acceptance_coverage=n/len(self.rows) if self.rows else None,
                    full_frame_correct_coverage=cr/len(self.rows) if self.rows else None)

    def same_count(self, n):
        if not 0 <= n <= len(self.scores):
            raise ValueError('Invalid acceptance target')
        result = self.point(n)
        boundary = self.scores[n-1] if n else None
        lo = bisect_left(self.scores, boundary) if n else 0
        hi = bisect_right(self.scores, boundary) if n else 0
        slots = n-lo
        base = self.prefix_bad[lo]
        tied_bad = self.prefix_bad[hi]-base
        tied_good = hi-lo-tied_bad
        result.update(risk_le=boundary, offline_GT_comparison_only=True,
                      identity_split_for_count=True,
                      tie_bounds=dict(FA_min=base+max(0, slots-tied_good),
                                      FA_max=base+min(slots, tied_bad),
                                      partial_tie=0 < slots < hi-lo))
        return result

    def same_cr(self, target):
        if not 0 <= target <= self.good:
            raise ValueError('Invalid CR target')
        boundary = self.good_scores[target-1] if target else None
        n = bisect_right(self.scores, boundary) if target else 0
        result = self.point(n)
        result.update(requested_CR=target, exact_CR=result['CR'] == target,
                      risk_le=boundary, offline_GT_comparison_only=True,
                      whole_cutoff_tie=True)
        return result

    def curve(self):
        result = [dict(self.point(0), risk_le=None)]
        for i, score in enumerate(self.scores):
            if i+1 == len(self.scores) or score != self.scores[i+1]:
                result.append(dict(self.point(i+1), risk_le=score))
        return result

    def auc(self):
        if not self.good or not self.wrong:
            return None
        total = 0.
        for r, score in zip(self.ordered, self.scores):
            if bad(r):
                total += bisect_left(self.good_scores, score)
                total += .5*(bisect_right(self.good_scores, score)-bisect_left(self.good_scores, score))
        return total/(self.good*self.wrong)


def compare_high_region(candidate, control, retention=.95):
    """All integer CR targets >=95%; no sampled grid or new policy selection."""
    if candidate.good != control.good or candidate.wrong != control.wrong:
        raise ValueError('Different labels/material')
    values = Counter(dict(better=0, equal=0, worse=0, tie_inconclusive=0))
    differences = []
    for target in range(math.ceil(retention*candidate.good), candidate.good+1):
        if not target:
            continue
        a, b = candidate.same_cr(target), control.same_cr(target)
        if not a['exact_CR'] or not b['exact_CR']:
            values['tie_inconclusive'] += 1
            continue
        delta = a['FA']-b['FA']
        differences.append(delta)
        values['better' if delta < 0 else 'worse' if delta > 0 else 'equal'] += 1
    return dict(**dict(values), FA_delta_min=min(differences) if differences else None,
                FA_delta_max=max(differences) if differences else None,
                retention_interval=[retention, 1.], correlated_targets_not_independent=True,
                empty_correct_support=candidate.good == 0)


def independent_check(rank, delivery, cutoff):
    """Separate scalar pair/count reconstruction; does not call curve/point/auc."""
    good = [r['audit_risks'][rank.method] for r in rank.rows if bad(r) is False]
    errors = [r['audit_risks'][rank.method] for r in rank.rows if bad(r) is True]
    expected_auc = (sum((e > g)+.5*(e == g) for e in errors for g in good)/
                    (len(errors)*len(good))) if errors and good else None
    if expected_auc is None:
        if rank.auc() is not None:
            raise ValueError('Single-class AUROC')
    elif abs(expected_auc-rank.auc()) > 1e-12:
        raise ValueError('Independent pair AUROC differs')
    expected = dict(FA=sum(s <= cutoff for s in errors),
                    CR=sum(s <= cutoff for s in good),
                    ED=sum(s > cutoff for s in errors),
                    FR=sum(s > cutoff for s in good),
                    MISSING=sum(r['pred'] is None for r in rank.rows))
    if delivery['states'] != expected:
        raise ValueError('Frozen states differ')
    # Exhaustively reconstruct every whole-tie curve prefix using an independent
    # grouped histogram, including empty acceptance and all outputs.
    hist = {}
    for r in rank.rows:
        if r['pred'] is not None:
            key = r['audit_risks'][rank.method]
            x = hist.setdefault(key, [0, 0])
            x[int(bad(r))] += 1
    expected_points = [(0, 0)]
    cr = fa = 0
    for key in sorted(hist):
        cr += hist[key][0]
        fa += hist[key][1]
        expected_points.append((cr, fa))
    if expected_points != [(p['CR'], p['FA']) for p in rank.curve()]:
        raise ValueError('Independent full curve differs')
    return dict(pair_comparisons=len(errors)*len(good), curve_points=len(expected_points),
                frozen_states_verified=True)
