"""TRAIN-only descriptive increment audit. No fitter or online decision rule."""
import math
from collections import Counter
from statistics import mean

VERSION = 'port_reliability_sparse_candidate_v1'
FEATURES = ('neighbor1_log_L', 'neighbor1_log_S', 'neighbor2_log_L', 'neighbor2_log_S')
STATES = ('FA', 'FR', 'ED', 'CR', 'MISSING')


def axes(box):
    if len(box) not in (5, 6) or not all(math.isfinite(x) for x in box):
        raise ValueError('Finite original-coordinate OBB required')
    if min(box[2:4]) <= 0:
        raise ValueError('Positive OBB edges required')
    return tuple(sorted(box[2:4], reverse=True))


def wrong(row):
    if row['pred'] is None:
        return None
    p, g = axes(row['pred']), axes(row['gt'])
    return any(abs(a - b) / b > .1 for a, b in zip(p, g))


def local_features(pred, neighbors):
    """GT-free, score-ordered neighbors from the frozen extraction; no oracle."""
    values = {key: None for key in FEATURES}
    if pred is None:
        if neighbors:
            raise ValueError('Missing output has neighbors')
        return values, ['missing', 'missing']
    L, S = axes(pred)
    for i, neighbor in enumerate(neighbors[:2], 1):
        l, s = axes(neighbor['box'])
        values['neighbor%d_log_L' % i] = math.log(l / L)
        values['neighbor%d_log_S' % i] = math.log(s / S)
    agreement = []
    for axis in ('L', 'S'):
        a, b = (values['neighbor%d_log_%s' % (i, axis)] for i in (1, 2))
        agreement.append('none' if a is None else 'single' if b is None else
            'both_positive' if a > 0 and b > 0 else
            'both_negative' if a < 0 and b < 0 else 'mixed_or_zero')
    return values, agreement


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    k = (len(values) - 1) * q
    i = int(k)
    return values[i] + (k - i) * (values[min(i + 1, len(values) - 1)] - values[i])


def distribution(values):
    return dict(n=len(values), mean=mean(values) if values else None,
        q10=quantile(values, .1), q25=quantile(values, .25),
        median=quantile(values, .5), q75=quantile(values, .75), q90=quantile(values, .9))


def pairs(wrong_values, good_values):
    n = len(wrong_values) * len(good_values)
    wins = sum(float(a > b) + .5 * (a == b) for a in wrong_values for b in good_values)
    return dict(wrong=len(wrong_values), good=len(good_values), pairs=n,
        wins=wins, AUC_higher= wins / n if n else None,
        AUC_lower=1 - wins / n if n else None)


def join(candidate_rows, reference_rows, cutoffs):
    if len({r['image'] for r in candidate_rows}) != len(candidate_rows) or len({r['image'] for r in reference_rows}) != len(reference_rows):
        raise ValueError('Duplicate image')
    ref = {r['image']: r for r in reference_rows}
    if set(ref) != {r['image'] for r in candidate_rows}:
        raise ValueError('TRAIN membership mismatch')
    output = []
    for row in candidate_rows:
        old = ref[row['image']]
        if row['reliability_role'] != 'train' or old['reliability_role'] != 'train' or row['split'] not in ('train', 'train_sim') or old['split'] != row['split']:
            raise ValueError('Only original TRAIN roles allowed')
        for key in ('pred', 'gt', 'image_size', 'sequence', 'domain', 'frame_id', 'temporal', 'original_simple_decision'):
            if row[key] != old[key]:
                raise ValueError('Frozen identity changed: ' + key)
        for key in ('partial_auc4', 'full_simple', 'fusion4', 'score_only'):
            if row['risks'][key] != old['risks'][key]:
                raise ValueError('Frozen risk changed: ' + key)
        available = row['temporal']['available']
        if not isinstance(available, bool):
            raise ValueError('Boolean history flag required')
        risk = old['risks']['crossfit_fusion'] if available else old['risks']['partial_auc4']
        if not available and old['risks']['crossfit_fusion'] != risk:
            raise ValueError('Fallback risk identity differs')
        for key in ('partial_auc4', 'crossfit_fusion', 'full_simple', 'score_only'):
            value = old['risks'][key]
            if row['pred'] is not None and (value is None or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError('Finite frozen risk required')
        accepted = row['pred'] is not None and risk <= cutoffs['supported' if available else 'fallback']
        if not isinstance(old['fallback_decision'], bool) or old['fallback_decision'] != accepted:
            raise ValueError('Frozen fallback decision changed')
        ev = row['candidate_evidence']
        neighbors = ev['neighbors']
        pool, chosen = row['candidate_pool'], row['candidate_chosen']
        indices = [n['index'] for n in neighbors]
        if len(set(indices)) != len(indices) or chosen in indices:
            raise ValueError('Self/duplicate neighbor')
        for n in neighbors:
            if not isinstance(n['index'], int) or not 0 <= n['index'] < len(pool) or n['box'] != pool[n['index']]:
                raise ValueError('Neighbor native source mismatch')
        if neighbors != sorted(neighbors, key=lambda n: (-n['box'][5], n['index'])):
            raise ValueError('Frozen neighbor ordering changed')
        x, agree = local_features(row['pred'], neighbors)
        y = wrong(row)
        decisions = dict(fallback_decision=accepted,
            partial_auc4=row['pred'] is not None and old['risks']['partial_auc4'] <= cutoffs['fallback'],
            full_simple=old['original_simple_decision']['size_accepted'] if row['pred'] is not None else False)
        states = {m: 'MISSING' if y is None else ('FA' if y else 'CR') if decision else ('ED' if y else 'FR')
            for m, decision in decisions.items()}
        output.append(dict(image=row['image'], sequence=row['sequence'], domain=row['domain'],
            history_available=available, temporal_primary=row['temporal']['primary'],
            neighbors=len(neighbors), features=x, agreement=agree, wrong=y,
            states=states, risks=dict(fallback_decision=risk, partial_auc4=old['risks']['partial_auc4'], full_simple=old['risks']['full_simple']),
            neighbor_source_indices=indices[:2]))
    return output


def summarize(rows):
    result = {}
    for method in ('fallback_decision', 'partial_auc4', 'full_simple'):
        counts = {}
        masks = dict(all=lambda r: True, no_neighbor=lambda r: r['neighbors'] == 0,
            one=lambda r: r['neighbors'] == 1, two=lambda r: r['neighbors'] == 2,
            three_plus=lambda r: r['neighbors'] >= 3, any=lambda r: r['neighbors'] >= 1,
            any_history=lambda r: r['neighbors'] >= 1 and r['history_available'])
        for name, predicate in masks.items():
            counter = Counter(r['states'][method] for r in rows if predicate(r))
            counts[name] = {s: counter[s] for s in STATES}
        result[method] = counts
    return result


def signal(rows, method):
    # Analyze existing accepted observations only: remaining FA versus retained CR.
    accepted = [r for r in rows if r['states'][method] in ('FA', 'CR') and r['neighbors'] > 0]
    # Label-free TRAIN risk quartiles are descriptive strata, never use thresholds.
    boundaries = [quantile([r['risks'][method] for r in accepted], q) for q in (.25, .5, .75)]
    features = {}
    for feature in FEATURES:
        valid = [r for r in accepted if r['features'][feature] is not None]
        bad = [r['features'][feature] for r in valid if r['states'][method] == 'FA']
        good = [r['features'][feature] for r in valid if r['states'][method] == 'CR']
        strata = []
        for seq in sorted({r['sequence'] for r in valid}):
            for bin_id in range(4):
                part = [r for r in valid if r['sequence'] == seq and
                    sum(r['risks'][method] > b for b in boundaries) == bin_id]
                a = [r['features'][feature] for r in part if r['states'][method] == 'FA']
                b = [r['features'][feature] for r in part if r['states'][method] == 'CR']
                if part:
                    strata.append(dict(sequence=seq, risk_quartile=bin_id, **pairs(a, b)))
        total = sum(s['pairs'] for s in strata)
        wins = sum(s['wins'] for s in strata)
        features[feature] = dict(FA=distribution(bad), CR=distribution(good),
            pooled=pairs(bad, good), same_video_risk_strata=strata,
            conditional_pairs=total, conditional_AUC_higher=wins / total if total else None,
            conditional_AUC_lower=1 - wins / total if total else None)
    agreement = {}
    for i, axis in enumerate(('L', 'S')):
        agreement[axis] = {name: {state: sum(r['agreement'][i] == name and r['states'][method] == state for r in rows)
            for state in STATES} for name in ('missing', 'none', 'single', 'both_positive', 'both_negative', 'mixed_or_zero')}
    return dict(risk_quartiles=boundaries, features=features, agreement=agreement,
        interpretation='Descriptive pair ordering, not conditional independence or out-of-video gain. No direction selected.')


def analyze(rows):
    groups = dict(all=rows)
    for key in ('domain', 'sequence'):
        for name in sorted({r[key] for r in rows}):
            groups[key + ':' + name] = [r for r in rows if r[key] == name]
    return dict(groups={name: summarize(part) for name, part in groups.items()},
        signals={method: {name: signal(part, method) for name, part in groups.items()}
            for method in ('fallback_decision', 'partial_auc4')},
        remaining_FA=[r for r in rows if r['states']['fallback_decision'] == 'FA'],
        status='DESCRIPTIVE_TRAIN_COMPLETE_NO_TRAINING_AUTHORIZED',
        recommendation_requires_human_review=True,
        limitations=['All predictions originate from detectors trained on TRAIN.',
            'Previously exposed TRAIN is development evidence, not held-out performance.',
            'Small correlated video samples cannot establish generalization.',
            'No fitted model, decision threshold, selected direction or performance gain.'])
