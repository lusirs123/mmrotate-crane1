"""Offline descriptive evidence only: no learned score or online GT input."""
import math
from collections import Counter
import numpy as np
from crane_project.utils.port_simple_component_reliability_v1 import descriptor

VERSION = 'port_reliability_conditional_evidence_v1'
NAMES = ('score_logit', 'relative_log_size', 'log_aspect')
SETTINGS = dict(level_cutoff=.0785511443740936, size_match_log_tolerance=.1,
                minimum_bad=5, minimum_good=5, screening_auc=.6,
                screening_interaction_contrast=.1, minimum_TRAIN_videos=2,
                minimum_VAL_videos=1)


def bad(row):
    if row['pred'] is None:
        return None
    a, b = sorted(row['pred'][2:4]), sorted(row['gt'][2:4])
    if any(not math.isfinite(x) or x <= 0 for x in a+b):
        raise ValueError('Invalid size truth/prediction')
    return max(abs(x-y)/y for x, y in zip(a, b)) > .1


def validate(rows, role):
    if role not in ('TRAIN', 'VAL'):
        raise ValueError('TEST prohibited')
    seen = set()
    for r in rows:
        if r['reliability_role'] != role.lower() or r['split'] not in (
                ('train', 'train_sim') if role == 'TRAIN' else ('val',)):
            raise ValueError('Source role mismatch')
        if r['image'] in seen:
            raise ValueError('Duplicate image')
        seen.add(r['image'])


def regime(r):
    if r['pred'] is None:
        return 'missing'
    t = r['temporal']
    if not t['available']:
        if t['primary'] is not None:
            raise ValueError('Unsupported history must not have numeric deviation')
        return 'unsupported'
    d = t['primary']
    if not math.isfinite(d) or d < 0:
        raise ValueError('Invalid temporal deviation')
    return 'low' if d <= SETTINGS['level_cutoff'] else 'high'


def counts(rows):
    b = sum(bad(r) is True for r in rows)
    g = sum(bad(r) is False for r in rows)
    return dict(frames=len(rows), bad=b, good=g, missing=len(rows)-b-g,
                error_rate=b/(b+g) if b+g else None)


def supported(c):
    return c['bad'] >= SETTINGS['minimum_bad'] and c['good'] >= SETTINGS['minimum_good']


def auc(rows, column, direction=1):
    # Average ranks count ties as half; no frame-independence significance claim.
    pairs = sorted((direction*float(descriptor(r['pred'], r['image_size'])[column]),
                    int(bad(r))) for r in rows if r['pred'] is not None)
    nb = sum(y for _, y in pairs); ng = len(pairs)-nb
    if not nb or not ng:
        return None
    numerator = 0.; good_before = 0; i = 0
    while i < len(pairs):
        j = i+1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]: j += 1
        b = sum(y for _, y in pairs[i:j]); g = j-i-b
        numerator += b*(good_before+.5*g); good_before += g; i = j
    return numerator/(nb*ng)


def freeze_train(train):
    validate(train, 'TRAIN')
    outputs = [r for r in train if r['pred'] is not None]
    low = [r for r in outputs if regime(r) == 'low']
    if not outputs or not low:
        raise ValueError('TRAIN outputs/low evidence required')
    x = np.array([descriptor(r['pred'], r['image_size']) for r in outputs])
    cuts = {name:np.quantile(x[:, i], [.25, .5, .75]).tolist() for i, name in enumerate(NAMES)}
    directions = {name:(1 if auc(low, i) is None or auc(low, i) >= .5 else -1)
                  for i, name in enumerate(NAMES)}
    return dict(settings=SETTINGS, quantiles=cuts, directions=directions,
                TRAIN_images=[r['image'] for r in train],
                direction_is_descriptive_not_a_selected_candidate=True)


def grouped(rows):
    result = {'all':rows}
    for field in ('domain', 'sequence'):
        for value in sorted({r[field] for r in rows}):
            result[field+':'+value] = [r for r in rows if r[field] == value]
    return result


def size_matched(low, frozen):
    pairs = []
    good = [r for r in low if bad(r) is False]
    for r in low:
        if bad(r) is not True: continue
        size = descriptor(r['pred'], r['image_size'])[1]
        choices = [(abs(size-descriptor(g['pred'], g['image_size'])[1]), g['image'], g)
                   for g in good if g['sequence'] == r['sequence']]
        if not choices: continue
        distance, _, g = min(choices, key=lambda x:(x[0], x[1]))
        if distance > SETTINGS['size_match_log_tolerance']: continue
        a, b = descriptor(r['pred'], r['image_size']), descriptor(g['pred'], g['image_size'])
        pair = dict(bad_image=r['image'], good_image=g['image'], size_distance=distance,
                    signed_difference={name:float((a[i]-b[i])*frozen['directions'][name])
                                       for i, name in enumerate(NAMES)})
        pairs.append(pair)
    return dict(pairs=pairs, matched_errors=len(pairs), unmatched_errors=sum(bad(r) is True for r in low)-len(pairs),
                unique_good_controls=len({p['good_image'] for p in pairs}),
                controls_reused=True, not_independent_pairs=True,
                concordance={name:(sum(p['signed_difference'][name] > 0 for p in pairs)
                                 +.5*sum(p['signed_difference'][name] == 0 for p in pairs))/len(pairs)
                             if pairs else None for name in NAMES})


def analyze(rows, frozen, cutoff):
    groups = {}; details = []
    lookup = {(r['sequence'], r['frame_id']):r for r in rows}
    for r in rows:
        accepted = r['pred'] is not None and r['risks']['fusion4'] <= cutoff
        label = bad(r)
        state = 'MISSING' if label is None else ('FA' if label else 'CR') if accepted else ('ED' if label else 'FR')
        history = [lookup.get((r['sequence'], i)) for i in r['temporal']['history_frames']]
        history_bad = sum(bad(h) is True for h in history if h is not None)
        details.append(dict(image=r['image'], sequence=r['sequence'], state=state,
                            regime=regime(r), history_bad_offline=history_bad,
                            history_lookup_complete=all(h is not None for h in history),
                            features=descriptor(r['pred'], r['image_size']).tolist() if r['pred'] is not None else None))
    for group, part in grouped(rows).items():
        outputs = [r for r in part if r['pred'] is not None]
        low = [r for r in outputs if regime(r) == 'low']
        statics = {}; interactions = {}
        for i, name in enumerate(NAMES):
            cuts = frozen['quantiles'][name]; bins = []
            for k in range(4):
                cell = [r for r in low if int(np.searchsorted(cuts, descriptor(r['pred'], r['image_size'])[i], side='right')) == k]
                bins.append(counts(cell))
            cells = {}
            for half in ('lower', 'upper'):
                for temporal in ('low', 'high'):
                    cell = [r for r in outputs if regime(r) == temporal and
                            ('lower' if descriptor(r['pred'], r['image_size'])[i] < cuts[1] else 'upper') == half]
                    cells[half+'_'+temporal] = counts(cell)
            rates = [cells[h+'_'+t]['error_rate'] for h in ('lower','upper') for t in ('low','high')]
            contrast = (rates[3]-rates[2])-(rates[1]-rates[0]) if all(x is not None for x in rates) else None
            interactions[name] = dict(cells=cells, contrast=contrast,
                                      sufficient_support=all(supported(c) for c in cells.values()))
            statics[name] = dict(low_auc=auc(low, i, frozen['directions'][name]), quartiles=bins)
        ids = {r['image'] for r in part}; d = [x for x in details if x['image'] in ids]
        states = Counter(x['state'] for x in d)
        groups[group] = dict(total=counts(part), regimes={k:counts([r for r in part if regime(r) == k])
                                for k in ('low','high','unsupported','missing')},
                            states={k:states[k] for k in ('FA','FR','ED','CR','MISSING')},
                            FA_regimes={k:sum(x['state']=='FA' and x['regime']==k for x in d)
                                        for k in ('low','high','unsupported')},
                            low_statics=statics, interactions=interactions,
                            size_matched=size_matched(low, frozen),
                            FR_with_bad_history=sum(x['state']=='FR' and x['history_bad_offline']>0 for x in d))
    return dict(groups=groups, details=details)


def prerequisite_screen(roles):
    # Fixed descriptive prerequisites, not significance tests or performance gates.
    static = {}; interaction = {}
    for name in NAMES:
        found = {}
        for role in ('TRAIN','VAL'):
            found[role] = [g for g, d in roles[role]['groups'].items() if g.startswith('sequence:')
                and supported(d['regimes']['low']) and d['low_statics'][name]['low_auc'] is not None
                and d['low_statics'][name]['low_auc'] >= SETTINGS['screening_auc']]
        static[name] = dict(videos=found, sufficient=(len(found['TRAIN'])>=SETTINGS['minimum_TRAIN_videos']
                                                     and len(found['VAL'])>=SETTINGS['minimum_VAL_videos']))
        found = {}
        for role in ('TRAIN','VAL'):
            found[role] = {str(sign):[g for g,d in roles[role]['groups'].items() if g.startswith('sequence:')
                and d['interactions'][name]['sufficient_support']
                and d['interactions'][name]['contrast'] is not None
                and (1 if d['interactions'][name]['contrast'] > 0 else -1)==sign
                and abs(d['interactions'][name]['contrast'])>=SETTINGS['screening_interaction_contrast']]
                for sign in (-1,1)}
        ok = any(len(found['TRAIN'][str(s)])>=SETTINGS['minimum_TRAIN_videos']
                 and len(found['VAL'][str(s)])>=SETTINGS['minimum_VAL_videos'] for s in (-1,1))
        interaction[name] = dict(videos=found, sufficient=ok)
    eligible = any(static[n]['sufficient'] and interaction[n]['sufficient'] for n in NAMES)
    return dict(status='DESCRIPTIVE_PREREQUISITE_PRESENT' if eligible else 'NO_SUFFICIENT_INTERACTION_EVIDENCE',
                static=static, interaction=interaction, automatic_training=False,
                candidate_effectiveness_not_tested=True, statistical_significance_not_claimed=True)
