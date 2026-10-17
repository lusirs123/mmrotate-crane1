"""Offline fixed single-axis weights; I forward and delivery are unchanged.

GT/domain/scale are used only for complete-TRAIN preparation and loss. Weight
mass n_g*w_g is balanced, not actual loss/gradient. near3 coefficient remains 1.
"""
import math
from collections import Counter

VERSION = 'port_geometry_size_axis_balance_v1'
GROUPS = ('undersized', 'oversized', 'near3')


def labels(gt, midpoint):
    g, m = sorted(gt[2:4], reverse=True), sorted(midpoint[2:4], reverse=True)
    e = [x/y-1 for x, y in zip(m, g)]
    return ['near3' if abs(x) <= .03 else 'undersized' if x < 0 else 'oversized' for x in e]


def weights_for_counts(counts):
    if set(counts) != set(GROUPS) or any(not isinstance(n, int) or n < 0 for n in counts.values()):
        raise ValueError('Complete integer axis counts required')
    if not counts['undersized'] or not counts['oversized']:
        raise ValueError('Missing error direction; do not treat as passed')
    total = counts['undersized']+counts['oversized']
    lo, hi = 0., float(total)
    for _ in range(100):
        mid = (lo+hi)/2
        if sum(min(4*counts[g], mid) for g in GROUPS[:2]) < total:
            lo = mid
        else:
            hi = mid
    w = dict(near3=1., **{g:min(4., hi/counts[g]) for g in GROUPS[:2]})
    if not math.isclose(sum(counts[g]*w[g] for g in GROUPS), sum(counts.values()), abs_tol=1e-9):
        raise ValueError('Mean-one weight accounting failed')
    return w


def prepare(rows, indices):
    """Return canonical weights in exactly packed present-view order."""
    counts, records = {}, []
    for i in indices:
        r = rows[i]
        if r['split'] not in ('train', 'train_sim') or r['domain'] not in ('real', 'sim') or r['scale'] not in (1., .5) or r['midpoint'] is None:
            raise ValueError('Only complete present TRAIN views may define weights')
        cell = r['domain']+'/'+str(r['scale']); group = labels(r['gt'], r['midpoint'])
        records.append(dict(image=r['image'], scale=r['scale'], cell=cell, groups=group))
        for j, edge in enumerate(('long', 'short')):
            counts.setdefault(cell+'/'+edge, Counter({g:0 for g in GROUPS}))[group[j]] += 1
    if len(records) != 5112 or len({(r['image'], r['scale']) for r in records}) != 5112 or len(counts) != 8:
        raise ValueError('Complete 5112 present views / eight cells required')
    cells = {c:dict(counts=dict(ns), weights=weights_for_counts(dict(ns))) for c, ns in counts.items()}
    values = [[cells[r['cell']+'/'+edge]['weights'][r['groups'][j]] for j, edge in enumerate(('long', 'short'))] for r in records]
    return dict(cells=cells, records=records, canonical_weights=values,
                interpretation='Fixed weighted sample mass only; no gradient/loss equality or near accuracy guarantee.')


def permute_axes(values, midpoint, boxes_model):
    """Canonical -> M raw -> B-model ROI. Same permutation is its inverse."""
    import torch
    if values.shape != (len(midpoint), 2):
        raise ValueError('Expected one two-axis value per view')
    swap = (midpoint[:, 2] < midpoint[:, 3]) ^ (boxes_model[:, 2] < boxes_model[:, 3])
    return torch.where(swap[:, None], values.flip(-1), values)


def loss(output, gt, midpoint, boxes_model, canonical_weights):
    import torch
    from torch.nn import functional as F
    from crane_project.utils import port_geometry_size_distribution_v1 as base
    terms = base.loss(output, gt, midpoint, boxes_model, 'decoded_control')
    w = canonical_weights.detach()
    if w.shape != output['delta_roi'].shape or not bool(torch.isfinite(w).all()) or not bool(((w > 0) & (w <= 4)).all()):
        raise ValueError('Invalid fixed per-axis weights')
    point = F.smooth_l1_loss(output['delta_roi'], terms['target'], beta=.1, reduction='none')
    roi_weights = permute_axes(w, midpoint, boxes_model)
    # 1/(2N) sum(w*loss); NEVER divide by a batch weight sum.
    terms['total'] = (point*roi_weights).mean()
    terms['unweighted'] = terms['decoded']
    terms['canonical_pointwise'] = permute_axes(point, midpoint, boxes_model)
    return terms
