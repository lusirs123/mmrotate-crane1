"""PRIME equations 1-8, adapted for a frozen-protocol LVSR feasibility review.

This module supplies training losses only. It does not change the LVSR head,
decoder, support, checkpoint layout or delivery rules. No optimizer is provided.
"""
import math

import torch
from torch.nn import functional as F

WINDOW = math.log(1.02)
EPS = 1e-6


def canonical_pair(value, boxes_model, midpoint):
    swap = (boxes_model[:, 2] < boxes_model[:, 3]) ^ (midpoint[:, 2] < midpoint[:, 3])
    return torch.where(swap[:, None], value.flip(-1), value)


def embedding(features, support):
    """Reuse the previous ConR descriptor, without another projection layer."""
    weighted = (features * support.detach().clamp(0, 1).sqrt()).flatten(1)
    return F.normalize(weighted, dim=1, eps=EPS), weighted.norm(dim=1)


def target_distance(a, b):
    # A diagnostic scale fixed before reading current TRAIN/VAL values.
    return torch.cdist(a.detach(), b.detach(), p=2) / WINDOW


def target_association(target, proxy_target, tau_target=1.):
    return (-tau_target * target_distance(target, proxy_target)).softmax(1)


def alignment_per_sample(z, target, proxy_features, proxy_target,
                         tau_feature=5., tau_target=1.):
    """Equations 5-7: every sample has a proxy reference, including batch1."""
    p = F.normalize(proxy_features, dim=1, eps=EPS)
    log_a = (-tau_feature * torch.cdist(z, p, p=2)).log_softmax(1)
    t = target_association(target, proxy_target, tau_target)
    return -(t * log_a).sum(1), t, log_a


def proxy_loss(proxy_features, proxy_target, tau_feature=5., tau_target=1., alpha=.005):
    """Equations 1-4: global off-diagonal normalization; squared cosine term."""
    count = len(proxy_features)
    if count < 2 or proxy_target.shape != (count, 2):
        raise ValueError('At least two paired two-dimensional proxies required')
    z = F.normalize(proxy_features, dim=1, eps=EPS)
    dt = target_distance(proxy_target, proxy_target)
    df = torch.cdist(z, z, p=2)
    other = ~torch.eye(count, dtype=torch.bool, device=z.device)
    log_p = (-tau_target * dt[other]).log_softmax(0)
    log_q = (-tau_feature * df[other]).log_softmax(0)
    kl = (log_p.exp() * (log_p - log_q)).sum()
    spread = (alpha * dt * (1 - z @ z.T).square())[other].sum()
    return kl - spread, dict(KL=kl.detach(), spread=spread.detach())


def kmeans_targets(target, count=20, max_iterations=100):
    """One deterministic farthest-first Lloyd run, using TRAIN targets only."""
    y = target.detach().double().cpu()
    if y.ndim != 2 or y.shape[1] != 2 or not bool(torch.isfinite(y).all()):
        raise ValueError('Expected finite two-dimensional TRAIN targets')
    if count < 2 or len(torch.unique(y, dim=0)) < count:
        raise ValueError('Not enough distinct TRAIN targets')
    first = int(((y - y.mean(0)).square().sum(1)).argmin())
    chosen = [first]
    while len(chosen) < count:
        nearest = torch.cdist(y, y[chosen]).square().amin(1)
        chosen.append(int(nearest.argmax()))
    centers = y[chosen].clone()
    last = None
    for iteration in range(1, max_iterations + 1):
        assignment = torch.cdist(y, centers).argmin(1)
        sizes = torch.bincount(assignment, minlength=count)
        if not bool((sizes > 0).all()):
            raise ValueError('Empty proxy cluster; stop without reinitialization search')
        centers = torch.stack([y[assignment == j].mean(0) for j in range(count)])
        if last is not None and torch.equal(assignment, last):
            break
        last = assignment
    final = torch.cdist(y, centers).argmin(1)
    sizes = torch.bincount(final, minlength=count)
    # A fixed-iteration K-means center remains a valid TRAIN-only reference.
    # Record exact assignment convergence instead of rejecting finite references
    # or silently spending extra iterations to satisfy an overly strict check.
    if not bool(torch.isfinite(centers).all()) or not bool((sizes > 0).all()):
        raise ValueError('Invalid finite-budget proxy centers')
    return centers.to(target.dtype), dict(iterations=iteration, counts=sizes.tolist(),
        inertia=float((y - centers[final]).square().sum()), initialization='one deterministic farthest-first',
        exact_assignment_converged=torch.equal(final, assignment),
        final_assignment_changes=int((final != assignment).sum()))


def new_proxy_features(count, dimension, dtype=torch.float32, seed=1703):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    # Explicit He normal initialization; leave the head/global RNG untouched.
    values = torch.randn(count, dimension, generator=generator, dtype=dtype) * math.sqrt(2. / dimension)
    return torch.nn.Parameter(values)


def combined_loss(size, align, proxy, weight_align=.001, weight_proxy=.001):
    if weight_align == 0 and weight_proxy == 0:
        return size
    return size + weight_align * align + weight_proxy * proxy
