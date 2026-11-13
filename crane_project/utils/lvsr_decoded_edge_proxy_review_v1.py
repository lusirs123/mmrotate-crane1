"""Review-only decoder-conditioned proxy objectives. Never imported by training.

Original hidden-channel indices are retained. The detached role mask affects
only the auxiliary descriptor, not the inference support, decoder or delivery.
"""
import torch
from torch.nn import functional as F
from crane_project.utils import lvsr_prime_review_v1 as prime


def masks(lengths, support):
    """N,81,2 pair lengths -> N,2,648 sqrt(W)-weighted original-slot masks."""
    n = len(lengths)
    if lengths.shape != (n, 81, 2) or support.shape != (n, 1, 9, 9):
        raise ValueError('Unexpected pair lengths or original support shape')
    with torch.no_grad():
        order = lengths.detach().sort(-1, descending=True)[1]
        # axes: sample, canonical edge, opposite-pair branch, channel, location.
        selected = order.transpose(1, 2)[:, :, None, None, :]
        branch = torch.arange(2, device=lengths.device)[None, None, :, None, None]
        role = (selected == branch).expand(n, 2, 2, 4, 81).reshape(n, 2, 648)
        weight = support.detach().clamp(0, 1).sqrt().expand(n, 8, 9, 9).flatten(1)
        weighted = role.to(weight.dtype) * weight[:, None]
    return weighted, role, order


def objectives(features, weighted_masks, target, proxy_features, centers):
    """Same20x648 bank, two 1D target projections, fixed mean over two edges.

    Both sample and proxies receive the SAME per-sample mask. Proxy arrangement
    is consequently conditional on that mask too; this is an adaptation, not
    the original PRIME objective. Temperatures/alpha match the existing trial.
    """
    n, k = len(features), len(centers)
    if features.shape != (n, 8, 9, 9) or weighted_masks.shape != (n, 2, 648):
        raise ValueError('Unexpected feature or auxiliary mask shape')
    if target.shape != (n, 2) or proxy_features.shape != (k, 648) or centers.shape != (k, 2):
        raise ValueError('Unexpected targets or proxy shape')
    weighted_masks = weighted_masks.detach()
    x = features.flatten(1)[:, None] * weighted_masks
    p = proxy_features[None, None] * weighted_masks[:, :, None]
    xnorm, pnorm = x.norm(dim=-1), p.norm(dim=-1)
    z = F.normalize(x, dim=-1, eps=prime.EPS)
    q = F.normalize(p, dim=-1, eps=prime.EPS)
    distance = (z[:, :, None] - q).norm(dim=-1)
    dt = (target.detach()[:, :, None] - centers.detach().T[None]).abs() / prime.WINDOW
    associations = (-dt).softmax(-1)
    alignment = -(associations * (-5 * distance).log_softmax(-1)).sum(-1)
    target_dist = (centers.T[:, :, None] - centers.T[:, None, :]).abs().detach() / prime.WINDOW
    feature_dist = torch.cdist(q.reshape(n*2, k, 648), q.reshape(n*2, k, 648)).reshape(n, 2, k, k)
    other = ~torch.eye(k, dtype=torch.bool, device=features.device)
    log_p = (-target_dist[:, other]).log_softmax(-1)
    log_q = (-5*feature_dist[:, :, other]).log_softmax(-1)
    kl = (log_p.exp()[None] * (log_p[None] - log_q)).sum(-1)
    cosine = q @ q.transpose(-1, -2)
    spread = (.005 * target_dist[None] * (1-cosine).square())[:, :, other].sum(-1)
    return dict(alignment=alignment, proxy=kl-spread, sample_norm=xnorm,
                proxy_norm=pnorm, weighted_masks=weighted_masks)
