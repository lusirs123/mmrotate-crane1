"""Training-only ConR-inspired regularization of the original 2480-param head.

Adaptation of ConR (ICLR 2024, BorealisAI/ConR): continuous two-edge log
residuals, existing natural batch, uniform density weights, no projection or
extra augmentations. The inference graph and original size loss are unchanged.
"""
import math
import torch
from torch.nn import functional as F
from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead
from crane_project.utils import port_size_qalign_v1 as model

CONR_WEIGHT = .001
TEMPERATURE = .07
WINDOW = math.log(1.02)
PUSHING = .01


class PairConRSizeHead(PairSpecificSizeHead):
    def forward(self, *args):
        if not self.training or not torch.is_grad_enabled():
            return super().forward(*args)
        captured = []
        handle = self.stem.register_forward_hook(lambda module, inputs, output: captured.append(output))
        try:
            out = super().forward(*args)
        finally:
            handle.remove()
        if captured:
            out['training_features'] = captured[0]
        return out


def canonical_pair(delta, bm, m):
    """Invert the inherited model-B/M axis permutations, without sorting values."""
    swap = (bm[:, 2] < bm[:, 3]) ^ (m[:, 2] < m[:, 3])
    return torch.where(swap[:, None], delta.flip(-1), delta)


def contrastive_loss(features, support, target, prediction):
    """Mean over the unchanged batch; anchors lacking positives/negatives give 0.

    Positive: target L-infinity distance <= log(1.02). Negative: targets farther
    apart but detached decoded predictions within that window. Self excluded.
    No target or prediction gradients; the auxiliary path only reaches the stem.
    """
    n = len(features)
    if n == 0 or target.shape != (n, 2) or prediction.shape != (n, 2):
        raise ValueError('Expected nonempty feature/target/prediction batch')
    w = support.detach().clamp(0, 1)
    embedding = (features * w.sqrt()).flatten(1)
    z = F.normalize(embedding, dim=1, eps=1e-6)
    distance = (target.detach()[:, None] - target.detach()[None, :]).abs().amax(-1) / WINDOW
    predicted = (prediction.detach()[:, None] - prediction.detach()[None, :]).abs().amax(-1) / WINDOW
    other = ~torch.eye(n, device=features.device, dtype=torch.bool)
    positive = (distance <= 1) & other
    negative = (distance > 1) & (predicted <= 1) & other
    active = positive.any(1) & negative.any(1)
    stats = dict(anchors=n, active_anchors=int(active.sum()), positive_pairs=int(positive.sum()),
                 negative_pairs=int(negative.sum()), zero_embeddings=int((embedding.norm(dim=1) <= 1e-6).sum()))
    if not bool(active.any()):
        return features.sum() * 0., stats
    similarity = z @ z.T / TEMPERATURE
    logits = similarity + torch.where(negative, PUSHING * distance, torch.zeros_like(distance))
    denominator = torch.logsumexp(logits[active].masked_fill(~(positive | negative)[active], -torch.inf), dim=1)
    mean_positive = (similarity[active] * positive[active]).sum(1) / positive[active].sum(1)
    return (denominator - mean_positive).sum() / n, stats


def update(h, opt, x, arm, torch_module=None, weight=CONR_WEIGHT):
    if arm != 'fixed_b':
        raise ValueError('Only the frozen fixed_b arm is authorized')
    h.train(); opt.zero_grad(); out = h(*x[:-1])
    size = model.size_loss(out, x[-1], x[-2], x[3])
    target = model.residual.size_targets(x[-1], x[-2], x[3])['log_residual']
    aux, stats = contrastive_loss(out['training_features'], out['common_support'],
        canonical_pair(target, x[3], x[-2]), canonical_pair(out['delta_roi'], x[3], x[-2]))
    # Exact original graph for the explicit zero-weight control.
    loss = size if weight == 0 else size + weight * aux
    if not bool(torch.isfinite(loss)):
        raise ValueError('Nonfinite loss')
    loss.backward()
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in h.parameters()):
        raise ValueError('Invalid gradient')
    stem = math.sqrt(sum(float(p.grad.double().square().sum()) for p in h.stem.parameters()))
    terminal = math.sqrt(sum(float(p.grad.double().square().sum()) for p in h.terminal.parameters()))
    norm = float(torch.nn.utils.clip_grad_norm_(h.parameters(), 10.)); opt.step()
    if any(not bool(torch.isfinite(p).all()) for p in h.parameters()):
        raise ValueError('Invalid parameters')
    return dict(loss=float(loss.detach()), size_loss=float(size.detach()), geometry_loss=0.,
        conr_loss=float(aux.detach()), weighted_conr_loss=weight*float(aux.detach()),
        stem_grad_norm=stem, terminal_grad_norm=terminal, grad_norm=norm, **stats)
