"""One PRIME-style training objective; original2480 LVSR inference unchanged."""
import math

import torch
from torch import nn

from crane_project.utils import lvsr_prime_review_v1 as reviewed
from crane_project.utils import port_size_qalign_v1 as geometry
from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead

SETTINGS = dict(count=20, dimension=648, tau_feature=5., tau_target=1., alpha=.005,
                weight_alignment=.001, weight_proxy=.001, proxy_lr_multiplier=1., proxy_clip_norm=10.)


class PairPrimeSizeHead(PairSpecificSizeHead):
    def forward(self, *args):
        if not self.training or not torch.is_grad_enabled():
            return super().forward(*args)
        captured = []
        hook = self.stem.register_forward_hook(lambda module, inputs, output: captured.append(output))
        try:
            out = super().forward(*args)
        finally:
            hook.remove()
        if captured:
            out['training_features'] = captured[0]
        return out


class ProxyBank(nn.Module):
    """Training-only state, deliberately outside the deployment head."""
    def __init__(self, targets):
        super().__init__()
        if targets.shape != (20, 2) or not bool(torch.isfinite(targets).all()):
            raise ValueError('Requires the fixed20 TRAIN-only two-dimensional targets')
        self.features = reviewed.new_proxy_features(20, 648, dtype=targets.dtype)
        self.register_buffer('targets', targets.detach().clone())


def objectives(head, bank, x):
    head.train()
    out = head(*x[:-1])
    size = geometry.size_loss(out, x[-1], x[-2], x[3])
    raw_target = geometry.residual.size_targets(x[-1], x[-2], x[3])['log_residual']
    target = reviewed.canonical_pair(raw_target, x[3], x[-2])
    z, norm = reviewed.embedding(out['training_features'], out['common_support'])
    per_sample, associations, _ = reviewed.alignment_per_sample(z, target, bank.features, bank.targets,
        SETTINGS['tau_feature'], SETTINGS['tau_target'])
    proxy, parts = reviewed.proxy_loss(bank.features, bank.targets,
        SETTINGS['tau_feature'], SETTINGS['tau_target'], SETTINGS['alpha'])
    return size, per_sample.mean(), proxy, dict(anchors=len(target),
        zero_embeddings=int((norm <= reviewed.EPS).sum()),
        target_entropy_mean=float(-(associations * associations.clamp_min(1e-30).log()).sum(1).mean()),
        proxy_KL=float(parts['KL']), proxy_spread=float(parts['spread']))


def update(head, bank, optimizer, x, weight_alignment=None, weight_proxy=None):
    wa = SETTINGS['weight_alignment'] if weight_alignment is None else weight_alignment
    wp = SETTINGS['weight_proxy'] if weight_proxy is None else weight_proxy
    if wa == 0 and wp == 0:
        # Exact original zero-auxiliary control, including its optimizer/clip path.
        from crane_project.tools.run_symeood_110_lvsr_v1 import update as original_update
        return original_update(head, optimizer, x, 'fixed_b', torch)
    optimizer.zero_grad()
    size, align, proxy, stats = objectives(head, bank, x)
    loss = reviewed.combined_loss(size, align, proxy, wa, wp)
    if not bool(torch.isfinite(loss)):
        raise ValueError('Nonfinite PRIME loss')
    loss.backward()
    parameters = list(head.parameters()) + list(bank.parameters())
    if any(q.grad is None or not bool(torch.isfinite(q.grad).all()) for q in parameters):
        raise ValueError('Invalid head or proxy gradient')
    stem = math.sqrt(sum(float(q.grad.double().square().sum()) for q in head.stem.parameters()))
    terminal = math.sqrt(sum(float(q.grad.double().square().sum()) for q in head.terminal.parameters()))
    # Preserve the original head clip10. Extra proxy gradients have their own cap.
    head_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 10.))
    proxy_norm = float(torch.nn.utils.clip_grad_norm_(bank.parameters(), SETTINGS['proxy_clip_norm']))
    optimizer.step()
    if any(not bool(torch.isfinite(q).all()) for q in parameters):
        raise ValueError('Nonfinite trained parameter')
    return dict(loss=float(loss.detach()), size_loss=float(size.detach()),
        alignment_loss=float(align.detach()), proxy_loss=float(proxy.detach()),
        weighted_alignment_loss=wa*float(align.detach()), weighted_proxy_loss=wp*float(proxy.detach()),
        stem_grad_norm=stem, terminal_grad_norm=terminal, grad_norm=head_norm, proxy_grad_norm=proxy_norm, **stats)
