"""Decoder-conditioned auxiliary proxies; original2480 LVSR inference intact."""
import math

import torch

from crane_project.utils import lvsr_decoded_edge_proxy_review_v1 as reviewed
from crane_project.utils import lvsr_prime_review_v1 as prime
from crane_project.utils import port_size_pair_prime_v1 as original
from crane_project.utils import port_size_qalign_v1 as geometry

SETTINGS = dict(original.SETTINGS)
ProxyBank = original.ProxyBank


class PairDecodedEdgeProxyHead(original.PairPrimeSizeHead):
    """The inherited forward has no auxiliary selection or proxy computation."""


class InvalidAuxiliaryDescriptor(ValueError):
    def __init__(self, details):
        self.details = details
        super().__init__('TRAIN selected descriptor/proxy norm<=original EPS or nonfinite: '+str(details))


def objectives(head, bank, x):
    head.train()
    out = head(*x[:-1])
    size = geometry.size_loss(out, x[-1], x[-2], x[3])
    n = len(x[-2])
    restored = geometry.residual.original.roi_delta_to_original(
        out['vectors'].detach().reshape(n,324,2), x[2], x[3], x[4]).reshape(n,81,4,2)
    lengths = torch.stack(((restored[:,:,0]-restored[:,:,2]).norm(dim=-1),
                           (restored[:,:,1]-restored[:,:,3]).norm(dim=-1)), -1)
    masks, _, order = reviewed.masks(lengths, out['common_support'])
    target = prime.canonical_pair(geometry.residual.size_targets(x[-1], x[-2], x[3])['log_residual'], x[3], x[-2])
    parts = reviewed.objectives(out['training_features'], masks, target, bank.features, bank.targets)
    sn, pn = parts['sample_norm'], parts['proxy_norm']
    bad_sample = (~torch.isfinite(sn)) | (sn <= prime.EPS)
    bad_proxy = (~torch.isfinite(pn)) | (pn <= prime.EPS)
    if bool(bad_sample.any()) or bool(bad_proxy.any()):
        raise InvalidAuxiliaryDescriptor(dict(sample_edge_count=int(bad_sample.sum()),
            sample_long_count=int(bad_sample[:,0].sum()), sample_short_count=int(bad_sample[:,1].sum()),
            proxy_count=int(bad_proxy.sum()), min_sample_norm=float(sn.detach().min()), min_proxy_norm=float(pn.detach().min()), EPS=prime.EPS))
    for name in ('alignment', 'proxy'):
        if not bool(torch.isfinite(parts[name]).all()): raise ValueError('Nonfinite edge-conditioned '+name)
    active = out['common_support'].flatten(1)>0
    long0 = order[:,:,0]==0
    mixed = (long0 & active).any(1) & ((~long0) & active).any(1)
    return size, parts['alignment'].mean(), parts['proxy'].mean(), dict(anchors=n,
        edge_descriptors=2*n,zero_embeddings=0,zero_proxy_embeddings=0,
        minimum_sample_norm=float(sn.detach().min()), minimum_proxy_norm=float(pn.detach().min()),
        mixed_role_views=int(mixed.sum()))


def update(head, bank, optimizer, x, weight_alignment=None, weight_proxy=None):
    wa = SETTINGS['weight_alignment'] if weight_alignment is None else weight_alignment
    wp = SETTINGS['weight_proxy'] if weight_proxy is None else weight_proxy
    if wa == 0 and wp == 0:
        from crane_project.tools.run_symeood_110_lvsr_v1 import update as original_update
        return original_update(head, optimizer, x, 'fixed_b', torch)
    optimizer.zero_grad()
    size, align, proxy, stats = objectives(head, bank, x)
    loss = prime.combined_loss(size, align, proxy, wa, wp)
    if not bool(torch.isfinite(loss)): raise ValueError('Nonfinite edge-conditioned total loss')
    loss.backward()
    if any(q.grad is None or not bool(torch.isfinite(q.grad).all()) for q in list(head.parameters())+list(bank.parameters())):
        raise ValueError('Invalid edge-conditioned head/proxy gradient')
    stem = math.sqrt(sum(float(q.grad.double().square().sum()) for q in head.stem.parameters()))
    terminal = math.sqrt(sum(float(q.grad.double().square().sum()) for q in head.terminal.parameters()))
    head_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 10.))
    proxy_norm = float(torch.nn.utils.clip_grad_norm_(bank.parameters(), SETTINGS['proxy_clip_norm']))
    optimizer.step()
    if any(not bool(torch.isfinite(q).all()) for q in list(head.parameters())+list(bank.parameters())):
        raise ValueError('Nonfinite updated edge-conditioned parameter')
    return dict(loss=float(loss.detach()),size_loss=float(size.detach()),alignment_loss=float(align.detach()),
        proxy_loss=float(proxy.detach()),weighted_alignment_loss=wa*float(align.detach()),weighted_proxy_loss=wp*float(proxy.detach()),
        stem_grad_norm=stem,terminal_grad_norm=terminal,grad_norm=head_norm,proxy_grad_norm=proxy_norm,**stats)
