"""S034: local vectors to annotated OBB side midpoints, sizes only delivered.

Forward is GT-free. D4 matching is OFFLINE supervision, not online matching.
These rectangle midpoints are not assumed to be visible physical keypoints.
"""
import torch
from torch import nn
from torch.nn import functional as F
from crane_project.utils import port_geometry_midpoint_edge_residual_v1 as residual

ARMS = ('size_only', 'size_plus_geometry')
PARAMETER_COUNT = 2800
GEOMETRY_WEIGHT = .1
PERMUTATIONS = tuple(tuple((s + direction*j) % 4 for j in range(4))
                     for direction in (1, -1) for s in range(4))


def grid(reference):
    c = torch.linspace(-.5, .5, 9, dtype=reference.dtype, device=reference.device)
    yy, xx = torch.meshgrid(c, c)
    return torch.stack((xx, yy), -1).reshape(1, 81, 2)


def encode(boxes, b, bm, xy):
    """OFFLINE for GT, also used online for the frozen M neutral reference."""
    q = residual.original.box_midpoints(boxes[:, :5])
    q = residual.original.original_to_roi(q, b, bm, xy)
    return q[:, None] - grid(b)[:, :, None]


def decode(vectors, support, b, bm, xy):
    """Inverse anisotropic affine FIRST, per-location norm/sort, fixed mean.

    Any supported invalid/degenerate pair invalidates the whole frame. Zero
    support never supplies a learned or GT-selected quality weight.
    """
    n = len(b)
    if vectors.shape != (n, 81, 4, 2):
        raise ValueError('Expected N x 81 x 4 x 2 vectors')
    weights = support.detach().reshape(n, 81).clamp(0, 1)
    active = weights > 0
    finite = torch.isfinite(vectors).all(-1).all(-1)
    safe = torch.where(torch.isfinite(vectors), vectors, torch.zeros_like(vectors))
    restored = residual.original.roi_delta_to_original(safe.reshape(n, 324, 2), b, bm, xy).reshape(n, 81, 4, 2)
    lengths = torch.stack(((restored[:, :, 0]-restored[:, :, 2]).norm(dim=-1),
                           (restored[:, :, 1]-restored[:, :, 3]).norm(dim=-1)), -1)
    local = lengths.sort(dim=-1, descending=True)[0]
    valid = ((finite & (lengths > 1e-6).all(-1)) | ~active).all(-1) & (weights.sum(-1) > 0)
    mean = (local*weights[:, :, None]).sum(1)/weights.sum(-1).clamp(min=1e-12)[:, None]
    variance = ((local-mean[:, None]).square()*weights[:, :, None]).sum(1)/weights.sum(-1).clamp(min=1e-12)[:, None]
    return dict(sizes=mean, local_sizes=local, valid=valid, variance=variance)


def geometric_loss(vectors, gt, support, b, bm, xy):
    """Per-location min over D4, mean8 coordinates, support-normalized meanN.

    A continuous piecewise-differentiable envelope; ties select the first
    branch. Targets outside the ROI remain unbounded continuous vectors.
    """
    target = encode(gt.detach(), b, bm, xy).detach()
    costs = torch.stack([F.smooth_l1_loss(vectors, target[:, :, list(p)],
                         beta=.1, reduction='none').mean((-1, -2)) for p in PERMUTATIONS], -1)
    best, choice = costs.min(-1)
    w = support.detach().reshape(len(b), 81).clamp(0, 1)
    if bool((w.sum(-1) == 0).any()):
        raise ValueError('No effective support in training batch')
    return ((best*w).sum(-1)/w.sum(-1)).mean(), choice


class LocalVectorSizeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(267, 8, 1), nn.ReLU(),
                                  nn.Conv2d(8, 8, 3, padding=1), nn.ReLU())
        self.terminal = nn.Conv2d(8, 8, 1)
        nn.init.zeros_(self.terminal.weight); nn.init.zeros_(self.terminal.bias)
        assert sum(p.numel() for p in self.parameters()) == PARAMETER_COUNT

    def forward(self, roi, support, b, bm, xy, m):
        b, _, _, _, _, m = residual.checked_inputs(roi, support, b, bm, xy, m)
        n = len(b)
        if not n:
            delta = b.new_empty(0, 2)
            return dict(residual.deliver_sizes(delta, b, bm, xy, m), delta_roi=delta,
                        vectors=b.new_empty(0, 81, 4, 2), vector_valid=b.new_empty(0, dtype=torch.bool))
        base = encode(m, b, bm, xy).detach()
        coords = grid(b).expand(n, -1, -1).transpose(1, 2).reshape(n, 2, 9, 9)
        channels = base.reshape(n, 81, 8).transpose(1, 2).reshape(n, 8, 9, 9)
        features = self.stem(torch.cat((roi.detach(), support.detach(), coords, channels), 1))
        change = self.terminal(features).permute(0, 2, 3, 1).reshape(n, 81, 4, 2)
        vectors = base + change
        decoded = decode(vectors, support, b, bm, xy)
        reference = decode(base, support, b, bm, xy)['sizes'].detach()
        canonical = (decoded['sizes'].clamp(min=1e-6)/reference.clamp(min=1e-6)).log()
        raw = torch.where((m[:, 2] < m[:, 3])[:, None], canonical.flip(-1), canonical)
        delta = residual.raw_roi_pair(raw, bm)
        delivery_delta = torch.where(decoded['valid'][:, None], delta, torch.full_like(delta, float('nan')))
        out = residual.deliver_sizes(delivery_delta, b, bm, xy, m)
        return dict(out, delta_roi=delta, vectors=vectors, vector_valid=decoded['valid'],
                    decoded_sizes=decoded['sizes'], local_size_variance=decoded['variance'])


def losses(output, gt, support, b, bm, xy, m, arm):
    if arm not in ARMS: raise ValueError('Unknown arm')
    if not bool(output['vector_valid'].all()) or not bool(torch.isfinite(output['vectors']).all()):
        raise ValueError('Invalid local vectors during training')
    size = residual.size_loss(output['delta_roi'], gt, m, bm)
    geometry, _ = geometric_loss(output['vectors'], gt, support, b, bm, xy)
    total = size + (GEOMETRY_WEIGHT*geometry if arm == ARMS[1] else 0.)
    return total, size, geometry
