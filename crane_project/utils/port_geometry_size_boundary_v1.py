"""Experimental size-only opposite-border readout; no online labels or policy.

Inspired by SABL coarse/fine localization and BorderDet border pooling, adapted
to a fixed M rectangle. This is not a reproduction of either full detector.
Native original/model swaps, anisotropic resize rounding, and old delivery
guards are reused without changing the sealed baseline implementation.
"""
import torch
from torch import nn
from torch.nn import functional as F

from crane_project.utils import port_geometry_midpoint_edge_residual_v1 as residual

SETTINGS = dict(seed=1703, in_channels=256, hidden_channels=8, roi_size=9,
                bins=9, log_range=.5, tangent_samples=9, prior_sigma_bins=1.,
                coarse_weight=.1, fine_beta=.1)
PARAMETER_COUNT = 2760

# OFFLINE target assignment and GT-free delivery are the existing contracts.
size_targets = residual.size_targets
deliver_sizes = residual.deliver_sizes


def bucket_grid(reference):
    return torch.linspace(-SETTINGS['log_range'], SETTINGS['log_range'],
                          SETTINGS['bins'], device=reference.device, dtype=reference.dtype)


def border_points(midpoint_original, boxes_model):
    """N x ROI-axis x +/-side x bucket x tangent x original-xy points.

    Symmetric sides share an extent. Tangential span is the OTHER frozen M
    dimension; neither individual side nor center/orientation is regressed.
    Raw M axes are mapped to model-B ROI order, not original-B canonical order.
    """
    m, bm = midpoint_original.detach(), boxes_model.detach()
    grid = bucket_grid(m)
    t = torch.linspace(-.5, .5, SETTINGS['tangent_samples'],
                       dtype=m.dtype, device=m.device)
    c, s = m[:, 4].cos(), m[:, 4].sin()
    u, v = torch.stack((c, s), -1), torch.stack((-s, c), -1)
    axes = torch.stack((u, v), 1)
    tangents = torch.stack((v, u), 1)
    sign = m.new_tensor([1., -1.])
    normal = (m[:, 2:4, None, None, None, None]*.5 *
              axes[:, :, None, None, None, :] *
              sign[None, None, :, None, None, None] *
              grid.exp()[None, None, None, :, None, None])
    tangent = (m[:, 2:4].flip(-1)[:, :, None, None, None, None] *
               tangents[:, :, None, None, None, :] *
               t[None, None, None, None, :, None])
    points = m[:, None, None, None, None, :2]+normal+tangent
    swap = (bm[:, 2] < bm[:, 3])[:, None, None, None, None, None]
    return torch.where(swap, points.flip(1), points)


def pool_borders(features, support, points, cb, xy, sides, rotation):
    """Sample the actual cached grid; interpolation creates no new evidence.

    Cache sample locations span [-.5,.5] including endpoints, as defined by
    port_geometry_refine_g_v1.sampling_grid(). Thus second-stage sampling uses
    align_corners=True; the parent's P3 sampling uses a DIFFERENT normalized
    grid with the half-cell shift and align_corners=False.
    Support-zero and out-of-ROI locations contribute zero. The ReLU stem is
    nonnegative, so masked zero cannot win over a supported positive feature.
    Fractional support weights sampled features; support means are reported.
    """
    n, h = features.shape[:2]
    k, p = SETTINGS['bins'], SETTINGS['tangent_samples']
    flat = points.reshape(n, 4*k*p, 2)
    local = torch.bmm(flat*xy[:, None]-cb[:, None, :2], rotation)/sides[:, None]
    grid = (2*local).reshape(n, 4*k, p, 2)
    inside = (local.abs() <= .5).all(-1).reshape(n, 1, 4*k, p)
    sampled_support = F.grid_sample(support.detach(), grid, mode='bilinear',
                                   padding_mode='zeros', align_corners=True).clamp(0, 1)
    sampled_support = sampled_support*inside.to(features.dtype)
    sampled = F.grid_sample(features, grid, mode='bilinear',
                            padding_mode='zeros', align_corners=True)
    pooled = (sampled*sampled_support).max(-1).values
    # Axes, side signs, and bins preserve the point construction order.
    paired = pooled.reshape(n, h, 2, 2, k).permute(0, 2, 4, 3, 1).reshape(n, 2, k, 2*h)
    coverage = sampled_support.mean(-1).reshape(n, 2, 2, k).permute(0, 1, 3, 2)
    return paired, coverage, local.reshape(n, 2, 2, k, p, 2)


def boundary_targets(gt_original, midpoint_original, boxes_model):
    """OFFLINE: no clipping/filtering, even for targets outside bucket range.

    Closest endpoint class plus an unrestricted fine target handles overflow.
    This is a virtual GT-size rectangle centered/oriented at M, not GT borders.
    """
    target = size_targets(gt_original, midpoint_original, boxes_model)['log_residual']
    grid = bucket_grid(target)
    index = (target[..., None]-grid).abs().argmin(-1)
    return dict(log_residual=target, bucket_index=index,
                fine_log=target-grid[index],
                outside_grid=target.abs() > SETTINGS['log_range'])


def boundary_loss(output, gt_original, midpoint_original, boxes_model):
    target = boundary_targets(gt_original, midpoint_original, boxes_model)
    logits, fine = output['bucket_logits'], output['fine_log']
    n = len(midpoint_original)
    shape = (n, 2, SETTINGS['bins'])
    if (not n or logits.shape != shape or fine.shape != shape or
            not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(fine).all())):
        raise ValueError('Expected nonempty finite boundary logits/fine offsets')
    coarse = F.cross_entropy(logits.reshape(-1, SETTINGS['bins']),
                             target['bucket_index'].reshape(-1))
    selected = fine.gather(-1, target['bucket_index'][..., None]).squeeze(-1)
    fine_loss = F.smooth_l1_loss(selected, target['fine_log'],
                               beta=SETTINGS['fine_beta'], reduction='mean')
    return dict(total=SETTINGS['coarse_weight']*coarse+fine_loss,
                coarse=coarse, fine=fine_loss, target=target)


class BoundarySizeHead(nn.Module):
    def __init__(self):
        super().__init__()
        h = SETTINGS['hidden_channels']
        self.stem = nn.Sequential(nn.Conv2d(259, h, 1), nn.ReLU(inplace=False),
                                  nn.Conv2d(h, h, 3, padding=1), nn.ReLU(inplace=False))
        # Two side features + original four geometry scalars + side supports
        # + fixed bucket coordinate. Linear weights are shared across buckets.
        self.u, self.v = nn.Linear(2*h+7, 2), nn.Linear(2*h+7, 2)
        for layer in (self.u, self.v):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        if sum(p.numel() for p in self.parameters()) != PARAMETER_COUNT:
            raise ValueError('Boundary parameter count differs')

    def forward(self, roi, support, boxes_original, boxes_model, scale_xy,
                midpoint_original):
        b, cb, xy, sides, rotation, m = residual.checked_inputs(
            roi, support, boxes_original, boxes_model, scale_xy, midpoint_original)
        n, r, k = len(b), SETTINGS['roi_size'], SETTINGS['bins']
        coord = torch.linspace(-.5, .5, r, device=b.device, dtype=b.dtype)
        yy, xx = torch.meshgrid(coord, coord)
        fixed = torch.stack((xx, yy), 0)[None].expand(n, -1, -1, -1)
        mu, bu = (residual.raw_roi_pair(x[:, 2:4], boxes_model) for x in (m, b))
        geometry = torch.cat(((mu/bu).log(), (cb[:, 2:4]/sides).log()), -1).detach()
        grid = bucket_grid(m)
        step = 2*SETTINGS['log_range']/(k-1)
        prior = -.5*(grid/(step*SETTINGS['prior_sigma_bins'])).square()
        if n:
            features = self.stem(torch.cat((roi.detach(), support.detach(), fixed), 1))
            paired, coverage, locations = pool_borders(
                features, support, border_points(m, boxes_model), cb, xy, sides, rotation)
            inputs = torch.cat((paired, geometry[:, None, None].expand(n, 2, k, 4),
                                coverage, grid[None, None, :, None].expand(n, 2, k, 1)), -1)
            prediction = torch.stack((self.u(inputs[:, 0]), self.v(inputs[:, 1])), 1)
            logits = prediction[..., 0]+prior
            fine = prediction[..., 1]
        else:
            logits, fine = b.new_empty(0, 2, k), b.new_empty(0, 2, k)
            coverage = b.new_empty(0, 2, k, 2)
            locations = b.new_empty(0, 2, 2, k, SETTINGS['tangent_samples'], 2)
        # torch1.8 refuses argmax even along a nonempty last axis if N=0.
        index = logits.argmax(-1) if n else torch.empty(0, 2, dtype=torch.long, device=b.device)
        delta = grid[index]+fine.gather(-1, index[..., None]).squeeze(-1)
        valid = torch.isfinite(logits).reshape(n, 2*k).all(-1) & torch.isfinite(fine).reshape(n, 2*k).all(-1)
        delta = torch.where(valid[:, None], delta, torch.full_like(delta, float('nan')))
        delivery = deliver_sizes(delta, b, boxes_model, scale_xy, m)
        return dict(delivery, delta_roi=delta, bucket_logits=logits,
                    bucket_probability=logits.softmax(-1), fine_log=fine,
                    bucket_index=index, border_support=coverage,
                    border_locations_roi=locations, geometry_inputs=geometry)
