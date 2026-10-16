"""DFL-style continuous log-size targets on frozen rotated M axes.

A bounded mechanism contrast, not a reproduction of D-FINE/FGL. The two arms
share this exact forward and decoder. GT enters loss/targets only. Python3.8+
and torch1.8+. Old source modules and delivery guards remain unchanged.
"""
import torch
from torch import nn
from torch.nn import functional as F
from crane_project.utils import port_geometry_midpoint_edge_residual_v1 as residual

VERSION = 'port_geometry_size_distribution_v1'
SETTINGS = dict(seed=1703, in_channels=256, hidden_channels=8, roi_size=9,
                bins=33, log_range=1., tangent_samples=9, decoded_beta=.1)
PARAMETER_COUNT = 2712
ARMS = ('decoded_control', 'distribution')
size_targets = residual.size_targets


def bucket_grid(reference):
    return torch.linspace(-SETTINGS['log_range'], SETTINGS['log_range'],
                          SETTINGS['bins'], dtype=reference.dtype, device=reference.device)


def decode(logits, grid):
    probability = logits.softmax(-1)
    reference = torch.zeros_like(logits).softmax(-1)
    neutral = (reference*grid).sum(-1)
    return dict(delta_roi=(probability*grid).sum(-1)-neutral,
                bucket_probability=probability, initial_expected_log=neutral)


def distribution_targets(target, logits, grid):
    """Exact adjacent-bin barycentric labels for the SAME centered decoder.

    No GT clipping and no hard-bucket fine branch. Out-of-range GT aborts.
    Endpoint clamps apply to bracket indices only, not target values.
    """
    if target.shape != logits.shape[:-1] or not target.numel() or not bool(torch.isfinite(target).all()):
        raise ValueError('Nonempty finite two-axis target required')
    neutral = (torch.zeros_like(logits).softmax(-1)*grid).sum(-1)
    value = target.detach()+neutral.detach()
    if bool(((value < grid[0]) | (value > grid[-1])).any()):
        raise ValueError('GT outside fixed distribution support; no clipping permitted')
    right = torch.searchsorted(grid.contiguous(), value.contiguous(), right=True).clamp(1, len(grid)-1)
    left = right-1
    wr = (value-grid[left])/(grid[right]-grid[left])
    wl = 1-wr
    return dict(left=left, right=right, weight_left=wl.detach(), weight_right=wr.detach(),
                reconstructed=wl*grid[left]+wr*grid[right]-neutral)


def loss(output, gt_original, midpoint_original, boxes_model, arm):
    if arm not in ARMS:
        raise ValueError('Unknown fixed supervision arm')
    target = size_targets(gt_original, midpoint_original, boxes_model)['log_residual']
    logits, delta = output['bucket_logits'], output['delta_roi']
    if not bool(torch.isfinite(logits).all() and torch.isfinite(delta).all()):
        raise ValueError('Nonfinite size output')
    assigned = distribution_targets(target, logits, output['grid'])
    logp = logits.log_softmax(-1)
    ce = -(assigned['weight_left']*logp.gather(-1, assigned['left'][..., None]).squeeze(-1)+
           assigned['weight_right']*logp.gather(-1, assigned['right'][..., None]).squeeze(-1)).mean()
    decoded = F.smooth_l1_loss(delta, target, beta=SETTINGS['decoded_beta'], reduction='mean')
    return dict(total=ce if arm == 'distribution' else decoded, soft_ce=ce,
                decoded=decoded, target=target, assignment=assigned)


def border_points(m, bm, grid):
    t = torch.linspace(-.5, .5, SETTINGS['tangent_samples'], dtype=m.dtype, device=m.device)
    c, s = m[:, 4].cos(), m[:, 4].sin()
    u, v = torch.stack((c, s), -1), torch.stack((-s, c), -1)
    axes, tangents = torch.stack((u, v), 1), torch.stack((v, u), 1)
    signs = m.new_tensor([1., -1.])
    normal = (m[:, 2:4, None, None, None, None]*.5*axes[:, :, None, None, None, :]*
              signs[None, None, :, None, None, None]*grid.exp()[None, None, None, :, None, None])
    tangent = (m[:, 2:4].flip(-1)[:, :, None, None, None, None]*
               tangents[:, :, None, None, None, :]*t[None, None, None, None, :, None])
    points = m[:, None, None, None, None, :2]+normal+tangent
    return torch.where((bm[:, 2] < bm[:, 3])[:, None, None, None, None, None], points.flip(1), points)


def pool_borders(features, support, points, cb, xy, sides, rotation):
    n, h = features.shape[:2]
    k, p = SETTINGS['bins'], SETTINGS['tangent_samples']
    local = torch.bmm(points.reshape(n, 4*k*p, 2)*xy[:, None]-cb[:, None, :2], rotation)/sides[:, None]
    grid = (2*local).reshape(n, 4*k, p, 2)
    inside = (local.abs() <= .5).all(-1).reshape(n, 1, 4*k, p)
    masks = F.grid_sample(support.detach(), grid, mode='bilinear', padding_mode='zeros',
                          align_corners=True).clamp(0, 1)*inside.to(features.dtype)
    sampled = F.grid_sample(features, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    pooled = (sampled*masks).max(-1).values
    paired = pooled.reshape(n, h, 2, 2, k).permute(0, 2, 4, 3, 1).reshape(n, 2, k, 2*h)
    coverage = masks.mean(-1).reshape(n, 2, 2, k).permute(0, 1, 3, 2)
    return paired, coverage


class DistributionSizeHead(nn.Module):
    def __init__(self):
        super().__init__()
        h = SETTINGS['hidden_channels']
        self.stem = nn.Sequential(nn.Conv2d(259, h, 1), nn.ReLU(inplace=False),
                                  nn.Conv2d(h, h, 3, padding=1), nn.ReLU(inplace=False))
        self.u, self.v = nn.Linear(2*h+7, 1), nn.Linear(2*h+7, 1)
        for layer in (self.u, self.v):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        if sum(p.numel() for p in self.parameters()) != PARAMETER_COUNT:
            raise ValueError('Size distribution parameter contract differs')

    def forward(self, roi, support, boxes_original, boxes_model, scale_xy, midpoint_original):
        b, cb, xy, sides, rotation, m = residual.checked_inputs(
            roi, support, boxes_original, boxes_model, scale_xy, midpoint_original)
        n, r, k = len(b), SETTINGS['roi_size'], SETTINGS['bins']
        grid = bucket_grid(m)
        coord = torch.linspace(-.5, .5, r, device=b.device, dtype=b.dtype)
        yy, xx = torch.meshgrid(coord, coord)
        fixed = torch.stack((xx, yy), 0)[None].expand(n, -1, -1, -1)
        mu, bu = (residual.raw_roi_pair(x[:, 2:4], boxes_model) for x in (m, b))
        geometry = torch.cat(((mu/bu).log(), (cb[:, 2:4]/sides).log()), -1).detach()
        if n:
            features = self.stem(torch.cat((roi.detach(), support.detach(), fixed), 1))
            paired, coverage = pool_borders(features, support, border_points(m, boxes_model, grid),
                                             cb, xy, sides, rotation)
            inputs = torch.cat((paired, geometry[:, None, None].expand(n, 2, k, 4),
                                coverage, grid[None, None, :, None].expand(n, 2, k, 1)), -1)
            logits = torch.stack((self.u(inputs[:, 0]).squeeze(-1), self.v(inputs[:, 1]).squeeze(-1)), 1)
        else:
            logits, coverage = b.new_empty(0, 2, k), b.new_empty(0, 2, k, 2)
        decoded = decode(logits, grid)
        decoded['delta_roi'] = torch.where(torch.isfinite(logits).reshape(n, 2*k).all(-1)[:, None],
            decoded['delta_roi'], torch.full_like(decoded['delta_roi'], float('nan')))
        delivery = residual.deliver_sizes(decoded['delta_roi'], b, boxes_model, xy, m)
        return dict(delivery, **decoded, bucket_logits=logits, grid=grid,
                    border_support=coverage, geometry_inputs=geometry)
