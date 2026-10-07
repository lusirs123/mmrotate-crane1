"""Size-only continuous opposite-border readout, with direct decoded supervision.

The frozen hard-bucket module supplies sampling and the identical 2760-parameter
stem/readouts. This experiment changes readout and its corresponding loss only.
GT, domain, video identity and downstream formulas never enter forward().
"""
import torch
from torch.nn import functional as F
from crane_project.utils import port_geometry_size_boundary_v1 as boundary

SETTINGS = dict(seed=1703, in_channels=256, hidden_channels=8, roi_size=9,
                bins=9, log_range=.5, tangent_samples=9, prior_sigma_bins=1.,
                decoded_beta=.1)
PARAMETER_COUNT = boundary.PARAMETER_COUNT
residual = boundary.residual
size_targets = boundary.size_targets
deliver_sizes = boundary.deliver_sizes
boundary_targets = boundary.boundary_targets
bucket_grid = boundary.bucket_grid
border_points = boundary.border_points
pool_borders = boundary.pool_borders


def continuous_decode(logits, fine, grid, prior):
    """Differentiable log-size mean, centered at the exact initial expectation.

    Use the same expanded shape and reduction for the reference expectation:
    symmetry alone need not give bitwise zero in float32. No detach, argmax,
    neutral-value branch or target clipping obstructs the decoded gradient.
    """
    probability = logits.softmax(-1)
    reference = prior.expand_as(logits).softmax(-1)
    initial_mean = (reference*grid).sum(-1)
    coarse_mean = (probability*grid).sum(-1)-initial_mean
    fine_mean = (probability*fine).sum(-1)
    delta = (probability*(grid+fine)).sum(-1)-initial_mean
    return dict(delta_roi=delta, bucket_probability=probability,
                expected_bucket_log=coarse_mean, expected_fine_log=fine_mean,
                initial_expected_log=initial_mean)


def continuous_loss(output, gt_original, midpoint_original, boxes_model):
    target = boundary_targets(gt_original, midpoint_original, boxes_model)
    delta = output['delta_roi']
    if (not len(midpoint_original) or delta.shape != target['log_residual'].shape or
            not bool(torch.isfinite(delta).all())):
        raise ValueError('Expected nonempty finite decoded size pair')
    decoded = F.smooth_l1_loss(delta, target['log_residual'],
                              beta=SETTINGS['decoded_beta'], reduction='mean')
    return dict(total=decoded, decoded=decoded, target=target)


class ContinuousBoundarySizeHead(boundary.BoundarySizeHead):
    """Same initialized stem/terminal weights, continuous extent delivery."""
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
            logits, fine = prediction[..., 0]+prior, prediction[..., 1]
        else:
            logits, fine = b.new_empty(0, 2, k), b.new_empty(0, 2, k)
            coverage = b.new_empty(0, 2, k, 2)
            locations = b.new_empty(0, 2, 2, k, SETTINGS['tangent_samples'], 2)
        decoded = continuous_decode(logits, fine, grid, prior)
        valid = (torch.isfinite(logits).reshape(n, 2*k).all(-1) &
                 torch.isfinite(fine).reshape(n, 2*k).all(-1))
        decoded['delta_roi'] = torch.where(valid[:, None], decoded['delta_roi'],
                                         torch.full_like(decoded['delta_roi'], float('nan')))
        delivery = deliver_sizes(decoded['delta_roi'], b, boxes_model, scale_xy, m)
        # Mode is diagnostic only; it does not participate in decoded delivery.
        index = logits.argmax(-1) if n else torch.empty(0, 2, dtype=torch.long, device=b.device)
        return dict(delivery, **decoded, bucket_logits=logits, fine_log=fine,
                    bucket_index=index, border_support=coverage,
                    border_locations_roi=locations, geometry_inputs=geometry)
