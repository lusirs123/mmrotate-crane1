"""One bounded change to the sealed R readout: geometry times bucket position.

All four geometry entries are retained, including the potentially redundant
context ratios. No extra parameters, GT, domain, video or temporal inputs.
"""
import torch
from crane_project.utils import port_geometry_size_distribution_v1 as base

VERSION = 'port_geometry_size_condition_v1'
PARAMETER_COUNT = base.PARAMETER_COUNT


class GeometryConditionSizeHead(base.DistributionSizeHead):
    def forward(self, roi, support, boxes_original, boxes_model, scale_xy, midpoint_original):
        b, cb, xy, sides, rotation, m = base.residual.checked_inputs(
            roi, support, boxes_original, boxes_model, scale_xy, midpoint_original)
        n, r, k = len(b), base.SETTINGS['roi_size'], base.SETTINGS['bins']
        grid = base.bucket_grid(m)
        coord = torch.linspace(-.5, .5, r, device=b.device, dtype=b.dtype)
        yy, xx = torch.meshgrid(coord, coord)
        fixed = torch.stack((xx, yy), 0)[None].expand(n, -1, -1, -1)
        mu, bu = (base.residual.raw_roi_pair(x[:, 2:4], boxes_model) for x in (m, b))
        geometry = torch.cat(((mu/bu).log(), (cb[:, 2:4]/sides).log()), -1).detach()
        if n:
            features = self.stem(torch.cat((roi.detach(), support.detach(), fixed), 1))
            paired, coverage = base.pool_borders(features, support, base.border_points(m, boxes_model, grid),
                                                cb, xy, sides, rotation)
            # Sole forward change versus R: the geometry term varies by bucket.
            conditioned = geometry[:, None, None]*grid[None, None, :, None]
            inputs = torch.cat((paired, conditioned.expand(n, 2, k, 4), coverage,
                                grid[None, None, :, None].expand(n, 2, k, 1)), -1)
            logits = torch.stack((self.u(inputs[:, 0]).squeeze(-1), self.v(inputs[:, 1]).squeeze(-1)), 1)
        else:
            logits, coverage = b.new_empty(0, 2, k), b.new_empty(0, 2, k, 2)
        decoded = base.decode(logits, grid)
        decoded['delta_roi'] = torch.where(torch.isfinite(logits).reshape(n, 2*k).all(-1)[:, None],
            decoded['delta_roi'], torch.full_like(decoded['delta_roi'], float('nan')))
        delivery = base.residual.deliver_sizes(decoded['delta_roi'], b, boxes_model, xy, m)
        return dict(delivery, **decoded, bucket_logits=logits, grid=grid,
                    border_support=coverage, geometry_inputs=geometry)
