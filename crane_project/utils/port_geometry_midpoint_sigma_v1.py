"""Original midpoint head with one explicit, checkpoint-bound prior width.

No global SETTINGS mutation. Sigma=1 delegates to the historical implementation
exactly; other widths change only the B-only prior and its neutral expectation.
"""
import math

import torch

from crane_project.utils import port_geometry_midpoint_v1 as original

SIGMAS = (.5, 1., 1.5)


def checked_sigma(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('Expected one fixed numeric sigma')
    value = float(value)
    if not math.isfinite(value) or value not in SIGMAS:
        raise ValueError('Only fixed sigma .5/1/1.5 is allowed')
    return value


def point_prior(reference, sigma_cells):
    sigma_cells = checked_sigma(sigma_cells)
    if sigma_cells == 1.:
        return original.point_prior(reference)
    r = original.SETTINGS['roi_size']
    coord = torch.linspace(-.5, .5, r, dtype=reference.dtype, device=reference.device)
    yy, xx = torch.meshgrid(coord, coord)
    grid = torch.stack((xx, yy), -1).reshape(r*r, 2)
    sigma = sigma_cells/(r-1)
    prior = -.5*(grid[None, None]-reference.detach()[:, :, None]).square().sum(-1)/sigma**2
    return grid, prior, torch.matmul(prior.softmax(-1), grid)


class SigmaMidpointHead(original.SpatialMidpointHead):
    def __init__(self, sigma_cells):
        self.prior_sigma_cells = checked_sigma(sigma_cells)
        super().__init__()

    def forward(self, roi, support, boxes_original, boxes_model, scale_xy):
        # Makes default checkpoint/state/forward reuse exact, not approximate.
        if self.prior_sigma_cells == 1.:
            return super().forward(roi, support, boxes_original, boxes_model, scale_xy)
        b, _, _, _, _ = original.frame(boxes_original, boxes_model, scale_xy)
        n, r = len(b), original.SETTINGS['roi_size']
        if (roi.shape != (n, 256, r, r) or support.shape != (n, 1, r, r) or
                any(x.device != b.device or x.dtype != b.dtype for x in (roi, support)) or
                not bool(torch.isfinite(roi).all()) or not bool(torch.isfinite(support).all()) or
                bool(((support < -1e-6) | (support > 1+1e-6)).any())):
            raise ValueError('Invalid frozen local ROI/support')
        anchors = original.box_midpoints(b[:, :5])
        reference = original.original_to_roi(anchors, b, boxes_model, scale_xy)
        grid, prior, neutral = point_prior(reference, self.prior_sigma_cells)
        fixed = grid.transpose(0, 1).reshape(1, 2, r, r).expand(n, -1, -1, -1)
        if n:
            logits = self.output(self.stem(torch.cat((roi.detach(), support.detach(), fixed), 1)))
        else:
            logits = b.new_empty(0, 4, r, r)
        probability = (logits.reshape(n, 4, r*r)+prior).softmax(-1)
        expectation = torch.matmul(probability, grid)
        delta = expectation-neutral
        local_points = reference+delta
        points = anchors+original.roi_delta_to_original(delta, b, boxes_model, scale_xy)
        decoded = original.decode_points(points, b, local_points,
            (delta == 0).reshape(n, 8).all(-1))
        return dict(decoded, points_original=points, points_roi=local_points,
            point_delta_roi=delta, anchor_roi=reference, neutral_roi=neutral,
            probabilities=probability.reshape(n, 4, r, r))
