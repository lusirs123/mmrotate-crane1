"""Same-capacity midpoint control: uniform neutral integral residual readout.

Only the Gaussian prior is removed. B anchors, neutral subtraction, network,
coordinate transforms and whole-box fallback remain. This does not separately
ablate the subtraction, and changes the probability/gradient/displacement range.
Historical midpoint and sigma implementations are deliberately left untouched.
"""
import torch

from crane_project.utils import port_geometry_midpoint_v1 as original

READOUT = 'uniform_neutral_residual'


def uniform_prior(reference):
    r = original.SETTINGS['roi_size']
    coord = torch.linspace(-.5, .5, r, dtype=reference.dtype, device=reference.device)
    yy, xx = torch.meshgrid(coord, coord)
    grid = torch.stack((xx, yy), -1).reshape(r*r, 2)
    prior = reference.new_zeros(len(reference), 4, r*r)
    # Evaluate on the actual finite grid. Hardcoding zero would lose the
    # bitwise identity guarantee if the reduction has rounding residuals.
    neutral = torch.matmul(prior.softmax(-1), grid)
    return grid, prior, neutral


class UniformMidpointHead(original.SpatialMidpointHead):
    readout = READOUT

    def forward(self, roi, support, boxes_original, boxes_model, scale_xy):
        b, _, _, _, _ = original.frame(boxes_original, boxes_model, scale_xy)
        n, r = len(b), original.SETTINGS['roi_size']
        if (roi.shape != (n, 256, r, r) or support.shape != (n, 1, r, r) or
                any(x.device != b.device or x.dtype != b.dtype for x in (roi, support)) or
                not bool(torch.isfinite(roi).all()) or not bool(torch.isfinite(support).all()) or
                bool(((support < -1e-6) | (support > 1+1e-6)).any())):
            raise ValueError('Invalid frozen local ROI/support')
        anchors = original.box_midpoints(b[:, :5])
        reference = original.original_to_roi(anchors, b, boxes_model, scale_xy)
        grid, prior, neutral = uniform_prior(reference)
        fixed = grid.transpose(0, 1).reshape(1, 2, r, r).expand(n, -1, -1, -1)
        if n:
            logits = self.output(self.stem(torch.cat((roi.detach(), support.detach(), fixed), 1)))
        else:
            logits = b.new_empty(0, 4, r, r)
        probability = (logits.reshape(n, 4, r*r)+prior).softmax(-1)
        delta = torch.matmul(probability, grid)-neutral
        local_points = reference+delta
        points = anchors+original.roi_delta_to_original(delta, b, boxes_model, scale_xy)
        decoded = original.decode_points(points, b, local_points,
            (delta == 0).reshape(n, 8).all(-1))
        return dict(decoded, points_original=points, points_roi=local_points,
            point_delta_roi=delta, anchor_roi=reference, neutral_roi=neutral,
            probabilities=probability.reshape(n, 4, r, r))
