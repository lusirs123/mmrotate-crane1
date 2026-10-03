"""Finite spatial midpoint readout of frozen aligned ROIs.

Online inputs contain no GT. Supervision and permutation matching are separate
offline functions. The four locations are OBB geometry, not visible keypoints.
Python 3.8 / torch 1.8 compatible; no detector integration or checkpoint export.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from crane_project.utils.port_geometry_refine_g_v1 import checked_original, wrap_pi
from crane_project.utils.port_structure_reliability_v1 import canonical_boxes

SETTINGS = dict(seed=1703, permutation_seed=1704, in_channels=256,
    hidden_channels=32, roi_size=9, context_multiplier=1.5,
    stride=8, min_context_cells=2., prior_sigma_cells=1., loss_beta=.1,
    steps_per_arm=200, batch_size=8, lr=.001, weight_decay=0., clip_norm=10.,
    max_center_over_b_short=.30, max_log_edge_residual=math.log(1.25),
    max_angle_residual_rad=math.pi/18, min_pair_length_px=1e-6,
    roi_bound_tolerance=1e-5)
PARAMETER_COUNT = 17696
POINT_NAMES = ('positive_b_long', 'positive_b_short',
               'negative_b_long', 'negative_b_short')


def box_midpoints(boxes):
    """Cyclic (+u,+v,-u,-v) midpoints; canonicalization is geometric only."""
    b = canonical_boxes(boxes)
    c, s = b[:, 4].cos(), b[:, 4].sin()
    u = torch.stack((c, s), dim=-1)*b[:, 2:3]*.5
    v = torch.stack((-s, c), dim=-1)*b[:, 3:4]*.5
    return b[:, None, :2]+torch.stack((u, v, -u, -v), dim=1)


def frame(boxes_original, boxes_model, scale_xy):
    checked_original(boxes_original)
    b, bm, scales = (x.detach() for x in (boxes_original, boxes_model, scale_xy))
    n = len(b)
    if (bm.shape != (n, 5) or scales.shape != (n, 2) or
            any(x.device != b.device or x.dtype != b.dtype for x in (bm, scales)) or
            not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any())):
        raise ValueError('Invalid detached model/original frame or xy scales')
    canonical_boxes(bm)
    expected = b[:, :4]*scales[:, [0, 1, 0, 1]]
    if (not torch.allclose(bm[:, :4], expected, atol=1e-4, rtol=1e-6) or
            not torch.allclose(wrap_pi(bm[:, 4]-b[:, 4]), torch.zeros_like(bm[:, 4]), atol=1e-5, rtol=0.)):
        raise ValueError('Raw w/h-angle restoration differs from frozen B')
    cb = canonical_boxes(bm)
    sides = (cb[:, 2:4]*SETTINGS['context_multiplier']).clamp(
        min=SETTINGS['stride']*SETTINGS['min_context_cells'])
    c, s = cb[:, 4].cos(), cb[:, 4].sin()
    # Columns are aligned model-space unit axes. Original-point transformations
    # use separate sx/sy, preserving native integer-resize rounding exactly.
    rotation = torch.stack((torch.stack((c, -s), -1),
                            torch.stack((s, c), -1)), dim=1)
    return b, cb, scales, sides, rotation


def original_to_roi(points, boxes_original, boxes_model, scale_xy):
    _, cb, scales, sides, rotation = frame(boxes_original, boxes_model, scale_xy)
    if points.shape != (len(cb), 4, 2) or not bool(torch.isfinite(points).all()):
        raise ValueError('Expected four finite original xy points')
    delta = points*scales[:, None]-cb[:, None, :2]
    return torch.bmm(delta, rotation)/sides[:, None]


def roi_delta_to_original(delta, boxes_original, boxes_model, scale_xy):
    _, _, scales, sides, rotation = frame(boxes_original, boxes_model, scale_xy)
    return torch.bmm(delta*sides[:, None], rotation.transpose(1, 2))/scales[:, None]


def target_points(gt_original, boxes_original, boxes_model, scale_xy):
    """OFFLINE only. Match all four cyclic permutations to frozen B anchors.

    Width exchange/pi shifts preserve the set. Squares require no artificial
    long-axis label: ties use the first cyclic permutation in canonical order.
    Matching never depends on the trained prediction, epoch or probe outcomes.
    """
    gt = gt_original.detach()
    anchors = box_midpoints(boxes_original.detach()[:, :5])
    reference = original_to_roi(anchors, boxes_original, boxes_model, scale_xy)
    points = box_midpoints(gt)
    if len(points) != len(anchors):
        raise ValueError('GT/B point count differs')
    candidates = torch.stack([torch.roll(points, shifts=-i, dims=1) for i in range(4)], 1)
    local = torch.stack([original_to_roi(candidates[:, i], boxes_original,
                                       boxes_model, scale_xy) for i in range(4)], 1)
    cost = (local-reference[:, None]).square().sum(dim=(2, 3))
    choice = cost.argmin(dim=1)
    index = torch.arange(len(points), device=points.device)
    return dict(original=candidates[index, choice].detach(),
                roi=local[index, choice].detach(), permutation=choice.detach(),
                reference_roi=reference.detach())


def point_loss(predicted_roi, targets):
    if predicted_roi.shape != targets['roi'].shape or not predicted_roi.numel():
        raise ValueError('Expected matched nonempty four-point targets')
    # Equal point/xy weight in fixed ROI-context units, no B-short normalization,
    # classification, size, angle or IoU auxiliary objective.
    parts = F.smooth_l1_loss(predicted_roi, targets['roi'].detach(),
        beta=SETTINGS['loss_beta'], reduction='none').mean(dim=(0, 2))
    return parts.mean(), parts


def point_prior(reference):
    """Fixed B-only prior and its exact truncated-grid neutral expectation."""
    r = SETTINGS['roi_size']
    coord = torch.linspace(-.5, .5, r, dtype=reference.dtype, device=reference.device)
    yy, xx = torch.meshgrid(coord, coord)
    grid = torch.stack((xx, yy), -1).reshape(r*r, 2)
    sigma = SETTINGS['prior_sigma_cells']/(r-1)
    prior = -.5*(grid[None, None]-reference.detach()[:, :, None]).square().sum(-1)/sigma**2
    return grid, prior, torch.matmul(prior.softmax(-1), grid)


def decode_points(points, boxes_original, local_points, zero_delta):
    """GT-free fixed rectangle projection and conservative full-B fallback.

    Center=mean4. Pair vectors u=p0-p2,v=p1-p3 yield
    theta=atan2(u_y-v_x,u_x+v_y); positive projected lengths reconstruct a
    rectangle. Raw B w/h association is retained even across a size crossing.
    Reject the entire candidate if finite/positive/ROI/center/size/angle checks
    fail; no clipping, confidence threshold or GT-selected acceptance.
    """
    b = boxes_original.detach()
    checked_original(b)
    if points.shape != (len(b), 4, 2) or local_points.shape != points.shape:
        raise ValueError('Invalid four-point decode shape')
    u, v = points[:, 0]-points[:, 2], points[:, 1]-points[:, 3]
    center = points.mean(dim=1)
    direction = torch.stack((u[:, 0]+v[:, 1], u[:, 1]-v[:, 0]), -1)
    theta = torch.atan2(direction[:, 1], direction[:, 0])
    axis = torch.stack((theta.cos(), theta.sin()), -1)
    normal = torch.stack((-theta.sin(), theta.cos()), -1)
    width, height = (u*axis).sum(-1), (v*normal).sum(-1)
    swap = b[:, 2] < b[:, 3]
    raw_angle = wrap_pi(theta-swap.to(b.dtype)*math.pi/2)
    # Equivalent directions are aligned to the original B representation.
    angle = b[:, 4]+wrap_pi(raw_angle-b[:, 4])
    candidate = torch.stack((center[:, 0], center[:, 1],
        torch.where(swap, height, width), torch.where(swap, width, height), angle, b[:, 5]), -1)
    # At initialization exact B must not depend on trigonometric round-off.
    candidate = torch.where(zero_delta[:, None], b, candidate)
    finite = torch.isfinite(candidate).all(-1) & torch.isfinite(points).reshape(len(b), 8).all(-1)
    positive = (width > SETTINGS['min_pair_length_px']) & (height > SETTINGS['min_pair_length_px'])
    direction_valid = direction.norm(dim=1) > SETTINGS['min_pair_length_px']
    inside = (local_points.abs() <= .5+SETTINGS['roi_bound_tolerance']).reshape(len(b), 8).all(-1)
    short = torch.minimum(b[:, 2], b[:, 3])
    center_ok = (candidate[:, :2]-b[:, :2]).norm(dim=1) <= SETTINGS['max_center_over_b_short']*short+1e-6
    ratios = candidate[:, 2:4]/b[:, 2:4]
    size_ok = ((ratios >= math.exp(-SETTINGS['max_log_edge_residual'])-1e-6) &
               (ratios <= math.exp(SETTINGS['max_log_edge_residual'])+1e-6)).all(-1)
    angle_ok = wrap_pi(candidate[:, 4]-b[:, 4]).abs() <= SETTINGS['max_angle_residual_rad']+1e-6
    checks = dict(finite=finite, positive=positive, direction_valid=direction_valid,
                  roi_inside=inside, center_bound=center_ok, edge_bound=size_ok, angle_bound=angle_ok)
    accepted = torch.stack(list(checks.values()), -1).all(-1)
    output = torch.where(accepted[:, None], candidate, b)
    checked_original(output)
    if not torch.equal(output[:, 5], b[:, 5]):
        raise ValueError('Point decoding changed frozen B score')
    pu, pv = axis*width[:, None]*.5, normal*height[:, None]*.5
    projected = center[:, None]+torch.stack((pu, pv, -pu, -pv), 1)
    return dict(
        boxes_original=output, candidate_original=candidate,
        candidate_valid=finite & positive & direction_valid,
        accepted=accepted, checks=checks,
        rectangle_projection_rms_px=(points-projected).square().sum(-1).mean(-1).sqrt())


class SpatialMidpointHead(nn.Module):
    """Shared conv -> four9x9 logits -> residual continuous expectations.

    p = B anchor + E(softmax(B prior+logits))-E(softmax(B prior)). Zero output
    weights give bitwise zero correction with nonzero point gradients.
    Coordinate channels are fixed local x/y, not labels or extra resolution.
    """
    def __init__(self):
        super().__init__()
        h = SETTINGS['hidden_channels']
        self.stem = nn.Sequential(nn.Conv2d(259, h, 1), nn.ReLU(inplace=False),
                                  nn.Conv2d(h, h, 3, padding=1), nn.ReLU(inplace=False))
        # Per-map bias cancels exactly in softmax and would be unidentifiable.
        self.output = nn.Conv2d(h, 4, 1, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, roi, support, boxes_original, boxes_model, scale_xy):
        b, _, _, _, _ = frame(boxes_original, boxes_model, scale_xy)
        n, r = len(b), SETTINGS['roi_size']
        if (roi.shape != (n, 256, r, r) or support.shape != (n, 1, r, r) or
                any(x.device != b.device or x.dtype != b.dtype for x in (roi, support)) or
                not bool(torch.isfinite(roi).all()) or not bool(torch.isfinite(support).all()) or
                bool(((support < -1e-6) | (support > 1+1e-6)).any())):
            raise ValueError('Invalid frozen local ROI/support')
        anchors = box_midpoints(b[:, :5])
        reference = original_to_roi(anchors, b, boxes_model, scale_xy)
        # Actual local locations, not displacement bins centered at the ROI
        # origin. A fixed B-only Gaussian prior makes each map point-specific.
        # Subtract its own expectation for exact identity despite truncation.
        grid, prior, neutral = point_prior(reference)
        fixed = grid.transpose(0, 1).reshape(1, 2, r, r).expand(n, -1, -1, -1)
        if n:
            logits = self.output(self.stem(torch.cat((roi.detach(), support.detach(), fixed), 1)))
        else:
            logits = b.new_empty(0, 4, r, r)
        probability = (logits.reshape(n, 4, r*r)+prior).softmax(-1)
        expectation = torch.matmul(probability, grid)
        delta = expectation-neutral
        local_points = reference+delta
        original = anchors+roi_delta_to_original(delta, b, boxes_model, scale_xy)
        decoded = decode_points(original, b, local_points, (delta == 0).reshape(n, 8).all(-1))
        return dict(decoded, points_original=original, points_roi=local_points,
                    point_delta_roi=delta, anchor_roi=reference, neutral_roi=neutral,
                    probabilities=probability.reshape(n, 4, r, r))
