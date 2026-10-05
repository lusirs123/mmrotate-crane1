"""Independent size-only residuals after frozen B24 + sigma1.5/epoch03.

The online forward has no GT, domain, sequence, time or calibration inputs.
Only OFFLINE size_targets()/size_loss() read GT. Raw width/height association,
the frozen M long-axis identity, and the actual original/model frame stay
separate. Python3.8 / torch1.8+; this is a finite candidate, not deployment.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from crane_project.utils import port_geometry_midpoint_v1 as original

SETTINGS = dict(seed=1703, in_channels=256, hidden_channels=8, roi_size=9,
    loss_beta=.1, edge_ratio_min=.8, edge_ratio_max=1.25,
    direction_atol_rad=1e-6)
PARAMETER_COUNT = 2818


def raw_roi_pair(values, boxes_model):
    """The model-B raw swap is an involution, for both edges and residuals.

    Do not substitute the original-B swap: actual sx/sy can change ordering.
    """
    if values.shape != (len(boxes_model), 2):
        raise ValueError('Expected one raw/ROI pair per model B')
    return torch.where((boxes_model[:, 2] < boxes_model[:, 3])[:, None],
                       values.flip(-1), values)


def checked_midpoint(b, midpoint_original):
    m = midpoint_original.detach()
    original.checked_original(m)
    if (m.shape != b.shape or m.dtype != b.dtype or m.device != b.device or
            not torch.equal(m[:, 5], b[:, 5])):
        raise ValueError('Invalid frozen M identity or score')
    # Validate the inherited M using the original native-dtype guard arithmetic.
    # New non-neutral size candidates use nominal bounds below, not this slack.
    s = original.SETTINGS
    ratios = m[:, 2:4]/b[:, 2:4]
    if (bool(((m[:, :2]-b[:, :2]).norm(dim=1) >
              s['max_center_over_b_short']*torch.minimum(b[:, 2], b[:, 3])+1e-6).any()) or
            bool((original.wrap_pi(m[:, 4]-b[:, 4]).abs() >
                  s['max_angle_residual_rad']+1e-6).any()) or
            bool(((ratios < math.exp(-s['max_log_edge_residual'])-1e-6) |
                  (ratios > math.exp(s['max_log_edge_residual'])+1e-6)).any())):
        raise ValueError('Frozen M violates the original B guards')
    return m


def checked_inputs(roi, support, boxes_original, boxes_model, scale_xy,
                   midpoint_original):
    b, cb, xy, sides, rotation = original.frame(
        boxes_original, boxes_model, scale_xy)
    m = checked_midpoint(b, midpoint_original)
    n, r = len(b), SETTINGS['roi_size']
    if (roi.shape != (n, 256, r, r) or support.shape != (n, 1, r, r) or
            any(x.dtype != b.dtype or x.device != b.device for x in (roi, support)) or
            not bool(torch.isfinite(roi).all()) or not bool(torch.isfinite(support).all()) or
            bool(((support < -1e-6) | (support > 1+1e-6)).any())):
        raise ValueError('Invalid frozen ROI/support')
    return b, cb, xy, sides, rotation, m


def size_targets(gt_original, midpoint_original, boxes_model):
    """OFFLINE: GT long/short -> frozen M raw order -> model-B ROI order.

    Assignment is fixed before new-size learning, independent of C, its loss,
    epoch and VAL. Original-B cyclic point matching is diagnostic only.
    """
    gt, m, bm = (v.detach() for v in (gt_original, midpoint_original, boxes_model))
    n = len(m)
    if (gt.shape != (n, 5) or bm.shape != (n, 5) or
            any(v.device != m.device or v.dtype != m.dtype for v in (gt, bm)) or
            not bool(torch.isfinite(gt).all()) or bool((gt[:, 2:4] <= 0).any())):
        raise ValueError('Invalid OFFLINE GT size targets')
    original.checked_original(m)
    long = torch.maximum(gt[:, 2], gt[:, 3])
    short = torch.minimum(gt[:, 2], gt[:, 3])
    ordered = torch.stack((long, short), -1)
    raw = torch.where((m[:, 2] < m[:, 3])[:, None], ordered.flip(-1), ordered)
    roi_goal = raw_roi_pair(raw, bm)
    roi_m = raw_roi_pair(m[:, 2:4], bm)
    return dict(raw_edges=raw, roi_edges=roi_goal,
                log_residual=(roi_goal/roi_m).log().detach())


def size_loss(delta_roi, gt_original, midpoint_original, boxes_model):
    """Loss BEFORE zero/fallback delivery; no GT clipping or error filtering."""
    target = size_targets(gt_original, midpoint_original, boxes_model)['log_residual']
    if not len(target) or delta_roi.shape != target.shape or not bool(torch.isfinite(delta_roi).all()):
        raise ValueError('Expected a nonempty finite pair of train residuals')
    return F.smooth_l1_loss(delta_roi, target, reduction='mean', beta=SETTINGS['loss_beta'])


def deliver_sizes(delta_roi, boxes_original, boxes_model, scale_xy,
                  midpoint_original):
    """GT-free whole-pair M fallback; zero/square preserve M exactly.

    Rebuild rectangle MIDPOINTS, not corners. Nonfinite/overflow candidates are
    never passed to canonical_boxes(): use M only for safe geometry arithmetic,
    then reject the actual invalid candidate.
    """
    b, _, _, _, _ = original.frame(boxes_original, boxes_model, scale_xy)
    m = checked_midpoint(b, midpoint_original)
    if delta_roi.shape != (len(m), 2) or delta_roi.dtype != m.dtype or delta_roi.device != m.device:
        raise ValueError('Residual pair shape/dtype/device differs')
    raw = raw_roi_pair(delta_roi, boxes_model)
    edges = m[:, 2:4]*raw.exp()
    candidate = torch.cat((m[:, :2], edges, m[:, 4:]), -1)
    n = len(m)
    finite = torch.isfinite(candidate).all(-1) & torch.isfinite(raw).all(-1)
    positive = (edges > 0).all(-1)
    safe = torch.where((finite & positive)[:, None], candidate, m)
    local = original.original_to_roi(original.box_midpoints(safe[:, :5]),
                                     b, boxes_model, scale_xy)
    neutral = (raw == 0).all(-1)
    square = m[:, 2] == m[:, 3]
    ratios = edges/b[:, 2:4]
    raw_order = (edges[:, 0] < edges[:, 1]) == (m[:, 2] < m[:, 3])
    canonical_m = original.wrap_pi(m[:, 4]+(m[:, 2] < m[:, 3]).to(m.dtype)*math.pi/2)
    canonical_c = original.wrap_pi(m[:, 4]+(edges[:, 0] < edges[:, 1]).to(m.dtype)*math.pi/2)
    s = original.SETTINGS
    checks = dict(finite=finite, positive=positive,
        original_b_edge_bound=((ratios >= SETTINGS['edge_ratio_min']) &
                               (ratios <= SETTINGS['edge_ratio_max'])).all(-1),
        m_raw_order=raw_order,
        m_canonical_direction=original.wrap_pi(canonical_c-canonical_m).abs() <= SETTINGS['direction_atol_rad'],
        roi_midpoints=(local.abs() <= .5+s['roi_bound_tolerance']).reshape(n, 8).all(-1),
        original_b_center_bound=(m[:, :2]-b[:, :2]).norm(dim=1) <=
            s['max_center_over_b_short']*torch.minimum(b[:, 2], b[:, 3])+1e-6,
        original_b_angle_bound=original.wrap_pi(m[:, 4]-b[:, 4]).abs() <=
            s['max_angle_residual_rad']+1e-6)
    accepted = torch.stack(list(checks.values()), -1).all(-1) & ~neutral & ~square
    output = torch.where(accepted[:, None], candidate, m)
    original.checked_original(output)
    if (not torch.equal(output[:, [0, 1, 4, 5]], m[:, [0, 1, 4, 5]]) or
            not torch.equal(output[:, 2] < output[:, 3], m[:, 2] < m[:, 3])):
        raise ValueError('Size delivery changed frozen M center/direction/score/axis')
    return dict(boxes_original=output, candidate_original=candidate,
        candidate_valid=finite & positive, delta_raw=raw, accepted=accepted,
        neutral_bypass=neutral, square_bypass=square, checks=checks,
        candidate_midpoints_roi=local)


class IndependentEdgeResidualHead(nn.Module):
    """Shared spatial stem, independent u/v profile linear size readouts."""
    def __init__(self):
        super().__init__()
        h, r = SETTINGS['hidden_channels'], SETTINGS['roi_size']
        self.stem = nn.Sequential(nn.Conv2d(259, h, 1), nn.ReLU(inplace=False),
                                  nn.Conv2d(h, h, 3, padding=1), nn.ReLU(inplace=False))
        self.u = nn.Linear(h*r+4, 1)
        self.v = nn.Linear(h*r+4, 1)
        for layer in (self.u, self.v):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        if sum(p.numel() for p in self.parameters()) != PARAMETER_COUNT:
            raise ValueError('Fixed 2818-parameter size structure differs')

    def forward(self, roi, support, boxes_original, boxes_model, scale_xy,
                midpoint_original):
        b, cb, _, sides, _, m = checked_inputs(
            roi, support, boxes_original, boxes_model, scale_xy, midpoint_original)
        n, r = len(b), SETTINGS['roi_size']
        # Fixed x/y channels, with independent actual sx/sy handled by frame().
        coord = torch.linspace(-.5, .5, r, dtype=b.dtype, device=b.device)
        yy, xx = torch.meshgrid(coord, coord)
        fixed = torch.stack((xx, yy), 0)[None].expand(n, -1, -1, -1)
        mu, bu = raw_roi_pair(m[:, 2:4], boxes_model), raw_roi_pair(b[:, 2:4], boxes_model)
        geometry = torch.cat(((mu/bu).log(), (cb[:, 2:4]/sides).log()), -1).detach()
        if n:
            features = self.stem(torch.cat((roi.detach(), support.detach(), fixed), 1))
            # u=x retains columns; v=y retains rows. Never global-pool both axes.
            profile_u = features.mean(dim=2).reshape(n, 8*r)
            profile_v = features.mean(dim=3).reshape(n, 8*r)
            delta = torch.cat((self.u(torch.cat((profile_u, geometry), -1)),
                               self.v(torch.cat((profile_v, geometry), -1))), -1)
        else:
            delta = b.new_empty(0, 2)
        result = deliver_sizes(delta, b, boxes_model, scale_xy, m)
        return dict(result, delta_roi=delta, geometry_inputs=geometry)
