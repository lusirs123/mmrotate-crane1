"""Bounded five-parameter refinement of frozen B outputs, Python 3.8/torch1.8.

The online head accepts the same detached inputs as G v1. Its added xy output
uses the ORIGINAL frozen B short edge; TRAIN labels enter only regression_loss.
The original G sources, sampling, three residuals and their loss stay intact.
"""
import torch
from torch import nn
from torch.nn import functional as F

from crane_project.utils import port_geometry_refine_g_v1 as base

CENTER_SETTINGS = dict(
    max_center_over_b_short=.30,
    center_coordinate_system='original_image_xy',
    center_scale='detached_original_B_short_edge',
    center_squash='z_xy / sqrt(1 + sum(z_xy**2))',
    loss_component_coefficient=1./3.,
    center_loss_beta=.1)
PARAMETER_COUNT = 184037
PART_NAMES = ['log_long', 'log_short', 'periodic_angle', 'center_x', 'center_y']


def apply_joint_residual(boxes_original, shape_residual, center_residual):
    """center_residual is dimensionless xy / ORIGINAL B short edge, radial<=.3.

    Preserve the reviewed size/angle representation and projection exactly.
    No image-boundary clipping, GT clipping or center-based output rejection.
    """
    shape, crossing = base.apply_residual(boxes_original, shape_residual)
    b = boxes_original.detach()
    if (center_residual.shape != (len(b), 2) or center_residual.device != b.device
            or center_residual.dtype != b.dtype
            or not bool(torch.isfinite(center_residual).all())):
        raise ValueError('Expected finite Nx2 center residuals with B device/dtype')
    if bool((center_residual.norm(dim=1) > CENTER_SETTINGS['max_center_over_b_short']+1e-7).any()):
        raise ValueError('Center residual exceeds fixed radial bound')
    short = torch.minimum(b[:, 2], b[:, 3])[:, None]
    refined = torch.cat((b[:, :2]+short*center_residual, shape[:, 2:]), dim=1)
    base.checked_original(refined)
    if not torch.equal(refined[:, 5], b[:, 5]):
        raise RuntimeError('Joint refinement changed B scores')
    return refined, crossing


class CenterLocalGeometryRefiner(base.LocalGeometryRefiner):
    """Same FC64/stem/descriptor, plus two zero-initialized output units."""
    def __init__(self):
        super().__init__()
        self.center_output = nn.Linear(base.SETTINGS['hidden_fc'], 2)
        nn.init.zeros_(self.center_output.weight)
        nn.init.zeros_(self.center_output.bias)

    def forward(self, roi, support, boxes_original, boxes_model):
        base.checked_original(boxes_original)
        n, r = len(boxes_original), base.SETTINGS['roi_size']
        if (roi.shape != (n, base.SETTINGS['in_channels'], r, r)
                or support.shape != (n, 1, r, r)):
            raise ValueError('Local feature/support dimensions differ')
        if not bool(torch.isfinite(roi).all()) or not bool(torch.isfinite(support).all()):
            raise ValueError('Nonfinite local features')
        if bool(((support < -1e-6) | (support > 1+1e-6)).any()):
            raise ValueError('Invalid local feature support')
        b = base.canonical_boxes(boxes_model.detach())
        if (len(b) != n or b.device != roi.device or b.dtype != roi.dtype
                or boxes_original.device != roi.device or boxes_original.dtype != roi.dtype):
            raise ValueError('Original/model box/local feature identity differs')
        geom = torch.stack((torch.log(b[:, 2]/base.SETTINGS['stride']),
                            torch.log(b[:, 3]/base.SETTINGS['stride']),
                            (2*b[:, 4]).sin(), (2*b[:, 4]).cos()), dim=1)
        if n:
            local = self.stem(torch.cat((roi.detach(), support.detach()), dim=1))
            hidden = self.hidden(torch.cat((local.reshape(n, -1), geom), dim=1))
            raw, xy = self.output(hidden), self.center_output(hidden)
        else:
            raw, xy = roi.new_empty(0, 3), roi.new_empty(0, 2)
        cap = raw.new_tensor([base.SETTINGS['max_log_edge_residual']]*2+
                             [base.SETTINGS['max_angle_residual_rad']])
        residual = raw.tanh()*cap
        # Smooth radial squash has finite nonzero Jacobian at zero and no
        # per-axis square bound that could allow sqrt(2) times the stated cap.
        if not bool(torch.isfinite(xy).all()):
            raise ValueError('Nonfinite raw center output')
        # Algebraically identical, but avoid overflow in xy**2 for large finite
        # activations. The auxiliary rescale cancels from the function; detach
        # it so max/tie selection cannot introduce a spurious gradient path.
        rescale = (xy.detach().abs().max(dim=1, keepdim=True)[0].clamp(min=1.)
                   if n else xy.new_ones((0, 1)))
        normalized = xy/rescale
        center = CENTER_SETTINGS['max_center_over_b_short']*normalized/torch.sqrt(
            rescale.reciprocal().square()+normalized.square().sum(dim=1, keepdim=True))
        refined, crossing = apply_joint_residual(boxes_original, residual, center)
        return dict(boxes_original=refined, residual=residual,
                    center_residual=center, edge_projection=crossing)


def regression_loss(refined_original, gt_original, frozen_b_original):
    """Original three-part loss + (p_x+p_y)/3, NOT a mean over five parts.

    Each part retains coefficient1/3; equivalently add2/3*mean(p_x,p_y).
    Center errors use .3*S_B, never a corrected edge or a clipped GT target.
    Returns raw component means for transparent loss/gradient diagnostics.
    """
    shape_loss, shape_parts = base.regression_loss(refined_original, gt_original)
    base.checked_original(frozen_b_original)
    b = frozen_b_original.detach()
    gt = base.canonical_boxes(gt_original.detach())
    if (len(b) != len(refined_original) or b.device != refined_original.device
            or b.dtype != refined_original.dtype):
        raise ValueError('Frozen B/reference identity differs')
    scale = CENTER_SETTINGS['max_center_over_b_short']*torch.minimum(b[:, 2], b[:, 3])[:, None]
    errors = (refined_original[:, :2]-gt[:, :2])/scale
    center_parts = F.smooth_l1_loss(errors, torch.zeros_like(errors),
        beta=CENTER_SETTINGS['center_loss_beta'], reduction='none').mean(dim=0)
    parts = torch.cat((shape_parts, center_parts))
    return shape_loss+CENTER_SETTINGS['loss_component_coefficient']*center_parts.sum(), parts


def initial_state_from_reference(reference):
    """Copy the entire untrained FC64 initialization; add only zero xy units."""
    head = CenterLocalGeometryRefiner()
    expected = head.state_dict()
    added = {'center_output.weight', 'center_output.bias'}
    if set(reference) != set(expected)-added:
        raise ValueError('Untrained FC64 initialization keys differ')
    if any(not bool((reference[k] == 0).all()) for k in ('output.weight', 'output.bias')):
        raise ValueError('Requires the untrained zero-output FC64 initialization')
    state = {}
    for key, target in expected.items():
        value = torch.zeros_like(target) if key in added else reference[key]
        if (value.shape != target.shape or value.dtype != target.dtype or value.device.type != 'cpu'
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Expected unchanged finite CPU initialization: '+key)
        state[key] = value.detach().clone()
    return state
