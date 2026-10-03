"""Independent center task on reviewed frozen B ROIs (Python3.8/torch1.8).

The unused shape output tensors preserve the reviewed joint initialization.
They are frozen and bypassed: raw B w/h/angle/score are copied exactly.
"""
import torch
from torch.nn import functional as F

from crane_project.utils import port_geometry_refine_g_v1 as base
from crane_project.utils.port_geometry_refine_g_center_v1 import (
    CENTER_SETTINGS, PARAMETER_COUNT, CenterLocalGeometryRefiner)

TRAINABLE_PARAMETER_COUNT = PARAMETER_COUNT - 3*(base.SETTINGS['hidden_fc']+1)
PART_NAMES = ['center_x', 'center_y']


class CenterOnlyLocalGeometryRefiner(CenterLocalGeometryRefiner):
    def __init__(self):
        super().__init__()
        self.output.requires_grad_(False)

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
            xy = self.center_output(hidden)
        else:
            xy = roi.new_empty(0, 2)
        if not bool(torch.isfinite(xy).all()):
            raise ValueError('Nonfinite raw center output')
        # Same smooth radial squash as joint, with overflow-safe rescaling.
        rescale = (xy.detach().abs().max(dim=1, keepdim=True)[0].clamp(min=1.)
                   if n else xy.new_ones((0, 1)))
        normalized = xy/rescale
        center = CENTER_SETTINGS['max_center_over_b_short']*normalized/torch.sqrt(
            rescale.reciprocal().square()+normalized.square().sum(dim=1, keepdim=True))
        frozen = boxes_original.detach()
        short = torch.minimum(frozen[:, 2], frozen[:, 3])[:, None]
        refined = torch.cat((frozen[:, :2]+short*center, frozen[:, 2:]), dim=1)
        base.checked_original(refined)
        if not torch.equal(refined[:, 2:], frozen[:, 2:]):
            raise RuntimeError('Center-only refinement changed B shape/score')
        return dict(boxes_original=refined, center_residual=center)


def regression_loss(refined_original, gt_original, frozen_b_original):
    """Only (p_x+p_y)/3; GT shape and corrected edges cannot change this loss."""
    base.checked_original(refined_original)
    base.checked_original(frozen_b_original)
    b, gt = frozen_b_original.detach(), gt_original.detach()
    if (not len(b) or gt.shape != (len(b), 5) or refined_original.shape != b.shape
            or b.device != refined_original.device or b.dtype != refined_original.dtype
            or gt.device != b.device or gt.dtype != b.dtype
            or not bool(torch.isfinite(gt[:, :2]).all())):
        raise ValueError('Expected nonempty paired finite center targets with B device/dtype')
    scale = CENTER_SETTINGS['max_center_over_b_short']*torch.minimum(b[:, 2], b[:, 3])[:, None]
    errors = (refined_original[:, :2]-gt[:, :2])/scale
    parts = F.smooth_l1_loss(errors, torch.zeros_like(errors),
        beta=CENTER_SETTINGS['center_loss_beta'], reduction='none').mean(dim=0)
    return parts.sum()/3., parts
