"""G v1: frozen-B local size/direction refinement; never predicts a center/score.

Online API accepts detached image features, B boxes and transform metadata only.
TRAIN supervision is a separate function. Python 3.8 / torch 1.8 compatible.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

# Reuse audited coordinate primitives; no reliability model, labels or policy.
from crane_project.utils.port_structure_reliability_v1 import (
    assert_detector_frozen, canonical_boxes, checked_meta, freeze_detector,
    map_boxes)


SETTINGS = dict(
    protocol='port_geometry_g_v1_train_preflight', seed=1703,
    feature_level=0, stride=8, in_channels=256, hidden_channels=32,
    roi_size=9, context_multiplier=1.5, min_context_cells=2., hidden_fc=64,
    max_log_edge_residual=math.log(1.25), max_angle_residual_rad=math.pi/18,
    loss_beta=.1, train_center_match_px=15.,
    steps_per_arm=200, batch_size=8, optimizer='Adam', lr=.001,
    weight_decay=0., clip_norm=10., fit_images_per_domain=24,
    probe_images_per_domain=8, scales=[1., .5],
    max_cpu_roi_bytes=32*2**20)
ARMS = ('ordinary', 'aligned')


def wrap_pi(value):
    return torch.remainder(value+math.pi/2, math.pi)-math.pi/2


def sampling_grid(boxes_model, feature_shape, mode):
    """Same rectangular context/grid; mode changes rotation alone.

    Feature index i maps to i*stride, as in B's anchor grid. grid_sample's
    align_corners=False normalization includes the required half-cell shift.
    """
    if mode not in ARMS:
        raise ValueError('Unknown fixed G sampling arm')
    b = canonical_boxes(boxes_model.detach())
    h, w = feature_shape
    if min(h, w) <= 0:
        raise ValueError('Empty feature resolution')
    side = (b[:, 2:4]*SETTINGS['context_multiplier']).clamp(
        min=SETTINGS['stride']*SETTINGS['min_context_cells'])
    v = torch.linspace(-.5, .5, SETTINGS['roi_size'], dtype=b.dtype, device=b.device)
    yy, xx = torch.meshgrid(v, v)
    x = xx[None]*side[:, None, None, 0]
    y = yy[None]*side[:, None, None, 1]
    angle = b[:, 4] if mode == 'aligned' else torch.zeros_like(b[:, 4])
    c, s = angle.cos()[:, None, None], angle.sin()[:, None, None]
    points = torch.stack((x*c-y*s+b[:, None, None, 0],
                          x*s+y*c+b[:, None, None, 1]), dim=-1)
    grid = 2*(points/SETTINGS['stride']+.5)/b.new_tensor([w, h])-1
    return grid.detach(), points.detach()


def sample_local(p3, boxes_model, meta, mode):
    """Bilinear sampling of frozen P3, masking padded source cells explicitly."""
    checked_meta(meta)
    if p3.ndim != 4 or p3.shape[0:2] != (1, SETTINGS['in_channels']):
        raise ValueError('Expected batch1 256-channel P3')
    if p3.requires_grad or p3.grad_fn is not None:
        raise ValueError('Frozen P3 must not retain a detector graph')
    h, w = p3.shape[-2:]
    boxes_model = boxes_model.detach()
    if boxes_model.device != p3.device or boxes_model.dtype != p3.dtype:
        raise ValueError('Feature and box device/dtype differ')
    grid, points = sampling_grid(boxes_model, (h, w), mode)
    n = len(boxes_model)
    if n == 0:
        r = SETTINGS['roi_size']
        return p3.new_empty(0, SETTINGS['in_channels'], r, r), p3.new_empty(0, 1, r, r), points
    y, x = torch.meshgrid(torch.arange(h, device=p3.device),
                          torch.arange(w, device=p3.device))
    valid = ((x*SETTINGS['stride'] < meta['img_shape'][1]) &
             (y*SETTINGS['stride'] < meta['img_shape'][0])).to(p3.dtype)[None, None]
    roi = F.grid_sample((p3*valid).expand(n, -1, -1, -1), grid,
                        align_corners=False, padding_mode='zeros')
    support = F.grid_sample(valid.expand(n, -1, -1, -1), grid,
                            align_corners=False, padding_mode='zeros')
    return roi.detach(), support.detach(), points


def checked_original(boxes):
    if boxes.ndim != 2 or boxes.shape[1] != 6:
        raise ValueError('Expected original-coordinate Nx6 B boxes including score')
    canonical_boxes(boxes[:, :5])
    if not bool(torch.isfinite(boxes).all()):
        raise ValueError('Nonfinite B score')


def apply_residual(boxes_original, residual):
    """Bounded canonical edges, original raw w/h representation retained.

    Crossing corrected long/short edges are projected to their geometric mean.
    At the square tie, direction uses the original canonical long direction,
    including when the original B raw representation had w<h.
    A zero residual preserves ALL six original B values bit for bit.
    """
    checked_original(boxes_original)
    b = boxes_original.detach()
    if residual.device != b.device or residual.dtype != b.dtype:
        raise ValueError('Residual and original box device/dtype differ')
    if residual.shape != (len(b), 3) or not bool(torch.isfinite(residual).all()):
        raise ValueError('Expected finite Nx3 residuals')
    cap = residual.new_tensor([SETTINGS['max_log_edge_residual']]*2 +
                              [SETTINGS['max_angle_residual_rad']])
    if bool((residual.abs() > cap+1e-7).any()):
        raise ValueError('Residual exceeds the fixed G bound')
    swap = b[:, 2] < b[:, 3]
    long = torch.maximum(b[:, 2], b[:, 3])*residual[:, 0].exp()
    short = torch.minimum(b[:, 2], b[:, 3])*residual[:, 1].exp()
    crossing = long < short
    equal = torch.exp((long.log()+short.log())*.5)
    long, short = torch.where(crossing, equal, long), torch.where(crossing, equal, short)
    unwrapped = b[:, 4]+residual[:, 2]
    # Identity at zero must still carry an angle gradient. Selecting b[:,4]
    # alone would incorrectly kill the zero-initialized angle supervision.
    direct = ((unwrapped >= -math.pi/2) & (unwrapped < math.pi/2)) | (residual[:, 2].detach() == 0)
    angle = torch.where(direct, unwrapped, wrap_pi(unwrapped))
    # Equal edges remove canonical_boxes' w<h flag. Retain the pre-projection
    # long direction explicitly, rather than creating a spurious pi/2 error.
    angle = torch.where(crossing & swap, wrap_pi(unwrapped+math.pi/2), angle)
    result = torch.stack((b[:, 0], b[:, 1], torch.where(swap, short, long),
                          torch.where(swap, long, short), angle, b[:, 5]), dim=1)
    checked_original(result)
    if not torch.equal(result[:, [0, 1, 5]], b[:, [0, 1, 5]]):
        raise RuntimeError('G changed a B center or score')
    return result, crossing


class LocalGeometryRefiner(nn.Module):
    """Identical head in both arms; no BN/dropout or trainable sampling offsets."""
    def __init__(self):
        super().__init__()
        c, h, r = SETTINGS['in_channels'], SETTINGS['hidden_channels'], SETTINGS['roi_size']
        self.stem = nn.Sequential(nn.Conv2d(c+1, h, 1), nn.ReLU(inplace=False),
                                  nn.Conv2d(h, h, 3, padding=1), nn.ReLU(inplace=False))
        self.hidden = nn.Sequential(nn.Linear(h*r*r+4, SETTINGS['hidden_fc']),
                                    nn.ReLU(inplace=False))
        self.output = nn.Linear(SETTINGS['hidden_fc'], 3)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, roi, support, boxes_original, boxes_model):
        checked_original(boxes_original)
        n, r = len(boxes_original), SETTINGS['roi_size']
        if roi.shape != (n, SETTINGS['in_channels'], r, r) or support.shape != (n, 1, r, r):
            raise ValueError('Local feature/support dimensions differ')
        if not bool(torch.isfinite(roi).all()) or not bool(torch.isfinite(support).all()):
            raise ValueError('Nonfinite local features')
        if bool(((support < -1e-6) | (support > 1+1e-6)).any()):
            raise ValueError('Invalid local feature support')
        b = canonical_boxes(boxes_model.detach())
        if len(b) != n or b.device != roi.device or b.dtype != roi.dtype:
            raise ValueError('Model box/local feature identity differs')
        geom = torch.stack((torch.log(b[:, 2]/SETTINGS['stride']),
                            torch.log(b[:, 3]/SETTINGS['stride']),
                            (2*b[:, 4]).sin(), (2*b[:, 4]).cos()), dim=1)
        # Detached cache/boxes are the sole online inputs. GT has no API path.
        if n:
            local = self.stem(torch.cat((roi.detach(), support.detach()), dim=1))
            raw = self.output(self.hidden(torch.cat((local.reshape(n, -1), geom), dim=1)))
        else:
            raw = roi.new_empty(0, 3)
        cap = raw.new_tensor([SETTINGS['max_log_edge_residual']]*2 +
                             [SETTINGS['max_angle_residual_rad']])
        residual = raw.tanh()*cap
        result, crossing = apply_residual(boxes_original, residual)
        return dict(boxes_original=result, residual=residual, edge_projection=crossing)


def regression_loss(refined_original, gt_original):
    """OFFLINE only: equally normalized edge/periodic-direction objectives.

    It trains only this head, not original SymKLD or B. No IoU/score weighting.
    """
    checked_original(refined_original)
    b = canonical_boxes(refined_original[:, :5])
    g = canonical_boxes(gt_original.detach())
    if len(g) != len(b) or not len(b):
        raise ValueError('Expected matched nonempty TRAIN boxes')
    errors = torch.cat((torch.log(b[:, 2:4]/g[:, 2:4])/SETTINGS['max_log_edge_residual'],
                        (wrap_pi(b[:, 4]-g[:, 4])/SETTINGS['max_angle_residual_rad'])[:, None]), dim=1)
    parts = F.smooth_l1_loss(errors, torch.zeros_like(errors),
                            beta=SETTINGS['loss_beta'], reduction='none').mean(dim=0)
    return parts.mean(), parts
