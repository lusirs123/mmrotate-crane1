"""Isolated current-frame reliability prototype; never modifies detector boxes.

Online inputs: detached P3, current detector boxes/scores, transform metadata.
GT and synthetic probes belong exclusively to the offline supervision helpers.
Compatible with the project's Python 3.8 / torch 1.8 environment.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


SETTINGS = dict(
    version='port_structure_reliability_v1_train_preflight', seed=1701,
    feature_level=0, stride=8, in_channels=256, hidden_channels=32,
    context_side_long_multiple=2., context_min_side_cells=4., roi_size=9,
    roi_pool_size=3, quality_hidden=64, response_sigma_cells=1.,
    center_reference_px=15., size_reference_relative=.10,
    angle_reference_deg=3., angle_assessable_gt_aspect=1.2,
    probe_center_px=15., probe_size_fraction=.15, probe_angle_deg=5.,
    probes_per_image=13, genuine_quality_weight=.5,
    structure_loss_weight=1., quality_loss_weight=1.,
    optimizer='SGD', lr=.001, momentum=0., weight_decay=0., clip_norm=10.,
    batch_size=1, steps_per_view=1, reset_branch_each_view=True)


def canonical_boxes(boxes):
    """Long/short le90; equivalent width exchange + pi/2 has identical meaning."""
    if boxes.ndim != 2 or boxes.shape[1] != 5:
        raise ValueError('Expected Nx5 boxes')
    if not bool(torch.isfinite(boxes).all()) or bool((boxes[:, 2:4] <= 0).any()):
        raise ValueError('Boxes must be finite with positive edges')
    swap = boxes[:, 2] < boxes[:, 3]
    angle = boxes[:, 4] + swap.to(boxes.dtype)*math.pi/2
    return torch.stack((boxes[:, 0], boxes[:, 1],
                        torch.maximum(boxes[:, 2], boxes[:, 3]),
                        torch.minimum(boxes[:, 2], boxes[:, 3]),
                        torch.remainder(angle + math.pi/2, math.pi)-math.pi/2), dim=1)


def checked_meta(meta):
    scale = [float(x) for x in meta['scale_factor']]
    if (len(scale) != 4 or not all(math.isfinite(x) and x > 0 for x in scale)
            or scale[0] != scale[2] or scale[1] != scale[3]):
        raise ValueError('Expected positive [sx,sy,sx,sy] scale_factor')
    ih, iw = meta['img_shape'][:2]
    oh, ow = meta['ori_shape'][:2]
    ph, pw = meta['pad_shape'][:2]
    if min(ih, iw, oh, ow) <= 0 or ih > ph or iw > pw:
        raise ValueError('Invalid top-left padded image geometry')
    # RResize keeps aspect ratio, with at most a pixel of rounding per axis.
    if abs(scale[0]-scale[1]) > 1./min(oh, ow) + 1e-6:
        raise ValueError('Anisotropic image resize is outside the fixed B contract')
    if meta.get('flip', False) and meta.get('flip_direction') not in (
            'horizontal', 'vertical', 'diagonal'):
        raise ValueError('Unknown reflection')
    return scale


def map_points(points, meta, inverse=False):
    scale = checked_meta(meta)
    if points.shape[-1] != 2 or not bool(torch.isfinite(points).all()):
        raise ValueError('Expected finite xy points')
    result = points.clone()
    sx = result.new_tensor(scale[:2])
    if not inverse:
        result = result*sx
    if meta.get('flip', False):
        direction = meta['flip_direction']
        if direction in ('horizontal', 'diagonal'):
            result[..., 0] = meta['img_shape'][1]-1-result[..., 0]
        if direction in ('vertical', 'diagonal'):
            result[..., 1] = meta['img_shape'][0]-1-result[..., 1]
    return result/sx if inverse else result


def map_boxes(boxes, meta, inverse=False, size_mode='detector'):
    """Preserve BOTH existing contracts, which differ after integer rounding.

    detector: w/sx,h/sy is SymEOODHead's existing inference rescale operation.
    annotation: w,h * sqrt(sx*sy) is RResize's existing GT operation.
    Reflection is undone before inverse scaling. Padding has no translation.
    """
    scale = checked_meta(meta)
    if size_mode not in ('detector', 'annotation'):
        raise ValueError('Unknown size convention')
    # Keep raw w/h association until detector component-wise scaling finishes.
    canonical_boxes(boxes)  # validate; do not reorder raw input here
    result = boxes.clone()
    sizes = scale[:2] if size_mode == 'detector' else [math.sqrt(scale[0]*scale[1])]*2
    if not inverse:
        result[:, :2] = result[:, :2]*result.new_tensor(scale[:2])
        result[:, 2:4] = result[:, 2:4]*result.new_tensor(sizes)
    if meta.get('flip', False):
        direction = meta['flip_direction']
        if direction in ('horizontal', 'diagonal'):
            result[:, 0] = meta['img_shape'][1]-1-result[:, 0]
        if direction in ('vertical', 'diagonal'):
            result[:, 1] = meta['img_shape'][0]-1-result[:, 1]
        if direction != 'diagonal':
            result[:, 4] = torch.remainder(math.pi-result[:, 4]+math.pi/2, math.pi)-math.pi/2
    if inverse:
        result[:, :2] = result[:, :2]/result.new_tensor(scale[:2])
        result[:, 2:4] = result[:, 2:4]/result.new_tensor(sizes)
    return result


def feature_points(shape, reference, stride=8):
    """Feature index i has coordinate i*stride, matching B's anchor grid.

    grid_sample uses align_corners=False with the corresponding +0.5 shift.
    This is a coordinate convention, not a claim about receptive-field support.
    """
    h, w = shape
    y, x = torch.meshgrid(torch.arange(h, device=reference.device, dtype=reference.dtype),
                          torch.arange(w, device=reference.device, dtype=reference.dtype))
    return torch.stack((x, y), dim=-1)*stride


def response_targets(axis_model, feature_shape, meta):
    """Native or weak finite axis, unsigned endpoints; no visibility targets."""
    checked_meta(meta)
    if (axis_model.shape != (2, 2) or not bool(torch.isfinite(axis_model).all())
            or float((axis_model[1]-axis_model[0]).square().sum()) <= 0):
        raise ValueError('Expected a nondegenerate finite axis')
    p = feature_points(feature_shape, axis_model, SETTINGS['stride'])
    v = axis_model[1]-axis_model[0]
    t = ((p-axis_model[0])*v).sum(dim=-1)/v.square().sum()
    nearest = axis_model[0] + t.clamp(0, 1)[..., None]*v
    sigma = SETTINGS['stride']*SETTINGS['response_sigma_cells']
    center = torch.exp(-.5*(p-axis_model.mean(dim=0)).square().sum(dim=-1)/sigma**2)
    line = torch.exp(-.5*(p-nearest).square().sum(dim=-1)/sigma**2)
    valid = ((p[..., 0] < meta['img_shape'][1]) &
             (p[..., 1] < meta['img_shape'][0])).to(axis_model.dtype)
    target = torch.stack((center, line), dim=0)[None]
    return target.detach(), valid[None, None].detach()


def balanced_response_loss(logits, target, valid):
    """Soft positive and negative masses normalized separately, valid image only."""
    if logits.shape != target.shape or valid.shape != (1, 1, *logits.shape[-2:]):
        raise ValueError('Response target/mask shape differs')
    positive, negative = target*valid, (1-target)*valid
    pos_n = positive.sum(dim=(0, 2, 3))
    neg_n = negative.sum(dim=(0, 2, 3))
    if bool((pos_n <= 0).any()) or bool((neg_n <= 0).any()):
        raise ValueError('Response supervision has no positive/negative support')
    losses = .5*((F.softplus(-logits)*positive).sum(dim=(0, 2, 3))/pos_n +
                 (F.softplus(logits)*negative).sum(dim=(0, 2, 3))/neg_n)
    return losses.mean(), losses


def quality_targets_original(boxes_original, gt_original):
    """Continuous q=exp(-error/reference); not calibrated correctness probability.

    Centers use original pixels; size=max relative long/short; direction is
    unsigned pi-periodic degrees, unassessed when GT long/short < 1.2.
    """
    b = canonical_boxes(boxes_original.detach())
    g = canonical_boxes(gt_original.detach().reshape(1, 5))
    center = (b[:, :2]-g[:, :2]).square().sum(dim=1).sqrt()
    size = (b[:, 2:4]/g[:, 2:4]-1).abs().max(dim=1)[0]
    angle = torch.remainder(b[:, 4]-g[:, 4]+math.pi/2, math.pi).sub(math.pi/2).abs()*180/math.pi
    errors = torch.stack((center, size, angle), dim=1)
    refs = errors.new_tensor([SETTINGS['center_reference_px'], SETTINGS['size_reference_relative'],
                              SETTINGS['angle_reference_deg']])
    target = torch.exp(-errors/refs)
    mask = torch.ones_like(target)
    if float(g[0, 2]/g[0, 3]) < SETTINGS['angle_assessable_gt_aspect']:
        mask[:, 2] = 0
    return target, mask, errors


def component_probes_original(gt_original):
    """Thirteen bounded offline GT probes. Never inference proposals or inputs tags."""
    g = canonical_boxes(gt_original.detach().reshape(1, 5))[0]
    rows, names = [g.clone()], ['gt_exact']
    for axis in (0, 1):
        for sign in (-1, 1):
            p = g.clone(); p[axis] += sign*SETTINGS['probe_center_px']
            rows.append(p); names.append('center_%s_%+d' % ('xy'[axis], sign))
    for edges, label in [((2, 3), 'joint_size'), ((2,), 'long_size'), ((3,), 'short_size')]:
        for sign in (-1, 1):
            p = g.clone()
            for edge in edges:
                p[edge] *= 1+sign*SETTINGS['probe_size_fraction']
            rows.append(p); names.append('%s_%+d' % (label, sign))
    for sign in (-1, 1):
        p = g.clone(); p[4] += sign*SETTINGS['probe_angle_deg']*math.pi/180
        rows.append(p); names.append('angle_%+d' % sign)
    return canonical_boxes(torch.stack(rows)), names


def quality_loss(predicted, targets, mask, genuine_count):
    if predicted.shape != targets.shape or mask.shape != targets.shape or genuine_count not in (0, 1):
        raise ValueError('Quality rows or genuine B count differs')
    if len(predicted) <= genuine_count:
        raise ValueError('Expected bounded TRAIN probes')
    value = F.smooth_l1_loss(predicted, targets.detach(), reduction='none')
    def mean(a, m):
        return (a*m).sum()/m.sum().clamp(min=1.)
    probe = mean(value[genuine_count:], mask[genuine_count:])
    genuine = mean(value[:genuine_count], mask[:genuine_count]) if genuine_count else value.sum()*0
    w = SETTINGS['genuine_quality_weight']
    total = (w*genuine+(1-w)*probe) if genuine_count else probe
    return total, dict(genuine=genuine, probe=probe)


def sample_square(feature, boxes_model, valid_map):
    """Isotropic, image-aligned square context; no rectangular ROI stretching."""
    b = canonical_boxes(boxes_model.detach())
    n, _, h, w = len(b), feature.shape[1], feature.shape[2], feature.shape[3]
    if feature.shape[0] != 1 or valid_map.shape != (1, 1, h, w):
        raise ValueError('This bounded prototype supports one image at a time')
    side = (b[:, 2]*SETTINGS['context_side_long_multiple']).clamp(
        min=SETTINGS['stride']*SETTINGS['context_min_side_cells'])
    r = SETTINGS['roi_size']
    local = torch.linspace(-.5, .5, r, device=b.device, dtype=b.dtype)
    yy, xx = torch.meshgrid(local, local)
    points = b[:, None, None, :2] + torch.stack((xx, yy), dim=-1)[None]*side[:, None, None, None]
    grid = 2*(points/SETTINGS['stride'] + .5)/b.new_tensor([w, h])-1
    # Invalid padded cells cannot become evidence through stem convolution/bias.
    sampled = F.grid_sample((feature*valid_map).expand(n, -1, -1, -1), grid,
                            align_corners=False, padding_mode='zeros')
    support = F.grid_sample(valid_map.expand(n, -1, -1, -1), grid,
                           align_corners=False, padding_mode='zeros')
    return sampled, support


class StructureComponentReliability(nn.Module):
    """A side branch. No bbox delta, filtering, sorting, detector train(), or GT API."""
    def __init__(self, use_structure=True):
        super().__init__()
        self.use_structure = bool(use_structure)
        c, h = SETTINGS['in_channels'], SETTINGS['hidden_channels']
        self.stem = nn.Sequential(nn.Conv2d(c, h, 1), nn.ReLU(inplace=False),
                                  nn.Conv2d(h, h, 3, padding=1), nn.ReLU(inplace=False))
        self.response = nn.Conv2d(h, 2, 1)
        # Ordinary ROI control retains exactly the same input dimensions/MLP.
        # Structure response block is zeroed for that control; no formal run here.
        d = h*SETTINGS['roi_pool_size']**2 + 3*SETTINGS['roi_size']**2 + 8
        self.quality = nn.Sequential(nn.Linear(d, SETTINGS['quality_hidden']), nn.ReLU(inplace=False),
                                      nn.Linear(SETTINGS['quality_hidden'], 3))

    def forward(self, p3, boxes_model, scores, meta):
        checked_meta(meta)
        if p3.ndim != 4 or p3.shape[:2] != (1, SETTINGS['in_channels']):
            raise ValueError('Expected one 256-channel frozen P3 tensor')
        b = canonical_boxes(boxes_model.detach())
        scores = scores.detach().reshape(-1)
        if len(scores) != len(b) or not bool(torch.isfinite(scores).all()) or bool(((scores < 0) | (scores > 1)).any()):
            raise ValueError('Expected one finite detector score per supplied box')
        # The network does not receive target/sequence/domain/probe identities.
        p = feature_points(p3.shape[-2:], p3, SETTINGS['stride'])
        valid = ((p[..., 0] < meta['img_shape'][1]) &
                 (p[..., 1] < meta['img_shape'][0])).to(p3.dtype)[None, None]
        shared = self.stem(p3.detach()*valid)
        logits = self.response(shared)
        if not len(b):
            return dict(response_logits=logits, qualities=shared.new_empty((0, 3)),
                        roi_support=shared.new_empty((0,)))
        image_roi, support = sample_square(shared, b, valid)
        response_roi, _ = sample_square(logits.sigmoid(), b, valid)
        if not self.use_structure:
            response_roi = response_roi*0
        ph, pw = meta['pad_shape'][:2]
        geometry = torch.stack((scores, b[:, 0]/pw, b[:, 1]/ph,
                                torch.log(b[:, 2]/max(ph, pw)), torch.log(b[:, 3]/max(ph, pw)),
                                torch.log(b[:, 2]/b[:, 3]), torch.sin(2*b[:, 4]), torch.cos(2*b[:, 4])), dim=1)
        descriptor = torch.cat((F.adaptive_avg_pool2d(image_roi, SETTINGS['roi_pool_size']).flatten(1),
                                 response_roi.flatten(1), support.flatten(1), geometry), dim=1)
        qualities = self.quality(descriptor).sigmoid()
        return dict(response_logits=logits, qualities=qualities,
                    roi_support=support.mean(dim=(1, 2, 3)))


def freeze_detector(detector):
    detector.eval()
    for p in detector.parameters():
        p.requires_grad_(False)
        p.grad = None
    return detector


def assert_detector_frozen(detector):
    if any(m.training for m in detector.modules()):
        raise ValueError('Detector or one of its modules left eval mode')
    if any(p.requires_grad or p.grad is not None for p in detector.parameters()):
        raise ValueError('Detector received gradients or was unfrozen')
