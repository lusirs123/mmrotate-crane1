"""Frozen box-conditioned input; only the identical small heads are trained."""
from copy import deepcopy
import math
import time
import numpy as np
import torch
from torch.nn import functional as F
from crane_project.utils import port_reliability_box_contrast_v1 as core
from crane_project.utils import port_reliability_feature_source_v1_torch as source

exported, optimizer = source.exported, source.optimizer


def make_models(device):
    torch.manual_seed(core.SETTINGS['seed'])
    first = source.ResidualRisk().to(device)
    return dict(original=first, contrast=deepcopy(first))


def box_features(p3, boxes_original, meta):
    """Online API has only frozen features, boxes and transform metadata."""
    from crane_project.utils import port_geometry_refine_g_v1 as geometry
    boxes = torch.as_tensor(boxes_original, dtype=p3.dtype, device=p3.device)
    if boxes.ndim != 2 or boxes.shape[1] != 6:
        raise ValueError('Expected original-coordinate Nx6 candidate boxes')
    model_boxes = geometry.map_boxes(boxes[:, :5], meta)
    roi, support, _ = geometry.sample_local(p3, model_boxes, meta, 'aligned')
    pooled = F.avg_pool2d(roi, 3, stride=3).reshape(len(boxes), 2304)
    # Independent CPU block layout/means, checked before storing feature bytes.
    reference = core.pool_reference(roi.cpu().numpy())
    value = pooled.cpu().numpy()
    if not np.allclose(value, reference, rtol=2e-5, atol=2e-5):
        raise ValueError('Box-conditioned spatial pool differs')
    if not np.isfinite(value).all(): raise ValueError('Nonfinite sampled box features')
    return value, support.cpu().numpy()


def losses(model, x, z, y, w, ax, az, ay, mask, arm):
    if arm not in core.ARMS or x.shape[0] != len(z) or y.shape != z.shape or w.shape != z.shape:
        raise ValueError('Invalid fixed quality objective')
    real = (F.binary_cross_entropy_with_logits(model(x.detach(), z.detach()), y.detach(),
                                               reduction='none')*w.detach()).mean()
    if arm == 'original': return real, real.new_zeros(())
    n, k = mask.shape
    if (ax.shape != (n, k, 259) or az.shape != ay.shape or ay.shape != mask.shape
            or n != len(x) or mask.dtype != torch.bool):
        raise ValueError('Invalid per-image contrast tensors')
    # Only valid entries enter the network/loss. Padding has no loss or gradient.
    positions = mask.nonzero(as_tuple=False)
    if not len(positions): return real, real.new_zeros(())
    logits = model(ax[mask].detach(), az[mask].detach())
    values = F.binary_cross_entropy_with_logits(logits, ay[mask].detach(), reduction='none')
    image_loss = values.new_zeros(n).index_add(0, positions[:, 0], values)
    per_image = image_loss/mask.sum(1).clamp(min=1).to(values.dtype)
    return real, core.SETTINGS['auxiliary_weight']*per_image.mean()


def update(model, optim, x, z, y, w, ax, az, ay, mask, arm, component_gradients=False):
    started = time.monotonic(); optim.zero_grad()
    real, auxiliary = losses(model, x, z, y, w, ax, az, ay, mask, arm)
    loss = real+auxiliary
    component = None
    params = list(model.parameters())
    if component_gradients:
        vectors = []
        for term in (real, auxiliary):
            gs = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True) if term.requires_grad else [None]*len(params)
            vector = torch.cat([torch.zeros_like(p).flatten() if g is None else g.detach().flatten()
                                for p, g in zip(params, gs)])
            if not bool(torch.isfinite(vector).all()): raise ValueError('Nonfinite component gradient')
            vectors.append(vector)
        nr, na = [float(v.norm()) for v in vectors]
        component = dict(real_gradient_norm=nr, weighted_auxiliary_gradient_norm=na,
                         auxiliary_over_real=na/nr if nr else None,
                         component_sum_gradient_norm=float((vectors[0]+vectors[1]).norm()))
    loss.backward()
    gradients = {n:float(p.grad.norm()) for n,p in model.named_parameters() if p.grad is not None}
    if not gradients or not all(math.isfinite(v) for v in gradients.values()):
        raise ValueError('Nonfinite or disconnected head gradient')
    before = float(torch.nn.utils.clip_grad_norm_(params, core.SETTINGS['clip_norm']))
    after = math.sqrt(sum(float(p.grad.norm())**2 for p in params if p.grad is not None))
    if not math.isfinite(before) or not math.isfinite(after): raise ValueError('Nonfinite clipping')
    optim.step()
    if not all(bool(torch.isfinite(p).all()) for p in params): raise ValueError('Nonfinite trained head')
    result = dict(loss=float(loss.detach()), real_loss=float(real.detach()),
                  weighted_auxiliary_loss=float(auxiliary.detach()), gradient_by_parameter=gradients,
                  gradient_norm_before=before, gradient_norm_after=after,
                  valid_contrasts=int(mask.sum()) if arm=='contrast' else 0,
                  elapsed_seconds=time.monotonic()-started)
    if component is not None: result['components'] = component
    return result
