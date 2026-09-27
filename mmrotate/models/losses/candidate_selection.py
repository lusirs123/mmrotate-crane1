"""Selection loss on the classification logits used by the K1 detector."""

import math

import torch
import torch.nn.functional as F


def candidate_selection_loss(logits, ious, *, score_threshold=0.05,
                             iou_threshold=0.5, margin=0.1):
    """Return one-image loss and chosen indices; geometry is a fixed target.

    ``logits`` and ``ious`` have the same flattened anchor order. The negative
    competitor is the strongest geometrically bad candidate or the deployed
    score threshold, whichever is higher. A missing good candidate supplies
    no label, so it contributes a differentiable zero rather than a false one.
    """
    if logits.ndim != 1 or ious.ndim != 1 or logits.shape != ious.shape:
        raise ValueError('logits and ious must be aligned one-dimensional arrays')
    if not 0.0 < score_threshold < 1.0:
        raise ValueError('score_threshold must be in (0, 1)')
    if not 0.0 < iou_threshold <= 1.0 or margin < 0.0:
        raise ValueError('invalid IoU threshold or margin')
    if not torch.isfinite(logits).all() or not torch.isfinite(ious).all():
        raise ValueError('non-finite selection inputs')
    zero = logits.sum() * 0.0
    good = ious.detach() >= iou_threshold
    if not bool(good.any()):
        return zero, None, None
    good_indices = torch.nonzero(good, as_tuple=False).flatten()
    best_good = int(good_indices[torch.argmax(logits[good])])
    bad_indices = torch.nonzero(~good, as_tuple=False).flatten()
    threshold_logit = logits.new_tensor(math.log(
        score_threshold / (1.0 - score_threshold)))
    if bad_indices.numel():
        best_bad = int(bad_indices[torch.argmax(logits[~good])])
        competitor = torch.maximum(logits[best_bad], threshold_logit)
    else:
        best_bad = None
        competitor = threshold_logit
    return F.relu(margin + competitor - logits[best_good]), best_good, best_bad
