"""Server-only trainable geometry reference. No quality head or B updates."""
import torch
from torch import nn
from torch.nn import functional as F


class SizeReference(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(256, 16, 1), nn.ReLU(inplace=False),
                                  nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=False))
        self.output = nn.Conv2d(16, 1, 3, padding=1)
        nn.init.constant_(self.output.bias, -4.)

    def forward(self, detached_p3):
        if detached_p3.requires_grad or detached_p3.grad_fn is not None:
            raise ValueError('Reference must receive detached frozen B features')
        if detached_p3.ndim != 4 or detached_p3.shape[:2] != (1, 256):
            raise ValueError('Expected one 256-channel P3')
        features = self.stem(detached_p3)
        return self.output(F.interpolate(features, scale_factor=4., mode='bilinear', align_corners=False))


def reference_loss(logits, target, valid):
    if logits.shape != target.shape or valid.shape != target.shape:
        raise ValueError('Reference shape mismatch')
    if (not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(target).all())
            or bool(((target < 0) | (target > 1)).any())):
        raise ValueError('Nonfinite or invalid reference values')
    mass = (target*valid).sum()
    if float(mass) <= 0: raise ValueError('No valid positive geometry supervision')
    p = logits.sigmoid()
    weight = torch.where(target > 0, .25*(target-p).abs(), .75*p.square())
    return (F.binary_cross_entropy_with_logits(logits, target, reduction='none')*weight*valid).sum()/mass
