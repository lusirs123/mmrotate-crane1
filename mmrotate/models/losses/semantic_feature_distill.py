"""Memory-bounded offline DINO feature distillation for SymEOOD."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticFeatureDistillation(nn.Module):
    """Match one student FPN level to one cached teacher feature map.

    The teacher is always detached.  Spatial token sampling bounds memory and
    keeps this auxiliary loss much smaller than a live DINO forward pass.
    """

    def __init__(self, student_channels=256, teacher_channels=1024,
                 loss_weight=0.05, feature_level=0, max_tokens=4096):
        super().__init__()
        if student_channels <= 0 or teacher_channels <= 0:
            raise ValueError('feature channels must be positive')
        if loss_weight < 0:
            raise ValueError('loss_weight must be non-negative')
        if feature_level < 0 or max_tokens <= 0:
            raise ValueError('feature_level/max_tokens must be valid')
        self.feature_level = int(feature_level)
        self.loss_weight = float(loss_weight)
        self.max_tokens = int(max_tokens)
        self.project = nn.Conv2d(
            int(student_channels), int(teacher_channels), 1, bias=False)

    @staticmethod
    def _student_level(features, level):
        if not isinstance(features, (tuple, list)) or level >= len(features):
            raise ValueError(
                'student features do not contain level {}'.format(level))
        return features[level]

    def forward(self, student_features, teacher_feature, spatial_mask=None):
        student = self._student_level(student_features, self.feature_level)
        if not isinstance(teacher_feature, torch.Tensor):
            raise ValueError('teacher_feature must be a tensor')
        if student.ndim != 4 or teacher_feature.ndim != 4:
            raise ValueError('distillation features must be [B,C,H,W]')
        if student.size(0) != teacher_feature.size(0):
            raise ValueError('student/teacher batch size mismatch')
        teacher = teacher_feature.detach().to(
            device=student.device, dtype=student.dtype, non_blocking=True)
        target_h, target_w = teacher.shape[-2:]
        if target_h * target_w > self.max_tokens:
            scale = (float(self.max_tokens) /
                     float(target_h * target_w)) ** 0.5
            target_h = max(1, int(target_h * scale))
            target_w = max(1, int(target_w * scale))
            teacher = F.interpolate(
                teacher, size=(target_h, target_w), mode='bilinear',
                align_corners=False)
        # Downsample before the 256->1024 projection.  Projecting a 128x128
        # P3 map first would create a needless 64 MiB FP32 activation/sample.
        student = F.interpolate(
            student, size=(target_h, target_w), mode='bilinear',
            align_corners=False)
        projected = self.project(student)
        projected = F.normalize(projected, dim=1, eps=1e-6)
        teacher = F.normalize(teacher, dim=1, eps=1e-6)
        per_token = 1.0 - (projected * teacher).sum(dim=1)
        if spatial_mask is not None:
            if spatial_mask.ndim != 4 or spatial_mask.size(1) != 1:
                raise ValueError('spatial_mask must be [B,1,H,W]')
            # Nearest downsampling can discard a one-cell foreground object.
            # A token is supervised if any source cell it covers is foreground.
            target_size = per_token.shape[-2:]
            if all(source >= target for source, target in zip(
                    spatial_mask.shape[-2:], target_size)):
                mask = F.adaptive_max_pool2d(
                    spatial_mask.float(), target_size)[:, 0] > 0
            else:
                mask = F.interpolate(
                    spatial_mask.float(), size=target_size,
                    mode='nearest')[:, 0] > 0
            if bool(mask.any()):
                per_token = per_token[mask]
            else:
                return projected.sum() * 0.0
        return torch.nan_to_num(
            per_token.mean() * self.loss_weight,
            nan=0.0, posinf=0.0, neginf=0.0)
