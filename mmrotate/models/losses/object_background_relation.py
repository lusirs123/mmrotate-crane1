"""Object/background relation loss for the cached DINO source experiment."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectBackgroundRelationDistillation(nn.Module):
    """Match teacher and student object/background cosine relation maps.

    The teacher is detached.  With ``protect_geometry=True`` the base FPN
    feature is detached before the classification adapter, so this auxiliary
    loss updates the adapter while the ordinary detection loss still trains
    the configured FPN and detection heads.
    """

    def __init__(self, loss_weight=0.05, feature_level=0,
                 protect_geometry=True):
        super().__init__()
        if loss_weight < 0 or feature_level < 0:
            raise ValueError('invalid object/background relation settings')
        self.loss_weight = float(loss_weight)
        self.feature_level = int(feature_level)
        self.protect_geometry = bool(protect_geometry)

    @staticmethod
    def _masks(boxes, meta, height, width, device):
        if boxes is None or boxes.numel() == 0:
            return None
        pad_h, pad_w = meta.get('pad_shape', meta['img_shape'])[:2]
        img_h, img_w = meta['img_shape'][:2]
        yy = ((torch.arange(height, device=device) + 0.5) * pad_h / height)[:, None]
        xx = ((torch.arange(width, device=device) + 0.5) * pad_w / width)[None, :]
        valid = (xx < img_w) & (yy < img_h)
        obj = torch.zeros((height, width), dtype=torch.bool, device=device)
        for box in boxes.detach():
            values = box[:5]
            if (not torch.isfinite(values).all().item()
                    or values[2] <= 0 or values[3] <= 0):
                continue
            cx, cy, bw, bh, theta = values
            dx, dy = xx - cx, yy - cy
            local_x = dx * torch.cos(theta) + dy * torch.sin(theta)
            local_y = -dx * torch.sin(theta) + dy * torch.cos(theta)
            obj |= ((local_x.abs() <= bw / 2)
                    & (local_y.abs() <= bh / 2))
        obj &= valid
        if not obj.any().item():
            return None
        binary = obj.float()[None, None]
        outer = F.max_pool2d(binary, 9, 1, 4)[0, 0] > 0
        inner = F.max_pool2d(binary, 3, 1, 1)[0, 0] > 0
        bg = outer & ~inner & valid
        if not bg.any().item():
            return None
        return obj, bg

    @staticmethod
    def _relation(features, obj):
        unit = F.normalize(features.float(), dim=0, eps=1e-6)
        prototype = F.normalize(unit[:, obj].mean(dim=1), dim=0, eps=1e-6)
        return (unit * prototype[:, None, None]).sum(dim=0)

    def forward(self, student_features, teacher_features, gt_bboxes,
                img_metas):
        if (not isinstance(student_features, (tuple, list))
                or self.feature_level >= len(student_features)):
            raise ValueError('student feature level is unavailable')
        student = student_features[self.feature_level]
        if not isinstance(teacher_features, torch.Tensor):
            raise ValueError('teacher_features must be a tensor')
        if student.ndim != 4 or teacher_features.ndim != 4:
            raise ValueError('relation features must be [B,C,H,W]')
        if student.size(0) != teacher_features.size(0):
            raise ValueError('student/teacher batch mismatch')
        teacher = teacher_features.detach().to(
            device=student.device, dtype=student.dtype, non_blocking=True)
        student = F.interpolate(student, size=teacher.shape[-2:],
                                mode='bilinear', align_corners=False)
        terms = []
        for index, (boxes, meta) in enumerate(zip(gt_bboxes, img_metas)):
            masks = self._masks(boxes, meta, teacher.shape[-2],
                                teacher.shape[-1], student.device)
            if masks is None:
                continue
            obj, bg = masks
            student_relation = self._relation(student[index], obj)
            teacher_relation = self._relation(teacher[index], obj)
            terms.append(0.5 * (
                F.mse_loss(student_relation[obj], teacher_relation[obj])
                + F.mse_loss(student_relation[bg], teacher_relation[bg])))
        if not terms:
            return student.sum() * 0.0
        return torch.nan_to_num(torch.stack(terms).mean() * self.loss_weight,
                                nan=0.0, posinf=0.0, neginf=0.0)
