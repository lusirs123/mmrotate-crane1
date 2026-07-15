"""Independent RGB platform-context sidecar.

The module intentionally has no dependency on the SymEOOD backbone/FPN.  It
predicts one internal platform heatmap from RGB and scores *existing* beam
candidates by sampling the heatmap inside ``K(beam_candidate)``.  It never
predicts or refines a beam box.
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    """Choose a GroupNorm group count that also works with batch size one."""
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvGNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True))


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1,
                      groups=in_channels, bias=False),
            nn.GroupNorm(_group_count(in_channels), in_channels),
            nn.SiLU(inplace=True))
        self.pointwise = ConvGNAct(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class PlatformContextSidecar(nn.Module):
    """Small trainable RGB encoder with a stride-4 heatmap decoder.

    GroupNorm is used instead of BatchNorm so that the probe can train one
    1024x1024 frame at a time without changing hidden running statistics.
    """

    def __init__(self, base_channels: int = 24, decoder_channels: int = 64):
        super().__init__()
        c1 = int(base_channels)
        c2, c3, c4 = c1 * 2, c1 * 4, c1 * 6
        dc = int(decoder_channels)

        self.stem = ConvGNAct(3, c1, stride=2)             # stride 2
        self.stage4 = DepthwiseSeparableBlock(c1, c2, 2)  # stride 4
        self.stage8 = nn.Sequential(
            DepthwiseSeparableBlock(c2, c3, 2),
            DepthwiseSeparableBlock(c3, c3, 1))           # stride 8
        self.stage16 = nn.Sequential(
            DepthwiseSeparableBlock(c3, c4, 2),
            DepthwiseSeparableBlock(c4, c4, 1))           # stride 16

        self.lat4 = nn.Conv2d(c2, dc, 1)
        self.lat8 = nn.Conv2d(c3, dc, 1)
        self.lat16 = nn.Conv2d(c4, dc, 1)
        self.fuse8 = ConvGNAct(dc, dc)
        self.fuse4 = ConvGNAct(dc, dc)
        self.heatmap_head = nn.Sequential(
            ConvGNAct(dc, dc // 2),
            nn.Conv2d(dc // 2, 1, 1))
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                        nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        # Start from a conservative low platform probability.
        nn.init.constant_(self.heatmap_head[-1].bias, -4.0)

    def forward(self, rgb01: torch.Tensor) -> torch.Tensor:
        """Return platform logits at stride 4.

        Args:
            rgb01: RGB tensor in [0, 1], shape [N, 3, H, W].
        """
        x2 = self.stem(rgb01)
        x4 = self.stage4(x2)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        p8 = self.fuse8(
            self.lat8(x8) + F.interpolate(
                self.lat16(x16), size=x8.shape[-2:], mode='bilinear',
                align_corners=False))
        p4 = self.fuse4(
            self.lat4(x4) + F.interpolate(
                p8, size=x4.shape[-2:], mode='bilinear',
                align_corners=False))
        return self.heatmap_head(p4)


def platform_heatmap_loss(logits: torch.Tensor, target: torch.Tensor,
                          focal_gamma: float = 2.0,
                          pos_alpha: float = 0.75,
                          dice_weight: float = 1.0
                          ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Focal BCE + Dice loss for the sparse platform mask."""
    if target.shape != logits.shape:
        target = F.interpolate(target, size=logits.shape[-2:], mode='nearest')
    target = target.to(dtype=logits.dtype)
    prob = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    pt = prob * target + (1.0 - prob) * (1.0 - target)
    alpha = pos_alpha * target + (1.0 - pos_alpha) * (1.0 - target)
    focal = (alpha * (1.0 - pt).pow(float(focal_gamma)) * bce).mean()

    intersection = (prob * target).sum(dim=(1, 2, 3))
    denominator = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0))
    dice = dice.mean()
    total = focal + float(dice_weight) * dice
    return total, dict(focal=focal.detach(), dice=dice.detach())


def candidate_platform_boxes(beam_boxes: torch.Tensor,
                             seq_k: Dict) -> torch.Tensor:
    """Map beam OBBs forward to internal platform OBBs with rigid K."""
    if beam_boxes.numel() == 0:
        return beam_boxes.new_zeros((0, 5))
    center = beam_boxes[:, :2]
    width = beam_boxes[:, 2].clamp(min=1e-6)
    height = beam_boxes[:, 3].clamp(min=1e-6)
    theta = beam_boxes[:, 4]
    width_is_long = width >= height
    long_len = torch.where(width_is_long, width, height)
    short_len = torch.where(width_is_long, height, width)
    long_theta = torch.where(width_is_long, theta, theta + math.pi / 2)
    ux = torch.stack([torch.cos(long_theta), torch.sin(long_theta)], dim=1)
    flip = ((ux[:, 0] < 0)
            | ((ux[:, 0].abs() < 1e-6) & (ux[:, 1] < 0)))
    ux = torch.where(flip[:, None], -ux, ux)
    uy = torch.stack([-ux[:, 1], ux[:, 0]], dim=1)
    mapped_center = (
        center
        + ux * (float(seq_k.get('offset_long_k', 0.0)) * long_len)[:, None]
        + uy * (float(seq_k.get('offset_short_k', 0.0)) * short_len)[:, None])
    mapped_w = float(seq_k['width_k']) * long_len
    mapped_h = float(seq_k['height_k']) * short_len
    mapped_theta = long_theta + float(seq_k.get('dtheta', 0.0))
    mapped_theta = torch.remainder(
        mapped_theta + math.pi / 2, math.pi) - math.pi / 2
    return torch.stack([
        mapped_center[:, 0], mapped_center[:, 1], mapped_w, mapped_h,
        mapped_theta], dim=1)


def sample_candidate_context(heatmap: torch.Tensor,
                             platform_boxes: torch.Tensor,
                             image_shape: Tuple[int, int],
                             grid_size: int = 3) -> torch.Tensor:
    """Read center/region platform evidence for many candidates at once."""
    if platform_boxes.numel() == 0:
        return platform_boxes.new_zeros((0,))
    if heatmap.ndim != 4 or heatmap.shape[0] != 1 or heatmap.shape[1] != 1:
        raise ValueError('heatmap must have shape [1, 1, H, W]')
    grid_size = max(int(grid_size), 1)
    coords = torch.linspace(
        -0.4, 0.4, grid_size, device=platform_boxes.device,
        dtype=platform_boxes.dtype)
    try:
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')
    except TypeError:  # PyTorch < 1.10
        yy, xx = torch.meshgrid(coords, coords)
    local = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    center = platform_boxes[:, :2]
    wh = platform_boxes[:, 2:4]
    theta = platform_boxes[:, 4]
    local_xy = local[None, :, :] * wh[:, None, :]
    cos_t = torch.cos(theta)[:, None]
    sin_t = torch.sin(theta)[:, None]
    x = (center[:, 0:1] + local_xy[:, :, 0] * cos_t
         - local_xy[:, :, 1] * sin_t)
    y = (center[:, 1:2] + local_xy[:, :, 0] * sin_t
         + local_xy[:, :, 1] * cos_t)
    image_h, image_w = int(image_shape[0]), int(image_shape[1])
    gx = 2.0 * x / max(float(image_w), 1.0) - 1.0
    gy = 2.0 * y / max(float(image_h), 1.0) - 1.0
    sampling_grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
    sampled = F.grid_sample(
        heatmap, sampling_grid, mode='bilinear', padding_mode='zeros',
        align_corners=False)[0, 0]
    region_score = sampled.mean(dim=1)
    center_index = sampled.shape[1] // 2
    center_score = sampled[:, center_index]
    return 0.5 * center_score + 0.5 * region_score
