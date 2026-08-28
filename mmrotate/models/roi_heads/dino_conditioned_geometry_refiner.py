"""Shared source-only geometry refiner for native-DINO rotated boxes.

The same module owns proposal canonicalization, target encoding, component
masking, and decoding in both the standalone trainer and formal runtime.  It
contains no domain, sequence, frame, threshold, or temporal-state logic.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrotate.core import build_bbox_coder, rbbox2roi
from mmrotate.models.builder import (ROTATED_HEADS, build_roi_extractor)


def canonicalize_le90(boxes):
    """Return the equivalent ``le90`` representation with width >= height."""
    if boxes.numel() == 0:
        return boxes
    result = boxes.clone()
    swap = result[..., 2] < result[..., 3]
    width = torch.where(swap, result[..., 3], result[..., 2])
    height = torch.where(swap, result[..., 2], result[..., 3])
    angle = torch.where(
        swap, result[..., 4] + math.pi / 2.0, result[..., 4])
    normalized = (
        torch.remainder(angle + math.pi / 2.0, math.pi) - math.pi / 2.0)
    already_in_range = ((angle >= -math.pi / 2.0)
                        & (angle < math.pi / 2.0))
    angle = torch.where(already_in_range, angle, normalized)
    result[..., 2] = width
    result[..., 3] = height
    result[..., 4] = angle
    return result


def map_original_obb_to_model(boxes, img_meta):
    """Map original-image OBBs exactly like ``RResize``/``RRandomFlip``."""
    boxes = canonicalize_le90(boxes)
    if boxes.numel() == 0:
        return boxes
    result = boxes.clone()
    scale = result.new_tensor(img_meta['scale_factor']).reshape(-1)
    if scale.numel() < 2:
        raise RuntimeError('img_meta.scale_factor is invalid')
    width_scale = scale[0]
    height_scale = scale[1]
    size_scale = torch.sqrt(width_scale * height_scale)
    result[:, 0] *= width_scale
    result[:, 1] *= height_scale
    result[:, 2:4] *= size_scale
    if bool(img_meta.get('flip', False)):
        direction = img_meta.get('flip_direction', 'horizontal')
        image_h, image_w = img_meta['img_shape'][:2]
        if direction in ('horizontal', 'diagonal'):
            result[:, 0] = image_w - result[:, 0] - 1
        if direction in ('vertical', 'diagonal'):
            result[:, 1] = image_h - result[:, 1] - 1
        if direction not in ('horizontal', 'vertical', 'diagonal'):
            raise RuntimeError('Unsupported flip direction: ' + direction)
        if direction != 'diagonal':
            result[:, 4] = torch.remainder(
                math.pi - result[:, 4] + math.pi / 2.0,
                math.pi) - math.pi / 2.0
    return canonicalize_le90(result)


def map_model_obb_to_original(boxes, img_meta):
    """Inverse of :func:`map_original_obb_to_model`."""
    if boxes.numel() == 0:
        return boxes
    result = canonicalize_le90(boxes)
    if bool(img_meta.get('flip', False)):
        # Every supported reflection is self-inverse.
        inverse_meta = dict(img_meta)
        inverse_meta['scale_factor'] = [1.0, 1.0, 1.0, 1.0]
        result = map_original_obb_to_model(result, inverse_meta)
    scale = result.new_tensor(img_meta['scale_factor']).reshape(-1)
    width_scale = scale[0]
    height_scale = scale[1]
    size_scale = torch.sqrt(width_scale * height_scale)
    result[:, 0] /= width_scale
    result[:, 1] /= height_scale
    result[:, 2:4] /= size_scale
    return canonicalize_le90(result)


@ROTATED_HEADS.register_module()
class DinoConditionedGeometryRefiner(nn.Module):
    """Refine a native-DINO OBB from frozen SymEOOD FPN RoI features."""

    def __init__(self,
                 roi_extractor=None,
                 in_channels=256,
                 roi_output_size=7,
                 fc_channels=256,
                 num_fcs=2,
                 refine_center=True,
                 refine_size=True,
                 refine_angle=True,
                 zero_init_output=True,
                 center_loss_weight=1.0,
                 size_loss_weight=1.0,
                 angle_loss_weight=1.0,
                 bbox_coder=None):
        super().__init__()
        self.refine_center = bool(refine_center)
        self.refine_size = bool(refine_size)
        self.refine_angle = bool(refine_angle)
        if not any((self.refine_center, self.refine_size,
                    self.refine_angle)):
            raise ValueError('At least one geometry component must be active')
        self.center_loss_weight = float(center_loss_weight)
        self.size_loss_weight = float(size_loss_weight)
        self.angle_loss_weight = float(angle_loss_weight)
        roi_extractor = dict(roi_extractor or dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlignRotated', out_size=int(roi_output_size),
                sample_num=2, clockwise=True),
            out_channels=int(in_channels),
            featmap_strides=[8, 16, 32, 64, 128]))
        self.roi_extractor = build_roi_extractor(roi_extractor)
        bbox_coder = dict(bbox_coder or dict(
            type='DeltaXYWHAOBBoxCoder', angle_range='le90',
            edge_swap=True, proj_xy=True,
            target_means=(0., 0., 0., 0., 0.),
            target_stds=(1., 1., 1., 1., 1.)))
        if (bbox_coder.get('angle_range') != 'le90'
                or not bbox_coder.get('edge_swap', False)
                or not bbox_coder.get('proj_xy', False)):
            raise ValueError(
                'Geometry refiner requires le90 + edge_swap + proj_xy')
        self.bbox_coder = build_bbox_coder(bbox_coder)

        flattened = int(in_channels) * int(roi_output_size) ** 2
        layers = []
        input_channels = flattened
        for _ in range(int(num_fcs)):
            layers.extend([nn.Linear(input_channels, int(fc_channels)),
                           nn.ReLU(inplace=True)])
            input_channels = int(fc_channels)
        self.shared_fcs = nn.Sequential(*layers)
        self.delta_head = nn.Linear(input_channels, 5)
        self._init_weights(bool(zero_init_output))

    def _init_weights(self, zero_init_output):
        for module in self.shared_fcs.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=1.0)
                nn.init.zeros_(module.bias)
        if zero_init_output:
            nn.init.zeros_(self.delta_head.weight)
            nn.init.zeros_(self.delta_head.bias)

    def active_component_mask(self, device=None, dtype=None):
        values = [self.refine_center, self.refine_center,
                  self.refine_size, self.refine_size, self.refine_angle]
        return torch.tensor(
            values, device=device, dtype=dtype or torch.float32)

    def mask_deltas(self, deltas):
        return deltas * self.active_component_mask(
            deltas.device, deltas.dtype).reshape(1, 5)

    @staticmethod
    def canonicalize_proposals(proposal_list):
        return [canonicalize_le90(item[:, :5]) for item in proposal_list]

    def forward(self, features, proposal_list):
        proposals = self.canonicalize_proposals(proposal_list)
        rois = rbbox2roi(proposals)
        if rois.shape[0] == 0:
            return features[0].new_zeros((0, 5))
        roi_features = self.roi_extractor(features, rois)
        hidden = self.shared_fcs(roi_features.flatten(1))
        return self.mask_deltas(self.delta_head(hidden))

    def encode_targets(self, proposal_list, gt_box_list):
        proposals = self.canonicalize_proposals(proposal_list)
        targets = []
        for proposal, gt_boxes in zip(proposals, gt_box_list):
            if proposal.shape[0] == 0:
                continue
            if proposal.shape[0] != 1 or gt_boxes.shape[0] != 1:
                raise RuntimeError(
                    'Geometry refiner requires one DINO OBB and one GT OBB')
            gt = canonicalize_le90(gt_boxes[:, :5])
            targets.append(self.bbox_coder.encode(proposal, gt))
        if not targets:
            device = proposal_list[0].device if proposal_list else 'cpu'
            return torch.zeros((0, 5), device=device)
        return self.mask_deltas(torch.cat(targets, dim=0))

    def decode_and_normalize(self, proposal_list, deltas, img_metas=None):
        """Shared train/runtime decode; return one tensor for each image."""
        proposals = self.canonicalize_proposals(proposal_list)
        counts = [int(item.shape[0]) for item in proposals]
        if sum(counts) != int(deltas.shape[0]):
            raise RuntimeError('Proposal/delta count mismatch')
        outputs = []
        offset = 0
        for index, (proposal, count) in enumerate(zip(proposals, counts)):
            if count == 0:
                outputs.append(proposal)
                continue
            current = self.mask_deltas(deltas[offset:offset + count])
            max_shape = (None if img_metas is None else
                         img_metas[index].get('img_shape'))
            decoded = self.bbox_coder.decode(
                proposal, current, max_shape=max_shape)
            outputs.append(canonicalize_le90(decoded))
            offset += count
        return outputs

    def loss(self, predicted_deltas, target_deltas):
        if predicted_deltas.shape != target_deltas.shape:
            raise RuntimeError('Geometry refiner prediction/target mismatch')
        if predicted_deltas.numel() == 0:
            zero = self.delta_head.weight.sum() * 0.0
            return dict(loss_geometry_refiner=zero)
        diagnostics = {}
        total = predicted_deltas.sum() * 0.0
        if self.refine_center:
            value = F.smooth_l1_loss(
                predicted_deltas[:, :2], target_deltas[:, :2])
            weighted = value * self.center_loss_weight
            diagnostics['refiner_center_objective'] = weighted.detach()
            total = total + weighted
        if self.refine_size:
            value = F.smooth_l1_loss(
                predicted_deltas[:, 2:4], target_deltas[:, 2:4])
            weighted = value * self.size_loss_weight
            diagnostics['refiner_size_objective'] = weighted.detach()
            total = total + weighted
        if self.refine_angle:
            residual = predicted_deltas[:, 4] - target_deltas[:, 4]
            value = (1.0 - torch.cos(2.0 * residual)).mean()
            weighted = value * self.angle_loss_weight
            diagnostics['refiner_angle_objective'] = weighted.detach()
            total = total + weighted
        diagnostics['loss_geometry_refiner'] = total
        return diagnostics

    def component_contract(self):
        return dict(
            representation='five_delta_xywha',
            angle_range='le90', edge_swap=True, proj_xy=True,
            refine_center=self.refine_center,
            refine_size=self.refine_size,
            refine_angle=self.refine_angle,
            active_component_mask=[bool(value) for value in
                                   self.active_component_mask().tolist()],
            domain_routing=False, sequence_frame_routing=False,
            temporal_state=False)
