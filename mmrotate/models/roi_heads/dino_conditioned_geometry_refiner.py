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
from mmrotate.models.losses.sym_kld_calculator import sym_kld


def _build_fc_tower(input_channels, fc_channels, num_fcs):
    layers = []
    current_channels = int(input_channels)
    for _ in range(int(num_fcs)):
        layers.extend([nn.Linear(current_channels, int(fc_channels)),
                       nn.ReLU(inplace=True)])
        current_channels = int(fc_channels)
    return nn.Sequential(*layers), current_channels


def _init_fc_tower(tower):
    for module in tower.modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, a=1.0)
            nn.init.zeros_(module.bias)


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
                 decoded_geometry_loss_weight=0.0,
                 temporal_size_loss_weight=0.0,
                 retention_loss_weight=0.0,
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
        self.decoded_geometry_loss_weight = float(
            decoded_geometry_loss_weight)
        self.temporal_size_loss_weight = float(temporal_size_loss_weight)
        self.retention_loss_weight = float(retention_loss_weight)
        if self.decoded_geometry_loss_weight < 0.0:
            raise ValueError(
                'Decoded geometry loss weight must be non-negative')
        if self.temporal_size_loss_weight < 0.0:
            raise ValueError('Temporal size loss weight must be non-negative')
        if self.retention_loss_weight < 0.0:
            raise ValueError('Retention loss weight must be non-negative')
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
        self.shared_fcs, output_channels = _build_fc_tower(
            flattened, fc_channels, num_fcs)
        self.delta_head = nn.Linear(output_channels, 5)
        self._init_weights(bool(zero_init_output))

    def _init_weights(self, zero_init_output):
        _init_fc_tower(self.shared_fcs)
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

    @staticmethod
    def _matched_gt_boxes(proposal_list, gt_box_list):
        if proposal_list is None or gt_box_list is None:
            raise RuntimeError(
                'Decoded geometry loss requires proposals and GT boxes')
        if len(proposal_list) != len(gt_box_list):
            raise RuntimeError('Proposal/GT batch-size mismatch')
        matched = []
        for proposal, gt_boxes in zip(proposal_list, gt_box_list):
            count = int(proposal.shape[0])
            if count == 0:
                continue
            if count != 1 or int(gt_boxes.shape[0]) != 1:
                raise RuntimeError(
                    'Geometry refiner requires one proposal and one GT OBB')
            matched.append(canonicalize_le90(gt_boxes[:, :5]))
        if matched:
            return torch.cat(matched, dim=0)
        device = proposal_list[0].device if proposal_list else 'cpu'
        return torch.zeros((0, 5), device=device)

    @staticmethod
    def _temporal_size_error_loss(predicted_deltas, target_deltas,
                                  pair_indices):
        if pair_indices is None or len(pair_indices) == 0:
            return predicted_deltas.sum() * 0.0
        pairs = torch.as_tensor(
            pair_indices, dtype=torch.long, device=predicted_deltas.device)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise RuntimeError('Temporal pair indices must have shape [N, 2]')
        if (pairs.numel() > 0 and
                (int(pairs.min()) < 0 or
                 int(pairs.max()) >= int(predicted_deltas.shape[0]))):
            raise RuntimeError('Temporal pair index is out of bounds')
        errors = predicted_deltas[:, 2:4] - target_deltas[:, 2:4]
        difference = errors[pairs[:, 1]] - errors[pairs[:, 0]]
        return F.smooth_l1_loss(difference, torch.zeros_like(difference))

    def _continuous_retention_loss(self, predicted_five,
                                   proposal_list, gt_box_list):
        """Keep accurate source anchors unchanged without a hard router.

        The detached continuous weight is derived only from source GT.  It is
        not an inference input and contains no domain/sequence/frame identity.
        Accurate anchors receive strong identity regularization; inaccurate
        anchors remain free to move under the geometry objective.
        """
        proposals = torch.cat([
            item for item in self.canonicalize_proposals(proposal_list)
            if item.shape[0] > 0], dim=0)
        gt_boxes = self._matched_gt_boxes(
            proposal_list, gt_box_list).to(proposals.device)
        if proposals.shape != gt_boxes.shape:
            raise RuntimeError('Retention proposal/GT shape mismatch')
        raw = torch.nan_to_num(
            sym_kld(proposals, gt_boxes),
            nan=16.0, posinf=16.0, neginf=0.0).clamp(min=0.0, max=16.0)
        quality = torch.exp(-torch.sqrt(raw + 1e-9)).detach()
        per_sample = F.smooth_l1_loss(
            predicted_five, torch.zeros_like(predicted_five),
            reduction='none').mean(dim=1)
        return (per_sample * quality).sum() / quality.sum().clamp(min=1e-6)

    def loss(self,
             predicted_deltas,
             target_deltas,
             proposal_list=None,
             gt_box_list=None,
             temporal_pair_indices=None):
        if predicted_deltas.shape != target_deltas.shape:
            raise RuntimeError('Geometry refiner prediction/target mismatch')
        if predicted_deltas.numel() == 0:
            zero = predicted_deltas.sum() * 0.0
            for parameter in self.parameters():
                zero = zero + parameter.sum() * 0.0
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
        if self.decoded_geometry_loss_weight > 0.0:
            decoded = self.decode_and_normalize(
                proposal_list, predicted_deltas)
            decoded = torch.cat(
                [item for item in decoded if item.shape[0] > 0], dim=0)
            gt_boxes = self._matched_gt_boxes(
                proposal_list, gt_box_list).to(decoded.device)
            if decoded.shape != gt_boxes.shape:
                raise RuntimeError('Decoded geometry/GT shape mismatch')
            raw = sym_kld(decoded, gt_boxes)
            value = (torch.sqrt(
                1.0 + torch.nan_to_num(
                    raw, nan=1e4, posinf=1e4, neginf=0.0)) - 1.0
                     ).clamp(max=10.0).mean()
            weighted = value * self.decoded_geometry_loss_weight
            diagnostics['refiner_decoded_geometry_objective'] = (
                weighted.detach())
            total = total + weighted
        if self.temporal_size_loss_weight > 0.0:
            value = self._temporal_size_error_loss(
                predicted_deltas, target_deltas, temporal_pair_indices)
            weighted = value * self.temporal_size_loss_weight
            diagnostics['refiner_temporal_size_objective'] = (
                weighted.detach())
            diagnostics['refiner_temporal_pair_count'] = (
                predicted_deltas.new_tensor(
                    0 if temporal_pair_indices is None
                    else len(temporal_pair_indices)))
            total = total + weighted
        if self.retention_loss_weight > 0.0:
            value = self._continuous_retention_loss(
                predicted_deltas, proposal_list, gt_box_list)
            weighted = value * self.retention_loss_weight
            diagnostics['refiner_continuous_retention_objective'] = (
                weighted.detach())
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
            decoded_geometry_loss_weight=self.decoded_geometry_loss_weight,
            temporal_size_loss_weight=self.temporal_size_loss_weight,
            retention_loss_weight=self.retention_loss_weight,
            domain_routing=False, sequence_frame_routing=False,
            temporal_state=False)


@ROTATED_HEADS.register_module()
class DinoConditionedDualTowerGeometryRefiner(
        DinoConditionedGeometryRefiner):
    """Compose an independent size tower and center-angle pose tower.

    Both towers consume the same Rotated ROIAlign tensor.  Their trainable
    fully connected layers are disjoint, preventing center/angle objectives
    from changing the size representation that controls DFR.
    """

    def __init__(self,
                 *args,
                 train_size_tower=True,
                 train_pose_tower=True,
                 train_roi_extractor=False,
                 evaluation_only=False,
                 **kwargs):
        if any(kwargs.get(name, True) is not True for name in (
                'refine_center', 'refine_size', 'refine_angle')):
            raise ValueError(
                'Dual-tower V2 requires center, size, and angle components')
        in_channels = int(kwargs.get('in_channels', 256))
        roi_output_size = int(kwargs.get('roi_output_size', 7))
        fc_channels = int(kwargs.get('fc_channels', 256))
        num_fcs = int(kwargs.get('num_fcs', 2))
        zero_init_output = bool(kwargs.get('zero_init_output', True))
        super().__init__(*args, **kwargs)
        del self.shared_fcs
        del self.delta_head
        flattened = in_channels * roi_output_size ** 2
        self.size_fcs, size_channels = _build_fc_tower(
            flattened, fc_channels, num_fcs)
        self.pose_fcs, pose_channels = _build_fc_tower(
            flattened, fc_channels, num_fcs)
        self.size_head = nn.Linear(size_channels, 2)
        self.pose_head = nn.Linear(pose_channels, 3)
        _init_fc_tower(self.size_fcs)
        _init_fc_tower(self.pose_fcs)
        if zero_init_output:
            nn.init.zeros_(self.size_head.weight)
            nn.init.zeros_(self.size_head.bias)
            nn.init.zeros_(self.pose_head.weight)
            nn.init.zeros_(self.pose_head.bias)
        self.train_size_tower = bool(train_size_tower)
        self.train_pose_tower = bool(train_pose_tower)
        self.train_roi_extractor = bool(train_roi_extractor)
        self.evaluation_only = bool(evaluation_only)
        if (self.evaluation_only and (
                self.train_size_tower or self.train_pose_tower
                or self.train_roi_extractor)):
            raise ValueError(
                'Evaluation-only dual tower cannot expose trainable parts')
        if (not self.evaluation_only
                and not (self.train_size_tower or self.train_pose_tower)):
            raise ValueError(
                'At least one dual-tower branch must be trainable')
        self._apply_trainability_contract()

    @staticmethod
    def _set_requires_grad(module, enabled):
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)

    def _apply_trainability_contract(self):
        self._set_requires_grad(self.size_fcs, self.train_size_tower)
        self._set_requires_grad(self.size_head, self.train_size_tower)
        self._set_requires_grad(self.pose_fcs, self.train_pose_tower)
        self._set_requires_grad(self.pose_head, self.train_pose_tower)
        self._set_requires_grad(
            self.roi_extractor, self.train_roi_extractor)
        if not self.train_size_tower:
            self.size_fcs.eval()
            self.size_head.eval()
        if not self.train_pose_tower:
            self.pose_fcs.eval()
            self.pose_head.eval()
        if not self.train_roi_extractor:
            self.roi_extractor.eval()

    def train(self, mode=True):
        super().train(mode)
        self._apply_trainability_contract()
        return self

    def forward(self, features, proposal_list):
        proposals = self.canonicalize_proposals(proposal_list)
        rois = rbbox2roi(proposals)
        if rois.shape[0] == 0:
            return features[0].new_zeros((0, 5))
        flattened = self.roi_extractor(features, rois).flatten(1)
        size = self.size_head(self.size_fcs(flattened))
        pose = self.pose_head(self.pose_fcs(flattened))
        deltas = torch.cat([pose[:, :2], size, pose[:, 2:3]], dim=1)
        return self.mask_deltas(deltas)

    def component_contract(self):
        contract = super().component_contract()
        contract.update(dict(
            architecture='dual_tower_size_pose_v2',
            roi_features_shared=True,
            trainable_fc_towers_shared=False,
            train_size_tower=self.train_size_tower,
            train_pose_tower=self.train_pose_tower,
            train_roi_extractor=self.train_roi_extractor,
            evaluation_only=self.evaluation_only,
            size_components=['dlogw', 'dlogh'],
            pose_components=['dx_local', 'dy_local', 'dangle']))
        return contract


@ROTATED_HEADS.register_module()
class DinoConditionedCausalHistoryRefiner(DinoConditionedGeometryRefiner):
    """Current-frame anchored refiner with rejectable causal history.

    Historical appearance and geometry can only contribute a bounded residual
    on top of the current-frame prediction.  With no valid history the result
    is exactly the current-only branch.  Sequence/frame identifiers are not
    accepted by this module and therefore cannot become routing features.
    """

    def __init__(self,
                 *args,
                 history_horizon=4,
                 max_history_center_delta=0.35,
                 max_history_log_size_delta=0.45,
                 max_history_angle_delta_deg=20.0,
                 history_gate_bias=-4.0,
                 **kwargs):
        self.history_horizon = int(history_horizon)
        if self.history_horizon <= 0:
            raise ValueError('history_horizon must be positive')
        fc_channels = int(kwargs.get('fc_channels', 256))
        in_channels = int(kwargs.get('in_channels', 256))
        roi_output_size = int(kwargs.get('roi_output_size', 7))
        num_fcs = int(kwargs.get('num_fcs', 2))
        zero_init_output = bool(kwargs.get('zero_init_output', True))
        super().__init__(*args, **kwargs)
        flattened = in_channels * roi_output_size ** 2
        self.history_fcs, history_channels = _build_fc_tower(
            flattened, fc_channels, num_fcs)
        self.relative_geometry_fcs = nn.Sequential(
            nn.Linear(5, fc_channels), nn.ReLU(inplace=True),
            nn.Linear(fc_channels, fc_channels), nn.ReLU(inplace=True))
        self.age_embedding = nn.Embedding(
            self.history_horizon + 1, fc_channels)
        self.history_attention = nn.Linear(fc_channels * 3, 1)
        self.history_fusion = nn.Sequential(
            nn.Linear(fc_channels + history_channels, fc_channels),
            nn.ReLU(inplace=True))
        self.history_delta_head = nn.Linear(fc_channels, 5)
        self.history_gate_head = nn.Linear(fc_channels, 1)
        _init_fc_tower(self.history_fcs)
        _init_fc_tower(self.relative_geometry_fcs)
        nn.init.normal_(self.age_embedding.weight, std=0.01)
        nn.init.xavier_uniform_(self.history_attention.weight)
        nn.init.zeros_(self.history_attention.bias)
        for module in self.history_fusion.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=1.0)
                nn.init.zeros_(module.bias)
        if zero_init_output:
            nn.init.zeros_(self.history_delta_head.weight)
            nn.init.zeros_(self.history_delta_head.bias)
        nn.init.zeros_(self.history_gate_head.weight)
        nn.init.constant_(self.history_gate_head.bias, float(history_gate_bias))
        bounds = [
            float(max_history_center_delta),
            float(max_history_center_delta),
            float(max_history_log_size_delta),
            float(max_history_log_size_delta),
            math.radians(float(max_history_angle_delta_deg))]
        if any(value <= 0.0 for value in bounds):
            raise ValueError('History residual bounds must be positive')
        self.register_buffer(
            'history_residual_bounds', torch.tensor(bounds))
        self.history_gate_bias = float(history_gate_bias)

    def _roi_hidden(self, features, proposal_list, tower):
        rois = rbbox2roi(self.canonicalize_proposals(proposal_list))
        if rois.shape[0] == 0:
            return features[0].new_zeros((0, self.delta_head.in_features))
        roi_features = self.roi_extractor(features, rois).flatten(1)
        return tower(roi_features)

    def _relative_geometry(self, current, history):
        batch, horizon, _ = history.shape
        current_flat = current[:, None, :].expand(
            batch, horizon, 5).reshape(-1, 5)
        history_flat = history.reshape(-1, 5)
        encoded = self.bbox_coder.encode(
            canonicalize_le90(current_flat),
            canonicalize_le90(history_flat))
        return torch.nan_to_num(
            encoded, nan=0.0, posinf=4.0, neginf=-4.0
        ).clamp(min=-4.0, max=4.0).reshape(batch, horizon, 5)

    def forward_causal(self,
                       current_features,
                       proposal_list,
                       history_features,
                       history_proposals,
                       history_valid_mask,
                       history_ages=None):
        batch = len(proposal_list)
        if any(int(item.shape[0]) != 1 for item in proposal_list):
            raise RuntimeError(
                'Causal history refiner requires one current proposal/image')
        if history_proposals.shape != (batch, self.history_horizon, 5):
            raise RuntimeError('History proposal tensor has invalid shape')
        if history_valid_mask.shape != (batch, self.history_horizon):
            raise RuntimeError('History validity tensor has invalid shape')
        if len(history_features) != len(current_features):
            raise RuntimeError('Current/history FPN level mismatch')
        if any(feature.shape[:2] != (
                batch, self.history_horizon) for feature in history_features):
            raise RuntimeError('History FPN tensor has invalid batch/horizon')
        current_hidden = self._roi_hidden(
            current_features, proposal_list, self.shared_fcs)
        flat_history_features = [
            feature.reshape(
                batch * self.history_horizon, *feature.shape[2:])
            for feature in history_features]
        flat_history_boxes = [
            history_proposals[index, age:age + 1]
            for index in range(batch)
            for age in range(self.history_horizon)]
        history_hidden = self._roi_hidden(
            flat_history_features, flat_history_boxes,
            self.history_fcs).reshape(batch, self.history_horizon, -1)
        current_boxes = torch.cat(
            self.canonicalize_proposals(proposal_list), dim=0)
        relative = self.relative_geometry_fcs(
            self._relative_geometry(
                current_boxes, history_proposals).reshape(-1, 5)
        ).reshape(batch, self.history_horizon, -1)
        if history_ages is None:
            history_ages = torch.arange(
                1, self.history_horizon + 1,
                device=history_proposals.device)[None, :].expand(batch, -1)
        ages = history_ages.to(dtype=torch.long).clamp(
            min=1, max=self.history_horizon)
        history_token = history_hidden + relative + self.age_embedding(ages)
        current_expanded = current_hidden[:, None, :].expand(
            -1, self.history_horizon, -1)
        attention_input = torch.cat(
            [current_expanded, history_token, relative], dim=-1)
        scores = self.history_attention(attention_input).squeeze(-1)
        valid = history_valid_mask.to(dtype=torch.bool)
        safe_scores = scores.masked_fill(~valid, -1e4)
        weights = torch.softmax(safe_scores, dim=1) * valid.to(scores.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1.0)
        context = (weights[..., None] * history_token).sum(dim=1)
        fused = self.history_fusion(torch.cat(
            [current_hidden, context], dim=-1))
        current_deltas = self.mask_deltas(self.delta_head(current_hidden))
        history_deltas = torch.tanh(self.history_delta_head(fused))
        bounds = self.history_residual_bounds.to(
            device=history_deltas.device, dtype=history_deltas.dtype)
        gate = torch.sigmoid(self.history_gate_head(fused))
        any_valid = valid.any(dim=1, keepdim=True).to(gate.dtype)
        correction = self.mask_deltas(
            history_deltas * bounds.reshape(1, 5))
        return current_deltas + any_valid * gate * correction

    def forward(self, features, proposal_list, **kwargs):
        if not kwargs:
            return super().forward(features, proposal_list)
        required = {
            'history_features', 'history_proposals',
            'history_valid_mask'}
        missing = required.difference(kwargs)
        if missing:
            raise RuntimeError(
                'Causal forward is missing: ' + ', '.join(sorted(missing)))
        return self.forward_causal(
            features, proposal_list,
            kwargs['history_features'], kwargs['history_proposals'],
            kwargs['history_valid_mask'], kwargs.get('history_ages'))

    def component_contract(self):
        contract = super().component_contract()
        contract.update(dict(
            architecture='current_anchored_causal_history_refiner_v1',
            history_horizon=self.history_horizon,
            strictly_causal=True,
            current_frame_anchored=True,
            bounded_history_residual=True,
            rejectable_history_gate=True,
            exact_current_only_when_no_history=True,
            history_gate_bias=self.history_gate_bias,
            history_residual_bounds=[
                float(value) for value in
                self.history_residual_bounds.detach().cpu().tolist()],
            domain_routing=False,
            sequence_frame_routing=False,
            temporal_state=False))
        return contract


@ROTATED_HEADS.register_module()
class K1AnchoredCausalPhaseGeometryRefiner(
        DinoConditionedCausalHistoryRefiner):
    """Preserve current K1 geometry while using DINO/history as context.

    The decoded box is anchored to the current frozen-K1 prediction whenever
    it exists.  A native-DINO box is only the anchor fallback; it otherwise
    supplies a second current-frame RoI token and the coordinate reference for
    strictly causal history.  Angle output uses a continuous double-angle
    phase vector ``(sin(2 da), cos(2 da)-1)``.  Therefore an all-zero output is
    an exact identity correction without evaluating ``atan2(0, 0)``.
    """

    def __init__(self, *args, conditioning_gate_bias=-2.0,
                 inference_component_mode='full', **kwargs):
        max_current_center_delta = float(
            kwargs.pop('max_current_center_delta', 0.12))
        max_current_log_size_delta = float(
            kwargs.pop('max_current_log_size_delta', 0.18))
        max_current_angle_delta_deg = float(
            kwargs.pop('max_current_angle_delta_deg', 12.0))
        in_channels = int(kwargs.get('in_channels', 256))
        roi_output_size = int(kwargs.get('roi_output_size', 7))
        fc_channels = int(kwargs.get('fc_channels', 256))
        num_fcs = int(kwargs.get('num_fcs', 2))
        zero_init_output = bool(kwargs.get('zero_init_output', True))
        super().__init__(*args, **kwargs)

        # Replace the scalar-angle current head with a continuous phase head.
        output_channels = int(self.delta_head.in_features)
        self.delta_head = nn.Linear(output_channels, 6)
        if zero_init_output:
            nn.init.zeros_(self.delta_head.weight)
            nn.init.zeros_(self.delta_head.bias)

        flattened = in_channels * roi_output_size ** 2
        self.conditioning_fcs, conditioning_channels = _build_fc_tower(
            flattened, fc_channels, num_fcs)
        self.conditioning_fusion = nn.Sequential(
            nn.Linear(output_channels + conditioning_channels, fc_channels),
            nn.ReLU(inplace=True))
        self.conditioning_gate_head = nn.Linear(fc_channels, 1)
        _init_fc_tower(self.conditioning_fcs)
        _init_fc_tower(self.conditioning_fusion)
        nn.init.zeros_(self.conditioning_gate_head.weight)
        nn.init.constant_(
            self.conditioning_gate_head.bias, float(conditioning_gate_bias))
        self.conditioning_gate_bias = float(conditioning_gate_bias)
        allowed_modes = {
            'full', 'current_only', 'center_only', 'k1_identity'}
        if inference_component_mode not in allowed_modes:
            raise ValueError(
                'Unsupported inference_component_mode: {}'.format(
                    inference_component_mode))
        self.inference_component_mode = str(inference_component_mode)
        current_bounds = [
            max_current_center_delta, max_current_center_delta,
            max_current_log_size_delta, max_current_log_size_delta,
            math.radians(max_current_angle_delta_deg)]
        if any(value <= 0.0 for value in current_bounds):
            raise ValueError('Current residual bounds must be positive')
        self.register_buffer(
            'current_residual_bounds', torch.tensor(current_bounds))

    def _apply_inference_component_mode(self, five_deltas):
        """Apply a parameter-free evaluation ablation to five-delta output.

        The switch never changes trained parameters or feature extraction.  It
        only determines which already-predicted residual components may reach
        the shared decoder.  This keeps trainer/runtime decoding identical and
        makes component attribution exact.
        """
        mode = self.inference_component_mode
        if mode == 'full' or mode == 'current_only':
            return five_deltas
        if mode == 'center_only':
            mask = five_deltas.new_tensor(
                [1.0, 1.0, 0.0, 0.0, 0.0]).reshape(1, 5)
            return five_deltas * mask
        if mode == 'k1_identity':
            return torch.zeros_like(five_deltas)
        raise RuntimeError('Invalid inference component mode')

    @staticmethod
    def _phase_to_five(deltas):
        if deltas.ndim != 2 or deltas.shape[1] != 6:
            raise RuntimeError('Phase deltas must have shape [N, 6]')
        sin_value = deltas[:, 4]
        cos_value = 1.0 + deltas[:, 5]
        angle = 0.5 * torch.atan2(sin_value, cos_value)
        return torch.cat([deltas[:, :4], angle[:, None]], dim=1)

    @staticmethod
    def _five_to_phase(deltas):
        if deltas.ndim != 2 or deltas.shape[1] != 5:
            raise RuntimeError('Five deltas must have shape [N, 5]')
        double_angle = 2.0 * deltas[:, 4]
        return torch.cat([
            deltas[:, :4],
            torch.sin(double_angle)[:, None],
            (torch.cos(double_angle) - 1.0)[:, None]], dim=1)

    def _mask_phase_deltas(self, deltas):
        values = [self.refine_center, self.refine_center,
                  self.refine_size, self.refine_size,
                  self.refine_angle, self.refine_angle]
        mask = deltas.new_tensor(values).reshape(1, 6)
        return deltas * mask

    def _bound_current_phase(self, raw_phase):
        raw_phase = self._mask_phase_deltas(raw_phase)
        raw_five = self._phase_to_five(raw_phase)
        bounds = self.current_residual_bounds.to(
            device=raw_five.device, dtype=raw_five.dtype)
        bounded = torch.tanh(raw_five / bounds.reshape(1, 5)) * bounds
        return self._mask_phase_deltas(self._five_to_phase(bounded))

    def _current_hidden(self, features, anchor_list,
                        conditioning_proposal_list):
        anchor_hidden = self._roi_hidden(
            features, anchor_list, self.shared_fcs)
        conditioning_hidden = self._roi_hidden(
            features, conditioning_proposal_list, self.conditioning_fcs)
        if anchor_hidden.shape != conditioning_hidden.shape:
            raise RuntimeError('Anchor/conditioning RoI count mismatch')
        fused = self.conditioning_fusion(torch.cat(
            [anchor_hidden, conditioning_hidden], dim=1))
        gate = torch.sigmoid(self.conditioning_gate_head(fused))
        return anchor_hidden + gate * fused

    def forward_causal(self,
                       current_features,
                       proposal_list,
                       history_features,
                       history_proposals,
                       history_valid_mask,
                       history_ages=None,
                       conditioning_proposal_list=None):
        batch = len(proposal_list)
        if conditioning_proposal_list is None:
            conditioning_proposal_list = proposal_list
        if len(conditioning_proposal_list) != batch:
            raise RuntimeError('Conditioning proposal batch-size mismatch')
        if (any(int(item.shape[0]) != 1 for item in proposal_list)
                or any(int(item.shape[0]) != 1
                       for item in conditioning_proposal_list)):
            raise RuntimeError(
                'K1-anchored refiner requires one anchor and one '
                'conditioning proposal/image')
        if history_proposals.shape != (batch, self.history_horizon, 5):
            raise RuntimeError('History proposal tensor has invalid shape')
        if history_valid_mask.shape != (batch, self.history_horizon):
            raise RuntimeError('History validity tensor has invalid shape')
        if len(history_features) != len(current_features):
            raise RuntimeError('Current/history FPN level mismatch')
        if any(feature.shape[:2] != (
                batch, self.history_horizon) for feature in history_features):
            raise RuntimeError('History FPN tensor has invalid batch/horizon')

        current_hidden = self._current_hidden(
            current_features, proposal_list, conditioning_proposal_list)
        flat_history_features = [
            feature.reshape(
                batch * self.history_horizon, *feature.shape[2:])
            for feature in history_features]
        flat_history_boxes = [
            history_proposals[index, age:age + 1]
            for index in range(batch)
            for age in range(self.history_horizon)]
        history_hidden = self._roi_hidden(
            flat_history_features, flat_history_boxes,
            self.history_fcs).reshape(batch, self.history_horizon, -1)
        conditioning_boxes = torch.cat(
            self.canonicalize_proposals(conditioning_proposal_list), dim=0)
        relative = self.relative_geometry_fcs(
            self._relative_geometry(
                conditioning_boxes, history_proposals).reshape(-1, 5)
        ).reshape(batch, self.history_horizon, -1)
        if history_ages is None:
            history_ages = torch.arange(
                1, self.history_horizon + 1,
                device=history_proposals.device)[None, :].expand(batch, -1)
        ages = history_ages.to(dtype=torch.long).clamp(
            min=1, max=self.history_horizon)
        history_token = history_hidden + relative + self.age_embedding(ages)
        current_expanded = current_hidden[:, None, :].expand(
            -1, self.history_horizon, -1)
        scores = self.history_attention(torch.cat(
            [current_expanded, history_token, relative], dim=-1)).squeeze(-1)
        valid = history_valid_mask.to(dtype=torch.bool)
        weights = torch.softmax(
            scores.masked_fill(~valid, -1e4), dim=1) * valid.to(scores.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1.0)
        context = (weights[..., None] * history_token).sum(dim=1)
        fused = self.history_fusion(torch.cat(
            [current_hidden, context], dim=-1))

        current_phase = self._bound_current_phase(
            self.delta_head(current_hidden))
        current_five = self.mask_deltas(self._phase_to_five(current_phase))
        history_raw = torch.tanh(self.history_delta_head(fused))
        bounds = self.history_residual_bounds.to(
            device=history_raw.device, dtype=history_raw.dtype)
        gate = torch.sigmoid(self.history_gate_head(fused))
        any_valid = valid.any(dim=1, keepdim=True).to(gate.dtype)
        correction = self.mask_deltas(history_raw * bounds.reshape(1, 5))
        if self.inference_component_mode == 'current_only':
            combined = current_five
        else:
            combined = current_five + any_valid * gate * correction
        combined = self._apply_inference_component_mode(combined)
        return self._mask_phase_deltas(self._five_to_phase(combined))

    def forward(self, features, proposal_list, **kwargs):
        conditioning = kwargs.pop(
            'conditioning_proposal_list', proposal_list)
        if not kwargs:
            hidden = self._current_hidden(
                features, proposal_list, conditioning)
            current = self._bound_current_phase(self.delta_head(hidden))
            five = self._apply_inference_component_mode(
                self.mask_deltas(self._phase_to_five(current)))
            return self._mask_phase_deltas(self._five_to_phase(five))
        required = {
            'history_features', 'history_proposals',
            'history_valid_mask'}
        missing = required.difference(kwargs)
        if missing:
            raise RuntimeError(
                'Causal forward is missing: ' + ', '.join(sorted(missing)))
        return self.forward_causal(
            features, proposal_list,
            kwargs['history_features'], kwargs['history_proposals'],
            kwargs['history_valid_mask'], kwargs.get('history_ages'),
            conditioning_proposal_list=conditioning)

    def encode_targets(self, proposal_list, gt_box_list):
        five = super().encode_targets(proposal_list, gt_box_list)
        return self._mask_phase_deltas(self._five_to_phase(five))

    def decode_and_normalize(self, proposal_list, deltas, img_metas=None):
        five = self.mask_deltas(self._phase_to_five(
            self._mask_phase_deltas(deltas)))
        return DinoConditionedGeometryRefiner.decode_and_normalize(
            self, proposal_list, five, img_metas=img_metas)

    def loss(self,
             predicted_deltas,
             target_deltas,
             proposal_list=None,
             gt_box_list=None,
             temporal_pair_indices=None):
        if predicted_deltas.shape != target_deltas.shape:
            raise RuntimeError('Phase refiner prediction/target mismatch')
        if predicted_deltas.numel() == 0:
            zero = predicted_deltas.sum() * 0.0
            for parameter in self.parameters():
                zero = zero + parameter.sum() * 0.0
            return dict(loss_geometry_refiner=zero)
        predicted_five = self.mask_deltas(
            self._phase_to_five(predicted_deltas))
        target_five = self.mask_deltas(self._phase_to_five(target_deltas))
        diagnostics = {}
        total = predicted_deltas.sum() * 0.0
        if self.refine_center:
            value = F.smooth_l1_loss(
                predicted_five[:, :2], target_five[:, :2])
            weighted = value * self.center_loss_weight
            diagnostics['refiner_center_objective'] = weighted.detach()
            total = total + weighted
        if self.refine_size:
            value = F.smooth_l1_loss(
                predicted_five[:, 2:4], target_five[:, 2:4])
            weighted = value * self.size_loss_weight
            diagnostics['refiner_size_objective'] = weighted.detach()
            total = total + weighted
        if self.refine_angle:
            predicted_vector = torch.stack([
                predicted_deltas[:, 4], 1.0 + predicted_deltas[:, 5]],
                dim=1)
            target_vector = torch.stack([
                target_deltas[:, 4], 1.0 + target_deltas[:, 5]], dim=1)
            predicted_vector = F.normalize(predicted_vector, dim=1, eps=1e-6)
            target_vector = F.normalize(target_vector, dim=1, eps=1e-6)
            value = F.smooth_l1_loss(predicted_vector, target_vector)
            weighted = value * self.angle_loss_weight
            diagnostics['refiner_phase_angle_objective'] = weighted.detach()
            total = total + weighted
        if self.decoded_geometry_loss_weight > 0.0:
            decoded = self.decode_and_normalize(
                proposal_list, predicted_deltas)
            decoded = torch.cat(
                [item for item in decoded if item.shape[0] > 0], dim=0)
            gt_boxes = self._matched_gt_boxes(
                proposal_list, gt_box_list).to(decoded.device)
            raw = sym_kld(decoded, gt_boxes)
            value = (torch.sqrt(1.0 + torch.nan_to_num(
                raw, nan=1e4, posinf=1e4, neginf=0.0)) - 1.0
                     ).clamp(max=10.0).mean()
            weighted = value * self.decoded_geometry_loss_weight
            diagnostics['refiner_decoded_geometry_objective'] = (
                weighted.detach())
            total = total + weighted
        if self.temporal_size_loss_weight > 0.0:
            value = self._temporal_size_error_loss(
                predicted_five, target_five, temporal_pair_indices)
            weighted = value * self.temporal_size_loss_weight
            diagnostics['refiner_temporal_size_objective'] = (
                weighted.detach())
            diagnostics['refiner_temporal_pair_count'] = (
                predicted_deltas.new_tensor(
                    0 if temporal_pair_indices is None
                    else len(temporal_pair_indices)))
            total = total + weighted
        if self.retention_loss_weight > 0.0:
            value = self._continuous_retention_loss(
                predicted_five, proposal_list, gt_box_list)
            weighted = value * self.retention_loss_weight
            diagnostics['refiner_continuous_retention_objective'] = (
                weighted.detach())
            total = total + weighted
        diagnostics['loss_geometry_refiner'] = total
        return diagnostics

    def component_contract(self):
        contract = super().component_contract()
        contract.update(dict(
            architecture='k1_anchored_causal_phase_refiner_v2',
            representation='six_delta_xywh_sin2a_cos2a_residual',
            current_k1_geometry_anchor=True,
            native_dino_anchor_fallback=True,
            native_dino_current_conditioning=True,
            continuous_double_angle_phase=True,
            zero_phase_is_exact_identity=True,
            bounded_current_residual=True,
            current_residual_bounds=[
                float(value) for value in
                self.current_residual_bounds.detach().cpu().tolist()],
            conditioning_gate_bias=self.conditioning_gate_bias,
            inference_component_mode=self.inference_component_mode,
            same_forward_all_domains=True,
            domain_routing=False,
            sequence_frame_routing=False,
            temporal_state=False))
        return contract


@ROTATED_HEADS.register_module()
class K1RetentiveCausalPhaseGeometryRefiner(
        K1AnchoredCausalPhaseGeometryRefiner):
    """V3 contract: continuous K1 retention plus source-pair consistency."""

    def component_contract(self):
        contract = super().component_contract()
        contract.update(dict(
            architecture='k1_retentive_causal_phase_refiner_v3',
            continuous_k1_retention=True,
            source_adjacent_pair_error_consistency=True,
            inference_sequence_input=False,
            domain_routing=False,
            sequence_frame_routing=False,
            temporal_state=False))
        return contract
