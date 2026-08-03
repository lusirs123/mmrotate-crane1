"""Causal multi-cue association for the protected native/S7 candidate pool.

The detector and both proposal lanes stay frozen.  This module learns only six
non-negative cue weights on source video and applies a native-first causal
selector at inference.  It never reads sequence identities as features and it
never accesses a future frame.
"""

import math
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn


CUE_NAMES = (
    'calibrated_score_logit',
    'negative_normalized_center_distance',
    'rotated_iou',
    'negative_log_scale_change',
    'periodic_angle_similarity',
    'dino_roi_appearance_similarity',
)


def _inverse_softplus(value: float) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError('Temporal cue weights must start positive')
    return math.log(math.expm1(value))


class S7TemporalAssociationScorer(nn.Module):
    """Positive linear fusion of score, motion, geometry and appearance."""

    def __init__(self, initial_weights=None):
        super().__init__()
        if initial_weights is None:
            initial_weights = [1.0, 1.0, 1.0, 0.5, 0.5, 0.5]
        if len(initial_weights) != len(CUE_NAMES):
            raise ValueError('Expected {} temporal cue weights'.format(
                len(CUE_NAMES)))
        initial = torch.tensor(
            [float(value) for value in initial_weights], dtype=torch.float32)
        if bool((initial <= 0.0).any().item()) or not bool(
                torch.isfinite(initial).all().item()):
            raise ValueError('Temporal cue weights must be finite and positive')
        self.raw_weights = nn.Parameter(torch.tensor(
            [_inverse_softplus(value) for value in initial.tolist()],
            dtype=torch.float32))
        self.register_buffer('initial_weights', initial)

    def weights(self) -> torch.Tensor:
        import torch.nn.functional as functional

        return functional.softplus(self.raw_weights)

    def forward(self, cues: torch.Tensor) -> torch.Tensor:
        if cues.ndim != 2 or cues.shape[1] != len(CUE_NAMES):
            raise ValueError('Temporal cues must have shape [N, {}]'.format(
                len(CUE_NAMES)))
        if not bool(torch.isfinite(cues).all().item()):
            raise ValueError('Temporal cues contain non-finite values')
        return cues @ self.weights()

    def prior_loss(self) -> torch.Tensor:
        return (self.weights() - self.initial_weights).square().mean()

    def state_summary(self) -> Dict[str, float]:
        values = self.weights().detach().cpu().tolist()
        return {name: float(value) for name, value in zip(CUE_NAMES, values)}


def _default_rotated_iou(current: torch.Tensor,
                         previous: torch.Tensor) -> torch.Tensor:
    from mmcv.ops import box_iou_rotated

    return box_iou_rotated(
        current.float(), previous.reshape(1, 5).float()).reshape(-1)


def build_temporal_cues(
        detections: torch.Tensor, embeddings: torch.Tensor,
        previous_box: torch.Tensor, previous_embedding: torch.Tensor,
        rotated_iou_fn: Optional[Callable] = None) -> torch.Tensor:
    """Build candidate cues relative to one strictly previous-frame state."""
    if detections.ndim != 2 or detections.shape[1] != 6:
        raise ValueError('Temporal detections must have shape [N, 6]')
    if embeddings.ndim != 2 or embeddings.shape[0] != detections.shape[0]:
        raise ValueError('Temporal embeddings must align with detections')
    if previous_box.numel() != 5 or previous_embedding.ndim != 1:
        raise ValueError('Previous temporal state has invalid shape')
    if previous_embedding.shape[0] != embeddings.shape[1]:
        raise ValueError('Previous embedding channel count does not match')
    if detections.shape[0] == 0:
        return detections.new_zeros((0, len(CUE_NAMES)))

    eps = 1e-6
    boxes = detections[:, :5]
    scores = detections[:, 5].clamp(eps, 1.0 - eps)
    score_logit = torch.log(scores) - torch.log1p(-scores)
    displacement = torch.linalg.vector_norm(
        boxes[:, :2] - previous_box[:2].reshape(1, 2), dim=1)
    current_diag = torch.linalg.vector_norm(
        boxes[:, 2:4].abs().clamp_min(eps), dim=1)
    previous_diag = torch.linalg.vector_norm(
        previous_box[2:4].abs().clamp_min(eps), dim=0)
    center_normalizer = (0.5 * (current_diag + previous_diag)).clamp_min(1.0)
    normalized_center = (displacement / center_normalizer).clamp(max=20.0)

    iou_function = rotated_iou_fn or _default_rotated_iou
    rotated_iou = iou_function(boxes, previous_box).to(
        device=boxes.device, dtype=boxes.dtype).reshape(-1).clamp(0.0, 1.0)
    if rotated_iou.shape[0] != boxes.shape[0]:
        raise ValueError('Rotated IoU function returned the wrong count')

    log_scale = torch.log(boxes[:, 2:4].abs().clamp_min(eps))
    previous_log_scale = torch.log(
        previous_box[2:4].abs().clamp_min(eps)).reshape(1, 2)
    scale_change = (log_scale - previous_log_scale).abs().mean(dim=1)
    angle_delta = boxes[:, 4] - previous_box[4]
    angle_similarity = torch.cos(2.0 * angle_delta)

    normalized_embedding = torch.nn.functional.normalize(
        embeddings.float(), dim=1, eps=eps)
    normalized_previous = torch.nn.functional.normalize(
        previous_embedding.float().reshape(1, -1), dim=1, eps=eps)
    appearance = (normalized_embedding * normalized_previous).sum(dim=1)

    return torch.stack((
        score_logit.clamp(-12.0, 12.0),
        -normalized_center,
        rotated_iou,
        -scale_change.clamp(max=12.0),
        angle_similarity,
        appearance.clamp(-1.0, 1.0)), dim=1)


def temporal_pair_losses(
        scorer: S7TemporalAssociationScorer, cues: torch.Tensor,
        gt_overlap: torch.Tensor, source_ids: torch.Tensor,
        riou_threshold: float, margin: float,
        retention_weight: float, gain_weight: float,
        prior_weight: float) -> Dict:
    """Source-only rank losses; quality is never used as a standalone score."""
    if gt_overlap.ndim != 1 or source_ids.ndim != 1:
        raise ValueError('Temporal overlaps and source ids must be vectors')
    if cues.shape[0] != gt_overlap.shape[0] or gt_overlap.shape != source_ids.shape:
        raise ValueError('Temporal training inputs have mismatched counts')
    fused = scorer(cues)
    zero = fused.sum() * 0.0
    retention = zero
    gain = zero
    native_indices = torch.nonzero(source_ids == 0, as_tuple=False).flatten()
    native_top = native_indices[0] if native_indices.numel() else None
    positive_indices = torch.nonzero(
        gt_overlap >= float(riou_threshold), as_tuple=False).flatten()
    wrong_indices = torch.nonzero(
        gt_overlap < float(riou_threshold), as_tuple=False).flatten()
    native_top_correct = bool(
        native_top is not None
        and gt_overlap[native_top] >= float(riou_threshold))
    positive = None
    if positive_indices.numel():
        positive = positive_indices[torch.argmax(gt_overlap[positive_indices])]

    retention_pair_count = 0
    gain_pair_count = 0
    if native_top_correct and wrong_indices.numel():
        competitor = wrong_indices[torch.argmax(fused[wrong_indices].detach())]
        retention = torch.relu(
            float(margin) + fused[competitor] - fused[native_top])
        retention_pair_count = 1
    elif native_top is not None and positive is not None:
        competitors = wrong_indices
        if competitors.numel():
            competitor = competitors[torch.argmax(fused[competitors].detach())]
            gain = torch.relu(
                float(margin) + fused[competitor] - fused[positive])
        else:
            gain = torch.relu(
                float(margin) + fused[native_top] - fused[positive])
        gain_pair_count = 1

    prior = scorer.prior_loss()
    return dict(
        loss_s7_temporal_retention=retention * float(retention_weight),
        loss_s7_temporal_gain=gain * float(gain_weight),
        loss_s7_temporal_prior=prior * float(prior_weight),
        s7_temporal_retention_pair_count=retention_pair_count,
        s7_temporal_gain_pair_count=gain_pair_count,
        s7_temporal_native_top1_correct=int(native_top_correct),
        s7_temporal_usable_candidate_count=int(positive_indices.numel()),
        s7_temporal_candidate_count=int(cues.shape[0]))


class CausalTemporalCandidateSelector:
    """Native-first selector with pending confirmation and exact reset rules."""

    def __init__(self, scorer: S7TemporalAssociationScorer,
                 max_candidates: int = 100, min_confirmations: int = 2,
                 override_margin: float = 0.25,
                 max_center_distance: float = 3.0,
                 min_rotated_iou: float = 0.05,
                 min_appearance_similarity: float = 0.20):
        self.scorer = scorer
        self.max_candidates = int(max_candidates)
        self.min_confirmations = int(min_confirmations)
        self.override_margin = float(override_margin)
        self.max_center_distance = float(max_center_distance)
        self.min_rotated_iou = float(min_rotated_iou)
        self.min_appearance_similarity = float(min_appearance_similarity)
        if self.max_candidates < 1 or self.min_confirmations < 1:
            raise ValueError('Temporal candidate and confirmation counts must be positive')
        self.reset()

    def reset(self):
        self.previous_box = None
        self.previous_embedding = None
        self.previous_seq = None
        self.previous_frame = None
        self.override_active = False
        self.pending_box = None
        self.pending_embedding = None
        self.pending_count = 0

    @staticmethod
    def _first_index(mask: torch.Tensor) -> Optional[int]:
        indices = torch.nonzero(mask, as_tuple=False).flatten()
        return None if indices.numel() == 0 else int(indices[0].item())

    def _continuous(self, seq: str, frame: int) -> bool:
        return bool(
            self.previous_box is not None
            and self.previous_seq == str(seq)
            and self.previous_frame is not None
            and int(frame) == int(self.previous_frame) + 1)

    def _continuity_ok(self, cue: torch.Tensor) -> bool:
        center_ok = float(-cue[1].item()) <= self.max_center_distance
        geometry_ok = float(cue[2].item()) >= self.min_rotated_iou
        appearance_ok = float(cue[5].item()) >= self.min_appearance_similarity
        return bool(center_ok and (geometry_ok or appearance_ok))

    def _store_previous(self, detections, embeddings, index, seq, frame):
        self.previous_box = detections[index, :5].detach().clone()
        self.previous_embedding = embeddings[index].detach().clone()
        self.previous_seq = str(seq)
        self.previous_frame = int(frame)

    def select(self, detections: torch.Tensor, embeddings: torch.Tensor,
               source_ids: torch.Tensor, seq: str, frame: int,
               valid_mask: Optional[torch.Tensor] = None) -> Dict:
        if detections.ndim != 2 or detections.shape[1] != 6:
            raise ValueError('Temporal selector expects [N, 6] detections')
        if (embeddings.ndim != 2
                or embeddings.shape[0] != detections.shape[0]
                or source_ids.shape != (detections.shape[0],)):
            raise ValueError('Temporal selector candidate metadata is misaligned')
        if valid_mask is None:
            valid_mask = torch.ones(
                detections.shape[0], dtype=torch.bool, device=detections.device)
        else:
            valid_mask = valid_mask.to(device=detections.device, dtype=torch.bool)
        bounded = torch.arange(
            detections.shape[0], device=detections.device) < self.max_candidates
        eligible = valid_mask & bounded
        native_index = self._first_index(eligible & (source_ids == 0))
        fallback_index = native_index
        if fallback_index is None:
            fallback_index = self._first_index(eligible)
        if fallback_index is None:
            self.reset()
            return dict(
                selected_index=None,
                order=torch.arange(detections.shape[0], device=detections.device),
                reason='no_valid_candidate', reset=True,
                override=False, pending_count=0,
                candidate_index=None, candidate_source=None,
                candidate_advantage=None, candidate_margin_ok=False,
                candidate_continuity_ok=False, candidate_override_ok=False)

        reset = not self._continuous(seq, frame)
        if reset:
            self.reset()
            selected = fallback_index
            reason = 'native_fallback_after_reset'
            candidate = None
            advantage = None
            margin_ok = False
            continuity_ok = False
            override_ok = False
            self._store_previous(detections, embeddings, selected, seq, frame)
        else:
            cues = build_temporal_cues(
                detections, embeddings, self.previous_box,
                self.previous_embedding)
            fused = self.scorer(cues)
            masked = fused.masked_fill(~eligible, float('-inf'))
            candidate = int(torch.argmax(masked).item())
            advantage = float(
                (fused[candidate] - fused[fallback_index]).detach().item())
            margin_ok = bool(advantage >= self.override_margin)
            continuity_ok = bool(self._continuity_ok(cues[candidate]))
            override_ok = bool(
                candidate != fallback_index
                and margin_ok
                and continuity_ok)
            if self.override_active and override_ok:
                selected = candidate
                reason = 'confirmed_override_continues'
            elif self.override_active:
                selected = fallback_index
                reason = 'native_fallback_after_override_failure'
                self.override_active = False
                self.pending_box = None
                self.pending_embedding = None
                self.pending_count = 0
            elif override_ok:
                if self.pending_box is None:
                    self.pending_count = 1
                else:
                    pending_cues = build_temporal_cues(
                        detections, embeddings, self.pending_box,
                        self.pending_embedding)
                    self.pending_count = (
                        self.pending_count + 1
                        if self._continuity_ok(pending_cues[candidate]) else 1)
                self.pending_box = detections[candidate, :5].detach().clone()
                self.pending_embedding = embeddings[candidate].detach().clone()
                if self.pending_count >= self.min_confirmations:
                    selected = candidate
                    reason = 'override_confirmed'
                    self.override_active = True
                    self.pending_box = None
                    self.pending_embedding = None
                    self.pending_count = 0
                else:
                    selected = fallback_index
                    reason = 'native_fallback_pending_confirmation'
            else:
                selected = fallback_index
                reason = 'native_fallback_no_override_evidence'
                self.pending_box = None
                self.pending_embedding = None
                self.pending_count = 0
            self._store_previous(detections, embeddings, selected, seq, frame)

        all_indices = torch.arange(detections.shape[0], device=detections.device)
        order = torch.cat((
            all_indices.new_tensor([selected]),
            all_indices[all_indices != selected]), dim=0)
        return dict(
            selected_index=int(selected), order=order,
            reason=reason, reset=bool(reset),
            override=bool(selected != fallback_index),
            selected_source=('native_s14' if int(source_ids[selected].item()) == 0
                             else 'supplement_s7'),
            native_fallback_index=(None if native_index is None else int(native_index)),
            candidate_index=(None if candidate is None else int(candidate)),
            candidate_source=(
                None if candidate is None else
                ('native_s14' if int(source_ids[candidate].item()) == 0
                 else 'supplement_s7')),
            candidate_advantage=(
                None if advantage is None else float(advantage)),
            candidate_margin_ok=bool(margin_ok),
            candidate_continuity_ok=bool(continuity_ok),
            candidate_override_ok=bool(override_ok),
            pending_count=int(self.pending_count),
            override_active=bool(self.override_active))
