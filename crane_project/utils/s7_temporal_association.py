"""Causal multi-cue association for the protected native/S7 candidate pool.

The detector and both proposal lanes stay frozen.  The original mode learns
six non-negative cue weights; the optional quality mode adds one dense,
source-supervised candidate-quality cue and may add a same-frame
relative-ranking term.  Both apply a native-first causal selector at
inference.  They never read sequence identities as features and never access
a future frame.
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
QUALITY_CUE_NAMES = CUE_NAMES + ('candidate_quality_logit',)


def _inverse_softplus(value: float) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError('Temporal cue weights must start positive')
    return math.log(math.expm1(value))


class S7TemporalAssociationScorer(nn.Module):
    """Positive linear fusion of score, motion, geometry and appearance."""

    def __init__(self, initial_weights=None, cue_names=None):
        super().__init__()
        self.cue_names = tuple(cue_names or CUE_NAMES)
        if initial_weights is None:
            initial_weights = [1.0, 1.0, 1.0, 0.5, 0.5, 0.5]
            if len(self.cue_names) == len(QUALITY_CUE_NAMES):
                initial_weights.append(0.5)
        if len(initial_weights) != len(self.cue_names):
            raise ValueError('Expected {} temporal cue weights'.format(
                len(self.cue_names)))
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
        if cues.ndim != 2 or cues.shape[1] != len(self.cue_names):
            raise ValueError('Temporal cues must have shape [N, {}]'.format(
                len(self.cue_names)))
        if not bool(torch.isfinite(cues).all().item()):
            raise ValueError('Temporal cues contain non-finite values')
        return cues @ self.weights()

    def prior_loss(self) -> torch.Tensor:
        return (self.weights() - self.initial_weights).square().mean()

    def state_summary(self) -> Dict[str, float]:
        values = self.weights().detach().cpu().tolist()
        return {name: float(value)
                for name, value in zip(self.cue_names, values)}


class S7CandidateQualityHead(nn.Module):
    """Predict continuous candidate max-RIoU from frozen ROI evidence."""

    SCALAR_CHANNELS = 6

    def __init__(self, embedding_channels: int, hidden: int = 128):
        super().__init__()
        if int(embedding_channels) <= 0 or int(hidden) <= 0:
            raise ValueError('Candidate quality dimensions must be positive')
        self.embedding_projection = nn.Sequential(
            nn.Linear(int(embedding_channels), int(hidden)),
            nn.LayerNorm(int(hidden)), nn.GELU())
        self.scalar_projection = nn.Sequential(
            nn.Linear(self.SCALAR_CHANNELS, int(hidden)), nn.GELU())
        self.output = nn.Linear(int(hidden) * 2, 1)
        # A zero output makes the new quality cue constant before training;
        # the audited affine/temporal ordering is therefore the exact start.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _scalar_features(detections: torch.Tensor,
                         source_ids: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        boxes = detections[:, :5]
        scores = detections[:, 5].clamp(eps, 1.0 - eps)
        score_logit = torch.log(scores) - torch.log1p(-scores)
        width = boxes[:, 2].abs().clamp_min(eps)
        height = boxes[:, 3].abs().clamp_min(eps)
        return torch.stack((
            score_logit.clamp(-12.0, 12.0),
            torch.log(width),
            torch.log(height),
            torch.log(width / height),
            torch.sin(boxes[:, 4]),
            source_ids.to(dtype=detections.dtype)), dim=1)

    def forward(self, embedding: torch.Tensor, detections: torch.Tensor,
                source_ids: torch.Tensor) -> torch.Tensor:
        if (embedding.ndim != 2 or detections.ndim != 2
                or detections.shape[1] != 6
                or source_ids.shape != (detections.shape[0],)
                or embedding.shape[0] != detections.shape[0]):
            raise ValueError('Candidate quality inputs are misaligned')
        if detections.shape[0] == 0:
            return detections.new_zeros((0,))
        scalars = self._scalar_features(detections, source_ids)
        hidden = torch.cat((
            self.embedding_projection(embedding.float()),
            self.scalar_projection(scalars.float())), dim=1)
        return self.output(hidden).reshape(-1)


class S7HighResCandidateQualityHead(nn.Module):
    """Fuse frozen semantic ROI evidence with a lightweight S7 ROI readout.

    The high-resolution branch is deliberately a quality readout, not a
    second detector.  Its output is initialized to zero so the phase-2
    native/S7 ordering is unchanged before source-only training.
    """

    SCALAR_CHANNELS = 6

    def __init__(self, embedding_channels: int, highres_channels: int,
                 hidden: int = 32):
        super().__init__()
        if (int(embedding_channels) <= 0 or int(highres_channels) <= 0
                or int(hidden) <= 0):
            raise ValueError('High-resolution quality dimensions must be positive')
        self.semantic_projection = nn.Sequential(
            nn.Linear(int(embedding_channels), int(hidden)),
            nn.LayerNorm(int(hidden)), nn.GELU())
        self.highres_projection = nn.Sequential(
            nn.Linear(int(highres_channels), int(hidden)),
            nn.LayerNorm(int(hidden)), nn.GELU())
        self.scalar_projection = nn.Sequential(
            nn.Linear(self.SCALAR_CHANNELS, int(hidden)), nn.GELU())
        self.output = nn.Linear(int(hidden) * 3, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, embedding: torch.Tensor, highres_embedding: torch.Tensor,
                detections: torch.Tensor,
                source_ids: torch.Tensor) -> torch.Tensor:
        if (embedding.ndim != 2 or highres_embedding.ndim != 2
                or detections.ndim != 2 or detections.shape[1] != 6
                or embedding.shape[0] != detections.shape[0]
                or highres_embedding.shape[0] != detections.shape[0]
                or source_ids.shape != (detections.shape[0],)):
            raise ValueError('High-resolution quality inputs are misaligned')
        if detections.shape[0] == 0:
            return detections.new_zeros((0,))
        scalars = S7CandidateQualityHead._scalar_features(
            detections, source_ids)
        hidden = torch.cat((
            self.semantic_projection(embedding.float()),
            self.highres_projection(highres_embedding.float()),
            self.scalar_projection(scalars.float())), dim=1)
        return self.output(hidden).reshape(-1)


class S7SelectivePromotionHead(nn.Module):
    """Predict S7-vs-native quality advantage and calibrated uncertainty.

    The head never assigns an unconditional score to the whole S7 lane.  It
    compares every eligible S7 candidate with the current native top-1 and
    returns a quality-advantage mean plus a positive uncertainty.  Zero mean
    initialization and positive uncertainty make the initial lower confidence
    bound negative, so an untrained head exactly falls back to native.
    """

    SCALAR_CHANNELS = 9

    def __init__(self, embedding_channels: int, hidden: int = 128,
                 initial_uncertainty: float = 0.5):
        super().__init__()
        if (int(embedding_channels) <= 0 or int(hidden) <= 0
                or float(initial_uncertainty) <= 0.0):
            raise ValueError(
                'Selective promotion dimensions and uncertainty must be '
                'positive')
        self.embedding_projection = nn.Sequential(
            nn.Linear(int(embedding_channels) * 4, int(hidden)),
            nn.LayerNorm(int(hidden)), nn.GELU())
        self.scalar_projection = nn.Sequential(
            nn.Linear(self.SCALAR_CHANNELS, int(hidden)), nn.GELU())
        self.advantage_output = nn.Linear(int(hidden) * 2, 1)
        self.uncertainty_output = nn.Linear(int(hidden) * 2, 1)
        nn.init.zeros_(self.advantage_output.weight)
        nn.init.zeros_(self.advantage_output.bias)
        nn.init.zeros_(self.uncertainty_output.weight)
        nn.init.constant_(
            self.uncertainty_output.bias,
            _inverse_softplus(float(initial_uncertainty)))

    @staticmethod
    def _pair_scalar_features(
            native_detection: torch.Tensor, s7_detections: torch.Tensor,
            native_quality: torch.Tensor,
            s7_quality: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        native_score = native_detection[5].clamp(eps, 1.0 - eps)
        s7_scores = s7_detections[:, 5].clamp(eps, 1.0 - eps)
        native_logit = torch.log(native_score) - torch.log1p(-native_score)
        s7_logits = torch.log(s7_scores) - torch.log1p(-s7_scores)
        native_box = native_detection[:5]
        s7_boxes = s7_detections[:, :5]
        displacement = torch.linalg.vector_norm(
            s7_boxes[:, :2] - native_box[:2].reshape(1, 2), dim=1)
        native_diag = torch.linalg.vector_norm(
            native_box[2:4].abs().clamp_min(eps), dim=0)
        s7_diag = torch.linalg.vector_norm(
            s7_boxes[:, 2:4].abs().clamp_min(eps), dim=1)
        center_distance = displacement / (
            0.5 * (native_diag + s7_diag)).clamp_min(1.0)
        native_log_scale = torch.log(
            native_box[2:4].abs().clamp_min(eps)).reshape(1, 2)
        s7_log_scale = torch.log(s7_boxes[:, 2:4].abs().clamp_min(eps))
        scale_change = (s7_log_scale - native_log_scale).abs().mean(dim=1)
        angle_similarity = torch.cos(
            2.0 * (s7_boxes[:, 4] - native_box[4]))
        native_quality = native_quality.reshape(()).expand_as(s7_quality)
        return torch.stack((
            native_logit.expand_as(s7_logits).clamp(-12.0, 12.0),
            s7_logits.clamp(-12.0, 12.0),
            (s7_logits - native_logit).clamp(-12.0, 12.0),
            native_quality.clamp(-12.0, 12.0),
            s7_quality.clamp(-12.0, 12.0),
            (s7_quality - native_quality).clamp(-12.0, 12.0),
            center_distance.clamp(max=20.0),
            scale_change.clamp(max=12.0),
            angle_similarity), dim=1)

    def forward(self, native_embedding: torch.Tensor,
                native_detection: torch.Tensor,
                native_quality: torch.Tensor,
                s7_embeddings: torch.Tensor,
                s7_detections: torch.Tensor,
                s7_quality: torch.Tensor):
        if (native_embedding.ndim != 1 or native_detection.shape != (6,)
                or native_quality.numel() != 1 or s7_embeddings.ndim != 2
                or s7_detections.ndim != 2
                or s7_detections.shape[1] != 6
                or s7_quality.shape != (s7_detections.shape[0],)
                or s7_embeddings.shape[0] != s7_detections.shape[0]
                or s7_embeddings.shape[1] != native_embedding.shape[0]):
            raise ValueError('Selective promotion pair inputs are misaligned')
        if s7_detections.shape[0] == 0:
            empty = s7_quality.new_zeros((0,))
            return empty, empty
        native = native_embedding.reshape(1, -1).expand_as(s7_embeddings)
        pair_embedding = torch.cat((
            native, s7_embeddings, s7_embeddings - native,
            s7_embeddings * native), dim=1)
        scalars = self._pair_scalar_features(
            native_detection, s7_detections, native_quality, s7_quality)
        hidden = torch.cat((
            self.embedding_projection(pair_embedding.float()),
            self.scalar_projection(scalars.float())), dim=1)
        advantage = self.advantage_output(hidden).reshape(-1)
        uncertainty = torch.nn.functional.softplus(
            self.uncertainty_output(hidden).reshape(-1)).clamp(
                min=1e-4, max=2.0)
        return advantage, uncertainty


class TwoFrameMotionState:
    """Minimal causal state for a two-frame constant-velocity prior.

    Only the two most recent selected boxes and ROI embeddings are retained.
    Sequence names are used solely to reset state at a boundary; they are
    never exposed to the learned head as input features.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.previous_box = None
        self.previous_embedding = None
        self.last_box = None
        self.last_embedding = None
        self.seq = None
        self.frame = None

    @property
    def ready(self) -> bool:
        return bool(self.previous_box is not None and self.last_box is not None)

    def continuous(self, seq: str, frame: int) -> bool:
        return bool(
            self.frame is not None and self.seq == str(seq)
            and int(frame) == int(self.frame) + 1)

    def prepare(self, seq: str, frame: int) -> bool:
        continuous = self.continuous(seq, frame)
        if not continuous:
            self.reset()
        return continuous

    def update(self, box: torch.Tensor, embedding: torch.Tensor,
               seq: str, frame: int):
        if box.numel() != 5 or embedding.ndim != 1:
            raise ValueError('Two-frame state inputs have invalid shapes')
        self.previous_box = self.last_box
        self.previous_embedding = self.last_embedding
        self.last_box = box.detach().clone()
        self.last_embedding = embedding.detach().clone()
        self.seq = str(seq)
        self.frame = int(frame)


def _periodic_angle_delta(current: torch.Tensor,
                          previous: torch.Tensor) -> torch.Tensor:
    """Return the shortest le90-compatible angle delta (period pi)."""
    delta = current - previous
    return 0.5 * torch.atan2(torch.sin(2.0 * delta),
                             torch.cos(2.0 * delta))


def _two_frame_candidate_motion_features(
        detections: torch.Tensor, embeddings: torch.Tensor,
        state: TwoFrameMotionState) -> torch.Tensor:
    """Return seven constant-velocity residual cues per candidate."""
    if not state.ready:
        raise ValueError('Two-frame motion features require two prior frames')
    if detections.ndim != 2 or detections.shape[1] != 6:
        raise ValueError('Two-frame detections must have shape [N, 6]')
    if (embeddings.ndim != 2
            or embeddings.shape[0] != detections.shape[0]
            or state.last_embedding.shape[0] != embeddings.shape[1]):
        raise ValueError('Two-frame embeddings are misaligned')
    if detections.shape[0] == 0:
        return detections.new_zeros((0, 7))

    eps = 1e-6
    previous = state.previous_box.to(
        device=detections.device, dtype=detections.dtype)
    last = state.last_box.to(
        device=detections.device, dtype=detections.dtype)
    predicted_center = last[:2] + (last[:2] - previous[:2])
    previous_log_scale = torch.log(previous[2:4].abs().clamp_min(eps))
    last_log_scale = torch.log(last[2:4].abs().clamp_min(eps))
    predicted_log_scale = last_log_scale + (
        last_log_scale - previous_log_scale)
    predicted_angle = last[4] + _periodic_angle_delta(last[4], previous[4])

    boxes = detections[:, :5]
    predicted_diag = torch.linalg.vector_norm(
        torch.exp(predicted_log_scale), dim=0).clamp_min(1.0)
    center_residual = (boxes[:, :2] - predicted_center.reshape(1, 2)) / (
        0.5 * (torch.linalg.vector_norm(
            boxes[:, 2:4].abs().clamp_min(eps), dim=1)
               + predicted_diag).clamp_min(1.0).reshape(-1, 1))
    center_norm = torch.linalg.vector_norm(center_residual, dim=1)
    scale_residual = (
        torch.log(boxes[:, 2:4].abs().clamp_min(eps))
        - predicted_log_scale.reshape(1, 2))
    angle_similarity = torch.cos(2.0 * (boxes[:, 4] - predicted_angle))
    normalized_embedding = torch.nn.functional.normalize(
        embeddings.float(), dim=1, eps=eps)
    normalized_last = torch.nn.functional.normalize(
        state.last_embedding.to(embeddings.device).float().reshape(1, -1),
        dim=1, eps=eps)
    appearance = (normalized_embedding * normalized_last).sum(dim=1)
    return torch.cat((
        center_residual.clamp(-20.0, 20.0),
        center_norm.clamp(max=20.0).reshape(-1, 1),
        scale_residual.clamp(-12.0, 12.0),
        angle_similarity.reshape(-1, 1),
        appearance.clamp(-1.0, 1.0).reshape(-1, 1)), dim=1)


class S7SmallTemporalRankerHead(nn.Module):
    """Tiny scalar-only native/S7 pair head for small-object ordering.

    The 24 inputs contain current score/quality/pair geometry plus two-frame
    constant-velocity residuals.  No sequence identity, dense feature map,
    optical flow, or additional backbone output is consumed.
    """

    SCALAR_CHANNELS = 24

    def __init__(self, hidden: int = 16, initial_uncertainty: float = 0.5):
        super().__init__()
        if int(hidden) <= 0 or float(initial_uncertainty) <= 0.0:
            raise ValueError('Small temporal ranker settings must be positive')
        self.hidden = nn.Sequential(
            nn.Linear(self.SCALAR_CHANNELS, int(hidden)), nn.GELU(),
            nn.Linear(int(hidden), int(hidden)), nn.GELU())
        self.advantage_output = nn.Linear(int(hidden), 1)
        self.uncertainty_output = nn.Linear(int(hidden), 1)
        nn.init.zeros_(self.advantage_output.weight)
        nn.init.zeros_(self.advantage_output.bias)
        nn.init.zeros_(self.uncertainty_output.weight)
        nn.init.constant_(
            self.uncertainty_output.bias,
            _inverse_softplus(float(initial_uncertainty)))

    @staticmethod
    def pair_features(
            native_embedding: torch.Tensor,
            native_detection: torch.Tensor, native_quality: torch.Tensor,
            s7_embedding: torch.Tensor, s7_detection: torch.Tensor,
            s7_quality: torch.Tensor,
            state: TwoFrameMotionState) -> torch.Tensor:
        if (native_embedding.ndim != 1 or s7_embedding.ndim != 1
                or native_embedding.shape != s7_embedding.shape
                or native_detection.shape != (6,)
                or s7_detection.shape != (6,)
                or native_quality.numel() != 1 or s7_quality.numel() != 1):
            raise ValueError('Small temporal ranker pair is misaligned')
        pair_static = S7SelectivePromotionHead._pair_scalar_features(
            native_detection, s7_detection.reshape(1, 6), native_quality,
            s7_quality.reshape(1))
        eps = 1e-6
        pair_appearance = torch.nn.functional.cosine_similarity(
            native_embedding.float().reshape(1, -1),
            s7_embedding.float().reshape(1, -1), dim=1, eps=eps)
        static = torch.cat((pair_static, pair_appearance.reshape(1, 1)), 1)
        pair_detections = torch.stack((native_detection, s7_detection), 0)
        pair_embeddings = torch.stack((native_embedding, s7_embedding), 0)
        motion = _two_frame_candidate_motion_features(
            pair_detections, pair_embeddings, state).reshape(1, 14)
        features = torch.cat((static, motion), dim=1)
        if features.shape != (1, S7SmallTemporalRankerHead.SCALAR_CHANNELS):
            raise RuntimeError('Unexpected small temporal feature dimension')
        return features

    def forward(self, features: torch.Tensor):
        if (features.ndim != 2
                or features.shape[1] != self.SCALAR_CHANNELS):
            raise ValueError('Small temporal features must have shape [N, 24]')
        hidden = self.hidden(features.float())
        advantage = self.advantage_output(hidden).reshape(-1)
        uncertainty = torch.nn.functional.softplus(
            self.uncertainty_output(hidden).reshape(-1)).clamp(
                min=1e-4, max=2.0)
        return advantage, uncertainty


def _native_and_quality_selected_s7(
        detections: torch.Tensor, source_ids: torch.Tensor,
        quality_logits: torch.Tensor, max_candidates: int,
        valid_mask: Optional[torch.Tensor] = None):
    """Choose native top-1 and one quality-ranked S7 from its lane top-K."""
    if (source_ids.shape != (detections.shape[0],)
            or quality_logits.shape != source_ids.shape):
        raise ValueError('Small temporal candidate metadata is misaligned')
    eligible = torch.ones_like(source_ids, dtype=torch.bool)
    if valid_mask is not None:
        if valid_mask.shape != source_ids.shape:
            raise ValueError('Small temporal valid mask is misaligned')
        eligible &= valid_mask.to(device=source_ids.device, dtype=torch.bool)
    native = torch.nonzero(eligible & (source_ids == 0),
                           as_tuple=False).flatten()
    s7_ranked = torch.nonzero(eligible & (source_ids == 1),
                              as_tuple=False).flatten()
    s7 = s7_ranked[:min(int(max_candidates), int(s7_ranked.numel()))]
    native_index = None
    s7_index = None
    if native.numel():
        native_index = native[torch.argmax(detections[native, 5])]
    if s7.numel():
        # torch.argmax returns the first maximum, preserving lane order for
        # deterministic ties in the frozen quality teacher.
        s7_index = s7[torch.argmax(quality_logits[s7])]
    return native_index, s7_index, int(s7.numel())


def small_temporal_ranker_losses(
        ranker_head: S7SmallTemporalRankerHead,
        embeddings: torch.Tensor, detections: torch.Tensor,
        source_ids: torch.Tensor, quality_logits: torch.Tensor,
        gt_overlap: torch.Tensor, state: TwoFrameMotionState,
        riou_threshold: float, advantage_gap: float = 0.10,
        promotion_margin: float = 0.10,
        uncertainty_multiplier: float = 1.0,
        quality_weight: float = 1.0, classification_weight: float = 1.0,
        retention_weight: float = 2.0, gain_weight: float = 1.0,
        prior_weight: float = 0.01, max_candidates: int = 20) -> Dict:
    """Train one quality-prefiltered S7/native pair using source GT only."""
    if gt_overlap.shape != source_ids.shape:
        raise ValueError('Small temporal overlap targets are misaligned')
    zero = ranker_head.advantage_output.bias.sum() * 0.0
    native_index, s7_index, candidate_count = _native_and_quality_selected_s7(
        detections, source_ids, quality_logits, max_candidates)
    base = dict(
        loss_s7_selective_quality=zero,
        loss_s7_selective_classification=zero,
        loss_s7_selective_retention=zero,
        loss_s7_selective_gain=zero,
        loss_s7_selective_prior=zero,
        s7_selective_candidate_count=candidate_count,
        s7_selective_positive_count=0,
        s7_selective_retention_pair_count=0,
        s7_selective_gain_pair_count=0,
        s7_selective_native_top1_correct=0,
        s7_selective_mean_uncertainty=0.0,
        s7_selective_mean_advantage_target=0.0,
        s7_small_temporal_history_ready=int(state.ready),
        _s7_small_temporal_selected_index=(
            None if native_index is None else int(native_index)))
    if native_index is None or s7_index is None or not state.ready:
        return base

    features = ranker_head.pair_features(
        embeddings[native_index], detections[native_index],
        quality_logits[native_index], embeddings[s7_index],
        detections[s7_index], quality_logits[s7_index], state)
    advantage, uncertainty = ranker_head(features)
    target_advantage = (
        gt_overlap[s7_index] - gt_overlap[native_index]).detach().clamp(-1, 1)
    variance = uncertainty.square().clamp_min(1e-6)
    quality_nll = (
        0.5 * (advantage[0] - target_advantage).square() / variance[0]
        + torch.log(uncertainty[0]))
    promote = bool(
        (target_advantage >= float(advantage_gap)).detach().item()
        and (gt_overlap[s7_index] >= float(riou_threshold)).detach().item())
    classification = torch.nn.functional.binary_cross_entropy_with_logits(
        advantage, advantage.new_tensor([float(promote)]))
    lower_bound = advantage[0] - float(
        uncertainty_multiplier) * uncertainty[0]
    selected_index = native_index
    if bool((lower_bound >= float(promotion_margin)).detach().item()):
        selected_index = s7_index
    native_correct = bool(
        (gt_overlap[native_index] >= float(riou_threshold)).detach().item())
    risky = bool(
        (gt_overlap[s7_index] < float(riou_threshold)).detach().item())
    retention = (
        torch.relu(lower_bound - float(promotion_margin))
        if native_correct and risky else zero)
    gain = (
        torch.relu(float(promotion_margin) - lower_bound)
        if (not native_correct and promote) else zero)
    base.update(
        loss_s7_selective_quality=quality_nll * float(quality_weight),
        loss_s7_selective_classification=(
            classification * float(classification_weight)),
        loss_s7_selective_retention=retention * float(retention_weight),
        loss_s7_selective_gain=gain * float(gain_weight),
        loss_s7_selective_prior=(
            advantage.square().mean() * float(prior_weight)),
        s7_selective_positive_count=int(promote),
        s7_selective_retention_pair_count=int(native_correct and risky),
        s7_selective_gain_pair_count=int(not native_correct and promote),
        s7_selective_native_top1_correct=int(native_correct),
        s7_selective_mean_uncertainty=float(uncertainty.detach().item()),
        s7_selective_mean_advantage_target=float(target_advantage.item()),
        _s7_small_temporal_selected_index=int(selected_index))
    return base


class CausalSmallTemporalRanker:
    """Native-protected two-frame inference with explicit abstention."""

    def __init__(self, ranker_head: S7SmallTemporalRankerHead,
                 max_candidates: int = 20,
                 uncertainty_multiplier: float = 1.0,
                 promotion_margin: float = 0.10):
        if (int(max_candidates) <= 0
                or float(uncertainty_multiplier) <= 0.0
                or float(promotion_margin) < 0.0):
            raise ValueError('Small temporal inference settings are invalid')
        self.ranker_head = ranker_head
        self.max_candidates = int(max_candidates)
        self.uncertainty_multiplier = float(uncertainty_multiplier)
        self.promotion_margin = float(promotion_margin)
        self.state = TwoFrameMotionState()

    def select(self, detections: torch.Tensor, embeddings: torch.Tensor,
               source_ids: torch.Tensor, quality_logits: torch.Tensor,
               seq: str, frame: int,
               valid_mask: Optional[torch.Tensor] = None) -> Dict:
        original_order = torch.arange(
            detections.shape[0], device=detections.device)
        continuous = self.state.prepare(seq, frame)
        native_index, s7_index, candidate_count = (
            _native_and_quality_selected_s7(
                detections, source_ids, quality_logits,
                self.max_candidates, valid_mask=valid_mask))
        if native_index is None:
            self.state.reset()
            return dict(
                order=original_order, selected_index=None, native_index=None,
                promoted=False, override=False, reset=not continuous,
                reason='native_missing', best_lower_bound=None,
                best_advantage=None, best_uncertainty=None,
                candidate_index=(None if s7_index is None else int(s7_index)),
                candidate_count=candidate_count, history_ready=False)

        selected = native_index
        reason = 'native_fallback_history_warmup'
        advantage_value = None
        uncertainty_value = None
        lower_bound_value = None
        history_ready = self.state.ready
        if history_ready and s7_index is not None:
            features = self.ranker_head.pair_features(
                embeddings[native_index], detections[native_index],
                quality_logits[native_index], embeddings[s7_index],
                detections[s7_index], quality_logits[s7_index], self.state)
            advantage, uncertainty = self.ranker_head(features)
            lower_bound = advantage[0] - (
                self.uncertainty_multiplier * uncertainty[0])
            advantage_value = float(advantage[0].detach().item())
            uncertainty_value = float(uncertainty[0].detach().item())
            lower_bound_value = float(lower_bound.detach().item())
            if bool((lower_bound >= self.promotion_margin).detach().item()):
                selected = s7_index
                reason = 's7_promoted_confident_two_frame_advantage'
            else:
                reason = 'native_fallback_uncertain_two_frame_advantage'
        elif history_ready:
            reason = 'native_fallback_no_s7_candidate'

        self.state.update(
            detections[selected, :5], embeddings[selected], seq, frame)
        remaining = original_order[original_order != selected]
        order = torch.cat((selected.reshape(1), remaining), 0)
        selected_value = int(selected.detach().item())
        native_value = int(native_index.detach().item())
        candidate_value = (None if s7_index is None
                           else int(s7_index.detach().item()))
        promoted = bool(selected_value != native_value)
        return dict(
            order=order, selected_index=selected_value,
            native_index=native_value, promoted=promoted,
            override=promoted, reset=not continuous, reason=reason,
            best_lower_bound=lower_bound_value,
            best_advantage=advantage_value,
            best_uncertainty=uncertainty_value,
            candidate_index=candidate_value,
            candidate_count=candidate_count,
            history_ready=bool(history_ready),
            selected_source=('supplement_s7' if promoted else 'native_s14'))


def native_protected_selective_promotion(
        promotion_head: S7SelectivePromotionHead,
        embeddings: torch.Tensor, detections: torch.Tensor,
        source_ids: torch.Tensor, quality_logits: torch.Tensor,
        max_candidates: int = 100, uncertainty_multiplier: float = 1.0,
        promotion_margin: float = 0.10) -> Dict:
    """Return a native-first order unless an S7 lower bound is sufficient."""
    if (embeddings.ndim != 2 or detections.ndim != 2
            or detections.shape[1] != 6
            or source_ids.shape != (detections.shape[0],)
            or quality_logits.shape != (detections.shape[0],)
            or embeddings.shape[0] != detections.shape[0]):
        raise ValueError('Selective promotion candidate pool is misaligned')
    if (int(max_candidates) <= 0 or float(uncertainty_multiplier) <= 0.0
            or float(promotion_margin) < 0.0):
        raise ValueError('Selective promotion inference settings are invalid')
    count = int(detections.shape[0])
    original_order = torch.arange(count, device=detections.device)
    native = torch.nonzero(source_ids == 0, as_tuple=False).flatten()
    if count == 0:
        return dict(
            order=original_order, selected_index=None,
            native_index=None, promoted=False, reason='empty_pool',
            best_lower_bound=None, candidate_count=0)
    if native.numel() == 0:
        return dict(
            # No native candidate exists, so there is neither a valid native
            # fallback nor a safe native-protected promotion decision.  Do
            # not fabricate index 0: callers must be able to distinguish this
            # state from selecting the first candidate in the pool.
            order=original_order, selected_index=None,
            native_index=None, promoted=False, reason='native_missing',
            best_lower_bound=None, candidate_count=0)
    native_index = native[torch.argmax(detections[native, 5])]
    # The limit belongs to the S7 lane, not to the globally merged pool.
    # Native detections may precede S7 detections in the calibrated global
    # order; using a global prefix would therefore discard valid S7 lane
    # candidates before the native-protected selector sees them.
    s7_ranked = torch.nonzero(source_ids == 1, as_tuple=False).flatten()
    s7 = s7_ranked[:min(int(max_candidates), int(s7_ranked.numel()))]
    selected = native_index
    promoted = False
    reason = 'native_fallback_no_s7_candidate'
    best_lower_bound = None
    best_uncertainty = None
    best_advantage = None
    if s7.numel():
        advantage, uncertainty = promotion_head(
            embeddings[native_index], detections[native_index],
            quality_logits[native_index], embeddings[s7], detections[s7],
            quality_logits[s7])
        lower_bound = advantage - float(
            uncertainty_multiplier) * uncertainty
        best_position = torch.argmax(lower_bound)
        best_lower_bound_tensor = lower_bound[best_position]
        best_lower_bound = float(best_lower_bound_tensor.detach().item())
        best_uncertainty = float(uncertainty[best_position].detach().item())
        best_advantage = float(advantage[best_position].detach().item())
        if best_lower_bound >= float(promotion_margin):
            selected = s7[best_position]
            promoted = True
            reason = 's7_promoted_confident_advantage'
        else:
            reason = 'native_fallback_uncertain_advantage'
    remaining = original_order[original_order != selected]
    order = torch.cat((selected.reshape(1), remaining), dim=0)
    return dict(
        order=order, selected_index=int(selected.detach().item()),
        native_index=int(native_index.detach().item()), promoted=promoted,
        reason=reason, best_lower_bound=best_lower_bound,
        best_advantage=best_advantage, best_uncertainty=best_uncertainty,
        candidate_count=int(s7.numel()))


def selective_promotion_losses(
        promotion_head: S7SelectivePromotionHead,
        embeddings: torch.Tensor, detections: torch.Tensor,
        source_ids: torch.Tensor, quality_logits: torch.Tensor,
        gt_overlap: torch.Tensor, riou_threshold: float,
        advantage_gap: float = 0.10, promotion_margin: float = 0.10,
        uncertainty_multiplier: float = 1.0,
        quality_weight: float = 1.0, classification_weight: float = 1.0,
        retention_weight: float = 2.0, gain_weight: float = 1.0,
        prior_weight: float = 0.01, max_candidates: int = 100) -> Dict:
    """Train source-only pair advantage, uncertainty and native abstention."""
    positive = (
        advantage_gap, uncertainty_multiplier, quality_weight,
        classification_weight, retention_weight, gain_weight, prior_weight,
        max_candidates)
    if any(float(value) <= 0.0 for value in positive):
        raise ValueError('Selective promotion loss settings must be positive')
    if float(promotion_margin) < 0.0:
        raise ValueError('Selective promotion margin must be non-negative')
    if (gt_overlap.ndim != 1 or gt_overlap.shape != source_ids.shape
            or gt_overlap.shape[0] != detections.shape[0]
            or quality_logits.shape != gt_overlap.shape):
        raise ValueError('Selective promotion training inputs are misaligned')

    zero = promotion_head.advantage_output.bias.sum() * 0.0
    native = torch.nonzero(source_ids == 0, as_tuple=False).flatten()
    # Keep training aligned with inference: native top-1 plus the S7 lane's
    # own top-K candidates.  Filtering first preserves the lane ranking even
    # when many native detections precede the S7 lane in the merged pool.
    s7_ranked = torch.nonzero(source_ids == 1, as_tuple=False).flatten()
    s7 = s7_ranked[:min(int(max_candidates), int(s7_ranked.numel()))]
    if native.numel() == 0 or s7.numel() == 0:
        return dict(
            loss_s7_selective_quality=zero,
            loss_s7_selective_classification=zero,
            loss_s7_selective_retention=zero,
            loss_s7_selective_gain=zero,
            loss_s7_selective_prior=zero,
            s7_selective_candidate_count=int(s7.numel()),
            s7_selective_positive_count=0,
            s7_selective_retention_pair_count=0,
            s7_selective_gain_pair_count=0,
            s7_selective_native_top1_correct=0,
            s7_selective_mean_uncertainty=0.0,
            s7_selective_mean_advantage_target=0.0)

    native_index = native[torch.argmax(detections[native, 5].detach())]
    advantage, uncertainty = promotion_head(
        embeddings[native_index], detections[native_index],
        quality_logits[native_index], embeddings[s7], detections[s7],
        quality_logits[s7])
    target_advantage = (
        gt_overlap[s7] - gt_overlap[native_index]).detach().clamp(-1.0, 1.0)
    variance = uncertainty.square().clamp_min(1e-6)
    quality_nll = (
        0.5 * (advantage - target_advantage).square() / variance
        + torch.log(uncertainty)).mean()
    promote_target = (
        (target_advantage >= float(advantage_gap))
        & (gt_overlap[s7] >= float(riou_threshold))).float()
    positive_count = int(promote_target.sum().item())
    negative_count = int(promote_target.numel() - positive_count)
    positive_scale = (
        min(8.0, float(negative_count) / float(max(1, positive_count)))
        if positive_count else 1.0)
    classification = torch.nn.functional.binary_cross_entropy_with_logits(
        advantage / max(float(advantage_gap), 1e-4), promote_target,
        pos_weight=advantage.new_tensor(positive_scale))
    lower_bound = advantage - float(
        uncertainty_multiplier) * uncertainty
    native_correct = bool(
        gt_overlap[native_index] >= float(riou_threshold))
    risky = gt_overlap[s7] < float(riou_threshold)
    retention = zero
    retention_pair_count = int(risky.sum().item()) if native_correct else 0
    if native_correct and bool(risky.any().item()):
        retention = torch.relu(
            lower_bound[risky].max() - float(promotion_margin))
    gain = zero
    gain_pair_count = 0
    if not native_correct and positive_count:
        eligible = torch.nonzero(
            promote_target > 0.5, as_tuple=False).flatten()
        best = eligible[torch.argmax(target_advantage[eligible])]
        gain = torch.relu(float(promotion_margin) - lower_bound[best])
        gain_pair_count = 1
    prior = advantage.square().mean()
    return dict(
        loss_s7_selective_quality=quality_nll * float(quality_weight),
        loss_s7_selective_classification=(
            classification * float(classification_weight)),
        loss_s7_selective_retention=retention * float(retention_weight),
        loss_s7_selective_gain=gain * float(gain_weight),
        loss_s7_selective_prior=prior * float(prior_weight),
        s7_selective_candidate_count=int(s7.numel()),
        s7_selective_positive_count=positive_count,
        s7_selective_retention_pair_count=retention_pair_count,
        s7_selective_gain_pair_count=gain_pair_count,
        s7_selective_native_top1_correct=int(native_correct),
        s7_selective_mean_uncertainty=float(
            uncertainty.detach().mean().item()),
        s7_selective_mean_advantage_target=float(
            target_advantage.mean().item()))


def candidate_quality_relative_ranking_loss(
        quality_logits: torch.Tensor, gt_overlap: torch.Tensor,
        margin: float = 0.25, min_gap: float = 0.10,
        max_pairs: int = 128) -> Dict:
    """Build deterministic same-frame source-only relative-quality pairs."""
    if quality_logits.ndim != 1 or gt_overlap.ndim != 1:
        raise ValueError('Relative quality inputs must be vectors')
    if quality_logits.shape != gt_overlap.shape:
        raise ValueError('Relative quality logits and targets are misaligned')
    if float(margin) <= 0.0 or float(min_gap) <= 0.0 or int(max_pairs) <= 0:
        raise ValueError(
            'Relative quality margin, gap and pair count must be positive')
    zero = quality_logits.sum() * 0.0
    if quality_logits.numel() < 2:
        return dict(
            loss_s7_candidate_quality_relative=zero,
            s7_candidate_quality_relative_pair_count=0,
            s7_candidate_quality_relative_active_count=0,
            s7_candidate_quality_relative_accuracy=0.0,
            s7_candidate_quality_relative_mean_gap=0.0)

    target = gt_overlap.detach().float().clamp(0.0, 1.0)
    order = torch.argsort(target, descending=True).detach().cpu().tolist()
    target_cpu = target.detach().cpu().tolist()
    positive_indices = []
    negative_indices = []
    # Pair each candidate with the nearest lower-quality candidate separated
    # by the configured target gap.  This gives broad rank supervision while
    # keeping the per-frame loss bounded and deterministic.
    for position, positive in enumerate(order):
        for negative in order[position + 1:]:
            if float(target_cpu[positive] - target_cpu[negative]) >= float(
                    min_gap):
                positive_indices.append(positive)
                negative_indices.append(negative)
                break
        if len(positive_indices) >= int(max_pairs):
            break
    if not positive_indices:
        return dict(
            loss_s7_candidate_quality_relative=zero,
            s7_candidate_quality_relative_pair_count=0,
            s7_candidate_quality_relative_active_count=0,
            s7_candidate_quality_relative_accuracy=0.0,
            s7_candidate_quality_relative_mean_gap=0.0)

    positive = quality_logits.new_tensor(positive_indices, dtype=torch.long)
    negative = quality_logits.new_tensor(negative_indices, dtype=torch.long)
    gaps = target[positive] - target[negative]
    logit_gaps = quality_logits[positive] - quality_logits[negative]
    per_pair = torch.relu(float(margin) - logit_gaps)
    return dict(
        loss_s7_candidate_quality_relative=per_pair.mean(),
        s7_candidate_quality_relative_pair_count=int(per_pair.numel()),
        s7_candidate_quality_relative_active_count=int(
            (per_pair > 0.0).sum().item()),
        s7_candidate_quality_relative_accuracy=float(
            (logit_gaps > 0.0).float().mean().item()),
        s7_candidate_quality_relative_mean_gap=float(gaps.mean().item()))


def candidate_quality_losses(
        quality_head: S7CandidateQualityHead, embedding: torch.Tensor,
        detections: torch.Tensor, source_ids: torch.Tensor,
        gt_overlap: torch.Tensor, riou_threshold: float,
        relative_margin: Optional[float] = None,
        relative_min_gap: float = 0.10,
        relative_max_pairs: int = 128) -> Dict:
    """Dense source-only continuous max-RIoU and optional relative ranking."""
    if gt_overlap.ndim != 1 or gt_overlap.shape[0] != detections.shape[0]:
        raise ValueError('Candidate quality targets are misaligned')
    logits = quality_head(embedding, detections, source_ids)
    if logits.numel() == 0:
        zero = quality_head.output.bias.sum() * 0.0
        result = dict(
            loss_s7_candidate_quality=zero,
            s7_candidate_quality_count=0,
            s7_candidate_quality_usable_count=0,
            s7_candidate_quality_mean_target=0.0,
            s7_candidate_quality_mean_prediction=0.0)
        if relative_margin is not None:
            result.update(candidate_quality_relative_ranking_loss(
                logits, gt_overlap, relative_margin, relative_min_gap,
                relative_max_pairs))
        return result
    target = gt_overlap.detach().float().clamp(0.0, 1.0)
    prediction = torch.sigmoid(logits)
    per_candidate = torch.nn.functional.smooth_l1_loss(
        prediction, target, reduction='none')
    # Give high-RIoU candidates more influence while retaining every
    # candidate as supervision; this is not a sparse gain-pair miner.
    weights = (1.0 + 3.0 * target).detach()
    loss = (per_candidate * weights).sum() / weights.sum().clamp_min(1e-6)
    result = dict(
        loss_s7_candidate_quality=loss,
        s7_candidate_quality_count=int(target.numel()),
        s7_candidate_quality_usable_count=int(
            (target >= float(riou_threshold)).sum().item()),
        s7_candidate_quality_mean_target=float(target.mean().item()),
        s7_candidate_quality_mean_prediction=float(prediction.mean().item()))
    if relative_margin is not None:
        result.update(candidate_quality_relative_ranking_loss(
            logits, gt_overlap, relative_margin, relative_min_gap,
            relative_max_pairs))
    return result


def candidate_student_losses(
        student_head: S7CandidateQualityHead,
        teacher_head: S7CandidateQualityHead,
        embedding: torch.Tensor, detections: torch.Tensor,
        source_ids: torch.Tensor, gt_overlap: torch.Tensor,
        riou_threshold: float, quality_weight: float = 1.0,
        relative_weight: float = 0.5, relative_margin: float = 0.25,
        relative_min_gap: float = 0.10, relative_max_pairs: int = 128,
        distillation_weight: float = 1.0,
        distillation_temperature: float = 1.0,
        supervised_frame_weight: float = 1.0) -> Dict:
    """Train a source-only student while retaining the fixed quality teacher.

    Source GT supplies continuous max-RIoU and same-frame relative-order
    supervision. Bernoulli distillation keeps the student close to the
    already source-gated phase-2 teacher. ``supervised_frame_weight`` may
    emphasize a geometrically small source frame, but is never an inference
    input.
    """
    positive = (
        quality_weight, relative_weight, distillation_weight,
        distillation_temperature, supervised_frame_weight)
    if any(float(value) <= 0.0 for value in positive):
        raise ValueError(
            'Student loss weights and temperature must be positive')
    with torch.no_grad():
        teacher_logits = teacher_head(
            embedding, detections, source_ids).detach()
    losses = candidate_quality_losses(
        student_head, embedding, detections, source_ids, gt_overlap,
        riou_threshold=riou_threshold,
        relative_margin=float(relative_margin),
        relative_min_gap=float(relative_min_gap),
        relative_max_pairs=int(relative_max_pairs))
    student_logits = student_head(embedding, detections, source_ids)
    if student_logits.numel() == 0:
        distillation = student_head.output.bias.sum() * 0.0
        mean_abs_error = 0.0
        top1_agreement = 1.0
    else:
        temperature = float(distillation_temperature)
        eps = 1e-6
        teacher_probability = torch.sigmoid(
            teacher_logits / temperature).clamp(eps, 1.0 - eps)
        student_probability = torch.sigmoid(
            student_logits / temperature).clamp(eps, 1.0 - eps)
        distillation = (
            teacher_probability * (
                torch.log(teacher_probability)
                - torch.log(student_probability))
            + (1.0 - teacher_probability) * (
                torch.log1p(-teacher_probability)
                - torch.log1p(-student_probability))).mean()
        distillation = distillation * (temperature ** 2)
        mean_abs_error = float(
            (student_logits.detach() - teacher_logits).abs().mean().item())
        top1_agreement = float(
            int(torch.argmax(student_logits.detach()).item())
            == int(torch.argmax(teacher_logits).item()))
    frame_weight = float(supervised_frame_weight)
    losses['loss_s7_student_quality'] = (
        losses.pop('loss_s7_candidate_quality')
        * float(quality_weight) * frame_weight)
    losses['loss_s7_student_relative'] = (
        losses.pop('loss_s7_candidate_quality_relative')
        * float(relative_weight) * frame_weight)
    losses['loss_s7_student_distillation'] = (
        distillation * float(distillation_weight))
    losses['s7_student_teacher_mean_abs_logit_error'] = mean_abs_error
    losses['s7_student_teacher_top1_agreement'] = top1_agreement
    losses['s7_student_supervised_frame_weight'] = frame_weight
    return losses


def static_candidate_rank_losses(
        quality_head: S7CandidateQualityHead, embedding: torch.Tensor,
        detections: torch.Tensor, source_ids: torch.Tensor,
        gt_overlap: torch.Tensor, riou_threshold: float,
        quality_weight: float = 1.0, relative_weight: float = 0.5,
        relative_margin: float = 0.25, relative_min_gap: float = 0.10,
        relative_max_pairs: int = 128, score_weight: float = 1.0,
        rank_margin: float = 0.25, retention_weight: float = 2.0,
        gain_weight: float = 1.0, prior_weight: float = 0.01) -> Dict:
    """Train a non-temporal source-only ranker on the merged candidate pool.

    The base detector score remains the reference ordering.  The learned
    quality logit is only a bounded additive residual in score-logit space,
    so the head starts as an exact no-op and is trained with two explicit
    protections: keep a correct native top-1 above hard negatives, and lift a
    usable S7 candidate when native top-1 is wrong.  All targets come from
    the current source frame; no sequence identity, target frame, or future
    state is used.
    """
    positive = (quality_weight, relative_weight, score_weight, rank_margin,
                retention_weight, gain_weight, prior_weight)
    if any(float(value) <= 0.0 for value in positive):
        raise ValueError('Static ranker weights and margin must be positive')
    if gt_overlap.ndim != 1 or source_ids.ndim != 1:
        raise ValueError('Static ranker overlaps and source ids must be vectors')
    if (gt_overlap.shape != source_ids.shape
            or gt_overlap.shape[0] != detections.shape[0]):
        raise ValueError('Static ranker inputs have mismatched counts')

    quality = candidate_quality_losses(
        quality_head, embedding, detections, source_ids, gt_overlap,
        riou_threshold=riou_threshold, relative_margin=float(relative_margin),
        relative_min_gap=float(relative_min_gap),
        relative_max_pairs=int(relative_max_pairs))
    logits = quality_head(embedding, detections, source_ids)
    zero = logits.sum() * 0.0
    if logits.numel() == 0:
        return dict(
            loss_s7_static_quality=zero,
            loss_s7_static_relative=zero,
            loss_s7_static_retention=zero,
            loss_s7_static_gain=zero,
            loss_s7_static_prior=zero,
            s7_static_retention_pair_count=0,
            s7_static_gain_pair_count=0,
            s7_static_native_top1_correct=0,
            s7_static_usable_candidate_count=0,
            s7_static_candidate_count=0,
            s7_static_hard_negative_count=0)

    base_scores = detections[:, 5].clamp(1e-6, 1.0 - 1e-6)
    base_logits = torch.log(base_scores) - torch.log1p(-base_scores)
    fused = base_logits + float(score_weight) * logits
    native = torch.nonzero(source_ids == 0, as_tuple=False).flatten()
    usable = torch.nonzero(
        gt_overlap >= float(riou_threshold), as_tuple=False).flatten()
    wrong = torch.nonzero(
        gt_overlap < float(riou_threshold), as_tuple=False).flatten()
    native_top = (native[torch.argmax(base_scores[native].detach())]
                  if native.numel() else None)
    native_top_correct = bool(
        native_top is not None
        and gt_overlap[native_top] >= float(riou_threshold))
    best_usable = (usable[torch.argmax(gt_overlap[usable].detach())]
                   if usable.numel() else None)
    retention = zero
    gain = zero
    retention_pair_count = 0
    gain_pair_count = 0
    hard_negative_count = 0
    if native_top_correct and wrong.numel():
        competitor = wrong[torch.argmax(fused[wrong].detach())]
        retention = torch.relu(
            float(rank_margin) + fused[competitor] - fused[native_top])
        retention_pair_count = 1
        hard_negative_count = 1
    elif native_top is not None and best_usable is not None:
        competitors = wrong[wrong != best_usable]
        if competitors.numel():
            competitor = competitors[torch.argmax(fused[competitors].detach())]
            gain = torch.relu(
                float(rank_margin) + fused[competitor] - fused[best_usable])
            hard_negative_count = 1
        else:
            gain = torch.relu(
                float(rank_margin) + fused[native_top] - fused[best_usable])
        gain_pair_count = 1

    relative = quality.get(
        'loss_s7_candidate_quality_relative', zero)
    return dict(
        loss_s7_static_quality=(quality['loss_s7_candidate_quality']
                                * float(quality_weight)),
        loss_s7_static_relative=relative * float(relative_weight),
        loss_s7_static_retention=retention * float(retention_weight),
        loss_s7_static_gain=gain * float(gain_weight),
        loss_s7_static_prior=logits.square().mean() * float(prior_weight),
        s7_static_retention_pair_count=retention_pair_count,
        s7_static_gain_pair_count=gain_pair_count,
        s7_static_native_top1_correct=int(native_top_correct),
        s7_static_usable_candidate_count=int(usable.numel()),
        s7_static_candidate_count=int(logits.numel()),
        s7_static_hard_negative_count=hard_negative_count,
        s7_static_quality_mean_target=quality.get(
            's7_candidate_quality_mean_target', 0.0),
        s7_static_quality_mean_prediction=quality.get(
            's7_candidate_quality_mean_prediction', 0.0),
        s7_static_relative_pair_count=quality.get(
            's7_candidate_quality_relative_pair_count', 0))


def highres_candidate_rank_losses(
        quality_head: S7HighResCandidateQualityHead,
        embedding: torch.Tensor, highres_embedding: torch.Tensor,
        detections: torch.Tensor, source_ids: torch.Tensor,
        gt_overlap: torch.Tensor, riou_threshold: float,
        quality_weight: float = 1.0, relative_weight: float = 0.5,
        relative_margin: float = 0.25, relative_min_gap: float = 0.10,
        relative_max_pairs: int = 128, score_weight: float = 1.0,
        rank_margin: float = 0.25, retention_weight: float = 2.0,
        gain_weight: float = 1.0, prior_weight: float = 0.01) -> Dict:
    """Train the first high-resolution ROI quality/ranking experiment.

    This is same-frame, source-only supervision.  The detector and proposal
    lanes are frozen; the learned residual is zero-initialized and the
    native candidate remains protected at inference unless a high-resolution
    S7 candidate clears the explicit margin.
    """
    positive = (quality_weight, relative_weight, score_weight, rank_margin,
                retention_weight, gain_weight, prior_weight)
    if any(float(value) <= 0.0 for value in positive):
        raise ValueError('High-resolution ranker settings must be positive')
    if (gt_overlap.ndim != 1 or source_ids.ndim != 1
            or gt_overlap.shape != source_ids.shape
            or gt_overlap.shape[0] != detections.shape[0]):
        raise ValueError('High-resolution ranker targets are misaligned')

    logits = quality_head(
        embedding, highres_embedding, detections, source_ids)
    zero = quality_head.output.bias.sum() * 0.0
    if logits.numel() == 0:
        return dict(
            loss_s7_highres_quality=zero,
            loss_s7_highres_relative=zero,
            loss_s7_highres_retention=zero,
            loss_s7_highres_gain=zero,
            loss_s7_highres_prior=zero,
            s7_highres_retention_pair_count=0,
            s7_highres_gain_pair_count=0,
            s7_highres_native_top1_correct=0,
            s7_highres_usable_candidate_count=0,
            s7_highres_candidate_count=0)

    target = gt_overlap.detach().float().clamp(0.0, 1.0)
    prediction = torch.sigmoid(logits)
    weights = (1.0 + 3.0 * target).detach()
    quality = (torch.nn.functional.smooth_l1_loss(
        prediction, target, reduction='none') * weights).sum() / (
            weights.sum().clamp_min(1e-6))
    relative_result = candidate_quality_relative_ranking_loss(
        logits, target, margin=float(relative_margin),
        min_gap=float(relative_min_gap), max_pairs=int(relative_max_pairs))

    scores = detections[:, 5].clamp(1e-6, 1.0 - 1e-6)
    score_logits = torch.log(scores) - torch.log1p(-scores)
    fused = score_logits + float(score_weight) * logits
    native = torch.nonzero(source_ids == 0, as_tuple=False).flatten()
    s7 = torch.nonzero(source_ids == 1, as_tuple=False).flatten()
    usable = torch.nonzero(target >= float(riou_threshold),
                           as_tuple=False).flatten()
    native_top = (native[torch.argmax(scores[native].detach())]
                  if native.numel() else None)
    native_correct = bool(native_top is not None and target[native_top]
                           >= float(riou_threshold))
    best_usable = (usable[torch.argmax(target[usable].detach())]
                   if usable.numel() else None)
    retention = zero
    gain = zero
    retention_count = 0
    gain_count = 0
    if native_correct:
        wrong = torch.nonzero(target < float(riou_threshold),
                              as_tuple=False).flatten()
        if wrong.numel():
            competitor = wrong[torch.argmax(fused[wrong].detach())]
            retention = torch.relu(
                float(rank_margin) + fused[competitor] - fused[native_top])
            retention_count = 1
    elif native_top is not None and best_usable is not None:
        competitors = torch.nonzero(target < float(riou_threshold),
                                    as_tuple=False).flatten()
        competitors = competitors[competitors != best_usable]
        if competitors.numel():
            competitor = competitors[torch.argmax(fused[competitors].detach())]
            gain = torch.relu(
                float(rank_margin) + fused[competitor] - fused[best_usable])
        else:
            gain = torch.relu(
                float(rank_margin) + fused[native_top] - fused[best_usable])
        gain_count = 1

    return dict(
        loss_s7_highres_quality=quality * float(quality_weight),
        loss_s7_highres_relative=(
            relative_result['loss_s7_candidate_quality_relative']
            * float(relative_weight)),
        loss_s7_highres_retention=retention * float(retention_weight),
        loss_s7_highres_gain=gain * float(gain_weight),
        loss_s7_highres_prior=logits.square().mean() * float(prior_weight),
        s7_highres_retention_pair_count=retention_count,
        s7_highres_gain_pair_count=gain_count,
        s7_highres_native_top1_correct=int(native_correct),
        s7_highres_usable_candidate_count=int(usable.numel()),
        s7_highres_candidate_count=int(logits.numel()),
        s7_highres_relative_pair_count=int(relative_result.get(
            's7_candidate_quality_relative_pair_count', 0)),
        s7_highres_quality_mean_target=float(target.mean().item()),
        s7_highres_quality_mean_prediction=float(prediction.mean().item()))


def unified_highres_candidate_rank_losses(
        quality_head: S7HighResCandidateQualityHead,
        embedding: torch.Tensor, highres_embedding: torch.Tensor,
        detections: torch.Tensor, source_ids: torch.Tensor,
        gt_overlap: torch.Tensor, riou_threshold: float,
        quality_weight: float = 1.0, relative_weight: float = 0.5,
        relative_margin: float = 0.25, relative_min_gap: float = 0.10,
        relative_max_pairs: int = 128, score_weight: float = 1.0,
        rank_margin: float = 0.25, retention_weight: float = 2.0,
        gain_weight: float = 1.0, prior_weight: float = 0.01,
        hard_pair_count: int = 8) -> Dict:
    """Train one source-only ranker over the unified native/S7 pool.

    The original high-resolution stage protects the native winner and mines
    one lane-level gain or retention pair.  This stage keeps the same frozen
    detector and ROI evidence, but mines hard pairs from the *whole* active
    pool.  Each source-GT quality level is paired with the highest current
    fused-score candidate at a lower quality level.  Native retention and S7
    gain pairs remain explicit, so the broader listwise signal cannot turn
    the ranker into an unconditional S7 promoter.
    """
    positive = (quality_weight, relative_weight, score_weight, rank_margin,
                retention_weight, gain_weight, prior_weight,
                hard_pair_count)
    if any(float(value) <= 0.0 for value in positive):
        raise ValueError('Unified high-resolution ranker settings must be positive')
    if (gt_overlap.ndim != 1 or source_ids.ndim != 1
            or gt_overlap.shape != source_ids.shape
            or gt_overlap.shape[0] != detections.shape[0]):
        raise ValueError('Unified high-resolution targets are misaligned')

    logits = quality_head(
        embedding, highres_embedding, detections, source_ids)
    zero = quality_head.output.bias.sum() * 0.0
    if logits.numel() == 0:
        return dict(
            loss_s7_highres_quality=zero,
            loss_s7_highres_relative=zero,
            loss_s7_highres_retention=zero,
            loss_s7_highres_gain=zero,
            loss_s7_highres_unified=zero,
            loss_s7_highres_prior=zero,
            s7_highres_retention_pair_count=0,
            s7_highres_gain_pair_count=0,
            s7_highres_unified_pair_count=0,
            s7_highres_unified_active_count=0,
            s7_highres_native_top1_correct=0,
            s7_highres_usable_candidate_count=0,
            s7_highres_candidate_count=0)

    target = gt_overlap.detach().float().clamp(0.0, 1.0)
    prediction = torch.sigmoid(logits)
    weights = (1.0 + 3.0 * target).detach()
    quality = (torch.nn.functional.smooth_l1_loss(
        prediction, target, reduction='none') * weights).sum() / (
            weights.sum().clamp_min(1e-6))
    relative_result = candidate_quality_relative_ranking_loss(
        logits, target, margin=float(relative_margin),
        min_gap=float(relative_min_gap), max_pairs=int(relative_max_pairs))

    scores = detections[:, 5].clamp(1e-6, 1.0 - 1e-6)
    score_logits = torch.log(scores) - torch.log1p(-scores)
    fused = score_logits + float(score_weight) * logits
    order = torch.argsort(target, descending=True).detach().cpu().tolist()
    target_cpu = target.detach().cpu().tolist()
    pairs = []

    def add_pair(positive_index, negative_index):
        pair = (int(positive_index), int(negative_index))
        if pair[0] != pair[1] and pair not in pairs:
            pairs.append(pair)

    # Hard-pair mine every distinct source-quality level against the current
    # fused-score leader among lower-quality candidates.  This directly
    # targets the observed rank-2/rank-3 failures instead of only comparing
    # the S7 lane winner with native top-1.
    for position, positive_index in enumerate(order):
        lower = [index for index in order[position + 1:]
                 if float(target_cpu[positive_index] - target_cpu[index])
                 >= float(relative_min_gap)]
        if not lower:
            continue
        negative = max(
            lower, key=lambda index: float(fused[index].detach().item()))
        add_pair(positive_index, negative)
        if len(pairs) >= int(hard_pair_count):
            break

    native = torch.nonzero(source_ids == 0, as_tuple=False).flatten()
    wrong = torch.nonzero(target < float(riou_threshold),
                          as_tuple=False).flatten()
    usable = torch.nonzero(target >= float(riou_threshold),
                           as_tuple=False).flatten()
    native_top = (native[torch.argmax(scores[native].detach())]
                  if native.numel() else None)
    native_correct = bool(native_top is not None and target[native_top]
                           >= float(riou_threshold))
    best_usable = (usable[torch.argmax(target[usable].detach())]
                   if usable.numel() else None)
    retention_count = 0
    gain_count = 0
    retention_pair = None
    gain_pair = None
    if native_correct and wrong.numel():
        competitor = wrong[torch.argmax(fused[wrong].detach())]
        retention_pair = (int(native_top.item()), int(competitor.item()))
        add_pair(native_top, competitor)
        retention_count = 1
    elif native_top is not None and best_usable is not None:
        competitors = wrong[wrong != best_usable]
        if competitors.numel():
            competitor = competitors[torch.argmax(fused[competitors].detach())]
            gain_pair = (int(best_usable.item()), int(competitor.item()))
            add_pair(best_usable, competitor)
        else:
            gain_pair = (int(best_usable.item()), int(native_top.item()))
            add_pair(best_usable, native_top)
        gain_count = 1

    if pairs:
        positive_indices = torch.tensor(
            [pair[0] for pair in pairs], dtype=torch.long,
            device=logits.device)
        negative_indices = torch.tensor(
            [pair[1] for pair in pairs], dtype=torch.long,
            device=logits.device)
        pair_gaps = fused[positive_indices] - fused[negative_indices]
        unified = torch.relu(
            float(rank_margin) - pair_gaps).mean()
        active = int((pair_gaps < float(rank_margin)).sum().item())

        def pair_hinge(pair):
            if pair is None:
                return zero
            positive_index, negative_index = pair
            return torch.relu(
                float(rank_margin) + fused[negative_index]
                - fused[positive_index])

        retention = pair_hinge(retention_pair)
        gain = pair_hinge(gain_pair)
    else:
        unified = zero
        retention = zero
        gain = zero
        active = 0

    return dict(
        loss_s7_highres_quality=quality * float(quality_weight),
        loss_s7_highres_relative=(
            relative_result['loss_s7_candidate_quality_relative']
            * float(relative_weight)),
        loss_s7_highres_retention=retention * float(retention_weight),
        loss_s7_highres_gain=gain * float(gain_weight),
        loss_s7_highres_unified=unified,
        loss_s7_highres_prior=logits.square().mean() * float(prior_weight),
        s7_highres_retention_pair_count=retention_count,
        s7_highres_gain_pair_count=gain_count,
        s7_highres_unified_pair_count=len(pairs),
        s7_highres_unified_active_count=active,
        s7_highres_native_top1_correct=int(native_correct),
        s7_highres_usable_candidate_count=int(usable.numel()),
        s7_highres_candidate_count=int(logits.numel()),
        s7_highres_relative_pair_count=int(relative_result.get(
            's7_candidate_quality_relative_pair_count', 0)),
        s7_highres_quality_mean_target=float(target.mean().item()),
        s7_highres_quality_mean_prediction=float(prediction.mean().item()))


def native_protected_unified_highres_ranking_from_logits(
        quality_logits: torch.Tensor, detections: torch.Tensor,
        source_ids: torch.Tensor, max_candidates: int = 32,
        score_weight: float = 1.0, promotion_margin: float = 0.25) -> Dict:
    """Select from the complete active native/S7 pool with native abstention.

    Unlike a lane-specific promotion helper, this function names the
    inference contract explicitly: all active candidates share one fused
    score, while a S7 winner must clear the native score by the fixed margin.
    The logits are supplied by the caller so the same readout is used for
    logging, selection, and source/target audits.
    """
    if detections.ndim != 2 or detections.shape[1] != 6:
        raise ValueError('Unified high-resolution ranking expects [N, 6] detections')
    if (quality_logits.shape != (detections.shape[0],)
            or source_ids.shape != (detections.shape[0],)):
        raise ValueError('Unified high-resolution ranking inputs are misaligned')
    if int(max_candidates) <= 0 or float(score_weight) <= 0.0:
        raise ValueError('Unified high-resolution ranking settings must be positive')
    if float(promotion_margin) < 0.0:
        raise ValueError('Unified high-resolution promotion margin must be non-negative')
    original = torch.arange(detections.shape[0], device=detections.device)
    native = torch.nonzero(source_ids == 0, as_tuple=False).flatten()
    s7 = torch.nonzero(source_ids == 1, as_tuple=False).flatten()
    if not native.numel():
        return dict(order=original, selected_index=None, native_index=None,
                    promoted=False, reason='native_missing', candidate_count=0)
    native_index = native[torch.argmax(detections[native, 5])]
    s7 = s7[:min(int(max_candidates), int(s7.numel()))]
    if not s7.numel():
        index = int(native_index.item())
        return dict(order=original, selected_index=index,
                    native_index=index, promoted=False,
                    reason='native_fallback_no_s7_candidate', candidate_count=0)
    active = torch.cat((native_index.reshape(1), s7), dim=0)
    scores = detections[active, 5].clamp(1e-6, 1.0 - 1e-6)
    fused = torch.log(scores) - torch.log1p(-scores)
    fused = fused + float(score_weight) * quality_logits[active]
    if not bool((quality_logits[active].detach().abs() > 1e-7).any().item()):
        return dict(order=original, selected_index=int(native_index.item()),
                    native_index=int(native_index.item()), promoted=False,
                    reason='native_fallback_zero_residual',
                    candidate_count=int(s7.numel()))
    best_position = 1 + torch.argmax(fused[1:])
    best_index = active[best_position]
    best_margin = fused[best_position] - fused[0]
    promoted = bool(
        best_index != native_index
        and float(best_margin.detach().item()) >= float(promotion_margin))
    selected = best_index if promoted else native_index
    remaining = original[original != selected]
    order = torch.cat((selected.reshape(1), remaining), dim=0)
    return dict(
        order=order, selected_index=int(selected.item()),
        native_index=int(native_index.item()), promoted=promoted,
        reason=('s7_promoted_unified_quality' if promoted
                else 'native_fallback_unified_margin'),
        candidate_count=int(s7.numel()),
        best_quality=float(quality_logits[best_index].detach().item()),
        best_margin=float(best_margin.detach().item()))


def native_protected_highres_promotion_from_logits(
        quality_logits: torch.Tensor, detections: torch.Tensor,
        source_ids: torch.Tensor, max_candidates: int = 32,
        score_weight: float = 1.0,
        promotion_margin: float = 0.25) -> Dict:
    """Apply one native-protected margin without recomputing ROI features."""
    if detections.ndim != 2 or detections.shape[1] != 6:
        raise ValueError('High-resolution promotion expects [N, 6] detections')
    if (quality_logits.shape != (detections.shape[0],)
            or source_ids.shape != (detections.shape[0],)):
        raise ValueError('High-resolution promotion inputs are misaligned')
    if int(max_candidates) <= 0 or float(score_weight) <= 0.0:
        raise ValueError('High-resolution promotion settings must be positive')
    if float(promotion_margin) < 0.0:
        raise ValueError('High-resolution promotion margin must be non-negative')
    original = torch.arange(detections.shape[0], device=detections.device)
    native = torch.nonzero(source_ids == 0, as_tuple=False).flatten()
    s7_ranked = torch.nonzero(source_ids == 1, as_tuple=False).flatten()
    if not native.numel():
        return dict(order=original, selected_index=None, native_index=None,
                    promoted=False, reason='native_missing', candidate_count=0)
    native_index = native[torch.argmax(detections[native, 5])]
    s7 = s7_ranked[:min(int(max_candidates), int(s7_ranked.numel()))]
    if not s7.numel():
        return dict(order=original, selected_index=int(native_index.item()),
                    native_index=int(native_index.item()), promoted=False,
                    reason='native_fallback_no_s7_candidate', candidate_count=0)
    active = torch.cat((native_index.reshape(1), s7), dim=0)
    quality = quality_logits[active]
    scores = detections[active, 5].clamp(1e-6, 1.0 - 1e-6)
    fused = torch.log(scores) - torch.log1p(-scores) + float(score_weight) * quality
    # A zero-initialized head is an exact no-op, which makes epoch-0 a true
    # phase-2 reference rather than a re-sorted copy of the same candidates.
    if not bool((quality.detach().abs() > 1e-7).any().item()):
        return dict(order=original, selected_index=int(native_index.item()),
                    native_index=int(native_index.item()), promoted=False,
                    reason='native_fallback_zero_residual',
                    candidate_count=int(s7.numel()))
    best_position = 1 + torch.argmax(fused[1:])
    best_index = active[best_position]
    promoted = bool(
        best_index != native_index
        and float((fused[best_position] - fused[0]).detach().item())
        >= float(promotion_margin))
    selected = best_index if promoted else native_index
    remaining = original[original != selected]
    order = torch.cat((selected.reshape(1), remaining), dim=0)
    return dict(
        order=order, selected_index=int(selected.item()),
        native_index=int(native_index.item()), promoted=promoted,
        reason=('s7_promoted_highres_quality' if promoted
                else 'native_fallback_highres_margin'),
        candidate_count=int(s7.numel()),
        best_quality=float(quality[best_position].detach().item()),
        best_margin=float((fused[best_position] - fused[0]).detach().item()))


def native_protected_highres_promotion(
        quality_head: S7HighResCandidateQualityHead,
        embedding: torch.Tensor, highres_embedding: torch.Tensor,
        detections: torch.Tensor, source_ids: torch.Tensor,
        max_candidates: int = 32, score_weight: float = 1.0,
        promotion_margin: float = 0.25) -> Dict:
    """Reorder one frame while retaining native-first abstention semantics."""
    if (embedding.shape[0] != detections.shape[0]
            or highres_embedding.shape[0] != detections.shape[0]):
        raise ValueError('High-resolution promotion inputs are misaligned')
    quality_logits = quality_head(
        embedding, highres_embedding, detections, source_ids)
    return native_protected_highres_promotion_from_logits(
        quality_logits, detections, source_ids,
        max_candidates=max_candidates, score_weight=score_weight,
        promotion_margin=promotion_margin)


def _default_rotated_iou(current: torch.Tensor,
                         previous: torch.Tensor) -> torch.Tensor:
    from mmcv.ops import box_iou_rotated

    return box_iou_rotated(
        current.float(), previous.reshape(1, 5).float()).reshape(-1)


def build_temporal_cues(
        detections: torch.Tensor, embeddings: torch.Tensor,
        previous_box: torch.Tensor, previous_embedding: torch.Tensor,
        rotated_iou_fn: Optional[Callable] = None,
        candidate_quality: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        cue_count = (len(QUALITY_CUE_NAMES) if candidate_quality is not None
                     else len(CUE_NAMES))
        return detections.new_zeros((0, cue_count))
    if candidate_quality is not None and candidate_quality.shape != (
            detections.shape[0],):
        raise ValueError('Candidate quality logits are misaligned')

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

    base = torch.stack((
        score_logit.clamp(-12.0, 12.0),
        -normalized_center,
        rotated_iou,
        -scale_change.clamp(max=12.0),
        angle_similarity,
        appearance.clamp(-1.0, 1.0)), dim=1)
    if candidate_quality is None:
        return base
    return torch.cat((base, candidate_quality.to(
        device=base.device, dtype=base.dtype).clamp(-12.0, 12.0).reshape(
            -1, 1)), dim=1)


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
               valid_mask: Optional[torch.Tensor] = None,
               quality_logits: Optional[torch.Tensor] = None) -> Dict:
        if detections.ndim != 2 or detections.shape[1] != 6:
            raise ValueError('Temporal selector expects [N, 6] detections')
        if (embeddings.ndim != 2
                or embeddings.shape[0] != detections.shape[0]
                or source_ids.shape != (detections.shape[0],)):
            raise ValueError('Temporal selector candidate metadata is misaligned')
        uses_quality = 'candidate_quality_logit' in self.scorer.cue_names
        if uses_quality and (quality_logits is None
                             or quality_logits.shape != (detections.shape[0],)):
            raise ValueError(
                'Quality-aware temporal selector requires aligned logits')
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
                self.previous_embedding, candidate_quality=quality_logits)
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
                        self.pending_embedding,
                        candidate_quality=quality_logits)
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
