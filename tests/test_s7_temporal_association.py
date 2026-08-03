import pytest
import torch

from crane_project.utils import s7_temporal_association as temporal


def _unit_iou(current, previous):
    del previous
    return torch.ones(current.shape[0], device=current.device)


def _detections():
    return torch.tensor([
        [10.0, 10.0, 8.0, 4.0, 0.0, 0.40],
        [11.0, 10.0, 8.0, 4.0, 0.0, 0.90],
    ])


def test_temporal_cues_combine_score_motion_geometry_and_appearance():
    detections = _detections()
    embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    cues = temporal.build_temporal_cues(
        detections, embeddings, detections[0, :5], embeddings[0],
        rotated_iou_fn=_unit_iou)
    assert cues.shape == (2, 6)
    assert cues[1, 0] > cues[0, 0]
    assert cues[:, 2].tolist() == pytest.approx([1.0, 1.0])
    assert cues[:, 5].tolist() == pytest.approx([1.0, 1.0])
    assert cues[0, 1] == pytest.approx(0.0)
    assert cues[1, 1] < 0.0


def test_temporal_pair_loss_updates_only_relative_association_weights():
    scorer = temporal.S7TemporalAssociationScorer()
    cues = torch.tensor([
        [0.0, -0.1, 0.8, -0.1, 1.0, 0.9],
        [2.0, -3.0, 0.0, -2.0, -1.0, -0.5],
    ])
    result = temporal.temporal_pair_losses(
        scorer, cues, gt_overlap=torch.tensor([0.8, 0.1]),
        source_ids=torch.tensor([0, 1]), riou_threshold=0.5,
        margin=0.5, retention_weight=2.0, gain_weight=1.0,
        prior_weight=0.01)
    assert result['s7_temporal_retention_pair_count'] == 1
    assert result['s7_temporal_gain_pair_count'] == 0
    total = (result['loss_s7_temporal_retention']
             + result['loss_s7_temporal_gain']
             + result['loss_s7_temporal_prior'])
    total.backward()
    assert scorer.raw_weights.grad is not None
    assert torch.isfinite(scorer.raw_weights.grad).all()


def test_causal_selector_falls_back_then_requires_two_confirmations(monkeypatch):
    monkeypatch.setattr(temporal, '_default_rotated_iou', _unit_iou)
    scorer = temporal.S7TemporalAssociationScorer()
    selector = temporal.CausalTemporalCandidateSelector(
        scorer, max_candidates=100, min_confirmations=2,
        override_margin=0.0, max_center_distance=3.0,
        min_rotated_iou=0.05, min_appearance_similarity=0.2)
    detections = _detections()
    embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    sources = torch.tensor([0, 1])

    first = selector.select(detections, embeddings, sources, 'seq', 1)
    assert first['selected_index'] == 0
    assert first['reason'] == 'native_fallback_after_reset'
    second = selector.select(detections, embeddings, sources, 'seq', 2)
    assert second['selected_index'] == 0
    assert second['reason'] == 'native_fallback_pending_confirmation'
    third = selector.select(detections, embeddings, sources, 'seq', 3)
    assert third['selected_index'] == 1
    assert third['reason'] == 'override_confirmed'
    assert third['override'] is True
    assert third['candidate_index'] == 1
    assert third['candidate_margin_ok'] is True
    assert third['candidate_continuity_ok'] is True
    assert third['candidate_override_ok'] is True


def test_causal_selector_resets_on_gap_and_never_reuses_future_state(monkeypatch):
    monkeypatch.setattr(temporal, '_default_rotated_iou', _unit_iou)
    selector = temporal.CausalTemporalCandidateSelector(
        temporal.S7TemporalAssociationScorer(), min_confirmations=1,
        override_margin=0.0)
    detections = _detections()
    embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    sources = torch.tensor([0, 1])
    selector.select(detections, embeddings, sources, 'seq', 1)
    selected = selector.select(detections, embeddings, sources, 'seq', 2)
    assert selected['selected_index'] == 1
    after_gap = selector.select(
        detections, embeddings, sources, 'seq', 4)
    assert after_gap['selected_index'] == 0
    assert after_gap['reset'] is True
    assert after_gap['reason'] == 'native_fallback_after_reset'
    assert after_gap['candidate_index'] is None
    assert after_gap['candidate_override_ok'] is False
