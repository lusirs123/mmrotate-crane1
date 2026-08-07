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


def test_quality_cue_is_optional_and_keeps_the_six_cue_compatibility():
    detections = _detections()
    embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    quality = torch.tensor([0.0, 1.5])
    cues = temporal.build_temporal_cues(
        detections, embeddings, detections[0, :5], embeddings[0],
        rotated_iou_fn=_unit_iou, candidate_quality=quality)
    assert cues.shape == (2, len(temporal.QUALITY_CUE_NAMES))
    assert cues[:, -1].tolist() == pytest.approx([0.0, 1.5])


def test_candidate_quality_head_is_constant_before_training_and_has_dense_gradients():
    head = temporal.S7CandidateQualityHead(embedding_channels=4, hidden=8)
    detections = torch.tensor([
        [10.0, 10.0, 8.0, 4.0, 0.0, 0.4],
        [11.0, 10.0, 8.0, 4.0, 0.1, 0.9],
        [12.0, 10.0, 7.0, 5.0, 0.2, 0.7]])
    embeddings = torch.randn(3, 4)
    source_ids = torch.tensor([0, 1, 1])
    logits = head(embeddings, detections, source_ids)
    assert logits.tolist() == pytest.approx([0.0, 0.0, 0.0])
    losses = temporal.candidate_quality_losses(
        head, embeddings, detections, source_ids,
        gt_overlap=torch.tensor([0.8, 0.2, 0.0]), riou_threshold=0.5)
    assert losses['s7_candidate_quality_count'] == 3
    assert losses['s7_candidate_quality_usable_count'] == 1
    losses['loss_s7_candidate_quality'].backward()
    assert head.output.weight.grad is not None
    assert torch.isfinite(head.output.weight.grad).all()


def test_candidate_quality_relative_ranking_builds_deterministic_pairs():
    logits = torch.zeros(3, requires_grad=True)
    result = temporal.candidate_quality_relative_ranking_loss(
        logits, torch.tensor([0.9, 0.5, 0.0]),
        margin=0.25, min_gap=0.1, max_pairs=2)
    assert result['s7_candidate_quality_relative_pair_count'] == 2
    assert result['s7_candidate_quality_relative_active_count'] == 2
    assert result['s7_candidate_quality_relative_accuracy'] == pytest.approx(0.0)
    assert result['s7_candidate_quality_relative_mean_gap'] == pytest.approx(
        0.45)
    result['loss_s7_candidate_quality_relative'].backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_candidate_quality_losses_can_add_relative_ranking_term():
    head = temporal.S7CandidateQualityHead(embedding_channels=4, hidden=8)
    detections = torch.tensor([
        [10.0, 10.0, 8.0, 4.0, 0.0, 0.4],
        [11.0, 10.0, 8.0, 4.0, 0.1, 0.9],
        [12.0, 10.0, 7.0, 5.0, 0.2, 0.7]])
    losses = temporal.candidate_quality_losses(
        head, torch.randn(3, 4), detections, torch.tensor([0, 0, 1]),
        gt_overlap=torch.tensor([0.8, 0.3, 0.0]), riou_threshold=0.5,
        relative_margin=0.25, relative_min_gap=0.1, relative_max_pairs=8)
    assert losses['s7_candidate_quality_relative_pair_count'] == 2
    total = (losses['loss_s7_candidate_quality']
             + losses['loss_s7_candidate_quality_relative'])
    total.backward()
    assert head.output.weight.grad is not None
    assert torch.isfinite(head.output.weight.grad).all()


def test_candidate_student_combines_source_quality_relative_and_distillation():
    teacher = temporal.S7CandidateQualityHead(embedding_channels=4, hidden=8)
    student = temporal.S7CandidateQualityHead(embedding_channels=4, hidden=8)
    with torch.no_grad():
        teacher.output.bias.fill_(0.5)
    detections = torch.tensor([
        [10.0, 10.0, 8.0, 4.0, 0.0, 0.4],
        [11.0, 10.0, 8.0, 4.0, 0.1, 0.9],
        [12.0, 10.0, 7.0, 5.0, 0.2, 0.7]])
    losses = temporal.candidate_student_losses(
        student, teacher, torch.randn(3, 4), detections,
        torch.tensor([0, 0, 1]), torch.tensor([0.8, 0.3, 0.0]),
        riou_threshold=0.5, supervised_frame_weight=2.0)
    total = (losses['loss_s7_student_quality']
             + losses['loss_s7_student_relative']
             + losses['loss_s7_student_distillation'])
    total.backward()
    assert student.output.weight.grad is not None
    assert torch.isfinite(student.output.weight.grad).all()
    assert teacher.output.weight.grad is None
    assert losses['s7_student_supervised_frame_weight'] == pytest.approx(2.0)


def test_static_candidate_ranker_uses_native_retention_and_s7_gain_pairs():
    head = temporal.S7CandidateQualityHead(embedding_channels=4, hidden=8)
    detections = torch.tensor([
        [10.0, 10.0, 8.0, 4.0, 0.0, 0.90],
        [11.0, 10.0, 8.0, 4.0, 0.1, 0.80],
        [12.0, 10.0, 7.0, 5.0, 0.2, 0.70]])
    embeddings = torch.randn(3, 4)
    losses = temporal.static_candidate_rank_losses(
        head, embeddings, detections, torch.tensor([0, 1, 1]),
        gt_overlap=torch.tensor([0.8, 0.6, 0.0]), riou_threshold=0.5,
        relative_margin=0.25, relative_min_gap=0.1, relative_max_pairs=8)
    assert losses['s7_static_retention_pair_count'] == 1
    assert losses['s7_static_gain_pair_count'] == 0
    assert losses['s7_static_hard_negative_count'] == 1
    total = sum(losses[name] for name in (
        'loss_s7_static_quality', 'loss_s7_static_relative',
        'loss_s7_static_retention', 'loss_s7_static_gain',
        'loss_s7_static_prior'))
    total.backward()
    assert head.output.weight.grad is not None
    assert torch.isfinite(head.output.weight.grad).all()


def test_static_candidate_ranker_can_train_usable_s7_gain_when_native_is_wrong():
    head = temporal.S7CandidateQualityHead(embedding_channels=4, hidden=8)
    detections = torch.tensor([
        [10.0, 10.0, 8.0, 4.0, 0.0, 0.90],
        [11.0, 10.0, 8.0, 4.0, 0.1, 0.80]])
    losses = temporal.static_candidate_rank_losses(
        head, torch.randn(2, 4), detections, torch.tensor([0, 1]),
        gt_overlap=torch.tensor([0.0, 0.7]), riou_threshold=0.5)
    assert losses['s7_static_retention_pair_count'] == 0
    assert losses['s7_static_gain_pair_count'] == 1


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


def test_quality_aware_selector_requires_and_consumes_aligned_quality_logits(
        monkeypatch):
    monkeypatch.setattr(temporal, '_default_rotated_iou', _unit_iou)
    scorer = temporal.S7TemporalAssociationScorer(
        cue_names=temporal.QUALITY_CUE_NAMES)
    selector = temporal.CausalTemporalCandidateSelector(
        scorer, min_confirmations=1, override_margin=0.0)
    detections = _detections()
    embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    sources = torch.tensor([0, 1])
    with pytest.raises(ValueError, match='aligned logits'):
        selector.select(detections, embeddings, sources, 'seq', 1)
    first = selector.select(
        detections, embeddings, sources, 'seq', 1,
        quality_logits=torch.tensor([0.0, 2.0]))
    assert first['selected_index'] == 0
    second = selector.select(
        detections, embeddings, sources, 'seq', 2,
        quality_logits=torch.tensor([0.0, 2.0]))
    assert second['candidate_index'] == 1


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


def test_selective_promotion_is_exact_native_fallback_at_initialization():
    head = temporal.S7SelectivePromotionHead(
        embedding_channels=4, hidden=8, initial_uncertainty=0.5)
    detections = torch.tensor([
        [10.0, 10.0, 8.0, 4.0, 0.0, 0.60],
        [11.0, 10.0, 8.0, 4.0, 0.1, 0.95]])
    result = temporal.native_protected_selective_promotion(
        head, torch.randn(2, 4), detections, torch.tensor([0, 1]),
        torch.tensor([0.0, 1.0]), promotion_margin=0.10)
    assert result['selected_index'] == 0
    assert result['promoted'] is False
    assert result['reason'] == 'native_fallback_uncertain_advantage'
    assert result['best_lower_bound'] < 0.10


def test_selective_promotion_preserves_pool_when_native_is_missing():
    head = temporal.S7SelectivePromotionHead(
        embedding_channels=4, hidden=8, initial_uncertainty=0.5)
    detections = torch.tensor([
        [10.0, 10.0, 8.0, 4.0, 0.0, 0.60],
        [11.0, 10.0, 8.0, 4.0, 0.1, 0.95],
        [12.0, 10.0, 7.0, 5.0, 0.2, 0.70]])
    original_order = torch.arange(detections.shape[0])
    result = temporal.native_protected_selective_promotion(
        head, torch.randn(3, 4), detections, torch.tensor([1, 1, 1]),
        torch.zeros(3), promotion_margin=0.10)
    assert result['selected_index'] is None
    assert result['native_index'] is None
    assert result['promoted'] is False
    assert result['reason'] == 'native_missing'
    assert result['order'].tolist() == original_order.tolist()
    assert result['candidate_count'] == 0


def test_selective_promotion_promotes_on_exact_margin_boundary():
    head = temporal.S7SelectivePromotionHead(
        embedding_channels=4, hidden=8, initial_uncertainty=0.5)
    with torch.no_grad():
        # Choose a bias whose LCB is exactly the configured promotion margin.
        head.advantage_output.bias.fill_(0.60)
        head.uncertainty_output.bias.fill_(
            temporal._inverse_softplus(0.50))
    detections = torch.tensor([
        [10.0, 10.0, 8.0, 4.0, 0.0, 0.60],
        [11.0, 10.0, 8.0, 4.0, 0.1, 0.95]])
    result = temporal.native_protected_selective_promotion(
        head, torch.randn(2, 4), detections, torch.tensor([0, 1]),
        torch.zeros(2), promotion_margin=0.10)
    assert result['best_lower_bound'] == pytest.approx(0.10, abs=1e-5)
    assert result['selected_index'] == 1
    assert result['promoted'] is True


def test_selective_promotion_can_promote_only_after_confident_advantage():
    head = temporal.S7SelectivePromotionHead(
        embedding_channels=4, hidden=8, initial_uncertainty=0.5)
    with torch.no_grad():
        head.advantage_output.bias.fill_(1.0)
        head.uncertainty_output.bias.fill_(-10.0)
    detections = torch.tensor([
        [10.0, 10.0, 8.0, 4.0, 0.0, 0.60],
        [11.0, 10.0, 8.0, 4.0, 0.1, 0.95]])
    result = temporal.native_protected_selective_promotion(
        head, torch.randn(2, 4), detections, torch.tensor([0, 1]),
        torch.tensor([0.0, 1.0]), promotion_margin=0.10)
    assert result['selected_index'] == 1
    assert result['promoted'] is True
    assert result['reason'] == 's7_promoted_confident_advantage'


def test_selective_promotion_uses_s7_lane_top100_not_global_prefix():
    head = temporal.S7SelectivePromotionHead(
        embedding_channels=2, hidden=8, initial_uncertainty=0.5)
    with torch.no_grad():
        for layer in (head.embedding_projection, head.scalar_projection):
            for module in layer:
                if hasattr(module, 'weight'):
                    module.weight.zero_()
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.zero_()
        head.advantage_output.weight.zero_()
        head.advantage_output.bias.fill_(1.0)
        head.uncertainty_output.weight.zero_()
        head.uncertainty_output.bias.fill_(-10.0)
    native_count = 101
    detections = torch.zeros(native_count + 1, 6)
    detections[:, 2:4] = torch.tensor([8.0, 4.0])
    detections[:, 5] = 0.10
    detections[:native_count, 5] = 0.90
    sources = torch.zeros(native_count + 1, dtype=torch.long)
    sources[-1] = 1
    result = temporal.native_protected_selective_promotion(
        head, torch.zeros(native_count + 1, 2), detections, sources,
        torch.zeros(native_count + 1), max_candidates=1,
        promotion_margin=0.10)
    assert result['candidate_count'] == 1
    assert result['selected_index'] == native_count
    assert result['promoted'] is True


def test_selective_promotion_losses_uses_s7_lane_top100_not_global_prefix():
    head = temporal.S7SelectivePromotionHead(
        embedding_channels=2, hidden=8, initial_uncertainty=0.5)
    native_count = 101
    detections = torch.zeros(native_count + 1, 6)
    detections[:, 2:4] = torch.tensor([8.0, 4.0])
    detections[:, 5] = 0.90
    sources = torch.zeros(native_count + 1, dtype=torch.long)
    sources[-1] = 1
    losses = temporal.selective_promotion_losses(
        head, torch.zeros(native_count + 1, 2), detections, sources,
        torch.zeros(native_count + 1), torch.tensor(
            [0.5] * native_count + [0.9]), riou_threshold=0.5,
        max_candidates=1)
    assert losses['s7_selective_candidate_count'] == 1


@pytest.mark.parametrize(
    'overlap,retention_pairs,gain_pairs', [
        ([0.8, 0.1, 0.0], 2, 0),
        ([0.0, 0.8, 0.1], 0, 1),
    ])
def test_selective_promotion_loss_trains_retention_and_gain_cases(
        overlap, retention_pairs, gain_pairs):
    head = temporal.S7SelectivePromotionHead(
        embedding_channels=4, hidden=8, initial_uncertainty=0.5)
    detections = torch.tensor([
        [10.0, 10.0, 8.0, 4.0, 0.0, 0.90],
        [11.0, 10.0, 8.0, 4.0, 0.1, 0.80],
        [12.0, 10.0, 7.0, 5.0, 0.2, 0.70]])
    losses = temporal.selective_promotion_losses(
        head, torch.randn(3, 4), detections, torch.tensor([0, 1, 1]),
        torch.tensor([0.0, 0.5, -0.5]), torch.tensor(overlap),
        riou_threshold=0.5)
    assert losses['s7_selective_retention_pair_count'] == retention_pairs
    assert losses['s7_selective_gain_pair_count'] == gain_pairs
    total = sum(losses[name] for name in (
        'loss_s7_selective_quality',
        'loss_s7_selective_classification',
        'loss_s7_selective_retention', 'loss_s7_selective_gain',
        'loss_s7_selective_prior'))
    total.backward()
    assert head.advantage_output.weight.grad is not None
    assert torch.isfinite(head.advantage_output.weight.grad).all()
