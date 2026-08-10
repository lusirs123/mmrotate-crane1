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


def test_highres_quality_head_and_rank_loss_use_two_roi_resolutions():
    head = temporal.S7HighResCandidateQualityHead(
        embedding_channels=4, highres_channels=3, hidden=8)
    detections = torch.tensor([
        [10.0, 10.0, 8.0, 4.0, 0.0, 0.90],
        [11.0, 10.0, 8.0, 4.0, 0.1, 0.80]])
    embedding = torch.randn(2, 4)
    highres = torch.randn(2, 3)
    source_ids = torch.tensor([0, 1])
    logits = head(embedding, highres, detections, source_ids)
    assert logits.tolist() == pytest.approx([0.0, 0.0])
    losses = temporal.highres_candidate_rank_losses(
        head, embedding, highres, detections, source_ids,
        gt_overlap=torch.tensor([0.0, 0.7]), riou_threshold=0.5)
    assert losses['s7_highres_gain_pair_count'] == 1
    total = sum(losses[name] for name in (
        'loss_s7_highres_quality', 'loss_s7_highres_relative',
        'loss_s7_highres_retention', 'loss_s7_highres_gain',
        'loss_s7_highres_prior'))
    total.backward()
    assert head.output.weight.grad is not None
    assert torch.isfinite(head.output.weight.grad).all()


def test_highres_promotion_is_native_noop_before_training():
    head = temporal.S7HighResCandidateQualityHead(
        embedding_channels=4, highres_channels=3, hidden=8)
    detections = torch.tensor([
        [10.0, 10.0, 8.0, 4.0, 0.0, 0.40],
        [11.0, 10.0, 8.0, 4.0, 0.0, 0.90]])
    result = temporal.native_protected_highres_promotion(
        head, torch.randn(2, 4), torch.randn(2, 3), detections,
        torch.tensor([0, 1]), max_candidates=1)
    assert result['reason'] == 'native_fallback_zero_residual'
    assert result['promoted'] is False
    assert result['order'].tolist() == [0, 1]


def test_highres_margin_grid_reuses_logits_and_changes_only_policy():
    detections = torch.tensor([
        [10.0, 10.0, 8.0, 4.0, 0.0, 0.50],
        [11.0, 10.0, 8.0, 4.0, 0.0, 0.50]])
    quality = torch.tensor([0.0, 0.23])
    source_ids = torch.tensor([0, 1])
    permissive = temporal.native_protected_highres_promotion_from_logits(
        quality, detections, source_ids, max_candidates=1,
        promotion_margin=0.20)
    conservative = temporal.native_protected_highres_promotion_from_logits(
        quality, detections, source_ids, max_candidates=1,
        promotion_margin=0.25)
    assert permissive['promoted'] is True
    assert permissive['order'].tolist() == [1, 0]
    assert conservative['promoted'] is False
    assert conservative['order'].tolist() == [0, 1]
    assert permissive['best_margin'] == pytest.approx(0.23)
    assert conservative['best_margin'] == pytest.approx(0.23)


def test_unified_highres_ranker_mines_whole_pool_pairs_and_backprops():
    head = temporal.S7HighResCandidateQualityHead(
        embedding_channels=4, highres_channels=3, hidden=8)
    detections = torch.tensor([
        [10.0, 10.0, 8.0, 4.0, 0.0, 0.90],
        [11.0, 10.0, 8.0, 4.0, 0.1, 0.80],
        [12.0, 10.0, 7.0, 5.0, 0.2, 0.70],
        [13.0, 10.0, 7.0, 5.0, 0.3, 0.60]])
    result = temporal.unified_highres_candidate_rank_losses(
        head, torch.randn(4, 4), torch.randn(4, 3), detections,
        torch.tensor([0, 1, 1, 0]),
        gt_overlap=torch.tensor([0.9, 0.7, 0.3, 0.1]),
        riou_threshold=0.5, hard_pair_count=8)
    assert result['s7_highres_unified_pair_count'] >= 3
    assert result['s7_highres_retention_pair_count'] == 1
    total = sum(result[name] for name in (
        'loss_s7_highres_quality', 'loss_s7_highres_relative',
        'loss_s7_highres_unified', 'loss_s7_highres_prior'))
    total.backward()
    assert head.output.weight.grad is not None
    assert torch.isfinite(head.output.weight.grad).all()


def test_unified_highres_inference_uses_one_pool_but_keeps_native_margin():
    detections = torch.tensor([
        [10.0, 10.0, 8.0, 4.0, 0.0, 0.90],
        [11.0, 10.0, 8.0, 4.0, 0.1, 0.80],
        [12.0, 10.0, 7.0, 5.0, 0.2, 0.70]])
    sources = torch.tensor([0, 1, 1])
    promoted = temporal.native_protected_unified_highres_ranking_from_logits(
        torch.tensor([0.0, 1.2, 0.1]), detections, sources,
        max_candidates=2, promotion_margin=0.25)
    assert promoted['promoted'] is True
    assert promoted['selected_index'] == 1
    assert promoted['reason'] == 's7_promoted_unified_quality'
    retained = temporal.native_protected_unified_highres_ranking_from_logits(
        torch.tensor([0.0, 0.1, 0.0]), detections, sources,
        max_candidates=2, promotion_margin=0.25)
    assert retained['promoted'] is False
    assert retained['selected_index'] == 0
    assert retained['reason'] == 'native_fallback_unified_margin'


def test_pairwise_takeover_zero_init_abstains_and_loss_backpropagates():
    head = temporal.S7HighResPairwiseTakeoverHead(
        embedding_channels=4, highres_channels=3, hidden=8,
        initial_uncertainty=0.25)
    detections = torch.tensor([
        [10.0, 10.0, 8.0, 4.0, 0.0, 0.90],
        [11.0, 10.0, 8.0, 4.0, 0.1, 0.80],
        [12.0, 10.0, 7.0, 5.0, 0.2, 0.70]])
    embedding = torch.randn(3, 4)
    highres = torch.randn(3, 3)
    sources = torch.tensor([0, 1, 1])
    prediction = head(embedding, highres, detections, sources)
    assert prediction['mean'].tolist() == pytest.approx([0.0, 0.0])
    assert prediction['uncertainty'].tolist() == pytest.approx([0.25, 0.25])
    selection = temporal.native_protected_pairwise_highres_takeover(
        prediction, detections, sources, uncertainty_multiplier=2.0,
        takeover_margin=0.05, deployment_score_thr=0.05)
    assert selection['promoted'] is False
    assert selection['selected_index'] == 0
    assert selection['best_lower_bound'] == pytest.approx(-0.5)

    losses = temporal.pairwise_highres_takeover_losses(
        head, embedding, highres, detections, sources,
        gt_overlap=torch.tensor([0.2, 0.9, 0.1]),
        riou_threshold=0.5, augmented_highres_embedding=highres + 0.1)
    assert losses['s7_takeover_gain_pair_count'] == 1
    assert losses['s7_takeover_ranking_pair_count'] > 0
    total = sum(value for name, value in losses.items()
                if name.startswith('loss_'))
    total.backward()
    assert head.output.weight.grad is not None
    assert torch.isfinite(head.output.weight.grad).all()


def test_pairwise_takeover_requires_deployable_score_and_positive_lcb():
    detections = torch.tensor([
        [10.0, 10.0, 8.0, 4.0, 0.0, 0.90],
        [11.0, 10.0, 8.0, 4.0, 0.1, 0.04],
        [12.0, 10.0, 7.0, 5.0, 0.2, 0.70]])
    sources = torch.tensor([0, 1, 1])
    prediction = dict(
        native_index=0, s7_indices=torch.tensor([1, 2]),
        raw_mean=torch.tensor([3.0, 2.0]),
        mean=torch.tensor([0.90, 0.40]),
        uncertainty=torch.tensor([0.10, 0.10]))
    selected = temporal.native_protected_pairwise_highres_takeover(
        prediction, detections, sources, uncertainty_multiplier=2.0,
        takeover_margin=0.05, deployment_score_thr=0.05)
    assert selected['promoted'] is True
    assert selected['selected_index'] == 2
    assert selected['eligible_count'] == 1
    conservative = temporal.native_protected_pairwise_highres_takeover(
        dict(prediction, mean=torch.tensor([0.90, 0.20])),
        detections, sources, uncertainty_multiplier=2.0,
        takeover_margin=0.05, deployment_score_thr=0.05)
    assert conservative['promoted'] is False
    assert conservative['selected_index'] == 0


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


def _two_frame_state():
    state = temporal.TwoFrameMotionState()
    state.update(torch.tensor([10.0, 10.0, 8.0, 4.0, 1.50]),
                 torch.tensor([1.0, 0.0]), 'seq', 1)
    state.update(torch.tensor([12.0, 11.0, 8.0, 4.0, -1.50]),
                 torch.tensor([1.0, 0.0]), 'seq', 2)
    return state


def test_two_frame_motion_uses_constant_velocity_and_periodic_angle():
    state = _two_frame_state()
    predicted_angle = -1.50 + temporal._periodic_angle_delta(
        torch.tensor(-1.50), torch.tensor(1.50))
    detections = torch.tensor([[14.0, 12.0, 8.0, 4.0,
                                float(predicted_angle), 0.9]])
    cues = temporal._two_frame_candidate_motion_features(
        detections, torch.tensor([[1.0, 0.0]]), state)
    assert cues.shape == (1, 7)
    assert cues[0, :5].tolist() == pytest.approx(
        [0.0, 0.0, 0.0, 0.0, 0.0], abs=1e-5)
    assert cues[0, 5].item() == pytest.approx(1.0, abs=1e-5)
    assert cues[0, 6].item() == pytest.approx(1.0, abs=1e-5)


def test_small_temporal_ranker_is_fixed_24_scalar_and_lightweight():
    head = temporal.S7SmallTemporalRankerHead(hidden=16)
    detections = torch.tensor([
        [14.0, 12.0, 8.0, 4.0, 0.0, 0.8],
        [14.5, 12.0, 7.0, 4.0, 0.1, 0.7]])
    embeddings = torch.tensor([[1.0, 0.0], [0.9, 0.1]])
    features = head.pair_features(
        embeddings[0], detections[0], torch.tensor(0.2),
        embeddings[1], detections[1], torch.tensor(0.8),
        _two_frame_state())
    assert features.shape == (1, 24)
    assert sum(parameter.numel() for parameter in head.parameters()) < 1000


def test_small_temporal_ranker_warms_up_then_promotes_only_on_confident_lcb():
    head = temporal.S7SmallTemporalRankerHead(hidden=16)
    with torch.no_grad():
        head.advantage_output.bias.fill_(1.0)
        head.uncertainty_output.bias.fill_(-10.0)
    selector = temporal.CausalSmallTemporalRanker(
        head, max_candidates=20, promotion_margin=0.1)
    detections = torch.tensor([
        [10.0, 10.0, 8.0, 4.0, 0.0, 0.8],
        [11.0, 10.0, 7.0, 4.0, 0.0, 0.7]])
    embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    sources = torch.tensor([0, 1])
    quality = torch.tensor([0.0, 1.0])
    first = selector.select(
        detections, embeddings, sources, quality, 'seq', 1)
    second = selector.select(
        detections, embeddings, sources, quality, 'seq', 2)
    third = selector.select(
        detections, embeddings, sources, quality, 'seq', 3)
    assert first['promoted'] is False and first['history_ready'] is False
    assert second['promoted'] is False and second['history_ready'] is False
    assert third['promoted'] is True and third['history_ready'] is True
    assert third['selected_index'] == 1


def test_small_temporal_ranker_resets_on_frame_gap_and_quality_prefilters_top20():
    head = temporal.S7SmallTemporalRankerHead(hidden=16)
    selector = temporal.CausalSmallTemporalRanker(head, max_candidates=20)
    detections = torch.zeros(23, 6)
    detections[:, 2:4] = torch.tensor([8.0, 4.0])
    detections[:, 5] = torch.linspace(0.99, 0.10, 23)
    sources = torch.ones(23, dtype=torch.long)
    sources[0] = 0
    quality = torch.zeros(23)
    quality[5] = 2.0
    quality[22] = 10.0
    native, s7, count = temporal._native_and_quality_selected_s7(
        detections, sources, quality, max_candidates=20)
    assert native.item() == 0
    assert s7.item() == 5
    assert count == 20
    embeddings = torch.randn(23, 2)
    selector.select(detections, embeddings, sources, quality, 'seq', 1)
    result = selector.select(
        detections, embeddings, sources, quality, 'seq', 3)
    assert result['reset'] is True
    assert result['history_ready'] is False


def test_small_temporal_training_state_follows_current_lcb_selection():
    head = temporal.S7SmallTemporalRankerHead(hidden=16)
    with torch.no_grad():
        head.advantage_output.bias.fill_(1.0)
        head.uncertainty_output.bias.fill_(-10.0)
    detections = torch.tensor([
        [14.0, 12.0, 8.0, 4.0, 0.0, 0.8],
        [14.5, 12.0, 7.0, 4.0, 0.1, 0.7]])
    embeddings = torch.tensor([[1.0, 0.0], [0.9, 0.1]])
    losses = temporal.small_temporal_ranker_losses(
        head, embeddings, detections, torch.tensor([0, 1]),
        torch.tensor([0.0, 1.0]), torch.tensor([0.1, 0.9]),
        _two_frame_state(), riou_threshold=0.5,
        promotion_margin=0.1, max_candidates=20)
    assert losses['_s7_small_temporal_selected_index'] == 1
