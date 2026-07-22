from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from crane_project.tools import retina_cls_contribution_probe as probe


def _args(**overrides):
    values = dict(
        seed=0, target_frames=[150, 155], source_frames=[104, 105],
        riou_thr=0.5, false_iou_thr=0.1, top_channels=3,
        reconstruction_atol=1e-4)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_protocol_keeps_target_dev_and_source_controls_small():
    assert probe.validate_args(_args()) == ([150, 155], [104, 105])
    with pytest.raises(ValueError, match='137..169'):
        probe.validate_args(_args(target_frames=[150, 170]))
    with pytest.raises(ValueError, match='1-3 unique'):
        probe.validate_args(_args(source_frames=[]))


class _AnchorGenerator:
    num_base_anchors = [3]


class _Head:
    cls_out_channels = 1
    anchor_generator = _AnchorGenerator()
    filter_padding_anchors = False


def test_candidate_layout_matches_location_major_anchor_order():
    scores = [torch.zeros(1, 3, 2, 2)]
    layout = probe.candidate_layout(scores, _Head(), (16, 16, 3))
    assert len(layout) == 12
    assert layout[0] == dict(
        level=0, row=0, col=0, anchor_id=0,
        output_channel=0, raw_level_index=0)
    assert layout[5]['row'] == 0
    assert layout[5]['col'] == 1
    assert layout[5]['anchor_id'] == 2
    assert layout[6]['row'] == 1
    assert layout[6]['col'] == 0
    assert layout[6]['anchor_id'] == 0


def test_exact_decomposition_reconstructs_conv_logit():
    conv = nn.Conv2d(2, 3, 3, padding=1, bias=True)
    torch.manual_seed(3)
    nn.init.normal_(conv.weight)
    nn.init.normal_(conv.bias)
    feature = torch.randn(1, 2, 4, 5)
    logits = conv(feature)
    location = dict(row=2, col=3, output_channel=1)
    actual = float(logits[0, 1, 2, 3].item())
    score = float(torch.sigmoid(logits[0, 1, 2, 3]).item())
    result = probe.exact_contributions(
        feature, conv, location, actual, score, 2, 1e-5)
    assert result['reconstruction_passed'] is True
    assert result['reconstructed_logit'] == pytest.approx(actual, abs=1e-5)
    assert (result['bias'] + result['positive_contribution_sum']
            + result['negative_contribution_sum']) == pytest.approx(
                actual, abs=1e-5)


def test_channel_comparison_marks_anchor_confound():
    def candidate(anchor, values):
        return dict(
            location=dict(anchor_id=anchor),
            decomposition=dict(per_channel=values))

    same = probe.compare_channel_contributions(
        candidate(1, [1.0, -1.0]), candidate(1, [0.5, -0.5]), 1)
    different = probe.compare_channel_contributions(
        candidate(1, [1.0, -1.0]), candidate(2, [0.5, -0.5]), 1)
    assert same['interpretation_valid_without_anchor_confound'] is True
    assert different['interpretation_valid_without_anchor_confound'] is False


def test_per_anchor_selection_keeps_geometry_and_score_views_separate():
    scores = torch.tensor([0.9, 0.2, 0.1, 0.8, 0.7, 0.3])
    ious = torch.tensor([0.0, 0.7, 0.8, 0.1, 0.6, 0.9])
    layout = [dict(anchor_id=index % 3) for index in range(6)]
    rows = probe.select_per_anchor_candidates(
        scores, ious, layout, num_anchors=3, riou_thr=0.5)
    assert rows[0]['highest_score']['index'] == 0
    assert rows[0]['best_usable_by_score'] is None
    assert rows[1]['best_usable_by_score']['index'] == 4
    assert rows[2]['best_usable_by_score']['index'] == 5
    assert rows[2]['dense_best_geometry']['riou'] == pytest.approx(0.9)


def test_classifier_filter_stats_exposes_anchor_imbalance():
    conv = nn.Conv2d(2, 3, 3, padding=1, bias=True)
    with torch.no_grad():
        conv.weight[0].fill_(1.0)
        conv.weight[1].fill_(0.1)
        conv.weight[2].fill_(0.01)
        conv.bias.copy_(torch.tensor([1.0, 2.0, 3.0]))
    rows = probe.classifier_filter_stats(conv)
    assert [row['bias'] for row in rows] == pytest.approx([1.0, 2.0, 3.0])
    assert rows[1]['weight_norm_relative_to_anchor0'] == pytest.approx(0.1)
    assert rows[2]['weight_norm_relative_to_anchor0'] == pytest.approx(0.01)


def _mock_candidate(anchor, rank, score, per_channel_values,
                    top_pos=None, top_neg=None):
    """Build a minimal candidate dict for comparison/consistency tests."""
    return dict(
        candidate_index=0,
        score=score,
        riou=0.5,
        rank=rank,
        location=dict(
            level=0, row=0, col=0, anchor_id=anchor,
            output_channel=anchor, raw_level_index=0),
        decomposition=dict(
            reconstruction_passed=True,
            actual_logit=-1.0,
            reconstructed_logit=-1.0,
            reconstruction_abs_error=1e-7,
            actual_score=score,
            reconstructed_score=score,
            score_abs_error=1e-7,
            bias=-1.0,
            contribution_sum=sum(per_channel_values),
            positive_contribution_sum=sum(v for v in per_channel_values if v > 0),
            negative_contribution_sum=sum(v for v in per_channel_values if v < 0),
            positive_channel_count=sum(1 for v in per_channel_values if v > 0),
            negative_channel_count=sum(1 for v in per_channel_values if v < 0),
            patch_norm=1.0,
            weight_norm=1.0,
            patch_weight_cosine=0.0,
            top_positive_channels=top_pos or [],
            top_negative_channels=top_neg or [],
            per_channel=per_channel_values))


def test_false_source_control_comparisons_finds_same_anchor():
    target_rows = [dict(
        frame=155, role='target_dev',
        false_candidate=_mock_candidate(0, 1, 0.2, [1.0, -0.5]),
        usable_candidate=_mock_candidate(2, 9000, 0.0001, [0.3, -0.8]),
        usable_vs_false=None, image_stats={}, dense_best_riou=0.5,
        decode_alignment={})]
    source_rows = [dict(
        frame=104, role='source_val_control',
        false_candidate=_mock_candidate(1, 50, 0.007, [0.5, -0.3]),
        usable_candidate=_mock_candidate(0, 1, 0.88, [2.0, 0.1]),
        usable_vs_false=None, image_stats={}, dense_best_riou=0.86,
        decode_alignment={})]
    results = probe.false_source_control_comparisons(
        target_rows, source_rows, 3)
    assert len(results) == 1
    assert results[0]['target_anchor_id'] == 0
    assert results[0]['source_anchor_id'] == 0
    assert results[0]['comparison']['same_anchor_filter'] is True
    assert results[0]['target_candidate_type'] == 'false'
    assert results[0]['source_candidate_type'] == 'usable'


def test_false_source_control_comparisons_marks_different_anchor():
    target_rows = [dict(
        frame=150, role='target_dev',
        false_candidate=_mock_candidate(1, 1, 0.005, [1.0, -0.5]),
        usable_candidate=None,
        usable_vs_false=None, image_stats={}, dense_best_riou=0.5,
        decode_alignment={})]
    source_rows = [dict(
        frame=104, role='source_val_control',
        false_candidate=_mock_candidate(1, 50, 0.007, [0.5, -0.3]),
        usable_candidate=_mock_candidate(0, 1, 0.88, [2.0, 0.1]),
        usable_vs_false=None, image_stats={}, dense_best_riou=0.86,
        decode_alignment={})]
    results = probe.false_source_control_comparisons(
        target_rows, source_rows, 3)
    assert len(results) == 1
    assert results[0]['comparison']['same_anchor_filter'] is False


def test_cross_frame_consistency_detects_shared_channels():
    top_pos = [
        dict(channel=10, contribution=0.5, patch_norm=1.0, weight_norm=0.5),
        dict(channel=20, contribution=0.3, patch_norm=1.0, weight_norm=0.5),
        dict(channel=30, contribution=0.2, patch_norm=1.0, weight_norm=0.5)]
    top_neg = [
        dict(channel=5, contribution=-0.5, patch_norm=1.0, weight_norm=0.5),
        dict(channel=15, contribution=-0.3, patch_norm=1.0, weight_norm=0.5),
        dict(channel=25, contribution=-0.2, patch_norm=1.0, weight_norm=0.5)]
    target_rows = [
        dict(frame=150, role='target_dev',
             false_candidate=_mock_candidate(1, 1, 0.005, [1.0, -0.5]),
             usable_candidate=_mock_candidate(
                 2, 6165, 0.0001, [0.3, -0.8], top_pos, top_neg),
             usable_vs_false=None, image_stats={}, dense_best_riou=0.5,
             decode_alignment={}),
        dict(frame=155, role='target_dev',
             false_candidate=_mock_candidate(0, 1, 0.2, [1.0, -0.5]),
             usable_candidate=_mock_candidate(
                 2, 9661, 0.00005, [0.3, -0.8], top_pos, top_neg),
             usable_vs_false=None, image_stats={}, dense_best_riou=0.5,
             decode_alignment={})]
    results = probe.cross_frame_consistency(target_rows, 3)
    # anchor=2 has 2 usable candidates across frames
    anchor2 = [r for r in results if r['anchor_id'] == 2][0]
    assert anchor2['frame_count'] == 2
    assert 10 in anchor2['consistent_positive_channels']
    assert 5 in anchor2['consistent_negative_channels']
    assert anchor2['top_positive_overlap_count'] == 3


def test_cross_frame_consistency_skips_single_frame_anchors():
    target_rows = [
        dict(frame=150, role='target_dev',
             false_candidate=_mock_candidate(1, 1, 0.005, [1.0, -0.5]),
             usable_candidate=_mock_candidate(
                 2, 6165, 0.0001, [0.3, -0.8]),
             usable_vs_false=None, image_stats={}, dense_best_riou=0.5,
             decode_alignment={})]
    results = probe.cross_frame_consistency(target_rows, 3)
    # Each anchor appears only once, so no consistency groups
    assert len(results) == 0


def test_build_decomposition_table_compacts_all_candidates():
    target_rows = [dict(
        frame=150, role='target_dev',
        false_candidate=_mock_candidate(1, 1, 0.005, [1.0, -0.5]),
        usable_candidate=_mock_candidate(2, 6165, 0.0001, [0.3, -0.8]),
        usable_vs_false=None, image_stats={}, dense_best_riou=0.5,
        decode_alignment={})]
    source_rows = [dict(
        frame=104, role='source_val_control',
        false_candidate=_mock_candidate(1, 50, 0.007, [0.5, -0.3]),
        usable_candidate=_mock_candidate(0, 1, 0.88, [2.0, 0.1]),
        usable_vs_false=None, image_stats={}, dense_best_riou=0.86,
        decode_alignment={})]
    table = probe.build_decomposition_table(target_rows, source_rows)
    assert len(table) == 4
    types = [(r['role'], r['candidate_type']) for r in table]
    assert ('target_dev', 'false') in types
    assert ('target_dev', 'usable') in types
    assert ('source_val_control', 'false') in types
    assert ('source_val_control', 'usable') in types
    # Check compact fields exist
    row = table[0]
    for key in ('rank', 'anchor_id', 'level', 'score', 'logit', 'bias',
                'contribution_sum', 'positive_contribution_sum',
                'negative_contribution_sum', 'positive_channel_count',
                'negative_channel_count', 'patch_norm', 'weight_norm',
                'patch_weight_cosine'):
        assert key in row
