import math
from types import SimpleNamespace

import pytest

from crane_project.tools import target_hard_negative_atlas_probe as atlas


def _args(**overrides):
    values = dict(
        seed=0,
        split='test',
        seq='real_seq02',
        start=137,
        end=169,
        candidate_source='main',
        pool_size=10000,
        riou_thr=0.5,
        false_iou_thr=0.1,
        false_peaks_per_frame=5,
        false_diversity_iou=0.3,
        false_search_limit=500,
        cluster_center_thr=0.06,
        cluster_log_area_thr=0.8,
        cluster_angle_thr=35.0,
        cluster_max_frame_gap=2,
        min_recurrent_frames=4,
        allow_noncanonical=False)
    values.update(overrides)
    return SimpleNamespace(**values)

def test_canonical_argument_gate():
    assert atlas.validate_args(_args()) is True
    with pytest.raises(ValueError, match='Canonical atlas'):
        atlas.validate_args(_args(start=138))
    assert atlas.validate_args(
        _args(start=138, allow_noncanonical=True)) is False


def test_original_box_mapping_is_isotropic_and_angle_preserving():
    box = atlas.decoded_box_to_original(
        [200.0, 100.0, 80.0, 40.0, 0.3],
        dict(scale_factor=[2.0, 2.0, 2.0, 2.0], flip=False))
    assert box == pytest.approx([100.0, 50.0, 40.0, 20.0, 0.3])
    with pytest.raises(ValueError, match='Anisotropic'):
        atlas.decoded_box_to_original(
            [1, 2, 3, 4, 0], dict(scale_factor=[2.0, 1.0]))


def _peak(frame, x, y, score=0.8, order=1, angle=0.0, area=0.01):
    return dict(
        frame=frame,
        peak_order=order,
        score=score,
        riou=0.0,
        normalized=dict(
            center_x=x,
            center_y=y,
            area_ratio=area,
            log_area=math.log(area),
            aspect_ratio=2.0,
            angle_deg=angle))


def test_recurrent_fixed_peak_forms_one_cluster():
    peaks = [_peak(frame, 0.20 + frame * 0.0002, 0.30)
             for frame in range(137, 143)]
    clusters = atlas.cluster_false_peaks(peaks)
    assert len(clusters) == 1
    assert clusters[0]['frame_count'] == 6
    assert clusters[0]['longest_consecutive'] == 6


def test_distant_or_gapped_peaks_do_not_merge():
    peaks = [
        _peak(137, 0.1, 0.1),
        _peak(138, 0.8, 0.8),
        _peak(142, 0.1, 0.1),
    ]
    clusters = atlas.cluster_false_peaks(peaks, max_frame_gap=2)
    assert len(clusters) == 3


def test_summary_separates_geometry_miss_and_recurrent_coverage():
    rows = [
        dict(frame=137, top1_is_false=True,
             top1_score=0.01, false_peaks=[],
             usable_candidate=dict(cls_rank=5000, score=0.001, fpn_level=0)),
        dict(frame=138, top1_is_false=True,
             top1_score=0.02, false_peaks=[],
             usable_candidate=dict(cls_rank=7000, score=0.002, fpn_level=1)),
        dict(frame=139, top1_is_false=True, top1_score=0.03,
             false_peaks=[], usable_candidate=None),
    ]
    clusters = [dict(
        cluster_id=3, frame_count=2, longest_consecutive=2,
        frames=[137, 138], peak_orders=[1, 1],
        occurrences=[dict(frame=137, peak_order=1),
                     dict(frame=138, peak_order=1)])]
    summary = atlas.build_summary(rows, clusters, min_recurrent_frames=2)
    assert summary['usable_frames'] == 2
    assert summary['geometry_miss_frames'] == [139]
    assert summary['usable_rank_median'] == pytest.approx(6000.0)
    assert summary['recurrent_top1_coverage'] == pytest.approx(2.0 / 3.0)
    assert summary['padding_anchors_removed_total'] == 0
    assert summary['padding_anchor_removed_ratio'] == 0.0


def test_summary_reports_padding_candidates_removed():
    rows = [dict(
        frame=137, top1_is_false=False, top1_score=0.8,
        false_peaks=[], usable_candidate=None,
        decode_alignment=[dict(
            anchors=60, anchors_before_content_filter=100,
            padding_anchors_removed=40)])]
    summary = atlas.build_summary(rows, [], min_recurrent_frames=2)
    assert summary['padding_anchors_removed_total'] == 40
    assert summary['padding_anchors_removed_median_per_frame'] == 40.0
    assert summary['padding_anchor_removed_ratio'] == pytest.approx(0.4)


def test_candidate_origin_separates_anchor_and_decoded_edge():
    origin = atlas.candidate_origin_geometry(
        box_img=[0.0, 300.0, 40.0, 20.0, 0.0],
        anchor_center_img=[400.0, 300.0],
        img_shape=(576, 1024, 3))
    assert origin['anchor_near_border'] is False
    assert origin['decoded_near_border'] is True
    assert origin['decoded_on_boundary'] is True
    assert origin['anchor_nearest_edge'] == 'left'
    assert origin['decoded_nearest_edge'] == 'left'
    assert origin['anchor_to_decoded_shift_px'] == pytest.approx(400.0)


def test_summary_reports_top1_origin_attribution():
    rows = [dict(
        frame=137, top1_is_false=True, top1_score=0.8,
        usable_candidate=None, decode_alignment=[],
        false_peaks=[dict(
            peak_order=1, candidate_kind='hard_false_background',
            fpn_level=0,
            origin=dict(
                anchor_near_border=False,
                decoded_near_border=True,
                decoded_on_boundary=True,
                anchor_to_decoded_shift_ratio=0.25))])]
    summary = atlas.build_summary(rows, [], min_recurrent_frames=2)
    assert summary['top1_source_anchor_near_border'] == 0
    assert summary['top1_decoded_near_border'] == 1
    assert summary['top1_decoded_on_boundary'] == 1
    assert summary['top1_anchor_to_decoded_shift_ratio_median'] == 0.25
