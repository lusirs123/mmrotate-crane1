import argparse
import math

import numpy as np
import pytest
import torch

from crane_project.tools import dino_teacher_source_roi_head_probe as roi_probe
from crane_project.tools import dino_teacher_topk_temporal_beam_audit as beam


def _args(**overrides):
    values = dict(
        seed=0, detector_candidate_limit=10000, beam_size=100,
        boundary_top_m=10, roi_chunk_size=16,
        min_roi_in_bounds=0.9, target_min_wins=26, max_mcml=5)
    values.update(overrides)
    return argparse.Namespace(**values)


def _candidate(rank, x, y=0.0, w=10.0, h=10.0, angle=0.0):
    return dict(
        dino_rank=rank,
        obb_original=[float(x), float(y), float(w), float(h), float(angle)])


def _model(q99=10.0, weight=1.0):
    return dict(
        center=np.zeros(5, dtype=np.float64),
        scale=np.ones(5, dtype=np.float64),
        q90_cost=float(q99) * 0.5,
        q99_cost=float(q99), transition_weight=float(weight))


def test_transition_vector_wraps_le90_equivalent_angles():
    previous = [0.0, 0.0, 10.0, 10.0, math.radians(89.0)]
    current = [0.0, 0.0, 10.0, 10.0, math.radians(-89.0)]
    vector = beam.transition_vector(previous, current)
    assert abs(vector[4]) == pytest.approx(math.radians(2.0))


def test_robust_transition_model_is_source_derived_and_finite():
    transitions = np.asarray([
        [0.1 + index * 0.001, 0.0, 0.0, 0.0, 0.0]
        for index in range(20)], dtype=np.float64)
    model = beam.robust_transition_model(transitions, beam_size=100)
    assert model['count'] == 20
    assert model['q90_cost'] >= 0.0
    assert model['q99_cost'] >= model['q90_cost']
    assert math.isfinite(model['transition_weight'])
    assert model['transition_weight'] > 0.0


def test_automatic_segments_uses_source_q99_not_target_labels():
    frames = [
        dict(frame=1, candidates=[_candidate(1, 0.0)]),
        dict(frame=2, candidates=[_candidate(1, 1.0)]),
        dict(frame=3, candidates=[_candidate(1, 100.0)]),
    ]
    segments = beam.automatic_segments(
        frames, _model(q99=1.0), boundary_top_m=1)
    assert segments == [(0, 2), (2, 3)]


def test_viterbi_prefers_smooth_second_rank_path():
    frames = [
        dict(frame=1, candidates=[
            _candidate(1, 0.0), _candidate(2, 10.0)]),
        dict(frame=2, candidates=[
            _candidate(1, 100.0), _candidate(2, 11.0)]),
        dict(frame=3, candidates=[
            _candidate(1, 0.0), _candidate(2, 12.0)]),
    ]
    selected = beam.viterbi_segment(
        frames, _model(q99=100.0, weight=2.0), beam_size=100)
    assert selected == [1, 1, 1]


def test_source_decoded_positive_uses_highest_score_usable_candidate():
    scores = torch.tensor([0.9, 0.8, 0.7])
    ious = torch.tensor([0.2, 0.6, 0.9])
    layout = [dict(level=0), dict(level=1), dict(level=0)]
    selected = beam.select_source_decoded_positive(
        scores, ious, layout, riou_thr=0.5)
    assert selected == 1


def test_refinement_splits_selected_transition_above_source_q99():
    frames = [
        dict(frame=1, candidates=[
            _candidate(1, 0.0), _candidate(2, 10.0)]),
        dict(frame=2, candidates=[
            _candidate(1, 100.0), _candidate(2, 0.0)]),
    ]
    selected, segments, segment_ids, transitions, refinement = (
        beam.refined_segmented_viterbi(
            frames, _model(q99=1.0, weight=0.0),
            beam_size=100, boundary_top_m=2))
    assert selected == [0, 0]
    assert [row['start_frame'] for row in segments] == [1, 2]
    assert segment_ids == [0, 1]
    assert transitions[1]['boundary_before'] is True
    assert transitions[1]['above_q99'] is True
    assert refinement['iteration_count'] == 2
    assert refinement['boundary_frames'] == [2]


def test_longest_miss_respects_frame_discontinuities():
    rows = [
        dict(frame=1, hit=False), dict(frame=2, hit=False),
        dict(frame=4, hit=False), dict(frame=5, hit=True)]
    assert beam.longest_miss(rows, 'hit') == 2


def test_decision_requires_real_selected_hits_and_mcml():
    summary = dict(
        frame_count=33, geometry_eligible_count=31,
        geometry_misses=[164, 167],
        temporal_selected=dict(hits=26, mcml=5))
    assert beam.make_decision(summary, _args()) == (
        'TEMPORAL_BEAM_RESTORES_ORDERING')
    summary['temporal_selected'] = dict(hits=20, mcml=4)
    assert beam.make_decision(summary, _args()) == (
        'TEMPORAL_BEAM_REDUCES_MCML_ONLY')


def test_load_frozen_roi_head_strictly_restores_checkpoint(tmp_path):
    head = roi_probe.TwoFCObjectnessHead(2, 2, 4)
    path = tmp_path / 'head.pth'
    torch.save(dict(
        state_dict=head.state_dict(), channels=2,
        pool_resolution=2, hidden_dim=4, source_only=True), path)
    loaded, metadata = beam.load_frozen_roi_head(
        str(path), torch.device('cpu'))
    assert metadata['channels'] == 2
    assert loaded.training is False
    assert all(not parameter.requires_grad
               for parameter in loaded.parameters())


def test_validate_args_rejects_beam_larger_than_candidate_pool():
    with pytest.raises(ValueError, match='cover the beam'):
        beam.validate_args(_args(
            detector_candidate_limit=50, beam_size=100))
