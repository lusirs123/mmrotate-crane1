from types import SimpleNamespace

import pytest
import torch

from crane_project.tools import anchor_training_coverage_probe as probe


def _args(**overrides):
    values = dict(
        seed=0, source_indexes=[0, 1], max_samples_per_source=10,
        gradient_samples_per_source=2, riou_thr=0.5,
        assignment_phases=['warmup', 'steady'])
    values.update(overrides)
    return SimpleNamespace(**values)


def test_validation_enforces_read_only_probe_bounds():
    assert probe.validate_args(_args()) == [0, 1]
    with pytest.raises(ValueError, match='gradient-samples'):
        probe.validate_args(_args(gradient_samples_per_source=11))
    with pytest.raises(ValueError, match='non-negative'):
        probe.validate_args(_args(source_indexes=[-1]))
    with pytest.raises(ValueError, match='assignment-phases'):
        probe.validate_args(_args(assignment_phases=['warmup', 'warmup']))


def test_assignment_phase_is_explicit_and_reaches_expected_topk():
    assigner = SimpleNamespace(
        _local_call_count=123, o2m=False,
        o2m_warmup_iters=2000, o2m_topk=9, topk=1)
    warmup = probe.configure_assignment_phase(assigner, 'warmup')
    assert assigner._local_call_count == 0
    assert warmup['expected_effective_topk'] == 9
    steady = probe.configure_assignment_phase(assigner, 'steady')
    assert steady['synthetic_current_iter'] == 4001
    assert assigner._local_call_count == 8002
    assert steady['expected_effective_topk'] == 1


def test_positive_anchor_counts_uses_location_major_anchor_order():
    # Positive indices 0/5/7 map to anchors 0/2/1 for A=3.
    labels = [torch.tensor([0, 1, 1, 1, 1, 0, 1, 0])]
    result = probe.positive_anchor_counts(
        labels, num_classes=1, num_anchors=3)
    assert result['total'] == 3
    assert result['by_anchor'] == [1, 1, 1]


def test_aggregate_source_separates_assignment_geometry_and_gradient():
    row = dict(
        gt_count=1,
        gt_geometry=[dict(symmetric_aspect=2.0)],
        positive_assignments=dict(by_anchor=[1, 0, 0]),
        per_anchor_geometry=[
            dict(anchor_id=0,
                 dense_best_geometry=dict(riou=0.8),
                 best_usable_by_score=dict(score=0.9)),
            dict(anchor_id=1,
                 dense_best_geometry=dict(riou=0.6),
                 best_usable_by_score=dict(score=0.1)),
            dict(anchor_id=2,
                 dense_best_geometry=dict(riou=0.2),
                 best_usable_by_score=None)],
        classification_gradient=dict(
            weight_grad_norms=[3.0, 2.0, 1.0],
            bias_grad_abs=[0.3, 0.2, 0.1]))
    result = probe.aggregate_source([row], 3)
    assert result['positive_assignment_fraction_by_anchor'] == [1.0, 0.0, 0.0]
    assert result['usable_frames_by_anchor'] == [1, 1, 0]
    assert result['dense_best_riou_median_by_anchor'] == pytest.approx(
        [0.8, 0.6, 0.2])
    assert result['weight_grad_norm_mean_by_anchor'] == pytest.approx(
        [3.0, 2.0, 1.0])


def test_expand_checkpoint_paths_deduplicates_current(tmp_path):
    first = tmp_path / 'epoch_2.pth'
    second = tmp_path / 'epoch_4.pth'
    first.touch()
    second.touch()
    paths = probe.expand_checkpoint_paths(
        str(second), [str(tmp_path / 'epoch_*.pth')])
    assert paths == [str(first), str(second)]
