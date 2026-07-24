from types import SimpleNamespace

import pytest
import torch

from crane_project.tools import p3_p4_multimodal_knn_audit as audit


def _args(**overrides):
    values = dict(
        seed=0,
        levels=[0, 1],
        physical_radius_px=16.0,
        min_target_gaussian=0.1,
        source_folds=5,
        neighbors=5,
        positive_quantile=0.1,
        negative_quantile=0.9,
        min_fold_votes=4,
        max_source_samples=0,
        riou_thr=0.5,
        false_iou_thr=0.1,
        source_min_accuracy=0.8,
        target_min_rescues=26,
        target_start=137,
        target_end=169,
        source_seq='real_seq07',
        config='crane_project/configs/crane_symeood_k1_brightaug.py',
        detector_checkpoint=(
            'work_dirs/crane_symeood_k1_brightaug/epoch_20.pth'),
        allow_noncanonical=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_canonical_protocol_locks_fixed_five_neighbor_audit():
    assert audit.validate_args(_args()) is True
    with pytest.raises(ValueError, match='include P3'):
        audit.validate_args(_args(levels=[1], allow_noncanonical=True))
    with pytest.raises(ValueError, match='Canonical multimodal-audit'):
        audit.validate_args(_args(neighbors=3))
    assert audit.validate_args(_args(
        neighbors=3, allow_noncanonical=True)) is False


def _multimodal_vectors():
    positive = torch.tensor([
        [1.0, 0.10], [1.0, 0.05], [1.0, 0.00],
        [1.0, -0.05], [1.0, -0.10],
        [-1.0, 0.10], [-1.0, 0.05], [-1.0, 0.00],
        [-1.0, -0.05], [-1.0, -0.10],
    ])
    negative = torch.tensor([
        [0.10, 1.0], [0.05, 1.0], [0.00, 1.0],
        [-0.05, 1.0], [-0.10, 1.0],
        [0.10, -1.0], [0.05, -1.0], [0.00, -1.0],
        [-0.05, -1.0], [-0.10, -1.0],
    ])
    return positive, negative


def test_knn_bank_recovers_two_opposite_positive_modes():
    positive, negative = _multimodal_vectors()
    bank = audit.build_knn_bank(positive, negative, neighbors=5)

    left_mode = audit.knn_scores(torch.tensor([-1.0, 0.0]), bank)
    right_mode = audit.knn_scores(torch.tensor([1.0, 0.0]), bank)
    background = audit.knn_scores(torch.tensor([0.0, 1.0]), bank)

    assert left_mode['cosine_preference_positive'] > 0.0
    assert left_mode['whitened_preference_positive'] > 0.0
    assert right_mode['cosine_preference_positive'] > 0.0
    assert right_mode['whitened_preference_positive'] > 0.0
    assert background['cosine_preference_positive'] < 0.0
    assert background['whitened_preference_positive'] < 0.0


def _simple_fold_models():
    positive = torch.stack([
        torch.ones(18),
        torch.ones(18) * 1.02,
        torch.ones(18) * 0.98,
        torch.ones(18) * 1.04,
        torch.ones(18) * 0.96,
    ])
    negative = -positive
    bank = audit.build_knn_bank(positive, negative, neighbors=5)
    return [dict(
        fold_id=index, bank=bank,
        cosine_preference_threshold=0.0,
        whitened_preference_threshold=0.0)
        for index in range(5)]


def test_multimodal_neighborhood_rescue_uses_source_grid_and_votes():
    feature = -torch.ones((1, 2, 9, 9))
    feature[0, :, 3:6, 4:7] = 1.0
    result = audit.multimodal_neighborhood_rescue(
        feature, base_x=36.0, base_y=36.0,
        false_x=4.0, false_y=4.0,
        img_shape=(72, 72, 3), stride=8.0,
        physical_radius_px=16.0,
        fold_models=_simple_fold_models(), min_fold_votes=4)
    assert result['location_count'] == 25
    assert result['rescued'] is True
    assert result['best_location']['fold_votes'] == 5
    assert result['best_location']['offset_cells'] == [0, 1]


def test_source_calibrated_threshold_rejects_weak_zero_margin_vote():
    feature = -torch.ones((1, 2, 9, 9))
    feature[0, :, 3:6, 4:7] = 1.0
    fold_models = _simple_fold_models()
    for model in fold_models:
        model['cosine_preference_threshold'] = 2.1
        model['whitened_preference_threshold'] = 2.1
    result = audit.multimodal_neighborhood_rescue(
        feature, base_x=36.0, base_y=36.0,
        false_x=4.0, false_y=4.0,
        img_shape=(72, 72, 3), stride=8.0,
        physical_radius_px=16.0,
        fold_models=fold_models, min_fold_votes=4)
    assert result['rescued'] is False
    assert result['best_location']['fold_votes'] == 0
    assert result['best_location']['mean_decision_margin'] < 0.0


def test_calibrated_threshold_uses_positive_floor_and_negative_ceiling():
    result = audit.calibrated_threshold(
        positive_values=[0.4, 0.5, 0.6, 0.7],
        negative_values=[-0.2, -0.1, 0.0, 0.65],
        positive_quantile=0.1, negative_quantile=0.9)
    assert result['positive_floor'] == pytest.approx(0.43)
    assert result['negative_ceiling'] == pytest.approx(0.455)
    assert result['threshold'] == pytest.approx(0.455)


def test_target_summary_keeps_vote_histogram_and_calibrated_margins():
    locations = [dict(
        fold_votes=3, rescues=False, is_mapped_center=True,
        mean_decision_margin=-0.2,
        mean_calibrated_cosine_margin=-0.1,
        mean_calibrated_whitened_margin=-0.2,
        source_ensemble=dict(fold_count=5))]
    row = dict(
        eligible=True,
        levels={
            '0': dict(rescued=False, best_location=locations[0],
                      locations=locations),
            '1': dict(rescued=False, best_location=locations[0],
                      locations=locations)})
    summary = audit.summarize_target([row], [0, 1])
    assert summary['levels']['0']['best_vote_histogram']['3'] == 1
    assert summary['levels']['0']['rescue_count'] == 0
    assert (summary['levels']['0']['best_decision_margin']['median']
            == pytest.approx(-0.2))


def _valid_source_summary():
    return dict(
        positive_cosine_accuracy=0.9,
        negative_cosine_accuracy=0.9,
        positive_whitened_accuracy=0.9,
        negative_whitened_accuracy=0.9)


def _target_summary(p3, p4):
    return dict(
        eligible_count=31,
        levels={'0': dict(rescue_count=p3),
                '1': dict(rescue_count=p4)})


def test_gate_closes_fpn_only_after_valid_multimodal_controls_fail():
    result = audit.make_gate(
        {0: _valid_source_summary(), 1: _valid_source_summary()},
        _target_summary(0, 0), [164, 167], _args())
    assert result['decision'] == 'CLOSE_ORDINARY_FPN_REPRESENTATION'


def test_gate_authorizes_only_the_level_that_multimodal_bank_rescues():
    p3 = audit.make_gate(
        {0: _valid_source_summary(), 1: _valid_source_summary()},
        _target_summary(27, 28), [164, 167], _args())
    assert p3['decision'] == 'MULTIMODAL_P3_RESCUE'

    p4 = audit.make_gate(
        {0: _valid_source_summary(), 1: _valid_source_summary()},
        _target_summary(0, 28), [164, 167], _args())
    assert p4['decision'] == 'MULTIMODAL_P4_RESCUE'


def test_gate_remains_inconclusive_when_source_knn_control_fails():
    invalid = _valid_source_summary()
    invalid['positive_cosine_accuracy'] = 0.79
    result = audit.make_gate(
        {0: invalid, 1: _valid_source_summary()},
        _target_summary(0, 0), [164, 167], _args())
    assert result['decision'] == 'P3_MULTIMODAL_INCONCLUSIVE'
