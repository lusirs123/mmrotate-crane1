from types import SimpleNamespace

import pytest
import torch

from crane_project.tools import frozen_p3_feature_alignment_audit as alignment
from crane_project.tools import p3_p4_neighborhood_rescue_audit as audit


def _args(**overrides):
    values = dict(
        seed=0,
        levels=[0, 1],
        physical_radius_px=16.0,
        min_target_gaussian=0.1,
        source_folds=5,
        min_fold_votes=4,
        positive_quantile=0.1,
        max_source_samples=0,
        riou_thr=0.5,
        false_iou_thr=0.1,
        source_min_accuracy=0.8,
        target_min_rescues=26,
        target_start=137,
        target_end=169,
        source_seq='real_seq07',
        config='crane_project/configs/crane_symeood_k1_brightaug.py',
        detector_checkpoint='work_dirs/crane_symeood_k1_brightaug/epoch_20.pth',
        allow_noncanonical=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_canonical_protocol_locks_levels_radius_and_epoch20():
    assert audit.validate_args(_args()) is True
    with pytest.raises(ValueError, match='Canonical neighborhood-audit'):
        audit.validate_args(_args(levels=[0]))
    with pytest.raises(ValueError, match='Canonical neighborhood-audit'):
        audit.validate_args(_args(physical_radius_px=24.0))
    assert audit.validate_args(_args(
        physical_radius_px=24.0, allow_noncanonical=True)) is False


def test_fixed_physical_radius_maps_to_p3_5x5_and_p4_3x3():
    assert audit.radius_in_cells(16.0, 8.0) == 2
    assert audit.radius_in_cells(16.0, 16.0) == 1
    with pytest.raises(ValueError, match='integer number of cells'):
        audit.radius_in_cells(15.0, 8.0)


def test_physical_mapping_and_valid_neighborhood_do_not_use_decoded_center():
    row, col = audit.physical_to_grid(
        x=36.0, y=36.0, stride=8.0,
        height=8, width=8, img_shape=(64, 64, 3))
    assert (row, col) == (4, 4)
    full = audit.enumerate_neighborhood(
        row, col, radius_cells=2, height=8, width=8,
        img_shape=(64, 64, 3), stride=8.0)
    assert len(full) == 25
    edge = audit.enumerate_neighborhood(
        0, 0, radius_cells=2, height=8, width=8,
        img_shape=(64, 64, 3), stride=8.0)
    assert len(edge) == 9


def _simple_prototypes():
    positive = torch.stack([
        torch.ones(9), torch.ones(9) * 1.1,
        torch.ones(9) * 0.9])
    negative = -positive
    return alignment.build_source_prototypes(positive, negative)


def _simple_ensemble():
    prototypes = _simple_prototypes()
    return [dict(
        fold_id=index, prototypes=prototypes,
        cosine_positive_threshold=0.0,
        whitened_positive_threshold=0.0)
        for index in range(5)]


def test_neighborhood_rescue_finds_positive_patch_with_fixed_search():
    feature = -torch.ones((1, 1, 9, 9))
    # A source-positive-like 3x3 patch is one cell to the right of the mapped
    # center. The matched false remains in the top-left background.
    feature[0, 0, 3:6, 4:7] = 1.0
    result = audit.neighborhood_rescue(
        feature, base_x=36.0, base_y=36.0,
        false_x=4.0, false_y=4.0,
        img_shape=(72, 72, 3), stride=8.0,
        physical_radius_px=16.0,
        fold_models=_simple_ensemble(), min_fold_votes=4)
    assert result['radius_cells'] == 2
    assert result['location_count'] == 25
    assert result['rescued'] is True
    assert result['best_location']['rescues'] is True
    assert result['best_location']['offset_cells'] == [0, 1]


def test_neighborhood_rescue_rejects_source_like_background_outside_gt():
    feature = -torch.ones((1, 1, 9, 9))
    feature[0, 0, 3:6, 5:8] = 1.0
    target_heatmap = torch.zeros((1, 9, 9))
    # The source-like patch center is outside the declared target heatmap.
    target_heatmap[0, 4, 4] = 1.0
    result = audit.neighborhood_rescue(
        feature, base_x=36.0, base_y=36.0,
        false_x=4.0, false_y=4.0,
        img_shape=(72, 72, 3), stride=8.0,
        physical_radius_px=16.0,
        fold_models=_simple_ensemble(), min_fold_votes=4,
        target_heatmap=target_heatmap,
        min_target_gaussian=0.1)
    assert result['location_count'] == 1
    assert result['locations'][0]['is_mapped_center'] is True
    assert result['rescued'] is False


def _valid_source_summary():
    return dict(
        positive_cosine_accuracy=0.95,
        negative_cosine_accuracy=0.95,
        positive_whitened_accuracy=0.90,
        negative_whitened_accuracy=0.90)


def _target_summary(p3, p4):
    return dict(
        eligible_count=31,
        levels={
            '0': dict(rescue_count=p3),
            '1': dict(rescue_count=p4),
        })


def test_gate_prefers_smaller_p3_sampling_when_both_levels_rescue():
    result = audit.make_gate(
        {0: _valid_source_summary(), 1: _valid_source_summary()},
        _target_summary(27, 29), [164, 167], _args())
    assert result['decision'] == 'AUTHORIZE_P3_LOCAL_SAMPLING'
    assert result['p3_pass'] is True
    assert result['p4_pass'] is True


def test_gate_routes_to_p4_only_when_p3_does_not_pass():
    result = audit.make_gate(
        {0: _valid_source_summary(), 1: _valid_source_summary()},
        _target_summary(12, 28), [164, 167], _args())
    assert result['decision'] == 'AUTHORIZE_P4_CROSS_LEVEL_OBJECTNESS'
    assert result['p3_pass'] is False
    assert result['p4_pass'] is True


def test_gate_closes_fpn_route_when_neither_level_rescues():
    result = audit.make_gate(
        {0: _valid_source_summary(), 1: _valid_source_summary()},
        _target_summary(8, 11), [164, 167], _args())
    assert result['decision'] == 'NO_FPN_RESCUE'
    assert result['p3_pass'] is False
    assert result['p4_pass'] is False


def test_gate_marks_p3_inconclusive_when_p3_source_calibration_fails():
    bad = _valid_source_summary()
    bad['positive_cosine_accuracy'] = 0.5
    result = audit.make_gate(
        {0: bad, 1: _valid_source_summary()},
        _target_summary(28, 10), [164, 167], _args())
    assert result['decision'] == 'P3_INCONCLUSIVE_SOURCE_CONTROL'


def test_gate_keeps_valid_p3_failure_when_only_p4_calibration_fails():
    bad_p4 = _valid_source_summary()
    bad_p4['negative_whitened_accuracy'] = 0.75
    result = audit.make_gate(
        {0: _valid_source_summary(), 1: bad_p4},
        _target_summary(0, 0), [164, 167], _args())
    assert result['decision'] == 'P3_NO_RESCUE_P4_INCONCLUSIVE'


def test_contiguous_folds_keep_each_time_block_together():
    fold_ids = audit.contiguous_fold_ids(10, 5)
    assert fold_ids == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert audit.quantile([0.0, 1.0, 2.0, 3.0], 0.1) == pytest.approx(0.3)
