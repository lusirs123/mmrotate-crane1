from types import SimpleNamespace

import pytest
import torch

from crane_project.tools import dark_signal_pathway_probe as probe


def _args(**overrides):
    values = dict(
        seed=0, split='test', seq='real_seq02',
        frames=[150, 155, 164, 167], candidate_source='main',
        riou_thr=0.5, false_iou_thr=0.1,
        signal_norm_ratio_thr=0.5)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_protocol_is_small_target_dev_reference_only_slice():
    assert probe.validate_args(_args()) == [150, 155, 164, 167]
    with pytest.raises(ValueError, match='test/real_seq02'):
        probe.validate_args(_args(split='val', seq='real_seq07'))
    with pytest.raises(ValueError, match='3-5 unique'):
        probe.validate_args(_args(frames=[150, 155]))
    with pytest.raises(ValueError, match='137..169'):
        probe.validate_args(_args(frames=[150, 155, 164, 170]))


def test_select_candidates_separates_false_and_usable_rank():
    scores = torch.tensor([0.8, 0.7, 0.2, 0.1])
    ious = torch.tensor([0.0, 0.2, 0.8, 0.6])
    false, usable = probe.select_candidates(scores, ious, 0.1, 0.5)
    assert false['index'] == 0
    assert false['rank'] == 1
    assert usable['index'] == 2
    assert usable['rank'] == 3


def test_pathway_hint_does_not_attribute_geometry_miss_to_classifier():
    hint = probe.pathway_hint(None, None, 0.5)
    assert hint['code'] == 'GEOMETRY_MISS'
    assert hint['classification_attribution_valid'] is False


def test_pathway_hint_uses_local_fpn_norm_ratio():
    usable = dict(score=0.001)
    weak = probe.pathway_hint(
        usable, dict(usable_to_false_norm_ratio=0.2), 0.5)
    retained = probe.pathway_hint(
        usable, dict(usable_to_false_norm_ratio=0.8), 0.5)
    assert weak['code'] == 'SIGNAL_ATTENUATION_SUSPECT'
    assert retained['code'] == 'HEAD_RANKING_SUSPECT'


def test_vector_comparison_reports_norm_ratio_and_cosine():
    result = probe.compare_vectors(
        torch.tensor([1.0, 0.0]), torch.tensor([2.0, 0.0]))
    assert result['cosine'] == pytest.approx(1.0)
    assert result['usable_to_false_norm_ratio'] == pytest.approx(0.5)
