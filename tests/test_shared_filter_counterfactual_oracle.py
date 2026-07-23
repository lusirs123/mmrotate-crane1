from types import SimpleNamespace

import pytest
import torch

from crane_project.tools import shared_filter_counterfactual_oracle as oracle


def _args(**overrides):
    values = dict(
        seed=0, split='test', seq='real_seq02', start=137, end=169,
        pool_size=10000, riou_thr=0.5, score_thr=0.05,
        tie_atol=1e-8,
        config='crane_project/configs/crane_symeood_k1_brightaug.py',
        checkpoint='work_dirs/crane_symeood_k1_brightaug/epoch_20.pth',
        allow_noncanonical=False)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_canonical_protocol_is_target_dev_only():
    assert oracle.validate_args(_args()) is True
    with pytest.raises(ValueError, match='Canonical oracle'):
        oracle.validate_args(_args(start=138))


class _Head:
    cls_out_channels = 1
    num_anchors = 3


def test_shared_logits_copy_anchor0_without_touching_input():
    original = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])
    shared = oracle.shared_anchor0_logits([original], _Head())[0]
    assert shared[:, :, 0, 0].tolist() == [[1.0, 1.0, 1.0]]
    assert original[:, :, 0, 0].tolist() == [[1.0, 2.0, 3.0]]


def _layout(count):
    return [dict(anchor_id=index % 3, level=0) for index in range(count)]


def test_candidate_metrics_reports_tie_rank_interval():
    scores = torch.tensor([0.9, 0.9, 0.9, 0.2])
    ious = torch.tensor([0.1, 0.2, 0.8, 0.0])
    result = oracle.candidate_metrics(
        scores, ious, _layout(4), pool_size=2,
        riou_thr=0.5, score_thr=0.05, tie_atol=1e-8)
    assert result['top1_tie']['count'] == 3
    assert result['top1_tie']['hit'] is True
    assert result['usable']['tie_best_rank'] == 1
    assert result['usable']['tie_worst_rank'] == 3
    assert result['pool']['tie_expanded_size'] == 3
    assert result['pool']['tie_expanded_hit'] is True


def test_summary_separates_raw_and_tie_aware_top1():
    metrics = oracle.candidate_metrics(
        torch.tensor([0.9, 0.9, 0.9]),
        torch.tensor([0.0, 0.0, 0.8]), _layout(3),
        pool_size=2, riou_thr=0.5, score_thr=0.05, tie_atol=1e-8)
    rows = [dict(frame=137, modes=dict(
        original=metrics, shared_anchor0=metrics))]
    summary = oracle.summarize(rows, 'shared_anchor0', 0.5)
    assert summary['top1_hits'] == 0
    assert summary['tie_aware_top1_hits'] == 1
    assert summary['top1_miss_run'] == 1
    assert summary['tie_aware_top1_miss_run'] == 0
