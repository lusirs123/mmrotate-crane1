import numpy as np
import pytest
import torch

from crane_project.tools import diagnose_v5_source_val_candidate_flow as probe


def test_inference_config_reads_detector_level_test_cfg():
    model_cfg = {'bbox_head': {'type': 'SymEOODHead'},
                 'test_cfg': {'score_thr': 0.05, 'max_per_img': 1}}
    assert probe.require_top1_test_cfg(model_cfg) == model_cfg['test_cfg']
    with pytest.raises(ValueError, match='postprocessing'):
        probe.require_top1_test_cfg({'bbox_head': {'type': 'SymEOODHead'}})


@pytest.mark.parametrize('ious,scores,expected', [
    ([0.1, 0.2], [0.9, 0.8], 'no_geometric_candidate'),
    ([0.8, 0.1], [0.04, 0.03], 'score_threshold'),
    ([0.8, 0.1], [0.3, 0.9], 'top1_ranking'),
    ([0.8, 0.1], [0.9, 0.3], 'success'),
])
def test_stage_classifies_actual_threshold_and_top1(monkeypatch, ious, scores,
                                                       expected):
    monkeypatch.setattr(probe, 'best_iou', lambda boxes, gt: torch.tensor(ious))
    row = probe.stage_summary(torch.zeros((2, 5)), torch.tensor(scores),
                              torch.zeros((1, 5)), 0.05)
    assert row['stage'] == expected
    assert row['candidate_count'] == 2


def test_saved_top1_must_match_trace_and_presence():
    meta = {'scale_factor': np.array([2., 2., 2., 2.])}
    box = np.array([20., 30., 10., 8., 0.1, 0.7])
    saved = [np.array([[10., 15., 5., 4., 0.1, 0.7]])]
    probe.validate_saved_output(box, saved, meta, 'frame')
    with pytest.raises(ValueError, match='differs'):
        probe.validate_saved_output(box, [np.array([[11., 15., 5., 4., 0.1, 0.7]])], meta, 'frame')
    with pytest.raises(ValueError, match='presence'):
        probe.validate_saved_output(None, saved, meta, 'frame')


def test_empty_dense_candidates(monkeypatch):
    monkeypatch.setattr(probe, 'best_iou', lambda boxes, gt: torch.empty(0))
    row = probe.stage_summary(torch.empty((0, 5)), torch.empty(0),
                              torch.zeros((1, 5)), 0.05)
    assert row['stage'] == 'no_geometric_candidate'
    assert row['top1_index'] is None


def test_top1_index_maps_back_to_unfiltered_candidates(monkeypatch):
    monkeypatch.setattr(probe, 'best_iou',
                        lambda boxes, gt: torch.tensor([0.1, 0.8, 0.9]))
    row = probe.stage_summary(torch.zeros((3, 5)),
                              torch.tensor([0.04, 0.06, 0.80]),
                              torch.zeros((1, 5)), 0.05)
    assert row['top1_index'] == 2
    assert row['stage'] == 'success'
