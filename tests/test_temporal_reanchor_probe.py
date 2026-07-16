from types import SimpleNamespace

import numpy as np
import pytest
import torch

from crane_project.tools import temporal_reanchor_probe as probe


def ranking_args():
    return SimpleNamespace(
        max_radius_px=100.0,
        min_radius_px=1.0,
        radius_mul=5.0,
        score_floor=0.0,
        max_cands=0,
        w_center=1.0,
        w_size=0.0,
        w_angle=0.0,
        w_score=0.0,
        angle_norm_deg=45.0,
    )


def test_local_oracle_is_distinct_from_heuristic_selection(monkeypatch):
    candidates = (
        torch.tensor([
            [0.0, 0.0, 10.0, 4.0, 0.0],
            [20.0, 0.0, 10.0, 4.0, 0.0],
        ]),
        torch.tensor([0.9, 0.01]),
        torch.tensor([0, 1]),
    )
    pred = np.array([0.0, 0.0, 10.0, 4.0, 0.0], dtype=np.float32)
    gt = np.array([20.0, 0.0, 10.0, 4.0, 0.0], dtype=np.float32)
    monkeypatch.setattr(
        probe, 'rotated_ious',
        lambda boxes, gt_box: boxes.new_tensor([0.1, 0.8]))

    ranking = probe.build_temporal_ranking(
        candidates, pred, ranking_args())
    selected = probe.select_reanchor_candidate(
        candidates, pred, ranking_args(), ranking=ranking)
    diagnostics = probe.candidate_oracle_diagnostics(
        candidates, gt, ranking, riou_thr=0.5)

    assert selected['index'] == 0
    assert diagnostics['dense_oracle_hit'] is True
    assert diagnostics['local_oracle_hit'] is True
    assert diagnostics['local_best_riou'] == pytest.approx(0.8)
    assert diagnostics['local_best_temporal_rank'] == 2


def test_summary_keeps_oracles_and_heuristic_separate():
    rows = [
        dict(frame=1, final_hit=False, after_hit=False,
             dense_oracle_hit=True, local_oracle_hit=True,
             heuristic_or_hit=False, bidir_hit=False,
             bidir_consistent=False),
        dict(frame=2, final_hit=False, after_hit=False,
             dense_oracle_hit=True, local_oracle_hit=False,
             heuristic_or_hit=False, bidir_hit=False,
             bidir_consistent=False),
        dict(frame=3, final_hit=False, after_hit=False,
             dense_oracle_hit=False, local_oracle_hit=False,
             heuristic_or_hit=False, bidir_hit=False,
             bidir_consistent=False),
    ]

    metrics = probe.build_summary_metrics(rows)

    assert metrics['dense_oracle']['hits'] == 2
    assert metrics['dense_oracle']['longest_miss']['length'] == 1
    assert metrics['local_oracle']['hits'] == 1
    assert metrics['local_oracle']['longest_miss']['length'] == 2
    assert metrics['heuristic_or']['hits'] == 0
    assert metrics['heuristic_or']['longest_miss']['length'] == 3


def test_teacher_force_updates_only_the_next_frame(monkeypatch):
    candidates = (
        torch.tensor([
            [0.0, 0.0, 10.0, 4.0, 0.0],
            [20.0, 0.0, 10.0, 4.0, 0.0],
        ]),
        torch.tensor([0.1, 0.1]),
        torch.tensor([0, 0]),
    )

    def decode(*call_args):
        fid = int(call_args[7])
        return candidates, dict(
            cx=float(10 * fid), cy=0.0, w=10.0, h=4.0, angle=0.0), {}

    monkeypatch.setattr(probe, 'decode_frame_candidates', decode)
    monkeypatch.setattr(probe, 'best_final_box', lambda *args: None)
    monkeypatch.setattr(
        probe, 'eval_hit',
        lambda *args: dict(
            riou=0.0, center_dist=0.0, gamma_error_deg=0.0,
            riou_hit=False, center_hit=True, hit=False))
    monkeypatch.setattr(
        probe, 'candidate_oracle_diagnostics',
        lambda *args: dict(
            dense_best_riou=0.0, dense_oracle_hit=False,
            dense_best_score=0.1, dense_best_score_rank=1,
            dense_best_level=0, local_best_riou=0.0,
            local_oracle_hit=False, local_usable_count=0,
            local_best_score=0.1, local_best_level=0,
            local_best_dist_to_pred=0.0,
            local_best_temporal_rank=1))
    args = ranking_args()
    args.data_root = 'unused'
    args.pred_dir = 'unused'
    args.split = 'test'
    args.seq = 'real_seq02'
    args.gpu = 0
    args.head = 'main'
    args.seed_mode = 'oracle-valid'
    args.riou_thr = 0.5
    args.center_thr = 25.0
    args.teacher_force_gt = True
    args.update_mode = 'oracle-hit'
    args.motion = 'linear'

    rows = probe.run_temporal_pass(
        None, None, None, False, args, [1, 2],
        [dict(fid=0, box=[0.0, 0.0, 10.0, 4.0, 0.0], source='anchor')],
        'fwd')

    assert rows[1]['pred_cx'] == 0.0
    assert rows[2]['pred_cx'] == 20.0
    assert rows[1]['update_source'] == 'teacher_gt'
    assert rows[2]['update_source'] == 'teacher_gt'


def test_joint_rank_uses_both_directional_predictions():
    candidates = (
        torch.tensor([
            [0.0, 0.0, 10.0, 4.0, 0.0],
            [4.0, 0.0, 10.0, 4.0, 0.0],
            [9.0, 0.0, 10.0, 4.0, 0.0],
            [20.0, 0.0, 10.0, 4.0, 0.0],
        ]),
        torch.ones(4),
        torch.zeros(4, dtype=torch.long),
    )
    args = ranking_args()
    fwd_pred = np.array([0.0, 0.0, 10.0, 4.0, 0.0], dtype=np.float32)
    bwd_pred = np.array([20.0, 0.0, 10.0, 4.0, 0.0], dtype=np.float32)
    fwd = probe.build_temporal_ranking(candidates, fwd_pred, args)
    bwd = probe.build_temporal_ranking(candidates, bwd_pred, args)

    selected = probe.select_joint_rank_candidate(
        candidates, fwd_pred, bwd_pred, fwd, bwd, args)

    assert selected['found'] is True
    assert selected['candidate_count'] == 4
    assert selected['index'] in (1, 2)


def test_bidir_time_weights_follow_frame_position():
    assert probe.bidir_time_weights(136, 136, 170) == pytest.approx(
        (1.0, 0.0, 0.0))
    assert probe.bidir_time_weights(153, 136, 170) == pytest.approx(
        (0.5, 0.5, 0.5))
    assert probe.bidir_time_weights(170, 136, 170) == pytest.approx(
        (0.0, 1.0, 1.0))

    with pytest.raises(ValueError, match='right_anchor'):
        probe.bidir_time_weights(136, 170, 136)


def test_time_weighted_rank_moves_selection_between_anchors():
    candidates = (
        torch.tensor([
            [0.0, 0.0, 10.0, 4.0, 0.0],
            [4.0, 0.0, 10.0, 4.0, 0.0],
            [9.0, 0.0, 10.0, 4.0, 0.0],
            [20.0, 0.0, 10.0, 4.0, 0.0],
        ]),
        torch.ones(4),
        torch.zeros(4, dtype=torch.long),
    )
    args = ranking_args()
    fwd_pred = np.array(
        [0.0, 0.0, 10.0, 4.0, 0.0], dtype=np.float32)
    bwd_pred = np.array(
        [20.0, 0.0, 10.0, 4.0, 0.0], dtype=np.float32)
    fwd = probe.build_temporal_ranking(candidates, fwd_pred, args)
    bwd = probe.build_temporal_ranking(candidates, bwd_pred, args)

    near_left = probe.select_joint_rank_candidate(
        candidates, fwd_pred, bwd_pred, fwd, bwd, args,
        directional_weights=(0.9, 0.1))
    near_right = probe.select_joint_rank_candidate(
        candidates, fwd_pred, bwd_pred, fwd, bwd, args,
        directional_weights=(0.1, 0.9))

    assert near_left['index'] == 0
    assert near_right['index'] == 3
    assert near_left['fwd_weight'] == pytest.approx(0.9)
    assert near_right['bwd_weight'] == pytest.approx(0.9)


def test_merge_bidir_rows_threads_time_weights(monkeypatch):
    candidates = (
        torch.tensor([
            [0.0, 0.0, 10.0, 4.0, 0.0],
            [20.0, 0.0, 10.0, 4.0, 0.0],
        ]),
        torch.ones(2),
        torch.zeros(2, dtype=torch.long),
    )
    rank_args = ranking_args()
    fwd_pred = np.array(
        [0.0, 0.0, 10.0, 4.0, 0.0], dtype=np.float32)
    bwd_pred = np.array(
        [20.0, 0.0, 10.0, 4.0, 0.0], dtype=np.float32)
    fwd_ranking = probe.build_temporal_ranking(
        candidates, fwd_pred, rank_args)
    bwd_ranking = probe.build_temporal_ranking(
        candidates, bwd_pred, rank_args)

    def pass_row(pred, ranking, selected_box):
        return dict(
            final_hit=False,
            reanchor_found=True,
            reanchor_box=selected_box,
            reanchor_riou=0.0,
            reanchor_hit=False,
            local_oracle_hit=True,
            local_best_riou=0.8,
            _cands=candidates,
            _pred_box=pred,
            _ranking=ranking,
            _gt_box=fwd_pred,
        )

    monkeypatch.setattr(
        probe, 'eval_hit',
        lambda box, *args: dict(
            riou=0.8 if box[0] == 0.0 else 0.0,
            center_dist=0.0, gamma_error_deg=0.0,
            riou_hit=box[0] == 0.0, center_hit=box[0] == 0.0,
            hit=box[0] == 0.0))
    args = rank_args
    args.bidir_select = 'time-weighted-rank'
    args.left_anchor = 136
    args.right_anchor = 170
    args.start = 137
    args.end = 169
    args.diverge_center_px = 25.0
    args.diverge_size_log = 0.5
    args.riou_thr = 0.5
    args.center_thr = 25.0
    args.head = 'main'
    args.teacher_force_gt = False
    fwd_rows = {137: pass_row(fwd_pred, fwd_ranking, [0.0] * 5)}
    bwd_rows = {137: pass_row(bwd_pred, bwd_ranking, [20.0] * 5)}

    row = probe.merge_bidir_rows(
        [137], fwd_rows, bwd_rows, args)[0]

    assert row['selection_mode'] == 'time-weighted-rank'
    assert row['joint_time_alpha'] == pytest.approx(1.0 / 34.0)
    assert row['joint_fwd_weight'] == pytest.approx(33.0 / 34.0)
    assert row['joint_bwd_weight'] == pytest.approx(1.0 / 34.0)
    assert row['selected_riou'] == pytest.approx(0.8)
    assert row['joint_hit'] is True
