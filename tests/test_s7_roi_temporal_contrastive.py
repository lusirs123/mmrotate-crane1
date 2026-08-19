import torch
import pytest
from types import SimpleNamespace

from crane_project.tools import (
    dino_teacher_s7_roi_temporal_contrastive_train as protocol34,
    dino_teacher_s7_roi_temporal_counterfactual_audit as protocol35,
)
from crane_project.tools.dino_teacher_rotated_labeller import (
    roi_temporal_holdout_selection_key,
    roi_temporal_representation_gate,
    summarize_roi_temporal_state_propagation,
)
from crane_project.utils.s7_temporal_association import (
    CausalRoiTemporalAdapterSelector,
    RoiTemporalContrastiveAdapter,
    roi_temporal_info_nce_loss,
)


def _identity_adapter(dim=4):
    adapter = RoiTemporalContrastiveAdapter(dim, dim, dim)
    adapter.projector = torch.nn.Identity()
    return adapter


def test_roi_temporal_adapter_outputs_normalized_vectors():
    adapter = RoiTemporalContrastiveAdapter(8, 4, 3)
    output = adapter(torch.randn(5, 8))
    assert output.shape == (5, 3)
    assert torch.allclose(output.norm(dim=1), torch.ones(5), atol=1e-5)


def test_roi_temporal_info_nce_prefers_aligned_positive():
    adapter = _identity_adapter()
    anchors = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    positives = anchors.clone()
    negatives = torch.tensor([[[0.0, 1.0, 0.0, 0.0],
                               [0.0, 0.0, 1.0, 0.0]]])
    good = roi_temporal_info_nce_loss(
        adapter, anchors, positives, negatives)
    bad = roi_temporal_info_nce_loss(
        adapter, anchors, negatives[:, 0],
        torch.cat((positives[:, None], negatives[:, 1:]), dim=1))
    assert float(good) < float(bad)


def test_roi_temporal_selector_resets_and_promotes_only_by_margin():
    adapter = _identity_adapter()
    selector = CausalRoiTemporalAdapterSelector(
        adapter, max_candidates=3, promotion_margin=0.05,
        motion_weight=0.0)
    detections = torch.tensor([
        [0.0, 0.0, 10.0, 10.0, 0.0, 0.9],
        [1.0, 0.0, 10.0, 10.0, 0.0, 0.8],
        [2.0, 0.0, 10.0, 10.0, 0.0, 0.7],
    ])
    source_ids = torch.tensor([0, 1, 1])
    first_embeddings = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    first = selector.select(
        detections, first_embeddings, source_ids, 'source', 0)
    assert first['selected_index'] == 0
    assert first['promotion'] is False

    second_embeddings = torch.tensor([
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    second = selector.select(
        detections, second_embeddings, source_ids, 'source', 1)
    assert second['selected_index'] == 1
    assert second['promotion'] is True
    assert second['order'].tolist() == [1, 0, 2]

    gap = selector.select(
        detections, second_embeddings, source_ids, 'source', 3)
    assert gap['selected_index'] == 0
    assert gap['reason'] == 'native_fallback_after_reset'


def test_roi_temporal_selector_does_not_select_invalid_supplement():
    adapter = _identity_adapter()
    selector = CausalRoiTemporalAdapterSelector(
        adapter, promotion_margin=0.0, motion_weight=0.0)
    detections = torch.tensor([
        [0.0, 0.0, 10.0, 10.0, 0.0, 0.9],
        [1.0, 0.0, 10.0, 10.0, 0.0, 0.8],
    ])
    embeddings = torch.tensor([
        [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    source_ids = torch.tensor([0, 1])
    selector.select(detections, embeddings, source_ids, 's', 0)
    result = selector.select(
        detections, embeddings, source_ids, 's', 1,
        valid_mask=torch.tensor([True, False]))
    assert result['selected_index'] == 0
    assert result['promotion'] is False


def test_roi_temporal_selector_keeps_native_plus_all_32_s7_candidates():
    adapter = _identity_adapter()
    selector = CausalRoiTemporalAdapterSelector(
        adapter, max_candidates=32, promotion_margin=0.05,
        motion_weight=0.0)
    detections = torch.zeros((33, 6), dtype=torch.float32)
    detections[:, 2:4] = 10.0
    detections[:, 5] = torch.linspace(0.9, 0.5, 33)
    source_ids = torch.cat((torch.zeros(1, dtype=torch.long),
                            torch.ones(32, dtype=torch.long)))
    first_embeddings = torch.zeros((33, 4), dtype=torch.float32)
    first_embeddings[0, 0] = 1.0
    selector.select(
        detections, first_embeddings, source_ids, 'source', 0)

    second_embeddings = torch.zeros((33, 4), dtype=torch.float32)
    second_embeddings[:, 1] = 1.0
    second_embeddings[-1] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    result = selector.select(
        detections, second_embeddings, source_ids, 'source', 1)
    assert result['selected_index'] == 32
    assert result['promotion'] is True
    assert result['order'][0].item() == 32


def test_roi_temporal_native_anchor_does_not_commit_s7_output():
    adapter = _identity_adapter()
    selector = CausalRoiTemporalAdapterSelector(
        adapter, max_candidates=2, promotion_margin=0.05,
        motion_weight=0.0, state_update_policy='native')
    detections = torch.tensor([
        [0.0, 0.0, 10.0, 10.0, 0.0, 0.9],
        [1.0, 0.0, 10.0, 10.0, 0.0, 0.8],
    ])
    source_ids = torch.tensor([0, 1])
    selector.select(
        detections, torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0, 0.0]]),
        source_ids, 'source', 0)
    promoted = selector.select(
        detections, torch.tensor([[0.0, 1.0, 0.0, 0.0],
                                  [1.0, 0.0, 0.0, 0.0]]),
        source_ids, 'source', 1)
    assert promoted['selected_source'] == 'supplement_s7'
    assert promoted['state_update_source'] == 'native_s14'
    # The third frame still compares against frame-1 native [0, 1], proving
    # that the promoted S7 [1, 0] was not recursively committed.
    third = selector.select(
        detections, torch.tensor([[0.0, 1.0, 0.0, 0.0],
                                  [1.0, 0.0, 0.0, 0.0]]),
        source_ids, 'source', 2)
    assert third['selected_source'] == 'native_s14'
    assert third['promotion'] is False


def test_holdout_selection_cannot_trade_temporal_for_cross_view():
    cross_view_heavy = {
        'overall': {'success_fraction': 0.95, 'margin_median': 0.30},
        'temporal': {'success_fraction': 0.70, 'margin_median': 0.08},
        'cross_view': {'success_fraction': 1.00, 'margin_median': 0.40},
    }
    balanced = {
        'overall': {'success_fraction': 0.89, 'margin_median': 0.20},
        'temporal': {'success_fraction': 0.82, 'margin_median': 0.12},
        'cross_view': {'success_fraction': 0.90, 'margin_median': 0.25},
    }
    assert (roi_temporal_holdout_selection_key(balanced)
            > roi_temporal_holdout_selection_key(cross_view_heavy))


def test_representation_gate_requires_each_branch_to_be_nonregressive():
    raw = {
        'temporal': {'success_fraction': 0.80, 'margin_median': 0.10},
        'cross_view': {'success_fraction': 0.90, 'margin_median': 0.20},
    }
    selected = {
        'temporal': {'success_fraction': 0.79, 'margin_median': 0.11},
        'cross_view': {'success_fraction': 0.99, 'margin_median': 0.30},
    }
    gate = roi_temporal_representation_gate(raw, selected, 100, 20)
    assert gate['passed'] is False
    assert gate['checks']['temporal_success_fraction_nonregression'] is False
    assert gate['checks']['cross_view_success_fraction_nonregression'] is True


def test_state_propagation_audit_reports_wrong_run_loss_and_recovery():
    def row(frame, hit, promotion=False, s7_selected=False):
        return {
            'split': 'val', 'seq': 'real_seq07', 'frame': frame,
            'metrics': {'top1_hit': hit},
            'temporal_selection': {
                'promotion': promotion,
                'selected_source': ('supplement_s7' if s7_selected
                                    else 'native_s14')},
        }

    baseline = [row(0, False), row(1, True), row(2, True)]
    candidate = [row(0, False, True, True),
                 row(1, False, True, True), row(2, True)]
    audit = summarize_roi_temporal_state_propagation(baseline, candidate)
    assert audit['promotion_count'] == 2
    assert audit['wrong_promotion_count'] == 2
    assert audit['max_consecutive_promotion_run'] == 2
    assert audit['max_consecutive_wrong_promotion_run'] == 2
    assert audit['max_consecutive_s7_state_run'] == 2
    assert audit['post_s7_state_old_correct_loss_frame_keys'] == [
        'val|real_seq07|1']
    assert audit['wrong_promotion_recovery_gap_max'] == 2
    assert audit['unresolved_wrong_promotion_incident_count'] == 0


def test_state_propagation_includes_s7_fallback_without_margin_promotion():
    def row(frame, hit, selected_source):
        return {
            'split': 'val', 'seq': 'real_seq07', 'frame': frame,
            'metrics': {'top1_hit': hit},
            'temporal_selection': {
                'promotion': False, 'selected_source': selected_source},
        }

    baseline = [row(0, False, 'native_s14'), row(1, True, 'native_s14')]
    candidate = [row(0, False, 'supplement_s7'),
                 row(1, False, 'native_s14')]
    audit = summarize_roi_temporal_state_propagation(baseline, candidate)
    assert audit['promotion_count'] == 0
    assert audit['s7_state_update_count'] == 1
    assert audit['post_s7_state_old_correct_loss_frame_keys'] == [
        'val|real_seq07|1']


def test_state_propagation_uses_actual_state_update_not_selected_output():
    def row(frame, hit, selected_source, state_source):
        return {
            'split': 'val', 'seq': 'real_seq07', 'frame': frame,
            'metrics': {'top1_hit': hit},
            'temporal_selection': {
                'promotion': selected_source == 'supplement_s7',
                'selected_source': selected_source,
                'state_update_source': state_source},
        }

    baseline = [row(0, True, 'native_s14', 'native_s14'),
                row(1, True, 'native_s14', 'native_s14')]
    candidate = [row(0, False, 'supplement_s7', 'native_s14'),
                 row(1, True, 'native_s14', 'native_s14')]
    audit = summarize_roi_temporal_state_propagation(baseline, candidate)
    assert audit['promotion_count'] == 1
    assert audit['wrong_promotion_count'] == 1
    assert audit['s7_state_update_count'] == 0
    assert audit['post_s7_state_old_correct_loss_count'] == 0


def _protocol34_args(tmp_path):
    support = tmp_path / 'support.json'
    checkpoint = tmp_path / 'base.pth'
    dino_checkpoint = tmp_path / 'dino.pth'
    dino_repo = tmp_path / 'dinov2'
    work_dir = tmp_path / 'work'
    for path in (support, checkpoint, dino_checkpoint):
        path.write_text('placeholder')
    dino_repo.mkdir()
    return SimpleNamespace(
        seed=0, support_result_json=str(support),
        eval_only_checkpoint=str(checkpoint),
        dinov2_checkpoint=str(dino_checkpoint), dinov2_repo=str(dino_repo),
        out_json=str(work_dir / 'result.json'), work_dir=str(work_dir),
        dino_gpus=[0, 1], head_gpu=2, batch_size=128,
        legacy_sdpa_query_chunk=256, epochs=4, lr=0.0003,
        temperature=0.07, promotion_margin=0.05, motion_weight=0.25)


def test_protocol34_accepts_only_three_visible_gpus(
        tmp_path, monkeypatch):
    args = _protocol34_args(tmp_path)
    monkeypatch.setattr(
        protocol34.labeller, 'load_roi_temporal_support_spec',
        lambda *_args: {})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1,2,3')
    protocol34.validate_args(args)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1,2,3')
    with pytest.raises(ValueError, match='at most three GPUs'):
        protocol34.validate_args(args)


def test_protocol34_rejects_unscaled_formal_optimizer_settings(
        tmp_path, monkeypatch):
    args = _protocol34_args(tmp_path)
    monkeypatch.setattr(
        protocol34.labeller, 'load_roi_temporal_support_spec',
        lambda *_args: {})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1,2,3')
    args.lr = 0.000225
    with pytest.raises(ValueError, match='locks --lr 0.0003'):
        protocol34.validate_args(args)


def test_protocol35_builds_read_only_three_gpu_route(tmp_path, monkeypatch):
    paths = {}
    for name in ('protocol34_result_json', 'adapter_checkpoint',
                 'eval_only_checkpoint', 'dinov2_checkpoint'):
        path = tmp_path / '{}.bin'.format(name)
        path.write_text('placeholder')
        paths[name] = str(path)
    dino_repo = tmp_path / 'dinov2'
    dino_repo.mkdir()
    args = SimpleNamespace(
        **paths, data_root='data', dinov2_repo=str(dino_repo),
        dino_gpus=[1, 2], head_gpu=0, legacy_sdpa_query_chunk=256,
        feature_cache_dir=str(tmp_path / 'cache'),
        work_dir=str(tmp_path / 'work'),
        out_json=str(tmp_path / 'work' / 'result.json'),
        empty_cache_interval=1, seed=0)
    monkeypatch.setattr(
        protocol35.labeller, 'load_roi_temporal_counterfactual_spec',
        lambda *_args: {})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1,2')
    protocol35.validate_args(args)
    argv = protocol35.build_locked_labeller_argv(args)
    assert '--source-roi-temporal-counterfactual-audit' in argv
    assert '--skip-target-eval' in argv
    assert argv[argv.index('--dino-gpus') + 1:argv.index('--head-gpu')] == [
        '1', '2']
