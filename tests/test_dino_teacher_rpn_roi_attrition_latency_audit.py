import argparse
import json

import pytest
import torch

from crane_project.tools import (
    dino_teacher_rpn_roi_attrition_latency_audit as audit)
from crane_project.tools import (
    dino_teacher_rpn_roi_attrition_v2_report as report_v2)
from crane_project.tools import (
    dino_teacher_token_scale_rpn_coverage_audit as coverage)


def _args(tmp_path, **overrides):
    paths = {}
    for name in ('coverage.json', 'labeller.pth', 'dino.pth'):
        path = tmp_path / name
        path.write_bytes(b'placeholder')
        paths[name] = str(path)
    values = dict(
        seed=0, source_rpn_datasets=['val:val'], target_slices=None,
        dino_gpus=[1, 2], head_gpu=0,
        patch_size=14, rpn_feat_channels=256, roi_fc_channels=1024,
        roi_samples=256, proposal_count=2000, max_detections=2000,
        dino_height=600, dino_max_long_side=1333,
        source_rpn_limit=0, source_fresh_latency_samples=10,
        latency_warmup=1, reconstruction_check_count=3,
        recall_ks=[20, 100, 1000, 2000], riou_thr=0.5,
        coverage_audit_json=paths['coverage.json'],
        source_selection_json=None,
        expected_source_retention_rate=0.985,
        labeller_checkpoint=paths['labeller.pth'],
        dinov2_checkpoint=paths['dino.pth'],
        target_feature_mode='fresh_fp32',
        out_json=str(tmp_path / 'result.json'))
    values.update(overrides)
    return argparse.Namespace(**values)


def test_validate_uses_default_target_slices(tmp_path):
    args = _args(tmp_path)
    audit.validate_args(args)
    assert [row['name'] for row in args.parsed_target_slices] == [
        'seq02_far', 'seq02_dark', 'seq03_small']


def test_validate_rejects_negative_latency_samples(tmp_path):
    with pytest.raises(ValueError, match='non-negative'):
        audit.validate_args(_args(tmp_path, source_fresh_latency_samples=-1))


def test_load_coverage_contract_checks_hashes(tmp_path, monkeypatch):
    args = _args(tmp_path)
    payload = dict(
        audit=coverage.AUDIT_NAME,
        isolation=dict(
            optimizer_steps=0, dino_parameters_unchanged=True,
            labeller_parameters_unchanged=True,
            target_labels_used_for_evaluation_only=True),
        protocol=dict(source_defined_token_bins=dict(
            lower=1.0, upper=2.0,
            labels=['source_small', 'source_medium', 'source_large'])),
        labeller_checkpoint_sha256='labeller',
        dinov2_checkpoint_sha256='dino')
    with open(args.coverage_audit_json, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle)
    monkeypatch.setattr(
        audit.common, 'file_sha256',
        lambda path: 'labeller' if path.endswith('labeller.pth') else 'dino')
    _payload, boundaries = audit.load_coverage_contract(
        args.coverage_audit_json, args)
    assert boundaries['lower'] == 1.0


def test_coverage_bins_can_be_reused_with_source_only_fc_cls_proof(
        tmp_path, monkeypatch):
    proof_path = tmp_path / 'selection.json'
    args = _args(tmp_path, source_selection_json=str(proof_path))
    coverage_payload = dict(
        audit=coverage.AUDIT_NAME,
        isolation=dict(
            optimizer_steps=0, dino_parameters_unchanged=True,
            labeller_parameters_unchanged=True,
            target_labels_used_for_evaluation_only=True),
        protocol=dict(source_defined_token_bins=dict(
            lower=1.0, upper=2.0,
            labels=['source_small', 'source_medium', 'source_large'])),
        labeller_checkpoint_sha256='old_labeller',
        dinov2_checkpoint_sha256='dino')
    with open(args.coverage_audit_json, 'w', encoding='utf-8') as handle:
        json.dump(coverage_payload, handle)
    proof = dict(
        decision='SOURCE_ONLY_EPOCH_SELECTED_TARGET_NOT_READ',
        source_only=True, target_data_read=False,
        min_exact_retention_rate=0.985,
        selected=dict(
            epoch=1, output_checkpoint_sha256='new_labeller',
            source_exact_retention=dict(
                baseline_correct_count=662,
                retained_correct_count=653)),
        parameter_invariants=dict(
            frozen_tensors_bit_identical=True,
            changed_parameter_names=[
                'roi_head.bbox_head.fc_cls.weight',
                'roi_head.bbox_head.fc_cls.bias']))
    with proof_path.open('w', encoding='utf-8') as handle:
        json.dump(proof, handle)

    def fake_hash(path):
        name = str(path)
        if name.endswith('labeller.pth'):
            return 'new_labeller'
        if name.endswith('dino.pth'):
            return 'dino'
        return 'proof'

    monkeypatch.setattr(audit.common, 'file_sha256', fake_hash)
    payload, boundaries = audit.load_coverage_contract(
        args.coverage_audit_json, args)
    assert boundaries['lower'] == 1.0
    assert payload['_reuse_contract']['mode'] == (
        'source_token_bins_only_with_fc_cls_change_proof')
    assert payload['_reuse_contract']['source_selection_proof'][
        'selected_epoch'] == 1


def test_coverage_mismatch_without_source_proof_is_rejected(
        tmp_path, monkeypatch):
    args = _args(tmp_path)
    payload = dict(
        audit=coverage.AUDIT_NAME,
        isolation=dict(
            optimizer_steps=0, dino_parameters_unchanged=True,
            labeller_parameters_unchanged=True,
            target_labels_used_for_evaluation_only=True),
        protocol=dict(source_defined_token_bins=dict(
            lower=1.0, upper=2.0,
            labels=['source_small', 'source_medium', 'source_large'])),
        labeller_checkpoint_sha256='old',
        dinov2_checkpoint_sha256='dino')
    with open(args.coverage_audit_json, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle)
    monkeypatch.setattr(
        audit.common, 'file_sha256',
        lambda path: 'new' if str(path).endswith('labeller.pth') else 'dino')
    with pytest.raises(RuntimeError, match='source-only selection proof'):
        audit.load_coverage_contract(args.coverage_audit_json, args)


def test_object_attrition_identifies_regression_loss(monkeypatch):
    def fake_riou(boxes, gt_boxes):
        if boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
            return boxes.new_zeros((boxes.shape[0], gt_boxes.shape[0]))
        same_center = (
            boxes[:, None, :2] == gt_boxes[None, :, :2]).all(dim=-1)
        return same_center.to(dtype=torch.float32)

    monkeypatch.setattr(audit, 'rotated_ious', fake_riou)
    proposals = torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.0, 0.9]])
    decoded = torch.tensor([[100.0, 100.0, 10.0, 10.0, 0.0]])
    probabilities = torch.tensor([[0.9, 0.1]])
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.0]])
    rows = audit.object_attrition_rows(
        proposals, decoded, probabilities, torch.tensor([0]),
        torch.cat([decoded, probabilities[:, :1]], dim=1),
        torch.zeros((0, 6)).numpy(), gt, gt.numpy(), 0.5)
    assert rows[0]['attrition_cause'] == (
        'ROI_REGRESSION_DESTROYS_RPN_GEOMETRY')


def test_object_attrition_accepts_empty_rpn_proposals():
    empty_boxes = torch.zeros((0, 5))
    rows = audit.object_attrition_rows(
        torch.zeros((0, 6)), empty_boxes, torch.zeros((0, 2)),
        torch.zeros((0,), dtype=torch.long), torch.zeros((0, 6)),
        torch.zeros((0, 6)).numpy(),
        torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.0]]).numpy(), 0.5)
    assert rows[0]['attrition_cause'] == 'RPN_MISS'
    assert rows[0]['best_rpn_geometry'] is None


def test_summarize_attrition_token_bins_do_not_recurse():
    obj = dict(
        attrition_cause='ROI_TOP1_RESTORED',
        source_token_bin='source_small',
        rpn=dict(best_usable_rank=1),
        roi_regression=dict(
            rpn_usable_survives_count=1, decoded_usable_count=1),
        post_nms=dict(best_usable_rank=1),
        post_valid_content=dict(best_usable_rank=1, top1_hit=True),
        best_foreground_among_rpn_usable=dict(
            foreground_over_background=True, foreground_rank=1))
    summary = audit.summarize_attrition([dict(objects=[obj])])
    assert summary['source_token_bins']['source_small']['object_count'] == 1
    assert summary['source_token_bins']['source_medium']['object_count'] == 0
    assert 'source_token_bins' not in summary['source_token_bins'][
        'source_small']


def test_attrition_summary_reports_frame_top1_hits_and_sequence_mcml():
    def obj(hit):
        return dict(
            attrition_cause=(
                'ROI_TOP1_RESTORED' if hit else
                'ROI_ORDERING_OR_NMS_REMOVES_GEOMETRY'),
            source_token_bin='source_small',
            rpn=dict(best_usable_rank=1),
            roi_regression=dict(
                rpn_usable_survives_count=1, decoded_usable_count=1),
            post_nms=dict(best_usable_rank=(1 if hit else None)),
            post_valid_content=dict(
                best_usable_rank=(1 if hit else None), top1_hit=hit),
            best_foreground_among_rpn_usable=dict(
                foreground_over_background=hit,
                foreground_rank=(1 if hit else 2)))

    rows = [
        dict(seq='a', frame=1, objects=[obj(False)]),
        dict(seq='a', frame=2, objects=[obj(False)]),
        dict(seq='b', frame=1, objects=[obj(False)]),
        dict(seq='b', frame=2, objects=[obj(True)]),
    ]
    summary = audit.summarize_attrition(rows, include_token_bins=False)
    assert summary['final_top1_hits'] == 1
    assert summary['final_top1_mcml'] == 2


def test_nms_indices_are_mapped_back_after_score_filtering():
    probabilities = torch.tensor([
        [0.0, 1.0], [0.7, 0.3], [0.8, 0.2]])
    mapped = audit.original_nms_indices(
        probabilities, 0.1, torch.tensor([1]))
    assert mapped.tolist() == [2]


def test_nms_suppression_attributes_false_kept_competitor(monkeypatch):
    monkeypatch.setattr(
        audit, 'rotated_ious',
        lambda boxes, gt: torch.tensor([[0.4]], dtype=torch.float32))
    decoded = torch.tensor([
        [0.0, 0.0, 10.0, 10.0, 0.0],
        [1.0, 0.0, 10.0, 10.0, 0.0]])
    probabilities = torch.tensor([[0.6, 0.4], [0.9, 0.1]])
    result = audit.nms_suppression_attribution(
        decoded, probabilities, torch.tensor([1]),
        torch.tensor([0.8, 0.1]), score_thr=0.0,
        nms_iou_thr=0.1, usable_riou_thr=0.5)
    assert result['status'] == 'SUPPRESSED_BY_FALSE_ROI'
    assert result['candidate_index'] == 0
    assert result['suppressor_index'] == 1
    assert result['suppressor_score_gap'] == pytest.approx(0.3)


def test_nms_suppression_reports_retained_usable_candidate():
    result = audit.nms_suppression_attribution(
        torch.zeros((1, 5)), torch.tensor([[0.8, 0.2]]),
        torch.tensor([0]), torch.tensor([0.7]),
        score_thr=0.0, nms_iou_thr=0.1, usable_riou_thr=0.5)
    assert result['status'] == 'RETAINED_BY_NMS'


def test_diagnose_prefers_rpn_then_regression_then_nms():
    base = dict(
        object_count=10, rpn_recall=1.0,
        rpn_geometry_survives_roi_regression=1.0,
        post_nms_recall=1.0, post_valid_recall=1.0,
        final_top1_recall=1.0)
    row = dict(base, rpn_recall=0.7)
    assert audit.diagnose(row) == 'RPN_COVERAGE_PRIMARY_LIMIT'
    row = dict(base, rpn_geometry_survives_roi_regression=0.7)
    assert audit.diagnose(row) == 'ROI_REGRESSION_PRIMARY_ATTRITION'
    row = dict(base, post_nms_recall=0.7)
    assert audit.diagnose(row) == (
        'ROI_CLASSIFICATION_ORDERING_OR_NMS_PRIMARY_ATTRITION')


def test_corrected_cause_allows_roi_recovery_after_rpn_miss():
    row = dict(
        rpn=dict(best_usable_rank=None),
        roi_regression=dict(decoded_usable_count=1),
        post_nms=dict(best_usable_rank=1),
        post_valid_content=dict(best_usable_rank=1, top1_hit=True))
    assert audit.attrition_cause_from_object(row) == 'ROI_TOP1_RESTORED'


def test_corrected_cause_attributes_terminal_failure_to_ordering():
    row = dict(
        rpn=dict(best_usable_rank=None),
        roi_regression=dict(decoded_usable_count=1),
        post_nms=dict(best_usable_rank=None),
        post_valid_content=dict(best_usable_rank=None, top1_hit=False))
    assert audit.attrition_cause_from_object(row) == (
        'ROI_ORDERING_OR_NMS_REMOVES_GEOMETRY')


def test_v2_report_preserves_raw_cause_and_rewrites_terminal_cause():
    obj = dict(
        attrition_cause='RPN_MISS', source_token_bin='source_small',
        rpn=dict(best_usable_rank=None),
        roi_regression=dict(
            rpn_usable_survives_count=0, decoded_usable_count=1,
            decoded_usable_foreground_rank=2),
        post_nms=dict(best_usable_rank=1),
        post_valid_content=dict(best_usable_rank=1, top1_hit=True),
        best_foreground_among_rpn_usable=None)
    rows = report_v2.corrected_rows([dict(objects=[obj])])
    corrected = rows[0]['objects'][0]
    assert corrected['raw_attrition_cause'] == 'RPN_MISS'
    assert corrected['attrition_cause'] == 'ROI_TOP1_RESTORED'
    assert corrected['rpn_initial_miss_recovered'] is True


def test_latency_summary_ignores_cached_none_values():
    rows = [dict(latency=dict(
        preprocess_ms=1.0, dino_ms=2.0, dino_to_head_ms=3.0,
        rpn_ms=4.0, roi_head_ms=5.0, roi_decode_ms=6.0,
        roi_nms_ms=7.0, valid_filter_ms=8.0,
        dino_branch_total_ms=36.0)),
            dict(latency={key: None for key in (
                'preprocess_ms', 'dino_ms', 'dino_to_head_ms', 'rpn_ms',
                'roi_head_ms', 'roi_decode_ms', 'roi_nms_ms',
                'valid_filter_ms', 'dino_branch_total_ms')})]
    summary = audit.summarize_latency(rows)
    assert summary['dino_ms']['sample_count'] == 1
    assert summary['dino_ms']['mean'] == pytest.approx(2.0)
