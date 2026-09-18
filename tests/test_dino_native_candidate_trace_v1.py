import hashlib
import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from crane_project.tools import dino_teacher_rotated_labeller as labeller


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _attribution_payload(head_sha, dino_sha):
    return dict(
        protocol='frozen_dino_native_s14_failure_attribution_v1',
        evidence_boundary=dict(
            target_slices_are_exposed=True,
            target_results_authorize_tuning=False),
        model_identity=dict(
            s7_enabled=False,
            dino_head_checkpoint_sha256=head_sha,
            frozen_dinov2_checkpoint_sha256=dino_sha),
        records=[
            dict(
                group='seq03_small', split='test',
                sequence='real_seq03', frame=144,
                review_required=True, attribution='ROI_REGRESSION'),
            dict(
                group='seq03_small', split='test',
                sequence='real_seq03', frame=145,
                review_required=True,
                attribution='ROI_ORDERING_OR_NMS'),
            dict(
                group='seq02_dark', split='test',
                sequence='real_seq02', frame=141,
                review_required=True,
                attribution='ROI_ORDERING_OR_NMS'),
            dict(
                group='seq03_small', split='test',
                sequence='real_seq03', frame=146,
                review_required=False, attribution='SUCCESS_TOP1'),
        ])


def _trace_args(tmp_path):
    return SimpleNamespace(
        eval_only_checkpoint='head.pth',
        native_candidate_trace_attribution_json='attribution.json',
        native_candidate_trace_groups=['seq03_small'],
        s7_residual=False, skip_target_eval=False,
        patch_size=14, dino_height=600, dino_max_long_side=1333,
        proposal_count=2000, max_detections=2000,
        roi_nms_iou_thr=0.1, riou_thr=0.5, feature_strides=None,
        out_json=str(tmp_path / 'trace.json'))


def test_trace_args_lock_formal_native_baseline(tmp_path):
    args = _trace_args(tmp_path)
    labeller.validate_native_candidate_trace_args(args)
    args.roi_nms_iou_thr = 0.5
    with pytest.raises(ValueError, match='locks ROI NMS IoU to 0.1'):
        labeller.validate_native_candidate_trace_args(args)

    args = _trace_args(tmp_path)
    args.feature_strides = [7, 14]
    with pytest.raises(ValueError, match='native stride 14'):
        labeller.validate_native_candidate_trace_args(args)

    args = _trace_args(tmp_path)
    (tmp_path / 'trace.json').write_text('{}')
    with pytest.raises(ValueError, match='Refusing to overwrite'):
        labeller.validate_native_candidate_trace_args(args)


def test_trace_spec_locks_identity_and_selects_review_small_frames(
        tmp_path, monkeypatch):
    head = tmp_path / 'head.pth'
    dino = tmp_path / 'dino.pth'
    head.write_bytes(b'head')
    dino.write_bytes(b'dino')
    attribution = tmp_path / 'attribution.json'
    attribution.write_text(json.dumps(
        _attribution_payload(_sha(head), _sha(dino))))

    class _Diag:
        def find_files(self, data_root, split, seq, frame):
            del data_root
            image = tmp_path / '{}_{}_{}.jpg'.format(split, seq, frame)
            annotation = tmp_path / '{}_{}_{}.txt'.format(
                split, seq, frame)
            image.write_bytes(b'image-' + str(frame).encode())
            annotation.write_text('annotation')
            return str(image), str(annotation)

    monkeypatch.setattr(
        labeller.common.entry_probe, 'get_diag', lambda: _Diag())
    args = SimpleNamespace(
        eval_only_checkpoint=str(head), dinov2_checkpoint=str(dino),
        native_candidate_trace_groups=['seq03_small'], data_root='unused',
        native_candidate_trace_all_records=False)
    spec = labeller.load_native_candidate_trace_spec(
        str(attribution), args)
    assert spec['groups'] == ['seq03_small']
    assert [row['frame'] for row in spec['records']] == [144, 145]
    assert all(row['attribution']['review_required']
               for row in spec['records'])


def test_trace_spec_can_select_all_records_without_repartitioning(
        tmp_path, monkeypatch):
    head = tmp_path / 'head.pth'
    dino = tmp_path / 'dino.pth'
    head.write_bytes(b'head')
    dino.write_bytes(b'dino')
    attribution = tmp_path / 'attribution.json'
    attribution.write_text(json.dumps(
        _attribution_payload(_sha(head), _sha(dino))))

    class _Diag:
        def find_files(self, data_root, split, seq, frame):
            del data_root
            image = tmp_path / '{}_{}_{}.jpg'.format(split, seq, frame)
            annotation = tmp_path / '{}_{}_{}.txt'.format(
                split, seq, frame)
            image.write_bytes(b'image-' + str(frame).encode())
            annotation.write_text('annotation')
            return str(image), str(annotation)

    monkeypatch.setattr(
        labeller.common.entry_probe, 'get_diag', lambda: _Diag())
    args = SimpleNamespace(
        eval_only_checkpoint=str(head), dinov2_checkpoint=str(dino),
        native_candidate_trace_groups=['seq03_small'], data_root='unused',
        native_candidate_trace_all_records=True)
    spec = labeller.load_native_candidate_trace_spec(
        str(attribution), args)
    assert spec['selection_mode'] == 'all_records'
    assert [row['frame'] for row in spec['records']] == [144, 145, 146]
    assert spec['records'][-1]['attribution']['attribution'] == 'SUCCESS_TOP1'


def test_trace_reproduction_accepts_locked_success_record():
    trace = {
        'best_rpn_riou': 0.55, 'best_decoded_riou': 0.67,
        'counts': dict(
            proposals=10, decoded=10, post_nms=4, post_valid=3),
        'final_post_valid_metrics': {'top1_hit': True}}
    result = labeller.validate_native_trace_reproduction(
        trace, {
            'attribution': 'SUCCESS_TOP1',
            'rpn_best_riou': 0.55, 'decoded_best_riou': 0.67,
            'candidate_counts': dict(
                rpn_proposals=10, roi_decoded=10,
                post_nms=4, post_valid=3)})
    assert result['passed'] is True
    assert result['candidate_counts']['post_nms']['matches'] is True


def test_trace_reproduction_rejects_postprocessing_count_mismatch():
    trace = {
        'best_rpn_riou': 0.55, 'best_decoded_riou': 0.67,
        'counts': dict(
            proposals=10, decoded=10, post_nms=8, post_valid=6),
        'final_post_valid_metrics': {'top1_hit': True}}
    attribution = {
        'attribution': 'SUCCESS_TOP1',
        'rpn_best_riou': 0.55, 'decoded_best_riou': 0.67,
        'candidate_counts': dict(
            rpn_proposals=10, roi_decoded=10,
            post_nms=4, post_valid=3)}
    with pytest.raises(RuntimeError, match='post_nms count'):
        labeller.validate_native_trace_reproduction(trace, attribution)


def test_trace_spec_rejects_checkpoint_or_evidence_boundary(
        tmp_path):
    head = tmp_path / 'head.pth'
    dino = tmp_path / 'dino.pth'
    head.write_bytes(b'head')
    dino.write_bytes(b'dino')
    payload = _attribution_payload('0' * 64, _sha(dino))
    attribution = tmp_path / 'attribution.json'
    attribution.write_text(json.dumps(payload))
    args = SimpleNamespace(
        eval_only_checkpoint=str(head), dinov2_checkpoint=str(dino),
        native_candidate_trace_groups=['seq03_small'], data_root='unused',
        native_candidate_trace_all_records=False)
    with pytest.raises(RuntimeError, match='head checkpoint identity'):
        labeller.load_native_candidate_trace_spec(str(attribution), args)

    payload['model_identity']['dino_head_checkpoint_sha256'] = _sha(head)
    payload['evidence_boundary']['target_results_authorize_tuning'] = True
    attribution.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match='evidence boundary'):
        labeller.load_native_candidate_trace_spec(str(attribution), args)


def test_suppressor_mapping_uses_actual_keep_and_higher_score(monkeypatch):
    ops = types.ModuleType('mmcv.ops')

    def fake_iou(boxes, kept_boxes):
        del kept_boxes
        return torch.tensor([[1.0], [0.8], [0.2]], dtype=boxes.dtype)

    ops.box_iou_rotated = fake_iou
    monkeypatch.setitem(sys.modules, 'mmcv.ops', ops)
    boxes = torch.zeros((3, 5), dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8, 0.7])
    result = labeller._nms_suppressor_indices(
        boxes, scores, torch.tensor([0]), 0.5)
    assert set(result) == {1}
    assert result[1]['candidate_id'] == 0
    assert result[1]['post_nms_rank'] == 1
    assert result[1]['rotated_iou'] == pytest.approx(0.8)


def test_source_quality_trace_skips_duplicate_official_roi_forward():
    class _RoiHead:
        def simple_test(self, *args, **kwargs):
            raise AssertionError('source audit must not repeat ROI forward')

    reconstructed = np.asarray(
        [[10.0, 20.0, 30.0, 40.0, 0.1, 0.9]], dtype=np.float32)
    official, audit = labeller._native_trace_official_output(
        SimpleNamespace(roi_head=_RoiHead()), object(), object(), {},
        reconstructed, True)
    assert official is reconstructed
    assert audit['duplicate_official_forward_executed'] is False
    assert audit['verification'] == (
        'not_run_covered_by_sampled_and_aggregate_checks')
    assert audit['shape_equal'] is None
    assert audit['allclose_atol_1e_4'] is None
    assert audit['max_abs_diff'] is None
    assert audit['expected_count'] is None


def test_normal_trace_keeps_exact_official_roi_reproduction_check():
    reconstructed = np.asarray(
        [[10.0, 20.0, 30.0, 40.0, 0.1, 0.9]], dtype=np.float32)

    class _RoiHead:
        calls = 0

        def simple_test(self, features, proposals, metas, rescale):
            del features, proposals, metas
            assert rescale is True
            self.calls += 1
            return [[reconstructed.copy()]]

    roi_head = _RoiHead()
    official, audit = labeller._native_trace_official_output(
        SimpleNamespace(roi_head=roi_head), object(), object(), {},
        reconstructed, False)
    assert np.array_equal(official, reconstructed)
    assert roi_head.calls == 1
    assert audit['duplicate_official_forward_executed'] is True
    assert audit['verification'] == 'per_frame_official_roi_output'

    wrong = reconstructed.copy()
    wrong[0, 0] += 1.0
    with pytest.raises(RuntimeError, match='did not reproduce ROI output'):
        labeller._native_trace_official_output(
            SimpleNamespace(roi_head=SimpleNamespace(
                simple_test=lambda *args, **kwargs: [[wrong]])),
            object(), object(), {}, reconstructed, False)


def test_best_decoded_candidate_summary_handles_empty_candidates():
    assert labeller._best_decoded_candidate_summary([]) == dict(
        candidate_id=None, riou=0.0, disposition=None)
    assert labeller._best_decoded_candidate_summary([
        dict(candidate_id=2, decoded_gt_riou=0.4,
             disposition='NMS_SUPPRESSED'),
        dict(candidate_id=5, decoded_gt_riou=0.7,
             disposition='POST_VALID_CONTENT'),
    ]) == dict(
        candidate_id=5, riou=0.7,
        disposition='POST_VALID_CONTENT')


def test_trace_summary_keeps_regression_and_nms_failures_separate():
    def row(attribution, decoded, post_nms, suppressed, wrong):
        stage = ('ROI_REGRESSION' if not decoded else
                 'NMS_SUPPRESSION' if not post_nms else
                 'FINAL_ORDERING')
        return dict(
            attribution=dict(attribution=attribution),
            reproduction=dict(
                passed=True,
                metrics=dict(
                    decoded_best_riou=dict(absolute_delta=1e-4)),
                candidate_counts=dict(
                    post_nms=dict(matches=True))),
            trace=dict(
                resolved_failure_stage=stage,
                reconstruction=dict(allclose_atol_1e_4=True),
                counts=dict(
                    decoded_usable=decoded, post_nms_usable=post_nms,
                    post_valid_usable=post_nms,
                    nms_suppressed_usable=suppressed,
                    nms_suppressed_usable_by_wrong_candidate=wrong)))

    summary = labeller.summarize_native_candidate_trace([
        row('ROI_REGRESSION', 0, 0, 0, 0),
        row('ROI_ORDERING_OR_NMS', 3, 0, 3, 2),
    ])
    assert summary['frame_count'] == 2
    assert summary['decoded_usable_frame_count'] == 1
    assert summary['post_nms_usable_frame_count'] == 0
    assert summary['nms_suppressed_usable_candidate_count'] == 3
    assert summary['frames_with_usable_suppressed_by_wrong_candidate'] == 1
    assert summary['metric_reproduction_absolute_tolerance'] == 1e-3
    assert summary['metric_reproduction_max_absolute_delta'] == 1e-4
    assert summary['candidate_count_reproduction_check_count'] == 2
    assert summary['candidate_count_reproduction_pass_count'] == 2
    assert summary['resolved_failure_stage_counts'] == {
        'NMS_SUPPRESSION': 1, 'ROI_REGRESSION': 1}
    assert summary['attributed_roi_regression_count'] == 1
    assert summary['attributed_roi_ordering_or_nms_count'] == 1


def test_trace_reproduction_requires_same_metrics_and_failure_stage():
    trace = dict(
        best_rpn_riou=0.55, best_decoded_riou=0.67,
        counts={},
        resolved_failure_stage='NMS_SUPPRESSION',
        final_post_valid_metrics=dict(top1_hit=False))
    attribution = dict(
        rpn_best_riou=0.55001, decoded_best_riou=0.67001,
        attribution='ROI_ORDERING_OR_NMS')
    result = labeller.validate_native_trace_reproduction(trace, attribution)
    assert result['absolute_tolerance'] == pytest.approx(1e-3)
    assert result['metrics']['decoded_best_riou']['absolute_delta'] == (
        pytest.approx(1e-5))

    # Multi-GPU and single-GPU executions can differ slightly in decoded
    # geometry. The formal tolerance accepts the observed 1.28e-4 drift.
    trace['best_decoded_riou'] = attribution['decoded_best_riou'] - 1.2821e-4
    labeller.validate_native_trace_reproduction(trace, attribution)

    trace['best_decoded_riou'] = 0.68
    with pytest.raises(RuntimeError, match='decoded_best_riou'):
        labeller.validate_native_trace_reproduction(trace, attribution)

    trace['best_decoded_riou'] = 0.67001
    trace['resolved_failure_stage'] = 'ROI_REGRESSION'
    with pytest.raises(RuntimeError, match='stage disagrees'):
        labeller.validate_native_trace_reproduction(trace, attribution)

    trace['resolved_failure_stage'] = 'FINAL_ORDERING'
    labeller.validate_native_trace_reproduction(trace, attribution)


def _source_audit_args(tmp_path, head, dino):
    return SimpleNamespace(
        eval_only_checkpoint=str(head), dinov2_checkpoint=str(dino),
        source_native_quality_reference_json=str(tmp_path / 'reference.json'),
        source_native_quality_feasibility_audit=True,
        native_candidate_trace_audit=False,
        skip_target_eval=True, s7_residual=False,
        source_train_datasets=['train:train'],
        source_val_datasets=['val:val'],
        patch_size=14, dino_height=600, dino_max_long_side=1333,
        proposal_count=2000, max_detections=2000,
        roi_nms_iou_thr=0.5, riou_thr=0.5, feature_strides=None,
        out_json=str(tmp_path / 'source_quality.json'))


def test_source_quality_reference_locks_model_and_source_identity(tmp_path):
    head = tmp_path / 'head.pth'
    dino = tmp_path / 'dino.pth'
    head.write_bytes(b'head')
    dino.write_bytes(b'dino')
    args = _source_audit_args(tmp_path, head, dino)
    payload = dict(
        selector='Frozen DINO ROI Classifier Source Interpolation Selector V1',
        selected_checkpoint_sha256=_sha(head),
        dinov2_checkpoint_sha256=_sha(dino), selected_alpha=0.5,
        protocol=dict(
            source_val_datasets=['val:val'],
            source_small_definition=dict(
                definition='source_train_short_token_lower_tertile',
                short_token_threshold=1.67),
            target_data_discovered=False, target_data_read=False,
            target_used_for_selection=False),
        isolation=dict(dino_frozen=True),
        source=dict(candidates=[dict(
            alpha=0.5,
            source_full_summary=dict(frame_count=738, top1_hits=677),
            source_small_summary=dict(frame_count=350, top1_hits=303))]))
    reference = tmp_path / 'reference.json'
    reference.write_text(json.dumps(payload))
    labeller.validate_source_native_quality_feasibility_args(args)
    spec = labeller.load_source_native_quality_reference(
        str(reference), args)
    assert spec['checkpoint_sha256'] == _sha(head)
    assert spec['small_token_threshold'] == pytest.approx(1.67)
    assert spec['expected_summary']['top1_hit_count'] == 677

    payload['protocol']['source_val_datasets'] = ['other:other']
    reference.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match='validation dataset identity'):
        labeller.load_source_native_quality_reference(
            str(reference), args)


def test_source_quality_audit_requires_target_skip_and_refuses_overwrite(
        tmp_path):
    head = tmp_path / 'head.pth'
    dino = tmp_path / 'dino.pth'
    head.write_bytes(b'head')
    dino.write_bytes(b'dino')
    args = _source_audit_args(tmp_path, head, dino)
    args.skip_target_eval = False
    with pytest.raises(ValueError, match='requires --skip-target-eval'):
        labeller.validate_source_native_quality_feasibility_args(args)
    args.skip_target_eval = True
    (tmp_path / 'source_quality.json').write_text('{}')
    with pytest.raises(ValueError, match='Refusing to overwrite'):
        labeller.validate_source_native_quality_feasibility_args(args)


def _candidate(candidate_id, riou, score, disposition,
               post_valid_rank=None, suppressor=None):
    return dict(
        candidate_id=candidate_id, decoded_gt_riou=riou,
        foreground_score=score, disposition=disposition,
        post_nms_rank=post_valid_rank, post_valid_rank=post_valid_rank,
        suppressor=suppressor)


def test_source_quality_compaction_records_wrong_nms_and_ordering_pairs():
    nms_trace = dict(
        candidates=[
            _candidate(
                0, 0.61, 0.40, 'NMS_SUPPRESSED',
                suppressor=dict(candidate_id=1, rotated_iou=0.7)),
            _candidate(1, 0.45, 0.90, 'POST_VALID_CONTENT', 1)],
        resolved_failure_stage='NMS_SUPPRESSION', best_rpn_riou=0.55,
        best_decoded_riou=0.61,
        final_post_valid_metrics=dict(top1_hit=False),
        counts=dict(post_nms_usable=0, post_valid_usable=0))
    compact = labeller._native_quality_compact_row(nms_trace, 0.5)
    assert compact['actionable_quality_conflict'] is True
    assert compact['wrong_nms_pair_count'] == 1
    assert compact['representative_wrong_nms_pairs'][0][
        'suppressor_riou'] == pytest.approx(0.45)

    ordering_trace = dict(
        candidates=[
            _candidate(0, 0.40, 0.90, 'POST_VALID_CONTENT', 1),
            _candidate(1, 0.65, 0.20, 'POST_VALID_CONTENT', 2)],
        resolved_failure_stage='FINAL_ORDERING', best_rpn_riou=0.6,
        best_decoded_riou=0.65,
        final_post_valid_metrics=dict(top1_hit=False),
        counts=dict(post_nms_usable=1, post_valid_usable=1))
    compact = labeller._native_quality_compact_row(ordering_trace, 0.5)
    assert compact['ordering_pair_present'] is True
    assert compact['final_ordering_pair']['usable_post_valid_rank'] == 2


def test_source_quality_support_gate_stops_when_cross_sequence_data_missing():
    def row(seq, conflict, pairs=0, small=True, hit=False):
        return dict(
            seq=seq, source_small=small,
            trace=dict(
                resolved_failure_stage=(
                    'FINAL_ORDERING' if conflict else 'TOP1_SUCCESS'),
                final_top1_hit=hit,
                actionable_quality_conflict=conflict,
                wrong_nms_pair_count=pairs,
                ordering_pair_present=bool(conflict and pairs == 0)))

    one_sequence = [row('seq_a', True, 2) for _ in range(20)]
    summary = labeller.summarize_source_native_quality_support(one_sequence)
    assert summary['support_gate']['passed'] is False
    assert summary['support_gate']['checks'][
        'minimum_cross_sequence_support'] is False

    supported = (
        [row('seq_a', True, 2) for _ in range(10)]
        + [row('seq_b', True, 2) for _ in range(10)])
    summary = labeller.summarize_source_native_quality_support(supported)
    assert summary['source_small_actionable_conflict_frame_count'] == 20
    assert summary['source_small_actionable_pair_count'] == 40
    assert summary['support_gate']['passed'] is True
