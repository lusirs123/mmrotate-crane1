import hashlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

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
        dinov2_model='dinov2_vitl14',
        feature_cache_dir=str(tmp_path / 'feature_cache'),
        proposal_count=2000, max_detections=2000,
        roi_nms_iou_thr=0.5, riou_thr=0.5, feature_strides=None,
        out_json=str(tmp_path / 'trace.json'))


def test_trace_args_lock_formal_native_baseline(tmp_path):
    args = _trace_args(tmp_path)
    labeller.validate_native_candidate_trace_args(args)
    args.roi_nms_iou_thr = 0.1
    with pytest.raises(ValueError, match='locks ROI NMS IoU to 0.5'):
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
        native_candidate_trace_groups=['seq03_small'], data_root='unused')
    spec = labeller.load_native_candidate_trace_spec(
        str(attribution), args)
    assert spec['groups'] == ['seq03_small']
    assert [row['frame'] for row in spec['records']] == [144, 145]
    assert all(row['attribution']['review_required']
               for row in spec['records'])


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
        native_candidate_trace_groups=['seq03_small'], data_root='unused')
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


def test_trace_summary_keeps_regression_and_nms_failures_separate():
    def row(attribution, decoded, post_nms, suppressed, wrong):
        stage = ('ROI_REGRESSION' if not decoded else
                 'NMS_SUPPRESSION' if not post_nms else
                 'FINAL_ORDERING')
        return dict(
            attribution=dict(attribution=attribution),
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
    assert summary['resolved_failure_stage_counts'] == {
        'NMS_SUPPRESSION': 1, 'ROI_REGRESSION': 1}
    assert summary['attributed_roi_regression_count'] == 1
    assert summary['attributed_roi_ordering_or_nms_count'] == 1


def test_trace_reproduction_requires_same_metrics_and_failure_stage():
    trace = dict(
        best_rpn_riou=0.55, best_decoded_riou=0.67,
        resolved_failure_stage='NMS_SUPPRESSION',
        final_post_valid_metrics=dict(top1_hit=False))
    attribution = dict(
        rpn_best_riou=0.55001, decoded_best_riou=0.67001,
        attribution='ROI_ORDERING_OR_NMS')
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
        source_native_quality_cache_only=True,
        source_native_quality_roi_chunk_size=256,
        source_native_quality_empty_cache_interval=10,
        native_candidate_trace_audit=False,
        skip_target_eval=True, s7_residual=False,
        source_train_datasets=['train:train'],
        source_val_datasets=['val:val'],
        patch_size=14, dino_height=600, dino_max_long_side=1333,
        dinov2_model='dinov2_vitl14',
        feature_cache_dir=str(tmp_path / 'feature_cache'),
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
    args.source_native_quality_cache_only = False
    with pytest.raises(ValueError, match='requires cache-only safe mode'):
        labeller.validate_source_native_quality_feasibility_args(args)
    args.source_native_quality_cache_only = True
    (tmp_path / 'source_quality.json').write_text('{}')
    with pytest.raises(ValueError, match='Refusing to overwrite'):
        labeller.validate_source_native_quality_feasibility_args(args)


def test_source_quality_cache_preflight_requires_all_valid_entries(tmp_path):
    head = tmp_path / 'head.pth'
    dino = tmp_path / 'dino.pth'
    image = tmp_path / 'val_seq01_00001.jpg'
    annotation = tmp_path / 'val_seq01_00001.txt'
    head.write_bytes(b'head')
    dino.write_bytes(b'dino')
    image.write_bytes(b'image')
    annotation.write_text('annotation')
    args = _source_audit_args(tmp_path, head, dino)
    record = dict(
        split='val', seq='val_seq01', frame=1,
        image=str(image), annotation=str(annotation))
    with pytest.raises(RuntimeError, match='cache is incomplete'):
        labeller.validate_source_feature_cache([record], args)

    cache = Path(labeller.cache_path(record, args))
    cache.parent.mkdir(parents=True)
    torch.save(dict(
        signature=labeller.cache_signature(record, args),
        feature=torch.ones((1, 1024, 4, 5), dtype=torch.float16),
        dino_meta={}), cache)
    result = labeller.validate_source_feature_cache([record], args)
    assert result['all_entries_valid'] is True
    assert result['in_channels'] == 1024


def test_chunked_roi_decode_preserves_candidate_order():
    class FakeHeads:
        def _decode_roi_candidates(
                self, feature, img_meta, proposals, rescale):
            del feature, img_meta, rescale
            value = proposals[:, :1]
            return value, value + 10, value + 20, value + 30

    proposals = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    outputs = labeller.FrozenDinoRotatedHeads._decode_roi_candidates_chunked(
        FakeHeads(), torch.empty(0), {}, proposals, True, 2)
    assert len(outputs) == 4
    assert outputs[0].reshape(-1).tolist() == [0, 2, 4, 6, 8]
    assert outputs[3].reshape(-1).tolist() == [30, 32, 34, 36, 38]


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
