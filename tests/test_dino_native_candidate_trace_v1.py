import hashlib
import json
import sys
import types
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
                 'USABLE_CANDIDATE_SURVIVES')
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
