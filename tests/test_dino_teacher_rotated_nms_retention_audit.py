import copy

from crane_project.tools import (
    dino_teacher_rotated_nms_retention_audit as selector)


def _object(hit):
    return dict(post_valid_content=dict(top1_hit=hit))


def _candidate(threshold, top1, small_r20, retention=1.0,
               detections=10.0):
    return dict(
        nms_iou_thr=threshold,
        summary=dict(
            final_top1_hits=top1, post_nms_recall_at_20=0.9),
        source_small=dict(
            post_nms_recall_at_20=small_r20,
            post_nms_recall_at_100=small_r20),
        source_retention=dict(exact_retention_rate=retention),
        mean_post_nms_detection_count=detections)


def test_exact_frame_retention_uses_source_keys():
    baseline = [
        dict(split='val', seq='a', frame=1, objects=[_object(True)]),
        dict(split='val', seq='a', frame=2, objects=[_object(False)])]
    candidate = [
        dict(split='val', seq='a', frame=1, objects=[_object(False)]),
        dict(split='val', seq='a', frame=2, objects=[_object(True)])]
    result = selector.retention_against_baseline(baseline, candidate)
    assert result['baseline_correct_count'] == 1
    assert result['retained_correct_count'] == 0
    assert result['exact_retention_rate'] == 0.0


def test_source_only_selection_prefers_small_r20_with_retention_gate():
    baseline = _candidate(0.1, top1=100, small_r20=0.8)
    better = _candidate(0.3, top1=100, small_r20=0.9)
    target_better_but_source_invalid = _candidate(
        0.5, top1=101, small_r20=0.95, retention=0.99)
    result = selector.select_candidate(
        [baseline, better, target_better_but_source_invalid], baseline)
    assert result['selected']['nms_iou_thr'] == 0.3
    assert result['eligible_thresholds'] == [0.1, 0.3]


def test_source_selection_tie_prefers_fewer_detections_then_lower_iou():
    baseline = _candidate(0.1, top1=100, small_r20=0.8)
    first = _candidate(
        0.2, top1=100, small_r20=0.9, detections=20.0)
    second = copy.deepcopy(first)
    second['nms_iou_thr'] = 0.3
    second['mean_post_nms_detection_count'] = 15.0
    result = selector.select_candidate([first, second], baseline)
    assert result['selected']['nms_iou_thr'] == 0.3
