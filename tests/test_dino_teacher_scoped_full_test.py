import argparse
import json
import os

import numpy as np

from crane_project.tools import dino_teacher_scoped_full_test as full_test


def _record(tmp_path, seq='real_seq02', frame=137):
    image = tmp_path / '{}_{:05d}.png'.format(seq, frame)
    image.write_bytes(b'image')
    return dict(
        split='test', seq=seq, frame=frame, image=str(image),
        annotation=str(tmp_path / 'ann.txt'))


def _scope(tmp_path, target_derived, eligible):
    path = tmp_path / 'scope.json'
    path.write_text(json.dumps(dict(
        scope_source='target_dev_protocol' if target_derived
        else 'acquisition_metadata',
        target_label_derived=target_derived,
        eligible_for_final_test=eligible,
        entries=[dict(
            split='test', seq='real_seq02', start=137, end=137,
            dino_enabled=True)])), encoding='utf-8')
    return str(path)


def test_final_scope_rejects_target_derived_manifest_without_confirmation(
        tmp_path):
    args = argparse.Namespace(
        scope_manifest=_scope(tmp_path, True, False),
        confirm_diagnosis_scope=False)
    try:
        full_test.load_final_scope(args, [_record(tmp_path)])
    except ValueError as error:
        assert 'diagnosis-only' in str(error)
    else:
        raise AssertionError('Expected target-derived scope rejection')


def test_diagnosis_scope_is_explicitly_labelled(tmp_path):
    args = argparse.Namespace(
        scope_manifest=_scope(tmp_path, True, False),
        confirm_diagnosis_scope=True)
    scope = full_test.load_final_scope(args, [_record(tmp_path)])
    assert scope['diagnosis_only'] is True
    assert scope['eligible_for_final_test'] is False


def test_external_scope_is_final_test_eligible(tmp_path):
    args = argparse.Namespace(
        scope_manifest=_scope(tmp_path, False, True),
        confirm_diagnosis_scope=False)
    scope = full_test.load_final_scope(args, [_record(tmp_path)])
    assert scope['diagnosis_only'] is False
    assert scope['eligible_for_final_test'] is True


def test_dota_export_and_pickle_keep_one_class_structure(tmp_path):
    records = [_record(tmp_path)]
    rows = [dict(policies=dict(scoped_dino_primary=dict(
        detections=[[10.0, 20.0, 30.0, 12.0, 0.0, 0.9]])))]
    task_dir = full_test.write_dota_predictions(
        rows, 'scoped_dino_primary', records, str(tmp_path / 'preds'))
    files = os.listdir(task_dir)
    assert files == ['real_seq02_00137.txt']
    line = (tmp_path / 'preds' / 'Task1_grab' /
            'real_seq02_00137.txt').read_text(encoding='utf-8').strip()
    assert len(line.split()) == 9

    output = tmp_path / 'results.pkl'
    full_test.write_results_pickle(
        rows, 'scoped_dino_primary', str(output))
    import pickle
    with output.open('rb') as handle:
        payload = pickle.load(handle)
    assert len(payload) == 1
    assert len(payload[0]) == 1
    assert payload[0][0].shape == (1, 6)


def test_miss_run_summary_breaks_on_frame_gaps():
    summary = full_test.miss_run_summary(
        [1, 2, 3, 5, 6], [False, True, False, False, False])
    assert summary['mcml'] == 2
    assert summary['longest_intervals'] == [
        {'start': 5, 'end': 6, 'length': 2}]


def test_sequence_failure_analysis_reports_three_metrics(
        monkeypatch, tmp_path):
    records = []
    rows = []
    for frame in (1, 2, 3, 4, 6, 7):
        record = _record(tmp_path, frame=frame)
        record['domain'] = 'real'
        records.append(record)
        output = [] if frame in (1, 6, 7) else [[0, 0, 10, 10, 0, 0.9]]
        rows.append(dict(
            seq='real_seq02', frame=frame,
            policies=dict(baseline=dict(
                detections=output,
                metrics=dict(
                    top1_hit=frame == 2,
                    top1_riou=0.8 if frame == 2 else 0.2)))))

    monkeypatch.setattr(
        full_test.labeller, 'parse_original_gt',
        lambda _path: np.asarray([[0, 0, 10, 10, 0]], dtype=np.float32))
    report = full_test.sequence_failure_analysis(
        rows, records, 'baseline')['real_seq02']
    assert report['silence_mcml']['mcml'] == 2
    assert report['silence_mcml']['longest_intervals'] == [
        {'start': 6, 'end': 7, 'length': 2}]
    assert report['center_mcml']['mcml'] == 2
    assert report['rotated_iou_mcml']['mcml'] == 2


def test_compact_sequence_mcml_keeps_paper_fields_only():
    metric = dict(
        mcml=3,
        longest_intervals=[{'start': 10, 'end': 12, 'length': 3}])
    compact = full_test.compact_sequence_mcml(dict(
        baseline=dict(real_seq02=dict(
            silence_mcml=metric,
            center_mcml=metric,
            rotated_iou_mcml=metric))))
    assert compact == dict(baseline=dict(real_seq02=dict(
        silence_mcml=3,
        silence_longest='10..12',
        center_mcml=3,
        center_longest='10..12',
        rotated_iou_mcml=3,
        rotated_iou_longest='10..12')))


def test_explicit_center_metrics_separates_denominators_and_thresholds():
    metrics = full_test.explicit_center_metrics(
        {'real/R_center(%)': 98.91, 'sim/R_center(%)': 100.0},
        {'real_R_center': 0.6476, 'sim_R_center': 1.0})
    assert metrics == dict(
        real_R_center_det_at_15px_percent=98.91,
        sim_R_center_det_at_15px_percent=100.0,
        real_R_center_all_at_25px_percent=64.76,
        sim_R_center_all_at_10px_percent=100.0)
