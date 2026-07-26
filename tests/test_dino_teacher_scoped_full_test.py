import argparse
import json
import os

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
