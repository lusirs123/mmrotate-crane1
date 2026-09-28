import json
import pickle

import numpy as np
import pytest

from crane_project.tools import audit_teacher_student_source_val_gap_v1 as audit


def teacher_file(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'FRAME_COUNT', 1)
    row = dict(seq='real_seq07', frame=1, top1_hit=True,
               top1_riou=0.8, top1_score=0.9)
    payload = dict(protocol=dict(source_val_datasets=['val:val']),
                   selected_alpha=0.5,
                   source=dict(candidates=[dict(alpha=0.5, frame_outcomes=[row])]))
    p = tmp_path / 'teacher.json'
    p.write_text(json.dumps(payload))
    return p, payload, row


def test_teacher_metric_and_alpha_consistency(tmp_path, monkeypatch):
    p, payload, row = teacher_file(tmp_path, monkeypatch)
    assert audit.load_teacher_outcomes(p)[0]['real_seq07|1']['correct']
    with pytest.raises(ValueError, match='already selected'):
        audit.load_teacher_outcomes(p, 0.75)
    row['top1_hit'] = False
    p.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='disagrees'):
        audit.load_teacher_outcomes(p)


def test_teacher_rejects_nonfinite_and_target_protocol(tmp_path, monkeypatch):
    p, payload, row = teacher_file(tmp_path, monkeypatch)
    row['top1_riou'] = float('nan')
    p.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        audit.load_teacher_outcomes(p)
    payload['protocol']['target_data_read'] = True
    p.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='protocol'):
        audit.load_teacher_outcomes(p)


def test_saved_output_validation_and_geometry():
    gt = np.array([20., 30., 10., 5., 0.])
    assert audit.student_outcome([np.array([[20, 30, 10, 5, 0, .7]])], gt)['correct']
    assert not audit.student_outcome([np.empty((0, 6))], gt)['output']
    for arr in (np.empty((0, 4)), np.array([[20, 30, -1, 5, 0, .7]]),
                np.array([[20, 30, 10, 5, 0, float('nan')]])):
        with pytest.raises(ValueError):
            audit.student_outcome([arr], gt)


def test_sidecar_missing_or_bad_hash_is_not_verified(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'FRAME_COUNT', 1)
    p = tmp_path / 'results.pkl'
    p.write_bytes(pickle.dumps([[np.empty((0, 6))]]))
    assert audit.load_student_predictions(p)[1] is None
    sidecar = tmp_path / 'results.pkl.provenance.json'
    sidecar.write_text('{}')
    with pytest.raises(ValueError, match='provenance'):
        audit.load_student_predictions(p)


def test_epoch24_predictions_cannot_be_labelled_epoch20(tmp_path, monkeypatch):
    gt_dir = tmp_path / 'annfiles'
    gt_dir.mkdir()
    teacher = {}
    for seq, count in [('real_seq07', 226), ('sim_seq10', 512)]:
        for i in range(count):
            (gt_dir / ('%s_%05d.txt' % (seq, i))).write_text('')
            teacher[audit.frame_key(seq, i)] = {}
    monkeypatch.setattr(audit, 'load_teacher_outcomes', lambda *args: (teacher, {}))
    monkeypatch.setattr(audit, 'load_student_predictions',
                        lambda *args: ([None] * 738, None, 'unverified'))
    ckpt = tmp_path / 'epoch_20.pth'
    ckpt.write_bytes(b'fixture')
    with pytest.raises(ValueError, match='epoch identity mismatch'):
        audit.build_report(gt_dir, 'unused', 'source_val_epoch24_results.pkl',
                           'historical', student_checkpoint=ckpt,
                           allow_unverified_student=True)
    with pytest.raises(ValueError, match='generation identity is unverified'):
        audit.build_report(gt_dir, 'unused', 'results.pkl', 'historical')
