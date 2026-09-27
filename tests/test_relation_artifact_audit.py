import json
from pathlib import Path
import pytest
from crane_project.tools import audit_k1_dino_object_background_relation_v5 as audit


def test_sidecar_rejects_changed_checkpoint(tmp_path):
    config, ckpt, pkl = [tmp_path / n for n in ('config', 'checkpoint', 'pkl')]
    for p in (config, ckpt, pkl):
        p.write_bytes(b'original')
    sidecar = tmp_path / 'sidecar.json'
    data = dict(config=str(config), config_sha256=audit.sha256(config),
                checkpoint=str(ckpt), checkpoint_sha256=audit.sha256(ckpt),
                results_pkl=str(pkl), results_pkl_sha256=audit.sha256(pkl), split='fixed_test')
    sidecar.write_text(json.dumps(data))
    assert audit._sidecar_matches(sidecar, config, ckpt, pkl, 'fixed_test')[0]
    ckpt.write_bytes(b'changed')
    assert not audit._sidecar_matches(sidecar, config, ckpt, pkl, 'fixed_test')[0]


def test_log_malformed_and_nan_rejected(tmp_path):
    p = tmp_path / 'run.log.json'
    p.write_text('{broken')
    with pytest.raises(ValueError, match='Malformed'):
        audit._log_summary(tmp_path)
    p.write_text(json.dumps(dict(mode='train', loss_object_background_relation=float('nan'))))
    with pytest.raises(ValueError, match='non-finite'):
        audit._log_summary(tmp_path)


def test_logs_kept_separate(tmp_path):
    for name in ('first', 'second'):
        (tmp_path / (name + '.log.json')).write_text('\n'.join(
            json.dumps(dict(mode='train', epoch=e, iter=50)) for e in (1, 24)))
    result = audit._log_summary(tmp_path)
    assert result['train_records'] == 4
    assert all(v['train_records'] == 2 for v in result['per_file'].values())


def test_runs_split_at_frame_gap_sequence_and_recovery():
    rows = [dict(domain='real', sequence=seq, frame=f, frame_key=seq+str(f),
                 baseline=dict(output=hit))
            for seq, f, hit in [('s1', 1, False), ('s1', 2, False),
                                ('s1', 9, False), ('s1', 10, True),
                                ('s2', 11, False)]]
    assert [r['length'] for r in audit.failure_runs(rows, 'baseline', 'output')] == [2, 1, 1]


def test_paired_denominator_and_churn():
    from crane_project.tools.audit_k1_dino_student_paired_test_v1 import _summarize
    def m(output, hit):
        return dict(output=output, center_hit=hit, riou_hit=hit, riou=0.7 if hit else 0.0)
    rows = [dict(baseline=m(True, True), student=m(False, False)),
            dict(baseline=m(False, False), student=m(True, True))]
    result = _summarize(rows)
    assert result['methods']['baseline']['center_hit_given_output'] == 1.0
    assert result['methods']['baseline']['output_coverage'] == 0.5
    assert result['paired']['baseline_only_center_hit_count'] == 1
    assert result['paired']['student_only_center_hit_count'] == 1


def test_prediction_chain_checks_gt_and_export_without_mutation(tmp_path):
    import pickle
    import numpy as np
    from crane_project.tools.ckpt_sweep import (
        prediction_provenance, dota_export_sha256)
    root = tmp_path
    gt = root / 'crane_project/data/crane_grab/val/annfiles'
    gt.mkdir(parents=True)
    (gt / 'real_seq01_00001.txt').write_text('0 0 2 0 2 2 0 2 grab 0\n')
    config, ckpt = root / 'config.py', root / 'epoch_16.pth'
    config.write_text('x=1')
    ckpt.write_bytes(b'checkpoint')
    pred = root / 'preds'
    pred.mkdir()
    pkl = pred / 'results.pkl'
    pkl.write_bytes(pickle.dumps([[np.empty((0, 6))]]))
    sidecar = Path(str(pkl) + '.provenance.json')
    sidecar.write_text(json.dumps(prediction_provenance(
        str(config), str(ckpt), str(pkl), str(gt), 'source_val')))
    dota = pred / 'Task1_grab'
    dota.mkdir()
    (dota / 'real_seq01_00001.txt').write_text('')
    export = Path(str(dota) + '.provenance.json')
    export.write_text(json.dumps(dict(protocol='crane_dota_export_provenance_v1',
        results_pkl_sha256=audit.sha256(pkl), frame_ids=['real_seq01_00001'],
        dota_sha256=dota_export_sha256(str(dota), ['real_seq01_00001']))))
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    audit.validate_prediction(root, config, ckpt, pkl, 'val')
    assert before == {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    (dota / 'real_seq01_00001.txt').write_text('tampered')
    with pytest.raises(ValueError, match='DOTA provenance'):
        audit.validate_prediction(root, config, ckpt, pkl, 'val')
    (dota / 'real_seq01_00001.txt').write_text('')
    (gt / 'real_seq01_00001.txt').write_text('tampered GT')
    with pytest.raises(RuntimeError, match='provenance mismatch'):
        audit.validate_prediction(root, config, ckpt, pkl, 'val')
