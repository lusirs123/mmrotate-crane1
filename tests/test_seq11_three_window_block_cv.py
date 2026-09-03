import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from crane_project.tools import (
    symeood_dino_seq11_block_cv_select as selector,
    symeood_dino_seq11_block_cv_support_audit as support)
from crane_project.tools.symeood_dino_seq11_block_cv_materialize import (
    load_manifest, materialize)


ROOT = Path(__file__).resolve().parents[1]
CV_MANIFEST = ROOT / (
    'crane_project/data_contracts/'
    'real_seq11_pilot_k1p9_three_window_block_cv_v1.json')
OLD_MANIFEST = ROOT / (
    'crane_project/data_contracts/'
    'real_seq11_pilot_k1p9_blocksplit_v1.json')


def _result(box):
    detections = (np.zeros((0, 6), dtype=np.float32) if box is None else
                  np.asarray([list(box) + [0.9]], dtype=np.float32))
    return [detections]


def _annotation(path):
    path.write_text('8 18 12 18 12 22 8 22 grab 0\n')


def test_manifest_has_three_disjoint_windows_and_preserves_fold3():
    manifest = load_manifest(CV_MANIFEST)
    folds = manifest['folds']
    assert [len(row['validation_stems']) for row in folds] == [10, 12, 11]
    assert len(manifest['validation_union']) == 33
    assert 'real_seq11_006710' in folds[2]['validation_stems']
    assert 'real_seq11_006711' in folds[2]['validation_stems']


def test_materializer_builds_three_exact_disjoint_views(tmp_path):
    old = json.loads(OLD_MANIFEST.read_text())
    frames = sorted(old['aux_train_frames'] + old['aux_val_frames'])
    data_root = tmp_path / 'data'
    source_root = data_root / 'source'
    (source_root / 'images').mkdir(parents=True)
    (source_root / 'annfiles').mkdir()
    records = []
    for frame in frames:
        stem = 'real_seq11_{:06d}'.format(frame)
        image = source_root / 'images' / (stem + '.jpg')
        image.write_bytes(('image-{}'.format(frame)).encode())
        _annotation(source_root / 'annfiles' / (stem + '.txt'))
        records.append(dict(
            filename=str(image), sequence='real_seq11', frame=frame,
            dino_invoked=True,
            dino_native_box=[10.0, 20.0, 4.0, 4.0, 0.0, 0.9],
            sym_eood_box=[10.0, 20.0, 4.0, 4.0, 0.0, 0.8]))
    audit_path = tmp_path / 'audit.json'
    audit_path.write_text(json.dumps(dict(
        protocol='source_owned_geometry_union_v2', records=records)))
    args = argparse.Namespace(
        data_root=str(data_root), source_split='source',
        audit_json=str(audit_path), cv_manifest=str(CV_MANIFEST),
        out_root=str(tmp_path / 'out'), mode='hardlink')

    report = materialize(args)

    assert report['passed'] is True
    assert report['oof_frame_count'] == 33
    assert report['clean_full59']['frame_count'] == 59
    clean_root = data_root / report['clean_full59']['split_name']
    assert len(list((clean_root / 'images').glob('*.jpg'))) == 59
    assert len(list((clean_root / 'annfiles').glob('*.txt'))) == 59
    assert [row['train']['frame_count']
            for row in report['fold_reports']] == [49, 47, 48]
    assert [row['val']['frame_count']
            for row in report['fold_reports']] == [10, 12, 11]
    assert all(row['train_val_overlap_count'] == 0
               for row in report['fold_reports'])


def test_support_audit_separates_missing_and_present_wrong(tmp_path):
    cv = load_manifest(CV_MANIFEST)
    old = json.loads(OLD_MANIFEST.read_text())
    frames = sorted(old['aux_train_frames'] + old['aux_val_frames'])
    data_root = tmp_path / 'data'
    source_root = data_root / 'source'
    (source_root / 'annfiles').mkdir(parents=True)
    (source_root / 'images').mkdir()
    gt = [10.0, 20.0, 4.0, 4.0, 0.0]
    bad = [40.0, 40.0, 4.0, 4.0, 0.0]
    present_wrong = set(sorted(cv['folds'][0]['validation_stems'])[:3])
    present_wrong.update(sorted(cv['folds'][1]['validation_stems'])[:3])
    missing = set(sorted(cv['folds'][2]['validation_stems'])[:7])
    results, records = [], []
    for frame in frames:
        stem = 'real_seq11_{:06d}'.format(frame)
        _annotation(source_root / 'annfiles' / (stem + '.txt'))
        (source_root / 'images' / (stem + '.jpg')).write_bytes(b'jpg')
        if stem in missing:
            k1_box = None
        elif stem in present_wrong:
            k1_box = bad
        else:
            k1_box = gt
        results.append(_result(k1_box))
        records.append(dict(
            filename=str(source_root / 'images' / (stem + '.jpg')),
            sequence='real_seq11', frame=frame, dino_invoked=True,
            dino_native_box=gt + [0.9], sym_eood_box=gt + [0.8]))
    result_path = tmp_path / 'k1.pkl'
    result_path.write_bytes(pickle.dumps(results))
    audit_path = tmp_path / 'audit.json'
    audit_path.write_text(json.dumps(dict(
        protocol='source_owned_geometry_union_v2', records=records)))
    args = argparse.Namespace(
        k1_results=str(result_path), audit_json=str(audit_path),
        cv_manifest=str(CV_MANIFEST), data_root=str(data_root),
        source_split='source', min_pooled_present_wrong=6,
        min_present_wrong_folds=2, min_pooled_missing=3,
        min_missing_folds=1, out_json=str(tmp_path / 'out.json'))

    report = support.audit(args)

    assert report['passed'] is True
    assert report['pooled_oof_33']['category_counts'][
        'k1_present_wrong_dino_hit'] == 6
    assert report['pooled_oof_33']['category_counts'][
        'k1_missing_dino_hit'] == 7
    assert report['present_wrong_supported_fold_count'] == 2
    assert report['missing_supported_fold_count'] == 1
    assert report['temporal_metrics_computed'] is False


def _gate(tmp_path, epoch, passed):
    payload = dict(
        protocol=selector.GATE_PROTOCOL,
        evidence_boundary=(
            'three_legacy_source_val_streams_plus_pooled_sparse_oof_33'),
        target_data_read=False, fixed_test_read=False,
        temporal_metrics_computed_on_oof=False,
        eligible_for_checkpoint_promotion=False,
        eligible_for_fixed_test=False,
        eligible_for_unknown_sequence_claim=False,
        epoch=epoch, passed=passed,
        input=dict(cv_manifest_sha256='manifest',
                   support_audit_sha256='support'),
        pooled_oof=dict(
            present_wrong_rescue_rate=0.2 + epoch / 100.0,
            missing_rescue_rate=0.9,
            candidate_mean_riou=0.7),
        folds=[
            dict(
                fold_id=fold_id,
                input=dict(checkpoint='fold{}_epoch_{}.pth'.format(
                    fold_id, epoch), checkpoint_sha256='sha'),
                source_val_metrics={
                    'real/mean_RIoU': 0.78,
                    'sim/mean_RIoU': 0.88,
                    'real/DFR(%/frame)': 3.0,
                    'sim/DFR(%/frame)': 2.2})
            for fold_id in range(1, 4)])
    path = tmp_path / 'epoch{}_gate.json'.format(epoch)
    path.write_text(json.dumps(payload))
    return str(path)


def test_selector_only_selects_passed_epoch(tmp_path):
    paths = [_gate(tmp_path, epoch, passed=epoch in (7, 9))
             for epoch in range(1, 11)]
    report = selector.select(paths)
    assert report['passing_epochs'] == [7, 9]
    assert report['selected']['epoch'] == 9
    assert report['eligible_for_final_all59_source_training'] is True
    assert report['eligible_for_fixed_test'] is False


def test_configs_require_explicit_fold_and_never_reference_test_split():
    train = (ROOT / ('crane_project/configs/'
                     'crane_symeood_dino_k1_retentive_causal_phase_refiner_'
                     'source_v3_seq11_blockcv.py')).read_text()
    val = (ROOT / ('crane_project/configs/'
                   'crane_symeood_dino_k1_retentive_causal_phase_refiner_'
                   'seq11_blockcv_val.py')).read_text()
    assert "SEQ11_CV_FOLD', '0'" in train
    assert 'source_only_k1_retentive_v3_seq11_blockcv_v1' in train
    assert "ann_file='test/annfiles/'" not in train + val
    assert "expected_split='test'" not in train + val
