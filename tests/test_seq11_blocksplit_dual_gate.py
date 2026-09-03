import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from crane_project.tools import (
    symeood_dino_seq11_aux_mechanism_gate as aux_gate,
    symeood_dino_seq11_aux_support_audit as support_audit,
    symeood_dino_seq11_block_split as splitter,
    symeood_dino_seq11_dual_source_select as selector)
from crane_project.tools.symeood_dino_causal_history_source_gate import (
    _seq11_blocksplit_legacy_preservation)


def _dota_annotation(path):
    path.write_text('8 18 12 18 12 22 8 22 grab 0\n')


def _manifest(path, train_frames, val_frames):
    all_frames = sorted(set(train_frames) | set(val_frames))
    payload = dict(
        protocol=splitter.PROTOCOL,
        sequence='real_seq11', source_split='source',
        all_frame_count=len(all_frames),
        aux_train_frame_count=len(train_frames),
        aux_val_frame_count=len(val_frames),
        holdout_rule=dict(
            kind='closed_source_frame_interval_with_near_neighbor_buffer',
            start_frame=min(val_frames), end_frame=max(val_frames)),
        aux_train_frames=train_frames, aux_val_frames=val_frames,
        temporal_metrics_authorized=False,
        independent_sequence_claim_authorized=False,
        fixed_test_use_authorized=False)
    path.write_text(json.dumps(payload))
    return payload


def _record(root, frame, dino_box=None):
    stem = 'real_seq11_{:06d}'.format(frame)
    return dict(
        filename=str(root / 'images' / (stem + '.jpg')),
        sequence='real_seq11', frame=frame, dino_invoked=True,
        dino_native_box=(dino_box if dino_box is not None else
                         [10.0, 20.0, 4.0, 4.0, 0.0, 0.9]),
        sym_eood_box=[10.0, 20.0, 4.0, 4.0, 0.0, 0.8])


def test_materializer_creates_disjoint_filtered_views(tmp_path):
    data_root = tmp_path / 'data'
    source = data_root / 'source'
    (source / 'images').mkdir(parents=True)
    (source / 'annfiles').mkdir()
    train_frames, val_frames = [1, 2], [10, 11]
    manifest = tmp_path / 'manifest.json'
    _manifest(manifest, train_frames, val_frames)
    for frame in train_frames + val_frames:
        stem = 'real_seq11_{:06d}'.format(frame)
        (source / 'images' / (stem + '.jpg')).write_bytes(
            b'image' + bytes([frame]))
        _dota_annotation(source / 'annfiles' / (stem + '.txt'))
    audit = tmp_path / 'audit.json'
    audit.write_text(json.dumps(dict(
        protocol=splitter.ALL_LANE_PROTOCOL,
        records=[_record(source, frame)
                 for frame in train_frames + val_frames])))
    args = argparse.Namespace(
        data_root=str(data_root), source_split='source',
        train_split='train_view', val_split='val_view',
        audit_json=str(audit), split_manifest=str(manifest),
        train_audit_json=str(tmp_path / 'train_audit.json'),
        val_audit_json=str(tmp_path / 'val_audit.json'),
        out_json=str(tmp_path / 'report.json'), mode='hardlink')

    report = splitter.materialize(args)

    assert report['aux_train_frame_count'] == 2
    assert report['aux_val_frame_count'] == 2
    assert report['train_val_overlap_count'] == 0
    train_audit = json.loads((tmp_path / 'train_audit.json').read_text())
    val_audit = json.loads((tmp_path / 'val_audit.json').read_text())
    assert len(train_audit['records']) == 2
    assert len(val_audit['records']) == 2
    assert {Path(row['filename']).stem for row in train_audit['records']} == {
        'real_seq11_000001', 'real_seq11_000002'}
    assert {Path(row['filename']).stem for row in val_audit['records']} == {
        'real_seq11_000010', 'real_seq11_000011'}


def _result(box):
    array = np.zeros((0, 6), dtype=np.float32) if box is None else np.asarray(
        [list(box) + [0.9]], dtype=np.float32)
    return [array]


def test_aux_gate_uses_formal_k1_and_never_temporal_metrics(
        tmp_path, monkeypatch):
    frames = list(range(100, 111))
    manifest_path = tmp_path / 'manifest.json'
    _manifest(manifest_path, [], frames)
    manifest = splitter._manifest(manifest_path)
    data_root = tmp_path / 'data'
    split = data_root / 'val_view'
    (split / 'annfiles').mkdir(parents=True)
    (split / 'images').mkdir()
    gt = [10.0, 20.0, 4.0, 4.0, 0.0]
    bad = [40.0, 40.0, 4.0, 4.0, 0.0]
    records = []
    candidate_results, k1_results = [], []
    for index, frame in enumerate(frames):
        stem = 'real_seq11_{:06d}'.format(frame)
        _dota_annotation(split / 'annfiles' / (stem + '.txt'))
        (split / 'images' / (stem + '.jpg')).write_bytes(b'jpg')
        records.append(_record(split, frame, dino_box=gt + [0.9]))
        candidate_results.append(_result(gt))
        k1_results.append(_result(bad if index < 3 else gt))
    audit_path = tmp_path / 'val_audit.json'
    audit_path.write_text(json.dumps(dict(
        protocol=splitter.ALL_LANE_PROTOCOL,
        auxiliary_split_role='aux-val',
        auxiliary_split_manifest_sha256=manifest['sha256'],
        records=records)))
    candidate_path = tmp_path / 'candidate.pkl'
    k1_path = tmp_path / 'k1.pkl'
    candidate_path.write_bytes(pickle.dumps(candidate_results))
    k1_path.write_bytes(pickle.dumps(k1_results))
    checkpoint = tmp_path / 'epoch_1.pth'
    checkpoint.write_bytes(b'checkpoint')
    contract = dict(
        auxiliary_split_manifest_sha256=manifest['sha256'])
    monkeypatch.setattr(
        aux_gate, '_checkpoint_contract',
        lambda *_args, **_kwargs: (
            str(checkpoint), 'checkpoint-sha', contract,
            False, True, False, True, True))
    args = argparse.Namespace(
        candidate_results=str(candidate_path),
        candidate_checkpoint=str(checkpoint),
        k1_reference_results=str(k1_path),
        aux_val_audit=str(audit_path),
        split_manifest=str(manifest_path), data_root=str(data_root),
        aux_val_split='val_view', expected_candidate_sha256=None,
        min_hard_support=3, out_json=str(tmp_path / 'out.json'))

    report = aux_gate.audit(args)

    assert report['passed'] is True
    assert report['metrics']['hard_rescued_hit_count'] == 3
    assert report['metrics']['k1_good_lost_count'] == 0
    assert report['temporal_metrics_computed'] is False
    assert 'MCML' not in json.dumps(report)

    support_args = argparse.Namespace(
        k1_reference_results=str(k1_path), aux_val_audit=str(audit_path),
        split_manifest=str(manifest_path), data_root=str(data_root),
        aux_val_split='val_view', min_hard_support=3,
        out_json=str(tmp_path / 'support.json'))
    support = support_audit.audit(support_args)
    assert support['passed'] is True
    assert support['k1_present_wrong_dino_hit_count'] == 3
    assert support['eligible_for_blocksplit_training'] is True


def _metrics():
    return {
        'real/R_center(%)': 99.0, 'sim/R_center(%)': 100.0,
        'real/mean_RIoU': 0.80, 'sim/mean_RIoU': 0.90,
        'real/DFR(%/frame)': 2.5, 'sim/DFR(%/frame)': 2.0,
        'real/ACI': 0.94, 'sim/ACI': 0.96,
        'sim/A-RMSE(deg)': 1.5,
        'real/TDR_w10(%)': 100.0, 'sim/TDR_w10(%)': 100.0,
        'real/MCML_max(frames)': 1, 'sim/MCML_max(frames)': 0}


def test_legacy_guardrails_allow_small_tradeoff_but_not_large_drop():
    reference = _metrics()
    candidate = dict(reference)
    candidate.update({
        'real/mean_RIoU': 0.795,
        'sim/DFR(%/frame)': 2.20,
        'sim/ACI': 0.958})
    assert _seq11_blocksplit_legacy_preservation(
        candidate, reference)['passed'] is True
    candidate['real/mean_RIoU'] = 0.78
    assert _seq11_blocksplit_legacy_preservation(
        candidate, reference)['passed'] is False


def _gate_file(tmp_path, epoch, kind, passed=True, rescued=2, gain=0.1):
    checkpoint = str(tmp_path / 'epoch_{}.pth'.format(epoch))
    if kind == 'legacy':
        payload = dict(
            protocol=selector.LEGACY_PROTOCOL,
            evidence_boundary='legacy_source_val_738_only',
            target_data_read=False, fixed_test_read=False,
            eligible_for_checkpoint_promotion=False,
            eligible_for_fixed_test=False,
            eligible_for_unknown_sequence_claim=False,
            passed=passed,
            input=dict(candidate_checkpoint=checkpoint,
                       candidate_checkpoint_sha256='sha{}'.format(epoch)),
            candidate_metrics=dict(
                _metrics(), **{
                    'real/MCML_max(frames)': 0,
                    'sim/MCML_max(frames)': 0}))
    else:
        payload = dict(
            protocol=selector.AUX_PROTOCOL,
            evidence_boundary='same_video_heldout_auxiliary_block_only',
            target_data_read=False, fixed_test_read=False,
            eligible_for_checkpoint_promotion=False,
            eligible_for_fixed_test=False,
            eligible_for_unknown_sequence_claim=False,
            passed=passed,
            input=dict(candidate_checkpoint=checkpoint,
                       candidate_checkpoint_sha256='sha{}'.format(epoch)),
            metrics=dict(hard_rescued_hit_count=rescued,
                         hard_mean_riou_gain=gain))
    path = tmp_path / '{}_{}.json'.format(kind, epoch)
    path.write_text(json.dumps(payload))
    return str(path)


def test_dual_selector_requires_both_halves_and_prioritizes_aux_gain(tmp_path):
    legacy, aux = [], []
    for epoch in range(1, 11):
        legacy.append(_gate_file(
            tmp_path, epoch, 'legacy', passed=epoch != 7))
        aux.append(_gate_file(
            tmp_path, epoch, 'aux', passed=epoch != 8,
            rescued=4 if epoch in (7, 9) else 2,
            gain=0.2 if epoch == 9 else 0.1))

    report = selector.select(legacy, aux)

    assert report['selected']['epoch'] == 9
    assert 7 not in report['passing_epochs']
    assert 8 not in report['passing_epochs']
    assert report['eligible_for_fixed_test'] is False


def test_dual_selector_stops_if_no_epoch_passes_both(tmp_path):
    legacy = [_gate_file(tmp_path, epoch, 'legacy', passed=epoch <= 5)
              for epoch in range(1, 11)]
    aux = [_gate_file(tmp_path, epoch, 'aux', passed=epoch > 5)
           for epoch in range(1, 11)]
    report = selector.select(legacy, aux)
    assert report['passed'] is False
    assert report['selected'] is None
    assert report['passing_epochs'] == []
    assert report['eligible_for_checkpoint_promotion'] is False
    assert report['decision'] == (
        'STOP_E1_NO_EPOCH_PASSED_BOTH_SOURCE_HALVES')
