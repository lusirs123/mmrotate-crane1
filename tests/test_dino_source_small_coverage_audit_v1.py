import json
from types import SimpleNamespace

import pytest

from crane_project.tools import dino_source_small_coverage_audit_v1 as audit


def _row(role, domain, seq, small, frame=0):
    return dict(
        role=role, annotation_split='train', image_split='train',
        domain=domain, seq=seq, frame=frame, object_count=1,
        short_token=1.0 if small else 3.0, source_small=small,
        image_sha256='a' * 64, annotation_sha256='b' * 64)


def test_coverage_separates_train_support_from_validation_gap():
    rows = []
    for frame in range(5):
        rows.append(_row('source_train', 'real', 'real_seq01', True, frame))
        rows.append(_row('source_train', 'sim', 'sim_seq02', True, frame))
        rows.append(_row(
            'source_validation', 'sim', 'sim_seq10', True, frame))
        rows.append(_row(
            'source_validation', 'real', 'real_seq07', False, frame))
    summary = audit.summarize_coverage(rows)
    assert summary['development_coverage']['source_train']['passed'] is True
    assert summary['development_coverage'][
        'source_validation']['passed'] is False
    assert summary['decision'] == (
        'SOURCE_TRAIN_AVAILABLE_VALIDATION_COVERAGE_INSUFFICIENT')


def test_coverage_requires_real_and_independent_sequences():
    rows = [_row('source_train', 'sim', 'sim_seq10', True, frame)
            for frame in range(10)]
    summary = audit.summarize_coverage(rows)
    gate = summary['development_coverage']['source_train']
    assert gate['qualifying_sequence_count'] == 1
    assert gate['qualifying_real_sequence_count'] == 0
    assert gate['passed'] is False


def test_frozen_definition_rejects_dataset_identity_mismatch(tmp_path):
    report = tmp_path / 'report.json'
    report.write_text(json.dumps(dict(
        audit='Frozen DINO Native-S14 Source Quality Feasibility Audit V2',
        protocol_version=2,
        decision=(
            'SOURCE_NATIVE_QUALITY_SUPPORT_INSUFFICIENT_COLLECT_NEW_'
            'LABELED_SEQUENCES_TARGET_NOT_READ'),
        protocol=dict(
            target_read=False,
            source_data=dict(
                train_datasets=['train:train'],
                val_datasets=['val:val']),
            source_small_definition=dict(
                definition='source_train_short_token_lower_tertile',
                short_token_threshold=1.67)))), encoding='utf-8')
    args = SimpleNamespace(
        source_train_datasets=['other:train'],
        source_val_datasets=['val:val'])
    with pytest.raises(RuntimeError, match='train dataset identity'):
        audit.load_frozen_definition(str(report), args)


def test_dota_short_side_reader_uses_grab_polygon_edges(tmp_path):
    annotation = tmp_path / 'frame.txt'
    annotation.write_text(
        '0 0 10 0 10 4 0 4 grab 0\n'
        '0 0 20 0 20 8 0 8 other 0\n', encoding='utf-8')
    assert audit.read_dota_grab_short_sides(str(annotation)) == [4.0]
