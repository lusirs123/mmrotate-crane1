import csv
import json
import sys

import pytest

from crane_project.tools import dino_native_s14_failure_attribution_v1 as audit


HEAD_SHA = 'a' * 64
DINO_SHA = 'b' * 64


def _object(cause, top1_hit, rpn_riou=0.6, decoded_riou=0.7):
    rank = 1 if top1_hit else None
    return dict(
        attrition_cause=cause,
        rpn=dict(best_riou=rpn_riou, best_usable_rank=2),
        roi_regression=dict(
            best_decoded_riou=decoded_riou,
            decoded_usable_count=3 if decoded_riou >= 0.5 else 0,
            decoded_usable_foreground_rank=2 if decoded_riou >= 0.5 else None),
        post_nms=dict(best_usable_rank=rank),
        post_valid_content=dict(
            top1_hit=top1_hit, best_usable_rank=rank,
            best_riou=0.65 if top1_hit else None),
        token_scale=dict(short_token=1.2, long_token=3.4),
        source_token_bin='source_small')


def _inputs():
    specification = dict(
        name='small', split='test', seq='real_seq03', start=1, end=3)
    causes = [
        ('ROI_TOP1_RESTORED', True, 0.6, 0.7),
        ('ROI_ORDERING_OR_NMS_REMOVES_GEOMETRY', False, 0.6, 0.7),
        ('ROI_REGRESSION_DESTROYS_RPN_GEOMETRY', False, 0.6, 0.4),
    ]
    attrition_rows = []
    coverage_rows = []
    for frame, (cause, hit, rpn, decoded) in enumerate(causes, 1):
        attrition_rows.append(dict(
            split='test', seq='real_seq03', frame=frame,
            counts=dict(rpn_proposals=10, roi_decoded=10,
                        post_nms=3, post_valid=2),
            objects=[_object(cause, hit, rpn, decoded)]))
        coverage_rows.append(dict(
            split='test', seq='real_seq03', frame=frame,
            objects=[dict(anchor=dict(
                max_hbb_iou=0.8, exact_assigned_positive_count=1))]))
    common_protocol = dict(
        riou_thr=0.5, target_used_for_training=False,
        target_used_for_checkpoint_selection=False)
    attrition = dict(
        audit=audit.EXPECTED_ATTRITION_AUDIT, protocol_version=2,
        labeller_checkpoint_sha256=HEAD_SHA,
        dinov2_checkpoint_sha256=DINO_SHA,
        protocol=common_protocol,
        target_diagnoses=dict(small=dict(
            specification=specification, rows=attrition_rows)))
    coverage = dict(
        audit=audit.EXPECTED_COVERAGE_AUDIT, protocol_version=1,
        labeller_checkpoint_sha256=HEAD_SHA,
        dinov2_checkpoint_sha256=DINO_SHA,
        protocol=common_protocol,
        target_diagnoses=dict(small=dict(
            specification=specification, rows=coverage_rows)))
    full0 = dict(frame_count=10, top1_hits=7, top1_mcml=2)
    small0 = dict(frame_count=5, top1_hits=3, top1_mcml=2)
    full1 = dict(frame_count=10, top1_hits=8, top1_mcml=1)
    small1 = dict(frame_count=5, top1_hits=4, top1_mcml=1)
    interpolation = dict(
        selector=audit.EXPECTED_SELECTOR, protocol_version=1,
        selected_checkpoint_sha256=HEAD_SHA,
        dinov2_checkpoint_sha256=DINO_SHA,
        selected_alpha=0.5,
        protocol=dict(target_data_read=False,
                      target_used_for_selection=False),
        source=dict(
            baseline=dict(source_full_summary=full0,
                          source_small_summary=small0),
            candidates=[dict(
                alpha=0.5, source_full_summary=full1,
                source_small_summary=small1,
                retention=dict(
                    retained_correct_count=7, baseline_correct_count=7,
                    newly_correct_count=1, newly_incorrect_count=0),
                source_gate=dict(passed=True))]))
    return attrition, coverage, interpolation


def _write_inputs(tmp_path):
    paths = {}
    for name, payload in zip(
            ('attrition', 'coverage', 'interpolation'), _inputs()):
        path = tmp_path / (name + '.json')
        path.write_text(json.dumps(payload), encoding='utf-8')
        paths[name] = str(path)
    return paths


def test_build_report_uses_authoritative_mutually_exclusive_causes(tmp_path):
    attrition, coverage, interpolation = _inputs()
    paths = _write_inputs(tmp_path)
    report = audit.build_report(
        attrition, coverage, interpolation, paths)
    summary = report['target_diagnosis_summary']['small']
    assert summary['frame_count'] == 3
    assert summary['top1_hit_count'] == 1
    assert summary['attribution_counts'] == {
        'ROI_ORDERING_OR_NMS': 1,
        'ROI_REGRESSION': 1,
        'SUCCESS_TOP1': 1,
    }
    failures = [row for row in report['records']
                if row['review_required']]
    assert [row['frame'] for row in failures] == [2, 3]
    assert report['evidence_boundary'][
        'ordering_and_nms_are_not_separable_from_retained_audit'] is True


def test_validation_rejects_mixed_checkpoint_identity():
    attrition, coverage, interpolation = _inputs()
    coverage['labeller_checkpoint_sha256'] = 'c' * 64
    with pytest.raises(ValueError, match='head checkpoint identities'):
        audit.validate_inputs(attrition, coverage, interpolation)


def test_validation_rejects_target_selected_interpolation():
    attrition, coverage, interpolation = _inputs()
    interpolation['protocol']['target_data_read'] = True
    with pytest.raises(ValueError, match='source-only'):
        audit.validate_inputs(attrition, coverage, interpolation)


def test_validation_locks_current_alpha05_source_safe_baseline():
    attrition, coverage, interpolation = _inputs()
    interpolation['selected_alpha'] = 0.25
    interpolation['source']['candidates'][0]['alpha'] = 0.25
    with pytest.raises(ValueError, match='requires alpha=0.5'):
        audit.validate_inputs(attrition, coverage, interpolation)

    attrition, coverage, interpolation = _inputs()
    interpolation['source']['candidates'][0]['source_gate']['passed'] = False
    with pytest.raises(ValueError, match='source gate'):
        audit.validate_inputs(attrition, coverage, interpolation)


def test_cli_writes_reproducible_json_csv_and_markdown(tmp_path, monkeypatch):
    paths = _write_inputs(tmp_path)
    out_dir = tmp_path / 'output'
    monkeypatch.setattr(sys, 'argv', [
        'audit', '--attrition', paths['attrition'],
        '--coverage', paths['coverage'],
        '--interpolation', paths['interpolation'],
        '--out-dir', str(out_dir)])
    audit.main()
    audit.main()

    result = json.loads((
        out_dir / 'frozen_dino_native_s14_failure_attribution_v1.json'
    ).read_text(encoding='utf-8'))
    assert result['protocol'] == audit.PROTOCOL
    assert len(result['records']) == 3
    with open(out_dir / 'frozen_dino_native_s14_failure_attribution_v1.csv',
              newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert rows[0]['review_required'] == 'True'
    markdown = (
        out_dir / 'frozen_dino_native_s14_failure_attribution_v1.md'
    ).read_text(encoding='utf-8')
    assert '目标困难切片已经暴露' in markdown
    assert '排序与 NMS' in markdown


def test_write_exact_refuses_different_existing_output(tmp_path):
    path = tmp_path / 'result.json'
    audit._write_exact(path, b'one')
    with pytest.raises(RuntimeError, match='Refusing to overwrite'):
        audit._write_exact(path, b'two')
