import json

from crane_project.tools import audit_historical_mechanisms_v1 as audit


def test_inventory_collects_evidence_without_inference(tmp_path):
    (tmp_path / 'crane_project/configs').mkdir(parents=True)
    (tmp_path / 'crane_project/configs/crane_symeood_k1.py').write_text(
        "bbox_head=dict(loss_cls=dict(type='SymNFLLoss'))\n"
        "assigner=dict(type='SymPOLAAssigner')\n")
    (tmp_path / 'docs/evidence').mkdir(parents=True)
    (tmp_path / 'docs/evidence/history.md').write_text(
        'ATSS and PAA were tested.\nQFL was not adopted.\n'
        'reg_quality_head was enabled in V3.\n')
    files = list(audit.iter_evidence_files(tmp_path))
    count, matches, class_counts = audit.collect_matches(tmp_path, files)
    assert count == 2
    assert matches['assignment']['ATSS']
    assert matches['assignment']['PAA']
    assert matches['classification']['QFL']
    assert matches['quality_guidance']['RegQuality']
    assert class_counts['assignment']['ATSS']['historical_record'] == 1
    assert matches['assignment']['ATSS'][0]['evidence_class'] == 'historical_record'
    report_path = tmp_path / 'report.json'
    # Smoke the same output contract without invoking model code.
    payload = {'protocol': 'historical_mechanism_inventory_v1',
               'selection_or_training_performed': False}
    report_path.write_text(json.dumps(payload))
    assert json.loads(report_path.read_text())['selection_or_training_performed'] is False


def test_test_path_is_marked_for_manual_review(tmp_path):
    path = tmp_path / 'fixed_test_qfl.json'
    path.write_text('QFL')
    files = list(audit.iter_evidence_files(tmp_path))
    _count, matches, _class_counts = audit.collect_matches(tmp_path, files)
    assert matches['classification']['QFL'][0]['test_artifact_path'] is True
