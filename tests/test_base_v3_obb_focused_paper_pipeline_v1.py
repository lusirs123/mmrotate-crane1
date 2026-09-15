import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from crane_project.tools import base_v3_obb_focused_paper_pipeline_v1 as pipeline


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / (
    'crane_project/configs/base_v3_obb_focused_paper_pipeline_v1.json')


def _config():
    return json.loads(CONFIG.read_text(encoding='utf-8'))


def test_real_config_and_all_bound_report_sources_validate():
    config = _config()
    pipeline.validate_config(config)
    bundle = pipeline.load_report_sources(config, ROOT)
    pipeline.validate_report_sources(bundle, config)
    assert bundle['metrics']['final_detection_metrics_complete'] is True
    assert bundle['diagnostic']['comparison_tolerance'] == 1e-12


def test_config_rejects_evidence_boundary_or_metric_inventory_drift():
    config = _config()
    changed = copy.deepcopy(config)
    changed['evidence_boundary']['parameter_tuning_authorized'] = True
    with pytest.raises(ValueError, match='parameter_tuning_authorized'):
        pipeline.validate_config(changed)
    changed = copy.deepcopy(config)
    changed['metric_inventory']['components'] = ['center', 'scale']
    with pytest.raises(ValueError, match='Component inventory'):
        pipeline.validate_config(changed)


def test_report_mode_writes_complete_and_separate_reliability_outputs(
        tmp_path):
    config = _config()
    bundle = pipeline.load_report_sources(config, ROOT)
    outputs = pipeline.write_reports(bundle, config, ROOT, tmp_path)
    assert set(outputs) == {
        'model_json', 'model_markdown',
        'reliability_json', 'reliability_markdown'}
    model = json.loads((tmp_path / config['outputs'][
        'model_report_json']).read_text(encoding='utf-8'))
    reliability = json.loads((tmp_path / config['outputs'][
        'reliability_report_json']).read_text(encoding='utf-8'))
    assert model['protocol'] == pipeline.MODEL_REPORT_PROTOCOL
    assert reliability['protocol'] == pipeline.RELIABILITY_REPORT_PROTOCOL
    assert len(model['metrics']['complete_obb']) == 9
    assert len(model['metrics']['components']) == 27
    assert len(model['metrics']['anchor_sources']) == 6
    assert reliability['joint_component_availability'][
        'v51_hybrid']['all']['partial_valid_count'] == 202
    assert reliability['matched_coverage_scale']['all'][
        'matched_coverage_tie_count'] == 1
    assert '不是基于 GT 的正确性标签' in reliability[
        'metric_definitions']['valid']


def test_bound_source_hash_mismatch_is_rejected():
    config = _config()
    changed = copy.deepcopy(config)
    changed['report_sources']['paper_metrics']['sha256'] = '0' * 64
    with pytest.raises(RuntimeError, match='paper_metrics SHA256 mismatch'):
        pipeline.load_report_sources(changed, ROOT)


def test_visualization_selects_states_without_gt_or_error_ranking(tmp_path):
    config = _config()
    bundle = pipeline.load_report_sources(config, ROOT)
    result = pipeline.write_visualizations(bundle, config, ROOT, tmp_path)
    manifest = json.loads(Path(result['manifest']['path']).read_text(
        encoding='utf-8'))
    assert manifest['gt_or_error_used_for_selection'] is False
    assert manifest['selected_frame_count'] == len(result['images'])
    assert manifest['selected_frame_count'] >= 3
    tuples = [tuple(row['state_tuple']) for row in manifest['records']]
    assert len(tuples) == len(set(tuples))
    assert all(Path(row['output']['path']).is_file()
               for row in manifest['records'])


def test_full_run_diagnostic_is_rebuilt_from_current_online_records():
    config = _config()
    bundle = pipeline.load_report_sources(config, ROOT)
    metric_contract = json.loads((ROOT / (
        'crane_project/data_contracts/'
        'base_v3_obb_true_online_paper_metrics_v3.json')).read_text(
            encoding='utf-8'))
    current = pipeline.build_current_diagnostic(
        bundle['online'], bundle['finalization'], metric_contract,
        bundle['identities']['paper_metrics'])
    assert current['generation_mode'] == (
        'from_current_full_run_online_records')
    assert current['comparison_tolerance'] == 1e-12
    assert current['joint_component_availability'][
        'v51_hybrid']['all']['partial_valid_count'] == 202
    assert current['scale_matched_coverage']['all'][
        'matched_coverage_tie_count'] == 1


def test_cli_report_is_a_minimal_end_to_end_entrypoint(tmp_path):
    environment = dict(os.environ)
    environment['PYTHONPATH'] = os.fspath(ROOT)
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    completed = subprocess.run([
        sys.executable, '-m',
        'crane_project.tools.base_v3_obb_focused_paper_pipeline_v1',
        '--config', os.fspath(CONFIG), '--mode', 'report',
        '--out-dir', os.fspath(tmp_path)], cwd=os.fspath(ROOT),
        env=environment, check=True, capture_output=True, text=True)
    receipt = json.loads(completed.stdout)
    assert set(receipt) == {
        'model_json', 'model_markdown',
        'reliability_json', 'reliability_markdown'}
    for item in receipt.values():
        assert Path(item['path']).is_file()
        assert len(item['sha256']) == 64


def test_cli_full_mode_requires_a_fresh_output_directory():
    environment = dict(os.environ)
    environment['PYTHONPATH'] = os.fspath(ROOT)
    completed = subprocess.run([
        sys.executable, '-m',
        'crane_project.tools.base_v3_obb_focused_paper_pipeline_v1',
        '--config', os.fspath(CONFIG), '--mode', 'full'],
        cwd=os.fspath(ROOT), env=environment, capture_output=True, text=True)
    assert completed.returncode != 0
    assert '--out-dir is required for full mode' in completed.stderr
