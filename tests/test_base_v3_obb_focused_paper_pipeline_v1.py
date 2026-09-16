import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from crane_project.tools import base_v3_obb_focused_paper_pipeline_v1 as pipeline
from crane_project.tools import base_v3_obb_true_online_finalization_v2 as finalization


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
        'reliability_json', 'reliability_markdown',
        'reliability_detail_markdown'}
    model = json.loads((tmp_path / config['outputs'][
        'model_report_json']).read_text(encoding='utf-8'))
    reliability = json.loads((tmp_path / config['outputs'][
        'reliability_report_json']).read_text(encoding='utf-8'))
    assert model['protocol'] == pipeline.MODEL_REPORT_PROTOCOL
    assert reliability['protocol'] == pipeline.RELIABILITY_REPORT_PROTOCOL
    assert len(model['metrics']['complete_obb']) == 9
    assert len(model['metrics']['components']) == 27
    assert len(model['metrics']['anchor_sources']) == 6
    assert model['metrics']['custom_temporal']['protocol'] == (
        'base_v3_obb_custom_temporal_metrics_v1')
    assert model['metrics']['custom_temporal']['values']['raw'][
        'real/R_center(%)'] == 97.38
    assert reliability['joint_component_availability'][
        'v51_hybrid']['all']['partial_valid_count'] == 202
    assert reliability['matched_coverage_scale']['all'][
        'matched_coverage_tie_count'] == 1
    assert '不是基于 GT 的正确性标签' in reliability[
        'metric_definitions']['valid']
    assert '## 自定义检测与时序指标' in (tmp_path / config['outputs'][
        'model_report_markdown']).read_text(encoding='utf-8')
    concise = (tmp_path / config['outputs'][
        'reliability_report_markdown']).read_text(encoding='utf-8')
    detail = (tmp_path / config['outputs'][
        'reliability_detail_markdown']).read_text(encoding='utf-8')
    assert '表格单元均为“输出率 / 输出准确率”' in concise
    assert '## 尺度同覆盖率比较' not in concise
    assert '## 尺度同覆盖率比较' in detail


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
    assert manifest['rendered_image_count'] == len(result['images'])
    assert manifest['selected_frame_count'] >= 3
    tuples = [tuple(row['state_tuple']) for row in manifest['records']]
    assert len(tuples) == len(set(tuples))
    assert all(Path(row['output']['path']).is_file()
               for row in manifest['records'])
    assert all(row['context_frames'] for row in manifest['records'])
    assert all(Path(context['output']['path']).is_file()
               for row in manifest['records']
               for context in row['context_frames'])


def test_gt_consistency_audit_is_read_only_and_records_actual_differences(
        tmp_path):
    config = _config()
    bundle = pipeline.load_report_sources(config, ROOT)
    result = pipeline.write_gt_consistency_audit(
        bundle, config, ROOT, tmp_path)
    audit = json.loads(Path(result['path']).read_text(encoding='utf-8'))
    assert audit['protocol'] == 'base_v3_obb_gt_consistency_audit_v1'
    assert audit['frame_count'] == 992
    assert audit['changes_bound_results'] is False
    assert audit['status'] == 'LOCAL_RECOMPUTATION_DIFFERENCES_PRESENT'
    assert audit['comparisons']['center']['max_abs_difference'] > 0
    assert audit['comparisons']['riou']['max_abs_difference'] > 0


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
        'reliability_json', 'reliability_markdown',
        'reliability_detail_markdown'}
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


def test_generated_run_can_be_loaded_and_all_embedded_hashes_validate():
    config = _config()
    directory = ROOT / (
        'work_dirs/base_v3_obb_reliability_baseline_v1/unified_full_run_v1')
    bundle = pipeline.load_run_sources(config, ROOT, directory)
    pipeline.validate_report_sources(bundle, config)
    assert bundle['online']['leakage_controls']['annotations_or_gt_read'] is False
    reliability = pipeline.build_reliability_report(bundle, config)
    assert '同一逐帧运行记录' in reliability['limits'][2]


def test_runtime_contract_updates_hashes_without_mutating_template():
    template = json.loads((ROOT / (
        'crane_project/data_contracts/'
        'base_v3_obb_true_online_paper_metrics_v3.json')).read_text(
            encoding='utf-8'))
    original = copy.deepcopy(template)
    runtime = pipeline._runtime_contract(
        template, dict(
            true_online_finalization_sha256='1' * 64,
            true_online_pipeline_sha256='2' * 64,
            online_observation_csv_sha256='3' * 64),
        dict(path='template.json', sha256='4' * 64, size_bytes=1),
        'evaluate')
    assert template == original
    assert runtime['expected_inputs']['true_online_finalization_sha256'] == (
        '1' * 64)
    assert runtime['runtime_binding']['stage'] == 'evaluate'
    assert runtime['runtime_binding']['template_contract']['sha256'] == '4' * 64


def test_infer_stage_exports_no_gt_observations_and_receipt(
        tmp_path, monkeypatch):
    config = _config()
    online = pipeline.load_report_sources(config, ROOT)['online']

    def fake_run(command, cwd, check):
        output = Path(command[command.index('--out-json') + 1])
        output.write_text(json.dumps(online), encoding='utf-8')
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(pipeline, 'validate_inference_inputs',
                        lambda config, root: None)
    monkeypatch.setattr(pipeline.subprocess, 'run', fake_run)
    result = pipeline.run_inference(config, ROOT, tmp_path, 'cpu')
    receipt = result['inference_receipt']
    assert receipt['gt_or_annotations_read'] is False
    assert receipt['online_pipeline']['sha256'] == (
        result['online_identity']['sha256'])
    header = Path(result['observation_csv_identity']['path']).read_text(
        encoding='utf-8').splitlines()[0]
    assert 'gt' not in header.lower()
    assert len(header.split(',')) >= 10


def test_observation_csv_refuses_overwriting_different_content(tmp_path):
    config = _config()
    online = pipeline.load_report_sources(config, ROOT)['online']
    path = tmp_path / 'observations.csv'
    first = finalization.write_observation_csv(path, online['records'])
    second = finalization.write_observation_csv(path, online['records'])
    assert first['sha256'] == second['sha256']
    path.write_text('different\n', encoding='utf-8')
    with pytest.raises(RuntimeError, match='Refusing to overwrite different'):
        finalization.write_observation_csv(path, online['records'])


def test_cli_infer_and_evaluate_require_stage_directories():
    environment = dict(os.environ)
    environment['PYTHONPATH'] = os.fspath(ROOT)
    infer = subprocess.run([
        sys.executable, '-m',
        'crane_project.tools.base_v3_obb_focused_paper_pipeline_v1',
        '--config', os.fspath(CONFIG), '--mode', 'infer'],
        cwd=os.fspath(ROOT), env=environment, capture_output=True, text=True)
    evaluate = subprocess.run([
        sys.executable, '-m',
        'crane_project.tools.base_v3_obb_focused_paper_pipeline_v1',
        '--config', os.fspath(CONFIG), '--mode', 'evaluate'],
        cwd=os.fspath(ROOT), env=environment, capture_output=True, text=True)
    assert infer.returncode != 0
    assert '--out-dir is required for infer mode' in infer.stderr
    assert evaluate.returncode != 0
    assert '--input-dir is required for evaluate mode' in evaluate.stderr
