"""Static contracts for the seq11-v2 replay stage."""

import ast
import argparse
import builtins
import hashlib
import io
import json
import math
import pickle
from pathlib import Path

import pytest

from crane_project.tools.symeood_dino_replay_schedule_audit import (
    audit as replay_audit)
from crane_project.tools.symeood_dino_seq11_formal_k1_identity import (
    audit as formal_k1_identity_audit)
from crane_project.utils.fixed_ratio_replay_schedule import (
    LEGACY_REPLAY_SCHEDULE_PROTOCOL, enumerate_replay_schedule,
    legacy_replay_schedule_contract, replay_schedule_contract)


ROOT = Path(__file__).resolve().parents[1]


def test_all_lane_collection_uses_all_251_and_ordinary_k1():
    path = ROOT / (
        'crane_project/configs/'
        'crane_symeood_dino_seq11_v2_all_lane_collect.py')
    text = path.read_text()
    assert "expected_frame_count=251" in text
    assert "baseline_config='crane_project/configs/crane_symeood_k1.py'" in text
    assert "frozen_symeood_checkpoint='work_dirs/crane_symeood_k1/epoch_24.pth'" in text
    assert "both_lanes_required_every_frame=True" in text
    assert "optimizer_steps=0" in text
    assert "fixed_test_read=False" in text
    assert 'val=seq11_dataset' in text
    assert 'test=seq11_dataset' in text
    assert "ann_file='test/" not in text


def test_v4_config_is_audited_source_only_and_fixed_budget():
    path = ROOT / (
        'crane_project/configs/'
        'crane_symeood_dino_k1_retentive_causal_phase_refiner_'
        'source_v4_seq11_v2_replay.py')
    text = path.read_text()
    tree = ast.parse(text)
    assert "'ALLOW_SEQ11_BLOCKSPLIT_SOURCE_TRAINING'" in text
    assert text.count("'ALLOW_AUXILIARY_SOURCE_TRAINING_INPUT'") >= 3
    assert 'auxiliary_source_train_frames=203' in text
    assert 'auxiliary_source_val_frames=48' in text
    assert 'auxiliary_train_val_overlap=0' in text
    assert "type='FixedRatioPairReplayDataset'" in text
    assert 'def _safe_data_root_child(value, role):' in text
    assert "child in {'train', 'train_sim', 'val', 'test'}" in text
    assert "startswith('extra_source_real_seq11_')" not in text
    assert 'original_batches_per_auxiliary_batch = 14' in text
    assert 'optimizer_steps_per_epoch = 1391' in text
    assert 'training_epochs = 10' in text
    assert 'total_optimizer_steps = optimizer_steps_per_epoch * training_epochs' in text
    assert "type='EpochBasedRunner'" in text
    assert 'base_teacher_retention_loss_weight=0.25' in text
    assert "selected_source_epoch=9" in text
    assert "ann_file='test/" not in text
    assert "expected_split='test'" not in text
    assert tree is not None


def test_replay_wrapper_keeps_each_pair_on_one_lane():
    path = ROOT / 'mmrotate/datasets/fixed_ratio_pair_replay.py'
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef)
               and node.name == 'FixedRatioPairReplayDataset')
    methods = {node.name for node in cls.body
               if isinstance(node, ast.FunctionDef)}
    assert {'_route', '__getitem__', 'set_epoch', 'replay_contract',
            'coverage_contract',
            'replay_route_for_optimizer_step'} <= methods
    text = path.read_text()
    assert 'batch = index // self.samples_per_batch' in text
    assert 'route_replay_batch(' in text
    assert "sample['source_replay_is_auxiliary']" in text
    assert 'epoch_offset = self.epoch * samples_per_epoch' in text
    assert 'replay_schedule_protocol=LEGACY_REPLAY_SCHEDULE_PROTOCOL' in text
    assert 'Unknown replay schedule protocol' in text


def test_replay_schedule_enumerates_exact_1391_route():
    routes = enumerate_replay_schedule(1391, 14)
    contract = replay_schedule_contract(1391, 14)

    assert len(routes) == 1391
    assert sum(not auxiliary for auxiliary, _ in routes) == 1299
    assert sum(auxiliary for auxiliary, _ in routes) == 92
    assert contract['protocol'] == 'fixed_ratio_pair_replay_schedule_v2'
    assert contract['scheduled_original_steps'] == 1299
    assert contract['scheduled_auxiliary_steps'] == 92
    assert contract['enumerated_total_steps'] == 1391
    legacy = legacy_replay_schedule_contract(1391, 14)
    assert legacy['protocol'] == LEGACY_REPLAY_SCHEDULE_PROTOCOL
    assert legacy['scheduled_original_steps'] == 1298
    assert legacy['scheduled_auxiliary_steps'] == 93
    assert legacy['enumerated_original_steps'] == 1299
    assert legacy['enumerated_auxiliary_steps'] == 92


def test_replay_schedule_audit_checks_offsets_coverage_and_determinism():
    args = argparse.Namespace(
        optimizer_steps_per_epoch=1391,
        original_batches_per_auxiliary_batch=14,
        samples_per_batch=2,
        training_epochs=10,
        original_sample_count=2781,
        auxiliary_sample_count=251,
        out_json='unused.json')

    report = replay_audit(args)

    assert report['passed'] is True
    assert report['checks']['exact_1391_route'] is True
    assert report['checks']['deterministic_reenumeration'] is True
    assert report['coverage']['original_unique_count'] == 2781
    assert report['coverage']['auxiliary_unique_count'] == 251
    assert report['per_epoch'][1]['original_sample_offset'] == 1299 * 2
    assert report['per_epoch'][1]['auxiliary_sample_offset'] == 92 * 2


def test_trainer_has_frozen_teacher_and_masked_retention():
    trainer = (ROOT / (
        'mmrotate/models/detectors/'
        'symeood_dino_geometry_refiner_trainer.py')).read_text()
    hook = (ROOT / (
        'mmrotate/core/hooks/'
        'geometry_refiner_contract_hook.py')).read_text()
    assert 'teacher_geometry_refiner = copy.deepcopy' in trainer
    assert '_phase_teacher_retention_loss' in trainer
    assert '_replay_auxiliary_mask' in trainer
    assert 'original = ~replay_auxiliary' in trainer
    assert 'refiner_base_v3_teacher_retention_objective' in trainer
    assert 'teacher_refiner_hash_unchanged' in trainer
    assert "'train_forward'" in trainer
    assert 'def runtime_forward_counts(self):' in trainer
    assert 'Base-V3 teacher received a gradient' in hook
    assert 'class FixedRatioReplayEpochHook' in hook
    assert 'fixed_ratio_replay_runtime_audit_v2' in hook
    assert "report['forward_counts']" in hook
    replay_class = hook.split('class FixedRatioReplayEpochHook', 1)[1]
    assert "priority = 'VERY_HIGH'" not in replay_class
    config = (ROOT / (
        'crane_project/configs/'
        'crane_symeood_dino_k1_retentive_causal_phase_refiner_'
        'source_v4_seq11_v2_replay.py')).read_text()
    assert ("dict(type='FixedRatioReplayEpochHook', priority='VERY_HIGH')"
            in config)


def test_seq11_v2_aux_evaluation_is_read_only_and_uses_48_frames():
    candidate = (ROOT / (
        'crane_project/configs/'
        'crane_symeood_dino_k1_retentive_causal_phase_refiner_'
        'source_v4_seq11_v2_aux_val.py')).read_text()
    reference = (ROOT / (
        'crane_project/configs/'
        'crane_symeood_k1_seq11_v2_aux_val_eval.py')).read_text()
    assert 'expected_frame_count=48' in candidate
    assert "paper_temporal=False" in candidate
    assert "paper_temporal=False" in reference
    assert "split_report.get('val_split', '')" in candidate
    assert "split_report.get('val_split', '')" in reference
    assert 'split_report = _read_json(split_report_path)' in candidate
    assert 'split_report = _read_json(split_report_path)' in reference
    assert 'as _handle' not in candidate + reference
    assert "ann_file='test/" not in candidate + reference
    assert 'optimizer' not in candidate
    assert 'optimizer' not in reference


def test_formal_k1_full251_config_is_source_only_and_identity_bound():
    config = (ROOT / (
        'crane_project/configs/'
        'crane_symeood_k1_seq11_v2_full251_eval.py')).read_text()
    identity = (ROOT / (
        'crane_project/tools/'
        'symeood_dino_seq11_formal_k1_identity.py')).read_text()

    assert "expected_checkpoint = 'work_dirs/crane_symeood_k1/epoch_24.pth'" in config
    assert "expected_frame_count=251" in config
    assert "prediction_coordinate_system='original_image_pixels'" in config
    assert "obb_convention='le90'" in config
    assert 'target_data_read=False' in config
    assert 'fixed_test_read=False' in config
    assert "ann_file='test/" not in config
    assert 'FRAME_ORDER_PROTOCOL' in identity
    assert 'results_sha256=_sha256_file(results_path)' in identity
    assert 'checkpoint_sha256=checkpoint_sha' in identity
    assert 'config_sha256=_sha256_file(config)' in identity
    assert 'source_manifest_sha256=_sha256_file(source_manifest)' in identity


def test_formal_k1_identity_binds_results_to_exact_frame_order(tmp_path):
    config = tmp_path / 'crane_symeood_k1_seq11_v2_full251_eval.py'
    config.write_text('\n'.join((
        "expected_checkpoint = 'work_dirs/crane_symeood_k1/epoch_24.pth'",
        'expected_frame_count=251',
        "prediction_coordinate_system='original_image_pixels'",
        "obb_convention='le90'",
        'target_data_read=False',
        'fixed_test_read=False')))
    checkpoint = tmp_path / 'epoch_24.pth'
    checkpoint.write_bytes(b'formal-k1')
    base_k1_config = tmp_path / 'crane_symeood_k1.py'
    base_k1_config.write_text('model = dict(type="EOODDetector")')
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    evaluation_summary = tmp_path / 'checkpoint_eval_summary.json'
    evaluation_summary.write_text(json.dumps([dict(
        config=str(config), checkpoint=str(checkpoint),
        checkpoint_sha256=checkpoint_sha, metric={'mAP': 1.0})]))
    source_manifest = tmp_path / 'split_manifest.json'
    source_root = tmp_path / 'data' / 'source'
    images = source_root / 'images'
    annotations = source_root / 'annfiles'
    images.mkdir(parents=True)
    annotations.mkdir()
    stems = []
    for frame in range(251):
        stem = 'real_seq11_{:06d}'.format(frame)
        stems.append(stem)
        (images / (stem + '.jpg')).write_bytes(b'image')
        (annotations / (stem + '.txt')).write_text('annotation')
    results = tmp_path / 'results.pkl'
    results.write_bytes(pickle.dumps([[[]] for _ in range(251)]))
    source_manifest.write_text(json.dumps(dict(
        protocol='real_seq11_source_k1p9_block_manifest_v2',
        all_frame_count=251, train_stems=stems, aux_val_stems=[])))
    inference_receipt = tmp_path / 'inference_receipt.json'
    inference_receipt.write_text(json.dumps(dict(
        protocol='mmdet_runtime_result_order_identity_v1',
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        checkpoint_sha256=checkpoint_sha,
        results_sha256=hashlib.sha256(results.read_bytes()).hexdigest(),
        result_count=251,
        runtime_dataset_order=[dict(
            result_index=index, frame_key=stem,
            dataset_filename=stem + '.jpg')
            for index, stem in enumerate(stems)],
        target_data_read=False, fixed_test_read=False)))
    args = argparse.Namespace(
        results=str(results), checkpoint=str(checkpoint), config=str(config),
        base_k1_config=str(base_k1_config),
        evaluation_summary=str(evaluation_summary),
        inference_receipt=str(inference_receipt),
        source_manifest=str(source_manifest), data_root=str(tmp_path / 'data'),
        source_split='source',
        frame_order_json=str(tmp_path / 'frame_order.json'),
        out_json=str(tmp_path / 'identity.json'))

    report = formal_k1_identity_audit(args)

    assert report['passed'] is True
    assert report['frame_count'] == 251
    assert report['missing_prediction_count'] == 251
    assert report['inputs']['frame_order_manifest_sha256']
    assert report['checks']['runtime_dataset_order_bound'] is True


def test_v2_cv_protocol_preregisters_isolation_metrics_and_stop_gates():
    path = ROOT / (
        'crane_project/data_contracts/'
        'real_seq11_pilot_k1p9_block_cv_v2_protocol.json')
    payload = json.loads(path.read_text())
    assert payload['fold_training_authorized'] is False
    assert payload['history_protocol']['history_horizon'] == 4
    assert payload['metric_pair_protocol']['cross_fold_pairs'] is False
    assert payload['metric_pair_protocol'][
        'pooled_dfr_aci_aggregation'].startswith('sum block numerators')
    assert payload['mcml_protocol'][
        'legacy_mcml_mean_exact_name'] == 'MCML_segment_max_mean'
    assert payload['support_protocol'][
        'both_bad_excluded_from_dino_rescue_denominator'] is True
    assert payload['support_protocol'][
        'per_fold_k1_present_wrong_dino_hit_min'] == 1
    assert payload['training_protocol'][
        'replay_schedule_protocol'] == 'fixed_ratio_pair_replay_schedule_v2'
    assert payload['target_data_read'] is False
    assert payload['fixed_test_read'] is False


def test_v2_identity_and_dependency_tools_fail_closed_before_training():
    dino = (ROOT / (
        'crane_project/tools/'
        'symeood_dino_seq11_dino_cache_identity.py')).read_text()
    history = (ROOT / (
        'crane_project/tools/'
        'symeood_dino_seq11_history_dependency_audit.py')).read_text()
    support = (ROOT / (
        'crane_project/tools/'
        'symeood_dino_seq11_block_cv_support_audit.py')).read_text()
    assert 'dino_forward_count_251' in dino
    assert 'source_manifest_exact_set' in dino
    assert 'history_dependency_edges' in history
    assert "frame - age" in history
    assert 'metric_pair_candidates' in history
    assert 'STOP_SEQ11_V2_MECHANISM_SUPPORT_INSUFFICIENT' in support
    assert "eligible_for_three_fold_training=False" in support


def test_server_inventory_is_fail_closed_and_never_reads_fixed_test():
    inventory = (ROOT / (
        'crane_project/tools/'
        'symeood_dino_seq11_v2_server_input_inventory.py')).read_text()

    assert 'visible_image_count_251' in inventory
    assert 'original_source_count_2781' in inventory
    assert 'official_source_val_count_738' in inventory
    assert 'base_v3_checkpoint_hash_matches_promotion' in inventory
    assert "payload.get('fixed_test_read') is True" in inventory
    assert "promotion.get('target_data_read') is False" in inventory
    assert "promotion.get('fixed_test_read') is False" in inventory
    assert 'STOP_SEQ11_V2_SERVER_INPUT_INVENTORY_FAILED' in inventory


def test_seq11_v2_three_way_baseline_is_source_only_and_preregistered():
    config = (ROOT / (
        'crane_project/configs/'
        'crane_symeood_dino_k1_retentive_causal_phase_refiner_'
        'base_v3_seq11_v2_full251_eval.py')).read_text()
    tool = (ROOT / (
        'crane_project/tools/'
        'symeood_dino_seq11_v2_baseline_compare.py')).read_text()
    contract_path = ROOT / (
        'crane_project/data_contracts/'
        'real_seq11_k1_dino_base_v3_three_way_baseline_v1.json')
    contract = json.loads(contract_path.read_text())

    assert 'expected_frame_count=251' in config
    assert "expected_checkpoint_source_train_frames = 2781" in config
    assert "expected_checkpoint_target_data_read = False" in config
    assert "expected_checkpoint_fixed_test_read = False" in config
    assert "evaluation_only=True" in config
    assert "optimizer_steps=0" in config
    assert "ann_file='test/" not in config
    assert "expected_split='test'" not in config

    assert 'deterministic_k1_else_dino' in tool
    assert 'base_v3_epoch9_refiner' in tool
    assert 'contiguous_gt_pair_count' in tool
    assert 'valid_pair_fraction' in tool
    assert 'MCML_segment_max_mean' in tool
    assert 'unrecovered_terminal_run_length' in tool
    assert 'both_bad_excluded_from_rescue_denominator=True' in tool
    assert 'no_automatic_cv_training_authorization=True' in tool
    assert "eligible_for_three_fold_training=False" in tool

    assert contract['three_fold_training_authorized'] is False
    assert contract['execution']['training'] is False
    assert contract['execution']['target_data_read'] is False
    assert contract['execution']['fixed_test_read'] is False
    assert contract['metric_protocol'][
        'both_bad_excluded_from_rescue_denominator'] is True
    assert contract['interpretation'][
        'automatic_cv_training_authorization'] is False


def test_result_identity_receipt_supports_generic_source_only_contract():
    source = (ROOT / 'tools/test.py').read_text()

    assert "cfg.get('formal_k1_full251_contract', None)" in source
    assert "cfg.get('source_only_result_contract', {})" in source
    assert 'evidence_contract=dict(evidence_contract)' in source
    assert "evidence_contract.get('target_data_read', None)" in source
    assert "evidence_contract.get('fixed_test_read', None)" in source
    assert "--runtime-audit-out" in source
    assert "mmdet_runtime_inference_resource_audit_v1" in source
    assert "runtime_audit_protocol" in source
    assert "torch.cuda.max_memory_allocated" in source
    assert "torch.cuda.max_memory_reserved" in source
    assert "runtime_forward_counts" in source
    assert "runtime_input_files" in source
    assert "runtime_model_contract" in source
    assert "effective_config_sha256" in source
    assert "effective_config=effective_config" in source


def test_runtime_audit_records_effective_model_and_merged_config(tmp_path):
    from argparse import Namespace

    from mmcv import Config
    from tools.test import _runtime_audit_record

    config_path = tmp_path / 'eval.py'
    checkpoint_path = tmp_path / 'epoch.pth'
    config_path.write_text('model = dict(type="Dummy")\n')
    checkpoint_path.write_bytes(b'checkpoint')
    cfg = Config(dict(
        model=dict(
            type='Dummy', geometry_refiner=dict(
                inference_component_mode='current_only')),
        source_only_result_contract=dict(
            target_data_read=False, fixed_test_read=False),
        runtime_input_files={}))

    class DummyModel:
        def runtime_forward_counts(self):
            return dict(inference_forward=1, dino_detector_forward=0)

        def runtime_inference_contract(self):
            return dict(
                evaluation_only=True,
                inference_component_mode='current_only',
                causal_history_features_computed=True,
                history_output_contribution=False,
                component_contract=dict(
                    inference_component_mode='current_only'))

    args = Namespace(
        config=str(config_path), checkpoint=str(checkpoint_path),
        cfg_options=None)
    record = _runtime_audit_record(cfg, args, DummyModel(), [object()])
    assert record['runtime_model_contract'][
        'inference_component_mode'] == 'current_only'
    assert record['effective_config']['model']['geometry_refiner'][
        'inference_component_mode'] == 'current_only'
    assert len(record['effective_config_sha256']) == 64
    assert record['cli_cfg_options'] is None


def test_base_v3_history_contribution_ablation_is_uniform_and_read_only():
    from mmcv import Config

    config_path = ROOT / (
        'crane_project/configs/'
        'crane_symeood_dino_k1_retentive_causal_phase_refiner_'
        'base_v3_seq11_v2_full251_current_only_eval_v2.py')
    config = (ROOT / (
        'crane_project/configs/'
        'crane_symeood_dino_k1_retentive_causal_phase_refiner_'
        'base_v3_seq11_v2_full251_current_only_eval.py')).read_text()
    config += config_path.read_text()
    tool = (ROOT / (
        'crane_project/tools/'
        'symeood_dino_seq11_v2_history_contribution_audit.py')).read_text()
    legacy_contract = json.loads((ROOT / (
        'crane_project/data_contracts/'
        'real_seq11_base_v3_history_contribution_ablation_v1.json')).read_text())
    contract = json.loads((ROOT / (
        'crane_project/data_contracts/'
        'real_seq11_base_v3_history_contribution_ablation_v2.json')).read_text())
    refiner = (ROOT / (
        'mmrotate/models/roi_heads/'
        'dino_conditioned_geometry_refiner.py')).read_text()

    assert "geometry_refiner=dict(inference_component_mode='current_only')" in config
    assert 'history_output_contribution=False' in config
    assert 'same_setting_all_frames=True' in config
    assert 'domain_routing=False' in config
    assert 'sequence_frame_routing=False' in config
    assert 'optimizer_steps=0' in config
    assert 'fixed_test_read=False' in config
    assert "ann_file='test/" not in config
    assert "expected_split='test'" not in config
    merged = Config.fromfile(str(config_path))
    assert merged.model.geometry_refiner.inference_component_mode == 'current_only'
    assert merged.model.evaluation_only is True
    assert merged.source_only_result_contract.protocol.endswith('_v2')
    assert merged.source_only_result_contract.runtime_audit_protocol.endswith(
        '_v2')
    assert merged.source_only_result_contract.optimizer_steps == 0
    assert merged.source_only_result_contract.fixed_test_read is False
    assert "if self.inference_component_mode == 'current_only':" in refiner
    assert 'combined = current_five' in refiner

    assert 'no_valid_history_invariance' in tool
    assert 'valid_history_count' in tool
    assert 'full_minus_current_only_riou' in tool
    assert 'geometry_overlay.png' in tool
    assert 'eligible_for_three_fold_training=False' in tool
    assert legacy_contract['protocol'].endswith('_v1')
    assert contract['paired_design']['same_checkpoint'] is True
    assert contract['paired_design'][
        'history_output_contribution_in_current_only_arm'] is False
    assert contract['interpretation'][
        'automatic_cv_training_authorization'] is False
    assert contract['execution']['training'] is False
    assert contract['execution']['fixed_test_read'] is False
    assert contract['three_fold_training_authorized'] is False
    assert contract['hard_fail_checks'][
        'no_valid_history_violation_max'] == 0
    assert contract['metric_protocol']['reject_cli_parameter_drift'] is True
    assert 'STOP_HISTORY_CONTRIBUTION_AUDIT_CONTRACT_FAILED' in tool
    assert "if not report['passed']" in tool


def test_history_ablation_rejects_metric_parameter_drift():
    from argparse import Namespace

    from crane_project.tools import (
        symeood_dino_seq11_v2_history_contribution_audit as audit)

    report = dict(metric_contract=dict(
        iou_threshold=0.5, center_threshold_px=25.0,
        angle_limit_deg=35.0, high_confidence_threshold=0.5))
    args = Namespace(
        iou_threshold=0.5, center_threshold_px=25.0,
        angle_limit_deg=35.0, high_confidence_threshold=0.5)
    locked = audit._locked_metric_parameters(report, args)
    assert locked == dict(
        iou_threshold=0.5, center_threshold_px=25.0,
        angle_limit_deg=35.0, high_confidence_threshold=0.5)
    args.iou_threshold = 0.49
    with pytest.raises(RuntimeError, match='differ from frozen'):
        audit._locked_metric_parameters(report, args)


def test_current_only_source_val_gate_is_preregistered_and_fail_closed():
    from crane_project.tools.symeood_dino_current_only_source_val_gate import (
        _relative_gate)

    config = (ROOT / (
        'crane_project/configs/'
        'crane_symeood_dino_k1_retentive_component_'
        'current_only_source_val.py')).read_text()
    contract = json.loads((ROOT / (
        'crane_project/data_contracts/'
        'base_v3_epoch9_current_only_source_val_gate_v1.json')).read_text())
    gate_source = (ROOT / (
        'crane_project/tools/'
        'symeood_dino_current_only_source_val_gate.py')).read_text()

    assert "inference_component_mode='current_only'" in config
    assert 'expected_checkpoint = promoted_checkpoint' not in config
    assert 'k1_retentive_v3_epoch9_promoted.pth' in config
    assert 'same_setting_real_sim=True' in config
    assert 'optimizer_steps=0' in config
    assert 'fixed_test_read=False' in config
    assert "ann_file='test/" not in config
    assert contract['status'].startswith('preregistered_before_')
    assert contract['relative_non_regression'][
        'real_center_drop_pp_max'] == 0.0
    assert contract['decision_policy'][
        'failure_allows_alternate_epoch_selection'] is False
    assert contract['decision_policy'][
        'failure_allows_threshold_change'] is False
    assert 'STOP_CURRENT_ONLY_SOURCE_NON_REGRESSION_FAILED' in gate_source
    assert "if not report['passed']" in gate_source

    reference = {
        'real/R_center(%)': 100.0, 'sim/R_center(%)': 100.0,
        'real/mean_RIoU': 0.80, 'sim/mean_RIoU': 0.90,
        'real/DFR(%/frame)': 2.0, 'sim/DFR(%/frame)': 2.0,
        'real/ACI': 0.95, 'sim/ACI': 0.96,
        'sim/A-RMSE(deg)': 5.0,
        'real/TDR_w10(%)': 100.0, 'sim/TDR_w10(%)': 100.0,
        'real/MCML_max(frames)': 0, 'sim/MCML_max(frames)': 0}
    passing = dict(reference)
    assert _relative_gate(
        passing, reference, contract['relative_non_regression'])['passed']
    failing = dict(reference)
    failing['real/R_center(%)'] = 99.99
    result = _relative_gate(
        failing, reference, contract['relative_non_regression'])
    assert result['passed'] is False
    assert result['checks']['real_center_no_drop'] is False


def test_current_only_source_val_config_resolves_without_parent_locals(
        monkeypatch):
    from mmcv import Config

    config_path = ROOT / (
        'crane_project/configs/'
        'crane_symeood_dino_k1_retentive_component_'
        'current_only_source_val.py')
    promotion = dict(
        decision='ALLOW_K1_RETENTIVE_CAUSAL_PHASE_FIXED_BENCHMARK_TEST',
        eligible_for_fixed_benchmark_test=True,
        target_data_read=False,
        fixed_test_read=False,
        selection=dict(selected=dict(epoch=9)),
        output=dict(
            checkpoint=(
                'work_dirs/crane_symeood_dino_k1_retentive_causal_phase_'
                'refiner_source_v3_seed3407/'
                'k1_retentive_v3_epoch9_promoted.pth'),
            checkpoint_sha256='a' * 64))
    original_open = builtins.open

    def patched_open(path, *args, **kwargs):
        if str(path).endswith('epoch9_source_promotion.json'):
            return io.StringIO(json.dumps(promotion))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, 'open', patched_open)
    merged = Config.fromfile(str(config_path))
    assert merged.expected_checkpoint.endswith(
        'k1_retentive_v3_epoch9_promoted.pth')
    assert merged.model.geometry_refiner.inference_component_mode == (
        'current_only')
    assert merged.data.test.ann_file == 'val/annfiles/'
    assert merged.source_only_result_contract.fixed_test_read is False


def test_current_only_failure_attribution_metric_decomposition():
    import numpy as np

    from crane_project.tools import (
        symeood_dino_current_only_source_val_attribution as attribution)

    gt = np.asarray([100.0, 100.0, 40.0, 20.0, 0.0])
    direct = attribution._angle_geometry(
        np.asarray([102.0, 100.0, 40.0, 20.0, math.radians(5.0)]), gt)
    penalized = attribution._angle_geometry(
        np.asarray([111.0, 100.0, 40.0, 20.0, 0.0]), gt)
    missing = attribution._angle_geometry(None, gt)
    assert direct['angle_metric_state'] == 'direct_periodic_angle_error'
    assert direct['angle_metric_error_deg'] == pytest.approx(5.0)
    assert penalized['angle_metric_state'] == 'center_error_penalty'
    assert penalized['angle_metric_error_deg'] == 90.0
    assert missing['angle_metric_state'] == 'missing_prediction_penalty'
    assert missing['angle_metric_squared_error_deg2'] == 8100.0
    summary = attribution._angle_method_summary([
        dict(candidate=direct), dict(candidate=penalized),
        dict(candidate=missing)], 'candidate')
    assert summary['direct_angle_frame_count'] == 1
    assert summary['center_penalty_frame_count'] == 1
    assert summary['missing_penalty_frame_count'] == 1
    assert summary['direct_angle_rmse_deg'] == pytest.approx(5.0)
    assert summary['penalty_squared_error_deg2'] == 16200.0


def test_current_only_failure_attribution_preserves_boundaries_and_role():
    from crane_project.tools import (
        symeood_dino_current_only_source_val_attribution as attribution)

    def row(frame, hit, sequence='seq01'):
        return dict(
            frame_key='sim_{}_{}'.format(sequence, frame),
            domain='sim', sequence=sequence, frame=frame,
            current_only=dict(riou_hit=hit))

    rows = [row(1, False), row(2, False), row(4, False),
            row(1, False, sequence='seq02')]
    runs = attribution._failure_runs(rows, 'current_only')
    assert [item['length'] for item in runs] == [2, 1, 1]

    contract = json.loads((ROOT / (
        'crane_project/data_contracts/'
        'base_v3_current_only_source_val_failure_attribution_v1.json'
    )).read_text())
    source = (ROOT / (
        'crane_project/tools/'
        'symeood_dino_current_only_source_val_attribution.py')).read_text()
    assert contract['prohibited_uses']['training_authorization'] is True
    assert contract['prohibited_uses']['fixed_test_authorization'] is True
    assert contract['metric_decomposition'][
        'sim_angle_center_threshold_px'] == 10.0
    assert 'seq11_dataset_has_diagnostic_value=True' in source
    assert 'CURRENT_ONLY_FAILURE_ATTRIBUTION_READY_' in source
