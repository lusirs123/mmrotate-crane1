"""Static integration contracts that do not require MMRotate locally."""

import ast
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _class_methods(path, class_name):
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == class_name)
    return {node.name: node for node in cls.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _called_attributes(node):
    return [child.attr for child in ast.walk(node)
            if isinstance(child, ast.Attribute)]


def test_simple_test_delegates_to_feature_path_without_second_extract():
    path = ROOT / 'mmrotate/models/detectors/sym_eood_detector.py'
    methods = _class_methods(path, 'SymEOOD')
    simple = _called_attributes(methods['simple_test'])
    reused = _called_attributes(methods['simple_test_from_features'])
    assert simple.count('extract_feat') == 1
    assert simple.count('simple_test_from_features') == 1
    assert 'extract_feat' not in reused


def test_trainer_and_runtime_reference_the_same_refiner_decode_contract():
    trainer = (ROOT / ('mmrotate/models/detectors/'
                       'symeood_dino_geometry_refiner_trainer.py')).read_text()
    runtime = (ROOT / ('mmrotate/models/detectors/'
                       'scoped_dino_lowlight_detector.py')).read_text()
    shared = (ROOT / ('mmrotate/models/roi_heads/'
                      'dino_conditioned_geometry_refiner.py')).read_text()
    assert 'decode_and_normalize' in trainer
    assert 'decode_and_normalize' in runtime
    assert 'class DinoConditionedGeometryRefiner' in shared
    assert "type='DeltaXYWHAOBBoxCoder'" in shared
    assert 'active_component_mask' in shared


def test_locked_configs_do_not_contain_forbidden_routing_or_test_input():
    full = (ROOT / ('crane_project/configs/'
                    'crane_symeood_dino_geometry_refiner_full_source_v1.py'))
    text = full.read_text()
    assert "target_data_read=False" in text
    assert "domain_routing=False" in text
    assert "sequence_frame_routing=False" in text
    assert "temporal_state=False" in text
    assert "expected_frame_count=2781" in text
    assert "expected_frame_count=738" in text
    assert "refine_center=True" in text
    assert "refine_size=True" in text
    assert "refine_angle=True" in text
    assert text.count("type='FormatDinoProposal'") == 2
    assert "ann_file='test/" not in text
    assert "expected_split='test'" not in text

    tree = ast.parse(text)
    data_assignment = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'data'
                for target in node.targets))
    data_keys = {
        keyword.arg for keyword in data_assignment.value.keywords}
    delete_keyword = next(
        keyword for keyword in data_assignment.value.keywords
        if keyword.arg == '_delete_')
    assert isinstance(delete_keyword.value, ast.Constant)
    assert delete_keyword.value.value is True
    assert 'samples_per_gpu' not in data_keys
    assert 'workers_per_gpu' not in data_keys
    assert {'train_dataloader', 'val_dataloader',
            'test_dataloader'}.issubset(data_keys)


def test_refiner_optimizer_is_exclusive_and_declared_by_config():
    hook = (ROOT / ('mmrotate/core/hooks/'
                    'geometry_refiner_contract_hook.py')).read_text()
    config = (ROOT / ('crane_project/configs/'
                      'crane_symeood_dino_geometry_refiner_full_source_v1.py')
              ).read_text()
    assert 'class GeometryRefinerOptimizerConstructor' in hook
    assert 'set(optimizer_ids) != set(refiner_ids)' in hook
    assert '_delete_=True' in config
    assert "constructor='GeometryRefinerOptimizerConstructor'" in config


def test_refiner_evaluation_requires_manual_source_gate_selection():
    path = ROOT / ('crane_project/configs/'
                   'crane_symeood_dino_geometry_refiner_full_source_v1.py')
    tree = ast.parse(path.read_text())
    assignment = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'evaluation'
                for target in node.targets))
    values = {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in assignment.value.keywords}
    assert values['_delete_'] is True
    assert 'save_best' not in values
    assert 'rule' not in values
    assert values['thresh_sim'] == 10.0
    assert values['thresh_real'] == 25.0
    assert values['weight_sim'] == 0.7
    assert values['weight_real'] == 0.3


def test_dual_tower_v2_is_source_val_only_and_evaluation_only():
    path = ROOT / ('crane_project/configs/'
                   'crane_symeood_dino_geometry_refiner_'
                   'dual_tower_source_val_v2.py')
    text = path.read_text()
    assert "type='DinoConditionedDualTowerGeometryRefiner'" in text
    assert 'evaluation_only=True' in text
    assert "architecture='dual_tower_size_pose_v2'" in text
    assert 'source_gate_passed=False' in text
    assert 'target_data_read=False' in text
    assert 'fixed_test_read=False' in text
    assert 'domain_routing=False' in text
    assert 'sequence_frame_routing=False' in text
    assert 'temporal_state=False' in text


def test_dual_tower_v21_trains_only_size_with_source_pair_supervision():
    path = ROOT / ('crane_project/configs/'
                   'crane_symeood_dino_geometry_refiner_'
                   'dual_tower_size_source_v21.py')
    text = path.read_text()
    assert "type='DinoConditionedDualTowerGeometryRefiner'" in text
    assert 'train_size_tower=True' in text
    assert 'train_pose_tower=False' in text
    assert 'train_roi_extractor=False' in text
    assert 'decoded_geometry_loss_weight=0.25' in text
    assert 'temporal_size_loss_weight=0.20' in text
    assert 'source_adjacent_pair_supervision=True' in text
    assert 'inference_sequence_input=False' in text
    assert 'shuffle=False' in text
    assert 'target_data_read=False' in text
    assert 'fixed_test_read=False' in text
    assert 'source_gate_passed=False' in text
    assert 'domain_routing=False' in text
    assert 'sequence_frame_routing=False' in text
    assert 'temporal_state=False' in text
    assert 'ann_file=\'test/' not in text


def test_v21_trainer_hashes_frozen_pose_and_builds_only_adjacent_pairs():
    path = ROOT / ('mmrotate/models/detectors/'
                   'symeood_dino_geometry_refiner_trainer.py')
    text = path.read_text()
    assert 'frozen_refiner_hash' in text
    assert 'frozen_refiner_hash_unchanged' in text
    assert '_temporal_pair_indices' in text
    assert "second[2] == first[2] + 1" in text
    assert 'temporal_pair_indices=temporal_pairs' in text
    hook = (ROOT / ('mmrotate/core/hooks/'
                    'geometry_refiner_contract_hook.py')).read_text()
    assert 'Frozen geometry-refiner component received a gradient' in hook
    assert 'frozen_refiner_hash_unchanged' in hook


def test_causal_history_source_config_is_unified_and_target_closed():
    path = ROOT / ('crane_project/configs/'
                   'crane_symeood_dino_causal_history_refiner_source_v1.py')
    text = path.read_text()
    assert "type='DinoConditionedCausalHistoryRefiner'" in text
    assert "history_horizon = 4" in text
    assert "type='LoadCausalHistoryFromAudit'" in text
    assert "type='PrepareCausalHistoryInputs'" in text
    assert "type='CausalHistoryProposalAugment'" in text
    assert "type='FormatCausalHistoryInputs'" in text
    assert 'current_frame_anchored=True' in text
    assert 'bounded_history_residual=True' in text
    assert 'rejectable_history_gate=True' in text
    assert 'history_identity_model_input=False' in text
    assert 'fixed_target_parameter_selection=False' in text
    assert 'dino_detector_forward_during_training=False' in text
    assert 'frozen_symeood_feature_forward=True' in text
    assert 'cached_dino_proposals_only=True' in text
    assert "baseline_config='crane_project/configs/crane_symeood_k1.py'" in text
    assert "baseline_checkpoint='work_dirs/crane_symeood_k1/epoch_24.pth'" in text
    assert 'target_data_read=False' in text
    assert 'fixed_test_read=False' in text
    assert 'domain_routing=False' in text
    assert 'sequence_frame_routing=False' in text
    assert "ann_file='test/" not in text
    assert "expected_split='test'" not in text
    assert "type='SetNoFlipMetadata'" in text
    assert "type='RRandomFlip'" not in text


def test_causal_trainer_reuses_frozen_backbone_for_history_without_state():
    path = ROOT / ('mmrotate/models/detectors/'
                   'symeood_dino_geometry_refiner_trainer.py')
    text = path.read_text()
    assert 'extract_causal_history_feat' in text
    assert 'with torch.no_grad()' in text
    assert 'history_images.reshape' in text
    assert "hasattr(self.geometry_refiner, 'forward_causal')" in text
    assert 'causal_history_frame_keys' not in text


def test_causal_source_smoke_is_source_only_and_reports_cuda_peaks():
    path = ROOT / ('crane_project/tools/'
                   'symeood_dino_causal_history_source_smoke.py')
    text = path.read_text()
    assert "'ALLOW_CAUSAL_HISTORY_SOURCE_TRAINING'" in text
    assert "'STOP_CAUSAL_HISTORY_SOURCE_SMOKE_ERROR'" in text
    assert 'target_data_read=False' in text
    assert 'fixed_test_read=False' in text
    assert 'checkpoint_written=False' in text
    assert 'optimizer_steps_in_memory=1' in text
    assert 'loss.backward()' in text
    assert 'optimizer.step()' in text
    assert 'build_dataset(cfg.data.test)' not in text
    assert 'max_memory_allocated' in text
    assert 'max_memory_reserved' in text
    assert "os.environ.get('CUDA_VISIBLE_DEVICES')" in text
    assert 'explicit_no_flip_metadata' in text
    hook = ROOT / ('mmrotate/core/hooks/'
                   'geometry_refiner_contract_hook.py')
    hook_text = hook.read_text()
    assert 'class CudaPeakMemoryContractHook' in hook_text
    assert 'cuda_peak_memory_rank' in hook_text


def test_causal_source_gate_uses_dual_reference_and_cannot_open_test():
    path = ROOT / ('crane_project/tools/'
                   'symeood_dino_causal_history_source_gate.py')
    text = path.read_text()
    assert "PROTOCOL = 'causal_history_refiner_source_gate_v1'" in text
    assert "evidence_boundary='source_val_only'" in text
    assert 'target_data_read=False' in text
    assert 'fixed_test_read=False' in text
    assert 'eligible_for_fixed_test=False' in text
    assert 'eligible_for_unknown_sequence_claim=False' in text
    assert 'native_dino_reference_metrics' in text
    assert 'sym_eood_reference_metrics' in text
    assert "parser.add_argument('--sym-reference-results', required=True)" in text
    assert 'average_gain_over_native_dino' in text
    assert 'sym_eood_geometry_preservation' in text
    assert "'ALLOW_CAUSAL_HISTORY_CHECKPOINT_PROMOTION'" in text
    assert "'source-val'" in text
    assert "'fixed-target'" not in text


def test_k1_anchored_phase_v2_is_source_only_unified_and_bounded():
    path = ROOT / (
        'crane_project/configs/'
        'crane_symeood_dino_k1_anchored_causal_phase_refiner_source_v2.py')
    text = path.read_text()
    assert "type='K1AnchoredCausalPhaseGeometryRefiner'" in text
    assert "frozen_baseline_config='crane_project/configs/crane_symeood_k1.py'" in text
    assert 'current_k1_geometry_anchor=True' in text
    assert 'native_dino_anchor_fallback=True' in text
    assert 'native_dino_current_conditioning=True' in text
    assert 'same_forward_all_domains=True' in text
    assert 'continuous_double_angle_phase=True' in text
    assert 'bounded_current_residual=True' in text
    assert "representation='six_delta_xywh_sin2a_cos2a_residual'" in text
    assert 'domain_routing=False' in text
    assert 'sequence_frame_routing=False' in text
    assert 'target_data_read=False' in text
    assert 'fixed_test_read=False' in text
    assert "ann_file='test/" not in text
    assert "expected_split='test'" not in text
    assert 'lr=2e-5' in text


def test_k1_anchor_is_generated_from_shared_frozen_features():
    path = ROOT / ('mmrotate/models/detectors/'
                   'symeood_dino_geometry_refiner_trainer.py')
    text = path.read_text()
    assert 'self.uses_k1_geometry_anchor' in text
    assert 'def _k1_results_and_proposals' in text
    assert 'self.baseline.simple_test_from_features' in text
    assert 'conditioning_proposal_list' in text
    assert 'Formal fallback: no DINO means the frozen K1 output' in text
    assert 'self.baseline.extract_feat(img)' in text


def test_causal_smoke_and_gate_accept_v2_without_opening_test():
    smoke = (ROOT / ('crane_project/tools/'
                     'symeood_dino_causal_history_source_smoke.py')).read_text()
    gate = (ROOT / ('crane_project/tools/'
                    'symeood_dino_causal_history_source_gate.py')).read_text()
    assert 'ALLOW_K1_ANCHORED_CAUSAL_PHASE_SOURCE_TRAINING' in smoke
    assert 'phase_head_gradient_nonzero' in smoke
    assert 'conditioning_head_gradients_finite' in smoke
    assert 'build_dataset(cfg.data.test)' not in smoke
    assert 'k1_anchored_causal_phase_refiner_source_gate_v2' in gate
    assert 'ALLOW_K1_ANCHORED_CAUSAL_PHASE_CHECKPOINT_PROMOTION' in gate
    assert 'eligible_for_fixed_test=False' in gate


def test_ordinary_k1_source_val_reference_config_never_reads_fixed_test():
    path = ROOT / ('crane_project/configs/'
                   'crane_symeood_k1_source_val_eval.py')
    text = path.read_text()
    assert "_base_ = ['./crane_symeood_k1.py']" in text
    assert "ann_file='val/annfiles/'" in text
    assert "img_prefix='val/images/'" in text
    assert "ann_file='test/annfiles/'" not in text
    assert "img_prefix='test/images/'" not in text
    assert 'source_val_dataset' in text


def test_v21_source_smoke_is_source_only_and_requires_real_gradients():
    path = ROOT / ('crane_project/tools/'
                   'symeood_dino_dual_tower_v21_source_smoke.py')
    text = path.read_text()
    assert "decision='STOP_DUAL_TOWER_V21_SOURCE_SMOKE_ERROR'" in text
    assert "'ALLOW_DUAL_TOWER_V21_SOURCE_TRAINING'" in text
    assert 'checkpoint_written=False' in text
    assert 'target_data_read=False' in text
    assert 'fixed_test_read=False' in text
    assert 'loss.backward()' in text
    assert 'optimizer.step()' in text
    assert 'pose_gradients_none' in text
    assert 'frozen_pose_hash_unchanged' in text
    assert 'one_adjacent_pair_reported' in text
    assert 'build_dataset(cfg.data.test)' not in text


def test_v21_relaxed_gate_does_not_read_or_authorize_fixed_test():
    path = ROOT / ('crane_project/tools/'
                   'symeood_dino_dual_tower_v21_source_gate.py')
    text = path.read_text()
    assert "evidence_boundary='source_val_only'" in text
    assert 'target_data_read=False' in text
    assert 'fixed_test_read=False' in text
    assert 'eligible_for_fixed_test=False' in text
    assert "'ALLOW_DUAL_TOWER_V21_CHECKPOINT_PROMOTION'" in text
    assert 'source_gate_passed=False' in text


def test_v21_promotion_is_the_only_source_to_fixed_test_bridge():
    path = ROOT / ('crane_project/tools/'
                   'symeood_dino_dual_tower_v21_promote.py')
    text = path.read_text()
    assert "SOURCE_GATE_PROTOCOL = (\n" not in text
    assert "SOURCE_GATE_PROTOCOL = 'dual_tower_v21_relaxed_composite" in text
    assert "PROMOTION_PROTOCOL = 'source_gated_dual_tower_v21" in text
    assert "source_gate_passed=True" in text
    assert "selected_source_epoch=int(expected_epoch)" in text
    assert "eligible_for_one_fixed_test=True" in text
    assert "eligible_for_unknown_sequence_claim=False" in text
    assert "decision='ALLOW_ONE_DUAL_TOWER_V21_FIXED_TEST'" in text
    assert "target_data_read=False" in text
    assert "fixed_test_read=False" in text


def test_v21_fixed_test_is_one_forward_without_identity_routing():
    config = ROOT / ('crane_project/configs/'
                     'crane_symeood_dino_geometry_refiner_'
                     'dual_tower_v21_fixed_test.py')
    text = config.read_text()
    assert "expected_frame_count=992" in text
    assert "expected_split='test'" in text
    assert "ann_file='test/annfiles/'" in text
    assert "source_gate_passed=True" in text
    assert "selected_source_epoch=7" in text
    assert "domain_routing=False" in text
    assert "sequence_frame_routing=False" in text
    assert "temporal_state=False" in text
    assert "evaluation_only=True" in text
    assert text.count("evaluation_only=True") >= 2
    assert "type='DinoConditionedDualTowerGeometryRefiner'" in text
    assert 'del _handle' in text

    audit = ROOT / ('crane_project/tools/'
                    'symeood_dino_dual_tower_v21_fixed_test_audit.py')
    audit_text = audit.read_text()
    assert "EXPECTED_FRAME_COUNT = 992" in audit_text
    assert "dino_detector_rerun=False" in audit_text
    assert "parameter_update_after_test=False" in audit_text
    assert "epoch_reselection_after_test=False" in audit_text
    assert "eligible_for_parameter_tuning_from_this_report=False" in audit_text
    assert "eligible_for_unknown_sequence_claim=False" in audit_text
    assert "real_mcml_max_le_5" in audit_text
    assert "sim_mcml_max_le_5" in audit_text


def test_dual_tower_package_does_not_promote_source_gate():
    path = ROOT / ('crane_project/tools/'
                   'symeood_dino_dual_tower_v2_package.py')
    text = path.read_text()
    assert "source_gate_passed=False" in text
    assert "fixed_test_read=False" in text
    assert "eligible_for_fixed_test=False" in text
    assert "size_state['delta_head.weight'][2:4]" not in text
    utility_path = ROOT / ('crane_project/utils/'
                           'dual_tower_geometry_refiner_checkpoint.py')
    utility = utility_path.read_text()
    assert "size_state['delta_head.weight'][2:4]" in utility
    assert 'torch.tensor([0, 1, 4]' in utility


def test_dual_tower_audit_requires_source_val_and_exact_components():
    path = ROOT / ('crane_project/tools/'
                   'symeood_dino_dual_tower_v2_audit.py')
    text = path.read_text()
    assert 'EXPECTED_FRAME_COUNT = 738' in text
    assert "evidence_boundary='source_val_only'" in text
    assert "target_data_read=False" in text
    assert "fixed_test_read=False" in text
    assert "eligible_for_fixed_test=False" in text
    assert "full[0], full[1], size[2], size[3], full[4]" in text
    assert 'expected_hybrid_equivalence' in text


def test_batch_fallback_slices_every_fpn_level():
    trainer = (ROOT / ('mmrotate/models/detectors/'
                       'symeood_dino_geometry_refiner_trainer.py')).read_text()
    assert 'feature[index:index + 1] for feature in features' in trainer
    expected = ('simple_test_from_features(\n'
                '                    image_features, [img_metas[index]]')
    assert expected in trainer


def test_validation_unwraps_only_one_augmentation_dimension():
    path = ROOT / ('mmrotate/models/detectors/'
                   'symeood_dino_geometry_refiner_trainer.py')
    text = path.read_text()
    tree = ast.parse(text)
    helper = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == '_unwrap_single_augmentation_proposals')
    assert 'is_tensor' in _called_attributes(helper)
    assert 'supports exactly one test augmentation' in text
    methods = _class_methods(path, 'SymEOODDinoGeometryRefinerTrainer')
    simple_calls = [
        node.func.id for node in ast.walk(methods['simple_test'])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    assert '_unwrap_single_augmentation_proposals' in simple_calls
    assert 'DINO proposal/meta batch-size mismatch' in text


def test_trainer_public_init_preserves_loaded_frozen_checkpoint():
    path = ROOT / ('mmrotate/models/detectors/'
                   'symeood_dino_geometry_refiner_trainer.py')
    methods = _class_methods(path, 'SymEOODDinoGeometryRefinerTrainer')
    init_method = methods['init_weights']
    calls = _called_attributes(init_method)
    assert 'frozen_parameter_hash' in calls
    assert 'init_weights' not in calls
    assert 'requires_grad_' in calls
    assert 'eval' in calls


def test_train_entrypoint_preserves_checkpoint_contract_metadata():
    text = (ROOT / 'tools/train.py').read_text()
    assert "dict(cfg.checkpoint_config.get('meta') or {})" in text
    assert 'checkpoint_meta.update' in text
    assert 'cfg.checkpoint_config.meta = checkpoint_meta' in text


def test_real_stack_smoke_keeps_source_only_boundary():
    path = ROOT / ('crane_project/tools/'
                   'symeood_dino_geometry_refiner_source_smoke.py')
    smoke = path.read_text()
    assert "target_data_read=False" in smoke
    assert "checkpoint_written=False" in smoke
    assert 'build_dataset(full_cfg.data.train)' in smoke
    assert 'build_dataset(full_cfg.data.val)' in smoke
    assert 'build_dataset(full_cfg.data.test)' not in smoke
    assert 'RRandomFlip, RResize' in smoke
    assert 'resized_matches_manual_expected' in smoke
    assert "'_matches_manual_expected'" in smoke
    assert "'_changed_from_resized'" in smoke
    assert 'loss.backward()' in smoke
    assert 'optimizer.step()' in smoke
    assert 'raw_model.init_weights()' in smoke
    assert 'public_init_preserved_frozen_checkpoint' in smoke
    assert 'source_val_forward_output_valid' in smoke
    assert 'optimizer_has_no_inherited_momentum' in smoke
    assert 'no_legacy_top_level_loader_args' in smoke
    assert 'manual_source_gate_no_auto_best' in smoke
    assert 'evaluation_thresholds_preserved' in smoke
    assert 'full_cfg = compat_cfg(full_cfg)' in smoke
    assert "decision='STOP_SOURCE_PREFLIGHT_FAILED'" in smoke
    assert 'frozen_parameter_and_buffer_hash_unchanged' in smoke
