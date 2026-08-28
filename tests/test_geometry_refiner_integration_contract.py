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


def test_batch_fallback_slices_every_fpn_level():
    trainer = (ROOT / ('mmrotate/models/detectors/'
                       'symeood_dino_geometry_refiner_trainer.py')).read_text()
    assert 'feature[index:index + 1] for feature in features' in trainer
    expected = ('simple_test_from_features(\n'
                '                    image_features, [img_metas[index]]')
    assert expected in trainer


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
    assert 'optimizer_has_no_inherited_momentum' in smoke
    assert "decision='STOP_SOURCE_PREFLIGHT_FAILED'" in smoke
    assert 'frozen_parameter_and_buffer_hash_unchanged' in smoke
