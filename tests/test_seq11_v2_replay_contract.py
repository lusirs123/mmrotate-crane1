"""Static contracts for the seq11-v2 replay stage."""

import ast
from pathlib import Path


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
            'coverage_contract'} <= methods
    text = path.read_text()
    assert 'batch = index // self.samples_per_batch' in text
    assert 'cycle = self.original_batches_per_auxiliary_batch + 1' in text
    assert "sample['source_replay_is_auxiliary']" in text
    assert 'epoch_offset = self.epoch * samples_per_epoch' in text


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
    assert 'Base-V3 teacher received a gradient' in hook
    assert 'class FixedRatioReplayEpochHook' in hook
