from pathlib import Path

import pytest

from crane_project.tools.preflight_k1_dino_warmstart_v2 import (
    EXPECTED_BASELINE_SHA256, validate_cache_reports,
    validate_configs, validate_source_identity)


class AttrDict(dict):
    __getattr__ = dict.__getitem__


def _pair(tmp_path):
    checkpoint = tmp_path / 'epoch_20.pth'
    checkpoint.write_bytes(b'placeholder')
    pipeline = [dict(type='LoadImageFromFile'),
                dict(type='LoadDinoFeatureFromCache',
                     cache_dir=str(tmp_path / 'cache'))]
    data = AttrDict(samples_per_gpu=2, train=[
        AttrDict(ann_file='train/annfiles/',
                 img_prefix='train/images/'),
        AttrDict(ann_file='train_sim/annfiles/',
                 img_prefix='train/images/')])
    shared = dict(data=data, data_root=str(tmp_path / 'data'),
                  optimizer=AttrDict(lr=0.00025),
                  optimizer_config=dict(grad_clip=10),
                  lr_config=dict(step=[3]),
                  runner=AttrDict(max_epochs=4),
                  checkpoint_config=dict(interval=1),
                  evaluation=dict(interval=1),
                  train_pipeline=pipeline, test_pipeline=[],
                  load_from=str(checkpoint), resume_from=None)
    control = AttrDict(shared, model=dict(
        bbox_head=dict(use_semantic_cls_adapter=True),
        semantic_distillation=None))
    distill = AttrDict(shared, model=dict(
        bbox_head=dict(use_semantic_cls_adapter=True),
        semantic_distillation=dict(enabled=True, loss_weight=0.05)))
    return checkpoint, control, distill


def test_preflight_accepts_only_source_selected_checkpoint_and_paired_config(
        tmp_path):
    checkpoint, control, distill = _pair(tmp_path)
    selection = dict(metric_protocol_version=2,
                     evidence_role='source_val_checkpoint_selection',
                     selected_checkpoint='epoch_20',
                     selected_path=str(checkpoint),
                     all_checkpoints=dict(epoch_20=dict(
                         checkpoint=str(checkpoint),
                         checkpoint_sha256=EXPECTED_BASELINE_SHA256)))
    final = dict(protocol='crane_ckpt_sweep_final_test_v2',
                 metric_protocol_version=2,
                 checkpoint=str(checkpoint),
                 checkpoint_sha256=EXPECTED_BASELINE_SHA256)
    assert validate_source_identity(
        selection, final, EXPECTED_BASELINE_SHA256,
        checkpoint) == 'epoch_20'
    assert validate_configs(control, distill, checkpoint)['loss_weight'] == .05
    selection['selected_checkpoint'] = 'epoch_24'
    with pytest.raises(ValueError, match='identity mismatch'):
        validate_source_identity(selection, final,
                                 EXPECTED_BASELINE_SHA256, checkpoint)


def test_preflight_rejects_hidden_data_schedule_or_model_difference(tmp_path):
    checkpoint, control, distill = _pair(tmp_path)
    distill['data'] = AttrDict(control.data, samples_per_gpu=1)
    with pytest.raises(ValueError, match='differ in data'):
        validate_configs(control, distill, checkpoint)
    distill['data'] = control.data
    distill['model']['bbox_head']['use_semantic_cls_adapter'] = False
    with pytest.raises(ValueError, match='model difference'):
        validate_configs(control, distill, checkpoint)
    distill['model']['bbox_head']['use_semantic_cls_adapter'] = True
    control['train_pipeline'] = [dict(type='LoadImageFromFile')]
    distill['train_pipeline'] = control.train_pipeline
    with pytest.raises(ValueError, match='same frozen DINO cache'):
        validate_configs(control, distill, checkpoint)


def test_configs_declare_same_base_and_separate_workdirs():
    root = Path(__file__).resolve().parents[1] / 'crane_project/configs'
    common = (root / 'crane_symeood_k1_dino_warmstart_common_v2.py').read_text()
    control = (root / 'crane_symeood_k1_dino_warmstart_control_v2.py').read_text()
    distill = (root / 'crane_symeood_k1_dino_warmstart_distill_v2.py').read_text()
    assert "load_from = 'work_dirs/crane_symeood_k1/epoch_20.pth'" in common
    assert "warmup_iters=100" in common
    assert "semantic_distillation=None" in control
    assert 'warmstart_common_v2.py' in control
    assert 'warmstart_common_v2.py' in distill
    assert 'warmstart_control_v2' in control
    assert 'warmstart_distill_v2' in distill


def test_cache_receipts_must_cover_both_source_splits(tmp_path):
    import json
    _checkpoint, control, _distill = _pair(tmp_path)
    paths = []
    for dataset, count in [('train:train', 2033),
                           ('train_sim:train', 748)]:
        path = tmp_path / (dataset.split(':')[0] + '.json')
        path.write_text(json.dumps(dict(
            protocol='dino_feature_cache_preflight_v1',
            datasets=[dataset], complete=True, image_count=count,
            valid_count=count, missing_or_ambiguous_count=0,
            cache_load_error_count=0, expected_channels=1024,
            expected_model='dinov2_vitl14',
            data_root=str(tmp_path / 'data'),
            cache_dir=str(tmp_path / 'cache'))))
        paths.append(path)
    assert len(validate_cache_reports(paths, control)) == 2
    payload = json.loads(paths[1].read_text())
    payload['valid_count'] = 747
    paths[1].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='cache receipt'):
        validate_cache_reports(paths, control)
