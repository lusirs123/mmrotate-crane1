"""Check that the new A/C comparison changes only feature supervision."""

import hashlib
from pathlib import Path

import pytest

from crane_project.tools import preflight_k1_dino_corrected_mask_ac_v4 as check


class AttrDict(dict):
    __getattr__ = dict.__getitem__


def _pair(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'epoch_20.pth'
    checkpoint.write_bytes(b'K1 fixture')
    monkeypatch.setattr(check, 'EXPECTED_BASELINE_SHA256',
                        hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    shared = dict(
        data=AttrDict(samples_per_gpu=2, train=[
            AttrDict(ann_file='train/annfiles/', img_prefix='train/images/'),
            AttrDict(ann_file='train_sim/annfiles/',
                     img_prefix='train/images/')]),
        train_pipeline=[dict(type='LoadDinoFeatureFromCache')],
        optimizer=AttrDict(lr=0.00025), runner=AttrDict(max_epochs=4),
        load_from=str(checkpoint), resume_from=None)
    common_model = dict(backbone=AttrDict(frozen_stages=4),
                        bbox_head=AttrDict(use_semantic_cls_adapter=True))
    a = AttrDict(shared, model=AttrDict(
        common_model, semantic_distillation=None),
        work_dir='work_dirs/crane_symeood_k1_dino_corrected_mask_a_v4')
    c = AttrDict(shared, model=AttrDict(
        common_model, semantic_distillation=AttrDict(
            enabled=True, mode='feature', scope='foreground',
            feature_level=0, loss_weight=0.05, max_tokens=4096,
            teacher_channels=1024, protect_geometry=False)),
        work_dir='work_dirs/crane_symeood_k1_dino_corrected_mask_c_v4')
    return checkpoint, [a, c]


def test_pair_contract_accepts_matching_arms(tmp_path, monkeypatch):
    checkpoint, configs = _pair(tmp_path, monkeypatch)
    assert check.validate_pair(configs) == checkpoint


def test_pair_contract_rejects_extra_changes(tmp_path, monkeypatch):
    _, configs = _pair(tmp_path, monkeypatch)
    configs[1]['data'] = AttrDict(configs[1].data, samples_per_gpu=1)
    with pytest.raises(ValueError, match='differ in data'):
        check.validate_pair(configs)
    configs[1]['data'] = configs[0].data
    configs[1].model.semantic_distillation['protect_geometry'] = True
    with pytest.raises(ValueError, match='feature-distillation contract'):
        check.validate_pair(configs)


def test_v4_configs_inherit_same_common_settings_and_separate_outputs():
    root = Path(__file__).resolve().parents[1] / 'crane_project/configs'
    for arm, override in [('a', 'semantic_distillation=None'),
                          ('c', 'protect_geometry=False')]:
        body = (root / ('crane_symeood_k1_dino_corrected_mask_{}_v4.py'
                        .format(arm))).read_text()
        assert 'fpn_gradient_common_v3.py' in body
        assert override in body
        assert 'corrected_mask_{}_v4'.format(arm) in body
