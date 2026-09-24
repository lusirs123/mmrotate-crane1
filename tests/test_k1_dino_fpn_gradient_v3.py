"""Guard the three-arm comparison against accidental extra differences."""

import hashlib
from pathlib import Path

import pytest

from crane_project.tools import preflight_k1_dino_fpn_gradient_v3 as probe


class AttrDict(dict):
    __getattr__ = dict.__getitem__


def _configs(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'epoch_20.pth'
    checkpoint.write_bytes(b'K1 fixture')
    monkeypatch.setattr(
        probe, 'EXPECTED_BASELINE_SHA256',
        hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    shared = dict(
        data=AttrDict(samples_per_gpu=2, train=[
            AttrDict(ann_file='train/annfiles/',
                     img_prefix='train/images/'),
            AttrDict(ann_file='train_sim/annfiles/',
                     img_prefix='train/images/')]),
        data_root='crane_project/data/crane_grab/',
        train_pipeline=[dict(type='LoadDinoFeatureFromCache')],
        test_pipeline=[],
        optimizer=AttrDict(lr=0.00025),
        optimizer_config={}, lr_config={},
        runner=AttrDict(max_epochs=4),
        checkpoint_config={}, evaluation={},
        load_from=str(checkpoint), resume_from=None)
    loss = dict(enabled=True, mode='feature', scope='foreground',
                feature_level=0, loss_weight=0.05, max_tokens=4096,
                teacher_channels=1024)
    configs = []
    for arm in 'ABC':
        distilled = None if arm == 'A' else AttrDict(
            loss, protect_geometry=(arm == 'B'))
        model = AttrDict(
            backbone=AttrDict(frozen_stages=4),
            bbox_head=AttrDict(use_semantic_cls_adapter=True),
            semantic_distillation=distilled)
        configs.append(AttrDict(shared, model=model))
    return checkpoint, configs


def test_three_arm_contract_accepts_only_matching_gradient_routing(
        tmp_path, monkeypatch):
    checkpoint, configs = _configs(tmp_path, monkeypatch)
    assert probe.validate_configs(configs) == checkpoint
    configs[2].model.semantic_distillation['protect_geometry'] = True
    with pytest.raises(ValueError, match='wrong distillation contract'):
        probe.validate_configs(configs)


def test_three_arm_contract_rejects_unmatched_freezing_and_data(
        tmp_path, monkeypatch):
    _, configs = _configs(tmp_path, monkeypatch)
    configs[1].model.backbone['frozen_stages'] = 1
    with pytest.raises(ValueError, match='does not freeze'):
        probe.validate_configs(configs)
    configs[1].model.backbone['frozen_stages'] = 4
    configs[2]['train_pipeline'] = []
    with pytest.raises(ValueError, match='differs in train_pipeline'):
        probe.validate_configs(configs)


def test_v3_configs_keep_distinct_arms_and_workdirs():
    root = Path(__file__).resolve().parents[1] / 'crane_project/configs'
    common = (root / 'crane_symeood_k1_dino_fpn_gradient_common_v3.py'
              ).read_text()
    assert "frozen_stages=4" in common
    assert "warmstart_common_v2.py" in common
    for arm, expected in [('a', 'semantic_distillation=None'),
                          ('b', 'protect_geometry=True'),
                          ('c', 'protect_geometry=False')]:
        body = (root / ('crane_symeood_k1_dino_fpn_gradient_{}_v3.py'
                        .format(arm))).read_text()
        assert expected in body
        assert 'fpn_gradient_common_v3.py' in body
        assert 'fpn_gradient_{}_v3'.format(arm) in body
