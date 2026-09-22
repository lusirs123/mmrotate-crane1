import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


loss_module = load_module(
    'semantic_feature_distill',
    'mmrotate/models/losses/semantic_feature_distill.py')
preflight = load_module(
    'dino_feature_cache_preflight_v1',
    'crane_project/tools/dino_feature_cache_preflight_v1.py')
exporter = load_module(
    'export_semantic_distilled_student_v1',
    'crane_project/tools/export_semantic_distilled_student_v1.py')


def test_feature_loss_detaches_teacher_and_updates_student_adapter():
    module = loss_module.SemanticFeatureDistillation(
        student_channels=4, teacher_channels=6, loss_weight=.2,
        feature_level=0, max_tokens=16)
    student = torch.randn(2, 4, 8, 8, requires_grad=True)
    teacher = torch.randn(2, 6, 4, 4, requires_grad=True)
    mask = torch.zeros(2, 1, 8, 8)
    mask[:, :, 2:5, 1:6] = 1
    loss = module((student,), teacher, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert student.grad is not None
    assert module.project.weight.grad is not None
    assert teacher.grad is None


def test_feature_loss_validates_batch_and_level():
    module = loss_module.SemanticFeatureDistillation(
        student_channels=4, teacher_channels=6, feature_level=1)
    with pytest.raises(ValueError, match='level 1'):
        module((torch.randn(1, 4, 2, 2),), torch.randn(1, 6, 2, 2))
    module = loss_module.SemanticFeatureDistillation(
        student_channels=4, teacher_channels=6)
    with pytest.raises(ValueError, match='batch size mismatch'):
        module((torch.randn(2, 4, 2, 2),), torch.randn(1, 6, 2, 2))


def test_empty_foreground_mask_does_not_distill_background():
    module = loss_module.SemanticFeatureDistillation(
        student_channels=4, teacher_channels=6)
    student = torch.randn(1, 4, 2, 2, requires_grad=True)
    teacher = torch.randn(1, 6, 2, 2)
    loss = module((student,), teacher, torch.zeros(1, 1, 2, 2))
    assert loss.item() == 0.0


def make_cache(tmp_path, corrupt=False):
    image_dir = tmp_path / 'data' / 'train' / 'images'
    image_dir.mkdir(parents=True)
    image = image_dir / 'real_seq01_00001.jpg'
    image.write_bytes(b'image')
    annotation_dir = tmp_path / 'data' / 'train' / 'annfiles'
    annotation_dir.mkdir(parents=True)
    (annotation_dir / 'real_seq01_00001.txt').write_text('label')
    stat = image.stat()
    cache_dir = tmp_path / 'cache' / 'train'
    cache_dir.mkdir(parents=True)
    cache = cache_dir / 'real_seq01_00001_deadbeef.pth'
    feature = torch.randn(1, 7 if corrupt else 8, 3, 5)
    torch.save(dict(
        signature=dict(
            image=dict(path=str(image), size=stat.st_size,
                       mtime_ns=stat.st_mtime_ns),
            dinov2_model='dinov2_vitl14'),
        feature=feature, dino_meta={}, frozen_dinov2=True), cache)
    return image, cache_dir.parent


def test_cache_preflight_accepts_complete_identity_bound_cache(tmp_path):
    _image, cache_dir = make_cache(tmp_path)
    report = preflight.inspect_cache(
        str(tmp_path / 'data'), str(cache_dir), ['train:train'],
        expected_channels=8)
    assert report['complete'] is True
    assert report['valid_count'] == 1
    assert report['records'][0]['status'] == 'valid'


def test_cache_preflight_rejects_wrong_channel_count(tmp_path):
    _image, cache_dir = make_cache(tmp_path, corrupt=True)
    report = preflight.inspect_cache(
        str(tmp_path / 'data'), str(cache_dir), ['train:train'],
        expected_channels=8)
    assert report['complete'] is False
    assert report['missing_or_ambiguous_count'] == 1


def test_cache_preflight_rejects_requested_split_without_images(tmp_path):
    _image, cache_dir = make_cache(tmp_path)
    report = preflight.inspect_cache(
        str(tmp_path / 'data'), str(cache_dir),
        ['train:train', 'train_sim:train'],
        expected_channels=8)
    assert report['complete'] is False
    assert report['missing_splits'] == ['train_sim']


def test_config_and_inference_keep_teacher_out_of_student_path():
    config = (ROOT / 'crane_project/configs/'
              'crane_symeood_k1_dino_semantic_distill_v1.py').read_text()
    detector = (ROOT / 'mmrotate/models/detectors/'
                'sym_eood_detector.py').read_text()
    head = (ROOT / 'mmrotate/models/dense_heads/'
            'sym_eood_head.py').read_text()
    assert "protect_geometry=True" in config
    assert "samples_per_gpu=1" in config
    simple_test = detector.split('    def simple_test(', 1)[1]
    simple_test = simple_test.split('    def ', 1)[0]
    assert 'teacher_features' not in simple_test
    method = head.split(
        '    def forward_semantic_distillation_features', 1)[1]
    method = method.split('    @force_fp32', 1)[0]
    assert 'feat.detach() if protect_geometry else feat' in method


def test_student_export_strips_only_training_adapter():
    payload = dict(
        state_dict={
            'backbone.layer.weight': torch.ones(1),
            'bbox_head.cls.weight': torch.ones(1) * 2,
            'semantic_distillation.project.weight': torch.ones(1) * 3},
        meta={'epoch': 2}, optimizer={'state': {}})
    output, removed = exporter.student_only_payload(payload)
    assert removed == ['semantic_distillation.project.weight']
    assert set(output['state_dict']) == {
        'backbone.layer.weight', 'bbox_head.cls.weight'}
    assert output['meta']['epoch'] == 2
    assert 'optimizer' not in output
    assert output['meta']['student_only_export_protocol'] == exporter.PROTOCOL
    assert 'semantic_distillation.project.weight' in payload['state_dict']


def test_student_export_rejects_non_distilled_checkpoint():
    with pytest.raises(ValueError, match='no semantic distillation'):
        exporter.student_only_payload(
            {'state_dict': {'backbone.weight': torch.ones(1)}})
