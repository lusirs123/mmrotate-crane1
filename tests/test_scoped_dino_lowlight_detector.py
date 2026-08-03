"""CPU-only tests for the integrated scoped-DINO detector policy."""

import abc
import importlib.util
import inspect
import pathlib
import runpy
import sys
import types

import numpy as np
import pytest
import torch
import torch.nn as nn


class _Registry:
    def register_module(self, force=False):
        del force
        return lambda cls: cls


class _RotatedBaseDetector(nn.Module):
    def __init__(self, init_cfg=None):
        del init_cfg
        super().__init__()

    @abc.abstractmethod
    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError


def _load_module():
    package_names = ('mmrotate', 'mmrotate.models',
                     'mmrotate.models.detectors')
    modules = {name: types.ModuleType(name) for name in package_names}
    for module in modules.values():
        module.__path__ = []
    builder = types.ModuleType('mmrotate.models.builder')
    builder.ROTATED_DETECTORS = _Registry()
    base = types.ModuleType('mmrotate.models.detectors.base')
    base.RotatedBaseDetector = _RotatedBaseDetector
    modules[builder.__name__] = builder
    modules[base.__name__] = base
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        root = pathlib.Path(__file__).resolve().parents[1]
        path = root / 'mmrotate/models/detectors/scoped_dino_lowlight_detector.py'
        name = 'mmrotate.models.detectors.scoped_dino_lowlight_detector'
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


MODULE = _load_module()
Detector = MODULE.ScopedDinoLowlightDetector


def test_integrated_detector_satisfies_base_abstract_interface():
    assert not inspect.isabstract(Detector)


def test_aug_test_is_explicitly_rejected_to_preserve_sequence_state():
    detector = _detector()
    with pytest.raises(RuntimeError, match='does not support'):
        detector.aug_test([], [], rescale=False)


def _detector(alpha=0.25):
    detector = Detector.__new__(Detector)
    nn.Module.__init__(detector)
    detector._alpha = alpha
    detector._stabilizer_enabled = True
    detector._previous_box = None
    detector._previous_seq = None
    detector._previous_frame = None
    return detector


def _box(cx, width, height, angle, score=0.8):
    return np.asarray([[cx, 4.0, width, height, angle, score]],
                      dtype=np.float32)


def test_integrated_stabilizer_matches_causal_geometry_rules():
    detector = _detector(alpha=0.5)
    first = detector._stabilize(_box(1, 10, 20, 0), 'seq02', 1)
    second = detector._stabilize(_box(2, 40, 80, 0), 'seq02', 2)
    assert np.array_equal(first, _box(1, 10, 20, 0))
    assert second[0, :2] == pytest.approx([2, 4])
    assert second[0, 2:4] == pytest.approx([20, 40])
    assert second[0, 5] == pytest.approx(0.8)


def test_integrated_stabilizer_resets_on_frame_gap():
    detector = _detector(alpha=0.25)
    detector._stabilize(_box(1, 10, 20, 0), 'seq02', 1)
    after_gap = detector._stabilize(_box(3, 40, 80, 0), 'seq02', 3)
    assert np.array_equal(after_gap, _box(3, 40, 80, 0))


def test_brightaug_checkpoint_keys_load_into_registered_baseline():
    detector = _detector()
    detector.baseline = nn.Linear(2, 1)
    state = {
        'weight': torch.tensor([[3.0, 4.0]]),
        'bias': torch.tensor([5.0]),
    }
    detector.load_state_dict(state, strict=True)
    assert detector.baseline.weight.detach().tolist() == [[3.0, 4.0]]
    assert detector.baseline.bias.detach().tolist() == [5.0]


def test_mmcv_recursive_loader_path_maps_unprefixed_checkpoint_keys():
    detector = _detector()
    detector.baseline = nn.Linear(2, 1)
    state = {
        'weight': torch.tensor([[6.0, 7.0]]),
        'bias': torch.tensor([8.0]),
    }
    # MMCV 1.x uses its own recursive loader and calls _load_from_state_dict
    # directly instead of the detector's public load_state_dict method.
    nn.Module.load_state_dict(detector, state, strict=True)
    assert detector.baseline.weight.detach().tolist() == [[6.0, 7.0]]
    assert detector.baseline.bias.detach().tolist() == [8.0]


def test_scope_lookup_is_closed_outside_declared_intervals(tmp_path):
    manifest = tmp_path / 'scope.json'
    manifest.write_text(
        '{"entries":[{"split":"test","seq":"real_seq02",'
        '"start":137,"end":169,"dino_enabled":true}]}')
    intervals = MODULE._load_scope(str(manifest), 'test')
    assert MODULE._in_scope(intervals, 'real_seq02', 137)
    assert MODULE._in_scope(intervals, 'real_seq02', 169)
    assert not MODULE._in_scope(intervals, 'real_seq02', 136)
    assert not MODULE._in_scope(intervals, 'real_seq03', 150)


def test_filename_parser_keeps_domain_prefix_used_by_scope_manifest():
    seq, frame = MODULE._sequence_frame('/tmp/real_seq02_00137.jpg')
    assert seq == 'real_seq02'
    assert frame == 137


def test_formal_config_builds_integrated_model_and_paper_metrics():
    root = pathlib.Path(__file__).resolve().parents[1]
    config = runpy.run_path(
        str(root / 'crane_project/configs/'
            'crane_symeood_scoped_dino_lowlight_v1.py'))
    assert config['model']['type'] == 'ScopedDinoLowlightDetector'
    assert config['model']['stabilizer']['alpha'] == 0.25
    assert config['model']['stabilizer']['target_used_for_selection'] is False
    assert config['evaluation']['paper_temporal'] is True


def test_unified_temporal_config_removes_target_slice_routing():
    root = pathlib.Path(__file__).resolve().parents[1]
    config = runpy.run_path(
        str(root / 'crane_project/configs/'
            'crane_symeood_scoped_dino_lowlight_s7_temporal_association_v1.py'))
    assert config['model']['scope_policy'] == 'all_frames'
    assert config['model']['scope_manifest'] is None
    assert config['model']['temporal_association']['enabled'] is True
    head = config['model']['dino_rescue']['head']
    assert head['s7_temporal_association'] is True
    assert head['s7_quality_suppression'] is False
    assert head['s7_lane_arbitration'] is False
    assert config['model']['temporal_association']['source_selected'] is True
    assert config['model']['temporal_association']['source_gate'] == dict(
        min_full_top1=688, min_small_top1=311, max_mcml=3)


def test_temporal_quality_config_is_source_only_and_unified():
    root = pathlib.Path(__file__).resolve().parents[1]
    config = runpy.run_path(
        str(root / 'crane_project/configs/'
            'crane_symeood_scoped_dino_lowlight_s7_temporal_quality_association_v1.py'))
    assert config['model']['scope_policy'] == 'all_frames'
    assert config['model']['scope_manifest'] is None
    assert config['model']['temporal_association']['target_used_for_selection'] is False
    head = config['model']['dino_rescue']['head']
    assert head['s7_temporal_association'] is True
    assert head['s7_temporal_quality_head'] is True
    assert head['s7_temporal_quality_hidden'] == 128
    assert config['s7_temporal_quality_training']['target_read'] is False
    assert config['s7_temporal_quality_training']['positive_promotion'] is False
    assert config['s7_temporal_quality_training']['gain_replay'] is False


def test_full_test_manifest_covers_stream_and_enables_exact_dark_slice():
    root = pathlib.Path(__file__).resolve().parents[1]
    manifest = root / (
        'crane_project/configs/scopes/'
        'full_test_seq02_lowlight_diagnosis.json')
    intervals = MODULE._load_scope(str(manifest), 'test')
    images = sorted((root / 'crane_project/data/crane_grab/test/images').glob(
        '*.jpg'))
    covered = 0
    enabled = []
    for path in images:
        seq, frame = MODULE._sequence_frame(str(path))
        matches = [(start, end, value)
                   for start, end, value in intervals.get(seq, ())
                   if start <= frame <= end]
        assert len(matches) == 1
        covered += 1
        if matches[0][2]:
            enabled.append((seq, frame))
    assert covered == 992
    assert enabled == [('real_seq02', frame)
                       for frame in range(137, 170)]
