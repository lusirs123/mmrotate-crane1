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


def test_filename_parser_accepts_webots_frame_names():
    seq, frame = MODULE._sequence_frame('/tmp/frame_00042.jpg')
    assert seq == 'frame'
    assert frame == 42


def test_formal_config_builds_integrated_model_and_paper_metrics():
    root = pathlib.Path(__file__).resolve().parents[1]
    config = runpy.run_path(
        str(root / 'crane_project/configs/'
            'crane_symeood_scoped_dino_lowlight_v1.py'))
    assert config['model']['type'] == 'ScopedDinoLowlightDetector'
    assert config['model']['stabilizer']['alpha'] == 0.25
    assert config['model']['stabilizer']['target_used_for_selection'] is False
    assert config['evaluation']['paper_temporal'] is True


def _formal_native_payload(alpha=0.5, s7_enabled=False):
    return dict(
        source_only_fc_cls_interpolation=dict(
            selector='Frozen DINO ROI Classifier Source Interpolation '
                     'Selector V1',
            protocol_version=1,
            alpha=alpha,
            target_data_read=False,
            source_gate=dict(passed=True)),
        s7_architecture=dict(enabled=s7_enabled),
        best_source_val_summary=dict(top1_hits=677, top1_mcml=3),
        best_source_small_val_summary=dict(top1_hits=303, top1_mcml=3))


def _formal_native_contract():
    return dict(
        selector='Frozen DINO ROI Classifier Source Interpolation Selector V1',
        protocol_version=1,
        alpha=0.5,
        require_target_unread=True,
        require_source_gate=True,
        require_s7_disabled=True,
        source_full=dict(top1_hits=677, top1_mcml=3),
        source_small=dict(top1_hits=303, top1_mcml=3))


def test_formal_native_checkpoint_contract_accepts_only_locked_baseline():
    MODULE._validate_dino_checkpoint_contract(
        _formal_native_payload(), _formal_native_contract())
    with pytest.raises(RuntimeError, match='interpolation mismatch'):
        MODULE._validate_dino_checkpoint_contract(
            _formal_native_payload(alpha=0.25), _formal_native_contract())
    with pytest.raises(RuntimeError, match='unexpectedly enables S7'):
        MODULE._validate_dino_checkpoint_contract(
            _formal_native_payload(s7_enabled=True),
            _formal_native_contract())


def test_formal_native_s14_component_config_is_pure_and_fixed():
    root = pathlib.Path(__file__).resolve().parents[1]
    config = runpy.run_path(
        str(root / 'crane_project/configs/'
            'crane_symeood_formal_dino_native_s14_v1.py'))
    model = config['model']
    assert model['type'] == 'FrozenDinoNativeS14Detector'
    assert model['scope_policy'] == 'all_frames'
    assert model['scope_manifest'] is None
    assert model['runtime_checkpoint_in_constructor'] is True
    assert model['stabilizer']['enabled'] is False
    assert model['temporal_association']['enabled'] is False
    assert 'baseline_config' not in model
    assert model['dino_checkpoint_contract']['alpha'] == pytest.approx(0.5)
    assert model['dino_rescue']['head']['feature_strides'] == [14]
    assert model['dino_rescue']['head']['s7_residual'] is False
    formal = config['formal_detection_contract']
    assert formal['symeood_enabled'] is False
    assert formal['brightaug_enabled'] is False
    assert formal['s7_enabled'] is False
    assert formal['target_scope'] is False


def test_pure_dino_simple_test_does_not_require_fusion_audit(monkeypatch):
    class _Common:
        @staticmethod
        def resize_and_normalize_bgr(image, height, patch_size,
                                     max_long_side):
            del image, height, patch_size, max_long_side
            return torch.zeros((1, 3, 14, 14)), dict(scale_factor=1.0)

        @staticmethod
        def extract_patch_grid(dino, tensor, patch_size):
            del dino, tensor, patch_size
            return torch.zeros((1, 1, 1, 1))

    class _Heads:
        _last_temporal_pool = None

        @staticmethod
        def simple_test(feature, feature_meta):
            del feature, feature_meta
            return np.asarray(
                [[10, 20, 30, 12, 0.1, 0.9]], dtype=np.float32)

    class _Labeller:
        @staticmethod
        def feature_meta(image_path, dino_meta):
            del image_path
            return dino_meta

        @staticmethod
        def filter_valid_rotated_detections(detections, feature_meta):
            del feature_meta
            return detections, dict()

    detector = Detector.__new__(Detector)
    nn.Module.__init__(detector)
    detector.baseline = None
    detector._fusion_policy = 'dino_primary'
    detector._scope_policy = 'all_frames'
    detector._scope_intervals = None
    detector._conditional_dino_selector = None
    detector._stabilizer_enabled = False
    detector._previous_box = None
    detector._previous_seq = None
    detector._previous_frame = None
    detector.__dict__['_dino_runtime'] = dict(
        common=_Common(), dino=object(), heads=_Heads(),
        labeller=_Labeller(), height=600, patch_size=14,
        max_long_side=1333, dino_device=torch.device('cpu'),
        head_device=torch.device('cpu'), temporal_selector=None)
    monkeypatch.setattr(
        MODULE.cv2, 'imread',
        lambda path, mode: np.zeros((32, 32, 3), dtype=np.uint8))

    result = detector.simple_test(
        torch.zeros((1, 3, 32, 32)),
        [dict(filename='/tmp/frame_00000.jpg',
              ori_shape=(32, 32, 3))],
        rescale=True)

    assert result[0][0].shape == (1, 6)
    assert result[0][0][0].tolist() == pytest.approx(
        [10, 20, 30, 12, 0.1, 0.9])


def test_formal_unified_config_keeps_symeood_and_common_dino_ranking():
    root = pathlib.Path(__file__).resolve().parents[1]
    config = runpy.run_path(
        str(root / 'crane_project/configs/'
            'crane_symeood_dino_unified_v1.py'))
    model = config['model']
    assert model['type'] == 'SymEOODDinoUnifiedDetector'
    assert model['baseline_config'].endswith(
        'crane_symeood_k1_brightaug.py')
    assert not pathlib.Path(model['baseline_config']).is_absolute()
    assert model['fusion_policy'] == 'sym_eood_proposal_dino_roi_union'
    assert model['scope_policy'] == 'all_frames'
    assert model['scope_manifest'] is None
    assert model['stabilizer']['enabled'] is False
    assert model['temporal_association']['enabled'] is False
    assert model['fusion_audit_enabled'] is True
    assert model['dino_head_checkpoint'].endswith(
        'dino_teacher_fc_cls_interpolation_v1/'
        'source_safe_interpolated_head.pth')
    assert not pathlib.Path(model['dino_head_checkpoint']).is_absolute()
    head = model['dino_rescue']['head']
    assert head['feature_strides'] == [14]
    assert head['roi_nms_iou_thr'] == pytest.approx(0.5)
    assert head['s7_residual'] is False
    assert config['dino_checkpoint_contract']['alpha'] == pytest.approx(0.5)
    formal = config['formal_detection_contract']
    assert formal['proposal_sources'] == [
        'symeood_k1_brightaug_top1', 'frozen_dino_native_s14_rpn']
    assert formal['sym_eood_checkpoint'].endswith(
        'crane_symeood_k1_brightaug/epoch_20.pth')
    assert formal['common_ranker'] == 'frozen_dino_roi_classifier_alpha05'
    assert formal['source_owned_geometry'] is True
    assert formal['raw_cross_model_score_comparison'] is False
    assert formal['target_scope'] is False
    assert formal['sequence_identity_routing'] is False
    assert formal['brightaug'] is True


def test_symeood_proposal_is_scaled_and_raw_score_is_discarded():
    baseline = np.asarray([[10, 20, 30, 40, 0.25, 0.123]], np.float32)
    proposal = MODULE.ScopedDinoLowlightDetector._sym_eood_proposals_for_dino(
        baseline,
        dict(scale_factor=np.asarray([2, 2, 2, 2], np.float32)),
        torch.device('cpu'))
    assert proposal.shape == (1, 6)
    assert proposal[0, :5].tolist() == pytest.approx(
        [20, 40, 60, 80, 0.25])
    assert proposal[0, 5].item() == pytest.approx(1.0)


class _UnifiedHeads:
    def __init__(self, scores):
        self.scores = torch.tensor(scores, dtype=torch.float32)

    def simple_test_proposals(self, feature, feature_meta):
        del feature, feature_meta
        native = torch.tensor([
            [1, 2, 3, 4, 0.1, 0.4],
            [5, 6, 7, 8, 0.2, 0.3],
        ], dtype=torch.float32)
        return [torch.empty(0)], [native]

    def _decode_roi_candidates(self, feature, feature_meta, proposals,
                               rescale):
        del feature, feature_meta, rescale
        # The final row intentionally differs from the external proposal.  A
        # correct source-owned selector must never return this regressed row
        # when SymEOOD wins.
        decoded = torch.tensor([
            [11, 12, 13, 14, 0.11],
            [21, 22, 23, 24, 0.22],
            [91, 92, 93, 94, 0.99],
        ], dtype=torch.float32)
        assert decoded.shape[0] == proposals.shape[0]
        return (decoded, self.scores, self.scores,
                torch.empty((3, 0), dtype=torch.float32))


class _AllValidLabeller:
    @staticmethod
    def valid_rotated_detection_mask(detections, feature_meta):
        del feature_meta
        return np.ones(detections.shape[0], dtype=bool)


def _unified_selector(scores):
    detector = Detector.__new__(Detector)
    nn.Module.__init__(detector)
    detector._test_score_thr = 0.05
    detector._conservative_selector = None
    detector.__dict__['_dino_runtime'] = dict(
        heads=_UnifiedHeads(scores), labeller=_AllValidLabeller())
    return detector


def test_unified_selector_preserves_symeood_geometry_when_external_wins():
    detector = _unified_selector([0.6, 0.7, 0.9])
    baseline = np.asarray([[31, 32, 33, 34, 0.33, 0.2]], np.float32)
    ranked, audit = detector._dino_test_with_sym_eood_proposal(
        torch.empty(0), dict(scale_factor=np.ones(4, np.float32)), baseline)
    assert ranked[0, :5].tolist() == pytest.approx(baseline[0, :5])
    assert ranked[0, 5] == pytest.approx(0.9)
    assert audit['selected_source'] == 'sym_eood'
    assert audit['sym_eood_geometry_preserved'] is True
    assert audit['sym_eood_original_box'] == pytest.approx(
        baseline[0].tolist())
    assert audit['sym_eood_original_score'] == pytest.approx(0.2)


def test_unified_selector_keeps_dino_regression_when_native_wins():
    detector = _unified_selector([0.95, 0.7, 0.9])
    baseline = np.asarray([[31, 32, 33, 34, 0.33, 0.2]], np.float32)
    ranked, audit = detector._dino_test_with_sym_eood_proposal(
        torch.empty(0), dict(scale_factor=np.ones(4, np.float32)), baseline)
    assert ranked[0, :5].tolist() == pytest.approx(
        [11, 12, 13, 14, 0.11])
    assert audit['selected_source'] == 'dino_native'


def test_unified_source_val_config_uses_val_split():
    root = pathlib.Path(__file__).resolve().parents[1]
    config = runpy.run_path(
        str(root / 'crane_project/configs/'
            'crane_symeood_dino_unified_source_val_v1.py'))
    assert config['data']['test']['ann_file'] == 'val/annfiles/'
    assert config['data']['test']['img_prefix'] == 'val/images/'
    assert not pathlib.Path(config['data']['test']['data_root']).is_absolute()


def test_fusion_audit_metadata_reports_hidden_runtime_parameters():
    detector = _detector()
    detector.baseline = nn.Linear(2, 1)
    detector._fusion_policy = 'sym_eood_proposal_dino_roi_union'
    detector._test_score_thr = 0.05
    detector._conservative_selector = None
    detector._conservative_takeover_calibration = None
    detector.__dict__['_dino_runtime'] = dict(
        dino=nn.Linear(3, 2), heads=nn.Linear(4, 3))
    counts = detector.fusion_audit_metadata()[
        'resource_summary']['parameter_counts']
    assert counts['sym_eood']['total'] == 3
    assert counts['dinov2']['total'] == 8
    assert counts['dino_heads']['total'] == 15
    assert counts['combined_runtime']['total'] == 26


def test_conservative_takeover_config_requires_source_calibration():
    root = pathlib.Path(__file__).resolve().parents[1]
    config = runpy.run_path(
        str(root / 'crane_project/configs/'
            'crane_symeood_dino_conservative_takeover_v2.py'))
    takeover = config['model']['conservative_takeover']
    assert takeover['enabled'] is True
    assert not pathlib.Path(takeover['calibration_json']).is_absolute()
    contract = config['formal_conservative_takeover_contract']
    assert contract['selection_split'] == 'val'
    assert contract['metric_protocol_version'] == 2
    assert contract['target_data_read'] is False
    assert contract['test_parameter_search'] is False


def test_lane_isolated_v3_configs_keep_source_and_fixed_test_separate():
    root = pathlib.Path(__file__).resolve().parents[1]
    source = runpy.run_path(str(
        root / 'crane_project/configs/'
        'crane_symeood_dino_lane_isolated_source_val_v3.py'))
    source_contract = source['formal_lane_isolated_source_contract']
    assert source_contract['split'] == 'val'
    assert source_contract['expected_frames'] == 738
    assert source_contract['target_data_read'] is False
    fixed = runpy.run_path(str(
        root / 'crane_project/configs/'
        'crane_symeood_dino_lane_isolated_conditional_v3.py'))
    conditional = fixed['model']['conditional_dino']
    assert conditional['enabled'] is True
    assert not pathlib.Path(conditional['calibration_json']).is_absolute()
    contract = fixed['formal_lane_isolated_conditional_contract']
    assert contract['selection_split'] == 'val'
    assert contract['independent_lane_state'] is True
    assert contract['cross_lane_geometry_rejection'] is False
    assert contract['target_scope'] is False
    assert contract['test_parameter_search'] is False


def test_conditional_v3_skips_entire_dino_runtime_when_not_triggered():
    from crane_project.utils.lane_isolated_conditional_dino import (
        LaneIsolatedConditionalDinoSelector)

    class _Baseline:
        @staticmethod
        def simple_test(img, img_metas, rescale=False):
            del img, img_metas, rescale
            return [[np.asarray(
                [[10, 10, 40, 20, 0.0, 0.9]], dtype=np.float32)]]

    detector = Detector.__new__(Detector)
    nn.Module.__init__(detector)
    detector.baseline = _Baseline()
    detector._fusion_policy = 'sym_eood_proposal_dino_roi_union'
    detector._scope_policy = 'all_frames'
    detector._scope_intervals = None
    detector._conditional_dino_selector = (
        LaneIsolatedConditionalDinoSelector(
            small_diag_ratio=0.0,
            max_sym_diag_change=0.2,
            max_sym_angle_change_deg=15.0,
            max_dino_diag_change=0.2,
            max_dino_angle_change_deg=15.0))
    detector._fusion_audit_enabled = True
    detector._fusion_audit_records = []
    detector._stabilizer_enabled = False
    detector._test_score_thr = 0.05
    result = detector.simple_test(
        torch.zeros((1, 3, 10, 10)),
        [dict(
            filename='/tmp/real_seq01_00001.jpg',
            ori_shape=(100, 100, 3))],
        rescale=True)
    assert result[0][0][0, 5] == pytest.approx(0.9)
    assert detector._fusion_audit_records[0]['dino_invoked'] is False
    assert '_dino_runtime' not in detector.__dict__


def test_symeood_proposal_fusion_rejects_more_than_top1():
    baseline = np.zeros((2, 6), np.float32)
    with pytest.raises(RuntimeError, match='top-1 only'):
        MODULE.ScopedDinoLowlightDetector._sym_eood_proposals_for_dino(
            baseline,
            dict(scale_factor=np.ones(4, np.float32)),
            torch.device('cpu'))


def test_symeood_proposal_fusion_rejects_invalid_geometry():
    baseline = np.asarray([[10, 20, 0, 40, 0.25, 0.8]], np.float32)
    with pytest.raises(RuntimeError, match='proposal is invalid'):
        MODULE.ScopedDinoLowlightDetector._sym_eood_proposals_for_dino(
            baseline,
            dict(scale_factor=np.ones(4, np.float32)),
            torch.device('cpu'))


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
