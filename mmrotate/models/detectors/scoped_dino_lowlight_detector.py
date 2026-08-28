"""Config-driven MMRotate detector plus frozen-DINO inference branch.

The registered MMRotate child owns the positional checkpoint passed to
``tools/test.py``.  DINOv2 and its source-trained rotated heads stay outside
PyTorch's module tree: the transformer is sharded over dedicated GPUs and must
not be moved or replicated by MMDataParallel.  Historical configs use scoped
replacement; the formal unified config instead adds the SymEOOD top-1 as a
proposal and ranks the complete union with one frozen DINO ROI head.
"""

import copy
import json
import math
import os
import re
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from ..builder import ROTATED_DETECTORS
from .base import RotatedBaseDetector


def _as_array(result):
    if not isinstance(result, (list, tuple)) or len(result) != 1:
        raise RuntimeError('Expected one-image detector result')
    classes = result[0]
    if not isinstance(classes, (list, tuple)) or len(classes) != 1:
        raise RuntimeError('Expected one-class detector result')
    array = np.asarray(classes[0], dtype=np.float32)
    if array.size == 0:
        return np.zeros((0, 6), dtype=np.float32)
    array = array.reshape((-1, 6))
    if not np.isfinite(array).all():
        raise RuntimeError('Detector produced non-finite boxes')
    return array.copy()


def _filename(meta):
    value = meta.get('filename')
    if isinstance(value, (list, tuple)):
        value = value[0]
    if not value:
        raise RuntimeError('Image metadata has no filename')
    return os.path.abspath(os.fspath(value))


def _sequence_frame(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    match = re.match(r'^((?:real|sim)_.+)_(\d+)$', stem)
    if match is None:
        raise RuntimeError('Cannot parse sequence/frame from {}'.format(path))
    return match.group(1), int(match.group(2))


def _original_image_shape(meta):
    value = meta.get('ori_shape')
    if value is None:
        value = meta.get('img_shape')
    if value is None or len(value) < 2:
        raise RuntimeError('Image metadata has no valid original shape')
    height, width = int(value[0]), int(value[1])
    if height <= 0 or width <= 0:
        raise RuntimeError('Image metadata has invalid original shape')
    return height, width


def _load_scope(path, split):
    with open(path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    entries = payload.get('entries')
    if not isinstance(entries, list) or not entries:
        raise ValueError('Scope manifest requires non-empty entries')
    intervals = {}
    for entry in entries:
        required = ('split', 'seq', 'start', 'end', 'dino_enabled')
        if any(key not in entry for key in required):
            raise ValueError('Scope manifest entry is incomplete')
        if str(entry['split']) != str(split):
            continue
        seq = str(entry['seq'])
        start, end = int(entry['start']), int(entry['end'])
        if end < start:
            raise ValueError('Scope manifest interval is reversed')
        bucket = intervals.setdefault(seq, [])
        for old_start, old_end, _old_enabled in bucket:
            if not (end < old_start or start > old_end):
                raise ValueError('Scope manifest has overlapping intervals')
        bucket.append((start, end, bool(entry['dino_enabled'])))
    return {seq: sorted(values) for seq, values in intervals.items()}


def _in_scope(intervals, seq, frame):
    for start, end, enabled in intervals.get(seq, ()):
        if start <= frame <= end:
            return bool(enabled)
    return False


def _validate_dino_checkpoint_contract(payload, contract):
    """Validate config-declared provenance for a formal DINO checkpoint.

    Historical scoped configs intentionally omit this contract.  A formal
    all-frame config uses it to prevent an old low-light, S7, or rejected
    experimental checkpoint from being loaded through the same runtime.
    """
    contract = dict(contract or {})
    if not contract:
        return
    interpolation = payload.get('source_only_fc_cls_interpolation')
    if not isinstance(interpolation, dict):
        raise RuntimeError(
            'Formal DINO checkpoint lacks source-only fc-cls interpolation '
            'provenance')
    expected_selector = contract.get('selector')
    if (expected_selector is not None
            and interpolation.get('selector') != expected_selector):
        raise RuntimeError('Formal DINO checkpoint selector mismatch')
    expected_protocol = contract.get('protocol_version')
    if (expected_protocol is not None
            and int(interpolation.get('protocol_version', -1))
            != int(expected_protocol)):
        raise RuntimeError('Formal DINO checkpoint protocol mismatch')
    expected_alpha = contract.get('alpha')
    if (expected_alpha is not None
            and not math.isclose(
                float(interpolation.get('alpha', float('nan'))),
                float(expected_alpha), rel_tol=0.0, abs_tol=1e-12)):
        raise RuntimeError('Formal DINO checkpoint interpolation mismatch')
    if (bool(contract.get('require_target_unread', True))
            and interpolation.get('target_data_read') is not False):
        raise RuntimeError('Formal DINO checkpoint provenance read target data')
    if (bool(contract.get('require_source_gate', True))
            and (interpolation.get('source_gate') or {}).get('passed')
            is not True):
        raise RuntimeError('Formal DINO checkpoint source gate did not pass')
    stored_s7 = payload.get('s7_architecture') or {'enabled': False}
    if (bool(contract.get('require_s7_disabled', True))
            and bool(stored_s7.get('enabled', False))):
        raise RuntimeError('Formal native-S14 checkpoint unexpectedly enables S7')

    for name, payload_key in (
            ('source_full', 'best_source_val_summary'),
            ('source_small', 'best_source_small_val_summary')):
        expected = contract.get(name)
        if expected is None:
            continue
        observed = payload.get(payload_key) or {}
        for metric in ('top1_hits', 'top1_mcml'):
            if metric in expected and int(observed.get(metric, -1)) != int(
                    expected[metric]):
                raise RuntimeError(
                    'Formal DINO checkpoint {} {} mismatch'.format(
                        name, metric))


@ROTATED_DETECTORS.register_module(force=True)
class ScopedDinoLowlightDetector(RotatedBaseDetector):
    """MMRotate baseline detector with a frozen DINOv2 inference branch."""

    def __init__(self,
                 baseline_config,
                 dino_rescue,
                 dino_head_checkpoint,
                 scope_manifest=None,
                 scope_split='test',
                 scope_policy='manifest',
                 fusion_policy='dino_primary',
                 stabilizer=None,
                 temporal_association=None,
                 dino_checkpoint_contract=None,
                 fusion_audit_enabled=False,
                 conservative_takeover=None,
                 conditional_dino=None,
                 pretrained=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            raise RuntimeError(
                'ScopedDinoLowlightDetector requires non-distributed inference '
                'because its DINOv2 backbone is explicitly sharded')
        super().__init__(init_cfg=init_cfg)
        self.fp16_enabled = False
        self._scope_split = str(scope_split)
        self._scope_policy = str(scope_policy)
        if self._scope_policy == 'all_frames':
            if scope_manifest is not None:
                raise ValueError(
                    'all_frames scope policy must not receive a manifest')
            self._scope_intervals = None
        elif self._scope_policy == 'manifest':
            if scope_manifest is None:
                raise ValueError('manifest scope policy requires a manifest')
            self._scope_intervals = _load_scope(
                scope_manifest, self._scope_split)
        else:
            raise ValueError('Unknown scope policy: {}'.format(
                self._scope_policy))
        self._fusion_policy = str(fusion_policy)
        if self._fusion_policy not in (
                'dino_primary', 'sym_eood_proposal_dino_roi_union'):
            raise ValueError('Unknown fusion policy: {}'.format(
                self._fusion_policy))
        self._fusion_audit_enabled = bool(fusion_audit_enabled)
        self._fusion_audit_records = []
        test_cfg = dict(test_cfg or {})
        self._test_score_thr = float(test_cfg.get('score_thr', 0.05))
        if not 0.0 <= self._test_score_thr < 1.0:
            raise ValueError('test_cfg.score_thr must be in [0, 1)')
        if (self._fusion_policy == 'sym_eood_proposal_dino_roi_union'
                and int(test_cfg.get('max_per_img', 1)) != 1):
            raise ValueError('Unified detector requires max_per_img=1')
        takeover_cfg = dict(conservative_takeover or {})
        self._conservative_takeover_enabled = bool(
            takeover_cfg.get('enabled', False))
        self._conservative_takeover_calibration = None
        self._conservative_selector = None
        if self._conservative_takeover_enabled:
            calibration_path = takeover_cfg.get('calibration_json')
            if not calibration_path:
                raise ValueError(
                    'Conservative takeover requires calibration_json')
            calibration_path = os.path.abspath(os.fspath(calibration_path))
            if not os.path.isfile(calibration_path):
                raise RuntimeError(
                    'Conservative takeover calibration does not exist: '
                    + calibration_path)
            with open(calibration_path, 'r') as handle:
                calibration = json.load(handle)
            if (calibration.get('protocol') !=
                    'source_calibrated_conservative_takeover_v2'):
                raise RuntimeError('Unexpected takeover calibration protocol')
            if int(calibration.get('metric_protocol_version', -1)) != 2:
                raise RuntimeError(
                    'Takeover calibration uses a stale metric protocol; '
                    're-run source-only calibration')
            if calibration.get('selection_split') != 'val':
                raise RuntimeError('Takeover calibration must use source val')
            if bool(calibration.get('target_data_read', True)):
                raise RuntimeError('Takeover calibration read target data')
            if not bool(calibration.get('eligible_for_test', False)):
                raise RuntimeError('Takeover calibration is not test-eligible')
            parameters = dict(calibration.get('selected_parameters') or {})
            from crane_project.utils.conservative_takeover import (
                ConservativeTakeoverSelector)
            self._conservative_selector = ConservativeTakeoverSelector(
                **parameters)
            self._conservative_takeover_calibration = dict(
                path=calibration_path,
                parameters=parameters,
                source_gate=calibration.get('source_gate'))
        conditional_cfg = dict(conditional_dino or {})
        self._conditional_dino_enabled = bool(
            conditional_cfg.get('enabled', False))
        self._conditional_dino_calibration = None
        self._conditional_dino_selector = None
        if (self._conditional_dino_enabled
                and self._conservative_takeover_enabled):
            raise ValueError(
                'Conditional DINO V3 and conservative takeover V2 are '
                'mutually exclusive')
        if (self._conditional_dino_enabled
                and self._fusion_policy
                != 'sym_eood_proposal_dino_roi_union'):
            raise ValueError(
                'Conditional DINO V3 requires unified SymEOOD-DINO fusion')
        if self._conditional_dino_enabled:
            calibration_path = conditional_cfg.get('calibration_json')
            if not calibration_path:
                raise ValueError(
                    'Conditional DINO V3 requires calibration_json')
            calibration_path = os.path.abspath(os.fspath(calibration_path))
            if not os.path.isfile(calibration_path):
                raise RuntimeError(
                    'Conditional DINO calibration does not exist: '
                    + calibration_path)
            with open(calibration_path, 'r') as handle:
                calibration = json.load(handle)
            if (calibration.get('protocol') !=
                    'source_calibrated_lane_isolated_conditional_dino_v3'):
                raise RuntimeError(
                    'Unexpected conditional DINO calibration protocol')
            if int(calibration.get('metric_protocol_version', -1)) != 2:
                raise RuntimeError(
                    'Conditional DINO calibration uses a stale metric '
                    'protocol')
            if calibration.get('selection_split') != 'val':
                raise RuntimeError(
                    'Conditional DINO calibration must use source val')
            if bool(calibration.get('target_data_read', True)):
                raise RuntimeError('Conditional DINO calibration read target')
            if not bool(calibration.get('eligible_for_fixed_test', False)):
                raise RuntimeError(
                    'Conditional DINO calibration is not fixed-test eligible')
            parameters = dict(calibration.get('selected_parameters') or {})
            from crane_project.utils.lane_isolated_conditional_dino import (
                LaneIsolatedConditionalDinoSelector)
            self._conditional_dino_selector = (
                LaneIsolatedConditionalDinoSelector(**parameters))
            self._conditional_dino_calibration = dict(
                path=calibration_path,
                parameters=parameters,
                source_gate=calibration.get('source_gate'),
                dino_invocation_rate=(calibration.get('selected_summary')
                                      or {}).get('dino_invocation_rate'))
        stabilizer = dict(stabilizer or {})
        self._alpha = float(stabilizer.get('alpha', 0.25))
        if not 0.0 < self._alpha <= 1.0:
            raise ValueError('stabilizer.alpha must be in (0, 1]')
        self._stabilizer_enabled = bool(stabilizer.get('enabled', True))
        self._previous_box = None
        self._previous_seq = None
        self._previous_frame = None

        if baseline_config is None:
            raise ValueError('Scoped DINO requires a baseline detector')
        # Keep the MMRotate detector as a registered child so its positional
        # checkpoint loads through the standard test entry.
        from mmcv import Config
        from mmcv.utils import import_modules_from_strings
        from mmrotate.models import build_detector

        baseline_path = os.path.abspath(os.fspath(baseline_config))
        baseline_cfg = Config.fromfile(baseline_path)
        imports = baseline_cfg.get('custom_imports')
        if imports:
            import_modules_from_strings(**imports)
        baseline_model_cfg = copy.deepcopy(baseline_cfg.model)
        baseline_model_cfg.pretrained = None
        self.baseline = build_detector(baseline_model_cfg)
        self.CLASSES = getattr(self.baseline, 'CLASSES', None)
        self._baseline_test_cfg = baseline_model_cfg.get('test_cfg')
        if self._baseline_test_cfg is None:
            raise RuntimeError('Baseline config has no model.test_cfg')
        if int(self._baseline_test_cfg.get('max_per_img', 0)) != 1:
            raise RuntimeError('Scoped DINO requires baseline max_per_img=1')

        # Lazy imports avoid a detector-registry cycle during MMRotate startup.
        from crane_project.tools import dino_teacher_common as common
        from crane_project.tools import dino_teacher_rotated_labeller as labeller

        dinov2 = dict(dino_rescue.get('dinov2', {}))
        head_cfg = dict(dino_rescue.get('head', {}))
        dino_gpus = [int(value) for value in dinov2.get('gpus', [])]
        head_gpu = int(head_cfg.get('gpu', 0))
        if not dino_gpus or head_gpu in dino_gpus:
            raise ValueError('DINO GPUs must be non-empty and separate from head GPU')
        if not os.path.isfile(dino_head_checkpoint):
            raise RuntimeError('DINO head checkpoint does not exist: {}'.format(
                dino_head_checkpoint))
        dino_devices = [torch.device('cuda:{}'.format(value))
                        for value in dino_gpus]
        head_device = torch.device('cuda:{}'.format(head_gpu))
        if not torch.cuda.is_available():
            raise RuntimeError('Frozen DINO inference requires CUDA')
        roi_nms_iou_thr = float(head_cfg.get('roi_nms_iou_thr', 0.1))
        if not 0.0 < roi_nms_iou_thr <= 1.0:
            raise ValueError('DINO ROI NMS IoU threshold must be in (0, 1]')
        feature_strides = head_cfg.get('feature_strides')
        if feature_strides is None:
            feature_strides = [int(dinov2.get('patch_size', 14))]
        feature_strides = sorted(set(int(value) for value in feature_strides))
        if (not feature_strides
                or int(dinov2.get('patch_size', 14)) not in feature_strides
                or any(value <= 0 for value in feature_strides)):
            raise ValueError(
                'DINO feature_strides must be positive and include patch size')
        args = SimpleNamespace(
            patch_size=int(dinov2.get('patch_size', 14)),
            rpn_feat_channels=int(head_cfg.get('rpn_feat_channels', 256)),
            roi_fc_channels=int(head_cfg.get('roi_fc_channels', 1024)),
            roi_samples=int(head_cfg.get('roi_samples', 256)),
            proposal_count=int(head_cfg.get('proposal_count', 2000)),
            max_detections=int(head_cfg.get('max_detections', 2000)),
            roi_nms_iou_thr=roi_nms_iou_thr,
            feature_strides=feature_strides,
            s7_residual=bool(head_cfg.get('s7_residual', False)),
            s7_channels=int(head_cfg.get('s7_channels', 128)),
            s7_rpn_feat_channels=int(head_cfg.get(
                's7_rpn_feat_channels', 128)),
            s7_proposal_count=int(head_cfg.get('s7_proposal_count', 500)),
            s7_nms_pre=int(head_cfg.get('s7_nms_pre', 2000)),
            s7_protected_merge=bool(head_cfg.get(
                's7_protected_merge', False)),
            s7_merge_init_bias=float(head_cfg.get(
                's7_merge_init_bias', -2.0)),
            s7_lane_arbitration=bool(head_cfg.get(
                's7_lane_arbitration', False)),
            s7_lane_hidden=int(head_cfg.get('s7_lane_hidden', 32)),
            s7_lane_max_adjustment=float(head_cfg.get(
                's7_lane_max_adjustment', 2.0)),
            s7_quality_suppression=bool(head_cfg.get(
                's7_quality_suppression', False)),
            s7_quality_hidden=int(head_cfg.get('s7_quality_hidden', 32)),
            s7_quality_max_suppression=float(head_cfg.get(
                's7_quality_max_suppression', 2.0)),
            s7_quality_init_risk_bias=float(head_cfg.get(
                's7_quality_init_risk_bias', 0.0)),
            s7_temporal_association=bool(head_cfg.get(
                's7_temporal_association', False)),
            s7_temporal_quality_head=bool(head_cfg.get(
                's7_temporal_quality_head', False)),
            s7_temporal_quality_hidden=int(head_cfg.get(
                's7_temporal_quality_hidden', 128)),
            s7_temporal_max_candidates=int(head_cfg.get(
                's7_temporal_max_candidates', 100)),
            s7_temporal_min_confirmations=int(head_cfg.get(
                's7_temporal_min_confirmations', 2)),
            s7_temporal_override_margin=float(head_cfg.get(
                's7_temporal_override_margin', 0.25)),
            s7_temporal_max_center_distance=float(head_cfg.get(
                's7_temporal_max_center_distance', 3.0)),
            s7_temporal_min_riou=float(head_cfg.get(
                's7_temporal_min_riou', 0.05)),
            s7_temporal_min_appearance=float(head_cfg.get(
                's7_temporal_min_appearance', 0.20)),
            s7_anchor_sizes=[float(value) for value in head_cfg.get(
                's7_anchor_sizes', [16, 32, 64, 128, 256])])
        dino, loaded_patch_size = common.load_frozen_dinov2(
            dinov2['repo'], dinov2['checkpoint'],
            dinov2.get('model', common.CANONICAL_MODEL), dino_devices,
            int(dinov2.get('legacy_sdpa_query_chunk', 512)))
        if int(loaded_patch_size) != args.patch_size:
            raise RuntimeError('DINO patch-size/checkpoint configuration mismatch')
        heads = labeller.FrozenDinoRotatedHeads(
            int(getattr(dino, 'embed_dim')), args).to(head_device)
        checkpoint = torch.load(dino_head_checkpoint, map_location='cpu')
        _validate_dino_checkpoint_contract(
            checkpoint, dino_checkpoint_contract)
        labeller.validate_checkpoint(checkpoint, int(getattr(dino, 'embed_dim')), args)
        labeller.load_heads_checkpoint_state(heads, checkpoint)
        dino.eval()
        heads.eval()
        for parameter in dino.parameters():
            parameter.requires_grad_(False)
        for parameter in heads.parameters():
            parameter.requires_grad_(False)
        temporal_cfg = dict(temporal_association or {})
        temporal_enabled = bool(temporal_cfg.get(
            'enabled', args.s7_temporal_association))
        if (self._fusion_policy == 'sym_eood_proposal_dino_roi_union'
                and temporal_enabled):
            raise ValueError(
                'SymEOOD proposal fusion cannot use an experimental '
                'temporal selector')
        source_selected = bool(temporal_cfg.get('source_selected', False))
        if source_selected and not temporal_enabled:
            raise ValueError(
                'source_selected temporal runtime requires temporal '
                'association to be enabled')
        if source_selected:
            source_gate = dict(temporal_cfg.get('source_gate', {}))
            gate = labeller.source_selected_checkpoint_gate(
                checkpoint,
                min_full_top1=int(source_gate.get('min_full_top1', 688)),
                min_small_top1=int(source_gate.get('min_small_top1', 311)),
                max_mcml=int(source_gate.get('max_mcml', 3)))
            if not gate['passed']:
                failed = sorted(
                    name for name, passed in gate['checks'].items()
                    if not passed)
                raise RuntimeError(
                    'Source-selected temporal checkpoint failed the '
                    'deployment gate; keep native S14 fallback. failed={}'
                    .format(','.join(failed)))
        if temporal_enabled != bool(args.s7_temporal_association):
            raise ValueError(
                'Temporal runtime and DINO head configuration disagree')
        temporal_selector = (
            labeller.temporal.CausalTemporalCandidateSelector(
                heads.s7_temporal_scorer,
                max_candidates=args.s7_temporal_max_candidates,
                min_confirmations=args.s7_temporal_min_confirmations,
                override_margin=args.s7_temporal_override_margin,
                max_center_distance=args.s7_temporal_max_center_distance,
                min_rotated_iou=args.s7_temporal_min_riou,
                min_appearance_similarity=args.s7_temporal_min_appearance)
            if temporal_enabled else None)
        # Keep both modules out of nn.Module._modules.  MMDataParallel must
        # never move the sharded transformer or replicate it to GPU 0.
        self.__dict__['_dino_runtime'] = dict(
            common=common, labeller=labeller, dino=dino, heads=heads,
            dino_device=dino_devices[0], head_device=head_device,
            height=int(dinov2.get('height', common.CANONICAL_DINO_HEIGHT)),
            max_long_side=int(dinov2.get(
                'max_long_side', common.CANONICAL_DINO_MAX_LONG_SIDE)),
            patch_size=args.patch_size,
            temporal_selector=temporal_selector)

    @staticmethod
    def _sym_eood_proposals_for_dino(baseline, feature_meta, device):
        """Map the source-safe SymEOOD top-1 OBB into DINO image space.

        The sixth detector-score column is deliberately replaced by one: the
        Oriented ROI head discards proposal scores and ranks every proposal
        with its own shared classifier.  This prevents incomparable SymEOOD
        and DINO scores from being mixed during fusion.
        """
        boxes = np.asarray(baseline, dtype=np.float32).reshape((-1, 6))
        if boxes.shape[0] == 0:
            return torch.zeros((0, 6), dtype=torch.float32, device=device)
        if boxes.shape[0] != 1:
            raise RuntimeError('Unified SymEOOD input must contain top-1 only')
        if (not np.isfinite(boxes).all()
                or np.any(boxes[:, 2:4] <= 0.0)):
            raise RuntimeError('Unified SymEOOD proposal is invalid')
        scale_factor = np.asarray(
            feature_meta['scale_factor'], dtype=np.float32).reshape(-1)
        if scale_factor.size < 4 or not np.isfinite(scale_factor[:4]).all():
            raise RuntimeError('Invalid DINO scale factor for proposal fusion')
        proposal = boxes[:, :5].copy()
        proposal[:, :4] *= scale_factor[:4]
        proposal = np.concatenate(
            [proposal, np.ones((1, 1), dtype=np.float32)], axis=1)
        return torch.as_tensor(proposal, dtype=torch.float32, device=device)

    def _dino_test_with_sym_eood_proposal(self, feature, feature_meta,
                                          baseline, sequence=None,
                                          frame=None):
        """Rank both proposal lanes while preserving source-owned geometry.

        The frozen DINO ROI classifier supplies the comparable foreground
        score for every proposal.  Native proposals use DINO-regressed OBBs;
        the external SymEOOD proposal keeps its original OBB because replacing
        that geometry was found to damage angle and temporal metrics.
        """
        runtime = self.__dict__['_dino_runtime']
        heads = runtime['heads']
        _features, proposal_list = heads.simple_test_proposals(
            feature, feature_meta)
        native = proposal_list[0]
        external = self._sym_eood_proposals_for_dino(
            baseline, feature_meta, native.device)
        union = torch.cat([native, external], dim=0)
        decoded, _log_odds, foreground_scores, _embedding = (
            heads._decode_roi_candidates(
                feature, feature_meta, union, rescale=True))
        if (decoded.shape[0] != union.shape[0]
                or foreground_scores.shape[0] != union.shape[0]):
            raise RuntimeError('Unified ROI candidate order was not preserved')

        native_count = int(native.shape[0])
        native_detections = torch.cat(
            [decoded[:native_count, :5],
             foreground_scores[:native_count, None]], dim=1)
        if external.shape[0] > 0:
            baseline_geometry = torch.as_tensor(
                np.asarray(baseline, dtype=np.float32)[:, :5],
                dtype=decoded.dtype, device=decoded.device)
            sym_detection = torch.cat(
                [baseline_geometry,
                 foreground_scores[native_count:, None]], dim=1)
        else:
            sym_detection = decoded.new_zeros((0, 6))
        detections = torch.cat(
            [native_detections, sym_detection], dim=0)
        source_ids = np.concatenate([
            np.zeros(native_detections.shape[0], dtype=np.int64),
            np.ones(sym_detection.shape[0], dtype=np.int64)])
        detections = detections.detach().cpu().numpy().astype(
            np.float32, copy=False)
        valid = runtime['labeller'].valid_rotated_detection_mask(
            detections, feature_meta)
        eligible = valid & (detections[:, 5] >= self._test_score_thr)
        eligible_indices = np.flatnonzero(eligible)
        if eligible_indices.size:
            local_order = np.argsort(
                -detections[eligible_indices, 5], kind='stable')
            order = eligible_indices[local_order]
            ranked = detections[order]
            ranked_sources = source_ids[order]
        else:
            ranked = detections[:0]
            ranked_sources = source_ids[:0]
        raw_selected_source = (
            'dino_native' if ranked_sources.size and ranked_sources[0] == 0
            else 'sym_eood' if ranked_sources.size else 'sym_eood_fallback')

        native_eligible = np.flatnonzero(
            eligible & (source_ids == 0))
        sym_eligible = np.flatnonzero(
            eligible & (source_ids == 1))
        native_top = (None if not native_eligible.size else detections[
            native_eligible[np.argmax(detections[native_eligible, 5])]])
        sym_top = (None if not sym_eligible.size else detections[
            sym_eligible[np.argmax(detections[sym_eligible, 5])]])
        takeover = None
        if self._conservative_selector is not None:
            if sequence is None or frame is None:
                raise RuntimeError(
                    'Conservative takeover requires sequence and frame')
            takeover = self._conservative_selector.select(
                sym_top, native_top, sequence, frame)
            selected = takeover['selected']
            ranked = (detections[:0] if selected is None else
                      np.asarray(selected, dtype=np.float32).reshape(1, 6))
            selected_source = takeover['selected_source']
        else:
            selected_source = raw_selected_source
        audit = dict(
            native_proposal_count=int(native.shape[0]),
            sym_eood_proposal_count=int(external.shape[0]),
            union_proposal_count=int(union.shape[0]),
            valid_candidate_count=int(valid.sum()),
            eligible_candidate_count=int(eligible.sum()),
            selected_source=selected_source,
            raw_selected_source=raw_selected_source,
            selected_common_score=(
                float(ranked[0, 5]) if ranked.shape[0] else None),
            sym_eood_common_score=(
                None if sym_top is None else float(sym_top[5])),
            dino_native_common_score=(
                None if native_top is None else float(native_top[5])),
            score_delta=(None if sym_top is None or native_top is None else
                         float(native_top[5] - sym_top[5])),
            sym_eood_box=(None if sym_top is None else
                          [float(value) for value in sym_top]),
            sym_eood_original_box=(
                None if np.asarray(baseline).size == 0 else
                [float(value) for value in np.asarray(
                    baseline, dtype=np.float32).reshape(-1, 6)[0, :6]]),
            sym_eood_original_score=(
                None if np.asarray(baseline).size == 0 else
                float(np.asarray(
                    baseline, dtype=np.float32).reshape(-1, 6)[0, 5])),
            dino_native_box=(None if native_top is None else
                             [float(value) for value in native_top]),
            score_threshold=float(self._test_score_thr),
            sym_eood_geometry_preserved=True,
            conservative_takeover_enabled=bool(
                self._conservative_selector is not None))
        if takeover is not None:
            audit.update(dict(
                previous_source=takeover['previous_source'],
                source_switched=takeover['source_switched'],
                takeover_reason=takeover['takeover_reason'],
                geometry_allowed=takeover['geometry_allowed'],
                diag_change=float(takeover['diag_change']),
                angle_change_deg=float(takeover['angle_change_deg'])))
        self.__dict__['_last_unified_fusion'] = audit
        return ranked, audit

    def fusion_audit_records(self):
        """Return a copy of per-frame source-attribution records."""
        return [dict(record) for record in self._fusion_audit_records]

    @staticmethod
    def _module_parameter_counts(module):
        total = trainable = 0
        if module is not None and hasattr(module, 'parameters'):
            for parameter in module.parameters():
                count = int(parameter.numel())
                total += count
                if parameter.requires_grad:
                    trainable += count
        return dict(total=int(total), trainable=int(trainable))

    def _runtime_resource_summary(self):
        runtime = self.__dict__.get('_dino_runtime', {})
        modules = dict(
            sym_eood=getattr(self, 'baseline', None),
            dinov2=runtime.get('dino'),
            dino_heads=runtime.get('heads'),
            geometry_refiner=getattr(self, 'geometry_refiner', None))
        parameter_counts = {
            name: self._module_parameter_counts(module)
            for name, module in modules.items()
        }
        parameter_counts['combined_runtime'] = dict(
            total=int(sum(row['total'] for row in parameter_counts.values())),
            trainable=int(sum(
                row['trainable'] for row in parameter_counts.values())))

        cuda_memory_mib = []
        if torch.cuda.is_available():
            mib = float(1024 ** 2)
            for index in range(torch.cuda.device_count()):
                cuda_memory_mib.append(dict(
                    logical_gpu=int(index),
                    device_name=torch.cuda.get_device_name(index),
                    allocated=round(
                        float(torch.cuda.memory_allocated(index)) / mib, 2),
                    reserved=round(
                        float(torch.cuda.memory_reserved(index)) / mib, 2),
                    peak_allocated=round(float(
                        torch.cuda.max_memory_allocated(index)) / mib, 2),
                    peak_reserved=round(float(
                        torch.cuda.max_memory_reserved(index)) / mib, 2)))
        return dict(
            parameter_counts=parameter_counts,
            cuda_memory_mib=cuda_memory_mib)

    def fusion_audit_metadata(self):
        """Return immutable protocol provenance for the saved audit."""
        return dict(
            fusion_policy=self._fusion_policy,
            score_threshold=float(self._test_score_thr),
            conservative_takeover_enabled=bool(
                self._conservative_selector is not None),
            conservative_takeover_calibration=(
                None if self._conservative_takeover_calibration is None else
                dict(self._conservative_takeover_calibration)),
            conditional_dino_enabled=bool(
                getattr(self, '_conditional_dino_selector', None) is not None),
            conditional_dino_calibration=(
                None if getattr(
                    self, '_conditional_dino_calibration', None) is None else
                dict(self._conditional_dino_calibration)),
            geometry_refiner_enabled=(
                getattr(self, 'geometry_refiner', None) is not None),
            geometry_refiner_checkpoint_contract=(
                None if getattr(
                    self, '_geometry_refiner_checkpoint_contract', None)
                is None else
                dict(self._geometry_refiner_checkpoint_contract)),
            runtime_forward_counts=dict(getattr(
                self, '_runtime_forward_counts', {})),
            resource_summary=self._runtime_resource_summary())

    def fusion_audit_protocol(self):
        if getattr(self, '_conditional_dino_selector', None) is not None:
            return 'lane_isolated_conditional_dino_v3'
        return 'source_owned_geometry_union_v2'

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys,
                              error_msgs):
        # ``tools/test.py`` supplies the original baseline checkpoint.  Map
        # its unprefixed keys into the registered child before MMDet recurses.
        if prefix == '':
            direct = [key for key in list(state_dict)
                      if not key.startswith('baseline.')]
            for key in direct:
                if key.startswith('module.'):
                    new_key = 'baseline.' + key[len('module.'):]
                else:
                    new_key = 'baseline.' + key
                state_dict[new_key] = state_dict.pop(key)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs)

    def load_state_dict(self, state_dict, strict=True):
        state_dict = state_dict.copy()
        if not any(key.startswith('baseline.') for key in state_dict):
            state_dict = {'baseline.' + key: value
                          for key, value in state_dict.items()}
        return super().load_state_dict(state_dict, strict=strict)

    def extract_feat(self, img):
        return self.baseline.extract_feat(img)

    def forward_train(self, *args, **kwargs):
        raise RuntimeError('ScopedDinoLowlightDetector is inference-only')

    def _reset_temporal(self):
        self._previous_box = None
        self._previous_seq = None
        self._previous_frame = None
        runtime = self.__dict__.get('_dino_runtime')
        if runtime is not None and runtime.get('temporal_selector') is not None:
            runtime['temporal_selector'].reset()
        if self._conservative_selector is not None:
            self._conservative_selector.reset()
        if getattr(self, '_conditional_dino_selector', None) is not None:
            self._conditional_dino_selector.reset()

    def _stabilize(self, detections, seq, frame):
        current = np.asarray(detections, dtype=np.float32).copy()
        if not self._stabilizer_enabled:
            return current
        if current.shape[0] == 0:
            self._reset_temporal()
            return current
        if (self._previous_seq != seq
                or self._previous_frame is None
                or int(frame) != int(self._previous_frame) + 1):
            self._previous_box = None
        if self._previous_box is not None and self._alpha < 1.0:
            previous = self._previous_box
            now = current[0, :5].copy()
            # Preserve the exact offline probe geometry: centers are current
            # values; only log-size and pi-periodic angle are smoothed.
            swapped = now.copy()
            swapped[2:4] = now[[3, 2]]
            swapped[4] = now[4] + math.pi / 2.0

            def cost(box):
                size = np.log(np.maximum(box[2:4], 1e-6)) - np.log(
                    np.maximum(previous[2:4], 1e-6))
                delta = 0.5 * math.atan2(
                    math.sin(2.0 * float(box[4] - previous[4])),
                    math.cos(2.0 * float(box[4] - previous[4])))
                return float(np.dot(size, size) + delta * delta)
            if cost(swapped) < cost(now):
                now = swapped
            prev_size = np.log(np.maximum(previous[2:4], 1e-6))
            cur_size = np.log(np.maximum(now[2:4], 1e-6))
            current[0, 2:4] = np.exp(
                (1.0 - self._alpha) * prev_size + self._alpha * cur_size)
            prev_vec = np.asarray([
                math.cos(2.0 * float(previous[4])),
                math.sin(2.0 * float(previous[4]))])
            cur_vec = np.asarray([
                math.cos(2.0 * float(now[4])),
                math.sin(2.0 * float(now[4]))])
            vector = (1.0 - self._alpha) * prev_vec + self._alpha * cur_vec
            current[0, 4] = 0.5 * math.atan2(
                float(vector[1]), float(vector[0]))
        self._previous_box = current[0, :5].copy()
        self._previous_seq = seq
        self._previous_frame = int(frame)
        return current

    def simple_test(self, img, img_metas, rescale=False):
        if len(img_metas) != 1 or int(img.shape[0]) != 1:
            raise RuntimeError(
                'Scoped DINO sequential inference requires batch size 1')
        counts = getattr(self, '_runtime_forward_counts', None)
        supports_feature_reuse = (
            hasattr(self.baseline, 'extract_feat')
            and hasattr(self.baseline, 'simple_test_from_features'))
        if supports_feature_reuse:
            baseline_features = self.baseline.extract_feat(img)
            if counts is not None:
                counts['symeood_backbone_fpn'] += 1
            baseline_result = self.baseline.simple_test_from_features(
                baseline_features, img_metas, rescale=rescale)
        else:
            if getattr(self, 'geometry_refiner', None) is not None:
                raise RuntimeError(
                    'Geometry refiner requires reusable SymEOOD FPN features')
            baseline_features = None
            baseline_result = self.baseline.simple_test(
                img, img_metas, rescale=rescale)
        baseline = _as_array(baseline_result)
        if (self._fusion_policy == 'sym_eood_proposal_dino_roi_union'
                and not rescale):
            raise RuntimeError(
                'Unified SymEOOD-DINO inference requires rescale=True so '
                'both proposal sources use original-image coordinates')
        meta = img_metas[0]
        image_path = _filename(meta)
        seq, frame = _sequence_frame(image_path)
        image_shape = _original_image_shape(meta)
        enabled = (True if self._scope_policy == 'all_frames' else
                   _in_scope(self._scope_intervals, seq, frame))
        if not enabled:
            self._reset_temporal()
            return [[baseline]]
        conditional_selector = getattr(
            self, '_conditional_dino_selector', None)
        conditional_trigger = None
        if conditional_selector is not None:
            sym_top = None if baseline.shape[0] == 0 else baseline[0]
            conditional_trigger = conditional_selector.begin_frame(
                sym_top, image_shape, seq, frame)
            if not conditional_trigger['invoke_dino']:
                decision = conditional_selector.finish_frame(None)
                selected_box = decision['selected']
                selected = (baseline[:0] if selected_box is None else
                            np.asarray(
                                selected_box, dtype=np.float32).reshape(1, 6))
                if self._fusion_audit_enabled:
                    record = dict(
                        filename=image_path,
                        sequence=seq,
                        frame=int(frame),
                        image_height=int(image_shape[0]),
                        image_width=int(image_shape[1]),
                        dino_invoked=False,
                        native_proposal_count=0,
                        sym_eood_proposal_count=int(baseline.shape[0] > 0),
                        union_proposal_count=int(baseline.shape[0] > 0),
                        valid_candidate_count=int(baseline.shape[0] > 0),
                        eligible_candidate_count=int(baseline.shape[0] > 0),
                        selected_source=decision['selected_source'],
                        raw_selected_source='not_computed',
                        output_source=decision['selected_source'],
                        output_score=(None if selected.shape[0] == 0 else
                                      float(selected[0, 5])),
                        selected_common_score=None,
                        sym_eood_common_score=None,
                        dino_native_common_score=None,
                        score_delta=None,
                        sym_eood_box=(
                            None if baseline.shape[0] == 0 else
                            [float(value) for value in baseline[0]]),
                        sym_eood_original_box=(
                            None if baseline.shape[0] == 0 else
                            [float(value) for value in baseline[0]]),
                        sym_eood_original_score=(
                            None if baseline.shape[0] == 0 else
                            float(baseline[0, 5])),
                        dino_native_box=None,
                        score_threshold=float(self._test_score_thr),
                        sym_eood_geometry_preserved=True,
                        conservative_takeover_enabled=False,
                        conditional_dino_enabled=True,
                        conditional_trigger_reasons=list(
                            decision['trigger_reasons']),
                        sym_normalized_diag=float(
                            decision['sym_normalized_diag']),
                        sym_diag_change=float(decision['sym_diag_change']),
                        sym_angle_change_deg=float(
                            decision['sym_angle_change_deg']),
                        dino_geometry_stable=bool(
                            decision['dino_geometry_stable']),
                        dino_diag_change=float(
                            decision['dino_diag_change']),
                        dino_angle_change_deg=float(
                            decision['dino_angle_change_deg']),
                        measurement_valid=bool(
                            decision['measurement_valid']),
                        measurement_risk_reasons=list(
                            decision['risk_reasons']),
                        selection_reason=decision['selection_reason'])
                    self.__dict__['_last_unified_fusion'] = dict(record)
                    self._fusion_audit_records.append(record)
                selected = self._stabilize(selected, seq, frame)
                return [[selected]]
        # Without a geometry refiner, release SymEOOD features exactly as the
        # historical runtime did.  A refiner deliberately retains the frozen
        # FPN tuple until the native DINO proposal is available; its memory
        # impact must be measured rather than assumed away.
        if (getattr(self, 'geometry_refiner', None) is None
                and baseline_features is not None):
            del baseline_features
        if torch.cuda.is_available():
            with torch.cuda.device(img.device):
                torch.cuda.empty_cache()
        runtime = self.__dict__['_dino_runtime']
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError('Cannot read image for DINO: {}'.format(image_path))
        tensor, dino_meta = runtime['common'].resize_and_normalize_bgr(
            image, runtime['height'], runtime['patch_size'],
            runtime['max_long_side'])
        tensor = tensor.to(runtime['dino_device'])
        if counts is not None:
            counts['dino'] += 1
        feature = runtime['common'].extract_patch_grid(
            runtime['dino'], tensor, runtime['patch_size'])
        del tensor
        head_device = runtime['head_device']
        feature = feature.to(head_device, dtype=torch.float32)
        feature_meta = runtime['labeller'].feature_meta(image_path, dino_meta)
        with torch.no_grad():
            if self._fusion_policy == 'sym_eood_proposal_dino_roi_union':
                dino_detections, fusion_audit = (
                    self._dino_test_with_sym_eood_proposal(
                        feature, feature_meta, baseline, seq, frame))
            else:
                dino_detections = runtime['heads'].simple_test(
                    feature, feature_meta)
            selector = runtime.get('temporal_selector')
            if selector is not None:
                pool = runtime['heads']._last_temporal_pool
                if pool is None or pool['detections'].shape[0] != dino_detections.shape[0]:
                    raise RuntimeError(
                        'Temporal runtime candidate metadata is unavailable')
                valid = runtime['labeller'].valid_rotated_detection_mask(
                    dino_detections, feature_meta)
                selection = selector.select(
                    pool['detections'], pool['embeddings'], pool['source_ids'],
                    seq, frame,
                    valid_mask=torch.as_tensor(
                        valid, dtype=torch.bool,
                        device=pool['detections'].device),
                    quality_logits=pool.get('quality_logits'))
                order = selection['order'].detach().cpu().numpy()
                dino_detections = dino_detections[order]
        if self._fusion_policy == 'sym_eood_proposal_dino_roi_union':
            if conditional_selector is not None:
                decision = conditional_selector.finish_frame(
                    fusion_audit.get('dino_native_box'))
                selected_box = decision['selected']
                dino_detections = (
                    dino_detections[:0] if selected_box is None else
                    np.asarray(selected_box, dtype=np.float32).reshape(1, 6))
                fusion_audit.update(dict(
                    dino_invoked=True,
                    selected_source=decision['selected_source'],
                    conditional_dino_enabled=True,
                    conditional_trigger_reasons=list(
                        decision['trigger_reasons']),
                    sym_normalized_diag=float(
                        decision['sym_normalized_diag']),
                    sym_diag_change=float(decision['sym_diag_change']),
                    sym_angle_change_deg=float(
                        decision['sym_angle_change_deg']),
                    dino_geometry_stable=bool(
                        decision['dino_geometry_stable']),
                    dino_diag_change=float(decision['dino_diag_change']),
                    dino_angle_change_deg=float(
                        decision['dino_angle_change_deg']),
                    measurement_valid=bool(decision['measurement_valid']),
                    measurement_risk_reasons=list(decision['risk_reasons']),
                    selection_reason=decision['selection_reason']))
                fusion_audit['selected_common_score'] = (
                    None if decision['selected_source'] != 'dino_native'
                    or decision['selected'] is None else
                    float(decision['selected'][5]))
            else:
                fusion_audit.update(dict(
                    dino_invoked=True,
                    conditional_dino_enabled=False,
                    measurement_valid=True,
                    measurement_risk_reasons=[]))
            fusion_audit.update(dict(
                image_height=int(image_shape[0]),
                image_width=int(image_shape[1])))
            self.__dict__['_last_unified_fusion'] = dict(fusion_audit)
        dino_detections, _stats = runtime['labeller'].filter_valid_rotated_detections(
            dino_detections, feature_meta)
        geometry_refiner = getattr(self, 'geometry_refiner', None)
        if geometry_refiner is not None:
            native_box = fusion_audit.get('dino_native_box')
            if native_box is None:
                selected = baseline
                output_source = 'sym_eood_fallback'
            else:
                from ..roi_heads.dino_conditioned_geometry_refiner import (
                    map_model_obb_to_original,
                    map_original_obb_to_model)
                original = torch.as_tensor(
                    np.asarray(native_box, dtype=np.float32)[:5],
                    device=baseline_features[0].device).reshape(1, 5)
                model_box = map_original_obb_to_model(original, meta)
                with torch.no_grad():
                    deltas = geometry_refiner(
                        baseline_features, [model_box])
                    refined = geometry_refiner.decode_and_normalize(
                        [model_box], deltas, img_metas=[meta])[0]
                    refined = map_model_obb_to_original(refined, meta)
                score = float(native_box[5])
                selected = np.concatenate([
                    refined.detach().cpu().numpy().astype(np.float32),
                    np.asarray([[score]], dtype=np.float32)], axis=1)
                output_source = 'dino_geometry_refiner'
                if counts is not None:
                    counts['geometry_refiner'] += 1
                fusion_audit.update(dict(
                    geometry_refiner_enabled=True,
                    geometry_refiner_input_box=[
                        float(value) for value in native_box],
                    geometry_refiner_output_box=[
                        float(value) for value in selected[0]],
                    geometry_refiner_contract=(
                        geometry_refiner.component_contract())))
        else:
            selected = (dino_detections[:1]
                        if dino_detections.shape[0] > 0 else baseline)
            output_source = (fusion_audit['selected_source']
                             if dino_detections.shape[0] > 0 else
                             'sym_eood_fallback')
        if (self._fusion_policy == 'sym_eood_proposal_dino_roi_union'
                and self._fusion_audit_enabled):
            record = dict(fusion_audit)
            record.update(dict(
                filename=image_path, sequence=seq, frame=int(frame),
                output_source=output_source,
                output_score=float(selected[0, 5])
                if selected.shape[0] else None))
            self._fusion_audit_records.append(record)
        selected = self._stabilize(selected, seq, frame)
        return [[selected]]

    def aug_test(self, imgs, img_metas, rescale=False):
        del imgs, img_metas, rescale
        raise RuntimeError(
            'Scoped DINO sequential inference does not support test-time '
            'augmentation; use the configured single-view test pipeline')


@ROTATED_DETECTORS.register_module(force=True)
class SymEOODDinoUnifiedDetector(ScopedDinoLowlightDetector):
    """All-frame SymEOOD proposal generation plus shared DINO ROI ranking."""

    def __init__(self, baseline_config, dino_rescue, dino_head_checkpoint,
                 dino_checkpoint_contract=None,
                 geometry_refiner=None,
                 geometry_refiner_checkpoint=None,
                 geometry_refiner_checkpoint_contract=None,
                 **kwargs):
        if baseline_config is None:
            raise ValueError('SymEOODDinoUnifiedDetector requires SymEOOD')
        requested_policy = kwargs.pop(
            'fusion_policy', 'sym_eood_proposal_dino_roi_union')
        if requested_policy != 'sym_eood_proposal_dino_roi_union':
            raise ValueError(
                'Unified detector locks SymEOOD-DINO ROI proposal fusion')
        requested_scope = kwargs.pop('scope_policy', 'all_frames')
        if requested_scope != 'all_frames':
            raise ValueError('Unified detector requires all-frame inference')
        if kwargs.get('scope_manifest') is not None:
            raise ValueError('Unified detector cannot use a target scope')
        super().__init__(
            baseline_config=baseline_config,
            dino_rescue=dino_rescue,
            dino_head_checkpoint=dino_head_checkpoint,
            dino_checkpoint_contract=dino_checkpoint_contract,
            fusion_policy='sym_eood_proposal_dino_roi_union',
            scope_policy='all_frames',
            **kwargs)
        self.geometry_refiner = None
        self._geometry_refiner_checkpoint_contract = None
        requested = geometry_refiner is not None
        if requested != (geometry_refiner_checkpoint is not None):
            raise ValueError(
                'Geometry refiner config and checkpoint must be supplied '
                'together')
        if requested:
            if getattr(self, '_conditional_dino_selector', None) is not None:
                raise ValueError(
                    'Geometry refiner cannot use conditional DINO routing')
            if self._conservative_selector is not None:
                raise ValueError(
                    'Geometry refiner cannot use learned/rule takeover')
            if self._stabilizer_enabled:
                raise ValueError(
                    'Geometry refiner runtime cannot use temporal stabilizer')
            from ..builder import build_head
            from crane_project.utils.geometry_refiner_checkpoint import (
                load_source_gated_geometry_refiner_checkpoint)
            self.geometry_refiner = build_head(dict(geometry_refiner))
            expected_refiner_contract = dict(
                refine_center=bool(self.geometry_refiner.refine_center),
                refine_size=bool(self.geometry_refiner.refine_size),
                refine_angle=bool(self.geometry_refiner.refine_angle))
            expected_refiner_contract.update(dict(
                geometry_refiner_checkpoint_contract or {}))
            contract = load_source_gated_geometry_refiner_checkpoint(
                self.geometry_refiner,
                geometry_refiner_checkpoint,
                expected_contract=expected_refiner_contract)
            self._geometry_refiner_checkpoint_contract = contract
            self.geometry_refiner.eval()
            for parameter in self.geometry_refiner.parameters():
                parameter.requires_grad_(False)
        self._runtime_forward_counts = dict(
            symeood_backbone_fpn=0, dino=0, geometry_refiner=0)
