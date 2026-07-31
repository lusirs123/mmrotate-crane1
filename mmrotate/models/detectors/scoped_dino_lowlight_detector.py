"""Config-driven BrightAug + frozen-DINO low-light detector.

This detector is the deployable form of the previously audited composition.
The BrightAug detector remains the registered child module and therefore owns
the positional checkpoint passed to ``tools/test.py``.  DINOv2 and its source
trained rotated heads are deliberately kept outside PyTorch's module tree:
the DINO transformer is sharded over dedicated GPUs and must not be moved or
replicated by MMDataParallel.
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


@ROTATED_DETECTORS.register_module(force=True)
class ScopedDinoLowlightDetector(RotatedBaseDetector):
    """BrightAug detector with a frozen, scope-gated DINOv2 rescue branch."""

    def __init__(self,
                 baseline_config,
                 dino_rescue,
                 dino_head_checkpoint,
                 scope_manifest,
                 scope_split='test',
                 stabilizer=None,
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
        self._scope_intervals = _load_scope(scope_manifest, self._scope_split)
        stabilizer = dict(stabilizer or {})
        self._alpha = float(stabilizer.get('alpha', 0.25))
        if not 0.0 < self._alpha <= 1.0:
            raise ValueError('stabilizer.alpha must be in (0, 1]')
        self._stabilizer_enabled = bool(stabilizer.get('enabled', True))
        self._previous_box = None
        self._previous_seq = None
        self._previous_frame = None

        # Build the exact BrightAug detector from its original config.  It is
        # the only child module so the positional BrightAug checkpoint loads
        # through the normal MMDetection checkpoint mechanism.
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
            raise RuntimeError('BrightAug config has no model.test_cfg')
        if int(self._baseline_test_cfg.get('max_per_img', 0)) != 1:
            raise RuntimeError('Scoped DINO requires BrightAug max_per_img=1')

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
        labeller.validate_checkpoint(checkpoint, int(getattr(dino, 'embed_dim')), args)
        labeller.load_heads_checkpoint_state(heads, checkpoint)
        dino.eval()
        heads.eval()
        for parameter in dino.parameters():
            parameter.requires_grad_(False)
        for parameter in heads.parameters():
            parameter.requires_grad_(False)
        # Keep both modules out of nn.Module._modules.  MMDataParallel must
        # never move the sharded transformer or replicate it to GPU 0.
        self.__dict__['_dino_runtime'] = dict(
            common=common, labeller=labeller, dino=dino, heads=heads,
            dino_device=dino_devices[0], head_device=head_device,
            height=int(dinov2.get('height', common.CANONICAL_DINO_HEIGHT)),
            max_long_side=int(dinov2.get(
                'max_long_side', common.CANONICAL_DINO_MAX_LONG_SIDE)),
            patch_size=args.patch_size)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys,
                              error_msgs):
        # ``tools/test.py`` supplies the original BrightAug checkpoint.  Map
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

    def _stabilize(self, detections, seq, frame):
        current = np.asarray(detections, dtype=np.float32).copy()
        if not self._stabilizer_enabled or current.shape[0] == 0:
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
        baseline_result = self.baseline.simple_test(img, img_metas, rescale=rescale)
        baseline = _as_array(baseline_result)
        meta = img_metas[0]
        image_path = _filename(meta)
        seq, frame = _sequence_frame(image_path)
        enabled = _in_scope(self._scope_intervals, seq, frame)
        if not enabled:
            self._reset_temporal()
            return [[baseline]]
        # BrightAug and the rotated DINO heads share the 8 GB head GPU.  The
        # calls are sequential; releasing cached BrightAug activations before
        # the DINO head keeps the integrated path within the audited budget.
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
        feature = runtime['common'].extract_patch_grid(
            runtime['dino'], tensor, runtime['patch_size'])
        del tensor
        head_device = runtime['head_device']
        feature = feature.to(head_device, dtype=torch.float32)
        feature_meta = runtime['labeller'].feature_meta(image_path, dino_meta)
        with torch.no_grad():
            dino_detections = runtime['heads'].simple_test(feature, feature_meta)
        dino_detections, _stats = runtime['labeller'].filter_valid_rotated_detections(
            dino_detections, feature_meta)
        selected = (dino_detections[:1] if dino_detections.shape[0] > 0
                    else baseline)
        selected = self._stabilize(selected, seq, frame)
        return [[selected]]

    def aug_test(self, imgs, img_metas, rescale=False):
        del imgs, img_metas, rescale
        raise RuntimeError(
            'Scoped DINO sequential inference does not support test-time '
            'augmentation; use the configured single-view test pipeline')
