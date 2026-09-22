# Copyright (c) OpenMMLab. All rights reserved.
import hashlib
import glob
import json
import os
import re

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines import LoadImageFromFile

from ..builder import ROTATED_PIPELINES


_SOURCE_OWNED_AUDIT_PROTOCOL = 'source_owned_geometry_union_v2'
_FRAME_RE = re.compile(
    r'^(?P<sequence>(?:real|sim)_seq\d+)_(?P<frame>\d+)$')


@ROTATED_PIPELINES.register_module()
class LoadDinoFeatureFromCache:
    """Load a frozen DINO patch grid without running DINO during training.

    The cache format is the one written by
    ``dino_teacher_rotated_labeller.extract_or_load_feature``.  Exactly one
    cache file must match each image.  The grid is transformed according to
    the already-applied random flip and resized to a fixed shape so MMCV can
    collate a batch without retaining variable-resolution teacher tensors.
    """

    def __init__(self, cache_dir, output_key='teacher_features',
                 output_size=(64, 64), expected_channels=1024,
                 expected_model='dinov2_vitl14',
                 strict_image_identity=True):
        self.cache_dir = os.path.abspath(os.fspath(cache_dir))
        self.output_key = str(output_key)
        self.output_size = tuple(int(value) for value in output_size)
        self.expected_channels = int(expected_channels)
        self.expected_model = str(expected_model)
        self.strict_image_identity = bool(strict_image_identity)
        if (len(self.output_size) != 2
                or min(self.output_size) <= 0
                or self.expected_channels <= 0):
            raise ValueError('invalid DINO cache output shape/channels')

    @staticmethod
    def _load(path):
        try:
            return torch.load(path, map_location='cpu', weights_only=False)
        except TypeError:
            return torch.load(path, map_location='cpu')

    def __call__(self, results):
        filename = results.get('filename', results.get('ori_filename'))
        if not filename:
            raise RuntimeError('Image filename is unavailable to DINO cache')
        stem = os.path.splitext(os.path.basename(os.fspath(filename)))[0]
        if _FRAME_RE.match(stem) is None:
            raise RuntimeError(
                'Image filename does not encode sequence/frame: ' + stem)
        candidates = sorted(glob.glob(os.path.join(
            self.cache_dir, '*', stem + '_*.pth')))
        matching = []
        stat = os.stat(filename)
        for path in candidates:
            payload = self._load(path)
            signature = payload.get('signature') or {}
            identity = signature.get('image') or {}
            same_size = int(identity.get('size', -1)) == int(stat.st_size)
            same_name = (os.path.basename(str(identity.get('path', '')))
                         == os.path.basename(os.fspath(filename)))
            same_mtime = int(identity.get('mtime_ns', -1)) == int(
                getattr(stat, 'st_mtime_ns', int(stat.st_mtime * 1e9)))
            clean_feature = 'paired_view' not in signature
            expected_model = signature.get('dinov2_model') == self.expected_model
            if clean_feature and expected_model and same_size and same_name and (
                    same_mtime or not self.strict_image_identity):
                matching.append((path, payload))
        if len(matching) != 1:
            raise RuntimeError(
                'Expected one valid DINO cache for {}, found {}'.format(
                    filename, len(matching)))
        path, payload = matching[0]
        feature = payload.get('feature')
        if (not payload.get('frozen_dinov2')
                or not isinstance(feature, torch.Tensor)
                or feature.ndim != 4 or feature.size(0) != 1
                or feature.size(1) != self.expected_channels
                or not bool(torch.isfinite(feature.float()).all().item())):
            raise RuntimeError('Invalid frozen DINO feature cache: ' + path)
        feature = feature[0].float()
        direction = results.get('flip_direction') if results.get('flip') else None
        if direction in ('horizontal', 'diagonal'):
            feature = torch.flip(feature, dims=(-1,))
        if direction in ('vertical', 'diagonal'):
            feature = torch.flip(feature, dims=(-2,))
        if direction not in (None, 'horizontal', 'vertical', 'diagonal'):
            raise RuntimeError('Unsupported DINO feature flip: ' + str(direction))
        # Both pipelines preserve aspect ratio and pad on the bottom/right.
        # Pad the non-square DINO grid before producing a fixed square tensor;
        # directly resizing HxW to 64x64 would distort token coordinates and
        # misalign them with the student's square 1024x1024 canvas.
        feature_h, feature_w = feature.shape[-2:]
        square_side = max(int(feature_h), int(feature_w))
        feature = F.pad(
            feature,
            (0, square_side - int(feature_w),
             0, square_side - int(feature_h)))
        feature = F.interpolate(
            feature.unsqueeze(0), size=self.output_size, mode='bilinear',
            align_corners=False)[0].half().contiguous()
        results[self.output_key] = DC(feature, stack=True)
        results['dino_feature_cache_path'] = path
        with open(path, 'rb') as handle:
            results['dino_feature_cache_sha256'] = hashlib.sha256(
                handle.read()).hexdigest()
        return results

    def __repr__(self):
        return ('{}(cache_dir={!r}, output_size={!r}, '
                'expected_channels={!r}, expected_model={!r})').format(
                    self.__class__.__name__, self.cache_dir,
                    self.output_size, self.expected_channels,
                    self.expected_model)


def dino_invocation_encoding(record, box_key='dino_native_box'):
    """Describe legacy/current encodings of an all-lane DINO forward.

    Older complete all-lane audits predate ``dino_invoked`` but always carry
    the native-box key; a null box then means a computed miss.  Explicit
    ``False``/``0`` and ``raw_selected_source=not_computed`` remain invalid.
    """
    if record.get('raw_selected_source') == 'not_computed':
        return 'explicit_not_computed'
    if 'dino_invoked' not in record:
        return ('legacy_implicit_complete_box_key'
                if box_key in record else 'missing_evidence')
    marker = record['dino_invoked']
    if marker is True:
        return 'boolean_true'
    if (isinstance(marker, int) and not isinstance(marker, bool)
            and marker == 1):
        return 'integer_one'
    return 'explicit_not_computed'


def dino_record_was_computed(record, box_key='dino_native_box'):
    return dino_invocation_encoding(record, box_key=box_key) in {
        'boolean_true', 'integer_one', 'legacy_implicit_complete_box_key'}


def _audit_frame_key(filename):
    stem = os.path.splitext(os.path.basename(os.fspath(filename)))[0]
    match = _FRAME_RE.match(stem)
    if match is None:
        raise RuntimeError(
            'DINO proposal audit filename does not encode sequence/frame: '
            + stem)
    return '{}|{}'.format(
        match.group('sequence'), int(match.group('frame')))


@ROTATED_PIPELINES.register_module()
class LoadDinoProposalFromAudit:
    """Load one precomputed native-DINO OBB before geometric transforms.

    The proposal is registered as a rotated ``bbox_field`` so ``RResize`` and
    ``RRandomFlip`` transform it with exactly the same code as the GT OBB.
    This loader never invokes DINO and deliberately rejects conditional or
    partially-computed audit files.
    """

    def __init__(self,
                 audit_json,
                 expected_frame_count=None,
                 expected_split=None,
                 box_key='dino_native_box',
                 output_key='dino_proposals'):
        self.audit_json = os.path.abspath(os.fspath(audit_json))
        self.expected_frame_count = (None if expected_frame_count is None else
                                     int(expected_frame_count))
        self.expected_split = (None if expected_split is None else
                               str(expected_split))
        self.box_key = str(box_key)
        self.output_key = str(output_key)
        with open(self.audit_json, 'rb') as handle:
            raw = handle.read()
        payload = json.loads(raw.decode('utf-8'))
        if payload.get('protocol') != _SOURCE_OWNED_AUDIT_PROTOCOL:
            raise RuntimeError(
                'Geometry refiner requires an unrouted all-lane audit')
        records = list(payload.get('records') or [])
        if (self.expected_frame_count is not None
                and len(records) != self.expected_frame_count):
            raise RuntimeError(
                'DINO audit frame-count mismatch: expected {} got {}'.format(
                    self.expected_frame_count, len(records)))
        self.audit_sha256 = hashlib.sha256(raw).hexdigest()
        self._boxes = {}
        for record in records:
            for required in ('filename', 'sequence', 'frame', self.box_key):
                if required not in record:
                    raise RuntimeError(
                        'DINO proposal audit record is missing ' + required)
            if not dino_record_was_computed(
                    record, box_key=self.box_key):
                raise RuntimeError(
                    'DINO must be computed on every audit input frame')
            key = _audit_frame_key(record['filename'])
            expected_key = '{}|{}'.format(
                str(record['sequence']), int(record['frame']))
            if key != expected_key:
                raise RuntimeError(
                    'DINO audit sequence/frame disagrees with filename: '
                    '{} != {}'.format(key, expected_key))
            if key in self._boxes:
                raise RuntimeError('Duplicate DINO audit frame: ' + key)
            box = record.get(self.box_key)
            if box is None:
                array = np.zeros((0, 5), dtype=np.float32)
            else:
                array = np.asarray(box, dtype=np.float32).reshape(-1)
                if (array.size < 5 or not np.isfinite(array[:5]).all()
                        or np.any(array[2:4] <= 0.0)):
                    raise RuntimeError('Invalid DINO OBB at ' + key)
                array = array[:5].reshape(1, 5).copy()
            self._boxes[key] = array

    def __call__(self, results):
        filename = results.get('filename', results.get('ori_filename'))
        if not filename:
            raise RuntimeError('Image filename is unavailable to DINO loader')
        key = _audit_frame_key(filename)
        if key not in self._boxes:
            raise RuntimeError('Image is absent from DINO audit: ' + key)
        if self.expected_split is not None:
            normalized = os.path.normpath(os.fspath(filename)).split(os.sep)
            if (self.expected_split not in normalized
                    and not (self.expected_split == 'source-train'
                             and 'train' in normalized)):
                raise RuntimeError(
                    'DINO audit/image split mismatch for {}: {}'.format(
                        self.expected_split, filename))
        results[self.output_key] = self._boxes[key].copy()
        bbox_fields = results.setdefault('bbox_fields', [])
        if self.output_key not in bbox_fields:
            bbox_fields.append(self.output_key)
        results['dino_audit_sha256'] = self.audit_sha256
        results['dino_proposal_frame_key'] = key
        return results

    def __repr__(self):
        return ('{}(audit_json={!r}, expected_frame_count={!r}, '
                'expected_split={!r}, output_key={!r})').format(
                    self.__class__.__name__, self.audit_json,
                    self.expected_frame_count, self.expected_split,
                    self.output_key)


@ROTATED_PIPELINES.register_module()
class FormatDinoProposal:
    """Format the custom DINO OBB field for MMCV collation/scatter."""

    def __init__(self, key='dino_proposals'):
        self.key = str(key)

    def __call__(self, results):
        if self.key not in results:
            raise RuntimeError('Missing DINO proposal field: ' + self.key)
        from mmcv.parallel import DataContainer as DC
        from mmdet.datasets.pipelines import to_tensor
        results[self.key] = DC(to_tensor(results[self.key]))
        return results

    def __repr__(self):
        return '{}(key={!r})'.format(self.__class__.__name__, self.key)


@ROTATED_PIPELINES.register_module()
class SetNoFlipMetadata:
    """Declare deterministic no-flip metadata for training pipelines.

    MMDetection's default ``Collect`` metadata contract requires both keys
    even when geometric flipping is intentionally disabled.  Writing them
    explicitly is more robust across MMCV/MMDetection versions than relying
    on ``RandomFlip(flip_ratio=0)`` side effects.
    """

    def __call__(self, results):
        if results.get('flip') not in (None, False):
            raise RuntimeError(
                'SetNoFlipMetadata cannot overwrite an enabled flip')
        results['flip'] = False
        results['flip_direction'] = None
        return results

    def __repr__(self):
        return self.__class__.__name__ + '()'


@ROTATED_PIPELINES.register_module()
class LoadCausalHistoryFromAudit:
    """Load strictly preceding source frames and cached DINO proposals.

    Sequence/frame identity is used only to preserve chronology.  It is never
    returned as a model feature and never selects a detector or output route.
    Missing, non-consecutive, unreadable, or proposal-missing history is
    represented by a false validity mask rather than crossing a boundary.
    """

    def __init__(self,
                 audit_json,
                 history_horizon=4,
                 expected_frame_count=None,
                 expected_split=None,
                 box_key='dino_native_box'):
        if int(history_horizon) <= 0:
            raise ValueError('history_horizon must be positive')
        self.history_horizon = int(history_horizon)
        self.expected_split = (None if expected_split is None else
                               str(expected_split))
        loader = LoadDinoProposalFromAudit(
            audit_json=audit_json,
            expected_frame_count=expected_frame_count,
            expected_split=expected_split,
            box_key=box_key)
        self.audit_sha256 = loader.audit_sha256
        self._boxes = dict(loader._boxes)
        self._filenames = {}
        with open(os.path.abspath(os.fspath(audit_json)), 'r',
                  encoding='utf-8') as handle:
            payload = json.load(handle)
        for record in payload['records']:
            self._filenames[_audit_frame_key(record['filename'])] = str(
                record['filename'])

    @staticmethod
    def _resolve_history_filename(current_filename, recorded_filename):
        sibling = os.path.join(
            os.path.dirname(os.path.abspath(os.fspath(current_filename))),
            os.path.basename(os.fspath(recorded_filename)))
        if os.path.isfile(sibling):
            return sibling
        if os.path.isfile(recorded_filename):
            return os.path.abspath(os.fspath(recorded_filename))
        return None

    def __call__(self, results):
        current_filename = results.get(
            'filename', results.get('ori_filename'))
        if not current_filename or 'img' not in results:
            raise RuntimeError(
                'Current image must be loaded before causal history')
        current_key = _audit_frame_key(current_filename)
        match = re.match(r'^(?P<sequence>.+)\|(?P<frame>\d+)$', current_key)
        sequence = match.group('sequence')
        frame = int(match.group('frame'))
        images = []
        proposals = []
        valid = []
        frame_keys = []
        for age in range(1, self.history_horizon + 1):
            key = '{}|{}'.format(sequence, frame - age)
            recorded = self._filenames.get(key)
            box = self._boxes.get(key)
            resolved = (None if recorded is None else
                        self._resolve_history_filename(
                            current_filename, recorded))
            usable = bool(
                resolved is not None and box is not None
                and np.asarray(box).shape == (1, 5))
            if usable:
                image = mmcv.imread(resolved, flag='color')
                usable = image is not None
            if not usable:
                image = np.zeros_like(results['img'])
                proposal = np.asarray(
                    [[0.0, 0.0, 1.0, 1.0, 0.0]], dtype=np.float32)
            else:
                proposal = np.asarray(box, dtype=np.float32).copy()
            images.append(image)
            proposals.append(proposal)
            valid.append(usable)
            frame_keys.append(key if usable else None)
        results['causal_history_images_raw'] = images
        results['causal_history_proposals_raw'] = proposals
        results['causal_history_valid_mask'] = np.asarray(
            valid, dtype=np.bool_)
        results['causal_history_ages'] = np.arange(
            1, self.history_horizon + 1, dtype=np.int64)
        results['causal_history_frame_keys'] = frame_keys
        results['causal_history_audit_sha256'] = self.audit_sha256
        return results


@ROTATED_PIPELINES.register_module()
class PrepareCausalHistoryInputs:
    """Resize, normalize, and pad causal images into model coordinates.

    This transform is intentionally used with a deterministic current-frame
    resize and no geometric random flip.  Appearance augmentation may still
    affect the current frame independently, which provides source-only
    degraded-current/history-clean supervision without coordinate ambiguity.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.to_rgb = bool(to_rgb)

    def __call__(self, results):
        required = (
            'causal_history_images_raw',
            'causal_history_proposals_raw',
            'causal_history_valid_mask', 'img_shape', 'pad_shape')
        missing = [key for key in required if key not in results]
        if missing:
            raise RuntimeError(
                'Causal history preparation is missing: ' + ', '.join(missing))
        image_h, image_w = results['img_shape'][:2]
        pad_h, pad_w = results['pad_shape'][:2]
        prepared_images = []
        prepared_boxes = []
        for image, box in zip(
                results['causal_history_images_raw'],
                results['causal_history_proposals_raw']):
            old_h, old_w = image.shape[:2]
            resized = mmcv.imresize(image, (image_w, image_h))
            resized = mmcv.imnormalize(
                resized, self.mean, self.std, self.to_rgb)
            resized = mmcv.impad(
                resized, shape=(pad_h, pad_w), pad_val=0)
            mapped = np.asarray(box, dtype=np.float32).copy()
            mapped[:, 0] *= float(image_w) / max(float(old_w), 1.0)
            mapped[:, 1] *= float(image_h) / max(float(old_h), 1.0)
            size_scale = np.sqrt(
                float(image_w) / max(float(old_w), 1.0)
                * float(image_h) / max(float(old_h), 1.0))
            mapped[:, 2:4] *= size_scale
            prepared_images.append(resized.astype(np.float32, copy=False))
            prepared_boxes.append(mapped)
        results['causal_history_images'] = np.stack(
            prepared_images, axis=0)
        results['causal_history_proposals'] = np.concatenate(
            prepared_boxes, axis=0)
        del results['causal_history_images_raw']
        del results['causal_history_proposals_raw']
        return results


@ROTATED_PIPELINES.register_module()
class CausalHistoryProposalAugment:
    """Source-only proposal corruption for learning history rejection.

    The ranges are declared in the experiment config and are independent of
    fixed-target results.  Invalid history stays invalid.  Corruption labels
    are retained for audit diagnostics but are not supplied to the model.
    """

    def __init__(self,
                 current_probability=0.5,
                 history_probability=0.35,
                 history_dropout_probability=0.25,
                 center_fraction=0.20,
                 log_size=0.30,
                 angle_deg=12.0):
        self.current_probability = float(current_probability)
        self.history_probability = float(history_probability)
        self.history_dropout_probability = float(
            history_dropout_probability)
        self.center_fraction = float(center_fraction)
        self.log_size = float(log_size)
        self.angle_rad = float(angle_deg) * np.pi / 180.0
        probabilities = (
            self.current_probability, self.history_probability,
            self.history_dropout_probability)
        if any(value < 0.0 or value > 1.0 for value in probabilities):
            raise ValueError('Causal history probabilities must be in [0, 1]')

    def _perturb(self, box):
        result = np.asarray(box, dtype=np.float32).copy()
        if result.shape != (1, 5):
            return result
        local = np.random.uniform(
            -self.center_fraction, self.center_fraction, size=2)
        angle = float(result[0, 4])
        dx = local[0] * float(result[0, 2])
        dy = local[1] * float(result[0, 3])
        result[0, 0] += np.cos(angle) * dx - np.sin(angle) * dy
        result[0, 1] += np.sin(angle) * dx + np.cos(angle) * dy
        result[0, 2:4] *= np.exp(np.random.uniform(
            -self.log_size, self.log_size, size=2))
        result[0, 4] += np.random.uniform(-self.angle_rad, self.angle_rad)
        return result

    def __call__(self, results):
        current = np.asarray(results['dino_proposals'], dtype=np.float32)
        current_corrupted = bool(
            current.shape == (1, 5)
            and np.random.rand() < self.current_probability)
        if current_corrupted:
            results['dino_proposals'] = self._perturb(current)
        history = np.asarray(
            results['causal_history_proposals'], dtype=np.float32).copy()
        valid = np.asarray(
            results['causal_history_valid_mask'], dtype=np.bool_).copy()
        corrupted = np.zeros(valid.shape, dtype=np.bool_)
        dropped = np.zeros(valid.shape, dtype=np.bool_)
        for index in range(len(valid)):
            if not valid[index]:
                continue
            if np.random.rand() < self.history_dropout_probability:
                valid[index] = False
                dropped[index] = True
            elif np.random.rand() < self.history_probability:
                history[index:index + 1] = self._perturb(
                    history[index:index + 1])
                corrupted[index] = True
        results['causal_history_proposals'] = history
        results['causal_history_valid_mask'] = valid
        results['causal_current_proposal_corrupted'] = current_corrupted
        results['causal_history_corrupted_mask'] = corrupted
        results['causal_history_dropped_mask'] = dropped
        return results


@ROTATED_PIPELINES.register_module()
class FormatCausalHistoryInputs:
    """Format fixed-horizon causal tensors for MMCV collation/scatter."""

    def __call__(self, results):
        from mmcv.parallel import DataContainer as DC
        from mmdet.datasets.pipelines import to_tensor
        images = np.asarray(results['causal_history_images'])
        if images.ndim != 4:
            raise RuntimeError('Causal history images must have shape K,H,W,C')
        images = np.ascontiguousarray(images.transpose(0, 3, 1, 2))
        results['causal_history_images'] = DC(
            to_tensor(images), stack=True, pad_dims=2)
        for key in ('causal_history_proposals',
                    'causal_history_valid_mask', 'causal_history_ages'):
            # These are fixed-shape Kx5/K tensors, not images.  MMCV 1.7's
            # default pad_dims=2 asserts on ndim <= 2 during collation.
            results[key] = DC(
                to_tensor(results[key]), stack=True, pad_dims=None)
        return results


@ROTATED_PIPELINES.register_module()
class LoadPatchFromImage(LoadImageFromFile):
    """Load an patch from the huge image.

    Similar with :obj:`LoadImageFromFile`, but only reserve a patch of
    ``results['img']`` according to ``results['win']``.
    """

    def __call__(self, results):
        """Call functions to add image meta information.

        Args:
            results (dict): Result dict with image in ``results['img']``.

        Returns:
            dict: The dict contains the loaded patch and meta information.
        """

        img = results['img']
        x_start, y_start, x_stop, y_stop = results['win']
        width = x_stop - x_start
        height = y_stop - y_start

        patch = img[y_start:y_stop, x_start:x_stop]
        if height > patch.shape[0] or width > patch.shape[1]:
            patch = mmcv.impad(patch, shape=(height, width))

        if self.to_float32:
            patch = patch.astype(np.float32)

        results['filename'] = None
        results['ori_filename'] = None
        results['img'] = patch
        results['img_shape'] = patch.shape
        results['ori_shape'] = patch.shape
        results['img_fields'] = ['img']
        return results
