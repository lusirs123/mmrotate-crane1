# Copyright (c) OpenMMLab. All rights reserved.
import hashlib
import json
import os
import re

import mmcv
import numpy as np
from mmdet.datasets.pipelines import LoadImageFromFile

from ..builder import ROTATED_PIPELINES


_SOURCE_OWNED_AUDIT_PROTOCOL = 'source_owned_geometry_union_v2'
_FRAME_RE = re.compile(
    r'^(?P<sequence>(?:real|sim)_seq\d+)_(?P<frame>\d+)$')


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
