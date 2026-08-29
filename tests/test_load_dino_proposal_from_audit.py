import importlib.util
import json
import pathlib
import sys
import types

import numpy as np
import pytest


class _Registry:
    def register_module(self, *args, **kwargs):
        del args, kwargs
        return lambda cls: cls


def _load_module():
    names = ('mmcv', 'mmdet', 'mmdet.datasets',
             'mmdet.datasets.pipelines', 'mmrotate', 'mmrotate.datasets',
             'mmrotate.datasets.builder', 'mmrotate.datasets.pipelines')
    modules = {name: types.ModuleType(name) for name in names}
    for module in modules.values():
        module.__path__ = []
    modules['mmdet.datasets.pipelines'].LoadImageFromFile = object
    modules['mmrotate.datasets.builder'].ROTATED_PIPELINES = _Registry()
    previous = {name: sys.modules.get(name) for name in names}
    sys.modules.update(modules)
    try:
        root = pathlib.Path(__file__).resolve().parents[1]
        path = root / 'mmrotate/datasets/pipelines/loading.py'
        name = 'mmrotate.datasets.pipelines.loading'
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


_DEFAULT = object()
_MISSING = object()


def _write(tmp_path, invoked=True, box=_DEFAULT,
           raw_selected_source=_MISSING):
    if box is _DEFAULT:
        box = [10., 20., 8., 4., 0.1, 0.8]
    record = dict(
        filename='/old/source/train/images/real_seq01_00001.jpg',
        sequence='real_seq01', frame=1, dino_native_box=box)
    if invoked is not _MISSING:
        record['dino_invoked'] = invoked
    if raw_selected_source is not _MISSING:
        record['raw_selected_source'] = raw_selected_source
    payload = dict(
        protocol='source_owned_geometry_union_v2', records=[record])
    path = tmp_path / 'audit.json'
    path.write_text(json.dumps(payload))
    return path


def test_loader_registers_native_dino_as_rotated_bbox_field(tmp_path):
    loader = MODULE.LoadDinoProposalFromAudit(
        _write(tmp_path), expected_frame_count=1,
        expected_split='source-train')
    results = loader(dict(
        filename='/new/source/train/images/real_seq01_00001.jpg',
        bbox_fields=['gt_bboxes']))
    assert results['bbox_fields'] == ['gt_bboxes', 'dino_proposals']
    assert results['dino_proposals'].dtype == np.float32
    assert results['dino_proposals'].shape == (1, 5)
    assert results['dino_proposal_frame_key'] == 'real_seq01|1'
    assert len(results['dino_audit_sha256']) == 64


def test_loader_rejects_conditional_or_partially_computed_audit(tmp_path):
    with pytest.raises(RuntimeError, match='computed on every'):
        MODULE.LoadDinoProposalFromAudit(
            _write(tmp_path, invoked=False), expected_frame_count=1)

    with pytest.raises(RuntimeError, match='computed on every'):
        MODULE.LoadDinoProposalFromAudit(
            _write(tmp_path, invoked=0), expected_frame_count=1)

    with pytest.raises(RuntimeError, match='computed on every'):
        MODULE.LoadDinoProposalFromAudit(
            _write(
                tmp_path, invoked=_MISSING,
                raw_selected_source='not_computed'),
            expected_frame_count=1)


@pytest.mark.parametrize('invoked', [1, _MISSING])
def test_loader_accepts_complete_legacy_invocation_encodings(
        tmp_path, invoked):
    loader = MODULE.LoadDinoProposalFromAudit(
        _write(tmp_path, invoked=invoked), expected_frame_count=1)
    results = loader(dict(filename='real_seq01_00001.jpg'))
    assert results['dino_proposals'].shape == (1, 5)


def test_loader_preserves_explicit_dino_missing_as_empty_proposal(tmp_path):
    loader = MODULE.LoadDinoProposalFromAudit(
        _write(tmp_path, box=None), expected_frame_count=1)
    results = loader(dict(filename='real_seq01_00001.jpg'))
    assert results['dino_proposals'].shape == (0, 5)


def test_loader_rejects_wrong_split(tmp_path):
    loader = MODULE.LoadDinoProposalFromAudit(
        _write(tmp_path), expected_frame_count=1, expected_split='val')
    with pytest.raises(RuntimeError, match='split mismatch'):
        loader(dict(
            filename='/new/source/train/images/real_seq01_00001.jpg'))


def test_no_flip_metadata_is_explicit_for_old_mmdetection_collect():
    transform = MODULE.SetNoFlipMetadata()
    result = transform(dict(img=np.zeros((4, 4, 3), dtype=np.uint8)))
    assert result['flip'] is False
    assert result['flip_direction'] is None
    with pytest.raises(RuntimeError, match='cannot overwrite'):
        transform(dict(flip=True, flip_direction='horizontal'))


def test_causal_history_loader_is_strictly_previous_and_never_crosses_gap(
        tmp_path, monkeypatch):
    records = []
    for frame in (1, 2, 4):
        filename = tmp_path / 'real_seq01_{:05d}.jpg'.format(frame)
        filename.write_bytes(b'image-placeholder')
        records.append(dict(
            filename=str(filename), sequence='real_seq01', frame=frame,
            dino_invoked=True,
            dino_native_box=[10. + frame, 20., 8., 4., 0.1, 0.8]))
    audit = tmp_path / 'history.json'
    audit.write_text(json.dumps(dict(
        protocol='source_owned_geometry_union_v2', records=records)))
    monkeypatch.setattr(
        MODULE.mmcv, 'imread',
        lambda path, flag='color': np.ones((8, 12, 3), dtype=np.uint8),
        raising=False)
    loader = MODULE.LoadCausalHistoryFromAudit(
        audit, history_horizon=3, expected_frame_count=3)
    current = str(tmp_path / 'real_seq01_00004.jpg')
    result = loader(dict(
        filename=current,
        img=np.zeros((8, 12, 3), dtype=np.uint8)))
    assert result['causal_history_frame_keys'] == [None, 'real_seq01|2',
                                                   'real_seq01|1']
    assert result['causal_history_valid_mask'].tolist() == [False, True, True]
    assert result['causal_history_ages'].tolist() == [1, 2, 3]
    assert result['causal_history_proposals_raw'][0].tolist() == [
        [0.0, 0.0, 1.0, 1.0, 0.0]]
