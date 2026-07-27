import argparse
import json
import types

import numpy as np
import pytest
import torch

from crane_project.tools import dino_teacher_common as common


class _FakeDino:
    def get_intermediate_layers(self, tensor, n=1):
        batch = tensor.shape[0]
        tokens = torch.arange(
            batch * 8 * 3, dtype=tensor.dtype,
            device=tensor.device).reshape(batch, 8, 3)
        return (tokens,)


class _FakeHubDino(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))
        self.patch_embed = argparse.Namespace(
            patch_size=torch.Size([14, 14]))


class _FakeShardedDino(torch.nn.Module):
    def __init__(self, blocks=6):
        super().__init__()
        self.patch_embed = torch.nn.Conv2d(3, 4, 1)
        self.blocks = torch.nn.ModuleList(
            [torch.nn.Linear(4, 4) for _ in range(blocks)])
        self.norm = torch.nn.LayerNorm(4)


def test_legacy_sdpa_matches_explicit_attention():
    torch.manual_seed(0)
    query = torch.randn(1, 2, 4, 3)
    key = torch.randn(1, 2, 4, 3)
    value = torch.randn(1, 2, 4, 5)
    previous = common._LEGACY_SDPA_QUERY_CHUNK
    common.configure_legacy_sdpa_query_chunk(2)
    try:
        actual = common.legacy_scaled_dot_product_attention(
            query, key, value)
    finally:
        common.configure_legacy_sdpa_query_chunk(previous)
    weights = torch.softmax(
        torch.matmul(query, key.transpose(-2, -1)) / (3.0 ** 0.5),
        dim=-1)
    assert torch.allclose(actual, torch.matmul(weights, value), atol=1e-6)


def test_install_sdpa_compatibility_only_when_missing():
    functional = types.SimpleNamespace()
    assert common.install_torch_sdpa_compatibility(functional) is True
    assert functional.scaled_dot_product_attention is (
        common.legacy_scaled_dot_product_attention)
    assert common.install_torch_sdpa_compatibility(functional) is False


def test_resize_normalize_preserves_aspect_and_patch_grid():
    image = np.zeros((20, 41, 3), dtype=np.uint8)
    tensor, meta = common.resize_and_normalize_bgr(image, 28, 14)
    assert tensor.shape == (1, 3, 28, 70)
    assert meta['resized_shape'] == [28, 57]
    assert meta['padded_shape'] == [28, 70]
    assert meta['scale'] == pytest.approx(1.4)


def test_extract_patch_grid_reconstructs_spatial_tokens():
    tensor = torch.zeros(1, 3, 4, 8)
    feature = common.extract_patch_grid(_FakeDino(), tensor, patch_size=2)
    assert feature.shape == (1, 3, 2, 4)


def test_load_frozen_dinov2_strictly_loads_and_freezes(
        tmp_path, monkeypatch):
    repo = tmp_path / 'dinov2'
    repo.mkdir()
    (repo / 'hubconf.py').write_text('# fake local hub\n')
    checkpoint = tmp_path / 'dinov2.pth'
    torch.save(_FakeHubDino().state_dict(), checkpoint)
    monkeypatch.setattr(
        torch.hub, 'load', lambda *args, **kwargs: _FakeHubDino())
    model, patch_size = common.load_frozen_dinov2(
        str(repo), str(checkpoint), 'dinov2_vitl14', torch.device('cpu'))
    assert patch_size == 14
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_load_frozen_dinov2_rejects_checkpoint_mismatch(
        tmp_path, monkeypatch):
    repo = tmp_path / 'dinov2'
    repo.mkdir()
    (repo / 'hubconf.py').write_text('# fake local hub\n')
    checkpoint = tmp_path / 'wrong.pth'
    torch.save({'other': torch.ones(2)}, checkpoint)
    monkeypatch.setattr(
        torch.hub, 'load', lambda *args, **kwargs: _FakeHubDino())
    with pytest.raises(RuntimeError, match='checkpoint/model mismatch'):
        common.load_frozen_dinov2(
            str(repo), str(checkpoint), 'dinov2_vitl14',
            torch.device('cpu'))


def test_load_frozen_dinov2_wraps_py38_patch_failure(
        tmp_path, monkeypatch):
    repo = tmp_path / 'dinov2'
    repo.mkdir()
    (repo / 'hubconf.py').write_text('# fake local hub\n')
    checkpoint = tmp_path / 'dinov2.pth'
    torch.save(_FakeHubDino().state_dict(), checkpoint)
    monkeypatch.setattr(
        common, '_load_local_dinov2',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            TypeError("unsupported operand type(s) for |: 'type' and 'NoneType'")))
    monkeypatch.setattr(common, '_legacy_annotation_error', lambda _error: True)
    monkeypatch.setattr(
        common.py38_patcher, 'patch_repo',
        lambda _repo: (_ for _ in ()).throw(RuntimeError('patch failed')))
    with pytest.raises(RuntimeError, match='compatibility patch failed'):
        common.load_frozen_dinov2(
            str(repo), str(checkpoint), 'dinov2_vitl14',
            torch.device('cpu'))


def test_sharding_uses_contiguous_balanced_blocks():
    assert common.contiguous_device_indices(24, 3) == (
        [0] * 8 + [1] * 8 + [2] * 8)
    model = _FakeShardedDino(blocks=6)
    metadata = common.shard_frozen_dinov2(
        model, [torch.device('cpu'), torch.device('cpu')])
    assert metadata['block_count'] == 6
    assert len(model._sym_dino_device_hooks) == 6


def test_atomic_json_replaces_nonfinite_values(tmp_path):
    output = tmp_path / 'result.json'
    replacements = common.write_json_atomic(
        str(output), {'value': float('nan')})
    payload = json.loads(output.read_text())
    assert replacements == 1
    assert payload['value'] is None
    assert payload['serialization']['nonfinite_values_replaced'] == 1


def test_discover_records_preserves_sorted_source_protocol(tmp_path):
    ann_dir = tmp_path / 'val' / 'annfiles'
    img_dir = tmp_path / 'val' / 'images'
    ann_dir.mkdir(parents=True)
    img_dir.mkdir(parents=True)
    for frame in (2, 1):
        name = 'real_seq07_{:05d}'.format(frame)
        (ann_dir / (name + '.txt')).write_text('', encoding='ascii')
        (img_dir / (name + '.jpg')).write_bytes(b'image')
    records = common.discover_labeled_records(str(tmp_path), 'val', limit=1)
    assert [(row['seq'], row['frame']) for row in records] == [
        ('real_seq07', 1)]
