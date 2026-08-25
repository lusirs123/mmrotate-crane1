import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from crane_project.tools import symeood_dino_source_inventory as inventory


def _box(x=10.0):
    return [x, 10.0, 8.0, 4.0, 0.0, 0.9]


def _write_sample(root, split, name):
    image_dir = root / split / 'images'
    ann_dir = root / split / 'annfiles'
    image_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_dir / name), np.full((32, 32, 3), 80, np.uint8))
    (ann_dir / (Path(name).stem + '.txt')).write_text(
        '6 8 14 8 14 12 6 12 grab 0\n')


def test_source_split_rejects_test_and_val():
    with pytest.raises(RuntimeError):
        inventory._safe_source_split('test')
    with pytest.raises(RuntimeError):
        inventory._safe_source_split('val_extra')
    assert inventory._safe_source_split('extra_source') == 'extra_source'


def test_collection_config_is_unrouted_and_source_guarded():
    path = Path(
        'crane_project/configs/crane_symeood_dino_source_inventory_v1.py')
    text = path.read_text()
    assert "conditional_dino=dict(enabled=False)" in text
    assert "conservative_takeover=dict(enabled=False)" in text
    assert "target_data_read=False" in text
    assert "test_parameter_search=False" in text
    assert "part == 'test' or part.startswith('val')" in text


def test_inventory_keeps_micro_and_sequence_macro_separate(tmp_path):
    split = 'extra_source'
    records = []
    for name, sym_x, dino_x in [
            ('real_seq11_000001.jpg', 10.0, 10.0),
            ('real_seq11_000002.jpg', 10.0, 10.0),
            ('real_seq12_000001.jpg', 100.0, 10.0)]:
        _write_sample(tmp_path, split, name)
        records.append(dict(
            filename=name,
            sequence=name.rsplit('_', 1)[0],
            frame=int(Path(name).stem.rsplit('_', 1)[1]),
            sym_eood_original_box=_box(sym_x),
            dino_native_box=_box(dino_x)))
    payload = dict(
        protocol='source_owned_geometry_union_v2',
        metadata=dict(conditional_dino_enabled=False),
        frame_count=3,
        records=records)
    result = inventory.inventory(
        payload, tmp_path, split, 0.5, 0.1, 1, 1)
    assert result['target_data_read'] is False
    assert result['parameter_update_count'] == 0
    assert result['router_support']['eligible'] is True
    assert result['recommended_use'] == 'PREREGISTER_SOURCE_ONLY_ROUTER_GATE'
    assert result['micro_metrics']['sym_hit_rate'] == pytest.approx(2 / 3)
    assert result['sequence_macro_metrics']['sym_hit_rate'] == pytest.approx(0.5)
