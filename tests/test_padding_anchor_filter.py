import importlib.util
from pathlib import Path
import runpy

import torch


def _load_anchor_utils():
    path = (Path(__file__).parents[1] / 'mmrotate' / 'core' / 'anchor'
            / 'utils.py')
    spec = importlib.util.spec_from_file_location('anchor_utils_standalone', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_content_mask_uses_img_shape_not_square_pad_shape():
    utils = _load_anchor_utils()
    anchors = torch.tensor([
        [4.0, 4.0, 8.0, 8.0, 0.0],
        [1020.0, 572.0, 8.0, 8.0, 0.0],
        [4.0, 580.0, 8.0, 8.0, 0.0],
        [1024.0, 100.0, 8.0, 8.0, 0.0],
    ])
    mask = utils.rotated_anchor_center_inside_flags(
        anchors, (576, 1024, 3))
    assert mask.tolist() == [True, True, False, False]


def test_content_mask_does_not_change_anchor_values():
    utils = _load_anchor_utils()
    anchors = torch.tensor([
        [100.0, 200.0, 32.0, 16.0, 0.25],
        [100.0, 700.0, 32.0, 16.0, 0.25],
    ])
    original = anchors.clone()
    mask = utils.rotated_anchor_center_inside_flags(
        anchors, (576, 1024, 3))
    assert torch.equal(anchors, original)
    assert torch.equal(anchors[mask], original[:1])


def test_padzero_diagnostic_only_overrides_eval_pipelines():
    config_path = (Path(__file__).parents[1] / 'crane_project' / 'configs'
                   / 'crane_symeood_k1_brightaug_valid_content_padzero_diag.py')
    config = runpy.run_path(str(config_path))
    assert 'train_pipeline' not in config
    assert config['data']['val']['pipeline'] is config['test_pipeline']
    assert config['data']['test']['pipeline'] is config['test_pipeline']
    pad = config['test_pipeline'][1]['transforms'][2]
    assert pad['type'] == 'Pad'
    assert pad['pad_val']['img'] == (0.0, 0.0, 0.0)
