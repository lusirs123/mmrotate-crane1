"""Check rotated foreground coverage and existing-artifact comparisons."""

import ast
import math
from pathlib import Path

import numpy as np
import torch

from crane_project.tools import audit_k1_dino_fpn_gradient_v3 as audit


ROOT = Path(__file__).resolve().parents[1]


def _mask_function():
    # Extract this pure tensor method without importing the CUDA/MMCV model.
    source = (ROOT / 'mmrotate/models/detectors/sym_eood_detector.py'
              ).read_text(encoding='utf-8')
    tree = ast.parse(source)
    detector = next(node for node in tree.body
                    if isinstance(node, ast.ClassDef)
                    and node.name == 'SymEOOD')
    method = next(node for node in detector.body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == '_build_distillation_foreground_mask')
    method.decorator_list = []
    namespace = dict(math=math)
    exec(compile(ast.Module(body=[method], type_ignores=[]),
                 '<distillation_mask>', 'exec'), namespace)
    return namespace[method.name]


def test_rotated_foreground_mask_encloses_ninety_degree_obb():
    mask_fn = _mask_function()
    feature = torch.zeros((1, 1, 16, 16))
    meta = [dict(img_shape=(16, 16, 3), pad_shape=(16, 16, 3))]
    upright = mask_fn(feature, meta, [torch.tensor([[8., 8., 10., 2., 0.]])])
    rotated = mask_fn(feature, meta, [torch.tensor(
        [[8., 8., 10., 2., math.pi / 2]])])
    assert upright[0, 0, 8, 4] == 1
    assert upright[0, 0, 4, 8] == 0
    assert rotated[0, 0, 4, 8] == 1
    assert rotated[0, 0, 8, 4] == 0


def test_artifact_weight_comparison_exposes_where_learning_occurred():
    left = {'backbone.x': torch.ones(2), 'neck.x': torch.zeros(2),
            'bbox_head.semantic_cls_adapter.weight': torch.zeros(1)}
    right = {'backbone.x': torch.ones(2), 'neck.x': torch.ones(2),
             'bbox_head.semantic_cls_adapter.weight': torch.ones(1)}
    comparison = audit.compare_states(left, right)
    assert comparison['backbone']['different_tensors'] == 0
    assert comparison['fpn']['different_tensors'] == 1
    assert comparison['fpn']['l2_delta'] == math.sqrt(2)
    assert comparison['classification_adapter']['different_tensors'] == 1


def test_pair_audit_finds_equal_aggregate_with_different_frames():
    target = np.array([10., 10., 4., 4., 0.])
    left = dict(boxes=[[10., 10., 4., 4., 0., .7], None],
                evaluated_boxes=[target.tolist(), None])
    right = dict(boxes=[None, [10., 10., 4., 4., 0., .7]],
                 evaluated_boxes=[None, target.tolist()])
    pair = audit._pair(left, right, [target, target], [0, 1])
    assert pair['left_only_output'] == 1
    assert pair['right_only_output'] == 1
    assert pair['left_only_center_hits'] == 1
    assert pair['right_only_center_hits'] == 1
