import importlib.util
from pathlib import Path

import torch


_MODULE_PATH = (Path(__file__).parents[1] / 'mmrotate' / 'models' / 'losses'
                / 'object_background_relation.py')
_SPEC = importlib.util.spec_from_file_location('object_background_relation',
                                                str(_MODULE_PATH))
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
ObjectBackgroundRelationDistillation = (
    _MODULE.ObjectBackgroundRelationDistillation)


def test_relation_loss_is_finite_and_reaches_adapter_only():
    loss_fn = ObjectBackgroundRelationDistillation(loss_weight=0.05)
    base = torch.randn(1, 4, 32, 32)
    adapter = torch.nn.Conv2d(4, 4, 1, bias=False)
    torch.nn.init.zeros_(adapter.weight)
    student = (base.detach() + adapter(base.detach()),)
    teacher = torch.randn(1, 6, 16, 16)
    boxes = [torch.tensor([[16., 16., 8., 6., 0.]])]
    metas = [dict(img_shape=(32, 32, 3), pad_shape=(32, 32, 3))]
    loss = loss_fn(student, teacher, boxes, metas)
    grad = torch.autograd.grad(loss, adapter.weight)[0]
    assert torch.isfinite(loss)
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad) > 0
    assert base.grad is None


def test_relation_loss_returns_zero_when_no_valid_background():
    loss_fn = ObjectBackgroundRelationDistillation()
    student = (torch.randn(1, 4, 4, 4, requires_grad=True),)
    teacher = torch.randn(1, 6, 4, 4)
    boxes = [torch.tensor([[16., 16., 32., 32., 0.]])]
    metas = [dict(img_shape=(32, 32, 3), pad_shape=(32, 32, 3))]
    loss = loss_fn(student, teacher, boxes, metas)
    assert loss.item() == 0.0
    loss.backward()
