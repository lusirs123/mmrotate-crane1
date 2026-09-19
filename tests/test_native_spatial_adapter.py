import torch

from crane_project.utils.native_spatial_adapter import NativeSpatialAdapter


def test_adapter_identity_and_gradient_path():
    module = NativeSpatialAdapter(8, 4)
    x = torch.randn(1, 8, 6, 5, requires_grad=True)
    y = module(x)
    assert y.shape == x.shape
    assert torch.equal(y.detach(), x.detach())
    y.square().mean().backward()
    assert module.residual_gate.grad is not None
    # Identity at initialization must not mean a dead residual branch.  The
    # zero output projection receives a usable gradient; after it moves, the
    # gate and earlier layers can receive gradients on the next step.
    assert module.refinement[-1].weight.grad is not None
    assert module.refinement[-1].weight.grad.abs().max().item() > 0.0
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    optimizer.step()
    module.zero_grad(set_to_none=True)
    module(x).square().mean().backward()
    assert module.residual_gate.grad.abs().max().item() > 0.0


def test_adapter_rejects_non_image_tensor():
    module = NativeSpatialAdapter(8, 4)
    try:
        module(torch.randn(8, 6, 5))
    except ValueError as exc:
        assert 'B,C,H,W' in str(exc)
    else:
        raise AssertionError('expected a shape validation error')
