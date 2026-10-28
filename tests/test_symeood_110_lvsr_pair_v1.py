"""Only the newly changed pair-specific connection/interface is checked here."""
import inspect
import pytest
import torch
from crane_project.utils import port_size_pair_specific_v1 as pair
from crane_project.utils import port_size_qalign_v1 as old
from crane_project.tools import run_symeood_110_lvsr_pair_v1 as runner


def fixture(raw_swap=False, model_swap=False, empty=False):
    b = torch.tensor([[128., 128., 99.9 if raw_swap else 100.1,
                       100.1 if raw_swap else 99.9, .2, .9]])
    sx = 1.003 if raw_swap and not model_swap else .997 if model_swap and not raw_swap else 1.
    xy = torch.tensor([[sx, 1.]])
    bm = b[:, :5].clone()
    bm[:, :4] *= xy[:, [0, 1, 0, 1]]
    assert bool(bm[0, 2] < bm[0, 3]) == model_swap
    roi, support = torch.randn(1, 256, 9, 9), torch.ones(1, 1, 9, 9)
    values = [roi, support, b, bm, xy, b.clone()]
    return [v[:0] for v in values] if empty else values


def test_parameters_independent_zero_terminals_and_original_forward():
    torch.manual_seed(1703)
    h = pair.PairSpecificSizeHead()
    assert sum(p.numel() for p in h.parameters()) == 2480
    assert sum(p.numel() for p in h.stem.parameters()) == 2440
    assert sum(p.numel() for p in h.terminal.parameters()) == 40
    assert pair.PairSpecificSizeHead.forward is old.AlignedSizeHead.forward
    for i in (0, 1):
        assert not h.terminal.branches[i].weight.detach().any()
        assert not h.terminal.branches[i].bias.detach().any()
    a, b = [list(s.parameters()) for s in h.stem.branches]
    assert all(x.data_ptr() != y.data_ptr() for x, y in zip(a, b))
    assert not torch.equal(a[0], b[0])
    assert 'gt' not in inspect.signature(h.forward).parameters


@pytest.mark.parametrize('raw_swap,model_swap', [(False, False), (False, True), (True, False), (True, True)])
def test_new_neutral_channel_assembly_and_axis_mapping(raw_swap, model_swap):
    x = fixture(raw_swap, model_swap)
    h = pair.PairSpecificSizeHead()
    assert torch.equal(h(*x)['boxes_original'], x[-1])
    with torch.no_grad():
        h.terminal.branches[0].bias.copy_(torch.tensor([.001, .002, .003, .004]))
        h.terminal.branches[1].bias.copy_(torch.tensor([-.001, -.002, -.003, -.004]))
    out = h(*x)
    base = old.v.encode(x[-1], *x[2:5])
    expected = torch.tensor([.001, .002, -.001, -.002, .003, .004, -.003, -.004])
    assert torch.allclose((out['vectors']-base).reshape(1, 81, 8), expected[None, None].expand(1, 81, 8), atol=1e-7, rtol=0)
    # Equal forced terminal output through the sealed head proves the new
    # branch adapters did not change downstream axis/decoder/delivery math.
    sealed = old.AlignedSizeHead('fixed_b')
    hook = sealed.terminal.register_forward_hook(lambda mod, args, value: expected[None, :, None, None].expand_as(value))
    reference = sealed(*x)
    hook.remove()
    for key in ('vectors', 'delta_roi', 'decoded_sizes', 'local_size_variance', 'candidate_original', 'boxes_original', 'vector_valid', 'accepted', 'common_support'):
        assert torch.equal(out[key], reference[key]), key
    assert torch.equal(out['boxes_original'][:, [0, 1, 4, 5]], x[-1][:, [0, 1, 4, 5]])


def test_pair_gradient_routes_and_original_loss_with_frozen_inputs():
    x = fixture()
    x[0].requires_grad_(True)
    h = pair.PairSpecificSizeHead()
    with torch.no_grad():
        for t in h.terminal.branches:
            t.weight.fill_(.001)
    output = h(*x)
    params = list(h.parameters())
    # Only point0/2 vector coordinates: second pair's parameters have zero VJP.
    gradients = torch.autograd.grad(output['vectors'][:, :, [0, 2]].square().sum(), params, retain_graph=True)
    named = list(h.named_parameters())
    assert all(not g.any() for (name, _), g in zip(named, gradients) if '.branches.1.' in name)
    assert any(g.any() for (name, _), g in zip(named, gradients) if '.branches.0.' in name)
    gt = x[-1][:, :5].clone()
    gt[:, 2:4] *= torch.tensor([[.98, 1.01]])
    loss = old.size_loss(output, gt, x[-1], x[3])
    assert torch.equal(loss, old.residual.size_loss(output['delta_roi'], gt, x[-1], x[3]))
    loss.backward()
    assert torch.isfinite(loss) and x[0].grad is None
    for s, t in zip(h.stem.branches, h.terminal.branches):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in list(s.parameters())+list(t.parameters()))
        assert sum(float(p.grad.abs().sum()) for p in s.parameters()) > 0
        assert sum(float(p.grad.abs().sum()) for p in t.parameters()) > 0


def test_inherited_missing_support_and_whole_pair_fallback_and_cli_guard():
    h = pair.PairSpecificSizeHead()
    empty = fixture(empty=True)
    assert h(*empty)['boxes_original'].shape == (0, 6)
    x = fixture()
    x[1].zero_()
    out = h(*x)
    assert not out['vector_valid'].any() and torch.equal(out['boxes_original'], x[-1])
    x[1].fill_(1.)
    with torch.no_grad():
        h.terminal.branches[0].bias.copy_(torch.tensor([10., 10., -10., -10.]))
    out = h(*x)
    assert not out['accepted'].any() and torch.equal(out['boxes_original'], x[-1])
    assert runner.parser().parse_args([]).stage == 'check'
    with pytest.raises(SystemExit):
        runner.parser().parse_args(['--stage', 'train'])
