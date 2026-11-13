"""CPU mathematical checks for the auxiliary mapping; no optimizer or update."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from crane_project.utils import lvsr_decoded_edge_proxy_review_v1 as r
    torch.set_num_threads(1)
    torch.manual_seed(3107)
    f = torch.randn(2, 8, 9, 9, requires_grad=True)
    lengths = torch.tensor([2., 1.]).expand(2, 81, 2).clone().requires_grad_()
    with torch.no_grad(): lengths[0, :30] = torch.tensor([1., 2.]); lengths[1, 0] = 1.
    w = torch.rand(2, 1, 9, 9); w[:, :, 0, 0] = 0
    bank = torch.randn(20, 648, requires_grad=True)
    centers = torch.randn(20, 2)*.05; target = torch.randn(2, 2)*.05
    masks, role, order = r.masks(lengths, w)
    assert not masks.requires_grad and bool(role.sum(1).eq(1).all())
    assert torch.equal(masks.sum(1), w.sqrt().expand(2,8,9,9).flatten(1))
    assert torch.equal(order, lengths.detach().sort(-1, descending=True)[1])
    a = r.objectives(f, masks, target, bank, centers)
    grad = torch.autograd.grad(a['alignment'][:,0].sum(), f, retain_graph=True)[0].flatten(1)
    assert bool(grad[(masks[:,0] == 0)].eq(0).all())
    assert torch.autograd.grad(a['alignment'].sum(), lengths, allow_unused=True)[0] is None
    modified = f.detach().flatten(1).clone(); modified[masks[:,0] == 0] += 100
    b = r.objectives(modified.reshape_as(f), masks, target, bank, centers)
    assert torch.equal(a['alignment'][:,0], b['alignment'][:,0])
    changed = target.clone(); changed[:,1] += 20
    b = r.objectives(f, masks, changed, bank, centers)
    assert torch.equal(a['alignment'][:,0], b['alignment'][:,0])
    # Consistent branch relabelling preserves objectives without exchanging
    # arbitrary hidden channels between the two roles inside a descriptor.
    swapf = f.reshape(2,2,4,81).flip(1).reshape_as(f)
    swapbank = bank.reshape(20,2,4,81).flip(1).reshape_as(bank)
    # Use the first sample without exact ties for this relabelling check.
    m, _, _ = r.masks(lengths[:1].flip(-1), w[:1])
    b = r.objectives(swapf[:1], m, target[:1], swapbank, centers)
    assert torch.allclose(a['alignment'][:1], b['alignment'], atol=1e-6, rtol=1e-6)
    assert torch.allclose(a['proxy'][:1], b['proxy'], atol=1e-5, rtol=1e-5)
    zero = torch.zeros_like(f, requires_grad=True)
    z = r.objectives(zero, masks, target, bank, centers)
    zg = torch.autograd.grad(z['alignment'].mean(), zero)[0]
    assert bool(z['sample_norm'].eq(0).all()) and bool(torch.isfinite(zg).all())
    assert not any(v.grad is not None for v in (f, bank, zero))
    print(json.dumps(dict(passed=True, checks=7, updates=0,
                         note='Zero norms are detected, not silently repaired or accepted for training.')))


if __name__ == '__main__': main()
