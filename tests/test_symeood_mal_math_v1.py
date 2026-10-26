import torch
import pytest
from crane_project.utils.symeood_mal_math_v1 import mal_terms, match_strength


def test_official_formula_and_detach():
    z=torch.tensor([[-2.],[.4],[1.]],dtype=torch.float64,requires_grad=True)
    q=torch.tensor([0.,1.,.2],dtype=torch.float64,requires_grad=True)
    y=torch.tensor([0,0,1]);w=torch.tensor([1.,1.,.5],dtype=torch.float64)
    m=mal_terms(z,y,q,w)
    expected=torch.nn.functional.binary_cross_entropy_with_logits(z,torch.tensor([[0.],[1.],[0.]],dtype=z.dtype),reduction='none')[:,0]
    expected=expected*torch.tensor([1.,1.,float(z[2].sigmoid().detach())**1.5*.5],dtype=z.dtype)
    torch.testing.assert_close(m,expected,rtol=0,atol=0)
    grad,gq=torch.autograd.grad(m.sum(),(z,q),allow_unused=True)
    assert gq is None
    torch.testing.assert_close(grad[2],.5*z[2].sigmoid().detach().pow(2.5))


def test_whole_rank_scalar_gradient_and_not_per_level():
    u=torch.tensor([1.,3.,2.],requires_grad=True)
    m=torch.tensor([10.,1.,.1],requires_grad=True)
    alpha,s,d=match_strength(u.sum(),m.sum())
    assert not d and not alpha.requires_grad
    torch.testing.assert_close(s,m.sum())
    gu,gm=torch.autograd.grad(s,(u,m),allow_unused=True)
    torch.testing.assert_close(gu,torch.ones(3)*alpha)
    assert gm is None and alpha>1


def test_ignored_empty_and_no_positive():
    for n in (0,3):
        z=torch.zeros(n,1,requires_grad=True)
        loss=mal_terms(z,torch.ones(n,dtype=torch.long),torch.zeros(n),torch.zeros(n)).sum()
        assert float(loss.detach())==0
        assert torch.equal(torch.autograd.grad(loss,z)[0],torch.zeros_like(z))


def test_zero_policy_and_uncapped_alpha():
    for u in (0.,1e-14,1e-12):
        a,s,d=match_strength(torch.tensor(u),torch.tensor(1e-13))
        assert d and a==1
    with pytest.raises(FloatingPointError): match_strength(torch.tensor(1e-13),torch.tensor(2e-12))
    a,_,_=match_strength(torch.tensor(1e-11),torch.tensor(10.))
    assert a>1e10
    for u,m in ((-1.,1.),(1.,float('nan'))):
        with pytest.raises(FloatingPointError):match_strength(torch.tensor(u),torch.tensor(m))


def test_quality_tolerance_extreme_logits_and_rng():
    state=torch.get_rng_state().clone()
    z=torch.tensor([[-1000.],[1000.]],requires_grad=True)
    l=mal_terms(z,torch.tensor([0,1]),torch.tensor([1.,0.]),torch.ones(2)).sum()
    assert torch.isfinite(l) and torch.isfinite(torch.autograd.grad(l,z)[0]).all()
    assert torch.equal(state,torch.get_rng_state())
    with pytest.raises(FloatingPointError):mal_terms(z,torch.tensor([0,1]),torch.tensor([1.01,0.]),torch.ones(2))
