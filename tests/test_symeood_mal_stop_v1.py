"""Regression evidence for the frozen control's non-shared loss minima."""
import pytest
import torch
import torch.nn.functional as F
from crane_project.utils.symeood_mal_math_v1 import mal_terms, match_strength


def objectives(logit):
    z=torch.tensor([[logit]],dtype=torch.float64,requires_grad=True)
    U=.25*.25*(1-z.sigmoid()).pow(2).mul(
        F.binary_cross_entropy_with_logits(z,torch.ones_like(z),reduction='none')).sum()
    M=.25*mal_terms(z,torch.tensor([0]),torch.tensor([.5],dtype=z.dtype),
                     torch.ones(1,dtype=z.dtype)).sum()
    return z,U,M


def test_near_zero_focal_does_not_imply_near_zero_soft_bce():
    z,U,M=objectives(10.)
    assert 0<float(U.detach())<1e-12 and float(M.detach())>1
    with pytest.raises(FloatingPointError,match='U near zero'):
        match_strength(U,M)


def test_matched_value_can_have_opposite_gradient():
    z,U,M=objectives(5.)
    alpha,S,_=match_strength(U,M)
    torch.testing.assert_close(S,M)
    gu=torch.autograd.grad(U,z,retain_graph=True)[0]
    gm=torch.autograd.grad(M,z,retain_graph=True)[0]
    gs=torch.autograd.grad(S,z)[0]
    torch.testing.assert_close(gs,alpha*gu)
    assert float(gs)<0<float(gm)


def test_soft_bce_retains_entropy_at_its_own_optimum():
    target=torch.tensor(.5**1.5,dtype=torch.float64)
    z=torch.logit(target).reshape(1,1).requires_grad_()
    M=.25*mal_terms(z,torch.tensor([0]),torch.tensor([.5],dtype=z.dtype),
                     torch.ones(1,dtype=z.dtype)).sum()
    gradient=torch.autograd.grad(M,z)[0]
    assert float(M.detach())>.1
    torch.testing.assert_close(gradient,torch.zeros_like(gradient),atol=1e-15,rtol=0)
