"""Full framework checks, executed in the server's frozen mmrotljj runtime."""
import torch
from unittest.mock import patch
from crane_project.utils.symeood_mal_v1 import SymEOODMALLossV1
from crane_project.utils.symeood_mal_math_v1 import match_strength
from mmrotate.models.losses.sym_nfl_loss import SymNFLLoss


def test_native_u_and_detached_matched_quality():
    for objective in ('U','MAL','S_MAL'):
        for labels in (torch.tensor([0,1,0,1]),torch.ones(4,dtype=torch.long)):
            z=torch.tensor([[-2.],[.4],[1.],[-1.]],requires_grad=True)
            boxes=torch.tensor([[5.,5.,4.,2.,0.]]*4,requires_grad=True)
            gt=boxes[labels==0].detach()
            weight=torch.tensor([1.,.5,1.,0.])
            old=SymNFLLoss(use_geometry_weight=False,loss_weight=.25)
            new=SymEOODMALLossV1(objective=objective,loss_weight=.25)
            state=torch.get_rng_state().clone()
            with patch('crane_project.utils.symeood_mal_v1.rbbox_overlaps',
                       side_effect=lambda p,g,is_aligned:torch.tensor([0.,1.])[:len(g)]):
                result=new(z,labels,boxes,gt,weight=weight,avg_factor=3.)
            r=new.records[0]
            native=old(z,labels,boxes,gt,weight=weight,avg_factor=3.)
            assert torch.equal(r['U'],native)
            assert torch.equal(torch.autograd.grad(r['U'],z,retain_graph=True)[0],
                               torch.autograd.grad(native,z,retain_graph=True)[0])
            alpha,s,_=match_strength(r['U'],r['M'])
            torch.testing.assert_close(torch.autograd.grad(s,z,retain_graph=True)[0],
                alpha*torch.autograd.grad(native,z,retain_graph=True)[0])
            assert torch.autograd.grad(r['M'],boxes,allow_unused=True)[0] is None
            assert int(new._local_iter)==int(old._local_iter)==1
            assert torch.equal(state,torch.get_rng_state())


def test_counters_and_many_fpn_one_scalar():
    loss=SymEOODMALLossV1(objective='S_MAL',loss_weight=.25)
    z=torch.zeros(2,1,requires_grad=True);labels=torch.ones(2,dtype=torch.long)
    boxes=torch.tensor([[5.,5.,4.,2.,0.]]*2)
    for rank_denominator in (2.,5.):
        loss.records=[]
        for level in range(5):
            loss(z+level,labels,boxes,boxes[:0],weight=torch.ones(2),avg_factor=rank_denominator)
        u=sum(r['U'] for r in loss.records);m=sum(r['M'] for r in loss.records)
        a,s,_=match_strength(u,m)
        torch.testing.assert_close(s,m)
        torch.testing.assert_close(torch.autograd.grad(s,z,retain_graph=True)[0],
                                   a*torch.autograd.grad(u,z)[0])
    assert int(loss._local_iter)==10
