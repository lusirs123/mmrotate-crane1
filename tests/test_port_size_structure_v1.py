from copy import deepcopy
import inspect
import io
import math
from types import SimpleNamespace
import pytest
import torch
from crane_project.utils import port_size_structure_v1 as m
from crane_project.tools import run_port_size_structure_v1 as r
from crane_project.utils import port_size_qalign_v1 as old
from test_port_size_native_p3_v1 import fixture


def heads():
    torch.manual_seed(1703);v=m.head(m.ARMS[0]);torch.manual_seed(1703);d=m.head(m.ARMS[1]);return v,d


def test_parameters_common_stem_exact_no_dead_parameters():
    v,d=heads()
    assert sum(p.numel() for p in v.parameters())==2800
    assert sum(p.numel() for p in d.parameters())==2746
    assert all(torch.equal(p,d.stem.state_dict()[k]) for k,p in v.stem.state_dict().items())
    for h in (v,d):assert not h.terminal.weight.detach().any() and not h.terminal.bias.detach().any()
    assert set(d.state_dict())==set(v.state_dict())


@pytest.mark.parametrize('swap,aniso,empty',[(False,False,False),(True,True,False),(False,False,True)])
def test_complete_neutral_and_all267_inputs(swap,aniso,empty):
    x=fixture(swap,aniso,empty)[:6];captured={};v,d=heads()
    for a,h in [('V',v),('D',d)]:
        handle=h.stem.register_forward_pre_hook(lambda mod,inputs,a=a:captured.update({a:inputs[0].clone()}))
        out=h(*x);assert torch.equal(out['boxes_original'],x[-1]);handle.remove()
    if not empty:
        assert torch.equal(captured['V'],captured['D']) and captured['D'].shape==(1,267,9,9)
        assert 'vector_valid' not in d(*x) and 'regression_valid' in d(*x)


@pytest.mark.parametrize('original_swap,model_swap',[(False,False),(False,True),(True,False),(True,True)])
def test_direct_canonical_to_raw_to_roi_independent_expected(original_swap,model_swap):
    x=list(fixture()[:6]);b=x[2];b[:,2:4]=torch.tensor([[99.9,100.1] if original_swap else [100.1,99.9]])
    x[4]=torch.tensor([[1.003,1.]]) if original_swap and not model_swap else torch.tensor([[.997,1.]]) if model_swap and not original_swap else torch.ones(1,2)
    x[3]=b[:,:5].clone();x[3][:,:4]*=x[4][:,[0,1,0,1]];x[5]=b.clone()
    assert bool(x[3][0,2]<x[3][0,3])==model_swap
    _,d=heads()
    with torch.no_grad():d.terminal.bias.copy_(torch.tensor([.001,-.001]))
    o=d(*x);raw=torch.tensor([[-.001,.001] if original_swap else [.001,-.001]])
    expected=raw.flip(-1) if model_swap else raw
    assert torch.equal(o['delta_roi'],expected)
    assert torch.allclose(o['candidate_original'][:,2:4],x[-1][:,2:4]*raw.exp(),atol=1e-5)
    assert torch.equal(o['boxes_original'][:,[0,1,4,5]],x[-1][:,[0,1,4,5]])


def test_direct_support_weighted_log_average_not_size_average():
    x=list(fixture()[:6]);_,d=heads();local=torch.zeros(1,2,9,9);local[0,0,:,0]=.04;local[0,1,:,8]=-.03
    hook=d.terminal.register_forward_hook(lambda mod,args,out:local)
    pair=m.read_pair(*x);w=pair['support'];expected=(local*w).sum((2,3))/w.sum((2,3))
    o=d(*x);canonical=o['delta_roi']
    # Fixture M and model B raw widths are canonical long.
    assert torch.allclose(canonical,expected,atol=1e-9)
    hook.remove()


def test_zero_partial_support_nonfinite_and_extreme_pair_fallback():
    x=list(fixture()[:6]);_,d=heads();x[1][:,:,:,0]=0;assert d(*x)['regression_valid'].all()
    with torch.no_grad():d.terminal.bias.fill_(100.)
    o=d(*x);assert not o['candidate_valid'].all() and torch.equal(o['boxes_original'],x[-1])
    with torch.no_grad():d.terminal.bias.fill_(float('nan'))
    o=d(*x);assert not o['regression_valid'].all() and torch.equal(o['boxes_original'],x[-1])
    x[1].zero_();o=m.head(m.ARMS[1])(*x);assert not o['regression_valid'].any() and torch.equal(o['boxes_original'],x[-1])


def test_square_and_near_square_protection():
    x=list(fixture()[:6]);x[2][:,2:4]=64.;x[3]=x[2][:,:5].clone();x[5]=x[2].clone();_,d=heads()
    with torch.no_grad():d.terminal.bias.copy_(torch.tensor([.01,-.01]))
    o=d(*x);assert o['square_bypass'].all() and torch.equal(o['boxes_original'],x[-1])
    x[2][:,2]=64.1;x[3]=x[2][:,:5].clone();x[5]=x[2].clone()
    with torch.no_grad():d.terminal.bias.copy_(torch.tensor([-.04,.04]))
    o=d(*x);assert not o['checks']['m_raw_order'].all() and torch.equal(o['boxes_original'],x[-1])


@pytest.mark.parametrize('arm',m.ARMS)
def test_loss_matches_original_gradient_freezing_and_adam_restore(arm):
    x=list(fixture()[:6]);x[0].requires_grad_(True);gt=x[-1][:,:5].clone();gt[:,2]*=.96;gt[:,3]*=1.04
    h=m.head(arm);opt=r.optimizer(h,torch)
    def step(h,opt):
        opt.zero_grad();o=h(*x);loss=m.size_loss(o,gt,x[-1],x[3],arm)
        assert torch.equal(loss,m.residual.size_loss(o['delta_roi'],gt,x[-1],x[3]));loss.backward()
        assert x[0].grad is None and all(p.grad is not None and torch.isfinite(p.grad).all() for p in h.parameters())
        stem=sum(float(p.grad.abs().sum()) for p in h.stem.parameters());torch.nn.utils.clip_grad_norm_(h.parameters(),10);opt.step();return stem
    assert step(h,opt)==0
    snapshot=dict(head=deepcopy(h.state_dict()),opt=deepcopy(opt.state_dict()),rng=torch.get_rng_state());buf=io.BytesIO();torch.save(snapshot,buf);buf.seek(0);got=torch.load(buf)
    clone=m.head(arm);clone.load_state_dict(got['head']);other=r.optimizer(clone,torch);other.load_state_dict(got['opt'])
    assert step(h,opt)>0;step(clone,other)
    assert r.equal(h.state_dict(),clone.state_dict(),torch) and r.equal(opt.state_dict(),other.state_dict(),torch)
    assert torch.equal(snapshot['rng'],got['rng'])
    assert 'gt' not in inspect.signature(h.forward).parameters


def test_V_matches_sealed_source_after_nonzero_weights():
    torch.manual_seed(9);h=m.head(m.ARMS[0]);torch.manual_seed(9);v=old.AlignedSizeHead('fixed_b');x=fixture()[:6]
    with torch.no_grad():h.terminal.bias.fill_(.001);v.terminal.bias.fill_(.001)
    a,b=h(*x),v(*x)
    for k in ['boxes_original','candidate_original','delta_roi','vectors','common_support','vector_valid']:assert torch.equal(a[k],b[k])


def test_optimizer_protocol_budget_and_no_GT_interface():
    p=r.protocol();fit=p['fit'];assert fit['lr']==.001 and fit['batch_size']==8 and fit['updates_per_epoch']==639 and fit['epochs_per_arm']==24
    assert fit['adam_betas']==[.9,.999] and fit['adam_eps']==1e-8 and fit['load_historical_checkpoint'] is False
    assert 'Only physical GPU3' in p['resource']
    assert all(v==0 or v is False for v in p['limits'].values())
    assert 'gt' not in inspect.signature(m.common_input).parameters
    assert '--epoch' not in r.parser().format_help()


def test_near_report_keeps_all_positive_damage_not_only_count():
    row=dict(image='sim_seq_x',domain='sim',sequence='sim_seq',frame_id=0,scale=1.,split='val',shard='val_s1',
      gt=[0,0,100,50,0],b=[0,0,100,50,0,.8],midpoint=[0,0,100,50,0,.8],edge_residual=[0,0,101,49,0,.8],size_delivery='accepted')
    d=r.near_report([row]);assert d['sim']['near']['long']['positive_harm_sum']==pytest.approx(.01)
    assert d['sim']['near']['short']['positive_harm_sum']==pytest.approx(.02)


@pytest.mark.parametrize('arm',m.ARMS)
def test_real_evaluation_interface_preguard_final_and_missing(arm,monkeypatch):
    x=list(fixture()[:6]);gt=x[-1][:,:5].clone();gt[:,2]*=.96
    row=dict(image='sim_seq_x',domain='sim',sequence='sim_seq',frame_id=0,scale=.5,split='train_sim',shard='train_s05',
        **{k:v for k,v in zip(('roi','support','boxes_original','boxes_model','scale_xy','midpoint_original','gt_original'),x+[gt])})
    missing=dict(row,image='sim_seq_missing',**{k:v[:0] for k,v in row.items() if torch.is_tensor(v) and k!='gt_original'})
    monkeypatch.setattr(r,'batch',lambda rows,torch:[torch.cat([rr[k] for rr in rows]) for k in ('roi','support','boxes_original','boxes_model','scale_xy','midpoint_original','gt_original')])
    h=m.head(arm);saved=r.evaluate(h,[row,missing],arm,torch)
    assert saved[0]['pre_guard']==saved[0]['edge_residual']==saved[0]['midpoint']
    assert saved[1]['edge_residual'] is None and saved[1]['size_delivery']=='no_output'
    assert ('vector_valid' in saved[0])==(arm==m.ARMS[0])
    assert ('regression_valid' in saved[0])==(arm==m.ARMS[1])
    if arm==m.ARMS[1]:
        with torch.no_grad():h.terminal.bias.copy_(torch.tensor([1.,-1.]))
        saved=r.evaluate(h,[row],arm,torch)[0]
        assert saved['size_delivery']=='fallback' and saved['pre_guard']!=saved['edge_residual'] and saved['edge_residual']==saved['midpoint']


def test_wrong_gpu_fails_before_creating_results(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','2')
    monkeypatch.setattr(r,'ROOT',tmp_path)
    out=tmp_path/'work_dirs'/r.VERSION/'unused'
    with pytest.raises(ValueError):r.main(SimpleNamespace(out_dir=out,resume_from=None,engineering_only=False))
    assert not out.exists()
