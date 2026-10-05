"""Tiny detached-image dispersion head; frozen predictions are never updated."""
from contextlib import contextmanager
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from crane_project.utils import port_size_residual_v4 as core


@contextmanager
def precision():
    """Disable TF32 for this candidate only; restore the baseline's settings."""
    old=[];deterministic=torch.backends.cudnn.deterministic;benchmark=torch.backends.cudnn.benchmark
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    for owner in (torch.backends.cuda.matmul, torch.backends.cudnn):
        if hasattr(owner, 'allow_tf32'):
            old.append((owner, owner.allow_tf32)); owner.allow_tf32=False
    try:
        yield
    finally:
        for owner,value in old:
            owner.allow_tf32=value
        torch.backends.cudnn.deterministic=deterministic;torch.backends.cudnn.benchmark=benchmark


class SizeDispersion(nn.Module):
    def __init__(self, standardizer):
        super().__init__()
        mean=torch.tensor(standardizer['mean'],dtype=torch.float32)
        scale=torch.tensor(standardizer['scale'],dtype=torch.float32)
        if mean.shape!=(3,) or scale.shape!=(3,) or not bool(torch.isfinite(mean).all()) or not bool((scale>0).all()) or not bool(torch.isfinite(scale).all()):
            raise ValueError('Invalid fitting-only standardizer')
        self.register_buffer('descriptor_mean',mean)
        self.register_buffer('descriptor_scale',scale)
        self.stem=nn.Conv2d(256,8,1)
        self.output=nn.Linear(19,2)
        nn.init.normal_(self.output.weight,mean=0.,std=core.SETTINGS['output_weight_std'])
        nn.init.constant_(self.output.bias,math.log(math.expm1(core.SETTINGS['initial_sigma']-core.SETTINGS['sigma_floor'])))

    def forward(self, roi, support, descriptor, arm):
        if arm not in core.ARMS:
            raise ValueError('Unknown dispersion arm')
        if roi.shape!=(1,256,9,9) or support.shape!=(1,1,9,9) or descriptor.shape!=(1,3):
            raise ValueError('Invalid online evidence shape')
        for value in (roi,support,descriptor):
            if value.requires_grad or value.grad_fn is not None or not bool(torch.isfinite(value).all()):
                raise ValueError('Only finite detached online evidence allowed')
            if value.device!=self.output.weight.device or value.dtype!=self.output.weight.dtype:
                raise ValueError('Evidence device/dtype differs')
        if bool(((support<0)|(support>1.00001)).any()) or (arm=='a1' and float(support.sum())<=1e-6):
            raise ValueError('No usable final-box image support')
        # 256 channels never enter a large fully-connected classifier. Only 16
        # spatial statistics of eight learned channels reach the two scales.
        y=F.relu(self.stem(roi)); mass=support.sum((2,3)).clamp(min=1e-6)
        mean=(y*support).sum((2,3))/mass
        variance=((y-mean[:,:,None,None])**2*support).sum((2,3))/mass
        image=torch.cat((mean,torch.sqrt(variance+core.SETTINGS['moment_epsilon'])),1)
        if arm=='a0': image=image*0.
        geometry=(descriptor-self.descriptor_mean)/self.descriptor_scale
        return self.output(torch.cat((geometry,image),1))


def loss_terms(raw, residual):
    if raw.shape!=(1,2) or residual.shape!=(1,2) or residual.requires_grad or residual.grad_fn is not None:
        raise ValueError('Expected fixed offline residual and two raw scales')
    if not bool(torch.isfinite(raw).all()) or not bool(torch.isfinite(residual).all()):
        raise ValueError('Nonfinite residual NLL input')
    # The two-scalar likelihood is evaluated in float64; the small head remains
    # float32. Negative NLL values are valid and must not trigger a failure.
    sigma=F.softplus(raw.double())+core.SETTINGS['sigma_floor']
    quadratic=.5*((residual.double()/sigma)**2).mean()
    normalization=sigma.log().mean()+.5*math.log(2*math.pi)
    return dict(total=quadratic+normalization,quadratic=quadratic,
                normalization=normalization,sigma=sigma)


def verify_autograd(raw_values, residual_values, device):
    raw=torch.tensor([raw_values],dtype=torch.float64,device=device,requires_grad=True)
    target=torch.tensor([residual_values],dtype=torch.float64,device=device)
    terms=loss_terms(raw,target); derivative=torch.autograd.grad(terms['total'],raw)[0][0].detach().cpu().numpy()
    oracle=core.nll_numpy(raw_values,residual_values)
    values=np.asarray(raw_values,dtype=float);finite=[]
    for axis in (0,1):
        positive=values.copy();negative=values.copy();positive[axis]+=1e-5;negative[axis]-=1e-5
        finite.append((core.nll_numpy(positive,residual_values)['total']-core.nll_numpy(negative,residual_values)['total'])/2e-5)
    if (not np.allclose(derivative,oracle['derivative'],rtol=1e-8,atol=1e-9) or
            not np.allclose(derivative,finite,rtol=1e-7,atol=1e-8) or
            not math.isclose(float(terms['total']),oracle['total'],rel_tol=1e-10,abs_tol=1e-10)):
        raise ValueError('Actual Torch NLL/gradient differs from independent oracle')
    return dict(passed=True,total=float(terms['total']),raw_gradient=derivative.tolist(),
                numpy_raw_gradient=oracle['derivative'],finite_difference_gradient=finite,
                precision='float64_two_scalar_likelihood')


def norm(parameters):
    return math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in parameters if p.grad is not None))


def update(raw, residual, model, optimizer, arm, measure=False):
    optimizer.zero_grad(); terms=loss_terms(raw,residual)
    derivative=torch.autograd.grad(terms['total'],raw,retain_graph=True)[0]
    with precision(): terms['total'].backward()
    groups={name:norm(list(module.parameters())) for name,module in (('stem',model.stem),('output',model.output))}
    before=norm(list(model.parameters()))
    if not math.isfinite(before) or any(not bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None):
        raise ValueError('Nonfinite candidate gradient')
    torch.nn.utils.clip_grad_norm_(model.parameters(),core.SETTINGS['clip_norm'])
    after=norm(list(model.parameters()))
    if not math.isfinite(after) or after>core.SETTINGS['clip_norm']+1e-5:
        raise ValueError('Invalid clipped candidate gradient')
    oracle=core.nll_numpy(raw.detach().cpu().numpy()[0],residual.detach().cpu().numpy()[0])
    if not np.allclose(derivative.detach().cpu().numpy()[0],oracle['derivative'],rtol=2e-5,atol=1e-6):
        raise ValueError('Raw NLL derivative differs; no multi-seed parameter decomposition used')
    record=dict(losses={k:float(terms[k].detach()) for k in ('total','quadratic','normalization')},
        sigma=terms['sigma'].detach().cpu().numpy()[0].tolist(),
        preclip_norm=before,postclip_norm=after,clip_retention=after/before if before>0 else 1.,
        parameter_groups_preclip=groups,raw_axis_gradient=derivative.detach().cpu().numpy()[0].tolist(),
        NLL_raw_component_gradient_norms={k:float(np.linalg.norm(oracle[k])) for k in
            ('derivative','quadratic_derivative','normalization_derivative')},
        raw_derivative_oracle_pass=True,a0_stem_zero_expected=arm=='a0',measured=measure)
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError('Nonfinite updated candidate state')
    return record
