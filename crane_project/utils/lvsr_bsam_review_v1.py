"""CPU review primitives, not a production BSAM training integration.

One TRAIN-only 2D neighborhood and capped inverse-root density adaptation.
Paper-style W->U and author-code W->W must remain separately named.
"""
from contextlib import contextmanager
import math
import torch
from torch.nn import functional as F

WINDOW=math.log(1.02)
WEIGHT_CAP=4.


def canonical_pair(v,bm,m):
    swap=(bm[:,2]<bm[:,3])^(m[:,2]<m[:,3])
    return torch.where(swap[:,None],v.flip(-1),v)


def target_density(targets,window=WINDOW):
    """Counts include self; estimate once from the original TRAIN views only."""
    if targets.ndim!=2 or targets.shape[1]!=2 or not len(targets) or window<=0:
        raise ValueError('Nonempty TRAIN Nx2 targets and positive window required')
    if not bool(torch.isfinite(targets).all()):raise ValueError('Nonfinite targets')
    t=targets.detach().double();counts=[]
    for start in range(0,len(t),128):
        counts.append(((t[start:start+128,None]-t[None]).abs().amax(-1)<=window).sum(-1))
    return torch.cat(counts)


def density_weights(counts,cap=WEIGHT_CAP):
    """Global mean1 and cap4 together; no per-batch renormalization."""
    c=counts.detach().double()
    if c.ndim!=1 or not len(c) or not bool(torch.isfinite(c).all()) or bool((c<=0).any()) or cap<=1:
        raise ValueError('Positive TRAIN counts and cap>1 required')
    raw=c.rsqrt();lo=0.;hi=float(1/raw.min())*cap
    for _ in range(80):
        mid=(lo+hi)/2
        if float((raw*mid).clamp_max(cap).mean())<1:lo=mid
        else:hi=mid
    result=(raw*((lo+hi)/2)).clamp_max(cap)
    if abs(float(result.mean())-1)>1e-12:raise ValueError('Global density weights not mean1')
    return result


def loss_edges(head,x):
    from crane_project.utils import port_size_qalign_v1 as q
    output=head(*x[:-1])
    if not bool(output['vector_valid'].all()):raise ValueError('Invalid review vectors')
    target=q.residual.size_targets(x[-1],x[-2],x[3])['log_residual']
    raw=F.smooth_l1_loss(output['delta_roi'],target,beta=.1,reduction='none')
    return canonical_pair(raw,x[3],x[-2]),output


def flat_gradient(loss,parameters):
    grad=torch.autograd.grad(loss,parameters,allow_unused=True)
    flat=torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(parameters,grad)])
    if not bool(torch.isfinite(flat).all()):raise ValueError('Nonfinite review gradient')
    return flat.detach()


@contextmanager
def displaced(parameters,flat):
    """copy_ restoration in finally: exact even if an exception interrupts."""
    parameters=list(parameters);saved=[p.detach().clone() for p in parameters]
    if flat.numel()!=sum(p.numel() for p in parameters) or not bool(torch.isfinite(flat).all()):
        raise ValueError('Invalid parameter displacement')
    try:
        with torch.no_grad():
            at=0
            for p in parameters:
                p.add_(flat[at:at+p.numel()].reshape_as(p));at+=p.numel()
        yield
    finally:
        with torch.no_grad():
            for p,old in zip(parameters,saved):p.copy_(old)
        if any(not torch.equal(p,old) for p,old in zip(parameters,saved)):
            raise ValueError('Parameter restoration differs')


def perturbation(gradient,radius):
    if radius<0 or not math.isfinite(radius):raise ValueError('Invalid radius')
    norm=float(gradient.norm())
    return torch.zeros_like(gradient) if norm==0 else gradient*(radius/norm)


def review_update(head,opt,x,weights,radius,second_weighted=False):
    """Disposable synthetic tests only; never called by formal/diagnostic fits."""
    from crane_project.tools.diagnose_lvsr_small_fit_v1 import step
    if radius==0 and not second_weighted:return step(head,opt,x)
    parameters=[p for p in head.parameters() if p.requires_grad]
    opt.zero_grad();edges,_=loss_edges(head,x)
    weights=weights.detach().to(edges)
    if weights.shape!=(len(edges),) or not bool(torch.isfinite(weights).all()) or bool((weights<=0).any()):
        raise ValueError('Positive per-view weights required')
    g=flat_gradient((edges*weights[:,None]).mean(),parameters)
    eps=perturbation(g,radius)
    # Backward at the perturbed state; restore parameters before base Adam.step.
    with displaced(parameters,eps):
        second,_=loss_edges(head,x)
        loss=(second*weights[:,None]).mean() if second_weighted else second.mean()
        loss.backward()
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):raise ValueError('Invalid update gradient')
    norm=float(torch.nn.utils.clip_grad_norm_(parameters,10.));opt.step()
    return dict(loss=float(loss.detach()),gradient_norm=norm,perturbation_norm=float(eps.norm()))
