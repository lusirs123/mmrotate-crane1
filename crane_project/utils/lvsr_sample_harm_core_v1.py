"""Pinned official harm solver, minimal compatibility/logging extraction.

Upstream avranasa/sample-harm-reduction ec532cb8f50ea51f096444fe3951921e8ef26cb8.
Only four solver functions: postponed annotations and no logging dependency;
get_sample_scales additionally returns logs/effective matrices for audit.
Numerical equations and upstream safeguards preserved. use_wandb MUST be False.

MIT License

Copyright (c) Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""
from __future__ import annotations
import torch

@torch.no_grad()
def lanczos_extremal(q, use_wandb=False) -> tuple[torch.Tensor, torch.Tensor]:
    n = q.shape[0]
    if n <= 256:
        # on a T4 gpu on BS >= 512 we saw Lanczos to be considerably faster
        eigs = torch.linalg.eigvalsh(q)
        if use_wandb: 
            wandb.log({ "eig_min before clamp": eigs[0] }, commit=False)
        eig_min = torch.clamp(eigs[0], min=1e-12)
        eig_max = torch.clamp(eigs[-1], min=1e-12)
        return eig_min, eig_max 
    iters=max(20, n//32)
    iters = min(iters, 128) # Good results even smaller cap (e.g. 64)
    if iters < 1:
        iters = 1
    v = torch.randn(n, device=q.device, dtype=q.dtype)
    v = v / v.norm()
    v_prev = torch.zeros_like(v)
    beta = torch.tensor(0.0, device=q.device, dtype=q.dtype)
    alphas, betas = [], []
    for _ in range(iters):
        w = q @ v
        alpha = v @ w
        w = w - alpha * v - beta * v_prev
        beta = w.norm()
        alphas.append(alpha)
        if beta.item() < 1e-12: break
        betas.append(beta)
        v_prev, v = v, w / beta
    
    alpha_diag = torch.stack(alphas)
    if len(alphas) <= 1:
        return alpha_diag[0], alpha_diag[0]
    
    
    k = len(alphas)
    t = torch.zeros((k, k), device=q.device, dtype=q.dtype)
    t.diagonal().copy_(alpha_diag)
    beta_stack = torch.stack(betas)[:k - 1]
    t.diagonal(1).copy_(beta_stack)
    t.diagonal(-1).copy_(beta_stack)
    
    eigs = torch.linalg.eigvalsh(t)
    
    # Ensure eigenvalues are positive to avoid negative condition numbers    
    if use_wandb: 
        wandb.log({ "eig_min before clamp": eigs[0] }, commit=False)
    return eigs[0], eigs[-1]


@torch.no_grad()
def get_sample_scales(Q1, Q2, Gdg, dgdg, 
                      smooth_condition_Q1,
                      smooth_dgdg,
                      LR, grad_scale_params):
    vmap_sum_grads = grad_scale_params["sum_grads"]
    warm_start_iter= grad_scale_params["warm_start_iter"]
    reg = grad_scale_params.get("reg", 1e-2)
    max_iter=grad_scale_params.get("max_iter", 20)
    tol= grad_scale_params.get("tol", 1e-5)
    use_wandb=grad_scale_params.get("use_wandb", False)
    constr = grad_scale_params.get("constraint", "both").lower()
    assert constr in ['box', 'ellipsoid', 'both'], f"Constraint {constr} not recognized"
    box_bound= grad_scale_params["box_bound"]
    radius_bound= grad_scale_params["radius_bound"]
    allow_neg_weights= grad_scale_params.get("allow_neg_weights", False)
    if LR < 1e-8: LR = 1e-8  # avoid division by zero 
    BS = Q1.shape[0]       
    
    # ==============================================
    # STABILITY CHECKS AND ADAPTIVE PRECONDITIONING
    # Safeguaard 1:
    # We check the condition number of Q1 using Lanczos iteration. If it drops too much
    # we increase the diagonal of Q1 (and Q2 if AdamW) to keep the problems equally
    # well-conditioned accross iterations. We track the condition number via an exponential 
    # moving average. If it goes above "max_raise_condition_Q * smoothed_condition", we add 
    # a scaled identity matrix.
    # Since Q is positive semi-definite and symmetric, all eigenvalues are non-negative,
    # and the singular values are equal to the eigenvalues (max eigenvalue is spectral norm).
    #
    # If max_cond = max_raise_condition_Q * smooth_condition_Q1, then the added identity 
    # matrix is multiplied by (eig_max - max_cond * eig_min) / (max_cond - 1) so as the 
    # new condition number (eig_max + alpha) / (eig_min + alpha) = max_cond.
    # 
    # Safeguard 2:
    # Also if current_dgdg > max_raise_dgdg * smoothed_dgdg we don't trust the current dgdg
    # and we scale it down to max_raise_dgdg * smoothed_dgdg       
    mom_stats_Q = grad_scale_params["mom_stats_Q"] 
    max_raise_condition_Q = grad_scale_params["max_raise_condition_Q"]
    max_raise_dgdg = grad_scale_params["max_raise_dgdg"]

    eig_min, eig_max = lanczos_extremal(Q1, use_wandb)
    # Ensure eigenvalues are positive to avoid negative condition numbers
    eig_min = torch.clamp(eig_min, min=1e-12)
    eig_max = torch.clamp(eig_max, min=1e-12)
    cond_Q1 = eig_max / eig_min
    if use_wandb: 
        wandb.log({ "cond_Q1_before": cond_Q1}, commit=False)
    changed_Q_or_dgdg = 0 # 0: no change, 1: changed Q, 2: changed dgdg, 3: changed both
    alpha = 0.0
    if smooth_condition_Q1.item() < 0:
        smooth_condition_Q1.copy_(cond_Q1)
    else:
        max_cond = max_raise_condition_Q * smooth_condition_Q1
        if cond_Q1 > max_cond:
            alpha = (eig_max - max_cond * eig_min) / (max_cond - 1.0 + 1e-10)
            assert alpha >= 0
            changed_Q_or_dgdg = 1
            Q1 += alpha * torch.eye(BS, device=Q1.device, dtype=Q1.dtype)
            if Q2 is not None:
                Q2 += alpha * torch.eye(BS, device=Q2.device, dtype=Q2.dtype)
            cond_Q1 = max_cond # After adjustment, cap it for EMA
            eig_max += alpha
            eig_min += alpha

        # exp moving average update
        smooth_condition_Q1.mul_(mom_stats_Q)
        smooth_condition_Q1.add_((1 - mom_stats_Q) * cond_Q1)
        
    # similarly for dgdg
    if smooth_dgdg.item() < 0:
        # Initialize smooth_dgdg
        # The +1 to compensate for the training routines using a warm start in their scheduler        
        smooth_dgdg.copy_(dgdg + 1) 
    max_dgdg = max_raise_dgdg * smooth_dgdg
    if dgdg > max_dgdg:
        changed_Q_or_dgdg = 2 if changed_Q_or_dgdg == 0 else 3
        # scale down dgdg to max_dgdg
        Gdg = Gdg * (max_dgdg / dgdg)
        dgdg = max_dgdg
    smooth_dgdg.mul_(mom_stats_Q)
    smooth_dgdg.add_((1 - mom_stats_Q) * dgdg)

    if vmap_sum_grads:
        q = (BS / LR) * Gdg 
    else:
        q = Gdg / LR             
    max_energy = max_w = None
    if constr in ['ellipsoid', 'both']:
        if vmap_sum_grads:
            max_energy = (BS * radius_bound / LR)**2 * dgdg   
        else:
            max_energy = (radius_bound / LR)**2 * dgdg  
    if constr in ['box', 'both'] :
        max_w = box_bound
    w_opt, to_log = ChambollePock_hinge(Q1, Q2, q, 
                        constraint=constr,
                        max_energy = max_energy,
                        max_w = max_w, 
                        allow_neg_weights=allow_neg_weights,
                        warm_start_iter=warm_start_iter,
                        reg=reg, max_iter=max_iter, tol=tol,
                        norm_Q1=eig_max) 
    grads_scales = 1 + w_opt 

    if use_wandb:   
        if Q2 is not None:
            Q2_diag = Q2.detach().diag()
            diag_Q2_avg_abs = Q2_diag.mean() # Q diagonal are all non-negative
            off_diag_Q2_avg_abs = (Q2.detach().abs().sum() - BS * diag_Q2_avg_abs) / (BS - 1)
            ratio_diag_vs_off_diag_Q2 = diag_Q2_avg_abs / (off_diag_Q2_avg_abs + 1e-10)     
        negative_corr = Q1.clamp(max=0.0).sum(dim=1)
        Q1_diag = Q1.detach().diag()
        portion_bad_corr = -negative_corr / Q1_diag.clamp(min=1e-10)            
        scales_std, scales_mean = torch.std_mean(grads_scales)
        to_log.update({
            "max_energy": max_energy,
            "avg_abs_diag_Q": Q1_diag.mean(),
            "avg_abs_diag_Q2": diag_Q2_avg_abs if Q2 is not None else None,
            "avg_row_sum_abs_off_diag_Q": (Q1.detach().abs().sum() - BS * Q1_diag.mean()) / (BS - 1), # per row average.
            "cond_Q1": cond_Q1.item(),
            "alpha in: Q+alpha*I": alpha.item() if isinstance(alpha, torch.Tensor) else alpha,
            "ratio diag vs off_diag Q2": ratio_diag_vs_off_diag_Q2.item() if Q2 is not None else None,
            "avg_abs_q": q.abs().mean(),
            "mean(portion_bad_corr)": portion_bad_corr.mean().item(),
            "mean(scale_opt)": scales_mean.item(),
            "std(scale_opt)": scales_std.item(),
            "changed_Q_or_dgdg": changed_Q_or_dgdg # 0: no change, 1: changed Q, 2: changed dgdg, 3: changed both
        })
        wandb.log(to_log, commit=False)
    return grads_scales, to_log, dict(Q1=Q1.clone(), Q2=Q2.clone(), q=q.clone(), energy=max_energy.clone(), guard=changed_Q_or_dgdg, diagonal_alpha=alpha)


@torch.no_grad()
def projection_ellipsoid_box(x0, Q, max_r, a, b):
    # Approximates projection of x0 onto the intersection of the ellipsoid defined by x^T Q x <= max_r
    # and the box defined by a <= x <= b. First projects onto the box and then moves towards the projected
    # direction of steepest energy descend (i.e. negative gradient of x^T Q x) on the ellipsoid surface.
    # 1. Box Projection
    if (a is not None) or (b is not None):
        x = torch.clamp(x0, min=a, max=b)
    else:
        x = x0
    
    # 2. Check Ellipsoid Constraint
    current_r = torch.dot(x, Q @ x)
    if current_r > max_r:
        x = (max_r / current_r)**0.5 * x  # box constraint first guarantees this radial projection is inside box
    
    return x


@torch.no_grad()
def ChambollePock_hinge(Q1, Q2, q,
                    constraint = 'both',
                    max_energy=None, max_w=None, allow_neg_weights=False,
                    warm_start_iter=0,
                    reg=0.0, max_iter=50, tol=1e-3, norm_Q1=None):
    """
    Solves the problem:
    min_w  sum(max(0, -Q1 @ w + q)) + epsilon * ||w||^2
    s.t.   w^T @ Q2 @ w <= max_energy (if constraint='ellipsoid' or 'both')
           -max_w <= w <= max_w (if constraint='box' or 'both')
           w >= 0 (if allow_neg_weights=False)

    Using the Chambolle-Pock algorithm.
    
    Args:
        Q1 (torch.Tensor): The operator matrix (B, B). Assumed symmetric semi-positive definite.
        Q2 (torch.Tensor): The constraint matrix (B, B).
        q (torch.Tensor): The linear term (B,).
        max_energy (float): The ellipsoid constraint bound.
        max_w (float): The box constraint bound.
        constraint (str): The type of constraint ('box' or 'ellipsoid' or 'both').
        reg (float): The regularization coefficient factor (default 0.01).
        allow_neg_weights (bool): If True, removes the w >= 0 constraint.
        max_iter (int): Maximum number of iterations.
        warm_start_iter (int): Number iteration of Projected GD for warm start
        tol (float): Tolerance for stopping.
    """
    assert warm_start_iter < max_iter // 2, "Warm start iterations should be less than half of max_iter."
    if constraint == 'ellipsoid': 
        assert max_energy is not None and max_w is None
        if max_energy <= 1e-10:
            # trivial case
            return torch.zeros_like(q), {"iterations": 0}
    elif constraint in ('box', 'both'):
        assert max_w is not None 
        if max_w <= 1e-10:
            # trivial case
            return torch.zeros_like(q), {"iterations": 0}
    else:   
        raise NotImplementedError
        
    orig_dtype = Q1.dtype
    if Q1.dtype in (torch.float16, torch.bfloat16):
        Q1 = Q1.to(torch.float32)
        Q2 = Q2.to(torch.float32)
        q = q.to(torch.float32) 
    
    device = Q1.device  
    B = Q1.shape[0]
    relu_q_sum = q.clamp(min=0.0).sum()
    obj_init = relu_q_sum.item()
    
    # --- 1. Initialization ---
    if q.isnan().any():
        print("Warning: NaNs detected in q. Returning zero weights.")
        return torch.zeros_like(q), {"iterations": -1}
    if torch.isnan(Q1).any() or torch.isnan(Q2).any():
        print("Warning: NaNs detected in Q1 or Q2. Returning zero weights.")
        return torch.zeros_like(q), {"iterations": -1}
        

    # --- 2. Step Sizes & Constants ---    
    if norm_Q1 is not None:
        norm_Q1 = norm_Q1
    else:
        try:
            #norm_Q1 = torch.linalg.matrix_norm(Q1, ord=2) # as dimmensions grow this can become the bottleneck!
            _, norm_Q1 = lanczos_extremal(Q1)
        except RuntimeError:
            norm_Q1 = torch.linalg.matrix_norm(Q1, ord='fro')
    tau = 0.5 / (norm_Q1 + 1e-8)
    sigma = 0.5 / (norm_Q1 + 1e-8)
    
    epsilon = 0.0
    if reg > 0.0:
        epsilon = reg * Q2.diag().sum() * relu_q_sum / (max_energy * B)     
    
    min_w = None
    if not allow_neg_weights:
        min_w = 0.0
    if constraint in ('box', 'both') and max_w is not None:
        if allow_neg_weights:
            min_w = -max_w

    # --- 3. Initialize Variables ---  
    z = torch.zeros(B, device=device, dtype=q.dtype)
    y = torch.zeros(B, device=device, dtype=q.dtype)
    num_iter_ws = 0
    if warm_start_iter > 0:
        # projected gradient descend (PGD) for a few steps
        # objective: min_w  sum(max(0, -Q1 @ w + q)) 
        lr = 0.9 / (norm_Q1 + 1e-8)
        obj = obj_init
        y = (q>0).to(q.dtype)
        reduction_iter = 0
        for _ in range(warm_start_iter):
            num_iter_ws += 1
            g = - Q1.T @ y
            z = z - lr * g
            if constraint == 'box':
                z = z.clamp_(min=min_w, max=max_w)
            elif constraint in ('both', 'ellipsoid'):
                z = projection_ellipsoid_box(z, Q2, max_energy, min_w, max_w)
            else:
                raise NotImplementedError
            
            linear_obj = -Q1 @ z + q
            new_obj = linear_obj.clamp(min=0.0).sum()
            if new_obj >= obj:
                lr *= 0.5
                reduction_iter += 1
                if reduction_iter >= 4:
                    break
            else:
                obj = new_obj
            y = (linear_obj > 0).to(q.dtype)
        if obj >= obj_init:
            z.zero_()
            y.zero_()
    z_bar = z.clone()                

    avg_change = 0.0
    dual_res = 0.0
    reduced_factor = 1.0

    # --- 4. Optimization Loop ---
    for k in range(num_iter_ws, max_iter):
        z_old_iter = z.clone() # Snapshot at start of iteration k
        y_old_iter = y.clone()
        
        # Recompute regularization factor for current tau
        d_reg = None
        if reg > 0.0:
            d_reg = 1.0 / (1.0 + 2 * tau * epsilon)

        # 1. Dual Update (uses z_bar from previous iteration)
        y_next = (y_old_iter + sigma * (-Q1 @ z_bar + q)).clamp_(min=0.0, max=1.0)

        # 2. Primal Update
        z_next = z_old_iter + tau * (Q1.T @ y_next)

        # 3. Regularization (Scale)
        if d_reg is not None:
            z_next *= d_reg

        # 4. Projections
        if constraint =='box':
            z_next_proj = z_next.clamp_(min=min_w, max=max_w)
        elif constraint in ('both', 'ellipsoid'):
            z_next_proj = projection_ellipsoid_box(z_next, Q2, max_energy, min_w, max_w)
        else:
            raise NotImplementedError
        if (z_next_proj - z_next).norm() > 2 * (z_next.norm() + 1e-8):
            # too far from unprojected point -> unstable 
            # (almost never needed)
            reduced_factor *= 0.5
            tau *= 0.5
            sigma *= 0.5
            continue 
        
        
        if torch.isnan(z_next_proj).any() or torch.isnan(y_next).any():
            print(f"CP failed to find stable step at iter {k}. Stopping.")
            break
        else:
            z = z_next_proj
            y = y_next

        # 5. Extrapolation (Standard CP)
        z_bar = z + 0.9 * (z - z_old_iter)

        # --- Convergence Check ---
        diff = (z - z_old_iter).abs()
        avg_change = diff.mean()
        dual_res = (y - y_old_iter).norm() / (y.norm() + 1e-10)

        if avg_change <= tol and k > 5:
            break
        
    # --- 3. Output ---
    w = z
    obj_final = (-Q1 @ w + q).clamp(min=0.0).sum().item()    
    Logs = {
        "iterations": k+1,
        "absolute_primal_residual": avg_change,
        "relative_dual_residual": dual_res,
        "epsilon": epsilon.item() if torch.is_tensor(epsilon) else epsilon,
        "CP objective rel. improvement": (obj_init - obj_final) / (obj_init + 1e-12),
        "CP tau": tau,
        "CP reduced_factor": reduced_factor
    }
    if constraint in ('ellipsoid','both'):
        R_final = torch.dot(w, Q2 @ w).item()
        Logs["ratio solution over max_energy"] = R_final / max_energy
    if torch.isnan(w).any():
        print("Warning: NaNs detected in w. Fallback to identity scales.")
        w = torch.zeros_like(w)

    if obj_final >= obj_init:
        w.zero_()
    return w.to(orig_dtype), Logs
