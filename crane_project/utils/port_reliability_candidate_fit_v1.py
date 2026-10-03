"""In-sample candidate identifiability diagnostic; never a deployment checkpoint."""
from copy import deepcopy
import torch
from crane_project.utils.port_structure_reliability_v1 import quality_loss


def capture(arm, p3, boxes, scores, meta):
    captured = []
    handle = arm.quality[0].register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().cpu().clone()))
    try:
        with torch.no_grad():
            q = arm(p3, boxes, scores, meta)['qualities'].detach().cpu()
    finally:
        handle.remove()
    if len(captured) != 1 or not torch.isfinite(captured[0]).all():
        raise ValueError('Expected one finite quality descriptor')
    mlp = deepcopy(arm.quality).cpu().eval()
    with torch.no_grad():
        delta = (mlp(captured[0]).sigmoid()-q).abs().max().item()
    if delta > 1e-6:
        raise ValueError('Cached CPU MLP does not reproduce original quality')
    return captured[0], q, mlp, delta


def objective(model, rows):
    return torch.stack([quality_loss(model(r['descriptor']).sigmoid(), r['target'],
                                    r['mask'], r['genuine_count'])[0] for r in rows]).mean()


def assess(model, rows):
    details, pairs = [], []
    with torch.no_grad():
        for row in rows:
            q = model(row['descriptor']).sigmoid()
            t, m, n = row['target'], row['mask'], row['genuine_count']
            def mae(start, end):
                mask = m[start:end]
                return [float(((q[start:end]-t[start:end]).abs()*mask)[:, c].sum()/mask[:, c].sum())
                        if mask[:, c].sum() > 0 else None for c in range(3)]
            details.append(dict(image=row['image'], domain=row['domain'], qualities=q.tolist(),
                                targets=t.tolist(), masks=m.tolist(), genuine_mae=mae(0,n), probe_mae=mae(n,len(q))))
            for j, name in enumerate(row['probe_names'][1:], start=n+1):
                component = 0 if name.startswith('center_') else 2 if name.startswith('angle_') else 1
                if not (m[n, component] > 0 and m[j, component] > 0):
                    continue
                target_gap = float(t[n,component]-t[j,component])
                gap = float(q[n,component]-q[j,component])
                pairs.append(dict(image=row['image'], domain=row['domain'], probe=name,
                    group=name.rsplit('_',1)[0], component=component, target_gap=target_gap,
                    predicted_gap=gap, gap_residual=gap-target_gap, correct_sign=gap>1e-6,
                    target_gap_fraction=gap/target_gap if target_gap>1e-6 else None))
    summary = {}
    for domain in ('all','real','sim'):
        for group in sorted(set(p['group'] for p in pairs)):
            selected = [p for p in pairs if p['group']==group and (domain=='all' or p['domain']==domain)]
            if selected:
                summary[domain+':'+group] = dict(count=len(selected), correct_sign=sum(p['correct_sign'] for p in selected),
                    mean_predicted_gap=sum(p['predicted_gap'] for p in selected)/len(selected),
                    mean_absolute_gap_residual=sum(abs(p['gap_residual']) for p in selected)/len(selected),
                    mean_target_gap=sum(p['target_gap'] for p in selected)/len(selected))
    errors = {}
    for domain in ('all','real','sim'):
        selected = [r for r in details if domain=='all' or r['domain']==domain]
        if selected:
            errors[domain] = {}
            for kind in ('genuine_mae','probe_mae'):
                errors[domain][kind] = []
                for c in range(3):
                    values = [r[kind][c] for r in selected if r[kind][c] is not None]
                    errors[domain][kind].append(sum(values)/len(values) if values else None)
    return dict(loss=float(objective(model,rows).detach()), views=details, pairs=pairs,
                pair_summary=summary, mean_view_component_errors=errors)


def fit(initial, rows, steps=2000, lr=.001, milestones=(100,500,1000,2000)):
    model = deepcopy(initial).cpu().train()
    for p in model.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    before = assess(model,rows)
    trajectory = []
    clipped_steps, max_norm = 0, 0.
    for step in range(1,steps+1):
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model,rows)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite diagnostic loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
        if not torch.isfinite(norm):
            raise ValueError('Nonfinite diagnostic gradient')
        clipped_steps += int(norm > 10.)
        max_norm = max(max_norm, float(norm))
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError('Nonfinite diagnostic parameters')
        if step in milestones or step==steps:
            trajectory.append(dict(step=step, pre_update_loss=float(loss.detach()),
                preclip_gradient_norm=float(norm), post_update=assess(model,rows)))
    return dict(before=before, trajectory=trajectory, final=assess(model,rows), optimizer_steps=steps, clipped_steps=clipped_steps,
                max_preclip_gradient_norm=max_norm), model
