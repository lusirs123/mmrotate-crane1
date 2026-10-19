"""S037 diagnostic adapter, not a new production head or training objective.

Validate natural M once; virtual Q never pretends to be a valid midpoint-head
output. Q has no GT/slot flag. Reuse the frozen V layers and original delivery
against natural M. Primary readout response is recorded before that guard.
"""
import math
import torch
from crane_project.utils import port_size_local_vectors_v1 as v

H = .0625
COUNT_EPS = 1e-6
SLOTS = ('natural', 'long_plus', 'long_minus', 'short_plus', 'short_minus')
VISUALS = ('normal', 'zero_visual_ablation', 'within_video_halfcycle')


def proposals(m):
    """Natural + four GT-free single-edge states, assignment fixed by M."""
    q = m.detach()[:, None].repeat(1, 5, 1)
    lr = (m[:, 2] < m[:, 3]).long()
    ids = torch.arange(len(m), device=m.device)
    for slot, edge, sign in ((1, lr, 1), (2, lr, -1), (3, 1-lr, 1), (4, 1-lr, -1)):
        q[ids, slot, 2+edge] *= math.exp(sign*H)
    safe = (q[:, :, 2] < q[:, :, 3]) == (m[:, 2] < m[:, 3])[:, None]
    return q, safe


def readout(head, roi, support, b, bm, xy, m, q):
    """Same frozen weights/evidence; only Q vector channels/base change.

    Current-frame input validation and final delivery validation are distinct.
    Axis crossings are marked unavailable by callers, never GT-rematched.
    """
    v.residual.checked_inputs(roi, support, b, bm, xy, m)
    if q.shape != m.shape or q.dtype != m.dtype or q.device != m.device:
        raise ValueError('Q shape/device/dtype')
    if not torch.isfinite(q).all() or (q[:, 2:4] <= 0).any():
        raise ValueError('Q must be finite and positive')
    if not torch.equal(q[:, [0, 1, 4, 5]], m[:, [0, 1, 4, 5]]):
        raise ValueError('Q changed center/angle/score')
    if not torch.equal(q[:, 2] < q[:, 3], m[:, 2] < m[:, 3]):
        raise ValueError('Q axis crossing: report unavailable, never relabel')
    n = len(m)
    base = v.encode(q, b, bm, xy).detach()
    coords = v.grid(b).expand(n, -1, -1).transpose(1, 2).reshape(n, 2, 9, 9)
    channels = base.reshape(n, 81, 8).transpose(1, 2).reshape(n, 8, 9, 9)
    z = head.stem(torch.cat((roi.detach(), support.detach(), coords, channels), 1))
    change = head.terminal(z).permute(0, 2, 3, 1).reshape(n, 81, 4, 2)
    decoded = v.decode(base+change, support, b, bm, xy)
    reference = v.decode(base, support, b, bm, xy)['sizes'].detach()
    canonical = (decoded['sizes'].clamp(min=1e-6)/reference.clamp(min=1e-6)).log()
    raw = torch.where((m[:, 2] < m[:, 3])[:, None], canonical.flip(-1), canonical)
    candidate = torch.cat((q[:, :2], q[:, 2:4]*raw.exp(), q[:, 4:]), -1)
    # Preserve official natural-M fallback; do not silently treat fallback as
    # a successful response to Q. Natural Q=M delta is exactly original V.
    total_raw = (q[:, 2:4]/m[:, 2:4]).log()+raw
    delta = v.residual.raw_roi_pair(total_raw, bm)
    delta = torch.where(decoded['valid'][:, None], delta, torch.full_like(delta, float('nan')))
    delivered = v.residual.deliver_sizes(delta, b, bm, xy, m)
    return dict(delivered, preguard_candidate=candidate,
                vector_valid=decoded['valid'], reference_vectors=base,
                vector_changes=change, q_delta_raw=raw)


def donor_indices(records):
    """Half-cycle in sorted image order within natural video/scale, no GT."""
    groups = {}
    for i, r in enumerate(records):
        groups.setdefault((r['sequence'], r['scale']), []).append(i)
    donors = {}
    for ids in groups.values():
        ids.sort(key=lambda i: records[i]['image'])
        for pos, i in enumerate(ids):
            donors[i] = ids[(pos+len(ids)//2) % len(ids)] if len(ids) > 1 else None
    return donors


def edge_evaluation(q, predicted, gt):
    """Offline GT evaluation only; near is protection, not mandatory zero."""
    before, after = q/gt-1., predicted/gt-1.
    delta = (predicted-q)/gt
    near = abs(before) <= .03
    group = 'near' if near else 'need_enlarge' if before < 0 else 'need_shrink'
    moved = abs(delta) > COUNT_EPS
    return dict(before_error=before, after_error=after, abs_error_delta=abs(after)-abs(before),
                update_gt_normalized=delta, required_group=group, hold=not moved,
                direction_correct=None if near else bool(moved and delta*before < 0),
                improved=abs(after) < abs(before)-COUNT_EPS,
                harmed=abs(after) > abs(before)+COUNT_EPS)
