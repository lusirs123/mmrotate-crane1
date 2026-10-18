"""Common point readout for cached versus native frozen P3. No GT online.

Unlike historical I, BOTH arms process features AFTER sampling. This common
readout is not checkpoint compatible with I and needs two fresh controls.
Native points are restricted to the SAME B context, with the SAME cached
support as the control. No learned positions, expanded context or new loss.
"""
import torch
from torch import nn
from torch.nn import functional as F
from crane_project.utils import port_geometry_size_distribution_v1 as old
from crane_project.utils.port_structure_reliability_v1 import checked_meta

VERSION = 'port_size_native_p3_v1'
PARAMETER_COUNT = 2328
ARMS = ('cached_p3', 'native_p3')


def read_points(roi, support, b, bm, xy, m, arm, p3=None, meta=None):
    """Return N,256,132,9 samples and identical A/B masks/local coordinates.

    Point order: model-B canonical axes, +/-, 33 bins, 9 tangent points.
    Original raw M axes generate points; actual sx/sy maps POINTS to model.
    Feature cell i corresponds to model pixel 8*i (not 8*(i+.5)).
    """
    if arm not in ARMS:
        raise ValueError('Unknown arm')
    b, cb, xy, sides, rotation, m = old.residual.checked_inputs(roi, support, b, bm, xy, m)
    n, k, p = len(b), old.SETTINGS['bins'], old.SETTINGS['tangent_samples']
    points = old.border_points(m, bm, old.bucket_grid(m))
    model_points = points.reshape(n, 4*k*p, 2)*xy[:, None]
    local = torch.bmm(model_points-cb[:, None, :2], rotation)/sides[:, None]
    local = local.reshape(n, 4*k, p, 2)
    inside = (local.abs() <= .5).all(-1)[:, None]
    if not n:
        samples, masks = roi.new_empty(0, 256, 4*k, p), support.new_empty(0, 1, 4*k, p)
    else:
        masks = F.grid_sample(support.detach(), 2*local, align_corners=True).clamp(0, 1)*inside
        if arm == 'cached_p3':
            samples = F.grid_sample(roi.detach(), 2*local, align_corners=True)
        else:
            if (p3 is None or meta is None or p3.ndim != 4 or p3.shape[:2] != (1, 256)
                    or p3.dtype != roi.dtype or p3.device != roi.device
                    or p3.requires_grad or p3.grad_fn is not None):
                raise ValueError('Detached batch1 256-channel native P3 required')
            checked_meta(meta)
            h, w = p3.shape[-2:]
            yy, xx = torch.meshgrid(torch.arange(h, device=p3.device), torch.arange(w, device=p3.device))
            valid = ((xx*8 < meta['img_shape'][1]) & (yy*8 < meta['img_shape'][0])).to(p3.dtype)[None, None]
            grid = (2*(model_points/8+.5)/p3.new_tensor([w, h])-1).reshape(n, 4*k, p, 2)
            samples = F.grid_sample((p3*valid).expand(n, -1, -1, -1), grid, align_corners=False)
        # Native is never allowed additional positions beyond the old context.
        samples = samples*inside.to(samples.dtype)
    mu, bu = (old.residual.raw_roi_pair(x[:, 2:4], bm) for x in (m, b))
    geometry = torch.cat(((mu/bu).log(), (cb[:, 2:4]/sides).log()), -1)
    return dict(samples=samples.detach(), masks=masks.detach(),
                local=local.permute(0, 3, 1, 2).detach(), geometry=geometry.detach(),
                model_points=model_points.detach(), inside=inside.detach())


class CommonPointSizeHead(nn.Module):
    def __init__(self):
        super().__init__()
        # Tangent-only spatial processing, independently per axis/side/bin.
        self.stem = nn.Sequential(nn.Conv2d(259, 8, 1), nn.ReLU(),
                                  nn.Conv2d(8, 8, (1, 3), padding=(0, 1)), nn.ReLU())
        self.u, self.v = nn.Linear(23, 1), nn.Linear(23, 1)
        for layer in (self.u, self.v):
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)
        assert sum(p.numel() for p in self.parameters()) == PARAMETER_COUNT

    def forward(self, read, b, bm, xy, m):
        n, k = len(b), old.SETTINGS['bins']
        grid = old.bucket_grid(m)
        if n:
            mask = read['masks'].detach()
            x = torch.cat((read['samples'].detach(), mask, read['local'].detach()), 1)
            feature = self.stem(x)
            pooled = (feature*mask).max(-1).values
            paired = pooled.reshape(n, 8, 2, 2, k).permute(0, 2, 4, 3, 1).reshape(n, 2, k, 16)
            coverage = mask.mean(-1).reshape(n, 2, 2, k).permute(0, 1, 3, 2)
            geom = read['geometry'][:, None, None]*grid[None, None, :, None]
            inputs = torch.cat((paired, geom.expand(n, 2, k, 4), coverage,
                                grid[None, None, :, None].expand(n, 2, k, 1)), -1)
            logits = torch.stack((self.u(inputs[:, 0]).squeeze(-1), self.v(inputs[:, 1]).squeeze(-1)), 1)
        else:
            logits, coverage = b.new_empty(0, 2, k), b.new_empty(0, 2, k, 2)
        decoded = old.decode(logits, grid)
        decoded['delta_roi'] = torch.where(torch.isfinite(logits).reshape(n, 2*k).all(-1)[:, None],
            decoded['delta_roi'], torch.full_like(decoded['delta_roi'], float('nan')))
        delivery = old.residual.deliver_sizes(decoded['delta_roi'], b, bm, xy, m)
        return dict(delivery, **decoded, bucket_logits=logits, grid=grid, border_support=coverage)


def protocol():
    return dict(protocol=VERSION, status='DESIGN_AND_ENGINEERING_ONLY_NO_TRAINING',
        arms=list(ARMS), parameter_count=PARAMETER_COUNT,
        scope='Same B context1.5x with16px floor, same M positions, cached support and common post-sampling point/tangent stem. No expansion/multiscale/learned offsets.',
        difference='Only raw feature sampling source in the common architecture. Historical I convolution-before-sampling is NOT this control.',
        tensors='P3[1,256,H,W]; ROI[N,256,9,9]; samples[N,256,132,9]; common support[N,1,132,9]; coords[N,2,132,9]; geometry[N,4]; logits[N,2,33].',
        labels='Offline original log(GT/M) assigned once to frozen M raw axes then model-B ROI axes. No GT in sampling or forward.',
        training=dict(images=2558, views=5116, present_views=5112, scales=[1.,.5], epochs=24,
            batch_size=8, updates_per_epoch=639, optimizer='Adam', lr=.001, weight_decay=0.,
            clip_norm=10., seed=1703, shuffle='Python Random(1703+epoch), original scale/image sorted present views',
            initialization='Both fresh identical state; terminal zero. No old checkpoint reuse.',
            loss='Unweighted SmoothL1 beta0.1 over all2N residuals; natural M only; no TRAIN admission gate.'),
        evaluation='Every epoch full887 natural VAL, identical opportunity; no TEST/depth method selection. Overall/Real/Sim/video custom/AP/3geometry/coverage and paired sizes.',
        ranking='Single lexicographic rule per arm: (Sim long MAE, Sim short MAE, Real long MAE, Real short MAE, epoch); select best eligible by this key, else same-key descriptive only.',
        eligibility=dict(direct='Sim both MAE<M-1e-6; Real both MAE<=M+0.001 relative units (0.1pp). B and I07/C20/P07 gaps disclosed.',
            joint='Sim meanabs(a),meanabs(r)<=M+1e-6 log units.',
            protection='Fixed-M near<=3% each edge each domain MAE<=M+0.001. M output/center/raw and canonical angle/score exact. Per-domain DFR not higher; MCML max/mean not higher; TDR numerator not lower with same denominator.',
            detection='Full-frame mean_RIoU overall>=M-1e-5 and each domain>=M-0.001. If no increase>1e-5 overall, call size contribution with detection held, not localization gain.',
            mechanism='Native chosen eligible vs cached chosen eligible (else descriptive control by same key): both Sim MAE lower>1e-6, Real both<=control+0.001, overall RIoU>=control-1e-5. Do not infer capacity/feature insufficiency from historical I comparison.',
            diagnostics_only='P95,bias,Real a/r,near-both,cross-edge intersections,joint10 transitions,AP,per-video changes; not additional gates.'),
        efficiency='Online increment<=5ms mean and<=128MiB allocator peak over common cached head on same warmed batch1 measurement; full process<2048MiB. Point cache float32 for both arms, no full P3 disk cache.',
        numerical='MAE relative not percent; epsilon1e-6; RIoU epsilon1e-5. Near change counts1e-12 diagnostic. No later tolerance tuning.',
        stop='Invalid engineering or resource cap stops. One paired24-epoch run only after separate implementation/execution authorization. No added levels/rounds on failure. Multiscale design only.',
        depth='Common detection/depth adoption NOT yet available: retain data roles, freeze legal development protocol/threshold before future fitting; independent validation missing; Unknown02 descriptive only, Raw-opt fixed.',
        limits=dict(training_updates=0, test_reads=0, depth_reads=0, automatic_promotion=False))
