"""S038: cached 2D area reindexing. No GT in read/forward; V stays sealed."""
import torch
from torch.nn import functional as F
from crane_project.utils import port_size_local_vectors_v1 as v

residual = v.residual
ARMS = ('fixed_b', 'q_aligned')
PARAMETER_COUNT = v.PARAMETER_COUNT


def read_pair(roi, support, b, bm, xy, q):
    b, cb, xy, sides, rotation, q = residual.checked_inputs(roi, support, b, bm, xy, q)
    n = len(b)
    u = v.grid(b).expand(n, -1, -1)
    # Column formula T_B, implemented with row vectors.
    original = (torch.bmm(u*sides[:, None], rotation.transpose(1, 2)) + cb[:, None, :2])/xy[:, None]
    angle = residual.original.wrap_pi(q[:, 4]-b[:, 4]) + b[:, 4]
    def rot(a):
        c,s = a.cos(),a.sin()
        return torch.stack((torch.stack((c,-s),-1),torch.stack((s,c),-1)),1)
    moved = torch.bmm(torch.bmm(original-b[:, None,:2], rot(b[:,4])) *
                      (q[:,2:4]/b[:,2:4])[:,None], rot(angle).transpose(1,2)) + q[:,None,:2]
    gq = torch.bmm(moved*xy[:,None]-cb[:,None,:2], rotation)/sides[:,None]
    grids = {'fixed_b':u, 'q_aligned':gq}
    features, masks, inside = {}, {}, {}
    for arm, g in grids.items():
        valid = (g.abs() <= .5+1e-5).all(-1).reshape(n,1,9,9)
        # Only arithmetic tolerance is clamped, genuine outside points are zero.
        grid = (2*g.clamp(-.5,.5)).reshape(n,9,9,2)
        if n:
            features[arm] = F.grid_sample(roi.detach(),grid,align_corners=True,padding_mode='zeros')*valid
            masks[arm] = F.grid_sample(support.detach(),grid,align_corners=True,padding_mode='zeros').clamp(0,1)*valid
        else:
            features[arm], masks[arm] = roi.detach(), support.detach()
        inside[arm] = valid
    w = torch.minimum(masks['fixed_b'],masks['q_aligned']).detach()
    for arm in ARMS:
        features[arm] = features[arm]*(w>0)
    return dict(features=features,support=w,source_support=masks,inside=inside,
                source_local=grids,logical=u,original_points={'fixed_b':original,'q_aligned':moved})


class AlignedSizeHead(v.LocalVectorSizeHead):
    def __init__(self, arm):
        if arm not in ARMS: raise ValueError('Unknown alignment arm')
        super().__init__()
        self.arm = arm

    def forward(self, roi, support, b, bm, xy, m):
        pair = read_pair(roi,support,b,bm,xy,m)
        out = super().forward(pair['features'][self.arm],pair['support'],b,bm,xy,m)
        out['common_support'] = pair['support']
        return out


def size_loss(output, gt, m, bm):
    if not bool(output['vector_valid'].all()):
        raise ValueError('Invalid local vectors/common support during training')
    return residual.size_loss(output['delta_roi'],gt,m,bm)


def cross_edge(rows, tolerance=1e-6):
    """Offline fixed-M subsets; retain ALL positive harm, not just intersection."""
    result={}
    for domain in ('overall','real','sim'):
        rr=[r for r in rows if (domain=='overall' or r['domain']==domain) and r['midpoint'] is not None and r['edge_residual'] is not None]
        before=[];after=[]
        for r in rr:
            gt=sorted(r['gt'][2:4],reverse=True)
            before.append([abs(x/y-1) for x,y in zip(sorted(r['midpoint'][2:4],reverse=True),gt)])
            after.append([abs(x/y-1) for x,y in zip(sorted(r['edge_residual'][2:4],reverse=True),gt)])
        before=torch.tensor(before,dtype=torch.float64).reshape(-1,2)
        after=torch.tensor(after,dtype=torch.float64).reshape(-1,2)
        change=after-before
        result[domain]=dict(outputs=len(rr),both_harmed=int((change>tolerance).all(-1).sum()),edges={})
        for j,name in enumerate(('long','short')):
            k=1-j; subset=(before[:,j]>.03)&(before[:,k]<=.03)
            good=change[:,j]<-tolerance; harm=change[:,k]>tolerance
            inter=subset&good&harm; denom=int(subset.sum())
            positive=torch.where(harm,change[:,k],torch.zeros_like(change[:,k]))
            result[domain]['edges'][name]=dict(denominator=denom,
                corrected_and_other_harmed=int(inter.sum()),
                intersection_other_harm_sum=float(positive[inter].sum()),
                intersection_other_harm_per_fixed_subset=float(positive[inter].sum())/denom if denom else None,
                all_other_positive_harm_sum_in_subset=float(positive[subset].sum()),
                all_other_positive_harm_sum=float(positive.sum()),
                both_harmed_in_subset=int((subset&(change>tolerance).all(-1)).sum()),
                beneficial_target_count=int((subset&good).sum()),
                beneficial_target_gain_sum=float((-change[:,j])[subset&good].sum()))
    return result
