"""TRAIN-only positive supply; current-weight features, original negatives.

Only an anchor lacking a native-batch label-positive may receive one external
positive. Primary batch/size loss/negative pool/inference are unchanged.
"""
import math
import torch
from torch.nn import functional as F
from crane_project.utils import port_size_pair_conr_v1 as c


def labels(records):
    tt=[]
    for r in records:
        t=c.model.residual.size_targets(r['gt_original'],r['midpoint_original'],r['boxes_model'])['log_residual']
        tt.append(c.canonical_pair(t,r['boxes_model'],r['midpoint_original'])[0])
    return torch.stack(tt).detach().cpu()


class PositiveIndex:
    """Metadata-only, immutable TRAIN search, independently computed view labels."""
    def __init__(self, records):
        if any(r['split']=='val' or not len(r['midpoint_original']) for r in records):
            raise ValueError('Present TRAIN views only')
        self.records=records;self.targets=labels(records);self.candidates=[]
        images=[r['image'] for r in records]
        for start in range(0,len(records),64):
            distance=(self.targets[start:start+64,None]-self.targets[None,:]).abs().amax(-1)
            for offset,dd in enumerate(distance):
                i=start+offset;r=records[i]
                valid=(dd<=c.WINDOW).nonzero().flatten().tolist()
                valid=[j for j in valid if images[j]!=r['image'] and
                    (records[j]['sequence']!=r['sequence'] or abs(records[j]['frame_id']-r['frame_id'])>30)]
                valid.sort(key=lambda j:(records[j]['sequence']==r['sequence'],float(dd[j]),j))
                self.candidates.append(valid)
        self.by_key={(r['image'],r['scale']):i for i,r in enumerate(records)}
        if len(self.by_key)!=len(records):raise ValueError('Duplicate TRAIN view identity')

    def select(self, indices):
        t=self.targets[indices];n=len(indices)
        native=(t[:,None]-t[None,:]).abs().amax(-1)<=c.WINDOW
        native.fill_diagonal_(False)
        main_images={self.records[i]['image'] for i in indices};keys=[];links=[];roles=[]
        for a,i in enumerate(indices):
            if bool(native[a].any()):continue
            mate=next((j for j in self.candidates[i] if self.records[j]['image'] not in main_images),None)
            if mate is None:continue
            if mate not in keys:keys.append(mate)
            links.append((a,keys.index(mate)))
            roles.append('cross_video' if self.records[mate]['sequence']!=self.records[i]['sequence'] else 'same_video_gap30')
        return dict(partner_indices=keys,links=links,roles=roles,missing_native=int((~native.any(1)).sum()),anchors=n)


def loss(main, target, prediction, partner, partner_target, links):
    """Rectangular anchor-to-key loss; external keys are designated positives only.

    No external negatives or additional anchor losses; preserves original N
    denominator and skips anchors without negatives. Both current feature paths
    receive gradients. Empty partner set calls the original implementation.
    """
    if partner is None or not len(partner['training_features']):
        return c.contrastive_loss(main['training_features'],main['common_support'],target,prediction)
    n=len(target)
    z=F.normalize((main['training_features']*main['common_support'].detach().sqrt()).flatten(1),dim=1,eps=1e-6)
    zz=F.normalize((partner['training_features']*partner['common_support'].detach().sqrt()).flatten(1),dim=1,eps=1e-6)
    dt=(target.detach()[:,None]-target.detach()[None,:]).abs().amax(-1)/c.WINDOW
    dp=(prediction.detach()[:,None]-prediction.detach()[None,:]).abs().amax(-1)/c.WINDOW
    other=~torch.eye(n,device=z.device,dtype=torch.bool)
    native_pos=(dt<=1)&other;native_neg=(dt>1)&(dp<=1)&other
    extra=torch.zeros(n,len(zz),dtype=torch.bool,device=z.device)
    for a,b in links:
        if bool(native_pos[a].any()) or bool(extra[a].any()):raise ValueError('Only one positive for a missing-positive anchor')
        if float((target[a].detach()-partner_target[b].detach()).abs().max())>c.WINDOW:raise ValueError('Invalid external positive label')
        extra[a,b]=True
    positive=torch.cat((native_pos,extra),1);negative=torch.cat((native_neg,torch.zeros_like(extra)),1)
    similarity=z@torch.cat((z,zz),0).T/c.TEMPERATURE
    distance=torch.cat((dt,torch.zeros_like(extra,dtype=dt.dtype)),1)
    active=positive.any(1)&negative.any(1)
    stats=dict(anchors=n,active_anchors=int(active.sum()),positive_pairs=int(positive.sum()),negative_pairs=int(negative.sum()),
        zero_embeddings=int(((main['training_features']*main['common_support'].detach().sqrt()).flatten(1).norm(dim=1)<=1e-6).sum()),external_pairs=len(links),external_views=len(zz))
    if not bool(active.any()):return (z.sum()+zz.sum())*0.,stats
    logits=similarity+torch.where(negative,c.PUSHING*distance,torch.zeros_like(distance))
    denominator=torch.logsumexp(logits[active].masked_fill(~(positive|negative)[active],-torch.inf),1)
    positive_mean=(similarity[active]*positive[active]).sum(1)/positive[active].sum(1)
    return (denominator-positive_mean).sum()/n,stats
