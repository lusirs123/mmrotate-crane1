"""Candidate-to-image direction evidence and an isolated angle-only diagnostic.

Online evidence takes only image-predicted line probabilities, candidate boxes
and observed geometry. Annotation templates are supplied by the OFFLINE caller.
No detection, depth, center/size quality or candidate ordering is changed here.
"""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn
from crane_project.utils import port_reliability_candidate_fit_v1 as candidate
from crane_project.utils.port_structure_reliability_v1 import (
    SETTINGS, canonical_boxes, checked_meta)

FEATURE_NAMES = ('cos_2_relative_angle', 'sin_2_relative_angle', 'line_anisotropy',
                 'finite_line_soft_iou', 'line_excess_peak', 'valid_context_fraction',
                 'direction_defined')


def atomic_json(path, value):
    encoded=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    def save(stream): stream.write(encoded)
    atomic_file(path,save)


def atomic_file(path,save):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='direction-cache-',suffix='.tmp',dir=str(path.parent))
    try:
        with os.fdopen(fd,'wb') as stream:
            save(stream); stream.flush(); os.fsync(stream.fileno())
        os.link(temp,str(path))  # exclusive publication; never overwrite evidence
    finally:
        os.unlink(temp)


def direction_features(line_probability, boxes_model, meta):
    """Seven bounded features, pi-periodic, original feature-grid convention.

    Full-grid moments use excess above local median in the existing isotropic
    square context. Undefined/flat evidence is encoded as zeros plus a flag,
    never assigned a trustworthy angle. Soft min/max overlap uses a candidate
    finite-axis Gaussian of the ORIGINAL 8-model-pixel sigma, not a GT axis.
    """
    checked_meta(meta)
    a=np.asarray(line_probability,dtype=np.float64)
    if a.ndim!=2 or not np.isfinite(a).all() or a.min()<0 or a.max()>1:
        raise ValueError('Expected a finite 2D image line-probability map in [0,1]')
    tensor=torch.as_tensor(boxes_model,dtype=torch.float64).detach().cpu()
    b=canonical_boxes(tensor).numpy()
    h,w=a.shape; ph,pw=meta['pad_shape'][:2]
    if (h*SETTINGS['stride'],w*SETTINGS['stride'])!=(ph,pw):
        raise ValueError('Line map does not match padded P3 stride geometry')
    y,x=np.mgrid[:h,:w]; points=np.stack((x,y),axis=-1)*SETTINGS['stride']
    valid=(points[...,0]<meta['img_shape'][1])&(points[...,1]<meta['img_shape'][0])
    result=[]; details=[]
    sigma=SETTINGS['stride']*SETTINGS['response_sigma_cells']
    for box in b:
        side=max(float(box[2])*SETTINGS['context_side_long_multiple'],
                 SETTINGS['stride']*SETTINGS['context_min_side_cells'])
        square=(np.abs(points[...,0]-box[0])<=side/2)&(np.abs(points[...,1]-box[1])<=side/2)
        region=square&valid; count=int(region.sum())
        # Denominator is the infinite feature lattice in this same square,
        # including cells outside the image/canvas; cropped contexts stay marked.
        nx=max(0,math.floor((box[0]+side/2)/SETTINGS['stride'])-math.ceil((box[0]-side/2)/SETTINGS['stride'])+1)
        ny=max(0,math.floor((box[1]+side/2)/SETTINGS['stride'])-math.ceil((box[1]-side/2)/SETTINGS['stride'])+1)
        support=count/max(nx*ny,1)
        info=dict(valid_cells=count,valid_context_fraction=support,direction_defined=False,
                  reference_angle_rad=None,weighted_center_model_px=None,anisotropy=0.,soft_iou=0.)
        values=[0.,0.,0.,0.,0.,support,0.]
        if count:
            p=points[region]; weights=np.maximum(a[region]-np.median(a[region]),0.)
            mass=float(weights.sum()); peak=float(weights.max())
            values[4]=peak
            if mass>1e-12 and peak>1e-12:
                center=(weights[:,None]*p).sum(axis=0)/mass
                delta=p-center; cov=(delta*weights[:,None]).T.dot(delta)/mass
                aa=float(cov[0,0]-cov[1,1]); bb=float(2*cov[0,1]); norm=math.hypot(aa,bb)
                trace=float(np.trace(cov)); anisotropy=norm/max(trace,1e-12)
                u=np.array([math.cos(box[4]),math.sin(box[4])])
                offset=p-box[:2]; along=np.clip(offset.dot(u),-box[2]/2,box[2]/2)
                dist_sq=((offset-along[:,None]*u)**2).sum(axis=1)
                template=np.exp(-.5*dist_sq/sigma**2)
                observed=weights/peak
                overlap=float(np.minimum(observed,template).sum()/max(np.maximum(observed,template).sum(),1e-12))
                info.update(weighted_center_model_px=center.tolist(),anisotropy=anisotropy,soft_iou=overlap)
                values[2]=anisotropy;values[3]=overlap
                if norm>1e-9 and trace>1e-9:
                    c,s=aa/norm,bb/norm; theta=box[4]
                    # cos/sin(2*(candidate - image reference)) avoid signed PCA eigenvectors.
                    values[0]=math.cos(2*theta)*c+math.sin(2*theta)*s
                    values[1]=math.sin(2*theta)*c-math.cos(2*theta)*s
                    values[6]=1.
                    info.update(direction_defined=True,reference_angle_rad=.5*math.atan2(bb,aa))
        result.append(values);details.append(info)
    features=np.asarray(result,dtype=np.float32).reshape(len(b),len(FEATURE_NAMES))
    if not np.isfinite(features).all():
        raise ValueError('Nonfinite explicit direction features')
    return torch.from_numpy(features),details


def analytic_direction_scores(features):
    """Offline diagnostic using original 3deg reference, not a calibrated gate."""
    result=[]
    for values in features.tolist():
        if values[6]!=1.:
            result.append(None)
        else:
            error=abs(.5*math.atan2(values[1],values[0]))*180/math.pi
            result.append(math.exp(-error/SETTINGS['angle_reference_deg']))
    return result


class AngleOnlyQuality(nn.Module):
    """Keep ep08 center/size logits frozen; fit only copied angle MLP.

    All controls have identical parameter dimensions. Extra columns initially
    zero, so every arm starts with the SAME ep08 score for every candidate.
    """
    def __init__(self, original_quality):
        super().__init__()
        self.base=deepcopy(original_quality).cpu().eval()
        for p in self.base.parameters(): p.requires_grad_(False)
        first,last=self.base[0],self.base[-1]
        self.raw_dim=first.in_features
        self.angle=nn.Sequential(nn.Linear(self.raw_dim+len(FEATURE_NAMES),first.out_features),
                                 nn.ReLU(inplace=False),nn.Linear(first.out_features,1))
        with torch.no_grad():
            self.angle[0].weight.zero_()
            self.angle[0].weight[:,:self.raw_dim].copy_(first.weight)
            self.angle[0].bias.copy_(first.bias)
            self.angle[-1].weight.copy_(last.weight[2:3])
            self.angle[-1].bias.copy_(last.bias[2:3])

    def forward(self, descriptor):
        if descriptor.ndim!=2 or descriptor.shape[1]!=self.raw_dim+len(FEATURE_NAMES):
            raise ValueError('Direction descriptor shape differs')
        with torch.no_grad(): center_size=self.base(descriptor[:,:self.raw_dim])[:,:2]
        return torch.cat((center_size,self.angle(descriptor)),dim=1)


def additional_metrics(model,rows,assessment):
    contribution=[]; pairs=assessment['pairs']; both={}; examples=[]
    with torch.no_grad():
        for row in rows:
            q=model(row['descriptor']).sigmoid();t=row['target'];m=row['mask'];n=row['genuine_count']
            loss=torch.nn.functional.smooth_l1_loss(q[:,2],t[:,2],reduction='none')*m[:,2]
            direction=.5*loss[:n].sum()/m[:n].sum().clamp(min=1)+.5*loss[n:].sum()/m[n:].sum().clamp(min=1)
            contribution.append(float(direction))
            examples.append(dict(image=row['image'],domain=row['domain'],gt=float(q[n,2]),
                minus_5_deg=float(q[n+11,2]),plus_5_deg=float(q[n+12,2]),
                saturated_all_three_high=bool((q[[n,n+11,n+12],2]>=1-1e-6).all())))
    for domain in ('all','real','sim'):
        selected=[p for p in pairs if p['group']=='angle' and (domain=='all' or p['domain']==domain)]
        images={p['image'] for p in selected}
        both[domain]=dict(eligible_views=len(images),both_signs_correct=sum(
            all(p['correct_sign'] for p in selected if p['image']==image) for image in images))
    assessment.update(direction_loss_contribution=sum(contribution)/len(contribution),
                      angle_both_signs=both,angle_examples=examples)
    return assessment


def fit(initial,rows,steps=2000,lr=.001,milestones=(100,500,1000,2000),notify=None):
    model=deepcopy(initial).cpu()
    original_state=deepcopy(model.base.state_dict())
    optimizer=torch.optim.Adam(model.angle.parameters(),lr=lr,weight_decay=0)
    def assessment():
        return additional_metrics(model,rows,candidate.assess(model,rows))
    before=assessment();trajectory=[]; clipped=0;max_norm=0.
    for step in range(1,steps+1):
        optimizer.zero_grad(set_to_none=True)
        # Original three-component quality loss; center/size are constant.
        loss=candidate.objective(model,rows)
        if not torch.isfinite(loss): raise ValueError('Nonfinite direction loss')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.angle.parameters(),10.)
        if not torch.isfinite(norm): raise ValueError('Nonfinite direction gradient')
        clipped+=int(norm>10);max_norm=max(max_norm,float(norm))
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.angle.parameters()):
            raise ValueError('Nonfinite direction parameters')
        if step in milestones or step==steps:
            item=dict(step=step,pre_update_loss=float(loss.detach()),preclip_gradient_norm=float(norm),post_update=assessment())
            trajectory.append(item)
            if notify is not None: notify(item)
    final=assessment()
    if any(not torch.equal(value,model.base.state_dict()[key]) for key,value in original_state.items()):
        raise ValueError('Frozen original quality MLP changed')
    if any(p.requires_grad or p.grad is not None for p in model.base.parameters()):
        raise ValueError('Frozen original quality MLP received gradients')
    for b,a in zip(before['views'],final['views']):
        if not np.array_equal(np.asarray(b['qualities'])[:,:2],np.asarray(a['qualities'])[:,:2]):
            raise ValueError('Center/size quality changed')
    return dict(before=before,trajectory=trajectory,final=final,optimizer_steps=steps,
                clipped_steps=clipped,max_preclip_gradient_norm=max_norm,center_size_quality_exact=True),model
