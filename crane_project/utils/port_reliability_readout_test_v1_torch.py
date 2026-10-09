"""Frozen ROI -> frozen stem. No detector, quality training, or GT input."""
import numpy as np
import torch

from crane_project.utils import port_reliability_readout_test_v1 as core
from crane_project.utils import port_reliability_redc_size_v1 as features
from crane_project.utils import port_reliability_redc_size_v1_torch as native


def validate_record(record, b, m, scale_xy):
    if set(record) != set(core.INPUT_NAMES):
        raise ValueError('Cache must contain only the six GT-free tensor fields')
    n = 0 if m is None else 1
    if (b is None) != (m is None):
        raise ValueError('Frozen B/M output presence differs')
    shapes = dict(roi=(n,256,9,9),support=(n,1,9,9),boxes_original=(n,6),
                  boxes_model=(n,5),scale_xy=(n,2),midpoint_original=(n,6))
    if any(not isinstance(v,torch.Tensor) or tuple(v.shape)!=shapes[k]
           or v.requires_grad or not bool(torch.isfinite(v).all()) for k,v in record.items()):
        raise ValueError('Cache shape, finite/detached contract differs')
    if n:
        if record['boxes_original'][0].tolist()!=b or record['midpoint_original'][0].tolist()!=m:
            raise ValueError('Cached B/M differs from reviewed delivered output')
        if record['scale_xy'][0].tolist()!=scale_xy or min(scale_xy)<=0:
            raise ValueError('Actual sx/sy restoration differs')
    if bool(((record['support'] < -1e-6) | (record['support'] > 1+1e-6)).any()):
        raise ValueError('Support values outside frozen contract')


def extract(head, records, online_inventory, device):
    # Inventory deliberately excludes all GT, offline errors and quality labels.
    allowed = {'image','b','midpoint','image_size','scale_xy'}
    if len(records)!=len(online_inventory) or any(set(r)!=allowed for r in online_inventory):
        raise ValueError('Online inventory contains labels or has missing frames')
    if head.training or any(p.requires_grad for p in head.parameters()):
        raise ValueError('Frozen eval head required')
    for record,r in zip(records,online_inventory):
        validate_record(record,r['b'],r['midpoint'],r['scale_xy'])
    before = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
    extracted = native.freeze_features(head,records,device)
    x=[];ids=[];maximum=np.zeros(6)
    for r,(pooled,decoded) in zip(online_inventory,extracted):
        replay=decoded['boxes_original'][0].detach().cpu().tolist() if len(decoded['boxes_original']) else None
        maximum=np.maximum(maximum,core.replay_box(r['midpoint'],replay))
        if r['midpoint'] is None:
            if pooled is not None:raise ValueError('Missing frame has a quality feature')
        else:
            x.append(features.descriptor(r['midpoint'],r['image_size'],pooled));ids.append(r['image'])
    if any(p.grad is not None for p in head.parameters()) or any(not torch.equal(v,head.state_dict()[k].cpu()) for k,v in before.items()):
        raise ValueError('Frozen head changed')
    return np.asarray(x).reshape(-1,291),ids,dict(maximum_native_replay_difference=maximum.tolist(),
        frames=len(records),outputs=len(ids),head_updates=0,new_detector_inferences=0,GT_online=False,
        stored_M_never_replaced=True)
