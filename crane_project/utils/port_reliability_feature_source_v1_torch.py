"""Observe actual native selection without changing the detector forward."""
from copy import deepcopy
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from crane_project.utils import port_reliability_redc_size_v1_torch as old
from crane_project.utils import port_reliability_feature_source_v1 as core

exported = old.exported
optimizer = old.optimizer
update = old.update


class ResidualRisk(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(258,16),nn.ReLU(),nn.Linear(16,8),nn.ReLU(),nn.Linear(8,1))
        nn.init.zeros_(self.network[4].weight); nn.init.zeros_(self.network[4].bias)
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, normalized, raw_logit):
        return -raw_logit+self.network(normalized[:,1:]).squeeze(-1)+self.beta[0]


def make_models(device):
    torch.manual_seed(1701)
    first = ResidualRisk().to(device)
    return dict(midpoint=first,native=deepcopy(first))


def trace_native(detector, image, metas):
    """No GT. Read actual decoder locals at return, including original topk.

    Python tracing observes the unmodified original method; masks below only
    reconstruct its index bookkeeping. No second topk or nearest-box match.
    """
    head = detector.bbox_head; convolution = head.retina_reg
    if (convolution.in_channels != 256 or convolution.kernel_size != (3,3)
            or convolution.padding != (1,1) or convolution.stride != (1,1)
            or convolution.dilation != (1,1) or convolution.groups != 1
            or head.cls_out_channels != 1 or not head.use_sigmoid_cls
            or head.test_cfg.get('max_per_img',1) != 1
            or getattr(head,'use_semantic_cls_adapter',False)
            or getattr(head,'use_score_context_modulation',False)):
        raise ValueError('Native plain single-class3x3 regression contract differs')
    if any(getattr(detector,k,None) is not None for k in ('pqa_head','reg_quality_head','aux_detach_cls_head','context_head')):
        raise ValueError('Auxiliary reranking/context route is outside this comparison')
    if any(getattr(head,k,None) is not None for k in ('candidate_selection','center_size_compensation','shape_compensation')):
        raise ValueError('Candidate/compensation route is outside the frozen B contract')
    if sys.gettrace() is not None: raise ValueError('An existing Python tracer cannot be overwritten')
    code = head._get_bboxes_single.__func__.__code__; calls = []; inputs = []; outputs = []
    def tracer(frame,event,arg):
        if frame.f_code != code: return None
        if event == 'return': calls.append((dict(frame.f_locals),arg))
        return tracer
    hook1 = convolution.register_forward_pre_hook(lambda module,args: inputs.append(args[0].detach()))
    hook2 = convolution.register_forward_hook(lambda module,args,out: outputs.append(out.detach()))
    try:
        sys.settrace(tracer)
        with torch.no_grad(): result = detector.simple_test(image,metas,rescale=True)
    finally:
        sys.settrace(None); hook1.remove(); hook2.remove()
    if len(calls) != 1 or len(inputs) != len(outputs): raise ValueError('Native single-frame call count differs')
    loc, decoded = calls[0]
    if len(inputs) != len(loc['cls_score_list']): raise ValueError('FPN/head invocation pairing differs')
    from mmrotate.models.dense_heads.sym_eood_head import rotated_anchor_center_inside_flags
    mapping = []; surviving_scores = []
    for level,(cls,anchors,feature) in enumerate(zip(loc['cls_score_list'],loc['mlvl_anchors'],inputs)):
        height,width = cls.shape[-2:]; per_cell = cls.shape[0]
        if feature.shape != (1,256,height,width) or convolution.out_channels != per_cell*5:
            raise ValueError('Anchor/cell/channel layout differs')
        scores = cls.permute(1,2,0).reshape(-1,1).sigmoid()
        indices = torch.arange(len(anchors),device=anchors.device)
        if head.filter_padding_anchors:
            mask = rotated_anchor_center_inside_flags(anchors,loc['img_shape'])
            indices,scores = indices[mask],scores[mask]
        if loc['cfg'].get('score_thr',0.)>0:
            mask = scores[:,0]>loc['cfg'].score_thr; indices,scores = indices[mask],scores[mask]
        for flat in indices.cpu().tolist():
            y,x = divmod(flat//per_cell,width)
            mapping.append((level,y,x,flat%per_cell,flat))
        surviving_scores.append(scores[:,0])
    if len(decoded[0]) == 0:
        if mapping: raise ValueError('Native missing selection has surviving anchors')
        return result,None,dict(present=False,levels=len(inputs),surviving=0)
    chosen = int(loc['topk_inds'][0]) if 'topk_inds' in loc else 0
    if len(mapping) != sum(len(s) for s in surviving_scores): raise ValueError('Selector index bookkeeping differs')
    level,y,x,anchor,flat = mapping[chosen]
    selected_scores = torch.cat(surviving_scores)
    if not torch.equal(selected_scores[chosen],decoded[0][0,5]): raise ValueError('Actual selected score does not replay')
    patch = F.pad(inputs[level],(1,1,1,1))[0,:,y:y+3,x:x+3].contiguous()
    replay = F.conv2d(patch[None],convolution.weight,convolution.bias)[0,:,0,0]
    actual = outputs[level][0,:,y,x]
    if not torch.allclose(replay,actual,rtol=1e-4,atol=1e-4): raise ValueError('Regression receptive field replay differs')
    # Scalar slicing reference is independent of Torch padding/layout.
    independent = core.patch_reference(inputs[level][0].cpu().numpy(),y,x)
    value = patch.reshape(2304).cpu().numpy()
    if not np.array_equal(value,independent): raise ValueError('Native2304 patch layout differs')
    return result,value,dict(present=True,level=level,y=y,x=x,anchor=anchor,flat_anchor_index=flat,
        selected_surviving_index=chosen,surviving=len(mapping),levels=len(inputs),
        feature_shape=list(inputs[level].shape),native_score=float(decoded[0][0,5]),
        tied_maximum=int((selected_scores==selected_scores.max()).sum()),
        actual_topk_observed=True,regression_max_replay_difference=float((replay-actual).abs().max()),
        raw_regression=actual[anchor*5:anchor*5+5].cpu().tolist(),
        scale_factor=np.asarray(metas[0]['scale_factor']).tolist(),GT_online=False)
