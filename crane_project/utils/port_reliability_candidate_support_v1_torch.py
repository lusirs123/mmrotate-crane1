"""Observe native decoder return locals without replacing its selection."""
import sys
import time
import numpy as np
import torch


def trace_candidates(detector,image,metas):
    head=detector.bbox_head
    if head.cls_out_channels!=1 or not head.use_sigmoid_cls or head.test_cfg.get('max_per_img',1)!=1:
        raise ValueError('Frozen single-class native Top1 required')
    if sys.gettrace() is not None:raise ValueError('Existing tracer cannot be replaced')
    code=head._get_bboxes_single.__func__.__code__;calls=[]
    def tracer(frame,event,arg):
        if frame.f_code!=code:return None
        if event=='return':calls.append((dict(frame.f_locals),arg))
        return tracer
    torch.cuda.synchronize(image.device);start=time.monotonic()
    try:
        sys.settrace(tracer)
        with torch.no_grad():result=detector.simple_test(image,metas,rescale=True)
    finally:sys.settrace(None)
    torch.cuda.synchronize(image.device);forward=time.monotonic()-start
    if len(calls)!=1:raise ValueError('Exactly one native decoding call required')
    loc,decoded=calls[0]
    if len(decoded[0])==0:
        return result,np.empty((0,6)),None,dict(forward_seconds=forward,surviving=0,score_thr=float(loc['cfg'].score_thr))
    pool=loc['mlvl_bboxes'];scores=loc['mlvl_scores'][:,0]
    if not torch.is_tensor(pool) or len(pool)!=len(scores):raise ValueError('Native pool layout differs')
    chosen=int(loc['topk_inds'][0]) if 'topk_inds' in loc else 0
    if len(pool)>1 and 'topk_inds' not in loc:raise ValueError('Actual Top-K source unavailable')
    original=pool.detach().clone()
    # With a sole survivor, original decoder aliases and rescales the pool.
    if len(pool)>1:original[:,:4]/=original.new_tensor(metas[0]['scale_factor'])
    values=torch.cat((original,scores[:,None]),1).cpu().numpy().astype(float)
    native=decoded[0][0].detach().cpu().numpy().astype(float)
    if not np.array_equal(values[chosen],native):raise ValueError('Selected candidate source does not replay native B')
    if np.any(values[:,5]<=float(loc['cfg'].score_thr)):raise ValueError('Native score filter was changed')
    return result,values,chosen,dict(forward_seconds=forward,surviving=len(pool),
        selected_index=chosen,score_thr=float(loc['cfg'].score_thr),native_selection_observed=True,
        GT_online=False,detector_forward_calls=1)
