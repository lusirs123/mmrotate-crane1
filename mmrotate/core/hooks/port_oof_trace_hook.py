"""Opt-in, single-GPU TRAIN trace. Never changes a loss or optimizer update."""
import inspect
import hashlib
import json
import math
from pathlib import Path
import random
import types
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.runner import HOOKS, Hook
from crane_project.utils.port_reliability_oof_followup_v1 import TRACE, VERSION, anomaly,json_safe
from crane_project.utils.port_reliability_oof_training_v2 import MemoryPlateau


def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    return value


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
        torch=torch.get_rng_state().clone(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
        cuda=[s.cpu().clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.backends.cudnn.benchmark=state['cudnn_benchmark']
    torch.backends.cudnn.deterministic=state['cudnn_deterministic']
    if state['cuda']: torch.cuda.set_rng_state_all(state['cuda'])


def runtime_state(model):
    return dict(assigner_local_call_count=model.bbox_head.assigner._local_call_count,
        training_modes={n:m.training for n,m in model.named_modules()})


def restore_runtime(model, state):
    model.bbox_head.assigner._local_call_count=state['assigner_local_call_count']
    for name, module in model.named_modules(): module.training=state['training_modes'][name]


def sum_loss(value):
    if isinstance(value,(tuple,list)): return sum(v.mean() for v in value)
    return value.mean()


def gradient_summary(grads, named):
    buckets={k:[] for k in ('all','backbone','neck','main_cls','main_reg','ATSS')}
    channels=None
    for (name,_), grad in zip(named,grads):
        if grad is None: continue
        buckets['all'].append(grad.detach().norm())
        category = ('backbone' if name.startswith('backbone.') else 'neck' if name.startswith('neck.')
            else 'main_cls' if name.startswith('bbox_head.retina_cls.')
            else 'main_reg' if name.startswith('bbox_head.retina_reg.')
            else 'ATSS' if name.startswith('aux_heads.') else None)
        if category: buckets[category].append(grad.detach().norm())
        if name=='bbox_head.retina_reg.weight':
            # anchor x (dx,dy,dw,dh,da), never confuse encoded dh with canonical short side.
            channels=grad.detach().reshape(-1,5,*grad.shape[1:]).square().sum(dim=(0,2,3,4)).sqrt().cpu().tolist()
    result={k:float(torch.stack(v).norm()) if v else 0. for k,v in buckets.items()}
    result['encoded_reg_weight_channel_norms']=channels
    return result


class TraceStopped(RuntimeError):
    """Expected diagnostic boundary; not a successful full training run."""


@HOOKS.register_module()
class PortOOFTraceHook(Hook):
    def __init__(self, directory, settings=None):
        self.directory=Path(directory); self.settings=dict(TRACE if settings is None else settings)
        self.losses=None; self.before=None; self.pending=None
        self.previous=None
        self.targets=[]; self.classification=[]; self.batch=None
        self.probe_memory=MemoryPlateau()

    def before_run(self, runner):
        if self.settings!=TRACE: raise ValueError('Unregistered trace contract')
        if getattr(runner.model,'device_ids',[]) and len(runner.model.device_ids)!=1:
            raise ValueError('Trace requires a single GPU')
        self.directory.mkdir(parents=True,exist_ok=False)
        self.log=(self.directory/'updates.jsonl').open('x')
        self.model=runner.model.module if hasattr(runner.model,'module') else runner.model
        runner._port_oof_trace=self
        self.install(self.model)

    def install(self, model, capture_failure=True):
        """Inspect the actual forward once; no second target assignment or NFL call."""
        original_forward=model.forward_train
        forward_signature=inspect.signature(original_forward)
        def forward(_model,*args,**kwargs):
            bound=forward_signature.bind(*args,**kwargs); bound.apply_defaults()
            arguments=dict(bound.arguments)
            for key,parameter in forward_signature.parameters.items():
                if parameter.kind==inspect.Parameter.VAR_KEYWORD:
                    arguments.update(arguments.pop(key,{}))
            self.batch=cpu_tree(arguments)
            self.gts=bound.arguments['gt_bboxes']; self.metas=bound.arguments['img_metas']
            self.targets.clear(); self.classification.clear()
            try:
                value=original_forward(*args,**kwargs)
            except Exception as exc:
                if not capture_failure: raise
                if self.previous is not None:
                    torch.save(self.previous,str(self.directory/'before_previous_update.pt'))
                self.before['batch']=self.batch
                torch.save(self.before,str(self.directory/'before_anomaly.pt'))
                after=self.snapshot(self.runner); after.update(batch=self.batch,forward_error=repr(exc))
                torch.save(after,str(self.directory/'after_forward_before_update.pt'))
                self.finish('FORWARD_FAILURE_QUARANTINED_BEFORE_OPTIMIZER',self.runner)
                raise
            self.losses=value
            return value
        model.forward_train=types.MethodType(forward,model)

        head=model.bbox_head; original_single=head.loss_single
        signature=inspect.signature(original_single)
        def single(_head,*args,**kwargs):
            args_dict=signature.bind(*args,**kwargs).arguments
            with torch.no_grad():
                deltas=args_dict['bbox_pred']; batch=deltas.shape[0]
                labels=args_dict['labels'].reshape(batch,-1)
                anchors=args_dict['anchors'].reshape(-1,5)
                targets=head.bbox_coder.decode(anchors,args_dict['bbox_targets'].reshape(-1,5)).reshape(batch,-1,5)
                pred=head.bbox_coder.decode(anchors,deltas.permute(0,2,3,1).reshape(-1,5)).reshape(batch,-1,5)
                limit=float(args_dict.get('decode_max_size') or 2048.)
                for i in range(batch):
                    positive=(labels[i]>=0)&(labels[i]<head.num_classes)
                    if not positive.any(): continue
                    gt=self.gts[i]; t=targets[i,positive]; p=pred[i,positive]
                    if len(gt)!=1: raise ValueError('Trace assumes the reviewed one-GT TRAIN protocol')
                    edges=(t[:,2:4].sort(1)[0]-gt[0,2:4].sort()[0]).abs().max()
                    cerror=(t[:,:2]-gt[0,:2]).abs().max()
                    axis=t[:,4]+(t[:,2]<t[:,3]).to(t.dtype)*math.pi/2
                    gt_axis=gt[0,4]+(gt[0,2]<gt[0,3]).to(gt.dtype)*math.pi/2
                    angle=torch.remainder(axis-gt_axis+math.pi/2,math.pi).sub(math.pi/2).abs().max()
                    self.targets.append(dict(image=Path(self.metas[i]['filename']).stem,
                        positives=int(positive.sum()), encoded_GT_center_size_max_error=float(torch.maximum(edges,cerror)),
                        encoded_GT_periodic_angle_error=float(angle),
                        pre_clamp_positive_count=int(((p[:,:2]<0).any(1)|(p[:,:2]>limit-1).any(1)|
                            (p[:,2:4]<1).any(1)|(p[:,2:4]>limit).any(1)).sum()),
                        encoded_delta_abs_max=deltas.permute(0,2,3,1).reshape(batch,-1,5)[i,positive].abs().amax(0).cpu().tolist()))
            return original_single(*args,**kwargs)
        head.loss_single=types.MethodType(single,head)

        nfl=head.loss_cls; original_spatial=nfl._image_spatial_weights; original_cls=nfl.forward
        cls_signature=inspect.signature(original_cls); self.spatial=None
        def spatial(_loss,*args,**kwargs):
            result=original_spatial(*args,**kwargs); self.spatial=result
            return result
        nfl._image_spatial_weights=types.MethodType(spatial,nfl)
        def classification(_loss,*args,**kwargs):
            self.spatial=None
            actual=original_cls(*args,**kwargs)
            bound=cls_signature.bind(*args,**kwargs); bound.apply_defaults(); v=bound.arguments
            with torch.no_grad():
                logits=v['pred_logits']; labels=v['labels']; mu,quality=self.spatial
                target=F.one_hot(labels,num_classes=logits.shape[1]+1)[:,:-1].float()
                probability=logits.sigmoid()
                if nfl.use_quality_target:
                    target=target*quality.unsqueeze(-1)
                    modulating=(target-probability).abs().pow(nfl.gamma)
                else: modulating=((1-probability)*target+probability*(1-target)).pow(nfl.gamma)
                per=F.binary_cross_entropy_with_logits(logits,target,reduction='none')*modulating*\
                    (nfl.alpha*target+(1-nfl.alpha)*(1-target))*mu.unsqueeze(-1)
                if v['weight'] is not None: per=per*v['weight'].reshape_as(per)
                if (v['reduction_override'] or nfl.reduction)!='mean' or v['avg_factor'] is None:
                    raise ValueError('Unexpected SymNFL reduction contract')
                per=per*nfl.loss_weight/v['avg_factor']
                pos=(labels>=0)&(labels<logits.shape[1])
                self.classification.append(dict(positive=float(per[pos].sum()),negative=float(per[~pos].sum()),
                    reconstructed=float(per.sum()),actual=float(actual.detach()),tau=float(nfl._get_current_tau())))
                if not torch.allclose(per.sum(),actual.detach(),atol=1e-5,rtol=1e-4):
                    raise ValueError('Diagnostic positive/negative decomposition differs from actual SymNFL')
            self.spatial=None
            return actual
        nfl.forward=types.MethodType(classification,nfl)

    def snapshot(self, runner):
        return dict(protocol=VERSION, diagnostic_only=True, iteration=runner.iter,
            model=cpu_tree(self.model.state_dict()), optimizer=cpu_tree(runner.optimizer.state_dict()),
            rng=rng_state(), runtime=runtime_state(self.model))

    def before_train_iter(self,runner):
        self.runner=runner
        if runner.iter>=self.settings['max_updates']:
            self.finish('WINDOW_LIMIT_REACHED_NOT_ROOT_CAUSE',runner)
            raise TraceStopped('Bounded TRAIN window completed; no automatic continuation')
        self.losses=None; self.batch=None
        # One CPU snapshot is retained, replaced each step. No GPU history or per-step disk weights.
        self.before=self.snapshot(runner)

    def after_train_iter(self,runner):
        losses=self.losses
        terms=dict(main_cls=sum_loss(losses['loss_cls']),main_reg=sum_loss(losses['loss_bbox']))
        auxiliary=[sum_loss(v) for k,v in losses.items() if 'loss' in k and k not in ('loss_cls','loss_bbox')]
        terms['ATSS']=sum(auxiliary) if auxiliary else terms['main_cls']*0.
        named=[(n,p) for n,p in self.model.named_parameters() if p.requires_grad]
        parameters=[p for _,p in named]
        actual=runner.outputs['loss']
        norms=gradient_summary(torch.autograd.grad(actual,parameters,retain_graph=True,allow_unused=True),named)
        values={k:float(v.detach()) for k,v in terms.items()}
        reasons=anomaly(values,norms['all'],self.settings)
        components={}; shared={}
        if (reasons or runner.iter<self.probe_memory.settings['warmup_steps'] or
                runner.iter%self.settings['component_interval']==0 or runner.iter>=self.settings['dense_start']):
            for name,term in terms.items():
                grads=torch.autograd.grad(term,parameters,retain_graph=True,allow_unused=True)
                components[name]=gradient_summary(grads,named)
                if name in ('main_cls','main_reg'):
                    shared[name]=torch.cat([(g.detach().flatten().cpu() if g is not None else torch.zeros(p.numel()))
                        for (n,p),g in zip(named,grads) if n.startswith(('backbone.','neck.'))])
                del grads
        cosine=None
        if shared:
            a,b=shared['main_cls'],shared['main_reg']; denominator=float(a.norm()*b.norm())
            cosine=float(torch.dot(a,b))/denominator if denominator else None
        self.pending=dict(step=runner.iter+1,epoch=runner.epoch+1,losses=values,
            total_loss=float(actual.detach()),lr=[g['lr'] for g in runner.optimizer.param_groups],
            training_statistics={k:float(sum_loss(v).detach()) for k,v in losses.items() if 'loss' not in k},
            gradient_before_clip=norms['all'],gradient_modules=norms,component_gradients=components,
            shared_cls_reg_cosine=cosine,classification=cpu_tree(self.classification),targets=cpu_tree(self.targets),
            images=[Path(m['filename']).stem for m in self.metas],metadata=self.batch['img_metas'],
            assigner_calls_before=self.before['runtime']['assigner_local_call_count'],
            assigner_calls_after=self.model.bbox_head.assigner._local_call_count,
            nfl_local_iter=int(self.model.bbox_head.loss_cls._local_iter),anomaly=reasons,optimizer_updated=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            try:
                self.pending['probe_memory']=self.probe_memory.observe(torch.cuda.memory_allocated(),torch.cuda.memory_reserved())
            except RuntimeError as exc:
                reasons.append('probe_memory_guard'); self.pending['memory_error']=str(exc)
        # Metadata may contain arrays; normalize it when logging, not when saving exact batch.
        if reasons:
            if self.previous is not None:
                torch.save(self.previous,str(self.directory/'before_previous_update.pt'))
            self.before['batch']=self.batch
            torch.save(self.before,str(self.directory/'before_anomaly.pt'))
            after=self.snapshot(runner); after['batch']=self.batch; after['trace']=self.pending
            torch.save(after,str(self.directory/'after_forward_before_update.pt'))
            self.record(); self.finish('ANOMALY_QUARANTINED_BEFORE_OPTIMIZER',runner)
            raise TraceStopped('Finite/nonfinite TRAIN anomaly: '+','.join(reasons))
        self.before['batch']=self.batch
        self.previous=self.before
        self.losses=None; self.gts=None; self.before=None; self.batch=None

    def record(self):
        def encode(v):
            if isinstance(v,np.ndarray): return v.tolist()
            if isinstance(v,np.generic): return v.item()
            if isinstance(v,torch.Tensor): return v.detach().cpu().tolist()
            raise TypeError(type(v).__name__)
        self.log.write(json.dumps(json_safe(self.pending),default=encode,allow_nan=False)+'\n'); self.log.flush()
        self.pending=None

    def finish(self,status,runner):
        pins={}
        for name in ('before_anomaly.pt','before_previous_update.pt','after_forward_before_update.pt'):
            path=self.directory/name
            if path.exists():
                digest=hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
                pins[name]=digest.hexdigest()
        with (self.directory/'completion.json').open('x') as f:
            json.dump(dict(protocol=VERSION,status=status,completed_updates=runner.iter,
                snapshot_sha256=pins,
                VAL_scored=False,TEST_read=False,diagnostic_weights_usable_as_OOF=False,
                snapshot_boundary='before forward and after forward, both before anomalous optimizer update',
                replay='saved transformed batch; not DataLoader-prefetch or complete future-run replay'),f,indent=2)
        self.log.close()


@HOOKS.register_module()
class PortOOFTraceCommitHook(Hook):
    """LOW priority: prove the ordinary strict optimizer actually completed."""
    def after_train_iter(self,runner):
        trace=runner._port_oof_trace
        if trace.pending is None: raise ValueError('Missing pre-update trace')
        grads=[p.grad for p in trace.model.parameters() if p.grad is not None]
        norm=float(torch.stack([g.detach().norm() for g in grads]).norm())
        trace.pending.update(optimizer_updated=True,gradient_after_clip=norm,
            expected_clip_fraction=min(1.,10./(trace.pending['gradient_before_clip']+1e-6)),
            allocated_bytes=torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            reserved_bytes=torch.cuda.memory_reserved() if torch.cuda.is_available() else 0)
        trace.previous['trace']=cpu_tree(trace.pending)
        trace.record()
