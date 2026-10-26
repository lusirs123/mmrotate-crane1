"""Opt-in MAL and S_MAL. Historical loss/head classes remain untouched."""
import hashlib
import json
from pathlib import Path

import torch
from mmcv.runner import HOOKS
from mmrotate.models.builder import ROTATED_HEADS, ROTATED_LOSSES
from mmrotate.models.losses.sym_nfl_loss import SymNFLLoss
from mmrotate.models.dense_heads.sym_eood_head import SymEOODHead
from mmrotate.core.bbox.iou_calculators.rotate_iou2d_calculator import rbbox_overlaps
from crane_project.utils.symeood_mal_math_v1 import mal_terms, match_strength
from crane_project.utils.symeood_usg_v1 import (
    USGContractHookV1, append_json, rng_state, restore_rng, tensor_digest, vector_metrics)


@ROTATED_LOSSES.register_module()
class SymEOODMALLossV1(SymNFLLoss):
    def __init__(self, objective='MAL', mal_gamma=1.5, mal_alpha=None, **kwargs):
        if objective not in ('U', 'MAL', 'S_MAL') or mal_gamma != 1.5 or mal_alpha is not None:
            raise ValueError('Frozen MAL contract differs')
        kwargs['use_geometry_weight'] = False
        super().__init__(**kwargs)
        if self.use_quality_target or self.isolate_images:
            raise ValueError('Legacy quality/image-isolation switches forbidden')
        self.objective, self.mal_gamma = objective, mal_gamma
        self.records = []

    def forward(self, pred_logits, labels, pred_bboxes, gt_bboxes,
                weight=None, avg_factor=None, **kwargs):
        # Exactly one native Focal call / counter advance. CF has no RNG/state.
        U = super().forward(pred_logits, labels, pred_bboxes, gt_bboxes,
                            weight=weight, avg_factor=avg_factor, **kwargs)
        positive = labels == 0
        if len(gt_bboxes) != int(positive.sum()):
            raise ValueError('GT must be aligned to existing positive indices')
        quality = pred_logits.new_zeros(len(labels))
        with torch.no_grad():
            if len(gt_bboxes):
                q = rbbox_overlaps(pred_bboxes[positive], gt_bboxes, is_aligned=True).reshape(-1)
                quality[positive] = q
        weights = torch.ones_like(quality) if weight is None else weight.reshape(-1)
        terms = mal_terms(pred_logits, labels, quality, weights, self.mal_gamma)
        M = terms.sum() / float(avg_factor) * self.loss_weight
        p = positive & (weights > 0)
        n = (~positive) & (weights > 0)
        # Use native U value/gradient; recomputed per-item terms are logging only.
        with torch.no_grad():
            target = positive.to(pred_logits)[:, None]
            prob = pred_logits.sigmoid()
            focal = torch.nn.functional.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
            focal *= (self.alpha*target+(1-self.alpha)*(1-target))*((1-prob)*target+prob*(1-target)).pow(self.gamma)
            ut = focal[:, 0]*weights*self.loss_weight/float(avg_factor)
        self.records.append(dict(U=U, M=M, logits=pred_logits, labels=labels,
            quality=quality, weight=weights, positive=p, negative=n,
            Upos=float(ut[p].sum()), Uneg=float(ut[n].sum()),
            Mpos=float(terms[p].detach().sum()*self.loss_weight/float(avg_factor)),
            Mneg=float(terms[n].detach().sum()*self.loss_weight/float(avg_factor))))
        return M if self.objective == 'MAL' else U


@ROTATED_HEADS.register_module()
class SymEOODMALHeadV1(SymEOODHead):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture_probe = False
        self.probe = None
        self.usg_stats = None

    def loss(self, *args, **kwargs):
        self.loss_cls.records = []
        try:
            result = super().loss(*args, **kwargs)
            rr = self.loss_cls.records
            if result is None or len(rr) != self.anchor_generator.num_levels:
                raise RuntimeError('Expected only main-head FPN classification calls')
            U, M = sum(r['U'] for r in rr), sum(r['M'] for r in rr)
            alpha, S, degenerate = match_strength(U, M)
            if self.loss_cls.objective == 'S_MAL':
                result['loss_cls'] = [alpha*r['U'] for r in rr]
            with torch.no_grad():
                q = torch.cat([r['quality'][r['positive']] for r in rr])
                p = torch.cat([r['logits'].detach().sigmoid()[:, 0][r['positive']] for r in rr])
                self.usg_stats = dict(U=float(U.detach()), M=float(M.detach()),
                    S=float(S.detach()), alpha=float(alpha), degenerate=degenerate,
                    Upos=sum(r['Upos'] for r in rr), Uneg=sum(r['Uneg'] for r in rr),
                    Mpos=sum(r['Mpos'] for r in rr), Mneg=sum(r['Mneg'] for r in rr),
                    positive=[int(r['positive'].sum()) for r in rr],
                    q=q.cpu().tolist(), target=q.pow(1.5).cpu().tolist(),
                    positive_score=p.cpu().tolist(), lowq_score=p[q < .5].cpu().tolist(),
                    negative_score_mean=[float(r['logits'].detach().sigmoid()[:, 0][r['negative']].mean())
                        if r['negative'].any() else None for r in rr])
            if self.capture_probe:
                self.probe = dict(U=U, M=M, S=S, records=rr,
                    other=sum(sum(v) if isinstance(v, (tuple, list)) else v
                        for k, v in result.items() if 'loss' in k and k != 'loss_cls'))
            return result
        finally:
            self.loss_cls.records = []


@HOOKS.register_module()
class MALContractHookV1(USGContractHookV1):
    """Reuse accepted identity/epoch recovery shell; MAL-specific probes below.

    Epoch-boundary recovery snapshots only; production queue never auto-resumes.
    All backward intermediates and probe gradients remain on GPU.
    """
    def before_run(self, runner):
        super().before_run(runner)
        from crane_project.utils.symeood_repro_trace_v1 import runtime
        actual=runtime()
        if actual['torch']!='1.13.1+cu117' or actual['mmcv']!='1.7.0' or actual['mmdet']!='2.25.1' or actual['cudnn']!=8500:
            raise RuntimeError('Frozen native runtime changed')
        append_json(Path(runner.work_dir)/('runtime_rank%d.jsonl'%self.rank),actual)
        append_json(Path(runner.work_dir)/('initial_state_rank%d.jsonl'%self.rank),
            dict(full_state_sha256=tensor_digest(self.model.state_dict().items())))
        if self.rank==0:
            path=Path(runner.work_dir)/'complete_initial_model.pth'
            if path.exists():raise RuntimeError('Initial snapshot exists')
            torch.save(dict(state_dict=self.model.state_dict(),rng=rng_state()),path)

    def after_train_iter(self, runner):
        super().after_train_iter(runner)
        append_json(Path(runner.work_dir)/('alpha_rank%d.jsonl'%self.rank),
            dict(iteration=runner.iter,epoch=runner.epoch+1,**self.head.usg_stats))
        # grad_norm is preclip total DDP gradient norm recorded by OptimizerHook.
        if runner.iter % 50 == 0:
            value = runner.log_buffer.val_history.get('grad_norm', [])
            norm = float(value[-1]) if value else None
            append_json(Path(runner.work_dir)/('clip_rank%d.jsonl' % self.rank),
                dict(iteration=runner.iter, epoch=runner.epoch+1, preclip_norm=norm,
                     clip_coefficient=min(1., 10/(norm+1e-6)) if norm is not None else None,
                     peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                     peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))

    def run_probes(self, runner):
        saved_rng = rng_state()
        buffers = {n:b.detach().clone() for n,b in self.model.named_buffers()}
        calls = self.head.assigner._local_call_count
        modes = {n:m.training for n,m in self.model.named_modules()}
        before = tensor_digest(self.model.named_parameters())
        self.head.capture_probe = True
        def move(x):
            if torch.is_tensor(x): return x.cuda()
            if isinstance(x,dict): return {k:move(v) for k,v in x.items()}
            if isinstance(x,(tuple,list)): return type(x)(move(v) for v in x)
            return x
        try:
            for index, path in enumerate(self.probe_paths):
                data = move(torch.load(path, map_location='cpu'))
                losses = self.model.forward_train(**data)
                probe = self.head.probe
                pars = [p for p in self.model.parameters() if p.requires_grad]
                logits = [r['logits'] for r in probe['records']]
                # Sequential gradients; no CPU saved-tensor hooks or recompute.
                gm = torch.autograd.grad(probe['M'], pars+logits, retain_graph=True, allow_unused=True)
                gs = torch.autograd.grad(probe['S'], pars+logits, retain_graph=True, allow_unused=True)
                row = dict(epoch=runner.epoch+1, probe=index, input=path,
                    objective=self.head.loss_cls.objective, **self.head.usg_stats,
                    gradients=dict(parameters=vector_metrics(gm[:len(pars)],gs[:len(pars)]),
                                   logits=vector_metrics(gm[len(pars):],gs[len(pars):])))
                del gm, gs
                total = sum(sum(v) if isinstance(v,(list,tuple)) else v
                            for k,v in losses.items() if 'loss' in k)
                gg = torch.autograd.grad(total, pars, allow_unused=True)
                norm = vector_metrics(gg,gg)['norm_G']
                row['total_gradient'] = dict(preclip_norm=norm,clip_coefficient=min(1.,10/(norm+1e-6)),
                                            scope='local fixed TRAIN probe, not DDP averaged')
                append_json(Path(runner.work_dir)/('probes_rank%d.jsonl'%self.rank),row)
                self.head.probe = None
                del gg, probe, losses, total, pars, logits, data
                with torch.no_grad():
                    for name,b in self.model.named_buffers(): b.copy_(buffers[name])
                self.head.assigner._local_call_count = calls
        finally:
            self.head.capture_probe = False
            self.head.probe = None
            self.head.loss_cls.records = []
            with torch.no_grad():
                for name,b in self.model.named_buffers(): b.copy_(buffers[name])
            self.head.assigner._local_call_count = calls
            for name,m in self.model.named_modules(): m.training = modes[name]
            restore_rng(saved_rng)
        if before != tensor_digest(self.model.named_parameters()):
            raise RuntimeError('Probe changed parameters')
        if not all(torch.equal(b,buffers[n]) for n,b in self.model.named_buffers()):
            raise RuntimeError('Probe changed buffers')
        append_json(Path(runner.work_dir)/('probe_conservation_rank%d.jsonl'%self.rank),
                    dict(epoch=runner.epoch+1,passed=True))
