"""Opt-in U/S/G experiment. Historical modules and their defaults are untouched."""
import hashlib
import json
import base64
import pickle
from pathlib import Path

import torch
from mmcv.runner import HOOKS, Hook
from mmrotate.models.builder import ROTATED_HEADS, ROTATED_LOSSES
from mmrotate.models.dense_heads.sym_eood_head import SymEOODHead
from mmrotate.models.losses.sym_nfl_loss import SymNFLLoss


def matched_objectives(U, G, floor=1e-12):
    if not bool(torch.isfinite(U)) or not bool(torch.isfinite(G)):
        raise FloatingPointError('Nonfinite U/G')
    if float(U.detach()) < 0 or float(G.detach()) < 0:
        raise FloatingPointError('Negative classification objective')
    degenerate = float(U.detach()) <= floor
    # Near zero, use U without a ratio. G must also be negligible under mu<=2.
    if degenerate:
        if float(G.detach()) > 2 * floor + 1e-15:
            raise FloatingPointError('Inconsistent near-zero G')
        alpha = U.detach().new_tensor(1.)
    else:
        alpha = (G / U).detach()
        if not 1 - 1e-5 <= float(alpha) <= 2 + 1e-5:
            raise FloatingPointError('Geometry multiplier outside [1,2]')
    return alpha, alpha * U, degenerate


@ROTATED_LOSSES.register_module()
class SymNFLUSGV1(SymNFLLoss):
    def __init__(self, objective='U', **kwargs):
        assert objective in ('U', 'S', 'G')
        kwargs['use_geometry_weight'] = objective != 'U'
        super().__init__(**kwargs)
        assert not self.isolate_images and not self.use_quality_target
        self.objective = objective
        self.records = []

    def forward(self, pred_logits, labels, pred_bboxes, gt_bboxes, **kwargs):
        native = super().forward(pred_logits, labels, pred_bboxes, gt_bboxes, **kwargs)
        original = self.use_geometry_weight
        self.use_geometry_weight = not original
        alternative = dict(kwargs, advance_iter=False)
        try:
            counterfactual = super().forward(pred_logits, labels, pred_bboxes, gt_bboxes, **alternative)
        finally:
            self.use_geometry_weight = original
        U, G = (counterfactual, native) if original else (native, counterfactual)
        self.records.append(dict(U=U, G=G, logits=pred_logits, labels=labels,
                                 weight=kwargs.get('weight'), pred=pred_bboxes,
                                 gt=gt_bboxes, avg_factor=kwargs.get('avg_factor'),
                                 tau=self._get_current_tau()))
        return U if self.objective in ('U', 'S') else G


@ROTATED_HEADS.register_module()
class SymEOODUSGHeadV1(SymEOODHead):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture_probe = False
        self.probe = None
        self.usg_stats = None

    def loss(self, *args, **kwargs):
        self.loss_cls.records = []
        try:
            result = super().loss(*args, **kwargs)
            if result is None:
                raise RuntimeError('No classification targets')
            records = self.loss_cls.records
            if len(records) != 5:
                raise RuntimeError('Expected exactly five main-head loss calls')
            U = sum(r['U'] for r in records)
            G = sum(r['G'] for r in records)
            alpha, S, degenerate = matched_objectives(U, G)
            if self.loss_cls.objective == 'S':
                result['loss_cls'] = [alpha * r['U'] for r in records]
            self.usg_stats = dict(U=float(U.detach()), G=float(G.detach()),
                                  S=float(S.detach()), alpha=float(alpha),
                                  degenerate=degenerate,
                                  tau=[r['tau'] for r in records],
                                  positive=[int((r['labels'] == 0).sum()) for r in records])
            if self.capture_probe:
                self.probe = dict(U=U, G=G, S=S, records=records,
                                  other=sum(sum(v) if isinstance(v, (tuple, list)) else v
                                            for k, v in result.items() if 'loss' in k and k != 'loss_cls'))
            return result
        finally:
            self.loss_cls.records = []


def tensor_digest(value):
    h = hashlib.sha256()
    for name, tensor in value:
        x = tensor.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(x.dtype).encode()); h.update(str(tuple(x.shape)).encode())
        h.update(x.numpy().tobytes())
    return h.hexdigest()


def vector_metrics(a, b):
    aa = bb = ab = 0.
    for x, y in zip(a, b):
        if x is not None:
            aa += float(x.detach().double().square().sum())
        if y is not None:
            bb += float(y.detach().double().square().sum())
        if x is not None and y is not None:
            ab += float((x.detach().double() * y.detach().double()).sum())
    return dict(norm_G=aa ** .5, norm_S=bb ** .5,
                cosine=ab / (aa * bb) ** .5 if aa and bb else None,
                ratio=(aa / bb) ** .5 if bb else None)


def append_json(path, data):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(data, allow_nan=False) + '\n')


def rng_state():
    import random
    import numpy as np
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())


def restore_rng(state):
    import random
    import numpy as np
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu()); torch.cuda.set_rng_state(state['cuda'].cpu())


def encode_rng(state):
    # MMCV logs runner.meta as JSON before custom before_run hooks execute.
    return base64.b64encode(pickle.dumps(state, protocol=4)).decode('ascii')


def decode_rng(payload):
    return pickle.loads(base64.b64decode(payload.encode('ascii')))


@HOOKS.register_module()
class USGContractHookV1(Hook):
    """Input identity, rank-specific recovery, and state-preserving probes.

    Recovery is supported only at completed epoch boundaries, after eval hooks.
    Worker RNG is seeded by the existing nonpersistent worker_init_fn.
    """
    def __init__(self, probe_paths, probe_epochs=(1, 12, 24)):
        self.probe_paths = probe_paths
        self.probe_epochs = tuple(probe_epochs)

    def before_run(self, runner):
        import torch.distributed as dist
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world = dist.get_world_size() if dist.is_initialized() else 1
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        self.model = model
        self.head = model.bbox_head
        self.resume_state = (runner.meta or {}).get('usg_recovery')
        if self.resume_state is not None:
            assert self.resume_state['world_size'] == self.world
            saved = self.resume_state['ranks'][self.rank]
            self.head.assigner._local_call_count = saved['assigner_calls']
            assert int(self.head.loss_cls._local_iter) == saved['loss_calls']
            self.pending_rng = decode_rng(saved['rng'])
        else:
            assert runner.iter == 0, 'Only complete USG recovery checkpoints can resume'
            self.pending_rng = None
        self.input_hash = hashlib.sha256()
        init = tensor_digest(model.named_parameters())
        append_json(Path(runner.work_dir) / ('identity_rank%d.jsonl' % self.rank),
                    dict(initial_parameters_sha256=init, seed=runner.meta['seed'],
                         objective=self.head.loss_cls.objective, resumed=bool(self.resume_state)))

    def before_train_epoch(self, runner):
        assert runner.data_loader.num_workers == 2
        assert not runner.data_loader.persistent_workers
        if self.pending_rng is not None:
            restore_rng(self.pending_rng)
            self.pending_rng = None
        self.input_hash = hashlib.sha256()

    def before_train_iter(self, runner):
        data = runner.data_batch
        # DataContainers hold CPU batch tensors before the DDP scatter.
        image = data['img'].data[0]
        payload = [('img', image)]
        for key in ('gt_bboxes', 'gt_labels'):
            payload.extend((key + str(i), t) for i, t in enumerate(data[key].data[0]))
        names = [m['filename'] for m in data['img_metas'].data[0]]
        self.input_hash.update(json.dumps(names).encode())
        self.input_hash.update(tensor_digest(payload).encode())

    def after_train_iter(self, runner):
        if not bool(torch.isfinite(runner.outputs['loss'])):
            raise FloatingPointError('Nonfinite native total loss')
        if runner.iter % 50 == 0:
            append_json(Path(runner.work_dir) / ('strength_rank%d.jsonl' % self.rank),
                        dict(epoch=runner.epoch + 1, iteration=runner.iter,
                             assigner_calls=self.head.assigner._local_call_count,
                             loss_calls=int(self.head.loss_cls._local_iter), **self.head.usg_stats))

    def after_train_epoch(self, runner):
        import torch.distributed as dist
        epoch = runner.epoch + 1
        append_json(Path(runner.work_dir) / ('inputs_rank%d.jsonl' % self.rank),
                    dict(epoch=epoch, sha256=self.input_hash.hexdigest(), batches=len(runner.data_loader)))
        if epoch in self.probe_epochs:
            self.run_probes(runner)
        # This hook runs LOWEST, after both checkpoint and validation hooks.
        state = dict(assigner_calls=self.head.assigner._local_call_count,
                     loss_calls=int(self.head.loss_cls._local_iter), rng=encode_rng(rng_state()))
        states = [None] * self.world
        if self.world > 1:
            dist.all_gather_object(states, state)
        else:
            states = [state]
        if self.rank == 0:
            runner.meta['usg_recovery'] = dict(world_size=self.world, ranks=states,
                                               epoch=epoch, iteration=runner.iter)
            # All epoch files are retained, no deletion, latest points only inside this run.
            runner.save_checkpoint(runner.work_dir, filename_tmpl='recovery_epoch_{}.pth', create_symlink=False)
        if self.world > 1:
            dist.barrier()

    def run_probes(self, runner):
        from mmcv.parallel import scatter
        saved_rng = rng_state()
        buffers = {n: b.detach().clone() for n, b in self.model.named_buffers()}
        calls = self.head.assigner._local_call_count
        modes = {n: m.training for n, m in self.model.named_modules()}
        before = tensor_digest(self.model.named_parameters())
        self.model.train()
        for m in self.model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()
        self.head.capture_probe = True
        features = []
        handle = self.model.neck.register_forward_hook(lambda m, i, o: features.extend(o))
        try:
            for i, path in enumerate(self.probe_paths):
                # Reuse frozen TRAIN augmented batches, never a live data-loader draw.
                data = torch.load(path, map_location='cpu')
                def move(x):
                    if torch.is_tensor(x):return x.cuda()
                    if isinstance(x, dict):return {k: move(v) for k, v in x.items()}
                    if isinstance(x, list):return [move(v) for v in x]
                    if isinstance(x, tuple):return tuple(move(v) for v in x)
                    return x
                features.clear()
                self.model.forward_train(**move(data))
                p = self.head.probe
                pars = [(n, t) for n, t in self.model.named_parameters() if t.requires_grad]
                logits = [r['logits'] for r in p['records']]
                targets = [t for n, t in pars] + features + logits
                grads = {k: torch.autograd.grad(p[k], targets, retain_graph=True, allow_unused=True) for k in ('U', 'G', 'S')}
                other = torch.autograd.grad(p['other'], [t for n, t in pars], allow_unused=True)
                n = len(pars)
                spaces = dict(logits=list(range(n + len(features), len(targets))),
                              cls_head=[j for j, (name, _) in enumerate(pars) if name.startswith('bbox_head.retina_cls.')],
                              FPN_parameters=[j for j, (name, _) in enumerate(pars) if name.startswith('neck.')])
                row = dict(epoch=runner.epoch + 1, probe=i, input=path,
                           objective=self.head.loss_cls.objective, **self.head.usg_stats,
                           gradient={s: dict(
                               G_vs_S=vector_metrics([grads['G'][j] for j in ids], [grads['S'][j] for j in ids]),
                               G_vs_U=vector_metrics([grads['G'][j] for j in ids], [grads['U'][j] for j in ids]),
                               S_vs_U=vector_metrics([grads['S'][j] for j in ids], [grads['U'][j] for j in ids])) for s, ids in spaces.items()},
                           total={}, levels=[])
                for k in grads:
                    combined = [y if x is None else x if y is None else x + y for x, y in zip(grads[k][:n], other)]
                    norm = vector_metrics(combined, combined)['norm_G']
                    row['total'][k] = dict(preclip_norm=norm, clip_coefficient=min(1., 10 / (norm + 1e-6)))
                for r in p['records']:
                    with torch.no_grad():
                        pred, gt = r['pred'], r['gt']
                        top = torch.zeros(len(pred), dtype=torch.bool, device=pred.device)
                        mu = pred.new_ones(len(pred))
                        if len(gt):
                            dist2 = ((pred[:, None, :2] - gt[None, :, :2]) ** 2).sum(-1).min(1)[0]
                            idx = dist2.topk(min(self.head.loss_cls.spatial_topk, len(pred)), largest=False)[1]
                            kld = self.head.loss_cls._min_sym_kld(pred[idx], gt)
                            mu[idx] = 1 + torch.exp(-(kld / r['tau']).clamp(-20, 20))
                            top[idx] = True
                        valid = r['weight'] > 0
                        pos = (r['labels'] == 0) & valid
                        neg = (r['labels'] != 0) & valid
                        # Same native Focal ingredients, used for logging only.
                        z = r['logits']; target = (r['labels'] == 0).float()[:, None]
                        sig = z.sigmoid(); pt = (1-sig)*target+sig*(1-target)
                        focal = torch.nn.functional.binary_cross_entropy_with_logits(z, target, reduction='none')
                        focal = focal * (.25*target+.75*(1-target))*pt.pow(2)
                        focal = focal[:, 0] * r['weight'] * self.head.loss_cls.loss_weight / r['avg_factor']
                        row['levels'].append(dict(positive=int(pos.sum()), negative=int(neg.sum()),
                            ignored=int((~valid).sum()), positive_topk=int((pos & top).sum()),
                            negative_topk=int((neg & top).sum()), mu_topk=[float(v) for v in mu[top].cpu()],
                            positive_increment=float((focal[pos]*(mu[pos]-1)).sum()),
                            negative_increment=float((focal[neg]*(mu[neg]-1)).sum()), tau=r['tau']))
                append_json(Path(runner.work_dir) / ('probes_rank%d.jsonl' % self.rank), row)
                self.head.probe = None
                del p, grads, other, targets, pars, logits, combined, row, r, features[:]
                # Counters and buffers are reset before every probe to the same epoch boundary.
                with torch.no_grad():
                    for name, b in self.model.named_buffers():b.copy_(buffers[name])
                self.head.assigner._local_call_count = calls
        finally:
            handle.remove(); self.head.capture_probe = False; self.head.probe = None
            self.head.loss_cls.records = []
            with torch.no_grad():
                for name, b in self.model.named_buffers():b.copy_(buffers[name])
            self.head.assigner._local_call_count = calls
            for name, m in self.model.named_modules():m.training = modes[name]
            restore_rng(saved_rng)
        assert tensor_digest(self.model.named_parameters()) == before
        assert all(torch.equal(b, buffers[n]) for n, b in self.model.named_buffers())
