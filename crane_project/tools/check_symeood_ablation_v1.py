#!/usr/bin/env python3
"""S4x ablation engineering acceptance (no training, no TEST, no dataset).

Verification ladder -- these are NOT full-model equivalence:
  L1 component : assigner / two losses / gradients (synthetic tensors)
  L2 head      : real SymEOODHead forward(feats) -> loss -> backward -> SGD step
  L3 full model: NOT performed here (backbone/FPN/data/training loop absent)

Historical reference sources come from --hist-dir and MUST match the blobs of
--hist-commit in this git repository; "differs from current" alone is not
accepted as historical identity.

Exit code 0 only if every required check passes. No check is deleted, skipped
or tolerance-relaxed to make a run pass.
"""
import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GROUPS = ('000', '100', '010', '001', '110', '101', '011', '111')
A0 = ('000', '010', '001', '011')
A1 = ('100', '110', '101', '111')
WHITELIST = ('work_dir', 'model.bbox_head.loss_cls.use_geometry_weight',
             'model.bbox_head.loss_bbox', 'model.train_cfg.assigner.cost_reg_mode')
ALPHA, GAMMA = 0.25, 2.0
LR, MOM = 0.01, 0.9
HIST_PATHS = ('mmrotate/core/bbox/assigners/sym_pola.py',
              'mmrotate/models/losses/sym_nfl_loss.py',
              'mmrotate/models/losses/sym_kld_loss.py')
HIST_OPERATOR = 'mmrotate/models/losses/sym_kld_calculator.py'
CHECKS = []


def record(section, name, ok, detail=None, required=True):
    CHECKS.append(dict(section=section, name=name, ok=bool(ok),
                       required=bool(required), detail=detail))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_blob(repo, commit, rel):
    r = subprocess.run(['git', '-C', str(repo), 'show', f'{commit}:{rel}'], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def flatten(cfg, prefix=''):
    out = {}
    for k, v in cfg.items():
        name = f'{prefix}.{k}' if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, name))
        else:
            out[name] = v
    return out


def load_hist(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def cls_info(cls):
    src = inspect.getsourcefile(cls)
    return dict(path=str(src), sha256=sha(src), module=cls.__module__,
                qualname=cls.__qualname__, id=hex(id(cls)))


# ------------------------------------------------------------------ synthetic data
def batch(torch, device, seed=1703, n_pos=12, n_neg=52, label_weight=1.0,
          gt=True, positives=True):
    g = torch.Generator().manual_seed(seed)
    n = n_pos + n_neg
    gt_box = torch.tensor([[120., 118., 96., 48., .30],
                           [300., 260., 64., 40., -.60],
                           [420., 180., 50., 50., 1.10]])
    gt_box = gt_box.to(device) if gt else torch.zeros(0, 5, device=device)
    centers = torch.rand(n, 2, generator=g) * 380. + 20.
    sizes = torch.rand(n, 2, generator=g) * 90. + 12.
    ang = (torch.rand(n, 1, generator=g) - .5) * 1.4
    labels = torch.full((n,), 1, dtype=torch.long, device=device)
    if positives:
        labels[:n_pos] = 0
    bw = torch.zeros(n, 5, device=device)
    if positives:
        bw[:n_pos] = 1.0
    return dict(pred=torch.cat((centers, sizes, ang), 1).to(device),
                logits=((torch.rand(n, 1, generator=g) - .5) * 2.).to(device),
                labels=labels, weights=torch.full((n,), float(label_weight), device=device),
                gt=gt_box, bbox_w=bw,
                gt_idx=torch.arange(n_pos, device=device) % max(len(gt_box), 1),
                avg_factor=float(max(int(bw[:, 0].sum()), 1)))


def run_assign(assigner, b, torch, target_iter):
    """AssignResult exposes gt_inds (the constructor arg is assigned_gt_inds)."""
    assigner._local_call_count = target_iter * 2      # assign() increments by 1
    r = assigner.assign(b['logits'].clone(), b['pred'].clone(),
                        torch.zeros(len(b['gt']), dtype=torch.long, device=b['pred'].device),
                        b['gt'].clone(), None, is_training=True)
    return dict(gt_inds=r.gt_inds.clone(),
                labels=None if r.labels is None else r.labels.clone(),
                num_gts=int(r.num_gts),
                max_overlaps=None if r.max_overlaps is None else r.max_overlaps.clone())


def same_assign(a, b, torch):
    if not torch.equal(a['gt_inds'], b['gt_inds']):
        return False, 'gt_inds'
    if (a['labels'] is None) != (b['labels'] is None):
        return False, 'labels presence'
    if a['labels'] is not None and not torch.equal(a['labels'], b['labels']):
        return False, 'labels'
    if a['num_gts'] != b['num_gts']:
        return False, 'num_gts'
    if (a['max_overlaps'] is None) != (b['max_overlaps'] is None):
        return False, 'max_overlaps presence'
    if a['max_overlaps'] is not None and not torch.equal(a['max_overlaps'], b['max_overlaps']):
        return False, 'max_overlaps'
    return True, 'ok'


def run_losses(loss_cls, loss_bbox, b, torch, train_iter=0):
    """Both losses + gradients + ONE SGD UPDATE OF THE LOGITS TENSOR ONLY."""
    if hasattr(loss_cls, '_local_iter'):
        loss_cls._local_iter.fill_(train_iter * 5)     # forward increments by 1
    theta = b['logits'].clone().requires_grad_(True)
    pred = b['pred'].clone().requires_grad_(True)
    lc = loss_cls(theta, b['labels'], pred_bboxes=pred, gt_bboxes=b['gt'],
                  weight=b['weights'], avg_factor=b['avg_factor'])
    pos = b['bbox_w'][:, 0] > 0
    if int(pos.sum()) and len(b['gt']):
        lb = loss_bbox(pred[pos], b['gt'][b['gt_idx']], b['bbox_w'][pos][:, 0],
                       avg_factor=b['avg_factor'])
    else:
        lb = pred.sum() * 0.0
    (lc + lb).backward()
    with torch.no_grad():
        theta -= LR * theta.grad
    return dict(cls=float(lc), bbox=float(lb), grad_logits=theta.grad.clone(),
                grad_pred=pred.grad.clone(), updated=theta.detach().clone(),
                local_iter=int(loss_cls._local_iter) if hasattr(loss_cls, '_local_iter') else -1)


def run_bbox(loss_bbox, b, torch):
    pred = b['pred'].clone().requires_grad_(True)
    pos = b['bbox_w'][:, 0] > 0
    lb = loss_bbox(pred[pos], b['gt'][b['gt_idx']], b['bbox_w'][pos][:, 0],
                   avg_factor=b['avg_factor'])
    lb.backward()
    return float(lb), pred.grad.clone()


def focal_reference(b, torch):
    p = torch.sigmoid(b['logits'])
    target = torch.nn.functional.one_hot(b['labels'], num_classes=2)[:, :-1].float()
    pt = (1 - p) * target + p * (1 - target)
    fw = (ALPHA * target + (1 - ALPHA) * (1 - target)) * pt ** GAMMA
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        b['logits'], target, reduction='none') * fw
    loss = loss * b['weights'].view(-1, 1)
    return float(loss.sum() / b['avg_factor']) * 0.25


# ------------------------------------------------------------------ head helpers
def head_feats(torch, device, img=1024, batch_size=2, channels=256):
    """FPN-like features whose spatial sizes follow strides 8..128."""
    g = torch.Generator().manual_seed(11)
    sizes = [(img // 8, img // 8), (img // 16, img // 16), (img // 32, img // 32),
             (img // 64, img // 64), (img // 128, img // 128)]
    return [torch.randn(batch_size, channels, h, w, generator=g).mul_(.05).to(device)
            for h, w in sizes]


def head_targets(torch, device, batch_size=2):
    gt = [torch.tensor([[120., 118., 96., 48., .30], [300., 260., 64., 40., -.60]],
                       device=device) for _ in range(batch_size)]
    labels = [torch.zeros(2, dtype=torch.long, device=device) for _ in range(batch_size)]
    metas = [dict(pad_shape=(1024, 1024, 3), img_shape=(1024, 1024, 3),
                  scale_factor=(1., 1.)) for _ in range(batch_size)]
    return gt, labels, metas


def sum_loss(out, torch):
    total = None
    for v in out.values():
        for x in (v if isinstance(v, (list, tuple)) else [v]):
            if torch.is_tensor(x) and x.numel():
                total = x if total is None else total + x
    return total


def train_step(head, opt, torch, feats, gt, labels, metas):
    head.train()
    opt.zero_grad()
    cls_scores, bbox_preds = head(feats)
    out = head.loss(cls_scores, bbox_preds, gt, labels, metas)
    total = sum_loss(out, torch)
    total.backward()
    grads = {n: (None if p.grad is None else p.grad.detach().clone())
             for n, p in head.named_parameters()}
    opt.step()
    params = {n: p.detach().clone() for n, p in head.named_parameters()}
    return dict(total=float(total), grads=grads, params=params, keys=sorted(out.keys()),
                graph=total.grad_fn is not None)


def compare_grads(a, b, torch):
    keys, mismatch, both_none = set(a) | set(b), [], 0
    for k in keys:
        x, y = a.get(k), b.get(k)
        if x is None and y is None:
            both_none += 1
            continue
        if x is None or y is None:
            mismatch.append(k + ':presence')
            continue
        if not torch.equal(x, y):
            mismatch.append(k)
    return mismatch, both_none


def grad_activity(grads):
    present = sum(1 for v in grads.values() if v is not None)
    nonzero = sum(1 for v in grads.values() if v is not None and float(v.abs().sum()) > 0)
    return present, nonzero


def counters(head):
    return dict(assign=int(head.assigner._local_call_count),
                iter=int(head.loss_cls._local_iter))


def set_counters(head, c):
    head.assigner._local_call_count = c['assign']
    head.loss_cls._local_iter.fill_(c['iter'])


def finish(receipt, out_dir):
    req_fail = [c for c in CHECKS if c['required'] and not c['ok']]
    receipt['checks'] = CHECKS
    receipt['summary'] = dict(total=len(CHECKS), failed=sum(1 for c in CHECKS if not c['ok']),
                              required_failed=len(req_fail), passed=not req_fail)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out/'engineering.json').write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str))
    lines = ['%-4s %-13s %-46s %s\n' % ('PASS' if c['ok'] else 'FAIL', c['section'], c['name'],
                                        '' if c['ok'] else json.dumps(c['detail'], default=str)[:140])
             for c in CHECKS]
    (out/'stdout.txt').write_text(''.join(lines) + json.dumps(receipt['summary']) + '\n')
    print(''.join(lines), end='')
    print(json.dumps(receipt['summary']))
    return 0 if not req_fail else 1


def probe_operators(torch, device):
    ok = True
    for name, loader in (('rbbox_overlaps', lambda: __import__(
            'mmrotate.core.bbox.iou_calculators', fromlist=['rbbox_overlaps']).rbbox_overlaps),
                         ('diff_iou_rotated_2d', lambda: __import__(
                             'mmcv.ops', fromlist=['diff_iou_rotated_2d']).diff_iou_rotated_2d)):
        try:
            op = loader()
            box = torch.tensor([[10., 10., 20., 10., .1]], device=device)
            v = float(op(box, box)[0, 0]) if name == 'rbbox_overlaps' \
                else float(op(box.unsqueeze(0), box.unsqueeze(0)).max())
            record('operators', f'{name}_on_{device.type}', v > .99, dict(value=v))
            ok = ok and v > .99
        except Exception as exc:
            ok = False
            record('operators', f'{name}_on_{device.type}', False, dict(error=repr(exc)))
    return ok


def main(args):
    import torch
    torch.set_num_threads(1)
    device = (torch.device('cuda:%d' % torch.cuda.current_device())
              if args.device == 'cuda' else torch.device('cpu'))
    from mmcv import Config
    from mmcv.utils import build_from_cfg
    from mmrotate.models.builder import ROTATED_LOSSES, ROTATED_HEADS
    from mmrotate.core.bbox.builder import ROTATED_BBOX_ASSIGNERS

    receipt = dict(protocol='symeood_ablation_engineering_v3', device=str(device),
                   argv=sys.argv, cwd=str(Path.cwd()), hist_commit=args.hist_commit,
                   scope='L1 component + L2 head; L3 full model NOT performed',
                   tolerance_policy='bit-identical (atol=0, rtol=0); no relaxation')

    # ------------------------------------------------------------ operators first
    if not probe_operators(torch, device):
        receipt['summary'] = dict(total=len(CHECKS), failed=len(CHECKS),
                                  required_failed=len(CHECKS), passed=False,
                                  note='required rotation-IoU operators unavailable on this '
                                       'device; rerun with --device cuda')
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out/'engineering.json').write_text(json.dumps(receipt, indent=2, sort_keys=True,
                                                       default=str))
        for c in CHECKS:
            print('%-4s %-13s %-46s %s' % ('PASS' if c['ok'] else 'FAIL', c['section'],
                                           c['name'], json.dumps(c['detail'], default=str)[:140]))
        print(json.dumps(receipt['summary']))
        return 1

    # ------------------------------------------------------------ config
    cfg_dir = ROOT/'crane_project/configs'
    cfgs = {g: Config.fromfile(str(cfg_dir/f'crane_symeood_k1_ablation_{g}_v1.py'))
            for g in GROUPS}
    base = Config.fromfile(str(cfg_dir/'crane_symeood_k1_ablation_v1_base.py'))
    flat_base = flatten(base)
    diff_report, bad = {}, []
    for g, cfg in cfgs.items():
        flat = flatten(cfg)
        changed = sorted(k for k in set(flat) | set(flat_base) if flat.get(k) != flat_base.get(k))
        diff_report[g] = [k for k in changed if k != 'work_dir']
        bad += [[g, k] for k in changed if k != 'work_dir'
                and not any(k == w or k.startswith(w + '.') for w in WHITELIST)]
    record('config', 'diff_whitelist', not bad, bad)
    blocks = ('data', 'optimizer', 'lr_config', 'runner', 'evaluation', 'checkpoint_config',
              'model.backbone', 'model.neck', 'model.bbox_head.anchor_generator',
              'model.bbox_head.bbox_coder', 'model.aux_bbox_head')
    inv = {b: all(flatten(c).get(k) == flat_base.get(k) for c in cfgs.values()
                  for k in {x for x in flat_base if x == b or x.startswith(b + '.')})
           for b in blocks}
    record('config', 'invariant_blocks', all(inv.values()), inv)

    exp = {g: dict(A='kld' if g[0] == '1' else 'iou',
                   B='SymKLDLoss' if g[1] == '1' else 'RotatedIoULoss',
                   C=g[2] == '1') for g in GROUPS}
    sw = {g: dict(A=cfgs[g].model.train_cfg.assigner.get('cost_reg_mode', 'kld'),
                  B=cfgs[g].model.bbox_head.loss_bbox['type'],
                  C=cfgs[g].model.bbox_head.loss_cls.get('use_geometry_weight', True))
          for g in GROUPS}
    record('config', 'group_switches', all(sw[g] == exp[g] for g in GROUPS),
           {g: [sw[g], exp[g]] for g in GROUPS if sw[g] != exp[g]})
    receipt['config_diff'] = diff_report
    receipt['switches'] = sw

    def build(g):
        c = cfgs[g]
        return (build_from_cfg(c.model.train_cfg.assigner, ROTATED_BBOX_ASSIGNERS),
                build_from_cfg(c.model.bbox_head.loss_cls, ROTATED_LOSSES),
                build_from_cfg(c.model.bbox_head.loss_bbox, ROTATED_LOSSES))

    a111, lc111, lb111 = build('111')
    built = dict(nfl=dict(alpha=lc111.alpha, gamma=lc111.gamma, tau_init=lc111.tau_init,
                          tau_min=lc111.tau_min, warmup_iters=lc111.warmup_iters,
                          loss_call_factor=lc111.loss_call_factor,
                          kld_chunk_size=lc111.kld_chunk_size, spatial_topk=lc111.spatial_topk,
                          loss_weight=lc111.loss_weight, isolate_images=lc111.isolate_images,
                          use_quality_target=lc111.use_quality_target,
                          strict_finite=lc111.strict_finite,
                          use_geometry_weight=lc111.use_geometry_weight),
                 kld=dict(eps=lb111.eps, reduction=lb111.reduction,
                          loss_weight=lb111.loss_weight, strict_finite=lb111.strict_finite),
                 pola=dict(cost_class=a111.cost_class, cost_reg=a111.cost_reg, topk=a111.topk,
                           o2m=a111.o2m, o2m_warmup_iters=a111.o2m_warmup_iters,
                           o2m_topk=a111.o2m_topk, tau_init=a111.tau_init, tau_min=a111.tau_min,
                           warmup_iters=a111.warmup_iters, eps=a111.eps,
                           cost_reg_mode=a111.cost_reg_mode))
    expected = dict(nfl=dict(alpha=.25, gamma=2.0, tau_init=10.0, tau_min=1.0, warmup_iters=1000,
                             loss_call_factor=5, kld_chunk_size=256, spatial_topk=300,
                             loss_weight=.25, isolate_images=False, use_quality_target=False,
                             strict_finite=False, use_geometry_weight=True),
                    kld=dict(eps=1e-6, reduction='mean', loss_weight=2.0, strict_finite=False),
                    pola=dict(cost_class=1.0, cost_reg=2.0, topk=1, o2m=False,
                              o2m_warmup_iters=2000, o2m_topk=9, tau_init=10.0, tau_min=1.0,
                              warmup_iters=1000, eps=1e-6, cost_reg_mode='kld'))
    record('config', 'built_module_defaults', built == expected,
           {k: [built[k], expected[k]] for k in built if built[k] != expected[k]})
    receipt['built_defaults'] = built

    # ------------------------------------------------------------ historical identity
    hp = Path(args.hist_dir).resolve()
    ident = dict(current={}, historical={}, commit=args.hist_commit)
    for name, cls in (('SymPOLAAssigner', a111.__class__), ('SymNFLLoss', lc111.__class__),
                      ('SymKLDLoss', lb111.__class__)):
        ident['current'][name] = cls_info(cls)
    hist, hist_ok = {}, True
    regs = {'SymPOLAAssigner': ROTATED_BBOX_ASSIGNERS, 'SymNFLLoss': ROTATED_LOSSES,
            'SymKLDLoss': ROTATED_LOSSES}
    files = {'SymPOLAAssigner': 'sym_pola.py', 'SymNFLLoss': 'sym_nfl_loss.py',
             'SymKLDLoss': 'sym_kld_loss.py'}
    before = {k: r.module_dict.get(k) for k, r in regs.items()}
    try:
        for name, fn in files.items():
            p = hp/fn
            rel = [h for h in HIST_PATHS if h.endswith(fn)][0]
            blob = git_blob(ROOT, args.hist_commit, rel)
            ok = p.is_file() and blob is not None and sha(p) == hashlib.sha256(blob).hexdigest()
            hist_ok = hist_ok and ok
            record('identity', f'{name}_matches_git_{args.hist_commit[:7]}', ok,
                   dict(file=str(p), exists=p.is_file(), git_blob=blob is not None,
                        file_sha=sha(p) if p.is_file() else None,
                        git_sha=hashlib.sha256(blob).hexdigest() if blob else None))
        op_blob = git_blob(ROOT, args.hist_commit, HIST_OPERATOR)
        record('identity', 'shared_operator_matches_history',
               op_blob is not None and hashlib.sha256(op_blob).hexdigest() == sha(ROOT/HIST_OPERATOR),
               dict(current=sha(ROOT/HIST_OPERATOR),
                    git=hashlib.sha256(op_blob).hexdigest() if op_blob else None))
        if hist_ok:
            mp = load_hist(hp/'sym_pola.py', 'mmrotate.core.bbox.assigners._hist_sym_pola')
            mn = load_hist(hp/'sym_nfl_loss.py', 'mmrotate.models.losses._hist_sym_nfl_loss')
            mk = load_hist(hp/'sym_kld_loss.py', 'mmrotate.models.losses._hist_sym_kld_loss')
            hist = dict(pola=mp.SymPOLAAssigner, nfl=mn.SymNFLLoss, kld=mk.SymKLDLoss)
            for name, cls in (('SymPOLAAssigner', hist['pola']), ('SymNFLLoss', hist['nfl']),
                              ('SymKLDLoss', hist['kld'])):
                ident['historical'][name] = cls_info(cls)
    finally:
        for k, r in regs.items():
            if before[k] is None:
                r.module_dict.pop(k, None)
            else:
                r.module_dict[k] = before[k]
        record('registry', 'restored_in_finally',
               all(r.module_dict.get(k) is before[k] for k, r in regs.items()),
               {k: [hex(id(before[k])) if before[k] else None,
                    hex(id(r.module_dict.get(k))) if r.module_dict.get(k) else None]
                for k, r in regs.items()})
    probe = build_from_cfg(cfgs['111'].model.bbox_head.loss_cls, ROTATED_LOSSES)
    record('registry', 'build_returns_current_class', type(probe) is type(lc111),
           dict(built=type(probe).__module__))
    if hist:
        record('identity', 'classes_are_distinct',
               all(ident['current'][k]['id'] != ident['historical'][k]['id']
                   for k in ident['current']))
    receipt['identity'] = ident

    b = batch(torch, device)
    b_neg = batch(torch, device, positives=False)
    b_empty = batch(torch, device, gt=False)
    b_w = batch(torch, device, label_weight=0.5)

    # ------------------------------------------------------------ assigner
    if hist:
        h_pola = hist['pola'](cost_class=1.0, cost_reg=2.0, o2m=False, topk=1,
                              o2m_warmup_iters=2000, o2m_topk=9, tau_init=10.0,
                              tau_min=1.0, warmup_iters=1000, eps=1e-6)
        for it in (0, 1999, 4000):
            for tag, bb in (('normal', b), ('no_positive', b_neg), ('empty_gt', b_empty)):
                cur = run_assign(a111, bb, torch, it)
                ref = run_assign(h_pola, bb, torch, it)
                ok, why = same_assign(cur, ref, torch)
                record('assigner', f'111_vs_hist it={it} {tag}', ok,
                       dict(reason=why, positives=int((cur['gt_inds'] > 0).sum()),
                            labels_none=cur['labels'] is None,
                            max_overlaps_none=cur['max_overlaps'] is None))

    # ------------------------------------------------------------ losses
    if hist:
        h_nfl = hist['nfl'](use_sigmoid=True, gamma=2.0, alpha=0.25, tau_init=10.0, tau_min=1.0,
                            warmup_iters=1000, loss_call_factor=5, eps=1e-6, reduction='mean',
                            loss_weight=0.25, kld_chunk_size=256).to(device)
        h_kld = hist['kld'](eps=1e-6, reduction='mean', loss_weight=2.0).to(device)
        for it in (0, 999, 4000):
            for tag, bb in (('normal', b), ('no_positive', b_neg), ('empty_gt', b_empty),
                            ('weighted', b_w)):
                c = run_losses(lc111, lb111, bb, torch, it)
                h = run_losses(h_nfl, h_kld, bb, torch, it)
                ok = (c['cls'] == h['cls'] and c['bbox'] == h['bbox']
                      and torch.equal(c['grad_logits'], h['grad_logits'])
                      and torch.equal(c['grad_pred'], h['grad_pred'])
                      and torch.equal(c['updated'], h['updated'])
                      and c['local_iter'] == h['local_iter'])
                record('losses', f'111_vs_hist it={it} {tag}', ok,
                       dict(cls_absdiff=abs(c['cls'] - h['cls']),
                            bbox_absdiff=abs(c['bbox'] - h['bbox']),
                            grad_absdiff=float((c['grad_logits'] - h['grad_logits']).abs().max()),
                            local_iter=[c['local_iter'], h['local_iter']]))

    # ------------------------------------------------------------ switches
    _, lc101, lb101 = build('101')
    _, lc110, lb110 = build('110')
    a100, _, _ = build('100')
    a000, _, _ = build('000')
    record('switches', 'B_classes_are_111_kld_101_iou',
           type(lb111).__name__ == 'SymKLDLoss' and type(lb101).__name__ == 'RotatedIoULoss',
           dict(l111=type(lb111).__name__, l101=type(lb101).__name__))
    v111, g111 = run_bbox(lb111, b, torch)
    v101, g101 = run_bbox(lb101, b, torch)
    record('switches', 'B_changes_loss_and_gradient',
           v111 != v101 and not torch.equal(g111, g101),
           dict(loss=[v111, v101], grad_maxdiff=float((g111 - g101).abs().max())))
    diff_a = {it: int((run_assign(a100, b, torch, it)['gt_inds']
                       != run_assign(a000, b, torch, it)['gt_inds']).sum()) for it in (0, 4000)}
    from mmrotate.core.bbox.assigners.sym_pola import iou_diou_cost
    from mmrotate.models.losses.sym_kld_calculator import sym_kld
    cost_iou = iou_diou_cost(b['pred'], b['gt'])
    cost_kld = sym_kld(b['pred'][:, None, :].expand(-1, len(b['gt']), -1),
                       b['gt'][None, :, :].expand(len(b['pred']), -1, -1))
    record('switches', 'A_geometric_cost_differs', not torch.equal(cost_iou, cost_kld),
           dict(max_absdiff=float((cost_iou - cost_kld).abs().max())))
    record('switches', 'A_changes_assignment_at_some_schedule', any(v > 0 for v in diff_a.values()),
           diff_a)
    c_on = run_losses(lc111, lb110, b, torch)['cls']
    c_off = run_losses(lc110, lb110, b, torch)['cls']
    record('switches', 'C_changes_loss', c_on != c_off, dict(delta=c_on - c_off))
    off_w = run_losses(lc110, lb110, b_w, torch)['cls']
    ref_w = focal_reference(b_w, torch)
    record('switches', 'C_off_matches_weighted_focal', abs(off_w - ref_w) < 1e-6,
           dict(off=off_w, ref=ref_w))
    record('switches', 'C_off_advances_local_iter',
           run_losses(lc110, lb110, b, torch, 3)['local_iter'] == 16)
    lc_iso = build_from_cfg(dict(cfgs['110'].model.bbox_head.loss_cls, isolate_images=True),
                            ROTATED_LOSSES).to(device)
    record('switches', 'C_off_dominates_isolate_images',
           run_losses(lc_iso, lb110, b, torch)['cls'] == c_off)

    # ------------------------------------------------------------ orthogonality
    ok_a0 = all(torch.equal(run_assign(build(A0[0])[0], b, torch, it)['gt_inds'],
                            run_assign(build(g)[0], b, torch, it)['gt_inds'])
                for it in (0, 1999, 4000) for g in A0[1:])
    ok_a1 = all(torch.equal(run_assign(build(A1[0])[0], b, torch, it)['gt_inds'],
                            run_assign(build(g)[0], b, torch, it)['gt_inds'])
                for it in (0, 1999, 4000) for g in A1[1:])
    record('orthogonality', 'A0_four_groups_same_assignment', ok_a0)
    record('orthogonality', 'A1_four_groups_same_assignment', ok_a1)

    # ------------------------------------------------------------ head level
    def head_cfg(g):
        c = cfgs[g]
        # SingleStageDetector injects train_cfg/test_cfg into bbox_head; replicate
        # that so the head is built exactly as in a real model.
        return dict(c.model.bbox_head, train_cfg=c.model.train_cfg,
                    test_cfg=c.model.test_cfg)

    builder = lambda g: (lambda: build_from_cfg(head_cfg(g), ROTATED_HEADS))
    feats = head_feats(torch, device)
    gt, labels, metas = head_targets(torch, device)
    try:
        h_cur = builder('111')().to(device)
        opt = torch.optim.SGD(h_cur.parameters(), lr=LR, momentum=MOM)
        state0 = {k: v.detach().clone() for k, v in h_cur.state_dict().items()}
        cnt0 = counters(h_cur)
        step = train_step(h_cur, opt, torch, feats, gt, labels, metas)
        record('head', 'forward_loss_backward_step_runs',
               step['graph'] and step['total'] == step['total'],
               dict(total=step['total'], keys=step['keys']))
        present, nonzero = grad_activity(step['grads'])
        record('head', 'gradients_present_and_nonzero', present > 0 and nonzero > 0,
               dict(present=present, nonzero=nonzero))
        updated = sum(1 for n, p in h_cur.named_parameters()
                      if not torch.equal(p.detach(), state0[n]))
        record('head', 'parameters_updated', updated > 0, dict(updated=updated))
        record('head', 'device_consistent',
               all(p.device == device for p in h_cur.parameters())
               and all(x.device == device for x in h_cur.buffers()), dict(device=str(device)))

        if hist:
            hh = builder('111')()
            for name, obj in (('assigner', hist['pola'](cost_class=1.0, cost_reg=2.0, o2m=False,
                                                        topk=1, o2m_warmup_iters=2000, o2m_topk=9,
                                                        tau_init=10.0, tau_min=1.0,
                                                        warmup_iters=1000, eps=1e-6)),
                              ('loss_cls', hist['nfl'](use_sigmoid=True, gamma=2.0, alpha=0.25,
                                                       tau_init=10.0, tau_min=1.0,
                                                       warmup_iters=1000, loss_call_factor=5,
                                                       eps=1e-6, reduction='mean',
                                                       loss_weight=0.25, kld_chunk_size=256)),
                              ('loss_bbox', hist['kld'](eps=1e-6, reduction='mean',
                                                        loss_weight=2.0))):
                setattr(hh, name, obj)
            hh = hh.to(device)
            hh.load_state_dict(state0)
            opt2 = torch.optim.SGD(hh.parameters(), lr=LR, momentum=MOM)
            step_h = train_step(hh, opt2, torch, feats, gt, labels, metas)
            record('head', 'loss_matches_historical_components',
                   step['total'] == step_h['total'] and step['keys'] == step_h['keys'],
                   dict(cur=step['total'], hist=step_h['total']))
            mism, both_none = compare_grads(step['grads'], step_h['grads'], torch)
            record('head', 'gradients_match_historical_components', not mism,
                   dict(mismatch=mism[:5], both_none=both_none, total=len(step['grads'])))
            pmism = [k for k in step['params']
                     if not torch.equal(step['params'][k], step_h['params'][k])]
            record('head', 'updated_parameters_match', not pmism, dict(mismatch=pmism[:5]))

        h_r = builder('111')().to(device)
        h_r.load_state_dict(state0)
        set_counters(h_r, cnt0)
        r_step = train_step(h_r, torch.optim.SGD(h_r.parameters(), lr=LR, momentum=MOM),
                            torch, feats, gt, labels, metas)
        record('head', 'initial_state_reload_reproduces_next_step',
               r_step['total'] == step['total']
               and all(torch.equal(r_step['params'][k], step['params'][k])
                       for k in step['params']))

        h_a = builder('111')().to(device)
        opt_a = torch.optim.SGD(h_a.parameters(), lr=LR, momentum=MOM)
        train_step(h_a, opt_a, torch, feats, gt, labels, metas)
        sd = {k: v.detach().clone() for k, v in h_a.state_dict().items()}
        osd = copy.deepcopy(opt_a.state_dict())
        cnt = counters(h_a)
        cont = train_step(h_a, opt_a, torch, feats, gt, labels, metas)
        h_b = builder('111')().to(device)
        h_b.load_state_dict(sd)
        opt_b = torch.optim.SGD(h_b.parameters(), lr=LR, momentum=MOM)
        opt_b.load_state_dict(osd)
        naive = train_step(h_b, opt_b, torch, feats, gt, labels, metas)
        record('resume', 'counter_state_not_in_checkpoint_disclosed', True,
               dict(note='assigner._local_call_count is a plain int, not saved in state_dict; '
                         'loss_cls._local_iter is a buffer and IS saved',
                    naive_loss=naive['total'], continuous_loss=cont['total'],
                    abs_diff=abs(naive['total'] - cont['total'])), required=False)
        h_c = builder('111')().to(device)
        h_c.load_state_dict(sd)
        opt_c = torch.optim.SGD(h_c.parameters(), lr=LR, momentum=MOM)
        opt_c.load_state_dict(osd)
        set_counters(h_c, cnt)
        resumed = train_step(h_c, opt_c, torch, feats, gt, labels, metas)
        record('resume', 'model_optimizer_counter_resume_matches',
               resumed['total'] == cont['total']
               and all(torch.equal(resumed['params'][k], cont['params'][k])
                       for k in cont['params']),
               dict(continuous=cont['total'], resumed=resumed['total']))
    except Exception as exc:
        record('head', 'head_level_check', False, dict(error=repr(exc)))

    return finish(receipt, args.out)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--hist-dir', type=Path, required=True)
    p.add_argument('--hist-commit', default='af8307ca175a46ffd7baa754116e28a2798b1a70')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    sys.exit(main(p.parse_args()))
