#!/usr/bin/env python3
"""One frozen 32-step C/ESH TRAIN pair. No epoch training, VAL, TEST or depth."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
COMMON = ROOT/'work_dirs/symeood_110_retrain_seed0_20261026_v1/downstream/lvsr'
REVIEW = COMMON/'edge_diagnostic/sample_harm_review'
CONTRACT = REVIEW/'short_validation_contract.json'
DEST = REVIEW/'run32'
CONTRACT_SHA = '38625ca3f63cfc979bf185d4f019477c0cd6aae9610d697ecb4c323d06948f46'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def plain(x):
    if isinstance(x, dict): return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [plain(v) for v in x]
    if hasattr(x, 'detach'): return float(x.detach().cpu())
    return x


def flat(tensors, torch): return torch.cat([v.reshape(-1) for v in tensors])


def edge_points(output, gt, m, bm, torch):
    from crane_project.utils.port_size_qalign_v1 import residual
    target = residual.size_targets(gt, m, bm)['log_residual']
    error = output['delta_roi']-target
    swap = (m[:, 2] < m[:, 3]) ^ (bm[:, 2] < bm[:, 3])
    canonical = torch.where(swap[:, None], error.flip(-1), error)
    return torch.nn.functional.smooth_l1_loss(canonical, torch.zeros_like(canonical), reduction='none', beta=.1)


def edge_gradients(points, params, torch):
    rows = []
    for value in points.reshape(-1):
        grads = torch.autograd.grad(value/points.numel(), params, retain_graph=True, allow_unused=False)
        rows.append(flat([g.detach() for g in grads], torch))
    G = torch.stack(rows)
    if not bool(torch.isfinite(G).all()): raise ValueError('Nonfinite edge gradient')
    return G


def preview(params, opt, gradients, torch):
    """Original Adam(wd0) preview on clones; no optimizer state insertion/write."""
    if len(opt.param_groups) != 1: raise ValueError('One original Adam group required')
    group = opt.param_groups[0]
    if group['weight_decay'] != 0 or group['amsgrad']: raise ValueError('Frozen Adam setting differs')
    lr, (beta1, beta2), eps = group['lr'], group['betas'], group['eps']
    ds, ks, ms, vs, oldm, oldv, steps = [], [], [], [], [], [], []
    offset = 0
    for p in params:
        g = gradients[offset:offset+p.numel()].reshape_as(p); offset += p.numel()
        state = opt.state.get(p, {})
        m = state['exp_avg'].detach().clone() if state else torch.zeros_like(p)
        v = state['exp_avg_sq'].detach().clone() if state else torch.zeros_like(p)
        step = int(state['step'])+1 if state else 1
        oldm.append(m.clone()); oldv.append(v.clone())
        m.lerp_(g, 1-beta1); v.mul_(beta2).addcmul_(g, g, value=1-beta2)
        denominator = v.sqrt().div_(math.sqrt(1-beta2**step)).add_(eps)
        factor = 1/((1-beta1**step)*denominator)
        ds.append(-lr*factor*m); ks.append((1-beta1)*factor)
        ms.append(m); vs.append(v); steps.append(step)
    assert offset == len(gradients) and len(set(steps)) == 1
    return dict(delta=flat(ds, torch), kappa=flat(ks, torch), next_m=flat(ms, torch),
                next_v=flat(vs, torch), old_m=flat(oldm, torch), old_v=flat(oldv, torch), step=steps[0])


def solve(G, d0, kappa, ema, settings, torch):
    from crane_project.utils import lvsr_sample_harm_core_v1 as core
    lr = .0001
    Q1 = (G*kappa)@G.T; Q2 = (G*kappa.square())@G.T
    Q1 = (Q1+Q1.T)/2; Q2 = (Q2+Q2.T)/2
    gd = G@d0; energy = (.2/lr)**2*d0.square().sum()
    raw = dict(Q1=Q1, Q2=Q2, q=gd/lr, energy=energy)
    if any(not bool(torch.isfinite(v).all()) for v in raw.values()): raise ValueError('Nonfinite raw solver quantity')
    if float(d0.double().norm()) <= 1e-12 or (float(Q1.abs().max()) == 0 and float(gd.clamp_min(0).sum()) == 0):
        return torch.ones(len(G), device=G.device), dict(iterations=0, degenerate=True), raw, raw
    weights, logs, effective = core.get_sample_scales(Q1.clone(), Q2.clone(), gd.clone(), d0.square().sum().clone(),
        ema[0], ema[1], lr, settings)
    w = weights-1
    if not bool(torch.isfinite(weights).all()) or float(w.min()) < -1e-6 or float(w.max()) > 2+1e-6:
        raise ValueError('Frozen solver weight constraint failed')
    actual_energy = w@(effective['Q2']@w)
    violation = float((actual_energy-effective['energy']).clamp_min(0))/max(float(effective['energy']), 1e-30)
    if violation > 1e-5: raise ValueError('Frozen effective ellipsoid constraint failed')
    logs = dict(logs, effective_energy_ratio=float(actual_energy)/max(float(effective['energy']), 1e-30),
        guard=effective.get('guard', 0), diagonal_alpha=plain(effective.get('diagonal_alpha', 0)),
        raw_objective_before=float(raw['q'].clamp_min(0).sum()),
        raw_objective_after=float((raw['q']-raw['Q1']@w).clamp_min(0).sum()),
        effective_objective_before=float(effective['q'].clamp_min(0).sum()),
        effective_objective_after=float((effective['q']-effective['Q1']@w).clamp_min(0).sum()),
        fallback_identity=bool(torch.equal(weights, torch.ones_like(weights))))
    if any(not bool(torch.isfinite(v).all()) for v in effective.values() if hasattr(v, 'isfinite')):
        raise ValueError('Nonfinite effective solver quantity')
    return weights.detach(), plain(logs), raw, effective


def parameter_grad_assign(params, vector):
    offset = 0
    for p in params:
        p.grad = vector[offset:offset+p.numel()].reshape_as(p).clone(); offset += p.numel()
    assert offset == len(vector)


def check_sources():
    if sha(CONTRACT) != CONTRACT_SHA: raise ValueError('Frozen design contract changed')
    c = read(CONTRACT)
    if sha(ROOT/c['sample_manifest']['path']) != c['sample_manifest']['sha256']: raise ValueError('Frozen samples changed')
    for name, digest in c['original_sources'].items():
        if sha(ROOT/name) != digest: raise ValueError('Original source changed: '+name)
    old_sources = read(ROOT/'crane_project/data_contracts/symeood_110_lvsr_c_ep7.json')['sources']
    for name in ('crane_project/tools/run_port_size_a20_protection_v1.py',
                 'crane_project/tools/run_port_size_native_train_v1.py',
                 'crane_project/utils/port_geometry_midpoint_v1.py',
                 'crane_project/utils/port_geometry_refine_g_v1.py',
                 'crane_project/utils/port_structure_reliability_v1.py',
                 'crane_project/utils/port_size_a20_protection_review_v1.py'):
        if sha(ROOT/name) != old_sources[name]: raise ValueError('Native C dependency changed: '+name)
    if c['budget']['per_arm_updates'] != 32 or c['unit'] != '16 canonical edge losses per eight-view batch, interleaved long/short':
        raise ValueError('Frozen budget/unit differs')
    return c


def tensor_digest(items):
    h = hashlib.sha256()
    for name, t in items:
        h.update(name.encode()); h.update(str((tuple(t.shape), t.dtype)).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_selected(c, samples, torch, files):
    formal = COMMON.parent/'midpoint/formal'
    manifest = read(formal/'roi_cache/cache_manifest.json')
    def checked(p, h):
        if sha(p) != h: raise ValueError('Input SHA differs: '+str(p))
        files[str(p.relative_to(ROOT))] = h
    checked(formal/'roi_cache/cache_manifest.json', c['fixed_identity']['cache_manifest_sha256'])
    artifacts = read(COMMON/'artifacts.json')['files']; rowpath = COMMON/'fixed_b/epoch_07_train.rows.jsonl.gz'
    checked(rowpath, artifacts['fixed_b/epoch_07_train.rows.jsonl.gz'])
    with gzip.open(rowpath, 'rt') as f: saved = {(r['image'], r['scale']): r for r in map(json.loads, f)}
    refpath = COMMON/'continuation_ep7/C/train_fixed_references.json'
    checked(refpath, c['fixed_identity']['train_fixed_references_sha256'])
    refs = {(r['image'], r['scale']): r for r in read(refpath)}
    selected = {(r['image'], r['scale']): r for k in ('update_batches', 'probe_batches') for b in samples[k] for r in b['views']}
    found = {}
    for name, digest in c['fixed_identity']['train_shards'].items():
        path = formal/'roi_cache'/name; checked(path, digest)
        payload = torch.load(str(path), map_location='cpu')
        if payload['identity'] != manifest['identity'] or payload['detector_state'] != manifest['detector_state']:
            raise ValueError('Formal shard provenance differs')
        for r in payload['records']:
            key = (r['image'], r['scale'])
            if key not in selected: continue
            v = saved[key]; sample = selected[key]
            if r['split'] not in ('train', 'train_sim') or r['gt_original'][0].tolist() != v['gt'] or r['boxes_original'][0].tolist() != v['b']:
                raise ValueError('TRAIN row/GT/B identity differs')
            if sample['near_M21'] != refs[key]['near']: raise ValueError('Fixed near membership differs')
            found[key] = dict(r, midpoint_original=r['boxes_original'].new_tensor([v['midpoint']]),
                             near_M21=sample['near_M21'], other_M21=sample['other_M21'], large10_M21=sample['large10_M21'])
        del payload
    if found.keys() != selected.keys(): raise ValueError('Missing bounded TRAIN views')
    path = ROOT/c['fixed_identity']['parent_path']; checked(path, c['fixed_identity']['parent_sha256'])
    payload = torch.load(str(path), map_location='cpu')
    if payload['input_identity']['detector_sha256'] != c['fixed_identity']['detector_sha256'] or payload['input_identity']['midpoint_sha256'] != c['fixed_identity']['midpoint_sha256']:
        raise ValueError('Parent frontend identity differs')
    return found, refs, payload


def pack(rows, torch, device):
    return [torch.cat([r[k] for r in rows]).to(device) for k in
        ('roi', 'support', 'boxes_original', 'boxes_model', 'scale_xy', 'midpoint_original', 'gt_original')]


def output_rows(output, points, rows):
    rr = []
    boxes = output['boxes_original'].detach().cpu().tolist()
    raw = output['candidate_original'].detach().cpu().tolist()
    ell = points.detach().cpu().tolist()
    for i, r in enumerate(rows):
        if boxes[i][0:2]+boxes[i][4:] != r['midpoint_original'][0].tolist()[0:2]+r['midpoint_original'][0].tolist()[4:]:
            raise ValueError('Center/angle/score/output identity changed')
        rr.append(dict(image=r['image'], scale=r['scale'], domain=r['domain'], sequence=r['sequence'],
            gt=r['gt_original'][0].tolist(), midpoint=r['midpoint_original'][0].tolist(), raw=raw[i], delivered=boxes[i],
            point_loss=ell[i], near_M21=r['near_M21'], other_M21=r['other_M21'], large10_M21=r['large10_M21'],
            accepted=bool(output['accepted'][i]), vector_valid=bool(output['vector_valid'][i]),
            candidate_valid=bool(output['candidate_valid'][i]),
            failed_checks=[k for k, v in output['checks'].items() if not bool(v[i])]))
    return rr


def evaluate(head, records, torch, device):
    was_training = head.training; head.eval(); result = []
    with torch.no_grad():
        for i in range(0, len(records), 8):
            rr = records[i:i+8]; x = pack(rr, torch, device); output = head(*x[:-1])
            points = edge_points(output, x[-1], x[-2], x[3], torch)
            result.extend(output_rows(output, points, rr))
    head.train(was_training)
    return result


def vector_stats(G, delta, torch):
    def val(v):
        return dict(harm_sum=float(v.clamp_min(0).sum()), harmed=int((v > 0).sum()), dot_per_edge=v.cpu().tolist())
    return dict(full=val(G.double()@delta.double()), stem=val(G[:, :2728].double()@delta[:2728].double()))


def run(gpu):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu): raise ValueError('Explicit single GPU binding required')
    c = check_sources(); samples = read(ROOT/c['sample_manifest']['path'])
    if DEST.exists(): raise ValueError('Existing short trial preserved; no automatic repeated run')
    import torch
    from crane_project.utils.port_size_qalign_v1 import AlignedSizeHead, size_loss
    from crane_project.utils import port_size_a20_protection_train_v1 as original
    from crane_project.tools.run_port_size_native_train_v1 import seed, equal, rng_get
    from crane_project.tools.run_port_size_a20_protection_v1 import update as native_update
    DEST.mkdir(); device = 'cuda:0'; torch.set_num_threads(1); torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.6, 0)
    runtime = dict(torch=str(torch.__version__), cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0))
    original_runtime = read(COMMON/'continuation_ep7/identity.json')['runtime']
    if any(runtime[k] != v for k, v in original_runtime.items()): raise ValueError('Original C environment differs')
    files = {str(CONTRACT.relative_to(ROOT)): CONTRACT_SHA,
             c['sample_manifest']['path']: c['sample_manifest']['sha256']}
    for name in ('crane_project/tools/run_symeood_110_lvsr_esh_short_v1.py', 'crane_project/utils/lvsr_sample_harm_core_v1.py',
                 'tests/test_symeood_110_lvsr_esh_short_v1.py'):
        files[name] = sha(ROOT/name)
    records, refs, parent = load_selected(c, samples, torch, files)
    def ordered(name): return [records[r['image'], r['scale']] for b in samples[name] for r in b['views']]
    updates, probes = ordered('update_batches'), ordered('probe_batches')
    if {r['image'] for r in updates} & {r['image'] for r in probes}: raise ValueError('Image identity leakage')
    inputs_before = tensor_digest((str(key)+k, r[k]) for key, r in sorted(records.items()) for k in
        ('roi','support','boxes_original','boxes_model','scale_xy','midpoint_original','gt_original'))
    write(DEST/'identity.json', dict(contract_sha256=CONTRACT_SHA, runtime=runtime, physical_gpu=gpu,
        parent_sha256=c['fixed_identity']['parent_sha256'], sources=files, started=datetime.now(timezone.utc).isoformat(),
        scope='64 temporary optimizer updates, no warmup updates, no VAL/TEST/depth',
        diagnostics='Both arms also measure16 edge VJPs per batch; control update itself uses unchanged native_update'))
    settings = {k:c['solver'][k] for k in ('sum_grads','constraint','box_bound','radius_bound','allow_neg_weights','reg','warm_start_iter','max_iter','tol','mom_stats_Q','max_raise_condition_Q','max_raise_dgdg')}
    settings['use_wandb'] = False
    starts = {}; finished = {}
    try:
        for arm in ('C', 'ESH'):
            dest = DEST/arm; dest.mkdir(); seed(torch)
            head = AlignedSizeHead('fixed_b').to(device); head.load_state_dict(parent['head_state'], strict=True)
            head.train(); params = tuple(head.parameters())
            initial = tensor_digest(list(head.named_parameters())+list(head.named_buffers()))
            starts[arm] = initial
            if len(starts) == 2 and starts['C'] != starts['ESH']: raise ValueError('Paired starting parameters differ')
            opt = torch.optim.Adam(params, lr=.0001, betas=(.9,.999), eps=1e-8, weight_decay=0., amsgrad=False)
            if opt.state: raise ValueError('Adam must start empty')
            ema = [torch.tensor(-1., device=device), torch.tensor(-1., device=device)]
            tensors = []; count = 0; buffers_before = tensor_digest(head.named_buffers())
            for role, rr in (('update', updates), ('probe', probes)):
                with gzip.open(dest/('step00_'+role+'.jsonl.gz'), 'xt') as f:
                    for r in evaluate(head, rr, torch, device): f.write(json.dumps(r, allow_nan=False)+'\n')
            with (dest/'steps.jsonl').open('x') as log:
                for step in range(1, 33):
                    tick = time.monotonic(); rr = updates[8*(step-1):8*step]; x = pack(rr, torch, device)
                    opt.zero_grad(); output = head(*x[:-1]); points = edge_points(output, x[-1], x[-2], x[3], torch)
                    if not bool(output['vector_valid'].all()) or not bool(output['candidate_valid'].all()): raise ValueError('Invalid training candidate')
                    before_rows = output_rows(output, points, rr)
                    loss = size_loss(output, x[-1], x[-2], x[3])
                    native_g = flat(torch.autograd.grad(loss, params, retain_graph=True), torch).detach()
                    G = edge_gradients(points, params, torch)
                    reconstruction = float((G.sum(0)-native_g).abs().max())
                    if not torch.allclose(G.sum(0), native_g, rtol=2e-5, atol=2e-7) or not torch.allclose(points.mean(), loss, rtol=1e-6, atol=1e-7):
                        raise ValueError('16 edge contribution reconstruction failed')
                    state_before = deepcopy(opt.state_dict()); rng_before = rng_get(torch)
                    base = preview(params, opt, native_g, torch)
                    if not equal(state_before, opt.state_dict(), torch) or not equal(rng_before, rng_get(torch), torch):
                        raise ValueError('Adam preview mutated optimizer/RNG')
                    if base['step'] != step: raise ValueError('Preview counter differs')
                    if arm == 'ESH':
                        weights, solver, raw, effective = solve(G, base['delta'], base['kappa'], ema, settings, torch)
                        gradient = G.T@weights
                    else:
                        weights = torch.ones(16, device=device); solver = dict(control=True); raw = effective = {}
                        gradient = native_g
                    if not bool(torch.isfinite(gradient).all()) or float(gradient.double().norm()) > 10 or float(native_g.double().norm()) > 10:
                        raise ValueError('Nonfinite gradient or clipping activation: stop frozen short trial')
                    weighted = preview(params, opt, gradient, torch)
                    if not equal(state_before, opt.state_dict(), torch): raise ValueError('Weighted preview changed state')
                    theta = flat([p.detach().clone() for p in params], torch)
                    predicted = base['delta']-.0001*base['kappa']*(G.T@(weights-1))
                    del output, points, loss
                    if arm == 'C':
                        packed_refs = original.pack_references(rr, refs, device, torch, training=True)
                        native_stats, _ = native_update(head, opt, x, packed_refs, 'C', torch)
                        if not torch.allclose(flat([p.grad for p in params], torch), native_g, atol=2e-7, rtol=2e-5):
                            raise ValueError('Native control gradient changed')
                    else:
                        opt.zero_grad(); parameter_grad_assign(params, gradient)
                        gradnorm = float(torch.nn.utils.clip_grad_norm_(params, 10.)); opt.step()
                        native_stats = dict(grad_norm=gradnorm)
                    count += 1
                    if any(int(opt.state[p]['step']) != count for p in params): raise ValueError('Real Adam must advance once')
                    actual = flat([p.detach() for p in params], torch)-theta
                    if not bool(torch.isfinite(actual).all()): raise ValueError('Nonfinite actual parameter update')
                    if not torch.allclose(actual, weighted['delta'], rtol=2e-4, atol=2e-7): raise ValueError('Real update differs from exact own-Adam preview')
                    actual_m = flat([opt.state[p]['exp_avg'] for p in params], torch)
                    actual_v = flat([opt.state[p]['exp_avg_sq'] for p in params], torch)
                    if not torch.allclose(actual_m, weighted['next_m'], rtol=1e-6, atol=1e-8) or not torch.allclose(actual_v, weighted['next_v'], rtol=1e-6, atol=1e-10):
                        raise ValueError('Adam moments/counter mismatch')
                    after_rows = evaluate(head, rr, torch, device)
                    before_ell = torch.tensor([r['point_loss'] for r in before_rows], dtype=torch.float64)
                    after_ell = torch.tensor([r['point_loss'] for r in after_rows], dtype=torch.float64)
                    norm0 = float(base['delta'].double().norm()); norm_actual = float(actual.double().norm())
                    residual = float((actual-predicted).double().norm())
                    prednorm = float(predicted.double().norm())
                    rec = dict(step=step, updates=count, scalar_edge_reconstruction_max=reconstruction,
                        gradient_norm=float(gradient.double().norm()), original_gradient_norm=float(native_g.double().norm()), clipping_coefficient=1.,
                        weights=weights.cpu().tolist(), solver=solver, ema=[float(v) for v in ema],
                        actual_delta_norm=norm_actual, original_preview_norm=norm0,
                        candidate_predicted_norm=prednorm, actual_predicted_residual=residual,
                        actual_predicted_relative_residual=residual/max(norm_actual, 1e-30),
                        actual_predicted_cosine=float(actual.double().dot(predicted.double()))/max(norm_actual*prednorm,1e-30),
                        actual_vs_original_radius=float((actual-base['delta']).double().norm())/max(norm0,1e-30),
                        proxy_vs_original_radius=float((predicted-base['delta']).double().norm())/max(norm0,1e-30),
                        second_moment_relative_change=float((actual_v-base['next_v']).double().norm())/max(float(base['next_v'].double().norm()),1e-30),
                        original_update_proxy=vector_stats(G, base['delta'], torch),
                        predicted_update_proxy=vector_stats(G, predicted, torch),
                        actual_update_proxy=vector_stats(G, actual, torch),
                        finite_current_batch_loss_delta=(after_ell-before_ell).reshape(-1).tolist(),
                        finite_current_batch_harm_sum=float((after_ell-before_ell).clamp_min(0).sum()/16),
                        actual_current_batch_loss_mean_before=float(before_ell.mean()), actual_current_batch_loss_mean_after=float(after_ell.mean()),
                        batch_before=before_rows, batch_after=after_rows,
                        elapsed_seconds=time.monotonic()-tick, peak_allocated_bytes=torch.cuda.max_memory_allocated(), native_stats=native_stats)
                    tensors.append(dict(step=step, G=G.cpu(), theta_before=theta.cpu(), actual_delta=actual.cpu(),
                        d0=base['delta'].cpu(), predicted_delta=predicted.cpu(), kappa=base['kappa'].cpu(), weights=weights.cpu(),
                        m_before=base['old_m'].cpu(), v_before=base['old_v'].cpu(),
                        unweighted_next_m=base['next_m'].cpu(), unweighted_next_v=base['next_v'].cpu(),
                        actual_next_m=actual_m.cpu(), actual_next_v=actual_v.cpu(),
                        raw={k:v.detach().cpu() for k,v in raw.items()},
                        effective={k:v.detach().cpu() if hasattr(v,'detach') else v for k,v in effective.items()}))
                    log.write(json.dumps(plain(rec), allow_nan=False)+'\n'); log.flush()
                    if step in (8,32):
                        for role, rows in (('update', updates), ('probe', probes)):
                            with gzip.open(dest/('step%02d_'%step+role+'.jsonl.gz'), 'xt') as f:
                                for r in evaluate(head, rows, torch, device): f.write(json.dumps(r, allow_nan=False)+'\n')
                    print(json.dumps(dict(arm=arm, step=step, updates=count, weights_mean=float(weights.mean()),
                                         actual_proxy_harm=rec['actual_update_proxy']['full']['harm_sum'])), flush=True)
                if count != 32 or buffers_before != tensor_digest(head.named_buffers()): raise ValueError('Budget or buffer changed')
            with (dest/'step_vectors.pt').open('xb') as f: torch.save(tensors, f)
            finished[arm] = dict(updates=count,initial_digest=initial,final_digest=tensor_digest(list(head.named_parameters())),
                final_optimizer_step=32,peak_allocated_bytes=torch.cuda.max_memory_allocated())
            del opt, head, tensors, params; torch.cuda.empty_cache()
        inputs_after = tensor_digest((str(key)+k, r[k]) for key, r in sorted(records.items()) for k in
            ('roi','support','boxes_original','boxes_model','scale_xy','midpoint_original','gt_original'))
        if inputs_before != inputs_after: raise ValueError('Frozen cache tensors changed')
        for name, digest in files.items():
            if sha(ROOT/name) != digest: raise ValueError('Original source/input bytes changed')
        check_sources()
        write(DEST/'completion.json', dict(status='complete', finished=datetime.now(timezone.utc).isoformat(),
            contract_sha256=CONTRACT_SHA, arms=finished, paired_initial_identical=True, actual_updates=64,
            updates_discarded=True, new_adoptable_checkpoints=0, original_files_unchanged=files,
            original_model_sources_unchanged=True, frozen_input_tensors_unchanged=True, TEST=False, depth=False, VAL=False,
            files={str(p.relative_to(DEST)):sha(p) for p in DEST.rglob('*') if p.is_file()}))
    except BaseException as error:
        write(DEST/'stopped.json',dict(error=repr(error),finished_arms=finished,automatic_retry=False,TEST=False,depth=False))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='store_true'); p.add_argument('--gpu', type=int, choices=range(4))
    args = p.parse_args()
    if args.run:
        if args.gpu is None: p.error('Explicit GPU required')
        run(args.gpu)
    else:
        print(json.dumps(dict(static_check=True, contract=check_sources()['protocol'], optimizer_updates=0)))
