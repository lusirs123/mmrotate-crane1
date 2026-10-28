#!/usr/bin/env python3
"""Bounded TRAIN-only frozen LVSR gradients. No optimizer or model writes."""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
COMMON = ROOT/'work_dirs/symeood_110_retrain_seed0_20261026_v1/downstream/lvsr'
OUT = COMMON/'edge_diagnostic/gradients_ep7_C6'
FORMAL = COMMON.parent/'midpoint/formal'
SEED = 1703
EPS = 1e-12
PARENTS = {
    'ep7': ('fixed_b/head_epoch_07.pth', 'dfa6206dd9eadd6875bd9523afa409a59d797bde3db926a41a8d0e33ea08eee6'),
    'C6': ('continuation_ep7/C/head_epoch_06.pth', '454a132c5bc4383121791c2b35a9f696672b13bc19c38463b694a8ec1a1fc6ac')}
DETECTOR = '7f35dacc308943949a8fd2e6aaad312d14651564903ec87c8c8dcec0552ec054'
MIDPOINT = '7aaaacc6c666116e466c9a99ac59d71acf7f77312dc86de224fcf965569fd9ac'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def choose(rows):
    """Eight disjoint batches per domain/scale; membership never sees C6."""
    chosen = []
    rng = random.Random(SEED)
    for domain in ('real', 'sim'):
        for scale in (1., .5):
            pool = sorted([r for r in rows if r['domain'] == domain and
                           r['scale'] == scale and r['midpoint'] is not None], key=lambda r: r['image'])
            picks = rng.sample(pool, 64)
            for i in range(8):
                chosen.append(dict(batch=len(chosen)+1, domain=domain, scale=scale,
                                   views=picks[8*i:8*i+8]))
    assert len({(r['image'], r['scale']) for b in chosen for r in b['views']}) == 256
    return chosen


def decompose(output, gt, m, bm, near, torch):
    """Original ROI target -> fixed M raw order -> canonical long/short."""
    from crane_project.utils.port_size_qalign_v1 import residual, size_loss
    from torch.nn import functional as F
    target = residual.size_targets(gt, m, bm)['log_residual']
    error = output['delta_roi']-target
    swap = (m[:, 2] < m[:, 3]) ^ (bm[:, 2] < bm[:, 3])
    canonical = torch.where(swap[:, None], error.flip(-1), error)
    points = F.smooth_l1_loss(canonical, torch.zeros_like(canonical), beta=.1, reduction='none')
    result = {'size': size_loss(output, gt, m, bm)}
    for j, name in enumerate(('long', 'short')):
        result[name] = points[:, j].mean()
        result[name+'_near'] = (points[:, j]*near[:, j]).mean()
        result[name+'_other'] = (points[:, j]*(~near[:, j])).mean()
    assert not target.requires_grad and not near.requires_grad and not gt.requires_grad
    return result, swap


def norm(x):
    return float(x.double().square().sum())**.5


def relation(a, b, total, applicable=True):
    na, nb, nt = norm(a), norm(b), norm(total)
    dot = float(a.double().dot(b.double())) if applicable else None
    cosine = dot/(na*nb) if applicable and min(na, nb) > EPS else None
    adot = float(a.double().dot(total.double())) if applicable else None
    bdot = float(b.double().dot(total.double())) if applicable else None
    return dict(applicable=applicable, a_norm=na, b_norm=nb, total_norm=nt,
                dot=dot, cosine=cosine, b_over_a=nb/na if applicable and na > EPS else None,
                a_zero_or_tiny=na <= EPS, b_zero_or_tiny=nb <= EPS,
                a_dot_total=adot, b_dot_total=bdot,
                a_cos_total=adot/(na*nt) if applicable and min(na, nt) > EPS else None,
                b_cos_total=bdot/(nb*nt) if applicable and min(nb, nt) > EPS else None)


def distribution(values):
    xs = sorted(float(x) for x in values if x is not None)
    if not xs:
        return dict(n=0, mean=None, minimum=None, p10=None, p25=None, median=None, p75=None, p90=None, maximum=None)
    def q(p):
        k = (len(xs)-1)*p; lo = int(k); hi = min(lo+1, len(xs)-1)
        return xs[lo]+(xs[hi]-xs[lo])*(k-lo)
    return dict(n=len(xs), mean=sum(xs)/len(xs), minimum=xs[0], p10=q(.1), p25=q(.25),
                median=q(.5), p75=q(.75), p90=q(.9), maximum=xs[-1])


def summarize(rows):
    grouped = defaultdict(list)
    for r in rows:
        for domain, scale in ((r['domain'], r['scale']), ('overall', 'all')):
            grouped[(r['checkpoint'], domain, scale, r['scope'], r['relation'])].append(r)
    out = []
    for key, rr in grouped.items():
        valid = [r for r in rr if r['cosine'] is not None]
        v = dict(zip(('checkpoint', 'domain', 'scale', 'scope', 'relation'), key))
        v.update(batches=len(rr), applicable=sum(r['applicable'] for r in rr), cosine_valid=len(valid),
                 zero_or_tiny=sum(r['a_zero_or_tiny'] or r['b_zero_or_tiny'] for r in rr),
                 negative_dot=sum(r['dot'] is not None and r['dot'] < 0 for r in rr),
                 negative_cos_fraction=sum(r['cosine'] < 0 for r in valid)/len(valid) if valid else None,
                 cosine_below_minus_0p1=sum(r['cosine'] < -.1 for r in valid),
                 cosine_below_minus_0p5=sum(r['cosine'] < -.5 for r in valid),
                 a_dot_total_negative=sum(r['a_dot_total'] is not None and r['a_dot_total'] < 0 for r in rr),
                 b_dot_total_negative=sum(r['b_dot_total'] is not None and r['b_dot_total'] < 0 for r in rr),
                 a_norm_gt_2b=sum(r['a_norm'] > 2*r['b_norm'] for r in rr if r['applicable']),
                 b_norm_gt_2a=sum(r['b_norm'] > 2*r['a_norm'] for r in rr if r['applicable']),
                 a_count_sum=sum(r['a_count'] for r in rr), b_count_sum=sum(r['b_count'] for r in rr))
        for name in ('cosine', 'a_norm', 'b_norm', 'b_over_a', 'dot', 'a_dot_total', 'b_dot_total'):
            v.update({name+'_'+k: x for k, x in distribution([r[name] for r in rr if r['applicable']]).items()})
        out.append(v)
    return out


def csv_write(path, rows):
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader(); writer.writerows(rows)


def run():
    # Small frozen head on original server CPU runtime; never initializes CUDA.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    from crane_project.utils.port_size_qalign_v1 import AlignedSizeHead
    from crane_project.utils import port_size_a20_protection_train_v1 as original_C
    from crane_project.tools.diagnose_port_center_size_d_v1 import gradient_vector
    torch.set_num_threads(1)
    if OUT.exists() and any(OUT.iterdir()):
        raise ValueError('Existing diagnostic preserved; inspect it, do not auto-repeat')
    OUT.mkdir(parents=True, exist_ok=True)
    before_files = {}
    def checked(path, expected=None):
        h = sha(path)
        if expected is not None and h != expected:
            raise ValueError('Source identity differs: '+str(path))
        before_files[str(path.relative_to(ROOT))] = h
        return path
    cproof = read(checked(ROOT/'crane_project/data_contracts/symeood_110_lvsr_c_ep7.json'))
    model_sources = ['crane_project/utils/port_size_qalign_v1.py', 'crane_project/utils/port_size_local_vectors_v1.py',
        'crane_project/utils/port_geometry_midpoint_edge_residual_v1.py',
        'crane_project/utils/port_geometry_midpoint_v1.py',
        'crane_project/utils/port_geometry_refine_g_v1.py',
        'crane_project/utils/port_structure_reliability_v1.py',
        'crane_project/utils/port_size_a20_protection_train_v1.py',
        'crane_project/utils/port_size_a20_protection_review_v1.py']
    for name in model_sources:
        checked(ROOT/name, cproof['sources'][name])
    checked(Path(__file__).resolve()); checked(ROOT/'crane_project/tools/diagnose_port_center_size_d_v1.py')
    checked(ROOT/'tests/test_symeood_110_lvsr_gradients_v1.py')
    manifest_path = checked(FORMAL/'roi_cache/cache_manifest.json')
    manifest = read(manifest_path)
    cp_proof = read(checked(ROOT/'crane_project/data_contracts/symeood_110_lvsr_cp_v1.json'))
    if sha(manifest_path) != cp_proof['inputs']['cache_manifest_sha256']:
        raise ValueError('Formal TRAIN cache manifest differs')
    artifacts = read(checked(COMMON/'artifacts.json'))['files']
    rowname = 'fixed_b/epoch_07_train.rows.jsonl.gz'
    with gzip.open(checked(COMMON/rowname, artifacts[rowname]), 'rt') as f:
        train = [json.loads(x) for x in f]
    if len(train) != 5116 or sum(r['midpoint'] is not None for r in train) != 5112:
        raise ValueError('Original TRAIN input count differs')
    refpath = ROOT/cp_proof['inputs']['train_fixed_references']
    refs = {(r['image'], r['scale']): r for r in read(checked(refpath, cp_proof['inputs']['train_reference_sha256']))}
    batches = choose(train)
    for b in batches:
        for r in b['views']:
            near = [abs(x/y-1) <= .03 for x, y in zip(sorted(r['midpoint'][2:4], reverse=True), sorted(r['gt'][2:4], reverse=True))]
            if near != refs[r['image'], r['scale']]['near']:
                raise ValueError('Fixed M21 near membership differs')
    sample = dict(seed=SEED, sampler='Random(1703), sorted image pools, real1/real0.5/sim1/sim0.5; sample64 without replacement per stratum',
        views=256, unique_images=len({r['image'] for b in batches for r in b['views']}),
        near_source='Frozen M21 per-edge absolute relative error <=0.03; original float64 memberships',
        batches=[dict(batch=b['batch'], domain=b['domain'], scale=b['scale'],
            views=[dict(image=r['image'], scale=r['scale'], split=r['split'], sequence=r['sequence'],
                        near=refs[r['image'], r['scale']]['near']) for r in b['views']]) for b in batches])
    write(OUT/'samples.json', sample)
    write(OUT/'protocol.json', dict(seed=SEED, device='cpu', checkpoints=PARENTS,
        scalar='L_size=(L_long+L_short)/2, SmoothL1 beta0.1 before guards; canonical M xor model-B swap',
        subset='near and complement each sum/Nbatch, not conditional mean; GT and masks detached',
        scopes={'stem': '2728 parameters, structurally shared convolution stem',
                'full': '2800 parameters: stem plus terminal 8 vector-coordinate filters (72 params); not two scalar edge outputs'},
        near_zero_norm=EPS, descriptive_cosine_bins=[0., -.1, -.5], descriptive_dominance_ratio=2.,
        optimizer_updates=0, clipping=False, optimizer_created=False, TEST_reads=0, depth_reads=0,
        limit='Raw gradients at two fixed snapshots, not temporal dominance, Adam direction, finite-step effect or causal proof',
        sources_before=before_files.copy()))
    selected = {(r['image'], r['scale']): r for b in batches for r in b['views']}
    cached = {}
    for name in ('train_s1.pt', 'train_s05.pt'):
        path = checked(FORMAL/'roi_cache'/name, manifest['files'][name])
        payload = torch.load(str(path), map_location='cpu')
        if payload['identity'] != manifest['identity'] or payload['detector_state'] != manifest['detector_state']:
            raise ValueError('Formal shard identity differs')
        for r in payload['records']:
            k = (r['image'], r['scale'])
            if k not in selected:
                continue
            saved = selected[k]
            if r['gt_original'][0].tolist() != saved['gt'] or r['boxes_original'][0].tolist() != saved['b']:
                raise ValueError('Selected TRAIN B/GT differs')
            if r['split'] not in ('train', 'train_sim'):
                raise ValueError('Only TRAIN tensors allowed')
            cached[k] = dict(r, midpoint_original=r['boxes_original'].new_tensor([saved['midpoint']]))
        del payload
    if cached.keys() != selected.keys():
        raise ValueError('Missing sampled TRAIN cache views')
    def tensor_digest(values):
        h = hashlib.sha256()
        for name, v in values:
            h.update(name.encode()); h.update(str((v.dtype, tuple(v.shape))).encode())
            h.update(v.detach().contiguous().numpy().tobytes())
        return h.hexdigest()
    def inputs_digest():
        return tensor_digest((str(k)+name, r[name]) for k, r in sorted(cached.items()) for name in
            ('roi', 'support', 'boxes_original', 'boxes_model', 'scale_xy', 'midpoint_original', 'gt_original'))
    input_before = inputs_digest()
    relations = [('long_short', 'long', 'short'), ('long_total', 'long', 'size'), ('short_total', 'short', 'size')]
    for e, other in (('long', 'short'), ('short', 'long')):
        relations += [(e+'_near_other', e+'_near', e+'_other'),
                      (e+'_near_cross', e+'_near', other), (e+'_near_own', e+'_near', e)]
    allrows, details = [], []
    with (OUT/'progress.jsonl').open('x') as progress:
        for checkpoint, (relative, expected) in PARENTS.items():
            path = checked(COMMON/relative, expected)
            payload = torch.load(str(path), map_location='cpu')
            if payload['input_identity']['detector_sha256'] != DETECTOR or payload['input_identity']['midpoint_sha256'] != MIDPOINT:
                raise ValueError('Frozen front-end identity differs')
            head = AlignedSizeHead('fixed_b'); head.load_state_dict(payload['head_state'], strict=True); head.train()
            # Architecture contains no stochastic layer or mutable running-stat buffer.
            before = tensor_digest(list(head.named_parameters())+list(head.named_buffers()))
            params = tuple(head.parameters()); names = [n for n, _ in head.named_parameters()]
            sizes = [p.numel() for p in params]; stem_end = sum(p.numel() for n, p in head.named_parameters() if n.startswith('stem.'))
            assert sum(sizes) == 2800 and stem_end == 2728
            for b in batches:
                rr = [cached[r['image'], r['scale']] for r in b['views']]
                x = [torch.cat([r[k] for r in rr]) for k in
                     ('roi', 'support', 'boxes_original', 'boxes_model', 'scale_xy', 'midpoint_original', 'gt_original')]
                assert not any(t.requires_grad for t in x)
                near = torch.tensor([refs[r['image'], r['scale']]['near'] for r in b['views']], dtype=torch.bool)
                output = head(*x[:-1]); losses, swap = decompose(output, x[-1], x[-2], x[3], near, torch)
                if not bool(output['vector_valid'].all()) or not bool(output['candidate_valid'].all()):
                    raise ValueError('Invalid raw training candidate')
                packed = original_C.pack_references(b['views'], refs, 'cpu', torch, training=True)
                control_loss = original_C.losses(output, x[-1], x[-2], x[3], packed, 'C', torch)['total']
                gradients = {}
                for key, loss in dict(losses, original_C=control_loss).items():
                    gg = gradient_vector(loss, params)
                    gradients[key] = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(gg, params)])
                def reconstruction(a, b, label):
                    if not torch.allclose(a, b, rtol=2e-5, atol=2e-7):
                        raise ValueError('Gradient reconstruction differs: '+label)
                    return float((a-b).abs().max())
                errors = {'size': reconstruction(gradients['size'], (gradients['long']+gradients['short'])/2, 'size')}
                errors['original_C'] = reconstruction(gradients['size'], gradients['original_C'], 'original_C')
                scalar_error = abs(float(losses['size']-(losses['long']+losses['short'])/2))
                assert scalar_error <= 1e-7 and float(control_loss) == float(losses['size'])
                for e in ('long', 'short'):
                    errors[e] = reconstruction(gradients[e], gradients[e+'_near']+gradients[e+'_other'], e)
                counts = {'long': 8, 'short': 8, 'size': 8}
                for j, e in enumerate(('long', 'short')):
                    counts[e+'_near'] = int(near[:, j].sum()); counts[e+'_other'] = 8-counts[e+'_near']
                for scope, end in (('stem', stem_end), ('full', 2800)):
                    for label, a, bname in relations:
                        result = relation(gradients[a][:end], gradients[bname][:end], gradients['size'][:end],
                                          counts[a] > 0 and counts[bname] > 0)
                        allrows.append(dict(checkpoint=checkpoint, batch=b['batch'], domain=b['domain'], scale=b['scale'],
                            scope=scope, relation=label, a=a, b=bname, a_count=counts[a], b_count=counts[bname],
                            a_loss=float(losses[a]), b_loss=float(losses[bname]), **result))
                terminal = {}
                for e in ('long', 'short'):
                    offset = 0; filters = None
                    for name, size in zip(names, sizes):
                        g = gradients[e][offset:offset+size]; offset += size
                        if name == 'terminal.weight': filters = g.reshape(8, -1).double().square().sum(-1)
                        if name == 'terminal.bias': filters += g.double().square()
                    terminal[e] = filters.sqrt().tolist()
                detail = dict(checkpoint=checkpoint, batch=b['batch'], domain=b['domain'], scale=b['scale'],
                    near_long=counts['long_near'], near_short=counts['short_near'], canonical_swaps=int(swap.sum()),
                    scalar_reconstruction_abs=scalar_error, gradient_reconstruction_max_abs=errors,
                    terminal_filter_grad_norms=terminal,
                    terminal_filters_active_for_both=sum(a > EPS and bb > EPS for a, bb in zip(terminal['long'], terminal['short'])))
                details.append(detail); progress.write(json.dumps(detail, allow_nan=False)+'\n'); progress.flush()
                print(json.dumps(dict(checkpoint=checkpoint, batch=b['batch'], complete=checkpoint == 'C6' and b['batch'] == 32)), flush=True)
            after = tensor_digest(list(head.named_parameters())+list(head.named_buffers()))
            if before != after or any(p.grad is not None for p in params):
                raise ValueError('Frozen parameters/buffers/grad fields mutated')
            del head, payload
    if input_before != inputs_digest() or torch.cuda.is_initialized():
        raise ValueError('Inputs changed or CUDA initialized')
    for name, expected in before_files.items():
        if sha(ROOT/name) != expected:
            raise ValueError('Original file changed: '+name)
    summary = summarize(allrows)
    csv_write(OUT/'batches.csv', allrows); csv_write(OUT/'summary.csv', summary)
    write(OUT/'integrity.json', dict(state='complete', finished=datetime.now(timezone.utc).isoformat(),
        torch=str(torch.__version__), cuda_build=torch.version.cuda, cudnn_build=torch.backends.cudnn.version(),
        device='cpu', cuda_initialized=False, model_forwards=64, views_per_checkpoint=256,
        updates=0, optimizer_created=False, clipping=False, parameters_and_buffers_unchanged=True,
        parameter_grad_fields_untouched=True, input_tensors_unchanged=True, original_files_unchanged=before_files,
        full_parameter_count=2800, stem_parameter_count=2728, batch_relation_rows=len(allrows), summary_rows=len(summary),
        TEST_reads=0, VAL_tensor_reads=0, depth_reads=0,
        scalar_reconstruction_max=max(d['scalar_reconstruction_abs'] for d in details),
        gradient_reconstruction_max=max(v for d in details for v in d['gradient_reconstruction_max_abs'].values()),
        files={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Run only the fixed 64 frozen TRAIN gradient batches')
    if parser.parse_args().run:
        run()
