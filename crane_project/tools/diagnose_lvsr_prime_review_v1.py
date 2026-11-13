"""One CPU-only PRIME feasibility audit. No optimizer, fitting, TEST or new cache."""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'work_dirs/lvsr_diagnostic_20261112_v1/shared_spatial_001'
SOURCES = ('crane_project/utils/lvsr_prime_review_v1.py',
           'crane_project/tools/diagnose_lvsr_prime_review_v1.py',
           'tests/test_lvsr_prime_review_v1.py')
SPEC = dict(version='lvsr_prime_cpu_review_20261113_v2', CPU_threads=1,
    CPU_budget_seconds=600, GPU=False, optimizer=False, training_updates=0, TEST=False,
    snapshots=['initial', 'L5', 'L24'], original_inference_parameters=2480,
    proxies=20, embedding_dimension=648, extra_training_parameters=12960,
    embedding='sqrt(common W)-weighted8x9x9 flattened map; L2 normalized eps1e-6; same descriptor as ConR',
    target='canonical long/short log(GT/M), TRAIN only for centers; no standardization',
    target_distance='Euclidean / log1.02', feature_distance='Euclidean of unit-normalized vectors',
    centers='single deterministic farthest-first Lloyd run, max100 iterations; no restarts',
    center_limit_handling='retain identical100-iteration centers; report strict convergence flag, no extra iterations',
    proxy_initialization='He normal, private CPU generator seed1703; same untrained proxies for each snapshot',
    tau_target=1., tau_feature=5., alpha=.005, weight_alignment=.001, weight_proxy=.001,
    settings_basis='C20/tau_feature5/alpha.005 from paper AgeDB base; target scale is prior2% diagnostic window; equal.001 diagnostic weights, not validated training hyperparameters',
    equations='paper1-8; no PRW/CB/LDAM, reweighting or augmentation',
    schedule='original epoch6 natural639 batches; full TRAIN/VAL feature gradients',
    parameter_gradient_batches='first16 natural batches plus every batch containing M joint error>10%',
    individual_stem_gradients='all difficult TRAIN views and first16 natural batches; overlap counted once',
    output_responses='first16 natural batches; Jacobian times unit auxiliary gradient, no Adam update',
    replay_atol=1e-4, replay_rtol=1e-6, effective_gradient_threshold=1e-12,
    adoption=False, download='grouped JSON/CSV only; no weights, embeddings, rows or gradient arrays')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def table(path, rows):
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def group_keys(record, target, error):
    for group in ('overall', record['domain'], record['sequence']):
        for j, edge in enumerate(('long', 'short')):
            subsets = ['all', 'shrink' if target[j] < math.log(.99) else
                       'enlarge' if target[j] > math.log(1.01) else 'near']
            if error[j] <= .03: subsets.append('accurate_M_le3pct')
            if error[j] > .1: subsets.append('difficult_M_gt10pct')
            for subset in subsets:
                yield group, edge, subset


def describe(values):
    if not values:
        return dict(n=0, mean=None, min=None, max=None, median=None)
    values = sorted(values)
    n = len(values)
    return dict(n=n, mean=sum(values)/n, min=values[0], max=values[-1],
                median=(values[(n-1)//2]+values[n//2])/2)


def gradient_relation(size, auxiliary):
    na, nb = float(size.norm()), float(auxiliary.norm())
    dot = float(size.dot(auxiliary))
    return dict(size_norm=na, weighted_alignment_norm=nb,
                ratio=nb/na if na else None, cosine=dot/(na*nb) if na and nb else None)


def main(args):
    hashes = {name: sha(ROOT/name) for name in SOURCES}
    if args.stage == 'check':
        print(json.dumps(dict(passed=True, spec=SPEC, source_SHA=hashes, run_started=False)))
        return
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    started = time.monotonic()
    import torch
    torch.set_num_threads(1)
    from types import SimpleNamespace
    from crane_project.utils import lvsr_prime_review_v1 as prime
    from crane_project.utils import port_size_qalign_v1 as model
    from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead
    from crane_project.utils import symeood_r4_lvsr_protocol_v1 as protocol
    from crane_project.tools import train_symeood_r4_lvsr_pair_v1 as original
    from crane_project.tools.run_symeood_110_lvsr_v1 import schedule_indices
    from crane_project.tools import diagnose_lvsr_direction_representation_v1 as common
    prefix = 'prime_review_v2'
    assert not (OUT/(prefix+'_protocol.json')).exists(), 'No automatic repeat'
    save(OUT/(prefix+'_protocol.json'), dict(spec=SPEC, source_SHA=hashes))
    old_optimizer = torch.optim.Optimizer.__init__
    old_handler = signal.getsignal(signal.SIGALRM)

    def forbidden(*unused, **kw):
        raise RuntimeError('PRIME review forbids optimizer creation')

    def timeout(*unused):
        raise RuntimeError('Fixed600-second CPU budget exceeded')

    torch.optim.Optimizer.__init__ = forbidden
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, max(1., 600-(time.monotonic()-started)))
    try:
        identity = protocol.checked_sources()
        train, val, input_receipt = original.load_inputs(protocol.paths(), identity, SimpleNamespace(torch=torch))
        digest = common.records_digest(train+val)
        present = [r for r in train if len(r['midpoint_original'])]
        assert len(train) == 5116 and len(present) == 5112 and len(val) == 887
        assert all(len(r['midpoint_original']) == 1 for r in present+val)
        old_root = protocol.BASE/'lvsr/attempt_001/train'
        old_selection = read(old_root/'selection.json')
        assert old_selection['selected_epoch'] is None
        pins = [protocol.paths()['inputs']/name for name in ('train.pt', 'val.pt', 'completion.json')]
        pins += [ROOT/protocol.parent.R4_CHECKPOINT, protocol.midpoint_directory()/'train/head_epoch_20.pth',
                 old_root/'head_epoch_05.pth', old_root/'head_epoch_24.pth', old_root/'selection.json']
        for snapshot in ('L5', 'L24'):
            pins.append(OUT.parent/'attempt_001'/(snapshot+'_all.rows.jsonl.gz'))
        pinned = {str(path.relative_to(ROOT)): sha(path) for path in pins}
        assert sha(old_root/'head_epoch_05.pth') == 'b272c50dfe6329414803d5f22c6e1e84d9c3db37b6bb2e31f1c7ed140f662c3e'
        assert sha(old_root/'head_epoch_24.pth') == '4e6baa0dd4bf568db68f63f39a835c2cbc4bb0ba08f2358b0ff3ed9a9ac8c5b3'
        metadata = {(r['image'], r['scale']): r for r in common.metadata(present+val)}
        tensors = torch.cat([prime.canonical_pair(model.residual.size_targets(r['gt_original'],
            r['midpoint_original'], r['boxes_model'])['log_residual'], r['boxes_model'], r['midpoint_original'])
            for r in present+val])
        direct = torch.tensor([metadata[r['image'], r['scale']]['target'] for r in present+val], dtype=tensors.dtype)
        assert torch.allclose(tensors, direct, atol=1e-7, rtol=1e-6)
        axis_max = float((tensors-direct).abs().max())
        centers, kmeans = prime.kmeans_targets(tensors[:5112], SPEC['proxies'])
        assert not centers.requires_grad
        inspection = read(OUT/'prime_review_clustering_inspection.json')
        center_sha = hashlib.sha256(centers.numpy().tobytes()).hexdigest()
        assert center_sha == inspection['centers_float32_sha'], 'The original100-iteration centers changed'
        center_directions = [[('shrink' if q < math.log(.99) else 'enlarge' if q > math.log(1.01) else 'near')
                              for q in row] for row in centers.tolist()]
        order = schedule_indices(5112, 6)
        ordered_train = [present[i] for i in order]
        assert sorted(order) == list(range(5112))
        order_sha = hashlib.sha256(json.dumps(order, separators=(',', ':')).encode()).hexdigest()
        target_map = {(r['image'], r['scale']): tensors[i] for i, r in enumerate(present+val)}
        feature_groups = defaultdict(list)
        individual_groups = defaultdict(list)
        natural_gradients, responses = [], defaultdict(list)
        snapshot_summary = {}
        forward_views = 0
        for snapshot, checkpoint in (('initial', None), ('L5', old_root/'head_epoch_05.pth'), ('L24', old_root/'head_epoch_24.pth')):
            torch.manual_seed(1703)
            head = PairSpecificSizeHead().cpu().train()
            if checkpoint:
                payload = torch.load(str(checkpoint), map_location='cpu')
                head.load_state_dict(payload['head_state'], strict=True)
                del payload
            initial_state = common.tensor_digest(head.state_dict())
            params = list(head.parameters()); names = [name for name, _ in head.named_parameters()]
            stem_indices = [i for i, name in enumerate(names) if name.startswith('stem')]
            proxy = prime.new_proxy_features(20, 648)
            proxy_state = proxy.detach().clone()
            flat = lambda values: torch.cat([q.reshape(-1) for q in values]).double()
            proxy_objective, proxy_parts = prime.proxy_loss(proxy, centers)
            proxy_gradient = torch.autograd.grad(.001*proxy_objective, proxy)[0].detach()
            reference = common.rows(OUT.parent/'attempt_001'/(snapshot+'_all.rows.jsonl.gz')) if checkpoint else None
            max_replay = 0.; gradient_batches = 0; difficult_effective = []; relations = defaultdict(list)
            with torch.no_grad():
                for record in train:
                    if not len(record['midpoint_original']):
                        output = head(*common.batch([record])[:-1])
                        assert len(output['boxes_original']) == 0
                        forward_views += 1
            for split, records in (('train', ordered_train), ('val', val)):
                for start in range(0, len(records), 8):
                    rr = records[start:start+8]
                    x = common.batch(rr)
                    captured = []
                    hook = head.stem.register_forward_hook(lambda module, inputs, output: captured.append(output))
                    try: output = head(*x[:-1])
                    finally: hook.remove()
                    forward_views += len(rr)
                    features = captured[0]
                    z, embedding_norm = prime.embedding(features, output['common_support'])
                    target = torch.stack([target_map[r['image'], r['scale']] for r in rr])
                    error = torch.tensor([metadata[r['image'], r['scale']]['M_error'] for r in rr])
                    difficult = (error > .1).any(1)
                    want_gradient = split == 'train' and (start < 128 or bool(difficult.any()))
                    alignment, t, log_a = prime.alignment_per_sample(z, target, proxy, centers)
                    feature_gradient = torch.autograd.grad(alignment.sum(), features, retain_graph=want_gradient)[0]
                    assert torch.isfinite(feature_gradient).all()
                    signal_norm = feature_gradient.flatten(1).norm(dim=1)
                    pred = prime.canonical_pair(output['delta_roi'], x[3], x[-2])
                    dt = (target[:, None]-target[None, :]).abs().amax(-1)/prime.WINDOW
                    dp = (pred.detach()[:, None]-pred.detach()[None, :]).abs().amax(-1)/prime.WINDOW
                    other = ~torch.eye(len(rr), dtype=torch.bool)
                    conr_active = ((dt <= 1)&other).any(1) & ((dt > 1)&(dp <= 1)&other).any(1)
                    nearest = ((target[:, None]-centers[None, :]).abs().amax(-1)).amin(1)
                    for j, record in enumerate(rr):
                        key = record['image'], record['scale']
                        assert torch.equal(output['boxes_original'][j, [0, 1, 4, 5]], x[-2][j, [0, 1, 4, 5]])
                        if snapshot == 'initial':
                            assert torch.equal(output['boxes_original'][j], x[-2][j])
                        else:
                            expected = torch.tensor(reference[key]['edge_residual'], dtype=x[-2].dtype)
                            maximum = float((output['boxes_original'][j]-expected).abs().max())
                            max_replay = max(max_replay, maximum)
                            assert torch.allclose(output['boxes_original'][j], expected, atol=1e-4, rtol=1e-6)
                        values = dict(effective=int(signal_norm[j] > 1e-12), conr=int(conr_active[j]),
                            zero=int(embedding_norm[j] <= prime.EPS), norm=float(signal_norm[j]),
                            entropy=float(-(t[j]*t[j].clamp_min(1e-30).log()).sum()),
                            top=float(t[j].max()), nearest=float(nearest[j]), loss=float(alignment[j].detach()))
                        view_split = 'val' if split == 'val' else 'train_s1' if record['scale'] == 1. else 'train_s05'
                        for cell in group_keys(record, target[j].tolist(), error[j].tolist()):
                            feature_groups[(snapshot, view_split) + cell].append(values)
                    if want_gradient:
                        gradient_batches += 1
                        size = model.size_loss(output, x[-1], x[-2], x[3])
                        gs = torch.autograd.grad(size, params, retain_graph=True)
                        ga = torch.autograd.grad(.001*alignment.mean(), params, retain_graph=True, allow_unused=True)
                        ga = [torch.zeros_like(q) if g is None else g for q, g in zip(params, ga)]
                        assert all(float(g.abs().sum()) == 0 for name, g in zip(names, ga) if name.startswith('terminal'))
                        align_proxy_gradient = torch.autograd.grad(.001*alignment.mean(), proxy, retain_graph=True)[0]
                        for block in ('all', 'stem', 'terminal'):
                            indices = [i for i, name in enumerate(names) if block == 'all' or name.startswith(block)]
                            relation = gradient_relation(flat([gs[i] for i in indices]), flat([ga[i] for i in indices]))
                            for metric in ('ratio', 'cosine'):
                                if relation[metric] is not None: relations[block+'/'+metric].append(relation[metric])
                            if start < 128:
                                natural_gradients.append(dict(snapshot=snapshot, batch=start//8+1, block=block,
                                    size_loss=float(size.detach()), alignment_loss=float(alignment.mean().detach()),
                                    proxy_alignment_gradient_norm=float(align_proxy_gradient.norm()),
                                    proxy_total_gradient_norm=float((align_proxy_gradient+proxy_gradient).norm()), **relation))
                        a, b = flat(gs), flat(ga)
                        for j, record in enumerate(rr):
                            if start < 128 or bool(difficult[j]):
                                individual = torch.autograd.grad(.001*alignment[j], [params[i] for i in stem_indices], retain_graph=True)
                                norm = float(flat(individual).norm())
                                if bool(difficult[j]): difficult_effective.append(norm > 1e-12)
                                for cell in group_keys(record, target[j].tolist(), error[j].tolist()):
                                    individual_groups[(snapshot,) + cell].append(norm)
                            if start < 128:
                                for edge in range(2):
                                    jac = flat(torch.autograd.grad(pred[j, edge], params, retain_graph=True))
                                    delta = -float(jac.dot(b))/float(b.norm()) if float(b.norm()) else 0.
                                    derivative = float(((pred[j, edge]-target[j, edge])/.1).clamp(-1, 1).detach())
                                    name = 'long' if edge == 0 else 'short'
                                    direction = 'shrink' if target[j, edge] < math.log(.99) else 'enlarge' if target[j, edge] > math.log(1.01) else 'near'
                                    responses[snapshot, name, direction].append((delta, derivative*delta))
                        assert not any(q.grad is not None for q in params+[proxy])
                    if start % 512 == 0:
                        print(json.dumps(dict(stage='signals', snapshot=snapshot, split=split,
                            done=min(start+8, len(records)), elapsed=time.monotonic()-started)), flush=True)
            assert common.tensor_digest(head.state_dict()) == initial_state and torch.equal(proxy, proxy_state)
            assert len(head.stem._forward_hooks) == 0
            snapshot_summary[snapshot] = dict(parameters=2480, proxies_extra=12960,
                state_unchanged=True, proxy_unchanged=True, replay_max_abs=max_replay,
                gradient_batches=gradient_batches, difficult_TRAIN_views=len(difficult_effective),
                difficult_actual_stem_effective=sum(difficult_effective),
                proxy_loss=float(proxy_objective.detach()), proxy_KL=float(proxy_parts['KL']),
                proxy_spread=float(proxy_parts['spread']), proxy_gradient_norm=float(proxy_gradient.norm()),
                relations={key: describe(value) for key, value in relations.items()})
            print(json.dumps(dict(stage='snapshot_complete', snapshot=snapshot,
                difficult_effective=sum(difficult_effective), elapsed=time.monotonic()-started)), flush=True)
        feature_rows = []
        for key, values in sorted(feature_groups.items()):
            feature_rows.append(dict(zip(('snapshot', 'split', 'group', 'edge', 'subset'), key),
                n=len(values), proxy_effective=sum(v['effective'] for v in values), conr_active=sum(v['conr'] for v in values),
                zero_embeddings=sum(v['zero'] for v in values),
                feature_gradient_norm_mean=sum(v['norm'] for v in values)/len(values),
                feature_gradient_norm_max=max(v['norm'] for v in values),
                target_entropy_mean=sum(v['entropy'] for v in values)/len(values),
                target_top_probability_mean=sum(v['top'] for v in values)/len(values),
                nearest_proxy_Linf_log_mean=sum(v['nearest'] for v in values)/len(values),
                alignment_loss_mean=sum(v['loss'] for v in values)/len(values)))
        individual_rows = [dict(zip(('snapshot', 'group', 'edge', 'subset'), key),
            n=len(values), actual_stem_effective=sum(v > 1e-12 for v in values), norm_mean=sum(values)/len(values),
            norm_min=min(values), norm_max=max(values)) for key, values in sorted(individual_groups.items())]
        response_rows = [dict(zip(('snapshot', 'edge', 'direction'), key), n=len(values),
            unit_aux_logsize_response_mean=sum(v[0] for v in values)/len(values),
            unit_aux_size_derivative_mean=sum(v[1] for v in values)/len(values),
            local_size_harmed=sum(v[1] > 1e-12 for v in values)) for key, values in sorted(responses.items())]
        assert all(sha(ROOT/name) == digest for name, digest in pinned.items())
        assert common.records_digest(train+val) == digest
        assert protocol.checked_sources() == identity
        assert all(sha(ROOT/name) == digest for name, digest in hashes.items())
        assert not torch.cuda.is_initialized()
        summary = dict(spec=SPEC, source_SHA=hashes, snapshots=snapshot_summary,
            training_views=5112, VAL_views=887, absent_TRAIN_views=4, axis_target_max_abs=axis_max,
            centers_TRAIN_only=centers.tolist(), center_directions=center_directions, kmeans=kmeans,
            same100_iteration_centers_SHA256=center_sha,
            original_stopped_review='prime_review_protocol.json; prime_review_failure.json',
            sample_order_sha256=order_sha, proxy_parameters=12960, inference_parameters=2480,
            combined_training_parameters=15440, equations='visually checked paper pages3-4, Eq1-8',
            author_code_confirmed=False, gradients_with_untrained_proxies=True,
            feature_rows=feature_rows, individual_stem_rows=individual_rows, natural_gradients=natural_gradients,
            first_order_responses=response_rows, inference_graph_modified=False, performance_evidence=False)
        save(OUT/(prefix+'_summary.json'), summary)
        table(OUT/(prefix+'_signals.csv'), feature_rows)
        table(OUT/(prefix+'_stem.csv'), individual_rows)
        table(OUT/(prefix+'_natural_gradients.csv'), natural_gradients)
        table(OUT/(prefix+'_responses.csv'), response_rows)
        save(OUT/(prefix+'_completion.json'), dict(passed=True, CPU_only=True, CUDA_initialized=False,
            optimizer_created=False, training_updates=0, TEST_access=False,
            weights_unchanged=True, proxies_unchanged=True, inputs_unchanged=True,
            original_selection_unchanged=True, source_SHA=hashes, pinned_SHA=pinned,
            forward_views=forward_views, elapsed_seconds=time.monotonic()-started,
            adopted=False, formal_training_started=False))
        suffixes = ['protocol.json', 'summary.json', 'signals.csv', 'stem.csv', 'natural_gradients.csv',
                    'responses.csv', 'completion.json']
        files = [prefix+'_'+suffix for suffix in suffixes]
        save(OUT/(prefix+'_manifest.json'), {name: sha(OUT/name) for name in files})
        print(json.dumps(dict(passed=True, elapsed_seconds=time.monotonic()-started,
                             snapshots=snapshot_summary)), flush=True)
    except BaseException as error:
        save(OUT/(prefix+'_failure.json'), dict(error=repr(error), updates=0, automatic_retry=False))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        torch.optim.Optimizer.__init__ = old_optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('check', 'run'), default='check')
    main(parser.parse_args())
