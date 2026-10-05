#!/usr/bin/env python3
"""Fixed size-only finite learning: check -> discarded2-step smoke ->200 steps.

Reuse sealed CPU ROIs and sigma1.5/epoch03 only. No detector construction,
image/dataset reads, feature extraction, old selector, TEST or formal training.
check is standard-library-only; smoke/finite require the existing CUDA torch
environment. No CLI for tuning seeds, budgets, architecture or thresholds.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import diagnose_port_geometry_midpoint_edge_feasibility_v1 as sealed
from crane_project.utils import port_geometry_midpoint_edge_residual_metrics_v1 as metrics

VERSION = 'port_geometry_midpoint_edge_residual_v1_finite_train_val'
PREFIX = 'crane_project/tools/port_geometry_midpoint_edge_residual_v1'
PROTOCOL = ROOT/(PREFIX+'_protocol.json')
SOURCES = ROOT/(PREFIX+'_sources.json')
PARENT = ROOT/sealed.SOURCES_NAME
PARENT_SHA = '3909ad63d284afc3511bae8810b42d824cd6127ff8ba9675d292ba0f7e7fc1af'
SETTINGS = dict(seed=1703, steps=200, smoke_steps=2, batch_size=8, lr=.001,
    weight_decay=0., clip_norm=10., scales=[1., .5], per_domain_scale=2,
    max_cpu_tensor_bytes=1024**3)
MODEL_SETTINGS = dict(seed=1703, in_channels=256, hidden_channels=8, roi_size=9,
    loss_beta=.1, edge_ratio_min=.8, edge_ratio_max=1.25, direction_atol_rad=1e-6)
NEW_SOURCES = {
    PREFIX+'_protocol.json',
    'crane_project/tools/preflight_port_geometry_midpoint_edge_residual_v1.py',
    'crane_project/utils/port_geometry_midpoint_edge_residual_v1.py',
    'crane_project/utils/port_geometry_midpoint_edge_residual_metrics_v1.py',
    'tests/test_port_geometry_midpoint_edge_residual_v1.py',
    sealed.SOURCES_NAME, 'crane_project/tools/eval_crane_offline.py'}


def protocol_document():
    return dict(protocol=VERSION, stages=['check', 'smoke', 'finite'], settings=SETTINGS,
        model_settings=MODEL_SETTINGS, parameter_count=2818,
        fixed_checkpoint=sealed.FIXED, cache_manifest_sha256=sealed.CACHE_SHA,
        parent_sources_sha256=PARENT_SHA, tolerances=metrics.TOLERANCES,
        architecture='Conv1x1(259,8,bias),ReLU,Conv3x3(8,8,pad1,bias),ReLU. Mean over y/x gives separate8x9 u/v profiles; each concat4 geometry ->Linear76to1. Two output weights/biases zero.2818 parameters.',
        geometry_inputs='log(M_roi_u/B_original_roi_u),log(M_roi_v/B_original_roi_v),log(B_model_u/context_u),log(B_model_v/context_v). Original and model B swaps stay separate; model B alone defines ROI axes.',
        objective='Mean two-axis SmoothL1(delta_roi-log(GT_roi/M_roi),beta.1), before zero/fallback delivery. GT canonical long/short follow frozen M raw order, then model-B swap. Old fixed-B cyclic matched target diagnostic only. No target clipping, size-error filtering or joint/temporal objective.',
        online='GT-free frozen ROI/support/B raw/B model/actual sx-sy/frozen M -> independent residuals. Copy M center/raw theta/score/output count. Exact zero and exact-square M bypass. Nominal original-B raw edge ratio[.8,1.25], frozen M raw order/canonical angle, original center/angle guards and native-dtype four-midpoint ROI bound. Any failure falls back whole pair to M.',
        sampling='Existing sealed64 TRAIN identities and roles: per-domain24 fit+8 probe, paired1.0/.5. Only inherited B-present center<15px fit views enter gradients, no replacements. Deterministic shuffled cyclic queue in each domain/scale,2 slots per queue per step; 1600 slots, not1600 unique images.',
        smoke='GPU2 updates, native parameter/optimizer/Python+torch CPU/CUDA RNG save/reload, exact prediction replay and gradient-path checks. TRAIN128 views only, no VAL tensor/evaluation. Existing selected VAL text is read for source binding only. Smoke weights/updates never initialize finite training.',
        finite='New seed1703 initialization,200 updates only. Evaluate neutral and final200 on TRAIN128 views and full standard VAL887. Indexed fixed M VAL reproduction precedes any new update. No intermediate weight selection, repeats or tuning. Save experimental independent weights on server only.',
        metrics='Same strict RIoU and contiguous sequence formulas in eval_crane_offline.py, canonical offline view only. Unrounded sequence values/event counts gate ACI/A-RMSE/DFR/MCML/MRF; original rounded output also retained. Three center denominators, conditional/full joint sizes, paired recovery/new failures, MAE/P95/max and signed errors.',
        gates='A003: preserve frozen state/center/raw+canonical angle/score/output. VAL both-domain joint-size counts strictly increase, each sequence does not decline; center+size coverage and TRAIN probe per-domain/scale do not decline. Every-domain/sequence two-edge MAE/P95/max nonincrease. All-frame meanRIoU nondecrease, no newRIoU<.5 or longer failure run. DFR/MCML max+mean/MRF nonincrease; ACI/A-RMSE and adjacent-pair/penalty counts identical. TDR_w10 denominator identical and hits nondecrease, MCML_limit5 nondecrease; MRF undefined pattern retained. Require engineering proof. Failure keeps M, never auto extends budget.',
        scope=dict(detector_updates=0, frozen_midpoint_updates=0,
            detector_constructed=False, feature_extractions=0, cache_rebuilt=False,
            test_access=False, selection_on_test=False, automatic_promotion=False,
            formal_training_approved=False, paper_modified=False, reliability_migrated=False),
        limitations=['Finite candidate is not measured improvement or a root-cause conclusion.',
            'TRAIN probe was used previously and is not independent generalization.',
            'Size loss does not guarantee RIoU/continuity/depth precision.',
            'Live detector/online image pipeline is not rerun; use sealed cache and GT-free head contract.',
            'GPU/weights/effect cannot be certified by local static tests.'])


def checked_sources():
    if sealed.sha(PARENT) != PARENT_SHA:
        raise ValueError('Fixed feasibility parent source identity differs')
    parent = sealed.checked_sources()
    manifest = sealed.read(SOURCES)
    required = set(parent['sources']) | NEW_SOURCES
    if (manifest['protocol'] != VERSION or manifest['parent_sources_sha256'] != PARENT_SHA or
            set(manifest['sources']) != required or sealed.read(PROTOCOL) != protocol_document()):
        raise ValueError('Independent-size source/protocol scope differs')
    for name, digest in manifest['sources'].items():
        if sealed.sha(ROOT/name) != digest:
            raise ValueError('Independent-size source SHA differs: '+name)
    if any(manifest['sources'][k] != v for k, v in parent['sources'].items()):
        raise ValueError('Fixed parent source closure differs')
    return dict(protocol_sha256=sealed.sha(PROTOCOL), sources_sha256=sealed.sha(SOURCES),
                sources=manifest['sources'], parent_identity=parent)


def select_views(records, roles):
    """Only sealed identity/role metadata determines selection; never geometry."""
    selected = []
    for r in records:
        if r['image'] not in roles:
            continue
        sample = roles[r['image']]
        if any(r[k] != sample[k] for k in ('domain', 'sequence', 'split')) or r['role'] != 'train':
            raise ValueError('Fixed TRAIN sample identity differs')
        selected.append(dict(r, sample_role=sample['role']))
    if (len(selected) != 128 or len({(r['image'], r['scale']) for r in selected}) != 128 or
            Counter((r['domain'], r['sample_role'], r['scale']) for r in selected) != Counter({
                (d, role, scale): n for d in ('real', 'sim') for role, n in (('fit',24), ('probe',8))
                for scale in SETTINGS['scales']})):
        raise ValueError('Missing/duplicate fixed64 paired TRAIN views; do not resample')
    return sorted(selected, key=lambda r: (r['scale'], r['image']))


def schedule(records):
    pools = {(d, scale): [i for i, r in enumerate(records) if r['sample_role'] == 'fit'
        and r['eligible'] and r['domain'] == d and r['scale'] == scale]
        for d in ('real', 'sim') for scale in SETTINGS['scales']}
    if any(len(values) < 2 for values in pools.values()):
        raise ValueError('Insufficient inherited eligible fit views; no resampling')
    rng = random.Random(SETTINGS['seed'])
    order = {key: rng.sample(values, len(values)) for key, values in pools.items()}
    offsets = {key: 0 for key in pools}
    result = []
    for _ in range(SETTINGS['steps']):
        indices = []
        for key in pools:
            for _ in range(SETTINGS['per_domain_scale']):
                if offsets[key] == len(order[key]):
                    order[key] = rng.sample(pools[key], len(pools[key]))
                    offsets[key] = 0
                indices.append(order[key][offsets[key]])
                offsets[key] += 1
        result.append(indices)
    return result


def load_records(cache_dir, cache, roles, torch, original, include_val):
    train, val, bytes_read, support = [], [], 0, {}
    for label in ('train_s1', 'train_s05') + (('val_s1',) if include_val else ()):
        path = cache_dir/(label+'.pt')
        if sealed.sha(path) != cache['files'][path.name]:
            raise ValueError('Sealed ROI shard SHA differs: '+path.name)
        payload = sealed.torch_load(torch, path)
        expected, role, _ = sealed.SHARDS[label]
        rows = payload['records']
        if (payload['identity'] != cache['identity'] or payload['detector_state'] != cache['detector_state'] or
                len(rows) != expected or len({r['image'] for r in rows}) != expected or
                Counter(r['sequence'] for r in rows) != Counter(sealed.TRAIN_COUNTS if role == 'train' else sealed.VAL_COUNTS)):
            raise ValueError('Cached source/state/frame counts differ: '+label)
        for r in rows:
            bytes_read += sealed.checked_record(r, label, torch, original)
            if bool((r['gt_original'][:, 2:4] <= 0).any()):
                raise ValueError('Nonpositive cached GT edges')
        if bytes_read > SETTINGS['max_cpu_tensor_bytes']:
            raise ValueError('CPU tensor budget exceeded')
        support[label] = dict(total=len(rows), inherited_eligible=sum(r['eligible'] for r in rows),
                              shard_sha256=cache['files'][path.name])
        if role == 'train':
            train.extend(dict(r, shard=label) for r in rows if r['image'] in roles)
        else:
            val.extend(dict(r, shard=label, sample_role='val') for r in rows)
        del payload, rows
    chosen = select_views(train, roles)
    support['selected'] = dict(identities=64, views=len(chosen), fit_views=sum(r['sample_role'] == 'fit' for r in chosen),
        probe_views=sum(r['sample_role'] == 'probe' for r in chosen), eligible=sum(r['eligible'] for r in chosen),
        excluded=[dict(image=r['image'], scale=r['scale'], role=r['sample_role']) for r in chosen if not r['eligible']])
    return chosen, sorted(val, key=lambda r: (r['sequence'], r['frame_id'])), support, bytes_read


def prepare_frozen(records, head, device, torch, original, model, references):
    result = []
    with torch.no_grad():
        for r in records:
            roi, support, b, bm, xy = [r[k].to(device) for k in sealed.TENSOR_KEYS[:-1]]
            output = head(roi, support, b, bm, xy)
            m = output['boxes_original']
            if (m.shape != b.shape or m.requires_grad or not torch.equal(m[:, 5], b[:, 5]) or
                    not all(sealed.native_source_guards(torch, original, b, m).values())):
                raise ValueError('Frozen M output/source guard differs')
            pred = m[0].cpu().tolist() if len(m) else None
            source = dict(image=r['image'], sequence=r['sequence'], domain=r['domain'], frame_id=r['frame_id'],
                scale=r['scale'], gt=r['gt_original'][0].tolist(), b=b[0].cpu().tolist() if len(b) else None,
                midpoint=pred, diagnosis=dict(baseline_joint10=False, baseline_center_hit=False))
            if pred is not None:
                source['diagnosis'].update(
                    baseline_joint10=sealed.geometry.joint_correct(pred[2:4], sealed.geometry.goal_edges(source['gt'], pred)),
                    baseline_center_hit=math.hypot(pred[0]-source['gt'][0], pred[1]-source['gt'][1]) < 15.)
            if r['shard'] == 'val_s1':
                sealed.compare_reference(source, references[r['image']])
                decision = bool(output['accepted'][0]) if len(m) else None
                if decision != references[r['image']]['accepted']:
                    raise ValueError('Fixed indexed M VAL fallback decision differs')
            diagnostic = None
            if len(m):
                gt = r['gt_original'].to(device)
                target = model.size_targets(gt, m, bm)
                old = original.target_points(gt, b, bm, xy)
                points = old['original']
                pair = torch.stack(((points[:, 0]-points[:, 2]).norm(dim=1),
                                    (points[:, 1]-points[:, 3]).norm(dim=1)), -1)
                raw_old = torch.where((b[:, 2] < b[:, 3])[:, None], pair.flip(-1), pair)
                diagnostic = dict(gt_raw_edges=target['raw_edges'][0].cpu().tolist(),
                    gt_roi_edges=target['roi_edges'][0].cpu().tolist(),
                    log_residual_target=target['log_residual'][0].cpu().tolist(),
                    fixed_b_matched_raw_edges=raw_old[0].cpu().tolist(),
                    fixed_b_permutation=int(old['permutation'][0]),
                    fixed_b_target_m_order_conflict=bool(((raw_old[:, 0] < raw_old[:, 1]) != (m[:, 2] < m[:, 3]))[0]),
                    fixed_b_target_axis_difference=not bool(torch.isclose(raw_old, target['raw_edges'], rtol=1e-6, atol=1e-4).all()),
                    fixed_b_target_max_relative_difference=float(((raw_old/target['raw_edges'])-1.).abs().max()),
                    original_b_swap=bool((b[:, 2] < b[:, 3])[0]), model_b_swap=bool((bm[:, 2] < bm[:, 3])[0]),
                    m_raw_swap=bool((m[:, 2] < m[:, 3])[0]),
                    gt_aspect_ratio=float(gt[:,2:4].max()/gt[:,2:4].min()),
                    gt_exact_square=bool((gt[:,2] == gt[:,3])[0]),
                    exact_target_m_order_conflict=bool(((target['raw_edges'][:,0] < target['raw_edges'][:,1]) != (m[:,2] < m[:,3]))[0]),
                    exact_target_outside_original_b=bool(((target['raw_edges']/b[:, 2:4] < .8) |
                        (target['raw_edges']/b[:, 2:4] > 1.25)).any()))
            result.append(dict(r, midpoint_original=m.detach().cpu().clone(), target_diagnostic=diagnostic,
                               frozen_m_accepted=bool(output['accepted'][0]) if len(m) else None))
    return result


def batch(records, indices, device, torch):
    if any(records[i]['sample_role'] != 'fit' or not records[i]['eligible'] for i in indices):
        raise ValueError('Only inherited eligible fit views enter gradients')
    if (len(indices) != 8 or Counter((records[i]['domain'], records[i]['scale']) for i in indices) !=
            Counter({(d, scale): 2 for d in ('real','sim') for scale in SETTINGS['scales']})):
        raise ValueError('Fixed batch8 needs two fit views per domain/scale')
    keys = sealed.TENSOR_KEYS[:-1]+('midpoint_original', 'gt_original')
    return [torch.cat([records[i][key] for i in indices]).to(device) for key in keys]


def seed_all(torch):
    random.seed(SETTINGS['seed'])
    torch.manual_seed(SETTINGS['seed'])
    torch.cuda.manual_seed_all(SETTINGS['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Keep the fixed-parent numeric policy; do not change TF32 flags and
    # silently alter frozen M replay. Actual flags are recorded in runtime.


def update(head, optimizer, frozen_head, records, indices, device, torch, model):
    owned = [id(p) for group in optimizer.param_groups for p in group['params']]
    if len(owned) != len(set(owned)) or set(owned) != {id(p) for p in head.parameters()}:
        raise ValueError('Optimizer must own only independent size parameters')
    roi, support, b, bm, xy, m, gt = batch(records, indices, device, torch)
    optimizer.zero_grad()
    output = head(roi, support, b, bm, xy, m)
    loss = model.size_loss(output['delta_roi'], gt, m, bm)
    if not bool(torch.isfinite(loss)):
        raise ValueError('Nonfinite size loss')
    loss.backward()
    if any(p.grad is not None or p.requires_grad for p in frozen_head.parameters()):
        raise ValueError('Frozen midpoint received gradients')
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in head.parameters()):
        raise ValueError('Missing/nonfinite independent size gradient')
    norm = lambda params: math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in params))
    stem, terminal = norm(head.stem.parameters()), norm(list(head.u.parameters())+list(head.v.parameters()))
    clipped = float(torch.nn.utils.clip_grad_norm_(head.parameters(), SETTINGS['clip_norm']))
    if not all(math.isfinite(v) for v in (stem, terminal, clipped)):
        raise ValueError('Nonfinite gradient norm')
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in head.parameters()):
        raise ValueError('Nonfinite updated size parameters')
    return dict(loss=float(loss.detach()), stem_grad_norm=stem, terminal_grad_norm=terminal,
                total_grad_norm_before_clip=clipped)


def evaluate_rows(head, records, device, torch):
    head.eval()
    result = []
    with torch.no_grad():
        for r in records:
            tensors = [r[k].to(device) for k in sealed.TENSOR_KEYS[:-1]+('midpoint_original',)]
            output = head(*tensors)
            b, m, c = tensors[2], tensors[-1], output['boxes_original']
            present = bool(len(m))
            status = ('no_output' if not present else 'square' if bool(output['square_bypass'][0]) else
                      'neutral' if bool(output['neutral_bypass'][0]) else 'accepted' if bool(output['accepted'][0]) else 'fallback')
            valid = present and bool(output['candidate_valid'][0])
            row = dict(image=r['image'], sequence=r['sequence'], domain=r['domain'], frame_id=r['frame_id'],
                scale=r['scale'], shard=r['shard'], split=r['split'], sample_role=r['sample_role'],
                eligible_train=r['eligible'], gt=r['gt_original'][0].tolist(),
                b=b[0].cpu().tolist() if present else None, midpoint=m[0].cpu().tolist() if present else None,
                edge_residual=c[0].cpu().tolist() if present else None,
                candidate=output['candidate_original'][0].cpu().tolist() if valid else None,
                delta_roi=output['delta_roi'][0].cpu().tolist() if present and bool(torch.isfinite(output['delta_roi'][0]).all()) else None,
                delta_raw=output['delta_raw'][0].cpu().tolist() if present and bool(torch.isfinite(output['delta_raw'][0]).all()) else None,
                residual_nonfinite=present and not bool(torch.isfinite(output['delta_roi'][0]).all()),
                size_delivery=status, failed_checks=[k for k, v in output['checks'].items() if present and not bool(v[0])],
                frozen_m_accepted=r['frozen_m_accepted'], target_diagnostic=r['target_diagnostic'])
            if c.shape != m.shape or not metrics.frozen_identity(row):
                raise ValueError('Independent size output changed frozen identity')
            result.append(row)
    return result


def write_rows(path, rows):
    with path.open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n')


def rng_state(torch):
    return dict(python=random.getstate(), cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(value, torch):
    random.setstate(value['python'])
    torch.set_rng_state(value['cpu'])
    torch.cuda.set_rng_state_all(value['cuda'])


def tree_equal(a, b, torch):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(tree_equal(a[k], b[k], torch) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(tree_equal(x, y, torch) for x, y in zip(a, b))
    return a == b


def replay_views(records):
    """Two existing eligible fit views per domain/scale for native replay."""
    selected = []
    for domain in ('real', 'sim'):
        for scale in SETTINGS['scales']:
            values = [r for r in records if r['sample_role'] == 'fit' and r['eligible']
                      and r['domain'] == domain and r['scale'] == scale]
            if len(values) < 2:
                raise ValueError('Insufficient fixed domain/scale replay views')
            selected.extend(values[:2])
    return selected


def save_reload(directory, stage, updates, head, optimizer, records, identity, proof, device, torch, model):
    """Experimental independent checkpoint only; no merge into B or M."""
    path = directory/('edge_smoke_step_0002.pth' if stage == 'smoke' else 'edge_finite_step_0200.pth')
    replay = replay_views(records)
    before = evaluate_rows(head, replay, device, torch)
    payload = dict(protocol=VERSION, stage=stage, updates=updates, identity=identity, proof=proof,
        head_state={k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
        head_digest=sealed.state_digest(head), optimizer_state=optimizer.state_dict(), rng=rng_state(torch),
        experimental_only=True, automatic_promotion=False, frozen_midpoint_updates=0, detector_updates=0)
    with path.open('xb') as stream:
        torch.save(payload, stream)
    digest = sealed.sha(path)
    restored = sealed.torch_load(torch, path)
    if (restored['protocol'] != VERSION or restored['stage'] != stage or restored['updates'] != updates or
            restored['identity'] != identity or restored['proof'] != proof or
            not tree_equal(payload, restored, torch)):
        raise ValueError('Independent checkpoint serialization differs')
    reloaded = model.IndependentEdgeResidualHead().to(device)
    reloaded.load_state_dict(restored['head_state'], strict=True)
    opt = torch.optim.Adam(reloaded.parameters(), lr=SETTINGS['lr'], weight_decay=SETTINGS['weight_decay'])
    opt.load_state_dict(restored['optimizer_state'])
    restore_rng(restored['rng'], torch)
    if (sealed.state_digest(reloaded) != payload['head_digest'] or
            not tree_equal(optimizer.state_dict(), opt.state_dict(), torch) or
            not tree_equal(restored['rng'], rng_state(torch), torch) or
            before != evaluate_rows(reloaded, replay, device, torch)):
        raise ValueError('Native GPU head/optimizer/RNG/prediction replay differs')
    return reloaded, opt, dict(path=path.name, sha256=digest, updates=updates,
        head_digest=payload['head_digest'], save_reload_exact=True, prediction_replay_views=len(before))


def checked_smoke(path, identity, cache, runtime):
    if not path:
        raise ValueError('finite requires --smoke-report from a successful matching GPU smoke')
    path = Path(path).resolve()
    value = sealed.read(path)
    if (value.get('status') != 'EDGE_RESIDUAL_GPU_SMOKE_COMPLETE_REVIEW_REQUIRED' or
            value.get('checkpoint', {}).get('path') != 'edge_smoke_step_0002.pth'):
        raise ValueError('Successful matching GPU smoke is required, not a failed/unrelated report')
    index = sealed.read(path.parent/'artifacts.json')
    if index['protocol'] != VERSION or any(index['files'].get(name) != sealed.sha(path.parent/name)
        for name in (path.name, 'protocol.json', 'source_identity.json', value['checkpoint']['path'])):
        raise ValueError('Smoke artifact SHA differs')
    if (value['protocol'] != VERSION or value['stage'] != 'smoke' or
            value['status'] != 'EDGE_RESIDUAL_GPU_SMOKE_COMPLETE_REVIEW_REQUIRED' or
            value['source_identity'] != identity or value['cache_manifest_sha256'] != sealed.CACHE_SHA or
            value['runtime'] != runtime or value['size_head_updates'] != 2 or
            value['val_tensor_access'] is not False or value['val_evaluated'] is not False or
            value['state_before']['midpoint'] != sealed.FIXED['head_digest'] or value['state_after']['midpoint'] != sealed.FIXED['head_digest'] or
            value['detector_state'] != cache['detector_state'] or value['checkpoint']['save_reload_exact'] is not True or
            not all(v is True for v in value['engineering'].values()) or not value['engineering'] or
            any(value[k] != expected for k, expected in protocol_document()['scope'].items()) or
            sealed.read(path.parent/'protocol.json') != protocol_document() or
            sealed.read(path.parent/'source_identity.json') != identity):
        raise ValueError('Successful matching fixed GPU smoke is required')
    return value


def publish(directory, completion):
    sealed.write(directory/'completion.json', completion)
    sealed.write(directory/'artifacts.json', dict(protocol=VERSION, files={
        p.name: sealed.sha(p) for p in directory.iterdir() if p.is_file() and p.name != 'artifacts.json'}))


def run(args):
    if args.stage == 'check':
        identity = checked_sources()
        sealed.sample_roles()
        print(json.dumps(dict(status='EDGE_RESIDUAL_SOURCE_CHECK_PASS', checked_source_files=len(identity['sources']),
            parameter_count=2818, settings=SETTINGS, scope=protocol_document()['scope']), sort_keys=True))
        return
    if not args.out_dir:
        raise ValueError('smoke/finite require --out-dir (fresh directory only)')
    directory = Path(args.out_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    completion = dict(protocol=VERSION, stage=args.stage, status='EDGE_RESIDUAL_FAILED',
        started_utc=datetime.now(timezone.utc).isoformat(), size_head_updates=0,
        val_access=False, indexed_selected_val_text_read=False, val_tensor_access=False,
        val_evaluated=False, **protocol_document()['scope'])
    try:
        identity = checked_sources()
        sealed.write(directory/'protocol.json', protocol_document())
        sealed.write(directory/'source_identity.json', identity)
        completion['source_identity'] = identity
        cache_dir = sealed.resolve_cache(args.cache_dir)
        cache = sealed.checked_cache(cache_dir, identity['parent_identity'])
        head_dir = Path(args.head_dir).resolve()
        selection, references, indexed = sealed.checked_selection(head_dir, identity['parent_identity'])
        completion.update(val_access=True, indexed_selected_val_text_read=True)
        import torch
        from crane_project.utils import port_geometry_midpoint_sigma_v1 as frozen_model
        from crane_project.utils import port_geometry_midpoint_edge_residual_v1 as model
        if model.SETTINGS != MODEL_SETTINGS or model.PARAMETER_COUNT != 2818:
            raise ValueError('Fixed independent size model settings differ')
        if os.environ.get('WORLD_SIZE', '1') != '1':
            raise ValueError('Single GPU only; do not use torchrun/DDP')
        device = torch.device(args.device)
        if device.type != 'cuda' or not torch.cuda.is_available():
            raise ValueError('smoke/finite require an available CUDA device')
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        runtime = dict(torch=str(torch.__version__), cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
            device=str(device), gpu=torch.cuda.get_device_name(device), dtype='torch.float32',
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            tf32_matmul=getattr(torch.backends.cuda.matmul, 'allow_tf32', None),
            tf32_cudnn=getattr(torch.backends.cudnn, 'allow_tf32', None))
        seed_all(torch)
        smoke = checked_smoke(args.smoke_report, identity, cache, runtime) if args.stage == 'finite' else None
        frozen = sealed.load_head(torch, frozen_model, head_dir, selection, cache, device)
        original = frozen_model.original
        frozen_before = sealed.state_digest(frozen)
        roles = sealed.sample_roles()
        train, val, support, bytes_read = load_records(cache_dir, cache, roles, torch, original, args.stage == 'finite')
        completion['val_tensor_access'] = bool(val)
        train = prepare_frozen(train, frozen, device, torch, original, model, references)
        val = prepare_frozen(val, frozen, device, torch, original, model, references)
        completion.update(runtime=runtime, support=support, cache_manifest_sha256=sealed.CACHE_SHA,
            detector_state=cache['detector_state'], frozen_checkpoint=sealed.FIXED,
            selected_val_replay_frames=len(val), selected_val_replay_pass=True if val else None,
            inputs=dict(cache_dir=str(cache_dir), head_dir=str(head_dir), indexed_selected_artifacts=indexed),
            input_tensor_bytes=bytes_read)
        batches = schedule(train)
        sealed.write(directory/'schedule.json', dict(seed=1703, steps=200,
            batches=[[dict(image=train[i]['image'], scale=train[i]['scale'], domain=train[i]['domain']) for i in batch] for batch in batches]))
        seed_all(torch)
        head = model.IndependentEdgeResidualHead().to(device)
        optimizer = torch.optim.Adam(head.parameters(), lr=SETTINGS['lr'], weight_decay=SETTINGS['weight_decay'])
        initial_digest = sealed.state_digest(head)
        if smoke is not None and (initial_digest != smoke['state_before']['size_head'] or initial_digest == smoke['state_after']['size_head']):
            raise ValueError('Finite head must restart from identical neutral initialization, never smoke weights')
        proof = dict(identity=identity, selected_artifacts=indexed, cache_manifest_sha256=sealed.CACHE_SHA,
            detector_state=cache['detector_state'], runtime=runtime, fixed_checkpoint=sealed.FIXED)
        initial_rows = evaluate_rows(head, train+val, device, torch)
        if any(r['midpoint'] != r['edge_residual'] for r in initial_rows):
            raise ValueError('Zero-size initialization changed M')
        initial_summary = metrics.evaluate(initial_rows) if val else None
        if val:
            completion['val_evaluated'] = True
            expected = {'real': (374,362,199), 'sim': (512,512,464)}
            for domain, counts in expected.items():
                value = initial_summary['val_s1']['all'][domain]['midpoint']
                actual = tuple(value[k]['numerator'] for k in ('output_coverage','center_correct_full_frame','joint_size10_full_frame'))
                if actual != counts:
                    raise ValueError('Fixed VAL output/center/joint-size counts differ')
            sealed.write(directory/'baseline_summary.json', initial_summary)
        write_rows(directory/'baseline_rows.jsonl', initial_rows)
        completion['state_before'] = dict(midpoint=frozen_before, size_head=initial_digest)
        count = SETTINGS['smoke_steps'] if args.stage == 'smoke' else SETTINGS['steps']
        gradients = []
        head.train()
        with (directory/'progress.jsonl').open('x', encoding='utf-8') as stream:
            for step, indices in enumerate(batches[:count], 1):
                value = update(head, optimizer, frozen, train, indices, device, torch, model)
                completion['size_head_updates'] = step
                gradients.append(value)
                event = dict(stage=args.stage, step=step, total=count, **value)
                stream.write(json.dumps(event, sort_keys=True, allow_nan=False)+'\n')
                stream.flush()
                if step in (1, 2) or step % 25 == 0 or step == count:
                    print('EDGE_RESIDUAL', args.stage, step, '/', count, 'loss', value['loss'], flush=True)
        if gradients[0]['stem_grad_norm'] != 0. or gradients[0]['terminal_grad_norm'] <= 0. or gradients[1]['stem_grad_norm'] <= 0.:
            raise ValueError('Expected zero first-step stem, trainable terminal and subsequent stem gradients')
        head, optimizer, checkpoint = save_reload(directory, args.stage, count, head, optimizer,
            train+val, identity, proof, device, torch, model)
        final_rows = evaluate_rows(head, train+val, device, torch)
        frozen_after = sealed.state_digest(frozen)
        engineering = dict(neutral_exact=True, gpu_save_reload=True,
            frozen_midpoint_state=frozen_before == frozen_after == sealed.FIXED['head_digest'],
            frozen_midpoint_no_grad=all(not p.requires_grad and p.grad is None for p in frozen.parameters()),
            whole_run_frozen_identity=all(metrics.frozen_identity(r) for r in final_rows),
            independent_parameter_count=sealed.state_digest(head)['parameter_count'] == 2818,
            first_stem_zero=gradients[0]['stem_grad_norm'] == 0., first_terminal_learns=gradients[0]['terminal_grad_norm'] > 0.,
            subsequent_stem_learns=gradients[1]['stem_grad_norm'] > 0.,
            fixed_head_weight_sha=sealed.sha(head_dir/sealed.FIXED['path']) == sealed.FIXED['sha256'],
            cache_manifest_sha=sealed.sha(cache_dir/'cache_manifest.json') == sealed.CACHE_SHA,
            source_closure=checked_sources() == identity)
        if not all(engineering.values()):
            raise ValueError('Engineering protection failed')
        if val:
            engineering.update(fixed_val_replay=len(val) == 887,
                smoke_not_reused=initial_digest == smoke['state_before']['size_head'], final_budget=completion['size_head_updates'] == 200)
            summary = metrics.evaluate(final_rows)
            gate = metrics.finite_gates(summary, engineering)
            sealed.write(directory/'summary.json', summary)
            sealed.write(directory/'finite_gate.json', gate)
            completion['finite_gate'] = dict(status=gate['status'], finite_joint_gate_pass=gate['finite_joint_gate_pass'], failed_checks=gate['failed_checks'])
        write_rows(directory/'final_rows.jsonl', final_rows)
        completion.update(status='EDGE_RESIDUAL_GPU_SMOKE_COMPLETE_REVIEW_REQUIRED' if args.stage == 'smoke' else
            'EDGE_RESIDUAL_FINITE_TRAIN_VAL_COMPLETE_REVIEW_REQUIRED',
            engineering=engineering, checkpoint=checkpoint,
            state_after=dict(midpoint=frozen_after, size_head=sealed.state_digest(head)),
            discarded_smoke_updates=2 if val else 0, smoke_weights_used_for_finite=False,
            evaluated_views=len(final_rows), val_frames=len(val), elapsed_seconds=time.monotonic()-started,
            peak_torch_memory_allocated_bytes=torch.cuda.max_memory_allocated(device))
        publish(directory, completion)
        print(completion['status'], flush=True)
        if val:
            for domain in ('real', 'sim'):
                group = summary['val_s1']['all'][domain]
                print('VAL', domain, 'joint_size10/full_frame',
                    group['midpoint']['joint_size10_full_frame'], '->', group['edge_residual']['joint_size10_full_frame'], flush=True)
            print(completion['finite_gate']['status'], 'failed_checks', completion['finite_gate']['failed_checks'], flush=True)
    except Exception as error:
        completion.update(error=str(error), error_type=type(error).__name__, elapsed_seconds=time.monotonic()-started)
        publish(directory, completion)
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--stage', choices=['check', 'smoke', 'finite'], required=True)
    result.add_argument('--out-dir')
    result.add_argument('--cache-dir')
    result.add_argument('--head-dir', default=sealed.DEFAULT_HEAD)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--smoke-report')
    return result


if __name__ == '__main__':
    run(parser().parse_args())
