#!/usr/bin/env python3
"""ideal -> smoke -> train -> assess, with fixed TRAIN split and no TEST.

CPU ideal uses reviewed numeric records. Server smoke/train learns only a new
image geometry reference. assess uses frozen cached B boxes and never chooses a
checkpoint, cutoff, or model. Every stage requires a NEW output directory.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import diagnose_port_reliability_separability_v1 as cached
from crane_project.tools import run_port_simple_reliability_v1 as parent
from crane_project.utils import port_size_reference_v1 as size
from crane_project.utils import port_reliability_separability_v1 as metrics
from crane_project.utils import port_simple_component_reliability_v1 as simple

PROTOCOL = ROOT/'crane_project/tools/port_size_reference_v1_protocol.json'
MANIFEST = ROOT/'crane_project/tools/port_size_reference_v1_sources.json'


def write(path, value):
    parent.write_new(path, value)


def sources():
    old, original, previous = cached.checked_sources()
    protocol = json.loads(PROTOCOL.read_text()); manifest = json.loads(MANIFEST.read_text())
    current = {p: parent.sha(ROOT/p) for p in manifest['sources']}
    if (manifest['protocol'] != size.VERSION or protocol['protocol'] != size.VERSION
            or manifest['sources'] != current or manifest['protocol_sha256'] != parent.sha(PROTOCOL)
            or manifest['parent_manifest_sha256'] != previous['manifest_sha256']
            or protocol['fit_frames']*protocol['epochs'] != protocol['optimizer_steps']
            or protocol['test_read'] or protocol['deployment_cutoff_created'] or protocol['quality_head_trained']):
        raise ValueError('Reviewed size reference sources/protocol differ')
    return protocol, dict(manifest_sha256=parent.sha(MANIFEST), sources=current, parent=previous), old, original


def prepare(args):
    protocol, src, old, original = sources()
    train, val, policy, proof = cached.inputs(args, old, original, src['parent'])
    split = size.partition(train, protocol)
    if len(split['fit']) != protocol['fit_frames'] or len(split['holdout']) != protocol['holdout_frames']:
        raise ValueError('Fixed TRAIN partition count differs')
    names = {k: [r['image'] for r in rows] for k, rows in split.items()}
    if any(set(names[a]) & set(names[b]) for a, b in (('fit', 'holdout'), ('fit', 'guard'), ('holdout', 'guard'))):
        raise ValueError('TRAIN roles overlap')
    numeric = json.loads(args.input_snapshot.read_text())['inputs']
    raw_sources = {r['image']: r for r in numeric}
    contract = dict(protocol=protocol, sources=src, input_proof=proof,
        partition=names, partition_sha256=simple.fingerprint(names),
        fixed_numeric_sha256=simple.fingerprint([train, val, policy]),
        frozen_b=original['frozen_b'], role='TRAIN_REFERENCE_FEASIBILITY',
        b_has_seen_detector_TRAIN=True, new_reference_has_not_seen_holdout_labels=True)
    old_val = {r['image']: r for r in (json.loads(x) for x in (args.val_dir/'val_qualities.jsonl').read_text().splitlines())}
    return protocol, src, train, val, policy, proof, split, raw_sources, contract, old_val


def numeric_meta(row, shrink=1., flip=False):
    w, h = row['image_size']; factor = min(1024./w, 1024./h)*shrink
    iw, ih = int(w*factor+.5), int(h*factor+.5)
    return dict(scale_factor=[iw/w, ih/h, iw/w, ih/h], img_shape=[ih, iw, 3],
                ori_shape=[h, w, 3], pad_shape=[1024, 1024, 3], flip=flip,
                flip_direction='horizontal' if flip else None)


def probe_record(row, reference, protocol):
    candidates = {name: b.tolist() for name, b in size.probes(row['gt'])}
    readings = {name: size.size_reading(b, reference, protocol['risk_scale']) for name, b in candidates.items()}
    baseline = readings['gt']['risk']
    pairs = {name: None if baseline is None else value['risk']-baseline
             for name, value in readings.items() if name.startswith(('long_', 'short_'))}
    invariant = {name: None if baseline is None else abs(value['risk']-baseline)
                 for name, value in readings.items() if name.startswith(('center_', 'angle_', 'equivalent'))}
    return dict(image=row['image'], domain=row['domain'], sequence=row['sequence'], gt_original=row['gt'],
                reference=reference, candidates_original=candidates, readings=readings, size_probe_risk_gaps=pairs,
                nuisance_control_max_risk_difference=None if baseline is None else max(invariant.values()))


def ideal(args, prepared):
    protocol, _, _, _, _, _, split, _, contract, _ = prepared
    # Two predetermined endpoints from each fit sequence, and two endpoints
    # from each held-out sequence. These are numerical, not learned results.
    selected = []
    for rows in (split['fit'], split['holdout']):
        groups = {}
        for r in rows: groups.setdefault(r['sequence'], []).append(r)
        for seq in sorted(groups): selected.extend([groups[seq][0], groups[seq][-1]])
    records = []
    for row in selected:
        for shrink, flip in ((1., False), (.5, True)):
            meta = numeric_meta(row, shrink, flip)
            heatmap, _ = size.target_map(row['gt'], meta)
            reference = size.reference_from_map(heatmap, row['gt'], meta, protocol)
            record = probe_record(row, reference, protocol)
            record.update(shrink=shrink, flip=flip, model_short_px=float(size.box_model(row['gt'], meta)[3]),
                numerical_size_relative_error=None if not reference['defined'] else
                max(abs(reference['long_original_px']/simple.canonical(row['gt'])[2]-1),
                    abs(reference['short_original_px']/simple.canonical(row['gt'])[3]-1)))
            record['passed'] = bool(reference['defined'] and record['numerical_size_relative_error'] <=
                protocol['ideal_reference_max_relative_size_error'] and min(record['size_probe_risk_gaps'].values()) > 1e-6
                and record['nuisance_control_max_risk_difference'] < 1e-10)
            records.append(record)
    clean = [r for r in records if r['shrink'] == 1.]
    passed = len(clean) == protocol['ideal_required_clean_views'] and all(r['passed'] for r in clean)
    report = dict(status='IDEAL_SIZE_READER_PASS' if passed else 'IDEAL_SIZE_READER_FAILED_REVIEW_REQUIRED',
        contract=contract, contract_sha256=simple.fingerprint(contract), rows=records,
        clean_pass=sum(r['passed'] for r in clean), clean_views=len(clean),
        half_scale_pass=sum(r['passed'] for r in records if r['shrink'] == .5),
        half_scale_role='resolution_and_reflection_diagnostic_not_training_view_or_gate_relaxation',
        target_is_geometry_proxy=True, learned_image_performance=False, test_read=False)
    write(args.out_dir/'ideal_report.json', report)
    print('Ideal clean', report['clean_pass'], '/', len(clean), report['status'], flush=True)
    return report['status']


def gate(path, expected_status, contract):
    if path is None: raise ValueError('Missing required preceding-stage report')
    value = json.loads(Path(path).read_text())
    completion_path = Path(path).parent/'completion.json'
    if (Path(path).parent/'failure.json').exists() or not completion_path.is_file():
        raise ValueError('Requires completed, failure-free preceding stage')
    completed = json.loads(completion_path.read_text())
    if (value.get('status') != expected_status or value.get('contract') != contract
            or value.get('contract_sha256') != simple.fingerprint(contract)
            or completed.get('status') != expected_status
            or completed.get('contract_sha256') != simple.fingerprint(contract)
            or completed.get('artifacts', {}).get(Path(path).name) != parent.sha(path)):
        raise ValueError('Preceding-stage gate or unchanged contract differs: '+str(path))
    return parent.sha(path)


def save_checkpoint(path, payload):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='size-reference-', suffix='.tmp', dir=str(path.parent))
    try:
        import torch
        with os.fdopen(fd, 'wb') as stream:
            torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
    finally: os.unlink(temporary)
    write(str(path)+'.sha.json', dict(sha256=parent.sha(path), contract_sha256=simple.fingerprint(payload['contract']),
                                    role=payload['role'], epoch=payload['epoch']))


def load_checkpoint(path, contract, role):
    import torch
    path = Path(path); marker = json.loads(Path(str(path)+'.sha.json').read_text())
    if marker['sha256'] != parent.sha(path): raise ValueError('Reference checkpoint SHA differs')
    payload = torch.load(str(path), map_location='cpu')
    if (payload['protocol'] != size.VERSION or payload['contract'] != contract
            or marker['contract_sha256'] != simple.fingerprint(contract)
            or payload['role'] != role or marker['role'] != role or marker['epoch'] != payload['epoch']):
        raise ValueError('Reference checkpoint role/contract differs')
    return payload


def runtime(args, contract):
    import torch
    from mmcv import Config
    from crane_project.utils.port_size_reference_v1_torch import SizeReference
    from crane_project.utils.port_structure_reliability_v1 import assert_detector_frozen
    from crane_project.tools.preflight_port_structure_reliability_v1 import state_digest as reviewed_digest
    if not torch.cuda.is_available(): raise RuntimeError('GPU stages require server CUDA')
    torch.manual_seed(contract['protocol']['seed']); torch.cuda.manual_seed_all(contract['protocol']['seed'])
    detector, pipeline = parent.build_online(args.b_checkpoint, args.gpu, {'frozen_b': contract['frozen_b']})
    assert_detector_frozen(detector)
    # The reviewed TRAIN cache uses separate parameter/buffer digests; the
    # lightweight stage-local parent digest has a different serialization.
    if reviewed_digest(detector) != contract['input_proof']['original_b_frozen_state']:
        raise ValueError('Loaded B differs from reviewed genuine TRAIN cache state')
    cfg = Config.fromfile(str(parent.B_CONFIG))
    if cfg.model.neck.out_channels != 256 or cfg.model.bbox_head.anchor_generator.strides[0] != 8:
        raise ValueError('Frozen P3 contract differs')
    model = SizeReference().cuda(args.gpu)
    torch.cuda.reset_peak_memory_stats(args.gpu)
    return torch, detector, pipeline, model


def view(row, raw_sources, detector, pipeline, gpu):
    import torch
    from mmcv.parallel import collate, scatter
    image_path = parent.DATA/row['split']/'images'/(row['image']+'.jpg')
    expected = raw_sources[row['image']]['image_sha256'] if row['split'] != 'val' else row['image_sha256']
    if parent.sha(image_path) != expected: raise ValueError('Current image bytes differ: '+row['image'])
    if row['split'] != 'val':
        ann = parent.DATA/row['split']/'annfiles'/(row['image']+'.txt')
        if parent.sha(ann) != raw_sources[row['image']]['annotation_sha256']:
            raise ValueError('TRAIN annotation bytes differ')
    item = pipeline(dict(img_info=dict(filename=str(image_path)), img_prefix=None))
    batch = scatter(collate([item], samples_per_gpu=1), [gpu])[0]
    if len(batch['img']) != 1 or len(batch['img_metas']) != 1: raise ValueError('Multiple inference views')
    image, metas = batch['img'][0], batch['img_metas'][0]; meta = metas[0]
    size.geometry(meta)
    if (meta.get('flip', False) or Path(meta['filename']).stem != row['image']
            or list(meta['ori_shape'][:2][::-1]) != row['image_size']): raise ValueError('Clean image identity differs')
    with torch.no_grad(): features = detector.extract_feat(image)
    if any(f.requires_grad or f.grad_fn is not None for f in features): raise ValueError('B graph not detached')
    if features[0].shape != (1, 256, 128, 128): raise ValueError('Expected frozen 1024-padded P3')
    return features, meta, metas


def payload(model, optimizer, contract, epoch, steps, role, b_state, gates):
    return dict(protocol=size.VERSION, contract=contract, role=role, epoch=epoch, steps=steps,
        state={k: v.detach().cpu() for k, v in model.state_dict().items()}, optimizer=optimizer.state_dict(),
        b_state=b_state, gates=gates)


def train_reference(args, prepared):
    protocol, _, _, _, _, _, split, raw_sources, contract, _ = prepared
    gates = dict(ideal=gate(args.ideal_report, 'IDEAL_SIZE_READER_PASS', contract))
    if args.mode == 'train': gates['smoke'] = gate(args.smoke_report, 'SIZE_REFERENCE_SMOKE_SAVE_RELOAD_PASS_DISCARDED', contract)
    torch, detector, pipeline, model = runtime(args, contract)
    from crane_project.utils.port_size_reference_v1_torch import SizeReference, reference_loss
    from crane_project.utils.port_structure_reliability_v1 import assert_detector_frozen
    before = parent.state_digest(detector)
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol['optimizer']['lr'], weight_decay=0.)
    role = 'smoke_discarded' if args.mode == 'smoke' else 'reference_train'
    steps, start = 0, 1
    if args.resume:
        if args.mode != 'train': raise ValueError('Only completed TRAIN epoch boundaries may resume')
        old = load_checkpoint(args.resume, contract, role)
        if old['gates'] != gates or old['b_state'] != before or not 0 < old['epoch'] < protocol['epochs'] or old['steps'] != old['epoch']*len(split['fit']):
            raise ValueError('Wrong resume boundary, gates or B state')
        model.load_state_dict(old['state'], strict=True); optimizer.load_state_dict(old['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v): state[k] = v.cuda(args.gpu)
        start, steps = old['epoch']+1, old['steps']
    initial_state = parent.state_digest(model)
    count = protocol['epochs'] if args.mode == 'train' else 1
    epochs = []
    with (args.out_dir/'train_steps.jsonl').open('x') as stream:
        for epoch in range(start, count+1):
            model.train(); losses = []; grads = []
            order = np.random.RandomState(protocol['seed']+epoch).permutation(len(split['fit']))
            rows = [split['fit'][int(i)] for i in order]
            if args.mode == 'smoke':
                real = next(r for r in split['fit'] if r['domain'] == 'real')
                sim = next(r for r in split['fit'] if r['domain'] == 'sim')
                rows = [real, sim, real, sim]
            for slot, row in enumerate(rows):
                features, meta, metas = view(row, raw_sources, detector, pipeline, args.gpu)
                raw_before = None
                if args.mode == 'smoke':
                    with torch.no_grad(): raw_before = deepcopy(detector.simple_test_from_features(features, metas, rescale=True))
                target, valid = size.target_map(row['gt'], meta)
                target = torch.as_tensor(target, device=features[0].device)[None, None]
                valid_tensor = torch.as_tensor(valid, device=features[0].device, dtype=target.dtype)[None, None]
                optimizer.zero_grad(); logits = model(features[0])
                loss = reference_loss(logits, target, valid_tensor)
                loss.backward()
                layer_gradients = {}
                for name, parameter in model.named_parameters():
                    if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                        raise ValueError('Reference parameter has missing/nonfinite gradient: '+name)
                    layer_gradients[name] = float(parameter.grad.norm())
                if args.mode == 'smoke' and (not any(value > 0 for name, value in layer_gradients.items() if name.startswith('stem.'))
                        or layer_gradients.get('output.weight', 0.) <= 0):
                    raise ValueError('Smoke reference image features are inactive')
                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), protocol['optimizer']['clip_norm']))
                if not math.isfinite(norm) or norm <= 0 or not bool(torch.isfinite(loss)):
                    raise ValueError('Inactive or nonfinite reference gradient')
                optimizer.step(); steps += 1; losses.append(float(loss.detach())); grads.append(norm)
                if args.mode == 'smoke':
                    from crane_project.tools.preflight_port_structure_reliability_v1 import flatten_prediction
                    with torch.no_grad(): raw_after = detector.simple_test_from_features(features, metas, rescale=True)
                    if not np.array_equal(flatten_prediction(raw_before), flatten_prediction(raw_after)):
                        raise ValueError('Side branch changed B raw outputs')
                record = dict(epoch=epoch, slot=slot, image=row['image'], loss=losses[-1], preclip_norm=norm,
                              target_mass=float(target.sum()), layer_gradient_norms=layer_gradients, detector_updates=0)
                stream.write(json.dumps(record, allow_nan=False)+'\n'); stream.flush()
                if slot % 64 == 0: print(args.mode, 'epoch', epoch, slot+1, '/', len(rows), row['image'], 'loss', losses[-1], flush=True)
                del logits, loss, target, valid_tensor
            model.eval()
            with torch.no_grad(): expected = model(features[0]).detach().cpu()
            path = args.out_dir/('epoch_%02d.pth' % epoch)
            save_checkpoint(path, payload(model, optimizer, contract, epoch, steps, role, before, gates))
            loaded = load_checkpoint(path, contract, role)
            fresh = SizeReference().cuda(args.gpu).eval(); fresh.load_state_dict(loaded['state'], strict=True)
            with torch.no_grad(): actual = fresh(features[0]).detach().cpu()
            if not torch.equal(expected, actual): raise ValueError('Reference save/reload changed same-view logits')
            del fresh, actual, expected, features
            epochs.append(dict(epoch=epoch, steps=steps, mean_loss=float(np.mean(losses)),
                gradient_median=float(np.median(grads)), clipped_steps=sum(g > protocol['optimizer']['clip_norm'] for g in grads)))
    assert_detector_frozen(detector)
    if parent.state_digest(detector) != before or parent.state_digest(model) == initial_state:
        raise ValueError('B changed or reference parameters did not update')
    expected_steps = protocol['smoke_steps'] if args.mode == 'smoke' else protocol['optimizer_steps']
    if steps != expected_steps: raise ValueError('Fixed reference optimization budget differs')
    status = 'SIZE_REFERENCE_SMOKE_SAVE_RELOAD_PASS_DISCARDED' if args.mode == 'smoke' else 'FIXED_EPOCH4_SIZE_REFERENCE_TRAIN_COMPLETE_REVIEW_REQUIRED'
    report = dict(status=status, contract=contract, contract_sha256=simple.fingerprint(contract), role=role,
        epochs=epochs, steps=steps, gates=gates, resumed_from_sha256=None if args.resume is None else parent.sha(args.resume),
        checkpoint_sha256=parent.sha(path), final_checkpoint=path.name, save_reload_logits_exact=True,
        b_state_before=before, b_state_after=parent.state_digest(detector), b_raw_smoke_exact=args.mode == 'smoke',
        detector_updates=0, quality_updates=0, test_read=False,
        allocated_peak_mib=torch.cuda.max_memory_allocated(args.gpu)/2**20,
        reserved_peak_mib=torch.cuda.max_memory_reserved(args.gpu)/2**20)
    write(args.out_dir/'train_report.json', report)
    print('Saved', args.out_dir/'train_report.json', status, flush=True)
    return status


def summary(rows, policy, protocol, old_val):
    result = {}
    groups = {'all': rows}
    groups.update({'domain:'+d: [r for r in rows if r['domain'] == d] for d in ('real', 'sim')})
    groups.update({'sequence:'+s: [r for r in rows if r['sequence'] == s] for s in sorted({r['sequence'] for r in rows})})
    for name, values in groups.items():
        output = [r for r in values if r['pred'] is not None]
        bad = [simple.geometry_errors(r['gt'], r['pred'])['size_max_relative'] > .1 for r in output]
        center = sum(simple.geometry_errors(r['gt'], r['pred'])['center_px'] < 15 for r in output)
        defined = [i for i, r in enumerate(output) if r['reading']['defined']]
        risks = dict(reference=[r['reading']['risk'] for r in output], score=[1-r['pred'][5] for r in output],
            simple=[float(simple.linear_risk(policy['models']['size'], simple.descriptor(r['pred'], r['image_size']))) for r in output])
        if output and all(r['image'] in old_val for r in output):
            risks['old_roi'] = [1-old_val[r['image']]['qualities']['roi'][1] for r in output]
        stats = dict(frames=len(values), outputs=len(output), raw_size_bad=sum(bad),
            output_coverage=len(output)/len(values) if values else None,
            center_correct_on_outputs=center/len(output) if output else None,
            all_frame_center_correct_coverage=center/len(values) if values else None,
            reference_defined=len(defined), unavailable_good=sum(not bad[i] for i in range(len(output)) if i not in defined),
            unavailable_bad=sum(bad[i] for i in range(len(output)) if i not in defined),
            common_defined_rank={m: metrics.rank_metrics([r[i] for i in defined], [bad[i] for i in defined]) for m, r in risks.items()},
            full_output_rank={m: metrics.rank_metrics(r, bad) for m, r in risks.items() if m != 'reference'},
            reference_size_error_relative_to_GT=error_distribution([r['reference_size_error'] for r in output if r['reference_size_error'] is not None]),
            coverage_points=[])
        order = sorted(defined, key=lambda i: (risks['reference'][i], output[i]['image']))
        for fraction in protocol['coverage_fractions']:
            requested = min(len(output), math.ceil(fraction*len(values)))
            count = min(requested, len(defined)); selected = set(order[:count])
            methods = {'reference': metrics.confusion(bad, [i in selected for i in range(len(output))], len(values), len(values))}
            for method in risks:
                if method == 'reference': continue
                comparator = sorted(range(len(output)), key=lambda i: (risks[method][i], output[i]['image']))
                accepted = set(comparator[:count])
                methods[method] = metrics.confusion(bad, [i in accepted for i in range(len(output))], len(values), len(values))
            stats['coverage_points'].append(dict(requested_fraction=fraction, requested_count=requested,
                actual_count=count, unavailable_not_accepted=True, matched_count_methods=methods,
                score_tie_bounds=metrics.same_count_reference(risks['score'], bad, [r['image'] for r in output], [count])[count]))
        result[name] = stats
    return result


def error_distribution(values):
    x = np.asarray(values, float)
    return dict(count=len(x), mean=float(x.mean()) if len(x) else None,
        p90=float(np.percentile(x, 90)) if len(x) else None,
        max=float(x.max()) if len(x) else None,
        within_10_percent=int(np.sum(x <= .1)))


def assess(args, prepared):
    protocol, _, _, val, policy, _, split, raw_sources, contract, old_val = prepared
    if args.reference_checkpoint is None: raise ValueError('assess requires fixed epoch_04 reference checkpoint')
    gate(args.reference_checkpoint.parent/'train_report.json', 'FIXED_EPOCH4_SIZE_REFERENCE_TRAIN_COMPLETE_REVIEW_REQUIRED', contract)
    saved = load_checkpoint(args.reference_checkpoint, contract, 'reference_train')
    completed = json.loads((args.reference_checkpoint.parent/'train_report.json').read_text())
    if (saved['epoch'] != protocol['epochs'] or saved['steps'] != protocol['optimizer_steps']
            or completed['status'] != 'FIXED_EPOCH4_SIZE_REFERENCE_TRAIN_COMPLETE_REVIEW_REQUIRED'
            or completed['contract'] != contract or completed['checkpoint_sha256'] != parent.sha(args.reference_checkpoint)
            or completed['steps'] != saved['steps'] or completed['gates'] != saved['gates']
            or completed['b_state_before'] != completed['b_state_after'] or completed['b_state_before'] != saved['b_state']
            or not completed['save_reload_logits_exact']): raise ValueError('Requires completed fixed epoch4 TRAIN bundle')
    torch, detector, pipeline, model = runtime(args, contract); model.load_state_dict(saved['state'], strict=True); model.eval()
    from crane_project.utils.port_structure_reliability_v1 import assert_detector_frozen
    before = parent.state_digest(detector)
    if before != saved['b_state']: raise ValueError('B state differs from reference TRAIN')
    reference_before = parent.state_digest(model); summaries = {}
    with (args.out_dir/'reference_rows.jsonl').open('x') as stream, (args.out_dir/'probe_rows.jsonl').open('x') as probe_stream:
        for role, rows in (('train_holdout', split['holdout']), ('val', val)):
            assessed = []; probe_rows = []
            for index, row in enumerate(rows):
                record = deepcopy(row); record.update(reference=None, reading=dict(defined=False, risk=None), reference_size_error=None)
                if row['pred'] is not None:
                    features, meta, _ = view(row, raw_sources, detector, pipeline, args.gpu)
                    with torch.no_grad(): probability = model(features[0]).sigmoid()[0, 0].cpu().numpy()
                    reference = size.reference_from_map(probability, row['pred'][:5], meta, protocol)
                    record.update(reference=reference, reading=size.size_reading(row['pred'][:5], reference, protocol['risk_scale']))
                    if reference['defined']:
                        g = simple.canonical(row['gt'])
                        record['reference_size_error'] = max(abs(reference['long_original_px']/g[2]-1), abs(reference['short_original_px']/g[3]-1))
                    # Perturbations are OFFLINE diagnostics. All use the same
                    # actual B context, never a GT-centred oracle context.
                    probe = probe_record(row, reference, protocol); probe.update(role=role, context_source='genuine_cached_B')
                    probe_stream.write(json.dumps(probe, allow_nan=False)+'\n')
                    probe_rows.append(probe)
                    del features, probability
                record['role'] = role
                stream.write(json.dumps(record, allow_nan=False)+'\n'); stream.flush(); assessed.append(record)
                if index % 64 == 0: print('Frozen reference', role, index+1, '/', len(rows), flush=True)
            summaries[role] = summary(assessed, policy, protocol, old_val)
            summaries[role]['probe_diagnostics'] = probe_summary(probe_rows)
    assert_detector_frozen(detector)
    if parent.state_digest(detector) != before or parent.state_digest(model) != reference_before:
        raise ValueError('Assessment updated B or reference')
    report = dict(status='FROZEN_SIZE_REFERENCE_ASSESSMENT_COMPLETE_REVIEW_REQUIRED', contract=contract,
        contract_sha256=simple.fingerprint(contract), summaries=summaries, checkpoint_sha256=parent.sha(args.reference_checkpoint),
        detector_updates=0, reference_updates=0, quality_updates=0, test_read=False, deployment_cutoff_created=False,
        cached_B_boxes_scores_and_missing_outputs_preserved=True,
        evidence_role='cached_box_scoring_new_image_reference_not_runtime_B_cache_parity_claim',
        old_roi_role='historical_epoch8_VAL_reference_trained_on_full_TRAIN_not_new_holdout_control',
        allocated_peak_mib=torch.cuda.max_memory_allocated(args.gpu)/2**20,
        reserved_peak_mib=torch.cuda.max_memory_reserved(args.gpu)/2**20)
    write(args.out_dir/'assessment.json', report)
    print('Saved', args.out_dir/'assessment.json', report['status'], flush=True)
    return report['status']


def probe_summary(rows):
    groups = {'all': rows}
    groups.update({'domain:'+d: [r for r in rows if r['domain'] == d] for d in ('real', 'sim')})
    result = {}
    for name, group in groups.items():
        defined = [r for r in group if r['reference']['defined']]
        gaps = [v for r in defined for v in r['size_probe_risk_gaps'].values()]
        result[name] = dict(views_with_B_output=len(group), defined_views=len(defined),
            correct_pairs=sum(v > 1e-6 for v in gaps), total_pairs=len(gaps),
            long_both_sides=sum(min(r['size_probe_risk_gaps'][k] for k in ('long_minus', 'long_plus')) > 1e-6 for r in defined),
            short_both_sides=sum(min(r['size_probe_risk_gaps'][k] for k in ('short_minus', 'short_plus')) > 1e-6 for r in defined),
            nuisance_max_difference=max((r['nuisance_control_max_risk_difference'] for r in defined), default=None),
            synthetic_probe_results_not_genuine_error_accuracy=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('ideal', 'smoke', 'train', 'assess'), required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--train-cache', type=Path, default=Path('work_dirs/port_reliability_train_support_v1_cache'))
    parser.add_argument('--input-snapshot', type=Path, default=Path('work_dirs/port_reliability_train_support_v1/train_input_snapshot.json'))
    parser.add_argument('--val-dir', type=Path, default=Path('work_dirs/port_reliability_branches_v1_val_cached_v1'))
    parser.add_argument('--policy', type=Path, default=Path('work_dirs/port_simple_reliability_v1_fit/policy.json'))
    parser.add_argument('--b-checkpoint', type=Path, default=Path('work_dirs/crane_symeood_k1_port_day2night_aug_b_v1/epoch_24.pth'))
    parser.add_argument('--ideal-report', type=Path)
    parser.add_argument('--smoke-report', type=Path)
    parser.add_argument('--reference-checkpoint', type=Path)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.resume is not None and args.mode != 'train': parser.error('--resume is TRAIN only')
    prepared = prepare(args); args.out_dir.mkdir(parents=True, exist_ok=False)
    write(args.out_dir/'input_check.json', prepared[8])
    try:
        if args.mode == 'ideal': status = ideal(args, prepared)
        elif args.mode in ('smoke', 'train'): status = train_reference(args, prepared)
        else: status = assess(args, prepared)
        if prepare(args)[8] != prepared[8]: raise ValueError('Inputs/sources changed during stage')
        artifacts = {p.name: parent.sha(p) for p in sorted(args.out_dir.iterdir()) if p.is_file()}
        write(args.out_dir/'completion.json', dict(status=status, mode=args.mode, artifacts=artifacts,
                                                  contract_sha256=simple.fingerprint(prepared[8]), test_read=False))
    except Exception as error:
        write(args.out_dir/'failure.json', dict(type=type(error).__name__, error=str(error), mode=args.mode))
        raise


if __name__ == '__main__': main()
