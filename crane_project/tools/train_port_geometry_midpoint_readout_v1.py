#!/usr/bin/env python3
"""One new uniform-neutral head versus saved sigma1.5 and frozen B on VAL.

check: text/hash audit only. preflight: one GPU, zero optimizer updates.
train: explicit, separately gated 24-epoch head fit, never detector training.
No TEST stage, detector construction, extraction, cache rebuilding or promotion.
"""
import argparse
from copy import deepcopy
import csv
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import train_port_geometry_midpoint_sigma_v1 as t
from crane_project.tools import eval_port_geometry_midpoint_sigma15_v1 as e
from crane_project.utils import port_geometry_midpoint_readout_v1 as model

f, g, torch = t.f, t.g, t.torch
VERSION = 'port_geometry_midpoint_readout_v1'
PROTOCOL = ROOT/'crane_project/tools/port_geometry_midpoint_readout_v1_protocol.json'
SOURCES = ROOT/'crane_project/tools/port_geometry_midpoint_readout_v1_sources.json'
PREFLIGHT_STATUS = 'READOUT_PREFLIGHT_PASS_ZERO_UPDATES_REVIEW_REQUIRED'
# These detector classes are not instantiated/called by this cache-only tool.
# Keep their historical hashes in old provenance, current hashes in our own
# manifest, and report the difference. Never rewrite a historical manifest.
DETECTOR_ONLY = frozenset((
    'mmrotate/core/bbox/assigners/sym_pola.py',
    'mmrotate/models/losses/sym_kld_loss.py',
    'mmrotate/models/losses/sym_nfl_loss.py',
    'mmrotate/models/dense_heads/sym_eood_head.py',
    'mmrotate/models/detectors/sym_eood_detector.py'))


def protocol_document():
    return dict(protocol=VERSION, readout=model.READOUT,
        reference=deepcopy(e.FIXED), reference_sigma_cells=1.5,
        settings=deepcopy(f.SETTINGS), midpoint_settings=deepcopy(f.m.SETTINGS),
        parameter_count=f.m.PARAMETER_COUNT, selection_config=deepcopy(f.SELECTION_CONFIG),
        control='Uniform zero prior Q=0: B anchor + E(softmax(logits))-E(softmax(0)) on the same finite9x9 grid. Both arms start exactly at B. Do not delete neutral subtraction.',
        variable='Remove only the B-only Gaussian spatial prior, retain neutral residual form. Distribution, gradients and available displacement range change. This is not a separate ablation of subtraction.',
        fairness='One new seed1703 head from scratch, same17696 parameters, targets, TRAIN eligibility, domain-balanced batches, Adam.001,clip10 and24 epochs/21648 updates. Reuse completed sigma1.5 epoch03 selected from all24 full-VAL checkpoints; never force epoch03 on the new arm.',
        cache='Reuse immutable original TRAIN1/.5 and VAL1 CPU ROI cache (3 shards). No detector load/forward/update, feature extraction or cache rebuild. Standard-scale full887 VAL including missing/mismatched frames.',
        reference_reuse='Verify original and sigma1.5 completed24-epoch artifacts, indexed selected rows/weight SHA, initial state, support, schedule, cache and runtime. Replay exact selected sigma1.5 full VAL before comparison. Mismatch fails; never silently retrain.',
        source_boundary='Cache-only execution: reconstruct historical reference identities from unchanged source manifests. Verify historical SHA for every dependency except the5 named detector classes whose methods are unused; record those historical/current differences and bind ALL current SHA in the new manifest. Runtime version checking may import MMRotate registries, but never instantiates/calls detector/assigner/loss methods or asserts changed files are historical bytes.',
        preflight='Text check loads no tensors. GPU preflight loads3 CPU shards, replays saved sigma1.5 full VAL, verifies new full-VAL zero identity and2 TRAIN gradient batches with no optimizer.step. Save/reload zero-update head exactly. No fitted weights or performance claim.',
        training='Explicit --stage train plus successful indexed preflight required. Fresh initialization and optimizer,24 complete VAL evaluations, unchanged ALL24 selection; export paired B/uniform/prior-sigma1.5 rows and every epoch. No automatic winner or replacement.',
        reporting='Output-frame center hit, output coverage and all-frame center correct coverage separately. Existing RIoU, DFR, ACI, penalty and pure sim angle, TDR, MCML, MRF; accepted/fallback/invalid-candidate counts. No new AP calculation or TEST data access.',
        checkpoint='New protocol and readout identity bound to sources/reference/cache, exact no-overwrite save/reload. Historical sigma loaders must reject the new checkpoint.',
        scope=dict(new_formal_conditions=1, reused_conditions=1, frozen_b_conditions=1,
            formal_epochs=24, formal_updates=21648, preflight_updates=0,
            detector_updates=0, detector_forward_calls=0, feature_extractions=0,
            cache_rebuilt=False, test_access=False, selection_on_test=False,
            automatic_promotion=False, test_repeatedly_exposed=True),
        limitations=['No separate identification of neutral subtraction, sigma optimality or stability across seeds.',
            'Sigma1.5 retained after exposed TEST return; reuse does not create independent confirmation.',
            'Original isotropic transform, actual sx/sy restoration, raw axes and depth interface unchanged; depth accuracy unverified.',
            'Protocol evaluator test-mode computes VAL temporal metrics; it does not access TEST images.'])


def recorded_reference_identity():
    """Historical identity is recorded provenance, not a current detector audit."""
    formal = t.read(f.SOURCES); sigma = t.read(t.SOURCES)
    evaluation = t.read(e.e.SOURCES); reference = t.read(e.SOURCES)
    if (t.read(f.PROTOCOL) != f.protocol_document() or
            t.read(t.PROTOCOL) != t.protocol_document() or
            t.read(e.e.PROTOCOL) != e.e.protocol_document() or
            t.read(e.PROTOCOL) != e.protocol_document() or
            formal['protocol'] != f.VERSION or sigma['protocol'] != t.VERSION or
            evaluation['protocol'] != e.e.VERSION or reference['protocol'] != e.VERSION or
            sigma['parent_sources_sha256'] != t.sha(f.SOURCES) or
            reference['training_sources_sha256'] != t.sha(t.SOURCES) or
            reference['evaluation_sources_sha256'] != t.sha(e.e.SOURCES)):
        raise ValueError('Historical source manifest/protocol changed')
    for document in (formal, sigma, evaluation):
        for name, digest in document['sources'].items():
            if reference['sources'].get(name) != digest:
                raise ValueError('Historical source closures disagree: '+name)
    differences = {}
    for name, digest in reference['sources'].items():
        actual = t.sha(ROOT/name)
        if actual != digest:
            if name not in DETECTOR_ONLY:
                raise ValueError('Executed historical dependency SHA differs: '+name)
            differences[name] = dict(historical_sha256=digest, current_sha256=actual,
                instantiated_or_called_by_this_tool=False)
    parent = dict(protocol_sha256=t.sha(f.PROTOCOL), sources_sha256=t.sha(f.SOURCES),
        sources=formal['sources'], frozen_b=deepcopy(g.ready.FROZEN_B))
    training = dict(protocol_sha256=t.sha(t.PROTOCOL), sources_sha256=t.sha(t.SOURCES),
        sources=sigma['sources'], parent_identity=parent)
    eval_identity = dict(training_identity=parent, sources=evaluation['sources'],
        protocol_sha256=t.sha(e.e.PROTOCOL), sources_sha256=t.sha(e.e.SOURCES))
    identity = dict(training_identity=training, evaluation_identity=eval_identity,
        sources=reference['sources'], protocol_sha256=t.sha(e.PROTOCOL), sources_sha256=t.sha(e.SOURCES))
    return identity, differences


def checked_sources():
    reference, differences = recorded_reference_identity()
    fixed = t.read(SOURCES)
    required = set(reference['sources']) | {
        str(e.SOURCES.relative_to(ROOT)), str(PROTOCOL.relative_to(ROOT)),
        'crane_project/utils/port_geometry_midpoint_readout_v1.py',
        'crane_project/tools/train_port_geometry_midpoint_readout_v1.py',
        'tests/test_port_geometry_midpoint_readout_v1.py'}
    if (fixed['protocol'] != VERSION or set(fixed['sources']) != required or
            fixed['reference_sources_sha256'] != t.sha(e.SOURCES) or
            t.read(PROTOCOL) != protocol_document()):
        raise ValueError('Readout fixed source/protocol scope differs')
    for name, digest in fixed['sources'].items():
        if t.sha(ROOT/name) != digest:
            raise ValueError('Readout source SHA differs: '+name)
    return dict(protocol_sha256=t.sha(PROTOCOL), sources_sha256=t.sha(SOURCES),
        sources=fixed['sources'], reference_identity=reference, detector_only_source_differences=differences)


def checked_contract(baseline_dir, selection_path, cache_dir):
    identity = checked_sources()
    reference_identity = identity['reference_identity']
    parent = reference_identity['training_identity']['parent_identity']
    cache = t.checked_cache_metadata(cache_dir, parent)
    baseline = t.audit_baseline(baseline_dir, parent, cache)
    selection, reference_proof = e.checked_selection(selection_path, reference_identity)
    if (reference_proof['cache_manifest_sha256'] != cache['manifest_sha256'] or
            reference_proof['training_runtime'] != cache['runtime'] or
            selection['proof']['baseline_files'] != baseline['files']):
        raise ValueError('Reference/source/cache/original training differs')
    directory = Path(selection_path).resolve().parent
    rows = e.read_rows(directory/'val_epoch_03.rows.jsonl')
    value = t.read(directory/'selected_val_compare.json')
    reference = dict(selection=selection, rows=rows, value=value, proof=reference_proof)
    proof = dict(identity=identity, cache_manifest_sha256=cache['manifest_sha256'],
        original_baseline_files=baseline['files'], reference=reference_proof)
    return cache, baseline, reference, proof


def save_checkpoint(path, head, optimizer, epoch, updates, proof, cache):
    if not isinstance(head, model.UniformMidpointHead):
        raise ValueError('Readout checkpoint requires the uniform head class')
    payload = dict(protocol=VERSION, readout=model.READOUT, proof=proof,
        epoch=epoch, updates=updates,
        head_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()},
        optimizer_state=deepcopy(optimizer.state_dict()) if optimizer is not None else None,
        rng=f.rng_state(), head_digest=g.state_digest(head),
        context=f.checkpoint_context(cache), frozen_b=deepcopy(g.ready.FROZEN_B))
    f.atomic_save(Path(path), payload)
    if not f.tree_equal(payload, torch.load(str(path), map_location='cpu')):
        raise ValueError('Readout checkpoint exact reload differs')
    return dict(path=Path(path).name, sha256=t.sha(path), epoch=epoch, updates=updates,
        readout=model.READOUT, head_digest=payload['head_digest'], save_reload_exact=True)


def load_head(path, proof, cache, device):
    payload = torch.load(str(path), map_location='cpu')
    if (payload['protocol'] != VERSION or payload.get('readout') != model.READOUT or
            payload['proof'] != proof or payload['context'] != f.checkpoint_context(cache) or
            payload['frozen_b'] != g.ready.FROZEN_B):
        raise ValueError('Readout checkpoint identity/source/cache differs')
    head = model.UniformMidpointHead().to(device)
    head.load_state_dict(payload['head_state'], strict=True)
    if g.state_digest(head) != payload['head_digest']:
        raise ValueError('Readout head digest differs')
    return head, payload


def replay_reference(selection_path, val, reference, cache, device):
    selected = reference['selection']['selected_checkpoint']
    path = Path(selection_path).resolve().parent/selected['path']
    if t.sha(path) != selected['sha256']:
        raise ValueError('Reference selected weight changed before replay')
    head, payload = t.load_head(path, 1.5, reference['selection']['proof'], cache, device)
    if (payload['epoch'] != selected['epoch'] or payload['updates'] != selected['updates'] or
            payload['head_digest'] != selected['head_digest']):
        raise ValueError('Reference selected checkpoint metadata differs')
    rows = f.evaluate(head, val, device)
    if rows != reference['rows'] or f.summaries(rows) != reference['value']:
        raise ValueError('Reference sigma1.5 full-VAL replay differs')
    return dict(selected_checkpoint=selected, frames=len(rows), exact_cached_val_replay=True)


def zero_identity(head, val, reference, device):
    rows = f.evaluate(head, val, device)
    if (any(r['b'] != r['midpoint'] for r in rows) or
            [(r['image'], r['b']) for r in rows] != [(r['image'], r['b']) for r in reference['rows']]):
        raise ValueError('New head initial output is not exact paired B')
    return rows


def preflight(train, val, initial, schedule, reference, proof, cache, device, out):
    f.seed_all(); head = model.UniformMidpointHead().to(device)
    head.load_state_dict(initial, strict=True)
    before = g.state_digest(head)
    zero_identity(head, val, reference, device)
    gradients = []
    for indices in f.epoch_batches(train, 1)[:2]:
        head.zero_grad()
        roi, support, b, bm, xy, gt = f.batch(train, indices, device)
        result = head(roi, support, b, bm, xy)
        loss, _ = f.m.point_loss(result['points_roi'], f.m.target_points(gt, b, bm, xy))
        loss.backward()
        params = list(head.parameters())
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params):
            raise ValueError('Missing/nonfinite preflight gradient')
        norm = float(torch.nn.utils.clip_grad_norm_(params, f.SETTINGS['clip_norm']))
        output_norm = float(head.output.weight.grad.double().norm())
        after = sum(float(p.grad.double().square().sum()) for p in params)**.5
        if (not bool(torch.isfinite(loss)) or not math.isfinite(norm) or
                output_norm <= 0 or after > f.SETTINGS['clip_norm']+1e-4):
            raise ValueError('Ineffective/nonfinite preflight output gradient')
        gradients.append(dict(loss=float(loss.detach()), output_grad_norm_after_clip=output_norm,
            grad_norm_before_clip=norm, grad_norm_after_clip=after, optimizer_updates=0))
    if g.state_digest(head) != before or before != schedule['initial_state']:
        raise ValueError('Zero-update preflight changed parameters/buffers')
    head.zero_grad()
    ckpt = save_checkpoint(out/'zero_update_head.pth', head, None, 0, 0, proof, cache)
    replay, payload = load_head(out/ckpt['path'], proof, cache, device)
    if payload['optimizer_state'] is not None or f.evaluate(head, f.smoke_views(val), device) != f.evaluate(replay, f.smoke_views(val), device):
        raise ValueError('Preflight zero-update save/reload differs')
    return dict(checkpoint=ckpt, initial_full_val_exact_b=True, gradient_batches=gradients,
        save_reload_exact=True, head_updates=0, fitted_weight=False)


def checked_preflight(path, proof, schedule):
    path = Path(path); report = t.read(path); index = t.read(path.parent/'artifacts.json')
    if (index['protocol'] != VERSION or index['files'].get(path.name) != t.sha(path) or
            report['protocol'] != VERSION or report['stage'] != 'preflight' or
            report['status'] != PREFLIGHT_STATUS or report['proof'] != proof or
            report['schedule'] != schedule or report['head_updates_total'] != 0 or
            report['test_access'] is not False or report['detector_updates'] != 0 or
            report['readout'] != model.READOUT or report['detector_forward_calls'] != 0 or
            report['feature_extractions'] != 0 or report['cache_tensor_loads'] != 3 or
            report['reference_replay']['exact_cached_val_replay'] is not True):
        raise ValueError('Requires matching successful zero-update preflight')
    value = report['preflight']; ckpt = value['checkpoint']; weight = path.parent/ckpt['path']
    if (ckpt['epoch'] != 0 or ckpt['updates'] != 0 or ckpt['readout'] != model.READOUT or
            ckpt['head_digest'] != schedule['initial_state'] or value['fitted_weight'] is not False or
            value['head_updates'] != 0 or value['save_reload_exact'] is not True or
            value['initial_full_val_exact_b'] is not True or len(value['gradient_batches']) != 2 or
            index['files'].get(weight.name) != t.sha(weight) or t.sha(weight) != ckpt['sha256']):
        raise ValueError('Preflight initial checkpoint/gradient evidence differs')
    for gradient in value['gradient_batches']:
        if (gradient['optimizer_updates'] != 0 or
                any(not math.isfinite(gradient[k]) for k in
                    ('loss', 'output_grad_norm_after_clip', 'grad_norm_before_clip', 'grad_norm_after_clip')) or
                gradient['output_grad_norm_after_clip'] <= 0 or
                gradient['grad_norm_after_clip'] > f.SETTINGS['clip_norm']+1e-4):
            raise ValueError('Preflight zero-update effective gradient evidence differs')
    return report


def train_condition(train, val, initial, proof, cache, reference, schedule, device, out, progress):
    f.seed_all(); head = model.UniformMidpointHead().to(device)
    head.load_state_dict(initial, strict=True)
    if g.state_digest(head) != schedule['initial_state']:
        raise ValueError('Fresh formal initialization differs')
    optimizer = f.optimizer_for(head)
    t.write(out/'val_baseline.json', f.summaries(zero_identity(head, val, reference, device)))
    entries = {}; values = {}; updates = 0
    for epoch in range(1, f.SETTINGS['epochs']+1):
        batches = f.epoch_batches(train, epoch); losses = []
        for step, indices in enumerate(batches, 1):
            result = t.tracked_update(head, optimizer, train, indices, device, progress,
                dict(readout=model.READOUT, epoch=epoch, step=step, updates=updates+1, phase='train'))
            updates += 1; losses.append(result['loss'])
            progress(dict(stage='train', epoch=epoch, step=step, epoch_steps=len(batches), updates=updates, **result))
        checkpoint = save_checkpoint(out/('head_epoch_%02d.pth'%epoch), head, optimizer, epoch, updates, proof, cache)
        rows = f.evaluate(head, val, device); value = f.summaries(rows)
        t.write(out/('val_epoch_%02d.json'%epoch), value)
        with (out/('val_epoch_%02d.rows.jsonl'%epoch)).open('x') as stream:
            for row in rows:
                stream.write(json.dumps(g.json_native(row), ensure_ascii=False, allow_nan=False)+'\n')
        name = checkpoint['path']; values[name] = value
        entries[name] = dict(checkpoint=checkpoint, metrics=value['groups']['overall']['midpoint']['metric_protocol_v2'])
        progress(dict(stage='epoch_complete', epoch=epoch, updates=updates,
            loss_mean=sum(losses)/len(losses), checkpoint=checkpoint, selection_metrics=entries[name]['metrics']))
        print('UNIFORM readout epoch', epoch, 'updates', updates, 'point_loss', sum(losses)/len(losses), flush=True)
    if updates != sum(schedule['epoch_steps']):
        raise ValueError('Uniform complete update budget differs')
    chosen, _, info = f.select_best_checkpoint(entries, f.SELECTION_CONFIG)
    if chosen is None:
        raise ValueError('No full-VAL checkpoint selected')
    del head, optimizer
    head, _ = load_head(out/chosen, proof, cache, device)
    rows = f.evaluate(head, val, device); value = f.summaries(rows)
    if value != values[chosen]:
        raise ValueError('Selected uniform head reload full-VAL differs')
    selection = dict(protocol=VERSION, readout=model.READOUT, proof=proof, split='val',
        selected_checkpoint=entries[chosen]['checkpoint'], selection_config=deepcopy(f.SELECTION_CONFIG),
        selection_info=info, all_checkpoints=entries, cache_manifest_sha256=cache['manifest_sha256'],
        comparison=f.comparison_checks(value), test_access=False, selection_on_test=False, automatic_promotion=False)
    t.write(out/'selection.json', selection); t.write(out/'selected_val_compare.json', value)
    return dict(selection=selection, value=value, rows=rows, updates=updates, all_epoch_values=values)


def comparison_rows(result, reference):
    rows = []
    for label, method, value, predictions, epoch in (
            ('b', 'b', reference['value'], reference['rows'], 24),
            (model.READOUT, 'midpoint', result['value'], result['rows'], result['selection']['selected_checkpoint']['epoch']),
            ('gaussian_prior_sigma1p5', 'midpoint', reference['value'], reference['rows'], 3)):
        for domain in ('real', 'sim'):
            summary = value['groups'][domain][method]; metrics = summary['metric_protocol_v2']
            subset = [r for r in predictions if r['domain'] == domain]
            accepted = sum(r['accepted'] is True for r in subset) if method != 'b' else None
            fallback = sum(r['accepted'] is False for r in subset) if method != 'b' else None
            invalid = sum(r['candidate_valid'] is False for r in subset) if method != 'b' else None
            rows.append(dict(condition=label, domain=domain, selected_epoch=epoch, split='val', scale=1.,
                frames=summary['frames'], output_frames=summary['output_frames'],
                output_center_correct_count=summary['conditional_center_correct_fraction']['numerator'],
                output_frame_r_center_pct=summary['output_center_hit_pct'], output_coverage_pct=summary['output_coverage_pct'],
                full_frame_center_correct_pct=summary['all_frame_center_hit_pct'],
                mean_riou=metrics[domain+'/mean_RIoU'], penalty_a_rmse_deg=metrics.get(domain+'/A-RMSE(deg)'),
                pure_angle_rmse_deg=summary['angle_error_deg']['rmse'] if domain == 'sim' else None,
                dfr_pct_per_frame=metrics[domain+'/DFR(%/frame)'], aci=metrics[domain+'/ACI'],
                tdr_w10_pct=metrics[domain+'/TDR_w10(%)'], mcml_max_frames=metrics[domain+'/MCML_max(frames)'],
                mcml_mean_frames=metrics[domain+'/MCML_mean(frames)'], mcml_pass=metrics[domain+'/MCML_pass(limit=5)'],
                mrf_frames=metrics.get(domain+'/MRF(frames)'), accepted_frames=accepted, fallback_frames=fallback,
                invalid_candidate_frames=invalid,
                accepted_output_pct=100.*accepted/summary['output_frames'] if accepted is not None and summary['output_frames'] else None))
    return rows


def publish(out):
    t.write(Path(out)/'artifacts.json', dict(protocol=VERSION,
        files={p.name:t.sha(p) for p in Path(out).iterdir() if p.is_file() and p.name != 'artifacts.json'}))


def safe_output(out, inputs):
    output = Path(out).resolve()
    for name in inputs:
        source = Path(name).resolve()
        if output == source or source in output.parents or output in source.parents:
            raise ValueError('Output must be disjoint from immutable inputs')
    if output.exists():
        raise FileExistsError('Output exists; preserve it and use a new --out-dir suffix.')


def run(args):
    if int(os.environ.get('WORLD_SIZE', '1')) != 1 or args.gpu < 0:
        raise ValueError('Use one logical GPU without torchrun/DDP')
    if args.stage == 'train' and not args.preflight_report:
        raise ValueError('--preflight-report required before any formal training')
    inputs = [args.baseline_dir, str(Path(args.reference_selection).resolve().parent.parent), args.cache_dir]
    if args.preflight_report:
        inputs.append(str(Path(args.preflight_report).resolve().parent))
    safe_output(args.out_dir, inputs)
    print('Readout', args.stage, ': one new uniform-neutral head; frozen B forward/updates=0;',
        'preflight optimizer updates=0; formal training requires --stage train.', flush=True)
    # Validate paths/contracts before creating output, so a missing input never
    # strands an empty directory that prevents retrying the same command.
    cache, baseline, reference, proof = checked_contract(args.baseline_dir, args.reference_selection, args.cache_dir)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=False)
    report = dict(protocol=VERSION, stage=args.stage, status='STARTED', proof=proof,
        readout=model.READOUT, head_updates_total=0, detector_updates=0, detector_forward_calls=0,
        feature_extractions=0, cache_tensor_loads=0, test_access=False, selection_on_test=False, automatic_promotion=False)
    started = time.perf_counter()
    with (out/'progress.jsonl').open('x') as stream:
        def progress(value):
            if value['stage'] == 'optimizer_step':
                report['head_updates_total'] += 1
            stream.write(json.dumps(g.json_native(value), ensure_ascii=False, allow_nan=False)+'\n'); stream.flush()
        try:
            t.write(out/'protocol.json', protocol_document()); t.write(out/'sources.json', t.read(SOURCES))
            t.write(out/'reuse_paths.json', dict(baseline_dir=str(Path(args.baseline_dir).resolve()),
                reference_selection=str(Path(args.reference_selection).resolve()), cache_dir=str(Path(args.cache_dir).resolve())))
            if args.stage == 'check':
                report['status'] = 'STATIC_READOUT_CONTRACT_PASS_NO_TENSOR_LOAD_NO_GPU_NO_UPDATES'
            else:
                runtime = f.runtime(args.gpu)
                if runtime != cache['runtime']:
                    raise ValueError('Original cache/runtime/GPU model differs')
                report['runtime'] = runtime; device = 'cuda:'+str(args.gpu)
                print('One logical GPU:', args.gpu, '; loading reviewed CPU ROI cache, no detector construction.', flush=True)
                train, val, loaded = f.load_cache(args.cache_dir, proof['identity']['reference_identity']['training_identity']['parent_identity'])
                loaded['manifest_sha256'] = t.sha(Path(args.cache_dir)/'cache_manifest.json')
                if loaded != cache:
                    raise ValueError('Loaded cache metadata differs')
                report['cache_tensor_loads'] = 3
                initial, schedule = t.verify_loaded_inputs(train, val, baseline)
                # Sigma schedule is stored in its arm completion, not proof.
                saved_schedule = t.read(Path(args.reference_selection).parent/'completion.json')['schedule']
                if schedule != saved_schedule or schedule['epoch_steps'] != [902]*24:
                    raise ValueError('Reference full24 schedule differs')
                report['schedule'] = schedule
                if args.stage == 'train':
                    checked_preflight(args.preflight_report, proof, schedule)
                    report['preflight_report_sha256'] = t.sha(args.preflight_report)
                print('Replaying fixed sigma1.5 epoch03 on all887 cached VAL frames.', flush=True)
                report['reference_replay'] = replay_reference(args.reference_selection, val, reference, cache, device)
                print('Reference replay exact; starting', args.stage, flush=True)
                torch.cuda.reset_peak_memory_stats(args.gpu)
                if args.stage == 'preflight':
                    report['preflight'] = preflight(train, val, initial, schedule, reference, proof, cache, device, out)
                    report['status'] = PREFLIGHT_STATUS
                else:
                    result = train_condition(train, val, initial, proof, cache, reference, schedule, device, out, progress)
                    if report['head_updates_total'] != 21648 or result['updates'] != 21648:
                        raise ValueError('Formal one-arm budget differs')
                    rows = comparison_rows(result, reference)
                    t.write(out/'val_readout_compare.json', dict(protocol=VERSION, proof=proof, split='val', scale=1.,
                        rows=rows, groups={model.READOUT:result['value']['groups'], 'gaussian_prior_sigma1p5':reference['value']['groups']},
                        all_epoch_values=result['all_epoch_values'], reference_selection=reference['selection']['selected_checkpoint'],
                        automatic_promotion=False, test_access=False))
                    with (out/'val_readout_compare.csv').open('x', newline='') as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
                    for row in rows:
                        print('VAL', row['condition'], row['domain'], 'epoch', row['selected_epoch'],
                            'R_center(output)', row['output_frame_r_center_pct'], 'coverage', row['output_coverage_pct'],
                            'R_center(all)', row['full_frame_center_correct_pct'], 'RIoU', row['mean_riou'],
                            'A-RMSE', row['penalty_a_rmse_deg'], 'DFR', row['dfr_pct_per_frame'], 'ACI', row['aci'], flush=True)
                    report.update(status='READOUT_FULL_TRAIN_VAL_COMPLETE_REVIEW_REQUIRED', epochs_completed=24,
                        selection=result['selection'], train_support=f.support_report(train))
                report['cuda_peak_bytes'] = dict(allocated=torch.cuda.max_memory_allocated(args.gpu), reserved=torch.cuda.max_memory_reserved(args.gpu))
            report['elapsed_seconds'] = time.perf_counter()-started
            progress(dict(stage='complete', status=report['status'])); t.write(out/'completion.json', report)
        except Exception as error:
            report.update(status='FAILED_READOUT_REVIEW_REQUIRED', error=type(error).__name__+': '+str(error),
                elapsed_seconds=time.perf_counter()-started)
            progress(dict(stage='failed', error=report['error'])); t.write(out/'failure.json', report)
            raise
        finally:
            stream.flush()
            publish(out)
    print('Saved', out/'completion.json', 'status', report['status'], flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('check', 'preflight', 'train'), required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--baseline-dir', default='work_dirs/crane_symeood_k1_port_day2night_midpoint_formal_v1')
    p.add_argument('--reference-selection', default=e.DEFAULT_SELECTION)
    p.add_argument('--cache-dir', default='work_dirs/port_geometry_midpoint_formal_v1_roi_cache')
    p.add_argument('--preflight-report')
    p.add_argument('--out-dir', required=True)
    return p


if __name__ == '__main__':
    run(parser().parse_args())
